# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashAttention sparse MLA backends.

Two siblings, one per FlashAttention generation, because the two kernels take
the sparse KV by completely different routes:

- ``FLASH_ATTN_MLA_SPARSE`` (FA3, Hopper) feeds the top-k list to the paged-KV
  kernel as a page-size-1 block table.
- ``FLASH_ATTN_MLA_SPARSE_FA4`` (FA4 cute-DSL, Blackwell) uses the MLA-absorbed
  ``qv`` kernel's native top-k gather over a flat KV cache. That kernel asserts
  gather and page table are mutually exclusive, so the two paths cannot be one
  ``fa_version`` switch inside a shared ``forward_mqa``.
"""

from dataclasses import dataclass
from functools import cache
from typing import Any, ClassVar

import torch

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonPrefillMetadata
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
    SparseMLACommonMetadataBuilder,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    MLAAttentionImpl,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import flash_attn_supports_mla
from vllm.v1.attention.backends.mla.sparse_utils import (
    flat_kv_row_view,
    triton_convert_req_index_to_global_index,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.workspace import current_workspace_manager
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

logger = init_logger(__name__)

# The FA4 MLA gather kernel packs the query heads of one token into a single
# 128-row cluster tile and asserts that count exactly.
FA4_GATHER_NUM_HEADS = 128
# gather_kv_indices' last dim must be a whole number of n-tiles.
FA4_GATHER_TOPK_MULTIPLE = 128


class FlashAttnMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_MLA_SPARSE"

    @staticmethod
    def get_builder_cls() -> type["FlashAttnMLASparseMetadataBuilder"]:
        return FlashAttnMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl[Any]]:
        return FlashAttnMLASparseImpl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 9

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if kv_cache_dtype not in (None, "auto", "float16", "bfloat16"):
            return (
                "FlashAttention MLA Sparse currently supports only FP16/BF16 KV cache"
            )

        if not flash_attn_supports_mla():
            return "FlashAttention MLA not supported on this device"

        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None and vllm_config.model_config is not None:
            if vllm_config.parallel_config.decode_context_parallel_size > 1:
                return "FlashAttention MLA Sparse does not support DCP for now"

            hf_config = vllm_config.model_config.hf_config
            if not hasattr(hf_config, "index_topk"):
                return "FlashAttention MLA Sparse requires model with index_topk"
        return None


@dataclass
class FlashAttnMLASparseMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    seq_lens: torch.Tensor
    block_size: int = 64
    topk_tokens: int = 2048
    num_decodes: int = 0
    num_prefills: int = 0
    num_decode_tokens: int = 0
    prefill_max_seq_len: int = 0
    prefill: MLACommonPrefillMetadata | None = None
    cp_kv_cache_interleave_size: int = 1


class FlashAttnMLASparseMetadataBuilder(
    SparseMLACommonMetadataBuilder[FlashAttnMLASparseMetadata]
):
    metadata_cls = FlashAttnMLASparseMetadata
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        num_q_heads = self.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        threshold = {16: 128, 32: 128, 64: 256, 128: 256}.get(num_q_heads, 256)
        self._init_reorder_batch_threshold(threshold, supports_spec_as_decode=True)


class FlashAttnMLASparseImpl(SparseMLACommonImpl[FlashAttnMLASparseMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: Any | None = None,
        **mla_args: Any,
    ) -> None:
        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashAttnMLASparseImpl does not support alibi, sliding window, "
                "or logits soft cap."
            )
        if kv_cache_dtype not in ("auto", "float16", "bfloat16"):
            raise NotImplementedError(
                "FlashAttnMLASparseImpl currently supports only FP16/BF16 KV cache."
            )

        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            indexer=indexer,
            topk_indices_buffer=topk_indices_buffer,
            **mla_args,
        )
        assert self.topk_indices_buffer is not None, (
            "Indexer or topk_indices_buffer required for sparse MLA"
        )
        self.supports_quant_query_input = False

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashAttnMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not isinstance(q, tuple):
            raise NotImplementedError(
                "FlashAttnMLASparseImpl expects split (q_nope, q_rope) input."
            )
        q_nope, q_rope = q
        num_actual_toks = q_rope.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        kv_rows, block_stride_rows = flat_kv_row_view(
            kv_c_and_k_pe_cache, attn_metadata.block_size
        )
        topk_indices, valid_counts = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_actual_toks],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            BLOCK_STRIDE_ROWS=block_stride_rows,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
        )

        cu_seqlens_q = torch.arange(
            0, num_actual_toks + 1, dtype=torch.int32, device=q_rope.device
        )
        k_cache = kv_rows[:, self.kv_lora_rank :].unsqueeze(1).unsqueeze(1)
        v_cache = kv_rows[:, : self.kv_lora_rank].unsqueeze(1).unsqueeze(1)

        out = flash_attn_varlen_func(
            q=q_rope,
            k=k_cache,
            v=v_cache,
            q_v=q_nope,
            max_seqlen_q=1,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=topk_indices.shape[1],
            seqused_k=valid_counts,
            block_table=topk_indices,
            softmax_scale=self.scale,
            causal=True,
            fa_version=3,
        )
        return out, None


def _fa4_cute_mla_available() -> str | None:
    """Reason the FA4 cute-DSL MLA entry point is unusable, or None."""
    from vllm.vllm_flash_attn.flash_attn_interface import (
        fa_version_unsupported_reason,
        is_fa_version_supported,
    )

    if not is_fa_version_supported(4):
        return f"FA4 unavailable: {fa_version_unsupported_reason(4)}"
    try:
        from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd  # noqa: F401
    except Exception as e:  # cute-DSL / cutlass import chain
        return f"FA4 cute-DSL entry point failed to import: {e!r}"
    return None


class FlashAttnMLASparseFA4Backend(AttentionBackend):
    """Blackwell sparse MLA on FA4's MLA-absorbed ``qv`` kernel."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    # The qv kernel asserts every descale is None, so a quantized KV cache has
    # no route through it; fp8_ds_mla stays FlashMLA's.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_MLA_SPARSE_FA4"

    @staticmethod
    def get_builder_cls() -> type["FlashAttnMLASparseFA4MetadataBuilder"]:
        return FlashAttnMLASparseFA4MetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl[Any]]:
        return FlashAttnMLASparseFA4Impl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # 512 latent + 64 RoPE; split back apart for the qv kernel.
        return [576]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 10

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if kv_cache_dtype not in (None, "auto", "bfloat16"):
            return "FA4 sparse MLA supports only a BF16 KV cache"

        if (reason := _fa4_cute_mla_available()) is not None:
            return reason

        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is None or vllm_config.model_config is None:
            return None

        if vllm_config.parallel_config.decode_context_parallel_size > 1:
            return "FA4 sparse MLA does not support DCP yet"

        hf_config = vllm_config.model_config.hf_text_config
        topk = getattr(hf_config, "index_topk", None)
        if topk is None:
            return "FA4 sparse MLA requires a model with index_topk"
        if topk % FA4_GATHER_TOPK_MULTIPLE != 0:
            return (
                f"FA4 sparse MLA requires index_topk divisible by "
                f"{FA4_GATHER_TOPK_MULTIPLE}, got {topk}"
            )

        kv_lora_rank = getattr(hf_config, "kv_lora_rank", None)
        qk_rope_head_dim = getattr(hf_config, "qk_rope_head_dim", None)
        if (kv_lora_rank, qk_rope_head_dim) != (512, 64):
            return (
                "FA4 sparse MLA requires kv_lora_rank=512 and qk_rope_head_dim=64, "
                f"got {kv_lora_rank} and {qk_rope_head_dim}"
            )
        return None


class FlashAttnMLASparseFA4MetadataBuilder(
    SparseMLACommonMetadataBuilder[FlashAttnMLASparseMetadata]
):
    metadata_cls = FlashAttnMLASparseMetadata
    # Nothing in the FA4 gather path is shaped by the batch: every query token is
    # its own kernel-level request with a fixed top-k width, so a captured graph
    # only needs the shapes to be stable, which UNIFORM_BATCH already gives.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        num_q_heads = self.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        threshold = {8: 128, 16: 128, 32: 128, 64: 256, 128: 1024}.get(
            num_q_heads, 1024
        )
        self._init_reorder_batch_threshold(threshold, supports_spec_as_decode=True)


@cache
def _fa4_varlen_scalars(
    num_tokens: int, num_kv_rows: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Constant per-token varlen metadata for the flat-cache gather.

    Every query token is its own kernel-level request of length 1, all of them
    pointing at offset 0 of one flat KV buffer whose full row count is the index
    bound the kernel's ``-1``-sentinel bitmask checks against.

    The cache must never evict: these pointers are baked into captured CUDA
    graphs, so freeing one would turn a later replay into a use-after-free.
    For the same reason the first call must happen outside graph capture, which
    the eager warmup runs guarantee.
    """
    cu_seqlens_q = torch.arange(num_tokens + 1, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.zeros(num_tokens + 1, dtype=torch.int32, device=device)
    seqused_k = torch.full((num_tokens,), num_kv_rows, dtype=torch.int32, device=device)
    return cu_seqlens_q, cu_seqlens_k, seqused_k


class FlashAttnMLASparseFA4Impl(SparseMLACommonImpl[FlashAttnMLASparseMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: Any | None = None,
        **mla_args: Any,
    ) -> None:
        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashAttnMLASparseFA4Impl does not support alibi, sliding "
                "window, or logits soft cap."
            )
        if kv_cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError(
                "FlashAttnMLASparseFA4Impl supports only a BF16 KV cache."
            )

        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            indexer=indexer,
            topk_indices_buffer=topk_indices_buffer,
            **mla_args,
        )
        assert self.topk_indices_buffer is not None, (
            "Indexer or topk_indices_buffer required for sparse MLA"
        )
        self.supports_quant_query_input = False

        if num_heads > FA4_GATHER_NUM_HEADS:
            raise NotImplementedError(
                f"FA4 sparse MLA packs the query heads of one token into a "
                f"{FA4_GATHER_NUM_HEADS}-row tile; got {num_heads} heads per rank."
            )

        vllm_config = get_current_vllm_config()
        self.max_varlen_tokens = vllm_config.scheduler_config.max_num_batched_tokens

        self.q_pad_buffers: tuple[torch.Tensor, torch.Tensor] | None = None
        if num_heads < FA4_GATHER_NUM_HEADS:
            logger.warning_once(
                "Padding num_heads from %d to %d for the FA4 sparse MLA kernel",
                num_heads,
                FA4_GATHER_NUM_HEADS,
            )
            self.q_pad_buffers = tuple(  # type: ignore[assignment]
                current_workspace_manager().get_simultaneous(
                    (
                        (
                            self.max_varlen_tokens,
                            FA4_GATHER_NUM_HEADS,
                            self.kv_lora_rank,
                        ),
                        torch.bfloat16,
                    ),
                    (
                        (
                            self.max_varlen_tokens,
                            FA4_GATHER_NUM_HEADS,
                            self.qk_rope_head_dim,
                        ),
                        torch.bfloat16,
                    ),
                )
            )

    def _pad_q_heads(
        self, ql_nope: torch.Tensor, q_pe: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.q_pad_buffers is None:
            return ql_nope, q_pe
        num_tokens, num_heads = q_pe.shape[:2]
        qv, q_rope = (buf[:num_tokens] for buf in self.q_pad_buffers)
        for dst, src in ((qv, ql_nope), (q_rope, q_pe)):
            dst[:, :num_heads].copy_(src)
            # The workspace is shared with other consumers, so the padding lanes
            # hold whatever ran last; zero them so the discarded head rows can
            # never turn into NaNs the profiler has to explain.
            dst[:, num_heads:].zero_()
        return qv, q_rope

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashAttnMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not isinstance(q, tuple):
            raise NotImplementedError(
                "FlashAttnMLASparseFA4Impl expects the split (ql_nope, q_pe) "
                "query; the qv kernel never sees a fused 576-dim head."
            )
        ql_nope, q_pe = q
        num_actual_toks, num_heads = q_pe.shape[:2]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        kv_rows, block_stride_rows = flat_kv_row_view(
            kv_c_and_k_pe_cache, attn_metadata.block_size
        )
        topk_indices, valid_counts = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_actual_toks],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            BLOCK_STRIDE_ROWS=block_stride_rows,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
        )

        qv, q_rope = self._pad_q_heads(ql_nope, q_pe)
        num_kv_rows = kv_rows.shape[0]
        cu_seqlens_q, cu_seqlens_k, seqused_k = _fa4_varlen_scalars(
            self.max_varlen_tokens, num_kv_rows, kv_rows.device
        )

        out = flash_attn_varlen_func(
            q=q_rope,
            k=kv_rows[:, self.kv_lora_rank :].unsqueeze(1),
            v=kv_rows[:, : self.kv_lora_rank].unsqueeze(1),
            q_v=qv,
            max_seqlen_q=1,
            cu_seqlens_q=cu_seqlens_q[: num_actual_toks + 1],
            max_seqlen_k=num_kv_rows,
            cu_seqlens_k=cu_seqlens_k[: num_actual_toks + 1],
            seqused_k=seqused_k[:num_actual_toks],
            gather_kv_indices=topk_indices,
            # Per-row early exit out of the top-k walk. The kernel rounds the
            # length up to a whole 128-wide block, which the indexer's layout
            # satisfies: its valid entries are a prefix and the padding past
            # them is -1 (the assumption FLASHMLA_SPARSE's topk_length already
            # makes). Whether this is None is part of FA4's compile key, so it
            # is passed unconditionally: a steady-state flip would JIT a second
            # kernel, or diverge from what a captured graph holds.
            gather_kv_valid_length=valid_counts,
            softmax_scale=self.scale,
            # Causality already lives in the indexer's top-k list; a causal mask
            # here would instead clamp the *flat cache row* index space.
            causal=False,
            fa_version=4,
        )
        # Rows whose top-k list is entirely -1 come back as exact zeros: the
        # kernel's sentinel bitmask masks every column, so the softmax
        # denominator is 0 and its epilogue writes 0 (and -inf LSE).
        return out[:, :num_heads], None
