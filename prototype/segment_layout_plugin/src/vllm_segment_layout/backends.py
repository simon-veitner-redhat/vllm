"""Model-owned attention backends for the segment layout.

Two ways to run the same attention pattern, both handed to the model's layers
through ``Attention(attn_backend=...)``:

* ``flex``: the stock FlexAttention encoder path with another mask closure.
  Same mechanism as vllm#65268, kept for an exact comparison.
* ``varlen``: the stock FlashAttention encoder path. The layout is block
  structured, so it is a plain varlen call where segment s has its own query
  rows and the key rows ``prefix + itself``. No mask, no T x T work.

The model state publishes the step's rows through a ``StepContext`` that the
model attaches to its own attention layers. Nothing travels through
``CommonAttentionMetadata``.
"""

import os
from dataclasses import dataclass

import torch

import vllm.v1.attention.backends.flex_attention as _flex

from vllm.config import get_layers_from_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.encoder_only_attention import (
    create_encoder_only_attention_backend,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    subclass_attention_backend,
    subclass_attention_backend_with_overrides,
)
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
)
from vllm.v1.attention.backends.flex_attention import (
    FlexAttentionBackend,
    FlexAttentionMetadataBuilder,
)

from .layout import StepRows

_SINGLE_BUILD = os.environ.get("SEGMENT_FLEX_SINGLE_BUILD") == "1"


@dataclass
class VarlenRows:
    num_tokens: int
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    kv_index: torch.Tensor
    max_q: int
    max_k: int


class StepContext:
    """Rows of the current engine step. Written by the model state in
    ``prepare_attn``, read by the backends of the same model."""

    def __init__(self) -> None:
        self.rows: StepRows | None = None
        self.varlen: VarlenRows | None = None

    def set(self, rows: StepRows | None, device: torch.device, varlen: bool) -> None:
        self.rows = rows
        self.varlen = None
        if rows is not None and varlen:
            self.varlen = VarlenRows(
                num_tokens=rows.num_tokens,
                cu_q=torch.from_numpy(rows.cu_q).to(device),
                cu_k=torch.from_numpy(rows.cu_k).to(device),
                kv_index=torch.from_numpy(rows.kv_index).to(device),
                max_q=rows.max_q,
                max_k=rows.max_k,
            )


def _context_of(vllm_config, layer_names) -> StepContext:
    layers = get_layers_from_vllm_config(vllm_config, Attention, layer_names)
    return next(iter(layers.values())).segment_step


class SegmentFlexBuilder(FlexAttentionMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.segment_step = _context_of(vllm_config, layer_names)
        # Full capacity and persistent: the mask closure keeps one shape, so a
        # new batch size does not recompile it.
        num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self._own = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        self._vis = torch.zeros(num_tokens, dtype=torch.int32, device=device)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        rows = self.segment_step.rows
        if rows is None or rows.num_tokens != common_attn_metadata.num_actual_tokens:
            # The stock mask, untouched.
            return super().build(common_prefix_len, common_attn_metadata, fast_build)

        own, vis = self._own, self._vis
        own[: rows.num_tokens].copy_(torch.from_numpy(rows.own))
        vis[: rows.num_tokens].copy_(torch.from_numpy(rows.vis))

        def segment_mask_mod(b, h, q_idx, kv_idx):
            kv_own = own[kv_idx]
            return (kv_own == own[q_idx]) | (kv_own == vis[q_idx])

        if _SINGLE_BUILD:
            # MEASUREMENT ONLY (SEGMENT_FLEX_SINGLE_BUILD=1): what the plugin
            # would do with a `metadata_cls` hook on the stock builder. Without
            # that hook the only way to get there is to swap the module global
            # for the duration of the call, which a real plugin must not do.
            class _SegmentMetadata(_flex.FlexAttentionMetadata):
                def get_bidirectional_mask_mod(self):
                    return segment_mask_mod

            stock = _flex.FlexAttentionMetadata
            _flex.FlexAttentionMetadata = _SegmentMetadata
            try:
                return super().build(common_prefix_len, common_attn_metadata, fast_build)
            finally:
                _flex.FlexAttentionMetadata = stock

        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        # COST: super().build() has already built the stock block mask, because
        # FlexAttentionMetadataBuilder.build names FlexAttentionMetadata and
        # builds the mask in one go. On a layout step the mask is built twice.
        metadata.get_mask_mod = lambda: segment_mask_mod
        metadata.mask_mod = segment_mask_mod
        metadata.block_mask = metadata.build_block_mask()
        return metadata


class SegmentVarlenImpl(FlashAttentionImpl):
    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        rows = layer.segment_step.varlen
        if (
            rows is None
            or attn_metadata is None
            or rows.num_tokens != attn_metadata.num_actual_tokens
        ):
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )
        num_tokens = rows.num_tokens
        descale_shape = (rows.cu_q.shape[0] - 1, self.num_kv_heads)
        # COPY of the call in FlashAttentionImpl._forward_encoder_attention
        # (flash_attn.py:1705-1731), with the key rows of each segment gathered
        # and their own cu_seqlens. The stock method takes one query_start_loc
        # for both sides, so it cannot be reused.
        flash_attn_varlen_func(
            q=query[:num_tokens],
            k=key.index_select(0, rows.kv_index),
            v=value.index_select(0, rows.kv_index),
            out=output[:num_tokens],
            cu_seqlens_q=rows.cu_q,
            cu_seqlens_k=rows.cu_k,
            max_seqlen_q=rows.max_q,
            max_seqlen_k=rows.max_k,
            softmax_scale=self.scale,
            causal=False,
            alibi_slopes=self.alibi_slopes,
            window_size=list(self.sliding_window),
            softcap=self.logits_soft_cap,
            fa_version=self.vllm_flash_attn_version,
            q_descale=layer._q_scale.expand(descale_shape)
            if self.supports_quant_query_input
            else None,
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            num_splits=1 if self.batch_invariant_enabled or self.fa4_hd256 else 0,
            s_aux=self.sinks,
        )
        return output


def segment_backend(kind: str) -> type[AttentionBackend]:
    """``flex`` or ``varlen``, wrapped like EncoderOnlyAttention wraps its backend
    (the wrapper's builder forces ``causal=False``)."""
    if kind == "flex":
        backend = subclass_attention_backend(
            "SegmentLayout_", FlexAttentionBackend, SegmentFlexBuilder
        )
    elif kind == "varlen":
        backend = subclass_attention_backend_with_overrides(
            "SegmentLayout_",
            FlashAttentionBackend,
            {"get_impl_cls": staticmethod(lambda: SegmentVarlenImpl)},
        )
    else:
        raise ValueError(f"segment_layout_backend must be flex or varlen, got {kind!r}")
    return create_encoder_only_attention_backend(backend)
