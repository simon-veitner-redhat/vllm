# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up FA4 CuTeDSL kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.prefill import get_mla_prefill_backend

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker


_MIN_CAUSAL_QUERY_LEN = 2
_MIXED_WARMUP_TOKENS = 3
_PACK_GQA_SMALL_TILE_MAX_ROWS = 64

_FA4_STATIC_QK_PAIRS = (
    (1, 1),
    (1, 16),
    (1, 128),
    (1, 512),
    (2, 2),
    (2, 16),
    (2, 128),
    (8, 8),
    (8, 128),
    (9, 9),
    (9, 128),
    (16, 16),
    (16, 128),
    (32, 32),
    (32, 128),
    (64, 64),
    (64, 256),
    (128, 128),
    (128, 129),
    (128, 512),
    (256, 256),
    (256, 512),
    (400, 400),
    (400, 512),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
)
_FA4_DYNAMIC_BQK_TRIPLES = (
    (1, 1, 1),
    (2, 1, 16),
    (8, 1, 128),
    (16, 1, 512),
    (32, 1, 2048),
    (64, 1, 512),
    (2, 8, 8),
    (4, 16, 64),
    (8, 32, 128),
    (16, 64, 128),
    (16, 128, 129),
    (16, 128, 512),
    (32, 64, 512),
    (64, 32, 2048),
    (128, 16, 4096),
)
_FA4_STATIC_SPLIT_REQUESTS = (1, 2, 17, 33, 65, 129)


def _capped_fa4_compile_classes(
    *,
    model_cap: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    ratio_query_lens: tuple[int, ...],
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int, int], ...]]:
    """Cap the finite compile list to the loaded scheduler and model."""
    static_candidates = _FA4_STATIC_QK_PAIRS + tuple(
        (query_len, model_cap) for query_len in ratio_query_lens
    )
    static_pairs: list[tuple[int, int]] = []
    for query_len, key_len in static_candidates:
        capped_key_len = min(key_len, model_cap)
        capped_pair = (
            min(query_len, capped_key_len, max_num_batched_tokens),
            capped_key_len,
        )
        if capped_pair not in static_pairs:
            static_pairs.append(capped_pair)

    dynamic_triples: list[tuple[int, int, int]] = []
    for batch_size, query_len, key_len in _FA4_DYNAMIC_BQK_TRIPLES:
        capped_batch_size = min(
            batch_size, max_num_seqs, max_num_batched_tokens
        )
        capped_key_len = min(key_len, model_cap)
        capped_triple = (
            capped_batch_size,
            min(
                query_len,
                capped_key_len,
                max_num_batched_tokens // capped_batch_size,
            ),
            capped_key_len,
        )
        if capped_triple not in dynamic_triples:
            dynamic_triples.append(capped_triple)
    return tuple(static_pairs), tuple(dynamic_triples)

def _causal_warmup_query_lens(
    max_warmup_tokens: int, num_queries_per_kv: tuple[int, ...]
) -> tuple[int, ...]:
    """Return query lengths that cover causal FA4 warmup paths.

    FA4 treats a one-token query as noncausal, so two is the shortest causal
    query. PackGQA changes tiles when the query length times
    ``num_queries_per_kv`` exceeds 64. Half the warmup budget also covers
    long-prefill scheduling.
    """
    query_lens = {_MIN_CAUSAL_QUERY_LEN, max_warmup_tokens // 2}
    query_lens.update(
        _PACK_GQA_SMALL_TILE_MAX_ROWS // ratio + 1 for ratio in num_queries_per_kv
    )
    return tuple(
        sorted(
            query_len
            for query_len in query_lens
            if _MIN_CAUSAL_QUERY_LEN <= query_len <= max_warmup_tokens
        )
    )


def _loaded_fa4_num_queries_per_kv(vllm_config: object) -> tuple[int, ...]:
    compilation_config = getattr(vllm_config, "compilation_config", None)
    static_forward_context = getattr(compilation_config, "static_forward_context", None)
    layers = getattr(static_forward_context, "values", None)
    if not callable(layers):
        return ()
    values = []
    for layer in layers():
        impl = getattr(layer, "impl", None)
        if getattr(impl, "vllm_flash_attn_version", None) != 4:
            continue
        ratio = getattr(impl, "num_queries_per_kv", None)
        if ratio is None:
            num_heads = getattr(impl, "num_heads", 1)
            num_kv_heads = getattr(impl, "num_kv_heads", 1)
            ratio = num_heads // num_kv_heads
        values.append(ratio)
    return tuple(dict.fromkeys(values))


@dataclass(frozen=True)
class _Fa4CompileConfig:
    num_heads: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_v: int
    num_queries_per_kv: int
    is_diffkv: bool
    packed_width: int
    q_dtype: torch.dtype
    has_descales: bool
    out_dtype: torch.dtype | None
    softmax_scale: float
    window_size: tuple[int, int]
    has_sink: bool


def _warm_fa4_compile_specs(worker: Worker) -> None:
    """Compile loaded dense and native-FP8 FA4 keys without running attention."""
    runner = worker.model_runner
    if (
        runner.is_pooling_model
        or not current_platform.is_device_capability(90)
    ):
        return

    vllm_config = runner.vllm_config
    if (
        vllm_config.model_config.use_mla
        or vllm_config.attention_config.flash_attn_version != 4
    ):
        return

    from vllm.v1.kv_cache_interface import CrossAttentionSpec

    kv_cache_config = getattr(runner, "kv_cache_config", None)
    kv_cache_groups = getattr(kv_cache_config, "kv_cache_groups", ())
    if any(
        isinstance(group.kv_cache_spec, CrossAttentionSpec)
        for group in kv_cache_groups
    ):
        return

    compilation_config = getattr(vllm_config, "compilation_config", None)
    static_forward_context = getattr(compilation_config, "static_forward_context", None)
    layers = getattr(static_forward_context, "values", None)
    cache_config = getattr(vllm_config, "cache_config", None)
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    block_size = getattr(cache_config, "block_size", None)
    model_cap = vllm_config.model_config.max_model_len
    max_num_seqs = getattr(scheduler_config, "max_num_seqs", 0)
    max_num_batched_tokens = getattr(
        scheduler_config, "max_num_batched_tokens", 0
    )
    if (
        not callable(layers)
        or not block_size
        or model_cap < 1
        or max_num_seqs < 1
        or max_num_batched_tokens < 1
    ):
        return

    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    from vllm.v1.attention.backends.flash_attn_diffkv import (
        FlashAttentionDiffKVBackend,
        FlashAttentionDiffKVImpl,
    )
    from vllm.v1.attention.backends.utils import get_kv_cache_layout
    from vllm.vllm_flash_attn.flash_attn_interface import (
        compile_flash_attn_varlen_func_from_specs,
    )
    max_pages = (model_cap + block_size - 1) // block_size

    configs: list[_Fa4CompileConfig] = []
    seen_configs: set[_Fa4CompileConfig] = set()
    for layer in layers():
        impl = getattr(layer, "impl", None)
        if getattr(impl, "vllm_flash_attn_version", None) != 4:
            continue
        fp8_mode = getattr(impl, "sm90_fa4_fp8_mode", None)
        if fp8_mode not in (None, "native"):
            continue
        attn_type = getattr(impl, "attn_type", None)
        if attn_type is not None and getattr(attn_type, "name", None) != "DECODER":
            continue
        num_heads = getattr(impl, "num_heads", None)
        num_kv_heads = getattr(impl, "num_kv_heads", None)
        head_dim_qk = getattr(impl, "head_size", None)
        if not num_heads or not num_kv_heads or not head_dim_qk:
            continue
        is_diffkv = isinstance(impl, FlashAttentionDiffKVImpl)
        if is_diffkv and fp8_mode != "native":
            continue
        if fp8_mode is None and not isinstance(impl, FlashAttentionImpl):
            continue
        head_dim_v = (
            (
                getattr(layer, "head_size_v", None)
                or getattr(impl, "head_size_v", None)
                or FlashAttentionDiffKVBackend.head_size_v
            )
            if is_diffkv
            else head_dim_qk
        )
        if not head_dim_v:
            continue
        ratio = getattr(impl, "num_queries_per_kv", num_heads // num_kv_heads)
        if not ratio:
            continue
        raw_window = getattr(impl, "sliding_window", None)
        window_size = (-1, -1) if raw_window is None else tuple(raw_window)
        if len(window_size) != 2:
            continue
        model_dtype = getattr(
            impl, "model_dtype", vllm_config.model_config.dtype
        )
        if fp8_mode == "native":
            q_dtype = torch.float8_e4m3fn
            out_dtype = getattr(impl, "native_fp8_out_dtype", None)
            if out_dtype is None:
                out_dtype = model_dtype
            has_descales = True
        else:
            if model_dtype not in (torch.bfloat16, torch.float16):
                continue
            q_dtype = model_dtype
            # The compile-only adapter reserves explicit out_dtype for native
            # FP8; ordinary attention defaults its output to q_dtype.
            out_dtype = None
            has_descales = False
        config = _Fa4CompileConfig(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_v=head_dim_v,
            num_queries_per_kv=ratio,
            is_diffkv=is_diffkv,
            packed_width=head_dim_qk + head_dim_v,
            out_dtype=out_dtype,
            q_dtype=q_dtype,
            has_descales=has_descales,
            softmax_scale=getattr(impl, "scale", head_dim_qk ** (-0.5)),
            window_size=window_size,
            has_sink=getattr(impl, "sinks", None) is not None,
        )
        if config not in seen_configs:
            seen_configs.add(config)
            configs.append(config)
    if not configs:
        return

    split_requests = _FA4_STATIC_SPLIT_REQUESTS
    dynamic_split_cap = (
        getattr(
            vllm_config.attention_config,
            "flash_attn_max_num_splits_for_cuda_graph",
            0,
        )
        or 0
    )
    seen_specs: set[tuple[tuple[str, object], ...]] = set()
    cache_layout = get_kv_cache_layout()
    for config in configs:
        short_boundary = max(
            2, _PACK_GQA_SMALL_TILE_MAX_ROWS // config.num_queries_per_kv
        )
        query_lens = tuple(
            sorted(
                {
                    min(model_cap, query_len)
                    for query_len in (
                        1,
                        min(2, model_cap),
                        short_boundary,
                        short_boundary + 1,
                        model_cap,
                    )
                    if query_len >= 1
                }
            )
        )
        static_pairs, dynamic_triples = _capped_fa4_compile_classes(
            model_cap=model_cap,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            ratio_query_lens=query_lens,
        )
        if cache_layout == "NHD":
            v_stride = (
                block_size * config.num_kv_heads * config.packed_width,
                config.num_kv_heads * config.packed_width,
                config.packed_width,
                1,
            )
        else:
            v_stride = (
                config.num_kv_heads * block_size * config.packed_width,
                config.packed_width,
                block_size * config.packed_width,
                1,
            )

        def compile_class(
            batch_size: int,
            query_len: int,
            key_len: int,
            num_splits: int,
            *,
            causal: bool,
            has_dynamic_splits: bool,
            has_dynamic_scheduler: bool,
        ) -> None:
            total_q = batch_size * query_len
            split_pointer_shape = (batch_size,) if has_dynamic_splits else None
            kwargs = {
                "q_shape": (
                    total_q,
                    config.num_heads,
                    config.head_dim_qk,
                ),
                "k_shape": (
                    max_pages,
                    block_size,
                    config.num_kv_heads,
                    config.head_dim_qk,
                ),
                "v_shape": (
                    max_pages,
                    block_size,
                    config.num_kv_heads,
                    config.head_dim_v,
                ),
                "q_dtype": config.q_dtype,
                "out_dtype": config.out_dtype,
                "v_stride": v_stride,
                "num_splits_dynamic_ptr_shape": split_pointer_shape,
                "num_splits_dynamic_ptr_stride": (
                    (1,) if has_dynamic_splits else None
                ),
                "s_aux_shape": (config.num_heads,) if config.has_sink else None,
                "s_aux_stride": (1,) if config.has_sink else None,
                "dynamic_scheduler_counter_shape": (
                    (1,) if has_dynamic_scheduler else None
                ),
                "dynamic_scheduler_counter_stride": (
                    (1,) if has_dynamic_scheduler else None
                ),
                "cu_seqlens_q_shape": (batch_size + 1,),
                "seqused_k_shape": (batch_size,),
                "seqused_k_stride": (1,),
                "page_table_shape": (batch_size, max_pages),
                "page_table_stride": (max_pages, 1),
                "q_descale_shape": () if config.has_descales else None,
                "q_descale_stride": () if config.has_descales else None,
                "k_descale_shape": () if config.has_descales else None,
                "k_descale_stride": () if config.has_descales else None,
                "v_descale_shape": () if config.has_descales else None,
                "v_descale_stride": () if config.has_descales else None,
                "max_seqlen_q": query_len,
                "max_seqlen_k": key_len,
                "softmax_scale": config.softmax_scale,
                "causal": causal,
                "window_size": list(config.window_size),
                "num_splits": num_splits,
                "fa_version": 4,
            }
            spec = tuple(
                (name, tuple(value) if isinstance(value, list) else value)
                for name, value in kwargs.items()
            )
            if spec not in seen_specs:
                seen_specs.add(spec)
                compile_flash_attn_varlen_func_from_specs(**kwargs)

        for query_len, key_len in static_pairs:
            for num_splits in split_requests:
                for causal in (True, False):
                    compile_class(
                        1,
                        query_len,
                        key_len,
                        num_splits,
                        causal=causal,
                        has_dynamic_splits=False,
                        has_dynamic_scheduler=False,
                    )
        for batch_size, query_len, key_len in dynamic_triples:
            if batch_size > 1:
                for causal in (True, False):
                    compile_class(
                        batch_size,
                        query_len,
                        key_len,
                        1,
                        causal=causal,
                        has_dynamic_splits=False,
                        has_dynamic_scheduler=True,
                    )
            if dynamic_split_cap > 1:
                for causal in (True, False):
                    compile_class(
                        batch_size,
                        query_len,
                        key_len,
                        dynamic_split_cap,
                        causal=causal,
                        has_dynamic_splits=True,
                        has_dynamic_scheduler=True,
                    )


def _warm_fa4_mla_prefill(worker: Worker) -> None:
    runner = worker.model_runner
    if runner.is_pooling_model:
        return

    vllm_config = runner.vllm_config
    if not vllm_config.model_config.use_mla:
        return
    try:
        backend_cls = get_mla_prefill_backend(vllm_config)
    except ValueError:
        # Fall back to the top-k MQA prefill path.
        return
    if backend_cls.get_name() != "FLASH_ATTN":
        return

    from vllm.v1.attention.backends.mla.prefill import flash_attn

    flash_attn.FA4_MLA_PREFILL_KERNEL.warmup(vllm_config)


def _warm_fa4_runtime_attention(worker: Worker) -> None:
    runner = worker.model_runner
    if runner.is_pooling_model:
        return

    vllm_config = runner.vllm_config
    if vllm_config.model_config.use_mla:
        try:
            backend_cls = get_mla_prefill_backend(vllm_config)
        except ValueError:
            # Fall back to the top-k MQA prefill path.
            return
        if backend_cls.get_name() != "FLASH_ATTN":
            return

    if (
        not current_platform.is_device_capability(90)
        or vllm_config.attention_config.flash_attn_version != 4
    ):
        return

    from vllm.v1.worker.gpu.warmup import run_mixed_prefill_decode_warmup

    if not worker.use_v2_model_runner:
        if vllm_config.model_config.use_mla:
            from vllm.v1.attention.backends.mla.flashattn_mla import (
                FlashAttnMLAMetadataBuilder,
            )

            max_warmup_tokens = min(
                vllm_config.scheduler_config.max_num_batched_tokens,
                vllm_config.model_config.max_model_len,
            )
            if max_warmup_tokens < 2:
                return
            runner._dummy_run(
                min(
                    FlashAttnMLAMetadataBuilder.reorder_batch_threshold + 1,
                    max_warmup_tokens,
                ),
                force_attention=True,
                is_profile=True,
                create_mixed_batch=True,
                skip_eplb=True,
                profile_seq_lens=max_warmup_tokens // 2,
            )
            for context_len in (min(128, max_warmup_tokens), max_warmup_tokens // 2):
                runner._dummy_run(
                    2,
                    force_attention=True,
                    is_profile=True,
                    skip_eplb=True,
                    profile_seq_lens=context_len,
                    num_reqs=1,
                )
        else:
            num_queries_per_kv = _loaded_fa4_num_queries_per_kv(vllm_config)
            if not num_queries_per_kv:
                return
            max_warmup_tokens = min(
                vllm_config.scheduler_config.max_num_batched_tokens,
                vllm_config.model_config.max_model_len,
            )
            for query_len in _causal_warmup_query_lens(
                max_warmup_tokens, num_queries_per_kv
            ):
                # Warm causal prefill at batch size 1.
                runner._dummy_run(
                    query_len,
                    force_attention=True,
                    is_profile=True,
                    skip_eplb=True,
                    profile_seq_lens=query_len,
                    num_reqs=1,
                )
                if (
                    runner.scheduler_config.max_num_seqs >= 2
                    and query_len < max_warmup_tokens
                ):
                    # One cached request decodes a token while a new request
                    # runs the shortest causal prefill.
                    runner._dummy_run(
                        _MIXED_WARMUP_TOKENS,
                        force_attention=True,
                        is_profile=True,
                        create_mixed_batch=True,
                        skip_eplb=True,
                        profile_seq_lens=[
                            query_len + 1,
                            _MIN_CAUSAL_QUERY_LEN,
                        ],
                    )
        return
    from vllm.v1.kv_cache_interface import CrossAttentionSpec, MambaSpec

    if any(
        isinstance(group.kv_cache_spec, (CrossAttentionSpec, MambaSpec))
        for group in runner.kv_cache_config.kv_cache_groups
    ):
        return
    max_warmup_tokens = min(
        vllm_config.scheduler_config.max_num_batched_tokens,
        vllm_config.model_config.max_model_len,
    )
    if vllm_config.model_config.use_mla:
        from vllm.v1.attention.backends.mla.flashattn_mla import (
            FlashAttnMLAMetadataBuilder,
        )

        absorbed_tokens = min(
            FlashAttnMLAMetadataBuilder.reorder_batch_threshold + 1,
            max_warmup_tokens,
        )
        run_mixed_prefill_decode_warmup(
            runner,
            worker.execute_model,
            worker.sample_tokens,
            absorbed_tokens,
            req_id_prefix=f"_fa4_mla_warmup_{absorbed_tokens}",
        )

    if vllm_config.model_config.use_mla:
        context_tokens = max_warmup_tokens // 2
        run_mixed_prefill_decode_warmup(
            runner,
            worker.execute_model,
            worker.sample_tokens,
            num_tokens=_MIXED_WARMUP_TOKENS,
            decode_prompt_len=context_tokens,
            decode_scheduled_tokens=1,
            req_id_prefix=f"_fa4_warmup_{max_warmup_tokens}",
        )
    else:
        num_queries_per_kv = _loaded_fa4_num_queries_per_kv(vllm_config)
        if not num_queries_per_kv:
            return
        for query_len in _causal_warmup_query_lens(
            max_warmup_tokens, num_queries_per_kv
        ):
            mixed_warmed = False
            if query_len < max_warmup_tokens:
                mixed_warmed = run_mixed_prefill_decode_warmup(
                    runner,
                    worker.execute_model,
                    worker.sample_tokens,
                    num_tokens=_MIXED_WARMUP_TOKENS,
                    decode_prompt_len=query_len,
                    decode_scheduled_tokens=1,
                    req_id_prefix=(f"_fa4_warmup_{max_warmup_tokens}_{query_len}"),
                )
            if not mixed_warmed:
                runner._dummy_run(
                    query_len,
                    skip_eplb=True,
                    is_profile=True,
                    num_reqs=1,
                )


def _warm_inkling_fa4_rel_attention(worker: Worker) -> None:
    from vllm.models.inkling.configs import InklingMMConfig, InklingModelConfig
    from vllm.models.inkling.nvidia.ops.fa4_rel_attention import (
        INKLING_FA4_REL_ATTENTION_KERNEL,
    )

    vllm_config = worker.vllm_config
    hf_config = vllm_config.model_config.hf_config
    if not isinstance(hf_config, (InklingMMConfig, InklingModelConfig)):
        return

    INKLING_FA4_REL_ATTENTION_KERNEL.warmup(vllm_config)


def fa4_cutedsl_warmup(worker: Worker) -> None:
    _warm_fa4_compile_specs(worker)
    _warm_fa4_mla_prefill(worker)
    _warm_fa4_runtime_attention(worker)
    _warm_inkling_fa4_rel_attention(worker)
