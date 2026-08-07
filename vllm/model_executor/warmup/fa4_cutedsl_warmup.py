# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up FA4 CuTeDSL attention kernels."""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.prefill import get_mla_prefill_backend

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker


@torch.no_grad()
def normalize_hopper_fa4_fp8_scales(worker: Worker) -> None:
    runner = worker.model_runner
    if runner.is_pooling_model:
        return

    vllm_config = runner.vllm_config
    if (
        vllm_config.attention_config.flash_attn_version != 4
        or not current_platform.is_device_capability(90)
        or not vllm_config.attention_config._hopper_fa4_fp8
    ):
        return

    from vllm.model_executor.layers.attention import Attention

    layers = vllm_config.compilation_config.static_forward_context
    for module in layers.values():
        if not isinstance(module, Attention):
            continue
        k_scale = float(module._k_scale_float)
        v_scale = float(module._v_scale_float)
        if (
            not isfinite(k_scale)
            or not isfinite(v_scale)
            or k_scale <= 0.0
            or v_scale <= 0.0
            or k_scale == 1.0
            or v_scale == 1.0
        ):
            raise ValueError(
                "Hopper FA4 FP8 requires finite positive non-unit per-tensor "
                "K/V scales"
            )
        module._q_scale.fill_(k_scale)
        module._k_scale.fill_(k_scale)
        module._v_scale.fill_(v_scale)
        module._q_scale_float = k_scale
        module._k_scale_cpu.fill_(k_scale)
        module._v_scale_cpu.fill_(v_scale)


def fa4_cutedsl_warmup(worker: Worker) -> None:
    runner = worker.model_runner
    if runner.is_pooling_model:
        return

    vllm_config = runner.vllm_config
    if not current_platform.is_device_capability(90):
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
        return
    if vllm_config.attention_config.flash_attn_version != 4:
        return
    normalize_hopper_fa4_fp8_scales(worker)

    from vllm.v1.worker.gpu.warmup import run_mixed_prefill_decode_warmup

    if vllm_config.model_config.use_mla:
        try:
            backend_cls = get_mla_prefill_backend(vllm_config)
        except ValueError:
            # Fall back to the top-k MQA prefill path.
            return
        if backend_cls.get_name() != "FLASH_ATTN":
            return

        from vllm.v1.attention.backends.mla.prefill import flash_attn

        if not flash_attn.FA4_MLA_PREFILL_KERNEL.get_warmup_keys(vllm_config):
            return
        flash_attn.FA4_MLA_PREFILL_KERNEL.warmup(vllm_config)

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
    if vllm_config.attention_config._hopper_fa4_fp8:
        # Close ordinary and dynamic decode, both persistent single-request
        # tiles, and both dynamic mixed-varlen tiles before graph capture.
        decode_shapes = (
            (min(4, runner.max_num_reqs), max_warmup_tokens // 2),
            (min(16, runner.max_num_reqs), min(16, max_warmup_tokens)),
        )
        for num_reqs, context_len in decode_shapes:
            runner._dummy_run(
                num_reqs,
                force_attention=True,
                uniform_decode=True,
                is_profile=True,
                skip_eplb=True,
                profile_seq_lens=context_len,
                num_reqs=num_reqs,
            )
        prefill_lengths = sorted(
            {min(16, max_warmup_tokens), min(17, max_warmup_tokens)}
        )
        for query_len in prefill_lengths:
            runner._dummy_run(
                query_len,
                force_attention=True,
                is_profile=True,
                skip_eplb=True,
                profile_seq_lens=query_len,
                num_reqs=1,
            )
        for prefill_len in prefill_lengths:
            run_mixed_prefill_decode_warmup(
                runner,
                worker.execute_model,
                worker.sample_tokens,
                prefill_len + 2,
                decode_prompt_len=prefill_len,
                num_decode_reqs=1,
                decode_scheduled_tokens=2,
                req_id_prefix=f"_fa4_fp8_warmup_{prefill_len}",
            )
        run_mixed_prefill_decode_warmup(
            runner,
            worker.execute_model,
            worker.sample_tokens,
            prefill_lengths[-1] + 2,
            decode_prompt_len=max_warmup_tokens // 2,
            num_decode_reqs=1,
            decode_scheduled_tokens=2,
            req_id_prefix=f"_fa4_fp8_warmup_long_{max_warmup_tokens}",
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

    # Long prefill compilation and long-context batched decode select scheduler
    # modes that the existing short-context model-runner warmup does not reach.
    num_long_decodes = min(4, runner.max_num_reqs - 1)
    decode_scheduled_tokens = 2
    run_mixed_prefill_decode_warmup(
        runner,
        worker.execute_model,
        worker.sample_tokens,
        num_long_decodes * decode_scheduled_tokens + 17,
        decode_prompt_len=max_warmup_tokens // 2,
        num_decode_reqs=num_long_decodes,
        decode_scheduled_tokens=decode_scheduled_tokens,
        req_id_prefix=f"_fa4_warmup_{max_warmup_tokens}",
    )
