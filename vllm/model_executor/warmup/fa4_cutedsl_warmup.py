# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up FA4 CuTeDSL attention kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.v1.attention.backends.mla.prefill import get_mla_prefill_backend

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker


def fa4_cutedsl_warmup(worker: Worker) -> None:
    runner = worker.model_runner
    if runner.is_pooling_model:
        return

    vllm_config = runner.vllm_config
    if vllm_config.attention_config.flash_attn_version != 4:
        return

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

    # Long prefill compilation and long-context batched decode select scheduler
    # modes that the existing short-context model-runner warmup does not reach.
    num_long_decodes = min(4, runner.max_num_reqs - 1)
    decode_scheduled_tokens = 2
    run_mixed_prefill_decode_warmup(
        runner,
        worker.execute_model,
        worker.sample_tokens,
        num_long_decodes * decode_scheduled_tokens + 3,
        decode_prompt_len=max_warmup_tokens // 2,
        num_decode_reqs=num_long_decodes,
        decode_scheduled_tokens=decode_scheduled_tokens,
        req_id_prefix=f"_fa4_warmup_{max_warmup_tokens}",
    )
