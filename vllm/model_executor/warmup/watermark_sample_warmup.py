# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up every specialization of the watermarked sampler's Triton kernel.

The generic sampler warmup always processes logits, so it only compiles
``_philox_gumbel_kernel`` for fp32 logits. The first plain request passes
model-dtype logits and would otherwise compile inside inference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.v1.watermarking.watermarker import Watermarker
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _philox_key(watermarker: Watermarker) -> int | None:
    from vllm.v1.watermarking.gumbel import GumbelWatermarker
    from vllm.v1.watermarking.prfs import PhiloxPRF

    if not isinstance(watermarker, GumbelWatermarker):
        return None
    prf = watermarker.prf
    return prf.key if type(prf) is PhiloxPRF else None


def _philox_sampler_keys(watermarker: Watermarker) -> list[int]:
    from vllm.v1.watermarking.gumbel import DualKeyGumbelWatermarker

    watermarkers: list[Watermarker] = [watermarker]
    if isinstance(watermarker, DualKeyGumbelWatermarker):
        if watermarker.alpha == 1:
            watermarkers = [watermarker.key_b_watermarker]
        elif watermarker.alpha != 0:
            watermarkers = [watermarker, watermarker.key_b_watermarker]
    return [key for key in map(_philox_key, watermarkers) if key is not None]


@torch.inference_mode()
def watermark_sample_warmup(worker: Worker) -> None:
    if worker.vllm_config.watermark_config is None:
        return
    # GumbelWatermarker only launches the kernel on CUDA.
    if not current_platform.is_cuda_alike():
        return

    from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler
    from vllm.v1.worker.gpu.sample.watermark import philox_gumbel_sample

    sampler = getattr(worker.model_runner, "sampler", None)
    if not isinstance(sampler, GPUWatermarkSampler):
        return

    model_config = worker.vllm_config.model_config
    device = worker.device
    try:
        keys = _philox_sampler_keys(sampler.watermarker)
        if not keys:
            return

        dtypes = {cast("torch.dtype", model_config.dtype), torch.float32}
        # The mask-free variant is only reachable with deduplication off.
        with_skip_mask = (
            (True,) if sampler.deduplicate_contexts != "none" else (True, False)
        )
        vocab_size = model_config.get_vocab_size()
        contexts = torch.zeros(
            (1, sampler.watermarker.context_width), dtype=torch.int32, device=device
        )
        # Dtypes match the runtime buffers; they are part of the kernel key.
        sampling_state = {
            "skip_mask": torch.zeros(1, dtype=torch.bool, device=device),
            "expanded_idx_mapping": torch.zeros(1, dtype=torch.int64, device=device),
            "temperatures": torch.ones(1, dtype=torch.float32, device=device),
            "seeds": torch.zeros(1, dtype=torch.int64, device=device),
            "positions": torch.zeros(1, dtype=torch.int64, device=device),
        }

        logger.info(
            "Warming up watermark sampler kernel (vocab=%d, keys=%d, dtypes=%s, "
            "skip_mask=%s).",
            vocab_size,
            len(keys),
            [str(dtype) for dtype in dtypes],
            list(with_skip_mask),
        )
        for dtype in dtypes:
            logits = torch.zeros((1, vocab_size), dtype=dtype, device=device)
            for key in keys:
                for use_skip_mask in with_skip_mask:
                    # Only the masked (mixed) path forwards use_fp64_gumbel.
                    philox_gumbel_sample(
                        logits,
                        contexts,
                        key,
                        use_fp64=use_skip_mask and sampler.use_fp64_gumbel,
                        **(sampling_state if use_skip_mask else {}),
                    )
    except Exception:
        logger.warning("Skipping watermark sampler warmup.", exc_info=True)
