# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.watermarking.watermarker import RandomSamplingState, Watermarker
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.watermark import watermark_prep

logger = init_logger(__name__)


class GPUWatermarkSampler(Sampler):
    def __init__(
        self,
        watermarker: Watermarker,
        *args,
        deduplicate_contexts: Literal["none", "single_turn", "all"] = "single_turn",
        deduplicate_contexts_max_history: int | None = 8192,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.watermarker = watermarker
        self.deduplicate_contexts = deduplicate_contexts
        self.deduplicate_contexts_max_history = deduplicate_contexts_max_history
        self.watermarking = UvaBackedTensor(
            self.sampling_states.max_num_reqs, dtype=torch.bool
        )
        self.watermarking.np.fill(True)
        self.watermarking.copy_to_uva()

    def add_request(
        self, req_idx: int, prompt_len: int, sampling_params: SamplingParams
    ) -> None:
        super().add_request(req_idx, prompt_len, sampling_params)
        self.watermarking.np[req_idx] = sampling_params.watermarking
        if sampling_params.watermarking and sampling_params.temperature == 0:
            logger.warning_once(
                "Watermarking is enabled, but greedy decoding "
                "(temperature=0) cannot be watermarked. This request will use "
                "ordinary greedy sampling."
            )

    def apply_staged_writes(self) -> None:
        super().apply_staged_writes()
        self.watermarking.copy_to_uva()

    def _sample_random(
        self,
        processed_logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        top_k: torch.Tensor | None,
        top_p: torch.Tensor | None,
        use_flashinfer: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        enabled = self.watermarking.np[idx_mapping_np] & (
            self.sampling_states.temperature.np[idx_mapping_np] != 0
        )
        if not np.any(enabled):
            return super()._sample_random(
                processed_logits,
                expanded_idx_mapping,
                idx_mapping_np,
                pos,
                top_k,
                top_p,
                use_flashinfer,
            )

        processed_logits = apply_top_k_top_p(processed_logits, top_k, top_p)
        contexts, skip_mask = self._prepare_watermark_inputs(expanded_idx_mapping)

        def random_sample(sample_logits: torch.Tensor) -> torch.Tensor:
            return gumbel_sample(
                sample_logits,
                expanded_idx_mapping,
                self.sampling_states.temperature.gpu,
                self.sampling_states.seeds.gpu,
                pos,
                apply_temperature=False,
                is_drafting=False,
                use_fp64=self.use_fp64_gumbel,
            )

        # Temperature-0 rows are in `skip_mask`; ordinary sampling is argmax there.
        sampling_state = RandomSamplingState(
            expanded_idx_mapping=expanded_idx_mapping,
            temperatures=self.sampling_states.temperature.gpu,
            seeds=self.sampling_states.seeds.gpu,
            positions=pos,
            use_fp64=self.use_fp64_gumbel,
        )
        output = self.watermarker.sample(
            processed_logits,
            contexts,
            random_sample,
            skip_mask=skip_mask,
            sampling_state=sampling_state,
        )
        return output.token_ids, output.logits

    def _prepare_watermark_inputs(
        self, expanded_idx_mapping: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Contexts and skip mask for the batch, in one launch."""
        return watermark_prep(
            self.req_states.all_token_ids.gpu,
            expanded_idx_mapping,
            self.req_states.prompt_len.gpu,
            self.req_states.total_len.gpu,
            self.watermarker.context_width,
            watermarking=self.watermarking.gpu,
            temperatures=self.sampling_states.temperature.gpu,
            max_history=self.deduplicate_contexts_max_history,
            include_prompt=self.deduplicate_contexts == "all",
            scan=self.deduplicate_contexts != "none",
        )

    def _get_contexts(self, expanded_idx_mapping: torch.Tensor) -> torch.Tensor:
        contexts, _ = watermark_prep(
            self.req_states.all_token_ids.gpu,
            expanded_idx_mapping,
            self.req_states.prompt_len.gpu,
            self.req_states.total_len.gpu,
            self.watermarker.context_width,
            include_prompt=self.deduplicate_contexts == "all",
        )
        return contexts
