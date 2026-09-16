# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The vllm:request_watermarked counter reports the engine's watermark decision."""

import prometheus_client
import pytest

from vllm.config import ModelConfig, VllmConfig
from vllm.v1.engine import FinishReason
from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.metrics.prometheus import unregister_vllm_metrics
from vllm.v1.metrics.stats import IterationStats, RequestStateStats

pytestmark = pytest.mark.cpu_test

TEST_MODEL = "distilbert/distilgpt2"


def _finished_iteration_stats(watermarked_values: list[bool | None]) -> IterationStats:
    iteration_stats = IterationStats()
    req_stats = RequestStateStats(arrival_time=0.0)
    req_stats.scheduled_ts = 0.1
    req_stats.first_token_ts = 0.5
    req_stats.last_token_ts = 2.0
    req_stats.num_generation_tokens = 10
    for idx, watermarked in enumerate(watermarked_values):
        iteration_stats.update_from_finished_request(
            finish_reason=FinishReason.STOP,
            request_id=f"req-{idx}",
            num_prompt_tokens=100,
            max_tokens_param=10,
            req_stats=req_stats,
            watermarked=watermarked,
        )
    return iteration_stats


def _samples(metric_name: str) -> dict[str, float]:
    """Map the ``watermarked`` label value to its counter value."""
    values: dict[str, float] = {}
    for metric in prometheus_client.REGISTRY.collect():
        if metric.name != metric_name:
            continue
        for sample in metric.samples:
            if sample.name == f"{metric_name}_total":
                values[sample.labels["watermarked"]] = sample.value
    return values


def test_request_watermarked_counter():
    config = VllmConfig(model_config=ModelConfig(model=TEST_MODEL))
    try:
        logger = PrometheusStatLogger(config)

        # Series are pre-created, so both label values export before any request.
        assert _samples("vllm:request_watermarked") == {"true": 0.0, "false": 0.0}

        logger.record(None, _finished_iteration_stats([True, False, None]))

        # The pooling request (None) is not counted under either label.
        assert _samples("vllm:request_watermarked") == {"true": 1.0, "false": 1.0}

        success_labels = {
            key
            for metric in prometheus_client.REGISTRY.collect()
            if metric.name == "vllm:request_success"
            for sample in metric.samples
            for key in sample.labels
        }
        # The new flag is its own counter, not a label on request_success.
        assert "watermarked" not in success_labels
    finally:
        unregister_vllm_metrics()
