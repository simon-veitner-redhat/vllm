# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.llm_engine import LLMEngine


def _validate(params: SamplingParams, configured: bool = True) -> None:
    processor = object.__new__(InputProcessor)
    processor.vllm_config = SimpleNamespace(
        watermark_config=object() if configured else None,
        reasoning_config=None,
    )
    processor.model_config = SimpleNamespace(
        return_sampling_mask=False,
        enable_trace_replay=True,
    )
    processor.speculative_config = None
    processor.structured_outputs_config = None
    processor.renderer = SimpleNamespace(tokenizer=None)
    with patch.object(SamplingParams, "verify"):
        processor._validate_params(params, ("generate",))


def _engine_core_request(params: SamplingParams) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id="request",
        prompt_token_ids=[1],
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def _input_processor() -> InputProcessor:
    processor = object.__new__(InputProcessor)
    processor.vllm_config = SimpleNamespace(watermark_config=object())
    return processor


@pytest.mark.parametrize(
    ("params", "message"),
    [
        (SamplingParams(temperature=0), "requires stochastic sampling"),
        (SamplingParams(trace_decode_token_ids=[1]), "Trace replay"),
    ],
)
def test_incompatible_watermarked_requests_are_rejected(params, message):
    with pytest.raises(VLLMValidationError, match=message):
        _validate(params)


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(temperature=0, watermarking=False),
        SamplingParams(trace_decode_token_ids=[1], watermarking=False),
    ],
)
def test_incompatible_modes_are_allowed_when_watermarking_is_disabled(params):
    _validate(params)


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(temperature=0),
        SamplingParams(trace_decode_token_ids=[1]),
    ],
)
def test_watermarking_checks_are_inactive_without_engine_config(params):
    _validate(params, configured=False)


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(seed=42),
        SamplingParams(n=2),
        SamplingParams(n=2, seed=42),
    ],
)
def test_seeded_and_parallel_sampling_are_allowed(params):
    _validate(params)


def test_direct_engine_request_is_rejected_before_id_mutation():
    engine = object.__new__(LLMEngine)
    engine.input_processor = _input_processor()
    request = _engine_core_request(SamplingParams(trace_decode_token_ids=[1]))

    with pytest.raises(VLLMValidationError, match="Trace replay"):
        engine.add_request(
            request.request_id,
            request,
            SamplingParams(watermarking=False),
        )

    assert request.request_id == "request"
    assert request.external_req_id is None


def test_direct_engine_request_uses_embedded_opt_out(monkeypatch):
    engine = object.__new__(LLMEngine)
    engine.input_processor = _input_processor()

    def accept(_request):
        raise RuntimeError("accepted")

    monkeypatch.setattr(engine.input_processor, "assign_request_id", accept)
    request = _engine_core_request(SamplingParams(temperature=0, watermarking=False))

    with pytest.raises(RuntimeError, match="accepted"):
        engine.add_request(
            request.request_id,
            request,
            SamplingParams(temperature=0),
        )


def test_direct_async_engine_request_is_rejected_before_side_effects():
    engine = object.__new__(AsyncLLM)
    engine.engine_core = SimpleNamespace(
        resources=SimpleNamespace(engine_dead=False), shutdown=lambda **kwargs: None
    )
    engine.output_handler = None
    engine.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=False)
    )
    engine.input_processor = _input_processor()
    request = _engine_core_request(SamplingParams(trace_decode_token_ids=[1]))

    async def add_request() -> None:
        await engine.add_request(
            request.request_id,
            request,
            SamplingParams(watermarking=False),
        )

    with pytest.raises(VLLMValidationError, match="Trace replay"):
        asyncio.run(add_request())

    assert request.request_id == "request"
    assert request.external_req_id is None
    assert engine.output_handler is None
