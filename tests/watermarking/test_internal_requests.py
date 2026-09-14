# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Requests vLLM issues for itself are never watermarked."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from vllm.entrypoints.generate.generative_scoring.serving import (
    GenerativeScoringRequest,
    ServingGenerativeScoring,
)
from vllm.entrypoints.speech_to_text.base.serving import SpeechToTextBaseServing
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.worker.gpu import warmup


class _Stop(Exception):
    pass


def test_language_detection_disables_watermarking():
    captured = None

    class EngineClient:
        def generate(self, prompt, sampling_params, request_id):
            nonlocal captured
            captured = sampling_params

            async def outputs():
                yield SimpleNamespace(
                    finished=True,
                    outputs=[SimpleNamespace(token_ids=[3])],
                )

            return outputs()

    class ModelCls:
        @staticmethod
        def get_language_detection_prompt(audio_chunk, asr_config):
            return {"type": "token", "prompt_token_ids": [1]}

        @staticmethod
        def get_language_token_ids(tokenizer):
            return [3, 4]

        @staticmethod
        def parse_language_detection_output(token_ids, tokenizer):
            return "en"

    serving = object.__new__(SpeechToTextBaseServing)
    serving.__dict__["model_cls"] = ModelCls
    serving.asr_config = SimpleNamespace()
    serving.tokenizer = SimpleNamespace()
    serving.engine_client = EngineClient()

    language = asyncio.run(serving._detect_language(np.zeros(4), "req-0"))

    assert language == "en"
    assert captured is not None
    assert not captured.watermarking


def test_generative_scoring_disables_watermarking():
    captured = None

    class EngineClient:
        errored = False

        def check_admission(self, num_requests):
            return None

        def generate(self, prompt, sampling_params, request_id, **kwargs):
            nonlocal captured
            captured = sampling_params
            raise _Stop

    async def check_model(request):
        return None

    async def build_prompts(request, tokenizer, max_model_len):
        return [{"type": "token", "prompt_token_ids": [1]}], [1]

    serving = object.__new__(ServingGenerativeScoring)
    serving.engine_client = EngineClient()
    serving.renderer = SimpleNamespace(tokenizer=SimpleNamespace())
    serving.model_config = SimpleNamespace(get_vocab_size=lambda: 32, max_model_len=128)
    serving.request_logger = None
    serving._check_model = check_model
    serving._maybe_get_adapters = lambda request: None
    serving._base_request_id = lambda raw_request, default=None: "req-0"
    serving._build_prompts = build_prompts
    serving._log_inputs = lambda *args, **kwargs: None

    request = GenerativeScoringRequest(query="q", items=["a"], label_token_ids=[3, 4])

    with pytest.raises(_Stop):
        asyncio.run(serving.create_generative_scoring(request, None))

    assert captured is not None
    assert not captured.watermarking


def test_kv_transfer_abort_stub_disables_watermarking():
    captured = None

    class EngineCore:
        async def add_request_async(self, request):
            nonlocal captured
            captured = request.sampling_params

        def shutdown(self, timeout=None):
            pass

    engine = object.__new__(AsyncLLM)
    engine.engine_core = EngineCore()

    asyncio.run(
        engine.notify_kv_transfer_request_rejected(
            "req-0",
            {"do_remote_prefill": True},
        )
    )

    assert captured is not None
    assert not captured.watermarking


def test_mixed_prefill_decode_warmup_disables_watermarking(monkeypatch):
    captured = None

    def new_request_data(*args, **kwargs):
        nonlocal captured
        captured = kwargs["sampling_params"]
        raise _Stop

    monkeypatch.setattr(warmup, "NewRequestData", new_request_data)

    kv_cache_spec = SimpleNamespace(block_size=16)
    model_runner = SimpleNamespace(
        is_pooling_model=False,
        max_num_reqs=8,
        max_model_len=128,
        model_state=SimpleNamespace(max_encoder_len=0),
        vllm_config=SimpleNamespace(num_lookahead_tokens=0),
        kv_cache_config=SimpleNamespace(
            num_blocks=1024,
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=kv_cache_spec)],
        ),
    )

    with pytest.raises(_Stop):
        warmup.run_mixed_prefill_decode_warmup(
            model_runner,
            lambda scheduler_output: None,
            lambda grammar_output: None,
            8,
        )

    assert captured is not None
    assert not captured.watermarking
