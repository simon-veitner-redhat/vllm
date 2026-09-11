# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from io import BytesIO

import pytest
from fastapi import UploadFile

from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionRequest,
)
from vllm.entrypoints.speech_to_text.translation.protocol import TranslationRequest
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import StructuredOutputsParams


def test_chat_request_disables_watermarking():
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}], watermarking=False
    )

    params = request.to_sampling_params(max_tokens=1, default_sampling_params={})

    assert not params.watermarking


def test_completion_request_disables_watermarking():
    request = CompletionRequest(prompt="hello", watermarking=False)

    params = request.to_sampling_params(max_tokens=1)

    assert not params.watermarking


def test_responses_request_disables_watermarking():
    request = ResponsesRequest(input="hello", watermarking=False)

    params = request.to_sampling_params(default_max_tokens=1)

    assert not params.watermarking


def test_batch_chat_request_disables_watermarking():
    request = BatchChatCompletionRequest(
        messages=[[{"role": "user", "content": "hello"}]], watermarking=False
    )

    converted = request.to_chat_completion_request(request.messages[0])
    params = converted.to_sampling_params(max_tokens=1, default_sampling_params={})

    assert not params.watermarking


@pytest.mark.parametrize(
    "request_cls,kwargs",
    [
        (ChatCompletionRequest, {"messages": [{"role": "user", "content": "hi"}]}),
        (CompletionRequest, {"prompt": "hi"}),
        (
            BatchChatCompletionRequest,
            {"messages": [[{"role": "user", "content": "hi"}]]},
        ),
    ],
)
@pytest.mark.parametrize("watermarking", [True, False])
def test_best_of_is_rejected(request_cls, kwargs, watermarking):
    with pytest.raises(VLLMValidationError, match="best_of.*not supported"):
        request_cls(**kwargs, best_of=2, watermarking=watermarking)


@pytest.mark.parametrize(
    "api_request",
    [
        ChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}],
            use_beam_search=True,
            watermarking=False,
        ),
        CompletionRequest(prompt="hello", use_beam_search=True, watermarking=False),
    ],
)
def test_beam_request_disables_watermarking(api_request):
    params = api_request.to_beam_search_params(max_tokens=1, default_sampling_params={})

    assert not params.watermarking


@pytest.mark.parametrize("request_cls", [TranscriptionRequest, TranslationRequest])
def test_speech_request_disables_watermarking(request_cls):
    request = request_cls(
        file=UploadFile(file=BytesIO(), filename="audio.wav"), watermarking=False
    )

    sampling_params = request.to_sampling_params(default_max_tokens=1)
    beam_params = request.to_beam_search_params(default_max_tokens=1)

    assert not sampling_params.watermarking
    assert not beam_params.watermarking


def test_chat_request_preserves_watermarking_with_structured_outputs():
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "choose A or B"}],
        temperature=0.8,
        watermarking=True,
        structured_outputs=StructuredOutputsParams(choice=["A", "B"]),
    )

    params = request.to_sampling_params(max_tokens=8, default_sampling_params={})

    assert params.watermarking
    assert params.structured_outputs is not None
    assert params.structured_outputs.choice == ["A", "B"]


def test_completion_request_preserves_watermarking_with_structured_outputs():
    request = CompletionRequest(
        prompt="choose A or B",
        temperature=0.8,
        watermarking=True,
        structured_outputs=StructuredOutputsParams(choice=["A", "B"]),
    )

    params = request.to_sampling_params(max_tokens=8)

    assert params.watermarking
    assert params.structured_outputs is not None
    assert params.structured_outputs.choice == ["A", "B"]


def test_responses_request_preserves_watermarking_with_structured_outputs():
    request = ResponsesRequest(
        input="choose A or B",
        temperature=0.8,
        watermarking=True,
        structured_outputs=StructuredOutputsParams(choice=["A", "B"]),
    )

    params = request.to_sampling_params(default_max_tokens=8)

    assert params.watermarking
    assert params.structured_outputs is not None
    assert params.structured_outputs.choice == ["A", "B"]
