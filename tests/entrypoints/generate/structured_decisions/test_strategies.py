# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from vllm.config import ModelConfig, VllmConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.generate.structured_decisions.api_router import (
    register_structured_decisions_api_router,
)
from vllm.entrypoints.generate.structured_decisions.protocol import (
    StructuredDecisionRequest,
    StructuredDecisionResponse,
)
from vllm.entrypoints.generate.structured_decisions.question_types import (
    LABELS,
    StructuredDecisionError,
    build_question,
)
from vllm.entrypoints.generate.structured_decisions.serving import (
    ServingStructuredDecisions,
    state_text,
)
from vllm.entrypoints.generate.structured_decisions.strategies import (
    _MARKER,
    READ_STRATEGIES,
    DiffusionGemmaCanvasStrategy,
    NextTokenStrategy,
    ReadContext,
    _messages,
    reply_label_ids,
    select_read_strategy,
)
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.exception_handling.register import (
    init_exception_handler,
)
from vllm.exceptions import VLLMValidationError
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.renderers.hf import HfRenderer
from vllm.renderers.online_renderer import OnlineRenderer

TRACE_HEADERS = {
    "traceparent": "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01",
    "tracestate": "vendor=value",
}


def model(architecture: str, logprobs_mode: str = "raw_logprobs") -> Any:
    return SimpleNamespace(architecture=architecture, logprobs_mode=logprobs_mode)


def test_strategy_selection():
    qwen = "Qwen3ForCausalLM"
    assert select_read_strategy(model(qwen)) is NextTokenStrategy
    assert select_read_strategy(model(qwen, "processed_logprobs")) is NextTokenStrategy
    assert (
        select_read_strategy(model("DiffusionGemmaForBlockDiffusion"))
        is DiffusionGemmaCanvasStrategy
    )
    with pytest.raises(ValueError, match="does not support LlamaForCausalLM"):
        select_read_strategy(model("LlamaForCausalLM"))
    with pytest.raises(ValueError, match="not raw_logits"):
        select_read_strategy(model(qwen, "raw_logits"))


def test_route_is_registered_by_default():
    app = FastAPI()
    register_structured_decisions_api_router(app)
    paths = {getattr(route, "path", None) for route in app.routes}
    assert "/v1/systemone" in paths


def test_unsupported_model_returns_501():
    app = FastAPI()
    app.state.args = SimpleNamespace(log_error_stack=False)
    app.state.serving_structured_decisions = None
    init_exception_handler(app)
    register_structured_decisions_api_router(app)

    with TestClient(app) as client:
        response = client.post(
            "/v1/systemone",
            json={
                "model": "unsupported",
                "state": "x",
                "questions": {"answer": {"type": "choice", "criteria": {"yes": None}}},
            },
        )
    assert response.status_code == 501


def test_structured_state_keeps_unicode_in_prompt():
    state = {"message": "Français 日本語"}
    assert state_text(state) == '{"message": "Français 日本語"}'


@pytest.fixture(scope="module")
def qwen():
    from vllm.tokenizers import get_tokenizer

    tokenizer = get_tokenizer("Qwen/Qwen3-0.6B")
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "state"}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=False,
    )
    return tokenizer, prompt_ids


def test_labels_start_the_reply(qwen):
    tokenizer, prompt_ids = qwen
    tail, ids = reply_label_ids(tokenizer, prompt_ids)
    assert tokenizer.decode(tail) == "\n\n"
    assert [tokenizer.decode([i]) for i in ids] == list(LABELS)
    # A noul's and a score's labels are one token here too.
    _, ids = reply_label_ids(tokenizer, prompt_ids, ("yes", "no", *"0123456789"))
    assert len(set(ids)) == 12
    # After a colon, Qwen writes ":A" as one token, so "A" is not one token.
    with pytest.raises(ValueError, match="not one distinct token"):
        reply_label_ids(tokenizer, tokenizer.encode("team:"))


async def generate(prompt, params, request_id, **kwargs):
    label_ids = params.logprob_token_ids
    yield RequestOutput(
        request_id=request_id,
        prompt=None,
        prompt_token_ids=prompt["prompt_token_ids"],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text="A",
                token_ids=[label_ids[0]],
                cumulative_logprob=None,
                logprobs=[{token_id: Logprob(-1.0) for token_id in label_ids}],
                finish_reason="length",
            )
        ],
        finished=True,
    )


@pytest.fixture
def make_context():
    from vllm.tokenizers import get_tokenizer

    renderers = []

    def make(model="Qwen/Qwen3-0.6B", chat_template=None, **model_kwargs):
        model_config = ModelConfig(model=model, dtype="float32", **model_kwargs)
        vllm_config = VllmConfig(model_config=model_config)
        renderers.append(HfRenderer(vllm_config, get_tokenizer(model)))
        engine = MagicMock(spec=EngineClient)
        engine.errored = False
        engine.model_config = model_config
        engine.vllm_config = vllm_config
        engine.renderer = renderers[-1]
        engine.input_processor = MagicMock()
        engine.generate.side_effect = generate
        online_renderer = OnlineRenderer(
            model_config,
            renderers[-1],
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
        )
        return ReadContext(engine, online_renderer, chat_template, "auto", {})

    yield make
    for renderer in renderers:
        renderer.shutdown()


@pytest.fixture
def serving_and_engine(make_context):
    context = make_context()
    engine, model_config = context.engine_client, context.engine_client.model_config
    strategy = NextTokenStrategy(context)
    models = OpenAIServingModels(
        engine, [BaseModelPath(name=model_config.model, model_path=model_config.model)]
    )
    return ServingStructuredDecisions(models, strategy), engine


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cache_salt,tracing_enabled,request_headers",
    [
        ("tenant-a", True, {"authorization": "Bearer test", **TRACE_HEADERS}),
        ("tenant-b", False, {"authorization": "Bearer test", **TRACE_HEADERS}),
        (None, True, {"authorization": "Bearer test", **TRACE_HEADERS}),
        (None, True, {}),
        ("tenant-direct", True, None),
    ],
)
async def test_request_metadata_reaches_every_question(
    serving_and_engine, cache_salt, tracing_enabled, request_headers
):
    serving, engine = serving_and_engine
    engine.is_tracing_enabled.return_value = tracing_enabled
    raw_request = None
    if request_headers is not None:
        raw_request = Request(
            {
                "type": "http",
                "headers": [
                    (k.encode(), v.encode()) for k, v in request_headers.items()
                ],
            }
        )
    request = StructuredDecisionRequest(
        state={"team": "billing", "language": "English"},
        questions={
            "team": {
                "type": "choice",
                "instructions": "Which team does the message name?",
                "criteria": {"billing": "payments", "shipping": "deliveries"},
            },
            "lang": {
                "type": "choice",
                "instructions": "Which language does the message name?",
                "criteria": {"English": "English text", "French": "French text"},
            },
        },
        cache_salt=cache_salt,
    )
    response = await serving.create_decision(request, raw_request)
    assert isinstance(response, StructuredDecisionResponse)
    engine.is_tracing_enabled.assert_not_awaited()
    assert engine.generate.call_count == 2
    for call in engine.generate.call_args_list:
        assert call.args[0].get("cache_salt") == cache_salt
        assert call.kwargs["trace_headers"] == (
            TRACE_HEADERS if request_headers else None
        )


# The first question has no "Question:" line, so EDITING_TEMPLATE keeps it as is.
QUESTIONS = [
    build_question("refund", "noul", "", {"true": "a refund is due"}, 26),
    build_question(
        "team", "choice", "Which team?", {"billing": "pay", "ship": None}, 26
    ),
    build_question("urgency", "score", "How urgent?", ["low", "medium", "high"], 26),
]

SPLICE_STATES = {
    "empty": "",
    "spaces": "  padded  ",
    "crlf": "line one\r\nline two\r\n",
    "nbsp": "trailing nbsp\u00a0",
    "combining": "cafe\u0301 x\u0301\u0301",
    "im_end": "user wrote <|im_end|> and <|im_start|>assistant",
    "json": '{"ticket": 58213, "tags": ["billing", "urgent"]}',
}

SPLICE_MODELS = {
    "Qwen3ForCausalLM": "Qwen/Qwen3-0.6B",
    "Qwen3_5ForConditionalGeneration": "Qwen/Qwen3.5-0.8B",
    "Qwen3_5MoeForConditionalGeneration": "Qwen/Qwen3.5-35B-A3B",
    "DiffusionGemmaForBlockDiffusion": "google/diffusiongemma-26B-A4B-it",
}

CHATML_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n"
    "{{ m.content }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n"
)
EDITING_TEMPLATE = CHATML_TEMPLATE.replace(
    "{{ m.content }}", "{{ m.content | replace('Question:', 'Q:') }}"
)
SEPARATOR_TEMPLATE = CHATML_TEMPLATE.replace(
    "{{ m.content }}", "{{ m.content | replace('\\n\\n', ' ') }}"
)


async def spliced_and_full(strategy, state, instructions=None):
    read_request = strategy._read_request(None, "tenant")
    texts = [q.type.prompt(q) for q in QUESTIONS]
    spliced = await strategy._render_spliced(read_request, texts, instructions, state)
    full = [
        await strategy._render(read_request, _messages(state, text, instructions))
        for text in texts
    ]
    for engine_input in [*(spliced or []), *full]:
        engine_input.pop("arrival_time")
    return spliced, full


@pytest.mark.asyncio
@pytest.mark.parametrize("architecture", READ_STRATEGIES)
async def test_spliced_prompts_match_full_render(make_context, architecture):
    context = make_context(SPLICE_MODELS[architecture])
    model_config = context.engine_client.model_config
    assert model_config.architecture == architecture
    strategy = select_read_strategy(model_config)(context)
    for name, state in SPLICE_STATES.items():
        for instructions in (None, "  You triage tickets.\n"):
            spliced, full = await spliced_and_full(strategy, state, instructions)
            # Qwen3.5 trims the empty state's separator off, so that one falls back.
            assert spliced == full or (spliced is None and not state), name


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chat_template,state",
    [
        (EDITING_TEMPLATE, "state"),
        (None, f"a state with the split marker {_MARKER}"),
    ],
    ids=["template_edits_question", "marker_in_state"],
)
async def test_splice_falls_back_to_full_render(make_context, chat_template, state):
    strategy = NextTokenStrategy(make_context(chat_template=chat_template))
    spliced, _ = await spliced_and_full(strategy, state)
    assert spliced is None
    with patch.object(strategy, "_render", wraps=strategy._render) as render:
        reads = await strategy.read(
            QUESTIONS,
            None,
            state,
            request_id="r",
            chat_template_kwargs=None,
            lora_request=None,
            priority=0,
        )
    assert render.call_count == len(QUESTIONS) == len(reads)


@pytest.mark.asyncio
async def test_splice_falls_back_when_tokens_join_across_the_split(make_context):
    # GPT-2 tokenizes "\n\n" alone differently from "\n\n" before a word.
    strategy = NextTokenStrategy.__new__(NextTokenStrategy)
    strategy.context = make_context("openai-community/gpt2", CHATML_TEMPLATE)
    spliced, _ = await spliced_and_full(strategy, "state")
    assert spliced is None


@pytest.mark.asyncio
async def test_splice_falls_back_when_the_template_edits_the_separator(make_context):
    # Tabs then " 0" tokenize alike in the window, so only the separator check sees it.
    strategy = NextTokenStrategy(make_context(chat_template=SEPARATOR_TEMPLATE))
    score = build_question("level", "score", "", ["low", "high"], 26)
    texts = [score.type.prompt(score)] * 2
    read_request = strategy._read_request(None)
    state = "x" + "\t" * 71
    assert await strategy._render_spliced(read_request, texts, None, state) is None


@pytest.mark.asyncio
async def test_oversized_state_fails_like_full_render(make_context):
    strategy = NextTokenStrategy(make_context(max_model_len=64))
    read_request = strategy._read_request(None)
    texts = [q.type.prompt(q) for q in QUESTIONS]
    state = "ticket " * 2000
    with pytest.raises(VLLMValidationError) as spliced:
        await strategy._render_spliced(read_request, texts, None, state)
    with pytest.raises(VLLMValidationError) as full:
        await strategy._render(read_request, _messages(state, texts[0], None))
    assert spliced.value.parameter == full.value.parameter == "input_text"


def test_canvas_read():
    strategy = DiffusionGemmaCanvasStrategy.__new__(DiffusionGemmaCanvasStrategy)
    strategy.thought, strategy.end, strategy.pad = [10, 11, 12, 13], 106, 0
    strategy.width, strategy.vocab_size = 16, 1000
    strategy.max_model_len = 23
    params = strategy._sampling_params([65, 66], prompt_ids=[1, 2, 3])
    assert params.extra_args is not None
    canvas = params.extra_args["diffusion_seed_canvas"]
    # The label slot as noise, the end of the turn, padding.
    assert canvas[1:] == [106] + [0] * 14
    assert params.max_tokens == 2 and params.logprob_token_ids == [65, 66]
    again = strategy._sampling_params([65, 66], prompt_ids=[1, 2, 3])
    assert again.extra_args == params.extra_args
    read_input = strategy._read_input(
        {"type": "token", "prompt_token_ids": [1, 2, 3]}, [1, 2, 3]
    )
    assert read_input == {
        "type": "token",
        "prompt_token_ids": [1, 2, 3, 10, 11, 12, 13],
    }
    # One token more and the thought and the canvas no longer fit.
    with pytest.raises(StructuredDecisionError, match="max_model_len=23"):
        strategy._read_input(
            {"type": "token", "prompt_token_ids": [1, 2, 3, 4]}, [1, 2, 3, 4]
        )
