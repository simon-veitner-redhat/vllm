# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from io import BytesIO
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pybase64 as base64
import pytest
import torch
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from PIL import Image

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
    parse_questions,
)
from vllm.entrypoints.generate.structured_decisions.strategies import (
    _MARKER,
    READ_STRATEGIES,
    DiffusionGemmaCanvasStrategy,
    JointSchemaStrategy,
    NextTokenStrategy,
    ReadContext,
    _messages,
    reply_label_ids,
    select_read_strategy,
    state_text,
)
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.exception_handling.register import (
    init_exception_handler,
)
from vllm.exceptions import VLLMValidationError
from vllm.inputs import mm_input
from vllm.logprobs import Logprob
from vllm.multimodal.media import MediaConnector, MediaWithBytes
from vllm.outputs import (
    CompletionOutput,
    PoolingOutput,
    PoolingRequestOutput,
    RequestOutput,
)
from vllm.renderers.hf import HfRenderer
from vllm.renderers.online_renderer import OnlineRenderer

TRACE_HEADERS = {
    "traceparent": "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01",
    "tracestate": "vendor=value",
}


def model(architecture: str, logprobs_mode: str = "raw_logprobs") -> Any:
    return SimpleNamespace(architecture=architecture, logprobs_mode=logprobs_mode)


def test_strategy_selection():
    qwen, generate = "Qwen3ForCausalLM", ("generate",)
    assert select_read_strategy(model(qwen), generate) is NextTokenStrategy
    assert (
        select_read_strategy(model(qwen, "processed_logprobs"), generate)
        is NextTokenStrategy
    )
    assert (
        select_read_strategy(model("DiffusionGemmaForBlockDiffusion"), generate)
        is DiffusionGemmaCanvasStrategy
    )
    with pytest.raises(ValueError, match="does not support LlamaForCausalLM"):
        select_read_strategy(model("LlamaForCausalLM"), generate)
    with pytest.raises(ValueError, match="not raw_logits"):
        select_read_strategy(model(qwen, "raw_logits"), generate)
    # A generative architecture served as a pooling model has no read strategy.
    with pytest.raises(ValueError, match="need the generate task"):
        select_read_strategy(model(qwen), ("embed",))
    # Clef-Omni scores options with its head, so it reads no logprobs.
    clef = model("ClefOmniForJointSchema", "raw_logits")
    assert select_read_strategy(clef, ("token_classify",)) is JointSchemaStrategy


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
def engine_and_renderer(make_context):
    context = make_context()
    return context.engine_client, context.online_renderer


def serving(engine, strategy) -> ServingStructuredDecisions:
    name = engine.model_config.model
    models = OpenAIServingModels(engine, [BaseModelPath(name=name, model_path=name)])
    return ServingStructuredDecisions(models, strategy)


@pytest.fixture
def serving_and_engine(engine_and_renderer):
    engine, online_renderer = engine_and_renderer
    strategy = NextTokenStrategy(ReadContext(engine, online_renderer, None, "auto", {}))
    return serving(engine, strategy), engine


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


# Clef-Omni encodes its own prompt, so it never splices.
SPLICE_ARCHITECTURES = [
    a for a, s in READ_STRATEGIES.items() if issubclass(s, NextTokenStrategy)
]


@pytest.mark.asyncio
@pytest.mark.parametrize("architecture", SPLICE_ARCHITECTURES)
async def test_spliced_prompts_match_full_render(make_context, architecture):
    context = make_context(SPLICE_MODELS[architecture])
    model_config = context.engine_client.model_config
    assert model_config.architecture == architecture
    strategy = select_read_strategy(model_config, ("generate",))(context)
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


JOINT_SCHEMA_PROMPT = """<|im_start|>system
Read the complete state and schema. Decide every field jointly. Each answer \
must be exactly one of that field's allowed options.<|im_end|>
<|im_start|>user
STATE:
{"customer":{"tier":"gold"},"ticket":"Charged twice"}

SCHEMA FIELDS:

FIELD 1
ID: team
TYPE: choice
INSTRUCTION: Which team?
ALLOWED OPTIONS:
OPTION 1: {"description":"payments","option_id":"billing"}
OPTION 2: {"option_id":"security"}
OPTION 3: {"description":"deliveries","option_id":"shipping"}
END FIELD

FIELD 2
ID: urgent
TYPE: noul
INSTRUCTION: urgent
ALLOWED OPTIONS:
OPTION 1: {"description":"money is moving","option_id":"true"}
OPTION 2: {"description":"The proposition is false or the answer is no.",\
"option_id":"false"}
END FIELD

FIELD 3
ID: tone
TYPE: score
INSTRUCTION: How upset?
ALLOWED OPTIONS:
OPTION 1: {"description":"calm","option_id":"0"}
OPTION 2: {"description":"annoyed","option_id":"1"}
OPTION 3: {"description":"furious","option_id":"2"}
END FIELD

<|im_end|>
<|im_start|>assistant
<think>

</think>

JOINT SCHEMA DECISIONS:"""

JOINT_SCHEMA_REQUEST: dict[str, Any] = dict(
    state={"ticket": "Charged twice", "customer": {"tier": "gold"}},
    questions={
        "team": {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {
                "shipping": "deliveries",
                "billing": "payments",
                "security": None,
            },
        },
        "urgent": {"type": "noul", "criteria": {"true": "money is moving"}},
        "tone": {
            "type": "score",
            "instructions": "How upset?",
            "criteria": ["calm", "annoyed", "furious"],
        },
    },
)


@pytest.fixture
def joint_schema(qwen, engine_and_renderer):
    tokenizer, _ = qwen
    engine, online_renderer = engine_and_renderer
    # The probability the head gives each option, by option id.
    scores = {"billing": 0.7, "security": 0.1, "shipping": 0.2}
    scores |= {"true": 0.9, "false": 0.1, "0": 0.2, "1": 0.5, "2": 0.3}

    async def encode(prompt, params, request_id, **kwargs):
        ids = prompt["prompt_token_ids"]
        spans = params.extra_kwargs["joint_schema"]["option_spans"]
        options = [json.loads(tokenizer.decode(ids[s:e])) for s, e in spans]
        data = torch.tensor([scores[o["option_id"]] for o in options])
        yield PoolingRequestOutput(request_id, PoolingOutput(data), ids, 0, True)

    engine.encode.side_effect = encode
    strategy = JointSchemaStrategy(
        ReadContext(engine, online_renderer, None, "auto", {})
    )
    return serving(engine, strategy), engine, tokenizer


@pytest.mark.asyncio
async def test_joint_schema_reads_every_question_at_once(joint_schema):
    serving, engine, tokenizer = joint_schema
    response = await serving.create_decision(
        StructuredDecisionRequest(**JOINT_SCHEMA_REQUEST, cache_salt="salt")
    )
    assert isinstance(response, StructuredDecisionResponse)

    engine.encode.assert_called_once()
    engine.check_admission.assert_called_once_with(1)
    prompt, params = engine.encode.call_args.args[:2]
    ids = prompt["prompt_token_ids"]
    assert tokenizer.decode(ids) == JOINT_SCHEMA_PROMPT
    assert prompt["cache_salt"] == "salt"
    assert params.task == "token_classify"
    schema = params.extra_kwargs["joint_schema"]
    assert schema["question_types"] == [1, 0, 2]
    assert schema["option_counts"] == [3, 2, 3]
    assert [tokenizer.decode(ids[s:e]) for s, e in schema["question_spans"]] == [
        "Which team?",
        "urgent",
        "How upset?",
    ]

    # Options are read in the model's order and answered in the request's.
    team = response.answers["team"]
    assert team["choice"] == "billing"
    assert list(team["probabilities"]) == ["shipping", "billing", "security"]
    assert team["probabilities"] == pytest.approx(
        {"shipping": 0.2, "billing": 0.7, "security": 0.1}
    )
    assert response.answers["urgent"]["noul"] == pytest.approx(0.9)
    assert response.answers["tone"]["score"] == pytest.approx(1.1)
    assert response.usage.input_tokens == len(ids)
    assert response.usage.output_tokens == 0
    assert all(d.label_mass == 1.0 for d in response.diagnostics.values())


@pytest.mark.asyncio
async def test_joint_schema_rejects_instructions(joint_schema):
    serving, engine, _ = joint_schema
    response = await serving.create_decision(
        StructuredDecisionRequest(**JOINT_SCHEMA_REQUEST, instructions="Be strict.")
    )
    assert isinstance(response, ErrorResponse)
    assert response.error.code == 400
    engine.encode.assert_not_called()


def test_joint_schema_truncates_the_state(joint_schema):
    serving, _, _ = joint_schema
    strategy = serving.strategy
    request = StructuredDecisionRequest(**JOINT_SCHEMA_REQUEST)
    questions = parse_questions(request, strategy.limits())
    prompt, schema, _ = strategy.encode(questions, "word " * 100)
    strategy.max_length = len(prompt) - 50
    truncated, truncated_schema, _ = strategy.encode(questions, "word " * 100)
    # The state loses 50 tokens and the questions stay at the end.
    assert len(truncated) == strategy.max_length
    assert truncated[-200:] == prompt[-200:]
    assert truncated_schema == schema
    strategy.max_length = 100
    with pytest.raises(StructuredDecisionError, match="questions take"):
        strategy.encode(questions, "word")


@pytest.mark.asyncio
async def test_media_needs_a_model_that_takes_it(serving_and_engine):
    serving, engine = serving_and_engine
    request = StructuredDecisionRequest(
        **JOINT_SCHEMA_REQUEST, images=["data:image/png;base64,AAAA"]
    )
    response = await serving.create_decision(request)
    assert isinstance(response, ErrorResponse)
    assert response.error.code == 400
    assert "does not take image inputs" in response.error.message
    engine.generate.assert_not_called()


@pytest.mark.asyncio
async def test_joint_schema_reads_media_before_the_state(joint_schema, monkeypatch):
    serving, engine, tokenizer = joint_schema
    strategy = serving.strategy
    strategy.__dict__["_connector"] = MediaConnector()
    image_pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    rendered: dict[str, Any] = {}

    async def process_for_engine_async(prompt, arrival_time):
        rendered.update(prompt)
        # The image's placeholder expands to three tokens.
        ids = [
            token
            for pad in prompt["prompt_token_ids"]
            for token in ([pad] * 3 if pad == image_pad else [pad])
        ]
        return mm_input(ids, {}, {}, {}, cache_salt=prompt.get("cache_salt"))

    renderer = strategy.context.online_renderer.renderer
    monkeypatch.setattr(renderer, "process_for_engine_async", process_for_engine_async)
    buffer = BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format="PNG")
    image = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    response = await serving.create_decision(
        StructuredDecisionRequest(**JOINT_SCHEMA_REQUEST, images=[image])
    )
    assert isinstance(response, StructuredDecisionResponse)

    placeholder = tokenizer.encode(
        "<|vision_start|><|image_pad|><|vision_end|>\n", add_special_tokens=False
    )
    assert rendered["prompt_token_ids"] == strategy.prefix + placeholder
    assert rendered["multi_modal_data"]["image"][0].size == (8, 8)
    assert rendered["mm_processor_kwargs"] == {
        "images_kwargs": {"size": strategy.IMAGE_SIZE}
    }
    prompt = engine.encode.call_args.args[0]
    assert prompt["type"] == "multimodal"
    ids = prompt["prompt_token_ids"]
    expanded = len(strategy.prefix) + len(placeholder) + 2
    assert ids[:expanded].count(image_pad) == 3
    state_and_questions = JOINT_SCHEMA_PROMPT.split("STATE:\n", 1)[1]
    assert tokenizer.decode(ids[expanded:]) == state_and_questions
    assert response.answers["team"]["choice"] == "billing"
    assert response.usage.input_tokens == len(ids)


@pytest.mark.asyncio
async def test_joint_schema_hears_video_soundtracks(joint_schema, monkeypatch):
    serving, engine, tokenizer = joint_schema
    strategy = serving.strategy
    video = MediaWithBytes(
        (np.zeros((4, 8, 8, 3), np.uint8), {"fps": 2.0}), b"video-bytes"
    )
    track = (np.zeros(160, np.float32), 16000)
    rendered: dict[str, Any] = {}

    class FakeConnector:
        async def fetch_image_async(self, url):
            raise AssertionError("no images in this test")

        async def load_from_url_async(self, url, media_io, *, fetch_timeout=None):
            return video

        async def fetch_audio_async(self, url):
            if url == "https://example.org/silent.mp4":
                raise ValueError("No audio stream found.")
            return track

    strategy.__dict__["_connector"] = FakeConnector()

    async def process_for_engine_async(prompt, arrival_time):
        rendered.clear()
        rendered.update(prompt)
        return mm_input(prompt["prompt_token_ids"], {}, {}, {})

    renderer = strategy.context.online_renderer.renderer
    monkeypatch.setattr(renderer, "process_for_engine_async", process_for_engine_async)

    response = await serving.create_decision(
        StructuredDecisionRequest(
            **JOINT_SCHEMA_REQUEST, videos=["https://example.org/talking.mp4"]
        )
    )
    assert isinstance(response, StructuredDecisionResponse)
    assert rendered["multi_modal_data"] == {"video": [video], "audio": [track]}
    assert rendered["mm_processor_kwargs"] == {
        "images_kwargs": {"size": strategy.IMAGE_SIZE},
        "use_audio_in_video": True,
    }

    # A video without a decodable soundtrack is watched silently.
    response = await serving.create_decision(
        StructuredDecisionRequest(
            **JOINT_SCHEMA_REQUEST, videos=["https://example.org/silent.mp4"]
        )
    )
    assert isinstance(response, StructuredDecisionResponse)
    assert rendered["multi_modal_data"] == {"video": [video]}
    assert rendered["mm_processor_kwargs"]["use_audio_in_video"] is False

    # Audio clips cannot be read with videos that have soundtracks.
    response = await serving.create_decision(
        StructuredDecisionRequest(
            **JOINT_SCHEMA_REQUEST,
            videos=["https://example.org/talking.mp4"],
            audio=["https://example.org/clip.wav"],
        )
    )
    assert isinstance(response, ErrorResponse)
    assert response.error.code == 400
