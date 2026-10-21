# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SlotLMStrategy: the /v1/systemone reads of Base Labs mammoth-1 and
elephant-1 must match their reference engine (qev, OneJev at 81ce62f with
serve.patch) byte for byte.

The golden prompts below were rendered by qev's own `DecisionEngine` methods
with the checkpoints' tokenizer. Set QEV_SRC to a OneJev checkout to also
compare against qev directly on random requests.
"""

import importlib
import json
import os
import random
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.generate.structured_decisions.protocol import (
    StructuredDecisionRequest,
)
from vllm.entrypoints.generate.structured_decisions.serving import parse_questions
from vllm.entrypoints.openai.decisions import slot_prompt
from vllm.entrypoints.openai.decisions.question_types import StructuredDecisionError
from vllm.entrypoints.openai.decisions.strategies import (
    ReadContext,
    SlotLMStrategy,
    select_read_strategy,
)
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.tokenizers import get_tokenizer

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

#: Hub id or local directory with the mammoth-1 / elephant-1 tokenizer (both
#: ship the same tokenizer.json and chat template).
TOKENIZER = os.environ.get("SLOT_TOKENIZER", "baselabs/elephant-1")

OVERRIDES = {
    "attention": {
        "full_attention": "block_bidirectional",
        "linear_attention": "causal",
    },
    "readout": "slot_lm",
    "head_dim": 256,
    "n_slots": 256,
}

#: The state and questions qev's DecisionEngine probes at load.
PROBE_STATE = {
    "ticket": "My card was charged twice for the same order and support has "
    "not replied in 3 days.",
    "customer": {"tenure_months": 14, "vip": False},
}
PROBE_QUESTIONS = {
    "a": {"type": "noul", "instructions": "Is the customer asking for money back?"},
    "b": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "billing": "charges and refunds",
            "technical": "bugs",
            "sales": "pricing",
        },
    },
    "c": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["calm", "irritated", "angry"],
    },
}

_PREFIX = (
    "<|im_start|>system\nApply the question to the state. Choose exactly one of "
    "the listed options. Respond with only its uppercase letter, with no "
    "explanation or reasoning.<|im_end|>\n<|im_start|>user\n<state>\n{\n"
    '  "ticket": "My card was charged twice for the same order and support has '
    'not replied in 3 days.",\n  "customer": {\n    "tenure_months": 14,\n'
    '    "vip": false\n  }\n}\n</state>\n\n'
)
_TAIL = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
#: qev's prompts for PROBE_QUESTIONS, and the suffix token counts it cut.
QEV_PROMPTS = {
    "a": (
        _PREFIX + "Question: Is the customer asking for money back?\n\nOptions:\n"
        "A. yes: the statement is true / the answer is yes\n"
        "B. no: the statement is false / the answer is no\n\n"
        "Answer with one letter: A, B." + _TAIL,
        60,
    ),
    "b": (
        _PREFIX + "Question: Which team should handle this?\n\nOptions:\n"
        "A. billing: charges and refunds\nB. technical: bugs\nC. sales: pricing"
        "\n\nAnswer with one letter: A, B, C." + _TAIL,
        52,
    ),
    "c": (
        _PREFIX + "Rate the state: How frustrated is the customer?\n\nOptions:\n"
        "A. level 0: calm\nB. level 1: irritated\nC. level 2: angry\n\n"
        "Answer with one letter: A, B, C." + _TAIL,
        58,
    ),
}
#: qev's shared prefix for PROBE_STATE, ending in `</`, `state`, `>`, `ĊĊ`.
QEV_PREFIX_LEN = 99


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer(TOKENIZER)


def make_strategy(tokenizer, *, blocks=True, head_dtype=torch.float32, hybrid=True):
    model_config = SimpleNamespace(
        model="baselabs/elephant-1",
        hf_config=SimpleNamespace(modeling_overrides=OVERRIDES),
        request_attention_blocks=blocks,
        head_dtype=head_dtype,
        max_model_len=32768,
        is_hybrid=hybrid,
        architecture="Qwen3_5ForSlotDecision",
        logprobs_mode="raw_logprobs",
    )
    engine = MagicMock(spec=EngineClient)
    engine.model_config = model_config
    engine.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=True)
    )

    async def generate(prompt, params, request_id, **kwargs):
        label_ids = params.logprob_token_ids or [0]
        # The first slot gets twice the weight of each other slot.
        logprobs = {t: Logprob(-2.0) for t in label_ids}
        logprobs[label_ids[0]] = Logprob(-2.0 + 0.6931471805599453)
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
                    logprobs=[logprobs],
                    finish_reason="length",
                )
            ],
            finished=True,
        )

    engine.generate.side_effect = generate
    online_renderer = SimpleNamespace(
        renderer=SimpleNamespace(get_tokenizer=lambda: tokenizer)
    )
    context = ReadContext(engine, online_renderer, None, "auto", {})
    return SlotLMStrategy(context), engine


async def read(strategy, request: StructuredDecisionRequest):
    questions = parse_questions(request, strategy.limits())
    reads = await strategy.read(
        questions,
        request.instructions,
        strategy.render_state(request.state),
        request_id="r",
        chat_template_kwargs=request.chat_template_kwargs,
        lora_request=None,
        priority=0,
        cache_salt=request.cache_salt,
    )
    return questions, reads


def test_strategy_is_registered():
    model = SimpleNamespace(
        architecture="Qwen3_5ForSlotDecision", logprobs_mode="raw_logprobs"
    )
    assert select_read_strategy(model, ("generate",)) is SlotLMStrategy


def test_slot_labels_are_single_distinct_tokens(tokenizer):
    strategy, _ = make_strategy(tokenizer)
    ids = strategy.slot_ids
    assert len(ids) == len(set(ids)) == 256
    assert slot_prompt.SLOT_LABELS[-1] == "JU"
    assert min(ids) == 32 and max(ids) == 213977
    assert ids[0] == tokenizer.encode("A", add_special_tokens=False)[0]


@pytest.mark.asyncio
async def test_reads_match_qev_byte_for_byte(tokenizer):
    strategy, engine = make_strategy(tokenizer)
    request = StructuredDecisionRequest(
        state=PROBE_STATE, questions=PROBE_QUESTIONS, cache_salt="salt"
    )
    questions, reads = await read(strategy, request)

    # The 99-token prefix is too short to be worth a warm-up.
    question_calls = engine.generate.call_args_list
    assert len(question_calls) == 3
    for q, call in zip(questions, question_calls):
        prompt, params = call.args[0], call.args[1]
        text, suffix_len = QEV_PROMPTS[q.id]
        ids = prompt["prompt_token_ids"]
        assert tokenizer.decode(ids) == text
        assert ids == tokenizer.encode(text, add_special_tokens=False)
        assert len(ids) == QEV_PREFIX_LEN + suffix_len
        assert prompt["cache_salt"] == "salt"
        assert params.max_tokens == 1 and params.temperature == 0.0
        assert params.extra_args == {
            "attention_blocks": [
                [0, QEV_PREFIX_LEN - 1],
                [QEV_PREFIX_LEN, len(ids) - 1],
            ]
        }
        assert params.logprob_token_ids == strategy.slot_ids[: len(q.options)]

    # Slot A is the first option: yes, the first choice, level 0.
    by_id = {q.id: (q, r) for q, r in zip(questions, reads)}
    q, r = by_id["a"]
    assert r.probs == pytest.approx([2 / 3, 1 / 3])
    assert q.type.answer(q, r.probs, r.label_mass)["noul"] == pytest.approx(2 / 3)
    q, r = by_id["b"]
    assert q.type.answer(q, r.probs, r.label_mass)["choice"] == "billing"
    q, r = by_id["c"]
    assert r.probs == pytest.approx([0.5, 0.25, 0.25])
    assert r.attention_blocks is True


@pytest.mark.asyncio
async def test_warm_up_prefills_exactly_the_prefix_block(tokenizer):
    strategy, engine = make_strategy(tokenizer)
    # Both questions start with the same words, so the token LCP runs past
    # the prefix block.
    questions = {
        "x": {"type": "noul", "instructions": "Is the ticket about billing?"},
        "y": {"type": "noul", "instructions": "Is the ticket about shipping?"},
    }
    request = StructuredDecisionRequest(state="word " * 200, questions=questions)
    await read(strategy, request)
    warm, *reads = engine.generate.call_args_list
    assert warm.args[2] == "r-prefix"
    warm_ids = warm.args[0]["prompt_token_ids"]
    prefix_len = len(warm_ids)
    assert tokenizer.decode(warm_ids).endswith("</state>\n\n")
    assert warm.args[1].extra_args == {"attention_blocks": [[0, prefix_len - 1]]}
    for call in reads:
        ids = call.args[0]["prompt_token_ids"]
        assert ids[:prefix_len] == warm_ids
        assert call.args[1].extra_args["attention_blocks"][0] == [0, prefix_len - 1]


@pytest.mark.asyncio
async def test_many_options_use_two_letter_slots(tokenizer):
    strategy, engine = make_strategy(tokenizer)
    criteria = {f"option {i}": {"id": i} if i % 2 else None for i in range(255)}
    request = StructuredDecisionRequest(
        state="plain text",
        questions={"many": {"type": "choice", "criteria": criteria}},
    )
    _, reads = await read(strategy, request)
    prompt, params = engine.generate.call_args.args[:2]
    text = tokenizer.decode(prompt["prompt_token_ids"])
    assert "<state>\nplain text\n</state>\n\nQuestion: Which option applies" in text
    assert '\nAA. option 26\nAB. option 27: {\n  "id": 27\n}\n' in text
    assert "Answer with one label: A, B, C," in text
    assert len(params.logprob_token_ids) == 255
    assert len(reads[0].probs) == 255


@pytest.mark.asyncio
async def test_causal_mode_sends_no_blocks(tokenizer, monkeypatch):
    monkeypatch.setenv("VLLM_MODELING_OVERRIDES", "causal")
    strategy, engine = make_strategy(tokenizer, blocks=False)
    request = StructuredDecisionRequest(state=PROBE_STATE, questions=PROBE_QUESTIONS)
    _, reads = await read(strategy, request)
    assert all(
        call.args[1].extra_args is None for call in engine.generate.call_args_list
    )
    assert all(r.attention_blocks is False for r in reads)


def test_refuses_to_start_without_mask_or_fp32_head(tokenizer):
    with pytest.raises(ValueError, match="block-bidirectional"):
        make_strategy(tokenizer, blocks=False)
    with pytest.raises(ValueError, match="fp32 LM head"):
        make_strategy(tokenizer, head_dtype=torch.bfloat16)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [
        {"instructions": "Be strict."},
        {"chat_template_kwargs": {"enable_thinking": True}},
    ],
)
async def test_refuses_untrained_prompt_options(tokenizer, extra):
    strategy, engine = make_strategy(tokenizer)
    request = StructuredDecisionRequest(
        state=PROBE_STATE, questions=PROBE_QUESTIONS, **extra
    )
    with pytest.raises(StructuredDecisionError):
        await read(strategy, request)
    engine.generate.assert_not_called()


@pytest.mark.asyncio
async def test_score_levels_are_capped(tokenizer):
    strategy, _ = make_strategy(tokenizer)
    request = StructuredDecisionRequest(
        state="s",
        questions={"s": {"type": "score", "criteria": [str(i) for i in range(11)]}},
    )
    with pytest.raises(StructuredDecisionError, match="at most 10 levels"):
        await read(strategy, request)


@pytest.mark.skipif("QEV_SRC" not in os.environ, reason="set QEV_SRC to OneJev")
@pytest.mark.asyncio
async def test_reads_match_qev_on_random_requests(tokenizer):
    """Tokens and block boundary of every read against qev's DecisionEngine."""
    sys.path.insert(0, os.environ["QEV_SRC"])
    qp = importlib.import_module("qev.prompt")
    qro = importlib.import_module("qev.readout")
    qschema = importlib.import_module("qev.schema")
    engine_cls = importlib.import_module("qev.engine").DecisionEngine

    qev = object.__new__(engine_cls)
    qev.tokenizer = tokenizer
    qev.style = qp.DEFAULT_STYLE
    qev.template_kwargs = {"enable_thinking": False}
    qev.readout = qro.Readout(qro.ReadoutConfig(attention="noncausal"))
    schema_types = {
        "noul": qschema.NoulQuestion,
        "choice": qschema.ChoiceQuestion,
        "score": qschema.ScoreQuestion,
    }

    rng = random.Random(0)
    words = ["alpha", " beta ", "Gamma\n", "délta", "", "  spaced  ", "x" * 40]

    def entry():
        r = rng.random()
        if r < 0.15:
            return None
        if r < 0.6:
            return rng.choice(words)
        return {"k": rng.choice(words), "n": [1, "é", None]}

    for _ in range(30):
        questions = {}
        for i in range(rng.randint(1, 4)):
            t = rng.choice(["noul", "choice", "score"])
            if t == "noul":
                crit = rng.choice([None, {"true": entry()}, {"false": entry()}])
            elif t == "choice":
                k = rng.choice([1, 2, 5, 27, 60])
                crit = {f"o{j}{rng.choice(words).strip()}": entry() for j in range(k)}
            else:
                crit = [f"l{j}{rng.choice(words)}" for j in range(rng.randint(2, 10))]
            questions[f"q{i}"] = {"type": t, "instructions": entry(), "criteria": crit}
        state = rng.choice(
            ["A plain ticket.", {"a": [1, 2], "é": "ü", "n": None}, None, [1, 2]]
        )

        strategy, engine = make_strategy(tokenizer)
        request = StructuredDecisionRequest(state=state, questions=questions)
        await read(strategy, request)
        calls = [
            c
            for c in engine.generate.call_args_list
            if not c.args[2].endswith("-prefix")
        ]

        rendered = [
            qp.render_question(qid, schema_types[q["type"]](**q))
            for qid, q in questions.items()
        ]
        prefix, suffixes, _ = qev._split(state, rendered)
        for call, suffix in zip(calls, suffixes):
            ids = call.args[0]["prompt_token_ids"]
            assert ids == prefix + suffix, json.dumps(questions)
            assert call.args[1].extra_args["attention_blocks"] == [
                [0, len(prefix) - 1],
                [len(prefix), len(ids) - 1],
            ]
