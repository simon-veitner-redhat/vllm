# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Detection of Base Labs checkpoints (mammoth-1, elephant-1) from the
`modeling_overrides` block of their config.json."""

import copy

import pytest
from transformers.models.qwen3_5 import Qwen3_5Config

from vllm.transformers_utils.model_arch_config_convertor import (
    Qwen3_5ModelArchConfigConvertor,
)
from vllm.transformers_utils.modeling_overrides import parse_modeling_overrides

pytestmark = pytest.mark.cpu_test

#: As shipped in both checkpoints' config.json.
OVERRIDES = {
    "attention": {
        "full_attention": "block_bidirectional",
        "linear_attention": "causal",
    },
    "readout": "slot_lm",
    "head_dim": 256,
    "n_slots": 256,
}


def test_parse():
    overrides = parse_modeling_overrides(OVERRIDES)
    assert overrides is not None
    assert overrides.block_attention and overrides.n_slots == 256
    assert parse_modeling_overrides(None) is None
    causal = parse_modeling_overrides({"attention": {"full_attention": "causal"}})
    assert causal is not None and not causal.block_attention


@pytest.mark.parametrize(
    "change,match",
    [
        ({"readout": "head"}, "unsupported readout"),
        ({"attention": {"full_attention": "noncausal"}}, "full_attention"),
        ({"attention": {"linear_attention": "block_bidirectional"}}, "linear"),
        ({"attention": {"sliding_attention": "causal"}}, "attention keys"),
        ({"temperature": 2.0}, "keys"),
        ({"n_slots": 300}, "n_slots"),
    ],
)
def test_parse_rejects(change, match):
    with pytest.raises(ValueError, match=match):
        parse_modeling_overrides({**OVERRIDES, **change})


def make_config(overrides=OVERRIDES) -> Qwen3_5Config:
    config = Qwen3_5Config()
    config.architectures = ["Qwen3_5ForConditionalGeneration"]
    if overrides is not None:
        config.modeling_overrides = copy.deepcopy(overrides)
    return config


def convert(config: Qwen3_5Config):
    return Qwen3_5ModelArchConfigConvertor(config, config.text_config).convert()


def test_detects_block_attention_and_slot_readout(monkeypatch):
    monkeypatch.delenv("VLLM_MODELING_OVERRIDES", raising=False)
    config = make_config()
    arch = convert(config)
    assert arch.request_attention_blocks
    assert arch.architectures == ["Qwen3_5ForSlotDecision"]
    assert config.head_dtype == "float32"
    # The flag lives on the text config too, for text-only sub-configs, and
    # converting again keeps it.
    assert config.text_config.request_attention_blocks
    assert convert(config).request_attention_blocks


def test_explicit_head_dtype_is_kept(monkeypatch):
    monkeypatch.delenv("VLLM_MODELING_OVERRIDES", raising=False)
    config = make_config()
    config.head_dtype = "model"
    convert(config)
    assert config.head_dtype == "model"


def test_causal_mode_keeps_the_readout(monkeypatch):
    monkeypatch.setenv("VLLM_MODELING_OVERRIDES", "causal")
    config = make_config()
    arch = convert(config)
    assert not arch.request_attention_blocks
    assert arch.architectures == ["Qwen3_5ForSlotDecision"]


def test_ignore_mode_serves_stock_qwen3_5(monkeypatch):
    monkeypatch.setenv("VLLM_MODELING_OVERRIDES", "ignore")
    config = make_config()
    arch = convert(config)
    assert not arch.request_attention_blocks
    assert arch.architectures == ["Qwen3_5ForConditionalGeneration"]
    assert getattr(config, "head_dtype", None) is None


def test_plain_qwen3_5_is_untouched(monkeypatch):
    monkeypatch.delenv("VLLM_MODELING_OVERRIDES", raising=False)
    arch = convert(make_config(overrides=None))
    assert not arch.request_attention_blocks
    assert arch.architectures == ["Qwen3_5ForConditionalGeneration"]


def test_unsupported_overrides_fail_at_load(monkeypatch):
    monkeypatch.delenv("VLLM_MODELING_OVERRIDES", raising=False)
    with pytest.raises(ValueError, match="unsupported readout"):
        convert(make_config({**OVERRIDES, "readout": "head"}))


def test_baseten_key_is_read_and_conflicts_fail(monkeypatch):
    monkeypatch.delenv("VLLM_MODELING_OVERRIDES", raising=False)
    config = make_config(overrides=None)
    config.baseten_modeling_overrides = copy.deepcopy(OVERRIDES)
    assert convert(config).request_attention_blocks

    config = make_config()
    config.baseten_modeling_overrides = copy.deepcopy(OVERRIDES)
    assert convert(config).request_attention_blocks

    config = make_config()
    config.baseten_modeling_overrides = {
        **OVERRIDES,
        "attention": {"full_attention": "causal", "linear_attention": "causal"},
    }
    with pytest.raises(ValueError, match="disagree"):
        convert(config)
