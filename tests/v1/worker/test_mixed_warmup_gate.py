# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for scheduler-realistic attention warmup."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
import torch

from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.warmup import fa4_cutedsl_warmup as fa4_warmup
from vllm.v1.worker.gpu.warmup import run_mixed_prefill_decode_warmup


def _fail(*args, **kwargs):
    raise AssertionError("worker callback must not run when warmup is skipped")


@pytest.mark.parametrize("max_num_reqs", [1, 0])
def test_mixed_warmup_skipped_for_single_seq(max_num_reqs):
    """A mixed prefill+decode step needs >=2 requests; with max_num_reqs < 2
    the warmup must be skipped without touching the worker callbacks."""
    runner = SimpleNamespace(is_pooling_model=False, max_num_reqs=max_num_reqs)

    assert (
        run_mixed_prefill_decode_warmup(
            runner,
            worker_execute_model=_fail,
            worker_sample_tokens=_fail,
            num_tokens=128,
        )
        is False
    )


def test_mixed_warmup_builds_multiple_decodes():
    connector = MagicMock()
    runner = SimpleNamespace(
        is_pooling_model=False,
        max_num_reqs=3,
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))
            ],
            num_blocks=128,
        ),
        kv_connector=connector,
    )
    outputs = []

    assert run_mixed_prefill_decode_warmup(
        runner,
        worker_execute_model=outputs.append,
        worker_sample_tokens=lambda _: None,
        num_tokens=9,
        num_decode_reqs=2,
        decode_scheduled_tokens=2,
    )

    mixed = outputs[2]
    assert len(mixed.scheduled_cached_reqs.req_ids) == 2
    assert sorted(mixed.num_scheduled_tokens.values()) == [2, 2, 5]
    assert len(outputs[3].finished_req_ids) == 3
    assert connector.set_disabled.call_args_list == [call(True), call(False)]


def test_v1_fa4_mla_warmup_covers_mixed_and_batch_one(monkeypatch):
    is_sm90 = False
    monkeypatch.setattr(
        fa4_warmup.current_platform,
        "is_device_capability",
        lambda _: is_sm90,
    )
    kernel = MagicMock()
    kernel.get_warmup_keys.return_value = [object()]
    flash_attn = ModuleType("vllm.v1.attention.backends.mla.prefill.flash_attn")
    flash_attn.FA4_MLA_PREFILL_KERNEL = kernel
    monkeypatch.setitem(sys.modules, flash_attn.__name__, flash_attn)
    monkeypatch.setattr(
        fa4_warmup,
        "get_mla_prefill_backend",
        lambda _: SimpleNamespace(get_name=lambda: "FLASH_ATTN"),
    )

    config = SimpleNamespace(
        attention_config=SimpleNamespace(
            flash_attn_version=4, _hopper_fa4_fp8=False
        ),
        model_config=SimpleNamespace(use_mla=True, max_model_len=8192),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
    )
    runner = SimpleNamespace(
        is_pooling_model=False,
        vllm_config=config,
        _dummy_run=MagicMock(),
    )
    worker = SimpleNamespace(model_runner=runner, use_v2_model_runner=False)

    # Preserve the pre-project MLA prefill warmup on non-SM90 architectures.
    config.attention_config.flash_attn_version = 3
    fa4_warmup.fa4_cutedsl_warmup(worker)
    kernel.warmup.assert_called_once_with(config)
    runner._dummy_run.assert_not_called()

    kernel.reset_mock()
    is_sm90 = True
    config.attention_config.flash_attn_version = 4
    fa4_warmup.fa4_cutedsl_warmup(worker)

    kernel.warmup.assert_called_once_with(config)
    assert runner._dummy_run.call_args_list == [
        call(
            513,
            force_attention=True,
            is_profile=True,
            create_mixed_batch=True,
            skip_eplb=True,
            profile_seq_lens=4096,
        ),
        call(
            2,
            force_attention=True,
            is_profile=True,
            skip_eplb=True,
            profile_seq_lens=128,
            num_reqs=1,
        ),
        call(
            2,
            force_attention=True,
            is_profile=True,
            skip_eplb=True,
            profile_seq_lens=4096,
            num_reqs=1,
        ),
    ]


def test_v2_fa4_fp8_warmup_covers_persistent_tiles(monkeypatch):
    monkeypatch.setattr(
        fa4_warmup.current_platform, "is_device_capability", lambda _: True
    )
    mixed_warmup = MagicMock(return_value=True)
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.warmup.run_mixed_prefill_decode_warmup",
        mixed_warmup,
    )
    layer = object.__new__(Attention)
    layer._q_scale = torch.tensor(1.0, requires_grad=True)
    layer._k_scale = torch.tensor(0.25, requires_grad=True)
    layer._v_scale = torch.tensor(0.5, requires_grad=True)
    layer._q_scale_float = 1.0
    layer._k_scale_float = 0.25
    layer._v_scale_float = 0.5
    layer._k_scale_cpu = torch.tensor(1.0)
    layer._v_scale_cpu = torch.tensor(1.0)
    config = SimpleNamespace(
        attention_config=SimpleNamespace(
            flash_attn_version=4,
            _hopper_fa4_fp8=True,
        ),
        model_config=SimpleNamespace(use_mla=False, max_model_len=8192),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
        compilation_config=SimpleNamespace(
            static_forward_context={"layer": layer}
        ),
    )
    runner = SimpleNamespace(
        is_pooling_model=False,
        vllm_config=config,
        max_num_reqs=16,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        _dummy_run=MagicMock(),
    )
    worker = SimpleNamespace(
        model_runner=runner,
        use_v2_model_runner=True,
        execute_model=MagicMock(),
        sample_tokens=MagicMock(),
    )

    config.attention_config.flash_attn_version = 3
    layer._k_scale_float = 1.0
    fa4_warmup.normalize_hopper_fa4_fp8_scales(worker)
    assert layer._q_scale.item() == 1.0
    layer._k_scale_float = 0.25
    config.attention_config.flash_attn_version = 4

    for scale_attr in ("_k_scale_float", "_v_scale_float"):
        original_scale = getattr(layer, scale_attr)
        setattr(layer, scale_attr, 1.0)
        with pytest.raises(ValueError, match="positive non-unit"):
            fa4_warmup.normalize_hopper_fa4_fp8_scales(worker)
        setattr(layer, scale_attr, original_scale)

    fa4_warmup.fa4_cutedsl_warmup(worker)

    assert layer._q_scale.item() == layer._k_scale.item() == 0.25
    assert layer._v_scale.item() == 0.5
    assert layer._q_scale_float == 0.25
    assert layer._k_scale_cpu.item() == 0.25
    assert layer._v_scale_cpu.item() == 0.5
    assert mixed_warmup.call_args_list == [
        call(
            runner,
            worker.execute_model,
            worker.sample_tokens,
            18,
            decode_prompt_len=16,
            num_decode_reqs=1,
            decode_scheduled_tokens=2,
            req_id_prefix="_fa4_fp8_warmup_16",
        ),
        call(
            runner,
            worker.execute_model,
            worker.sample_tokens,
            19,
            decode_prompt_len=17,
            num_decode_reqs=1,
            decode_scheduled_tokens=2,
            req_id_prefix="_fa4_fp8_warmup_17",
        ),
        call(
            runner,
            worker.execute_model,
            worker.sample_tokens,
            19,
            decode_prompt_len=4096,
            num_decode_reqs=1,
            decode_scheduled_tokens=2,
            req_id_prefix="_fa4_fp8_warmup_long_8192",
        ),
        call(
            runner,
            worker.execute_model,
            worker.sample_tokens,
            25,
            decode_prompt_len=4096,
            num_decode_reqs=4,
            decode_scheduled_tokens=2,
            req_id_prefix="_fa4_warmup_8192",
        ),
    ]
