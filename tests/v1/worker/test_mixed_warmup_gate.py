# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for scheduler-realistic attention warmup."""

import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from vllm.config import get_current_vllm_config_or_none, set_current_vllm_config
from vllm.model_executor.warmup import fa4_cutedsl_warmup as fa4_warmup
from vllm.v1.worker import gpu_worker
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


@pytest.mark.parametrize("hopper_fa4_fp8", [True, False])
def test_fa4_fp8_warmup_config_context(monkeypatch, hopper_fa4_fp8):
    config = SimpleNamespace(
        attention_config=SimpleNamespace(_hopper_fa4_fp8=hopper_fa4_fp8),
    )
    outer_config = SimpleNamespace(
        attention_config=SimpleNamespace(_hopper_fa4_fp8=False)
    )
    worker = object.__new__(gpu_worker.Worker)
    worker.vllm_config = config
    observed = []

    def observe_fa4_dispatch(_):
        from vllm.vllm_flash_attn.flash_attn_interface import (
            _hopper_fa4_fp8_enabled,
        )

        observed.append(
            (get_current_vllm_config_or_none(), _hopper_fa4_fp8_enabled())
        )

    monkeypatch.setattr(
        gpu_worker.Worker,
        "_compile_or_warm_up_model",
        observe_fa4_dispatch,
    )

    with set_current_vllm_config(outer_config):
        gpu_worker.Worker.compile_or_warm_up_model(worker)
        assert get_current_vllm_config_or_none() is outer_config

    expected = config if hopper_fa4_fp8 else outer_config
    assert observed == [(expected, hopper_fa4_fp8)]


@pytest.mark.parametrize("hopper_fa4_fp8", [True, False])
def test_fa4_fp8_memory_profile_config_context(monkeypatch, hopper_fa4_fp8):
    config = SimpleNamespace(
        attention_config=SimpleNamespace(_hopper_fa4_fp8=hopper_fa4_fp8),
    )
    outer_config = SimpleNamespace(
        attention_config=SimpleNamespace(_hopper_fa4_fp8=False)
    )
    worker = object.__new__(gpu_worker.Worker)
    worker.vllm_config = config
    observed = []

    def observe(label):
        def inner(_):
            observed.append((label, get_current_vllm_config_or_none()))
            return 1

        return inner

    monkeypatch.setattr(
        gpu_worker.Worker,
        "_determine_available_memory",
        observe("worker"),
    )
    with set_current_vllm_config(outer_config):
        assert gpu_worker.Worker.determine_available_memory(worker) == 1
        assert get_current_vllm_config_or_none() is outer_config

    expected = config if hopper_fa4_fp8 else outer_config
    assert observed == [("worker", expected)]


@pytest.mark.parametrize("hopper_fa4_fp8", [True, False])
def test_fa4_fp8_memory_profile_compile_monitor_reset(monkeypatch, hopper_fa4_fp8):
    from vllm.vllm_flash_attn import flash_attn_interface

    sentinel = {"forward": frozenset({object()}), "combine": frozenset()}
    flash_attn_interface._fa4_compile_cache_baseline = sentinel
    config = SimpleNamespace(
        attention_config=SimpleNamespace(_hopper_fa4_fp8=hopper_fa4_fp8),
    )
    worker = object.__new__(gpu_worker.Worker)
    worker.vllm_config = config
    observed = []
    monkeypatch.setattr(
        gpu_worker.Worker,
        "_determine_available_memory",
        lambda _: observed.append(
            flash_attn_interface._fa4_compile_cache_baseline
        )
        or 1,
    )

    assert gpu_worker.Worker.determine_available_memory(worker) == 1

    assert observed == [None if hopper_fa4_fp8 else sentinel]
    flash_attn_interface.disarm_flash_attn_compile_monitor()


@pytest.mark.parametrize("hopper_fa4_fp8", [True, False])
def test_fa4_fp8_execute_model_config_context(monkeypatch, hopper_fa4_fp8):
    config = SimpleNamespace(
        attention_config=SimpleNamespace(_hopper_fa4_fp8=hopper_fa4_fp8),
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=False)
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
    )
    outer_config = SimpleNamespace(
        attention_config=SimpleNamespace(_hopper_fa4_fp8=False)
    )
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=1)
    observed = []

    def observe_fa4_dispatch(*args):
        from vllm.vllm_flash_attn.flash_attn_interface import (
            _hopper_fa4_fp8_enabled,
        )

        observed.append(
            (args, get_current_vllm_config_or_none(), _hopper_fa4_fp8_enabled())
        )

    worker = object.__new__(gpu_worker.Worker)
    worker.vllm_config = config
    worker._pp_send_work = []
    worker.use_v2_model_runner = True
    worker.model_runner = SimpleNamespace(
        execute_model=observe_fa4_dispatch,
        is_pooling_model=False,
    )
    worker.annotate_profile = lambda _: nullcontext()
    monkeypatch.setattr(
        gpu_worker,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True),
    )

    with set_current_vllm_config(outer_config):
        assert gpu_worker.Worker.execute_model(worker, scheduler_output) is None
        assert get_current_vllm_config_or_none() is outer_config

    expected = config if hopper_fa4_fp8 else outer_config
    assert observed == [
        ((scheduler_output, None), expected, hopper_fa4_fp8)
    ]


def test_v1_fa4_mla_warmup_covers_mixed_and_batch_one(monkeypatch):
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
        attention_config=SimpleNamespace(flash_attn_version=4),
        model_config=SimpleNamespace(use_mla=True, max_model_len=8192),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
    )
    runner = SimpleNamespace(
        is_pooling_model=False,
        vllm_config=config,
        _dummy_run=MagicMock(),
    )
    worker = SimpleNamespace(model_runner=runner, use_v2_model_runner=False)

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
