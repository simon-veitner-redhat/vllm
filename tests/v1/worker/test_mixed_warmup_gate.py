# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for scheduler-realistic attention warmup."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
import torch

from vllm.model_executor.warmup import fa4_cutedsl_warmup as fa4_warmup
from vllm.v1.worker.gpu import warmup as gpu_warmup
from vllm.v1.worker.gpu.warmup import run_mixed_prefill_decode_warmup


def _fail(*args, **kwargs):
    raise AssertionError("worker callback must not run when warmup is skipped")


@pytest.mark.parametrize("max_num_reqs", [0, 1])
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




def _compile_worker(*impls):
    config = SimpleNamespace(
        attention_config=SimpleNamespace(
            flash_attn_version=4,
            flash_attn_max_num_splits_for_cuda_graph=32,
        ),
        cache_config=SimpleNamespace(block_size=16),
        compilation_config=SimpleNamespace(
            static_forward_context={
                f"attention_{index}": SimpleNamespace(impl=impl)
                for index, impl in enumerate(impls)
            }
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=64,
            max_num_batched_tokens=100,
        ),
        model_config=SimpleNamespace(
            use_mla=False,
            max_model_len=100,
            dtype=torch.bfloat16,
        ),
    )
    runner = SimpleNamespace(
        is_pooling_model=False,
        vllm_config=config,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        _dummy_run=MagicMock(side_effect=_fail),
    )
    return SimpleNamespace(
        model_runner=runner,
        vllm_config=config,
        use_v2_model_runner=True,
        execute_model=MagicMock(side_effect=_fail),
        sample_tokens=MagicMock(side_effect=_fail),
    )


def _assert_dynamic_scheduler_spec_pairing(compile_calls) -> None:
    for item in compile_calls:
        kwargs = item.kwargs
        has_counter = kwargs["dynamic_scheduler_counter_shape"] == (1,)
        assert kwargs["dynamic_scheduler_counter_stride"] == (
            (1,) if has_counter else None
        )


def _native_impl(
    *,
    num_heads=16,
    num_kv_heads=2,
    head_size=128,
    head_size_v=None,
    model_dtype=torch.bfloat16,
    scale=0.125,
    sliding_window=(-1, -1),
    sinks=None,
    diffkv=False,
):
    values = {
        "vllm_flash_attn_version": 4,
        "sm90_fa4_fp8_mode": "native",
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_size": head_size,
        "num_queries_per_kv": num_heads // num_kv_heads,
        "model_dtype": model_dtype,
        "scale": scale,
        "sliding_window": sliding_window,
        "sinks": sinks,
    }
    if not diffkv:
        return SimpleNamespace(**values)
    from vllm.v1.attention.backends.flash_attn_diffkv import (
        FlashAttentionDiffKVImpl,
    )

    impl = object.__new__(FlashAttentionDiffKVImpl)
    for name, value in values.items():
        setattr(impl, name, value)
    impl.head_size_v = head_size_v
    return impl


def _ordinary_impl(*, model_dtype=torch.bfloat16):
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    impl = object.__new__(FlashAttentionImpl)
    values = {
        "vllm_flash_attn_version": 4,
        "sm90_fa4_fp8_mode": None,
        "num_heads": 16,
        "num_kv_heads": 2,
        "head_size": 128,
        "num_queries_per_kv": 8,
        "model_dtype": model_dtype,
        "scale": 0.125,
        "sliding_window": (-1, -1),
        "sinks": None,
    }
    for name, value in values.items():
        setattr(impl, name, value)
    return impl


def test_dense_native_compile_specs_cover_structural_matrix(
    monkeypatch: pytest.MonkeyPatch,
):
    first = _native_impl()
    duplicate = _native_impl()
    second = _native_impl(
        num_heads=32,
        num_kv_heads=1,
        head_size=96,
        model_dtype=torch.float16,
        scale=0.25,
        sliding_window=(127, 0),
    )
    worker = _compile_worker(first, duplicate, second)
    compile_specs = MagicMock()
    monkeypatch.setattr(
        fa4_warmup.current_platform,
        "is_device_capability",
        lambda capability: capability == 90,
    )
    monkeypatch.setattr(
        "vllm.v1.attention.backends.utils.get_kv_cache_layout",
        lambda: "NHD",
    )
    monkeypatch.setattr(
        "vllm.vllm_flash_attn.flash_attn_interface."
        "compile_flash_attn_varlen_func_from_specs",
        compile_specs,
    )

    fa4_warmup._warm_fa4_compile_specs(worker)
    _assert_dynamic_scheduler_spec_pairing(compile_specs.call_args_list)

    static_calls = [
        item
        for item in compile_specs.call_args_list
        if item.kwargs["num_splits_dynamic_ptr_shape"] is None
        and item.kwargs["dynamic_scheduler_counter_shape"] is None
    ]
    unsplit_dynamic_calls = [
        item
        for item in compile_specs.call_args_list
        if item.kwargs["num_splits_dynamic_ptr_shape"] is None
        and item.kwargs["dynamic_scheduler_counter_shape"] == (1,)
    ]
    split_dynamic_calls = [
        item
        for item in compile_specs.call_args_list
        if item.kwargs["num_splits_dynamic_ptr_shape"] is not None
    ]
    assert len(compile_specs.call_args_list) == 504
    assert len(static_calls) == 420
    assert len(unsplit_dynamic_calls) == 40
    assert len(split_dynamic_calls) == 44
    assert len(
        {
            tuple(
                (
                    name,
                    tuple(value) if isinstance(value, list) else value,
                )
                for name, value in item.kwargs.items()
            )
            for item in compile_specs.call_args_list
        }
    ) == 504
    assert all(
        item.kwargs["q_dtype"] == torch.float8_e4m3fn
        and item.kwargs["q_descale_shape"] == ()
        and item.kwargs["q_descale_stride"] == ()
        and item.kwargs["k_descale_shape"] == ()
        and item.kwargs["k_descale_stride"] == ()
        and item.kwargs["v_descale_shape"] == ()
        and item.kwargs["v_descale_stride"] == ()
        for item in compile_specs.call_args_list
    )
    assert {
        item.kwargs["num_splits"] for item in static_calls
    } == set(fa4_warmup._FA4_STATIC_SPLIT_REQUESTS)
    assert {item.kwargs["causal"] for item in static_calls} == {True, False}
    assert all(
        item.kwargs["q_shape"][0] == item.kwargs["max_seqlen_q"]
        and item.kwargs["max_seqlen_q"] <= item.kwargs["max_seqlen_k"] <= 100
        and item.kwargs["cu_seqlens_q_shape"] == (2,)
        and item.kwargs["seqused_k_shape"] == (1,)
        and item.kwargs["page_table_shape"] == (1, 7)
        for item in static_calls
    )
    assert all(
        item.kwargs["num_splits"] == 1
        and item.kwargs["q_shape"][0]
        == item.kwargs["seqused_k_shape"][0]
        * item.kwargs["max_seqlen_q"]
        and item.kwargs["cu_seqlens_q_shape"]
        == (item.kwargs["seqused_k_shape"][0] + 1,)
        and item.kwargs["page_table_shape"]
        == (item.kwargs["seqused_k_shape"][0], 7)
        and item.kwargs["seqused_k_shape"][0] > 1
        and item.kwargs["dynamic_scheduler_counter_stride"] == (1,)
        for item in unsplit_dynamic_calls
    )
    assert {item.kwargs["causal"] for item in unsplit_dynamic_calls} == {
        True,
        False,
    }
    assert all(
        item.kwargs["num_splits"] == 32
        and item.kwargs["q_shape"][0]
        == item.kwargs["num_splits_dynamic_ptr_shape"][0]
        * item.kwargs["max_seqlen_q"]
        and item.kwargs["cu_seqlens_q_shape"]
        == (item.kwargs["num_splits_dynamic_ptr_shape"][0] + 1,)
        and item.kwargs["seqused_k_shape"]
        == item.kwargs["num_splits_dynamic_ptr_shape"]
        and item.kwargs["page_table_shape"]
        == (item.kwargs["num_splits_dynamic_ptr_shape"][0], 7)
        and item.kwargs["max_seqlen_q"] <= item.kwargs["max_seqlen_k"] <= 100
        and item.kwargs["q_shape"][0] <= 100
        and item.kwargs["dynamic_scheduler_counter_shape"] == (1,)
        for item in split_dynamic_calls
    )
    windowed_call = next(
        item.kwargs
        for item in split_dynamic_calls
        if item.kwargs["q_shape"][1:] == (32, 96)
        and item.kwargs["causal"] is False
        and item.kwargs["max_seqlen_k"] == 64
    )
    assert windowed_call["v_stride"] == (3072, 192, 192, 1)
    assert windowed_call["out_dtype"] == torch.float16
    assert windowed_call["softmax_scale"] == 0.25
    assert windowed_call["window_size"] == [127, 0]
    assert windowed_call["s_aux_shape"] is None
    worker.model_runner._dummy_run.assert_not_called()
    worker.execute_model.assert_not_called()
    worker.sample_tokens.assert_not_called()


@pytest.mark.parametrize("model_dtype", [torch.bfloat16, torch.float16])
def test_ordinary_compile_specs_cover_long_static_and_dynamic_classes(
    monkeypatch: pytest.MonkeyPatch,
    model_dtype: torch.dtype,
):
    worker = _compile_worker(
        _ordinary_impl(model_dtype=model_dtype),
        _ordinary_impl(model_dtype=model_dtype),
    )
    worker.vllm_config.model_config.max_model_len = 1800
    worker.vllm_config.scheduler_config.max_num_batched_tokens = 1800
    compile_specs = MagicMock()
    monkeypatch.setattr(
        fa4_warmup.current_platform,
        "is_device_capability",
        lambda capability: capability == 90,
    )
    monkeypatch.setattr(
        "vllm.v1.attention.backends.utils.get_kv_cache_layout",
        lambda: "NHD",
    )
    monkeypatch.setattr(
        "vllm.vllm_flash_attn.flash_attn_interface."
        "compile_flash_attn_varlen_func_from_specs",
        compile_specs,
    )

    fa4_warmup._warm_fa4_compile_specs(worker)

    static_calls = [
        item
        for item in compile_specs.call_args_list
        if item.kwargs["num_splits_dynamic_ptr_shape"] is None
        and item.kwargs["dynamic_scheduler_counter_shape"] is None
    ]
    unsplit_dynamic_calls = [
        item
        for item in compile_specs.call_args_list
        if item.kwargs["num_splits_dynamic_ptr_shape"] is None
        and item.kwargs["dynamic_scheduler_counter_shape"] == (1,)
    ]
    split_dynamic_calls = [
        item
        for item in compile_specs.call_args_list
        if item.kwargs["num_splits_dynamic_ptr_shape"] is not None
    ]
    assert len(compile_specs.call_args_list) == 430
    assert len(static_calls) == 372
    assert len(unsplit_dynamic_calls) == 28
    assert len(split_dynamic_calls) == 30
    assert len(
        {
            tuple(
                (
                    name,
                    tuple(value) if isinstance(value, list) else value,
                )
                for name, value in item.kwargs.items()
            )
            for item in compile_specs.call_args_list
        }
    ) == 430
    assert all(
        item.kwargs["q_dtype"] == model_dtype
        and item.kwargs["out_dtype"] is None
        and (item.kwargs["out_dtype"] or item.kwargs["q_dtype"]) == model_dtype
        and item.kwargs["q_descale_shape"] is None
        and item.kwargs["q_descale_stride"] is None
        and item.kwargs["k_descale_shape"] is None
        and item.kwargs["k_descale_stride"] is None
        and item.kwargs["v_descale_shape"] is None
        and item.kwargs["v_descale_stride"] is None
        for item in compile_specs.call_args_list
    )
    long_static = [
        item.kwargs
        for item in static_calls
        if item.kwargs["q_shape"] == (1800, 16, 128)
        and item.kwargs["max_seqlen_q"] == 1800
        and item.kwargs["max_seqlen_k"] == 1800
    ]
    assert len(long_static) == 12
    assert {
        (item["num_splits"], item["causal"]) for item in long_static
    } == {
        (num_splits, causal)
        for num_splits in fa4_warmup._FA4_STATIC_SPLIT_REQUESTS
        for causal in (True, False)
    }
    assert all(
        item["q_dtype"] == model_dtype
        and (item["out_dtype"] or item["q_dtype"]) == model_dtype
        for item in long_static
    )
    assert any(item["num_splits"] > 1 for item in long_static)
    assert all(
        item.kwargs["dynamic_scheduler_counter_shape"] == (1,)
        and item.kwargs["num_splits"] == 1
        for item in unsplit_dynamic_calls
    )
    assert all(
        item.kwargs["dynamic_scheduler_counter_shape"] == (1,)
        and item.kwargs["num_splits_dynamic_ptr_shape"]
        == item.kwargs["seqused_k_shape"]
        and item.kwargs["num_splits"] == 32
        for item in split_dynamic_calls
    )
    worker.model_runner._dummy_run.assert_not_called()
    worker.execute_model.assert_not_called()
    worker.sample_tokens.assert_not_called()


def test_native_compile_specs_emit_diagnosed_classes(
    monkeypatch: pytest.MonkeyPatch,
):

    worker = _compile_worker(_native_impl())
    worker.vllm_config.model_config.max_model_len = 512
    worker.vllm_config.scheduler_config.max_num_batched_tokens = 2048
    compile_specs = MagicMock()
    monkeypatch.setattr(
        fa4_warmup.current_platform,
        "is_device_capability",
        lambda capability: capability == 90,
    )
    monkeypatch.setattr(
        "vllm.v1.attention.backends.utils.get_kv_cache_layout",
        lambda: "NHD",
    )
    monkeypatch.setattr(
        "vllm.vllm_flash_attn.flash_attn_interface."
        "compile_flash_attn_varlen_func_from_specs",
        compile_specs,
    )

    fa4_warmup._warm_fa4_compile_specs(worker)

    assert len(compile_specs.call_args_list) == 394
    assert {
        (item.kwargs["num_splits"], item.kwargs["causal"])
        for item in compile_specs.call_args_list
        if item.kwargs["q_shape"] == (400, 16, 128)
        and item.kwargs["max_seqlen_q"] == 400
        and item.kwargs["max_seqlen_k"] == 400
        and item.kwargs["num_splits_dynamic_ptr_shape"] is None
    } == {
        (num_splits, causal)
        for num_splits in fa4_warmup._FA4_STATIC_SPLIT_REQUESTS
        for causal in (True, False)
    }
    assert {
        item.kwargs["causal"]
        for item in compile_specs.call_args_list
        if item.kwargs["q_shape"] == (2048, 16, 128)
        and item.kwargs["max_seqlen_q"] == 128
        and item.kwargs["max_seqlen_k"] == 129
        and item.kwargs["num_splits_dynamic_ptr_shape"] == (16,)
    } == {True, False}
    unsplit_b16 = [
        item.kwargs
        for item in compile_specs.call_args_list
        if item.kwargs["q_shape"] == (2048, 16, 128)
        and item.kwargs["max_seqlen_q"] == 128
        and item.kwargs["max_seqlen_k"] == 129
        and item.kwargs["num_splits"] == 1
        and item.kwargs["num_splits_dynamic_ptr_shape"] is None
    ]
    assert len(unsplit_b16) == 2
    assert {item["causal"] for item in unsplit_b16} == {True, False}
    assert all(
        item["dynamic_scheduler_counter_shape"] == (1,)
        and item["dynamic_scheduler_counter_stride"] == (1,)
        for item in unsplit_b16
    )
    assert {
        item.kwargs["causal"]
        for item in compile_specs.call_args_list
        if item.kwargs["q_shape"] == (64, 16, 128)
        and item.kwargs["max_seqlen_q"] == 1
        and item.kwargs["max_seqlen_k"] == 512
        and item.kwargs["num_splits_dynamic_ptr_shape"] == (64,)
    } == {True, False}
    worker.model_runner._dummy_run.assert_not_called()
    worker.execute_model.assert_not_called()
    worker.sample_tokens.assert_not_called()


@pytest.mark.parametrize(
    ("cache_layout", "expected_stride"),
    [
        ("NHD", (10240, 640, 320, 1)),
        ("HND", (10240, 320, 5120, 1)),
    ],
)
def test_native_diffkv_compile_specs_cover_packed_sink_dynamic_and_dedupe(
    monkeypatch: pytest.MonkeyPatch,
    cache_layout: str,
    expected_stride: tuple[int, ...],
):
    first = _native_impl(
        head_size=192,
        head_size_v=128,
        sliding_window=(127, 0),
        sinks=object(),
        diffkv=True,
    )
    duplicate = _native_impl(
        head_size=192,
        head_size_v=128,
        sliding_window=(127, 0),
        sinks=object(),
        diffkv=True,
    )
    worker = _compile_worker(first, duplicate)
    from vllm.v1.kv_cache_interface import MambaSpec

    worker.model_runner.kv_cache_config.kv_cache_groups = [
        SimpleNamespace(kv_cache_spec=object.__new__(MambaSpec))
    ]
    compile_specs = MagicMock()
    monkeypatch.setattr(
        fa4_warmup.current_platform,
        "is_device_capability",
        lambda capability: capability == 90,
    )
    monkeypatch.setattr(
        "vllm.v1.attention.backends.utils.get_kv_cache_layout",
        lambda: cache_layout,
    )
    monkeypatch.setattr(
        "vllm.vllm_flash_attn.flash_attn_interface."
        "compile_flash_attn_varlen_func_from_specs",
        compile_specs,
    )

    fa4_warmup._warm_fa4_compile_specs(worker)

    # A coexisting Mamba cache group must not suppress DiffKV compilation.
    static_calls = [
        item
        for item in compile_specs.call_args_list
        if item.kwargs["num_splits_dynamic_ptr_shape"] is None
        and item.kwargs["dynamic_scheduler_counter_shape"] is None
    ]
    unsplit_dynamic_calls = [
        item
        for item in compile_specs.call_args_list
        if item.kwargs["num_splits_dynamic_ptr_shape"] is None
        and item.kwargs["dynamic_scheduler_counter_shape"] == (1,)
    ]
    split_dynamic_calls = [
        item
        for item in compile_specs.call_args_list
        if item.kwargs["num_splits_dynamic_ptr_shape"] is not None
    ]
    assert len(compile_specs.call_args_list) == 246
    assert len(static_calls) == 204
    assert len(unsplit_dynamic_calls) == 20
    assert len(split_dynamic_calls) == 22
    assert len(
        {
            tuple(
                (
                    name,
                    tuple(value) if isinstance(value, list) else value,
                )
                for name, value in item.kwargs.items()
            )
            for item in compile_specs.call_args_list
        }
    ) == 246
    dynamic = next(
        item.kwargs
        for item in split_dynamic_calls
        if item.kwargs["num_splits_dynamic_ptr_shape"] == (4,)
        and item.kwargs["max_seqlen_q"] == 16
        and item.kwargs["max_seqlen_k"] == 64
        and item.kwargs["causal"]
    )
    unsplit_dynamic = next(
        item.kwargs
        for item in unsplit_dynamic_calls
        if item.kwargs["seqused_k_shape"] == (4,)
        and item.kwargs["max_seqlen_q"] == 16
        and item.kwargs["max_seqlen_k"] == 64
        and item.kwargs["causal"]
    )
    assert unsplit_dynamic["q_shape"] == (64, 16, 192)
    assert unsplit_dynamic["k_shape"] == (7, 16, 2, 192)
    assert unsplit_dynamic["v_shape"] == (7, 16, 2, 128)
    assert unsplit_dynamic["v_stride"] == expected_stride
    assert unsplit_dynamic["window_size"] == [127, 0]
    assert unsplit_dynamic["s_aux_shape"] == (16,)
    assert unsplit_dynamic["num_splits"] == 1
    assert unsplit_dynamic["num_splits_dynamic_ptr_shape"] is None
    assert unsplit_dynamic["cu_seqlens_q_shape"] == (5,)
    assert unsplit_dynamic["page_table_shape"] == (4, 7)
    assert unsplit_dynamic["dynamic_scheduler_counter_shape"] == (1,)
    assert unsplit_dynamic["dynamic_scheduler_counter_stride"] == (1,)
    assert dynamic["q_shape"] == (64, 16, 192)
    assert dynamic["k_shape"] == (7, 16, 2, 192)
    assert dynamic["v_shape"] == (7, 16, 2, 128)
    assert dynamic["v_stride"] == expected_stride
    assert dynamic["window_size"] == [127, 0]
    assert dynamic["s_aux_shape"] == (16,)
    assert dynamic["s_aux_stride"] == (1,)
    assert dynamic["num_splits_dynamic_ptr_stride"] == (1,)
    assert dynamic["cu_seqlens_q_shape"] == (5,)
    assert dynamic["seqused_k_shape"] == (4,)
    assert dynamic["page_table_shape"] == (4, 7)
    assert dynamic["page_table_stride"] == (7, 1)
    assert dynamic["dynamic_scheduler_counter_shape"] == (1,)
    assert dynamic["dynamic_scheduler_counter_stride"] == (1,)
    assert dynamic["num_splits"] == 32
    worker.model_runner._dummy_run.assert_not_called()
    worker.execute_model.assert_not_called()
    worker.sample_tokens.assert_not_called()


@pytest.mark.parametrize(
    "route",
    [
        "pooling",
        "non_sm90",
        "mla",
        "fa3_config",
        "local_fa2",
        "ordinary_unsupported_dtype",
        "ordinary_diffkv",
        "cross_attention",
        "encoder",
    ],
)
def test_fa4_compile_specs_skip_ineligible_routes(
    monkeypatch: pytest.MonkeyPatch,
    route: str,
):
    from vllm.v1.kv_cache_interface import CrossAttentionSpec

    impl = _native_impl()
    worker = _compile_worker(impl)
    is_sm90 = route != "non_sm90"
    if route == "pooling":
        worker.model_runner.is_pooling_model = True
    elif route == "mla":
        worker.vllm_config.model_config.use_mla = True
    elif route == "fa3_config":
        worker.vllm_config.attention_config.flash_attn_version = 3
    elif route == "local_fa2":
        impl.vllm_flash_attn_version = 2
    elif route == "ordinary_unsupported_dtype":
        impl = _ordinary_impl(model_dtype=torch.float32)
        worker = _compile_worker(impl)
    elif route == "ordinary_diffkv":
        impl = _native_impl(diffkv=True)
        impl.sm90_fa4_fp8_mode = None
        worker = _compile_worker(impl)
    elif route == "cross_attention":
        worker.model_runner.kv_cache_config.kv_cache_groups = [
            SimpleNamespace(kv_cache_spec=object.__new__(CrossAttentionSpec))
        ]
    elif route == "encoder":
        impl.attn_type = SimpleNamespace(name="ENCODER")

    compile_specs = MagicMock()
    monkeypatch.setattr(
        fa4_warmup.current_platform,
        "is_device_capability",
        lambda _: is_sm90,
    )
    monkeypatch.setattr(
        "vllm.vllm_flash_attn.flash_attn_interface."
        "compile_flash_attn_varlen_func_from_specs",
        compile_specs,
    )

    fa4_warmup._warm_fa4_compile_specs(worker)

    compile_specs.assert_not_called()
    worker.model_runner._dummy_run.assert_not_called()
    worker.execute_model.assert_not_called()
    worker.sample_tokens.assert_not_called()


def test_legacy_native_compile_specs_use_real_static_context(
    monkeypatch: pytest.MonkeyPatch,
):
    full = _native_impl(
        num_heads=16,
        num_kv_heads=1,
        head_size=192,
        sliding_window=(-1, -1),
    )
    swa_layers = [
        _native_impl(
            num_heads=16,
            num_kv_heads=1,
            head_size=192,
            sliding_window=(127, 0),
        )
        for _ in range(4)
    ]
    worker = _compile_worker(full, *swa_layers)
    worker.use_v2_model_runner = False
    worker.vllm_config.model_config.max_model_len = 512
    worker.vllm_config.scheduler_config.max_num_batched_tokens = 2048
    compile_specs = MagicMock()
    monkeypatch.setattr(
        fa4_warmup.current_platform,
        "is_device_capability",
        lambda capability: capability == 90,
    )
    monkeypatch.setattr(
        "vllm.v1.attention.backends.utils.get_kv_cache_layout",
        lambda: "NHD",
    )
    monkeypatch.setattr(
        "vllm.vllm_flash_attn.flash_attn_interface."
        "compile_flash_attn_varlen_func_from_specs",
        compile_specs,
    )

    fa4_warmup._warm_fa4_compile_specs(worker)

    assert len(compile_specs.call_args_list) == 788
    assert {
        tuple(item.kwargs["window_size"])
        for item in compile_specs.call_args_list
    } == {(-1, -1), (127, 0)}
    q400_split_two = [
        item.kwargs
        for item in compile_specs.call_args_list
        if item.kwargs["q_shape"] == (400, 16, 192)
        and item.kwargs["max_seqlen_q"] == 400
        and item.kwargs["max_seqlen_k"] == 400
        and item.kwargs["num_splits"] == 2
        and item.kwargs["causal"]
        and item.kwargs["num_splits_dynamic_ptr_shape"] is None
    ]
    assert len(q400_split_two) == 2
    assert {tuple(item["window_size"]) for item in q400_split_two} == {
        (-1, -1),
        (127, 0),
    }
    assert all(
        item["dynamic_scheduler_counter_shape"] is None
        for item in q400_split_two
    )
    worker.model_runner._dummy_run.assert_not_called()
    worker.execute_model.assert_not_called()
    worker.sample_tokens.assert_not_called()


def test_v2_fa4_dense_warmup_covers_causal_query_lengths(monkeypatch):
    monkeypatch.setattr(
        fa4_warmup.current_platform,
        "is_device_capability",
        lambda _: True,
    )
    config = SimpleNamespace(
        attention_config=SimpleNamespace(flash_attn_version=4),
        compilation_config=SimpleNamespace(
            static_forward_context={
                "attention": SimpleNamespace(
                    impl=SimpleNamespace(
                        vllm_flash_attn_version=4, num_queries_per_kv=8
                    )
                )
            }
        ),
        model_config=SimpleNamespace(
            use_mla=False,
            max_model_len=8192,
            hf_config=SimpleNamespace(model_type="test"),
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
    )
    runner = SimpleNamespace(
        is_pooling_model=False,
        vllm_config=config,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        _dummy_run=MagicMock(),
    )
    worker = SimpleNamespace(
        model_runner=runner,
        vllm_config=config,
        use_v2_model_runner=True,
        execute_model=MagicMock(),
        sample_tokens=MagicMock(),
    )
    mixed_warmup = MagicMock(return_value=True)
    monkeypatch.setattr(
        gpu_warmup,
        "run_mixed_prefill_decode_warmup",
        mixed_warmup,
    )

    fa4_warmup.fa4_cutedsl_warmup(worker)

    assert mixed_warmup.call_args_list == [
        call(
            runner,
            worker.execute_model,
            worker.sample_tokens,
            num_tokens=3,
            decode_prompt_len=2,
            decode_scheduled_tokens=1,
            req_id_prefix="_fa4_warmup_8192_2",
        ),
        call(
            runner,
            worker.execute_model,
            worker.sample_tokens,
            num_tokens=3,
            decode_prompt_len=9,
            decode_scheduled_tokens=1,
            req_id_prefix="_fa4_warmup_8192_9",
        ),
        call(
            runner,
            worker.execute_model,
            worker.sample_tokens,
            num_tokens=3,
            decode_prompt_len=4096,
            decode_scheduled_tokens=1,
            req_id_prefix="_fa4_warmup_8192_4096",
        ),
    ]


def _legacy_dense_worker(
    max_tokens: int, max_num_seqs: int, num_queries_per_kv: int = 8
):
    config = SimpleNamespace(
        attention_config=SimpleNamespace(flash_attn_version=4),
        compilation_config=SimpleNamespace(
            static_forward_context={
                "attention": SimpleNamespace(
                    impl=SimpleNamespace(
                        vllm_flash_attn_version=4,
                        num_queries_per_kv=num_queries_per_kv,
                    )
                )
            }
        ),
        model_config=SimpleNamespace(
            use_mla=False,
            max_model_len=max_tokens,
            hf_config=SimpleNamespace(model_type="test"),
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_tokens),
    )
    runner = SimpleNamespace(
        is_pooling_model=False,
        vllm_config=config,
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        _dummy_run=MagicMock(),
    )
    return SimpleNamespace(
        model_runner=runner,
        vllm_config=config,
        use_v2_model_runner=False,
    )


def test_v1_fa4_dense_warmup_covers_causal_query_lengths(monkeypatch):
    monkeypatch.setattr(
        fa4_warmup.current_platform,
        "is_device_capability",
        lambda _: True,
    )
    worker = _legacy_dense_worker(8192, 2)

    fa4_warmup.fa4_cutedsl_warmup(worker)

    calls = worker.model_runner._dummy_run.call_args_list
    assert [call.args[0] for call in calls] == [2, 3, 9, 3, 4096, 3]
    assert [call.kwargs["profile_seq_lens"] for call in calls] == [
        2,
        [3, 2],
        9,
        [10, 2],
        4096,
        [4097, 2],
    ]
    assert ["create_mixed_batch" in call.kwargs for call in calls] == [
        False,
        True,
        False,
        True,
        False,
        True,
    ]
    for call_args in calls:
        assert call_args.kwargs["force_attention"]
        assert call_args.kwargs["is_profile"]
        assert call_args.kwargs["skip_eplb"]


def test_v2_fa4_dense_warmup_skips_local_fa2_attention(monkeypatch):
    monkeypatch.setattr(
        fa4_warmup.current_platform,
        "is_device_capability",
        lambda _: True,
    )
    config = SimpleNamespace(
        attention_config=SimpleNamespace(flash_attn_version=4),
        compilation_config=SimpleNamespace(
            static_forward_context={
                "attention": SimpleNamespace(
                    impl=SimpleNamespace(vllm_flash_attn_version=2)
                )
            }
        ),
        model_config=SimpleNamespace(
            use_mla=False,
            max_model_len=8192,
            hf_config=SimpleNamespace(model_type="test"),
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
    )
    runner = SimpleNamespace(
        is_pooling_model=False,
        vllm_config=config,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
    )
    worker = SimpleNamespace(
        model_runner=runner,
        vllm_config=config,
        use_v2_model_runner=True,
        execute_model=MagicMock(),
        sample_tokens=MagicMock(),
    )
    mixed_warmup = MagicMock()
    monkeypatch.setattr(
        gpu_warmup,
        "run_mixed_prefill_decode_warmup",
        mixed_warmup,
    )

    fa4_warmup.fa4_cutedsl_warmup(worker)

    mixed_warmup.assert_not_called()


def test_v2_fa4_dense_warmup_seeds_batch_one_when_mixed_is_infeasible(
    monkeypatch,
):
    max_tokens = 3
    monkeypatch.setattr(
        fa4_warmup.current_platform,
        "is_device_capability",
        lambda _: True,
    )
    config = SimpleNamespace(
        attention_config=SimpleNamespace(flash_attn_version=4),
        compilation_config=SimpleNamespace(
            static_forward_context={
                "attention": SimpleNamespace(
                    impl=SimpleNamespace(
                        vllm_flash_attn_version=4, num_queries_per_kv=32
                    )
                )
            }
        ),
        model_config=SimpleNamespace(
            use_mla=False,
            max_model_len=max_tokens,
            hf_config=SimpleNamespace(model_type="test"),
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_tokens),
    )
    runner = SimpleNamespace(
        is_pooling_model=False,
        vllm_config=config,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        _dummy_run=MagicMock(),
    )
    worker = SimpleNamespace(
        model_runner=runner,
        vllm_config=config,
        use_v2_model_runner=True,
        execute_model=MagicMock(),
        sample_tokens=MagicMock(),
    )
    mixed_warmup = MagicMock(return_value=False)
    monkeypatch.setattr(
        gpu_warmup,
        "run_mixed_prefill_decode_warmup",
        mixed_warmup,
    )

    fa4_warmup.fa4_cutedsl_warmup(worker)

    assert runner._dummy_run.call_args_list == [
        call(query_len, skip_eplb=True, is_profile=True, num_reqs=1)
        for query_len in (2, 3)
    ]
    mixed_warmup.assert_called_once()


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
        attention_config=SimpleNamespace(flash_attn_version=4),
        model_config=SimpleNamespace(
            use_mla=True,
            max_model_len=8192,
            hf_config=SimpleNamespace(model_type="test"),
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
    )
    runner = SimpleNamespace(
        is_pooling_model=False,
        vllm_config=config,
        _dummy_run=MagicMock(),
    )
    worker = SimpleNamespace(
        model_runner=runner,
        vllm_config=config,
        use_v2_model_runner=False,
    )

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




def test_fa4_mla_prefill_sm90_warmup_key_contract(monkeypatch):
    import vllm.envs as envs
    from vllm.model_executor.layers.attention import mla_attention
    from vllm.v1.attention.backends.mla.prefill import flash_attn

    mla_dims = SimpleNamespace(
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
    )
    monkeypatch.setattr(mla_attention, "get_mla_dims", lambda _: mla_dims)
    monkeypatch.setattr(
        flash_attn.current_platform,
        "is_device_capability",
        lambda capability: capability == 90,
    )
    monkeypatch.setattr(
        flash_attn.current_platform,
        "is_device_capability_family",
        lambda _: False,
    )
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)

    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        get_num_attention_heads=lambda _: 16,
    )
    config = SimpleNamespace(
        attention_config=SimpleNamespace(flash_attn_version=4),
        model_config=model_config,
        parallel_config=SimpleNamespace(),
    )

    keys = flash_attn.FA4_MLA_PREFILL_KERNEL.get_warmup_keys(config)

    intended_shape_pairs = {
        (1, 128),
        (129, 512),
        (1, 4096),
        (129, 4096),
    }
    actual_axes = {
        (
            key.cu_seqlens_q_shape[0] - 1,
            (key.max_seqlen_q, key.max_seqlen_k),
            key.causal,
            key.return_softmax_lse,
        )
        for key in keys
    }
    expected_axes = {
        (batch_size, shape_pair, causal, return_lse)
        for batch_size in (1, 2)
        for shape_pair in intended_shape_pairs
        for causal in (False, True)
        for return_lse in (False, True)
    }
    assert len(keys) == 32
    assert actual_axes == expected_axes

    config.attention_config.flash_attn_version = 3
    assert flash_attn.FA4_MLA_PREFILL_KERNEL.get_warmup_keys(config) == []
