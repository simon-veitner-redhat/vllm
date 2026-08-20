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


def test_mixed_warmup_skips_exact_multi_decode_capacity():
    runner = SimpleNamespace(
        is_pooling_model=False,
        max_num_reqs=3,
        max_model_len=128,
        model_state=SimpleNamespace(max_encoder_len=0),
        vllm_config=SimpleNamespace(num_lookahead_tokens=1),
        kv_cache_config=SimpleNamespace(
            num_blocks=6,
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=4))
            ],
        ),
    )

    assert not run_mixed_prefill_decode_warmup(
        runner,
        worker_execute_model=_fail,
        worker_sample_tokens=_fail,
        num_tokens=9,
        decode_prompt_len=3,
        num_decode_reqs=2,
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


def _capture_compile_specs(
    monkeypatch: pytest.MonkeyPatch,
    worker,
    *,
    cache_layout: str = "NHD",
) -> list[dict]:
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
    _assert_worker_callbacks_not_called(worker)
    return [item.kwargs for item in compile_specs.call_args_list]


def _assert_worker_callbacks_not_called(worker) -> None:
    worker.model_runner._dummy_run.assert_not_called()
    worker.execute_model.assert_not_called()
    worker.sample_tokens.assert_not_called()


def _compile_classes(specs: list[dict]) -> tuple[list[dict], ...]:
    static = [
        spec
        for spec in specs
        if spec["num_splits_dynamic_ptr_shape"] is None
        and spec["dynamic_scheduler_counter_shape"] is None
    ]
    unsplit_dynamic = [
        spec
        for spec in specs
        if spec["num_splits_dynamic_ptr_shape"] is None
        and spec["dynamic_scheduler_counter_shape"] == (1,)
    ]
    split_dynamic = [
        spec
        for spec in specs
        if spec["num_splits_dynamic_ptr_shape"] is not None
    ]
    assert static and unsplit_dynamic and split_dynamic
    assert len(static) + len(unsplit_dynamic) + len(split_dynamic) == len(specs)
    assert all(
        (
            spec["dynamic_scheduler_counter_shape"],
            spec["dynamic_scheduler_counter_stride"],
        )
        in {(None, None), ((1,), (1,))}
        for spec in specs
    )
    assert all(
        (
            spec["dynamic_scheduler_counter_shape"],
            spec["dynamic_scheduler_counter_stride"],
        )
        == ((1,), (1,))
        for spec in split_dynamic
    )
    assert {
        spec["num_splits"] for spec in static
    } == set(fa4_warmup._FA4_STATIC_SPLIT_REQUESTS)
    assert all(
        spec["q_shape"][0] == spec["max_seqlen_q"]
        and spec["max_seqlen_q"] <= spec["max_seqlen_k"]
        and spec["cu_seqlens_q_shape"] == (2,)
        and spec["seqused_k_shape"] == (1,)
        for spec in static
    )
    assert all(
        spec["num_splits"] == 1
        and spec["q_shape"][0]
        == spec["seqused_k_shape"][0] * spec["max_seqlen_q"]
        and spec["cu_seqlens_q_shape"]
        == (spec["seqused_k_shape"][0] + 1,)
        and spec["seqused_k_shape"][0] > 1
        for spec in unsplit_dynamic
    )
    assert all(
        spec["num_splits"] == 32
        and spec["q_shape"][0]
        == spec["num_splits_dynamic_ptr_shape"][0] * spec["max_seqlen_q"]
        and spec["cu_seqlens_q_shape"]
        == (spec["num_splits_dynamic_ptr_shape"][0] + 1,)
        and spec["seqused_k_shape"] == spec["num_splits_dynamic_ptr_shape"]
        for spec in split_dynamic
    )
    assert all(
        {spec["causal"] for spec in compile_class} == {True, False}
        for compile_class in (static, unsplit_dynamic, split_dynamic)
    )

    def freeze(value):
        if isinstance(value, list):
            return tuple(value)
        return value

    assert len(
        {tuple((name, freeze(value)) for name, value in spec.items()) for spec in specs}
    ) == len(specs)
    return static, unsplit_dynamic, split_dynamic


def _matching(specs: list[dict], **expected) -> list[dict]:
    return [
        spec
        for spec in specs
        if all(spec[name] == value for name, value in expected.items())
    ]

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
    full = _native_impl(
        num_kv_heads=1,
        head_size=192,
    )
    duplicate = _native_impl(
        num_kv_heads=1,
        head_size=192,
    )
    swa = _native_impl(
        num_kv_heads=1,
        head_size=192,
        sliding_window=(127, 0),
    )
    fp16 = _native_impl(
        num_heads=32,
        num_kv_heads=1,
        head_size=96,
        model_dtype=torch.float16,
        scale=0.25,
        sliding_window=(127, 0),
    )
    worker = _compile_worker(full, duplicate, swa, fp16)
    worker.use_v2_model_runner = False
    worker.vllm_config.model_config.max_model_len = 512
    worker.vllm_config.scheduler_config.max_num_batched_tokens = 2048

    specs = _capture_compile_specs(monkeypatch, worker)
    static, unsplit_dynamic, split_dynamic = _compile_classes(specs)
    assert all(
        spec["q_dtype"] == torch.float8_e4m3fn
        and spec["q_descale_shape"] == spec["q_descale_stride"] == ()
        and spec["k_descale_shape"] == spec["k_descale_stride"] == ()
        and spec["v_descale_shape"] == spec["v_descale_stride"] == ()
        for spec in specs
    )
    assert {spec["out_dtype"] for spec in specs} == {
        torch.bfloat16,
        torch.float16,
    }

    q400 = _matching(
        static,
        q_shape=(400, 16, 192),
        max_seqlen_q=400,
        max_seqlen_k=400,
    )
    assert {
        (spec["num_splits"], spec["causal"], tuple(spec["window_size"]))
        for spec in q400
    } == {
        (num_splits, causal, window)
        for num_splits in fa4_warmup._FA4_STATIC_SPLIT_REQUESTS
        for causal in (True, False)
        for window in ((-1, -1), (127, 0))
    }
    assert all(spec["dynamic_scheduler_counter_shape"] is None for spec in q400)

    for compile_class, expected in (
        (
            unsplit_dynamic,
            {
                "q_shape": (2048, 16, 192),
                "max_seqlen_q": 128,
                "max_seqlen_k": 129,
                "num_splits": 1,
                "num_splits_dynamic_ptr_shape": None,
            },
        ),
        (
            split_dynamic,
            {
                "q_shape": (2048, 16, 192),
                "max_seqlen_q": 128,
                "max_seqlen_k": 129,
                "num_splits_dynamic_ptr_shape": (16,),
            },
        ),
        (
            split_dynamic,
            {
                "q_shape": (64, 16, 192),
                "max_seqlen_q": 1,
                "max_seqlen_k": 512,
                "num_splits_dynamic_ptr_shape": (64,),
            },
        ),
    ):
        assert {
            spec["causal"] for spec in _matching(compile_class, **expected)
        } == {True, False}

    windowed = next(
        spec
        for spec in split_dynamic
        if spec["q_shape"][1:] == (32, 96)
        and spec["causal"] is False
        and spec["max_seqlen_k"] == 64
    )
    assert windowed["v_stride"] == (3072, 192, 192, 1)
    assert windowed["out_dtype"] == torch.float16
    assert windowed["softmax_scale"] == 0.25
    assert windowed["window_size"] == [127, 0]
    assert windowed["s_aux_shape"] is None


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

    specs = _capture_compile_specs(monkeypatch, worker)
    static, _, _ = _compile_classes(specs)
    assert all(
        spec["q_dtype"] == model_dtype
        and spec["out_dtype"] is None
        and spec["q_descale_shape"] is None
        and spec["q_descale_stride"] is None
        and spec["k_descale_shape"] is None
        and spec["k_descale_stride"] is None
        and spec["v_descale_shape"] is None
        and spec["v_descale_stride"] is None
        for spec in specs
    )
    long_static = _matching(
        static,
        q_shape=(1800, 16, 128),
        max_seqlen_q=1800,
        max_seqlen_k=1800,
    )
    assert {
        (spec["num_splits"], spec["causal"]) for spec in long_static
    } == {
        (num_splits, causal)
        for num_splits in fa4_warmup._FA4_STATIC_SPLIT_REQUESTS
        for causal in (True, False)
    }


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
    impl_kwargs = {
        "head_size": 192,
        "head_size_v": 128,
        "sliding_window": (127, 0),
        "sinks": object(),
        "diffkv": True,
    }
    worker = _compile_worker(_native_impl(**impl_kwargs), _native_impl(**impl_kwargs))
    from vllm.v1.kv_cache_interface import MambaSpec

    worker.model_runner.kv_cache_config.kv_cache_groups = [
        SimpleNamespace(kv_cache_spec=object.__new__(MambaSpec))
    ]
    specs = _capture_compile_specs(
        monkeypatch,
        worker,
        cache_layout=cache_layout,
    )
    _, unsplit_dynamic, split_dynamic = _compile_classes(specs)
    assert all(
        spec["q_dtype"] == torch.float8_e4m3fn
        and spec["out_dtype"] == torch.bfloat16
        and spec["q_descale_shape"] == spec["q_descale_stride"] == ()
        and spec["k_descale_shape"] == spec["k_descale_stride"] == ()
        and spec["v_descale_shape"] == spec["v_descale_stride"] == ()
        for spec in specs
    )

    common = {
        "q_shape": (64, 16, 192),
        "k_shape": (7, 16, 2, 192),
        "v_shape": (7, 16, 2, 128),
        "v_stride": expected_stride,
        "window_size": [127, 0],
        "s_aux_shape": (16,),
        "cu_seqlens_q_shape": (5,),
        "seqused_k_shape": (4,),
        "page_table_shape": (4, 7),
        "max_seqlen_q": 16,
        "max_seqlen_k": 64,
        "causal": True,
    }
    unsplit = _matching(
        unsplit_dynamic,
        **common,
        num_splits=1,
        num_splits_dynamic_ptr_shape=None,
    )
    split = _matching(
        split_dynamic,
        **common,
        num_splits=32,
        num_splits_dynamic_ptr_shape=(4,),
    )
    assert len(unsplit) == len(split) == 1
    assert split[0]["s_aux_stride"] == (1,)
    assert split[0]["num_splits_dynamic_ptr_stride"] == (1,)
    assert split[0]["page_table_stride"] == (7, 1)


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
