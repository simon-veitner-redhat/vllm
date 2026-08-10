# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for scheduler-realistic attention warmup."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from vllm.model_executor.warmup import fa4_cutedsl_warmup as fa4_warmup
from vllm.v1.worker.gpu import warmup as gpu_warmup
from vllm.v1.worker.gpu.warmup import run_mixed_prefill_decode_warmup


def _fail(*args, **kwargs):
    raise AssertionError("worker callback must not run when warmup is skipped")


@pytest.mark.parametrize(
    ("max_warmup_tokens", "num_queries_per_kv", "expected_query_lens"),
    [
        (1, (1,), ()),
        (2, (1,), (2,)),
        (3, (32,), (2, 3)),
        (100, (1,), (2, 50, 65)),
        (8192, (8,), (2, 9, 4096)),
    ],
)
def test_fa4_causal_warmup_query_lens_are_feasible_and_distinct(
    max_warmup_tokens, num_queries_per_kv, expected_query_lens
):
    assert (
        fa4_warmup._causal_warmup_query_lens(
            max_warmup_tokens, num_queries_per_kv
        )
        == expected_query_lens
    )


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


def test_mixed_warmup_multi_decode_lookahead_exact_capacity():
    connector = MagicMock()
    runner = SimpleNamespace(
        is_pooling_model=False,
        max_num_reqs=3,
        max_model_len=128,
        model_state=SimpleNamespace(max_encoder_len=0),
        vllm_config=SimpleNamespace(num_lookahead_tokens=1),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=4))
            ],
            num_blocks=6,
        ),
        kv_connector=connector,
    )
    outputs = []

    # Two 4-token decodes plus one lookahead token need two blocks each;
    # the 7-token prefill plus lookahead needs two more. Block zero is not
    # allocated, so exactly six available blocks must skip the six-block run.
    assert not run_mixed_prefill_decode_warmup(
        runner,
        worker_execute_model=_fail,
        worker_sample_tokens=_fail,
        num_tokens=9,
        decode_prompt_len=3,
        num_decode_reqs=2,
    )

    runner.kv_cache_config.num_blocks = 7
    assert run_mixed_prefill_decode_warmup(
        runner,
        worker_execute_model=outputs.append,
        worker_sample_tokens=lambda _: None,
        num_tokens=9,
        decode_prompt_len=3,
        num_decode_reqs=2,
    )

    mixed = outputs[2]
    assert len(mixed.scheduled_cached_reqs.req_ids) == 2
    assert sorted(mixed.num_scheduled_tokens.values()) == [1, 1, 7]
    allocated_ids = {
        block_id
        for output in outputs[:3]
        for new_req in output.scheduled_new_reqs
        for block_ids in new_req.block_ids
        for block_id in block_ids
    }
    allocated_ids.update(
        block_id
        for new_block_ids in mixed.scheduled_cached_reqs.new_block_ids
        if new_block_ids is not None
        for block_ids in new_block_ids
        for block_id in block_ids
    )
    assert allocated_ids == set(range(1, runner.kv_cache_config.num_blocks))
    assert len(outputs[3].finished_req_ids) == 3
    assert connector.set_disabled.call_args_list == [call(True), call(False)]


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
        model_config=SimpleNamespace(use_mla=False, max_model_len=8192),
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
        model_config=SimpleNamespace(use_mla=False, max_model_len=max_tokens),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_tokens),
    )
    runner = SimpleNamespace(
        is_pooling_model=False,
        vllm_config=config,
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        _dummy_run=MagicMock(),
    )
    return SimpleNamespace(model_runner=runner, use_v2_model_runner=False)


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
        model_config=SimpleNamespace(use_mla=False, max_model_len=8192),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
    )
    runner = SimpleNamespace(
        is_pooling_model=False,
        vllm_config=config,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
    )
    worker = SimpleNamespace(
        model_runner=runner,
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


@pytest.mark.parametrize("max_tokens", [2, 3])
def test_v2_fa4_dense_warmup_seeds_batch_one_when_mixed_is_infeasible(
    monkeypatch, max_tokens
):
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
        model_config=SimpleNamespace(use_mla=False, max_model_len=max_tokens),
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

    expected_query_lens = (2,) if max_tokens == 2 else (2, 3)
    assert runner._dummy_run.call_args_list == [
        call(query_len, skip_eplb=True, is_profile=True, num_reqs=1)
        for query_len in expected_query_lens
    ]
    assert mixed_warmup.call_count == (1 if max_tokens == 3 else 0)


@pytest.mark.parametrize(
    ("max_tokens", "max_num_seqs", "num_queries_per_kv", "expected_tokens"),
    [
        (3, 2, 32, [2, 3, 3]),
        (4, 1, 8, [2]),
    ],
)
def test_v1_fa4_dense_warmup_respects_capacity(
    monkeypatch, max_tokens, max_num_seqs, num_queries_per_kv, expected_tokens
):
    monkeypatch.setattr(
        fa4_warmup.current_platform,
        "is_device_capability",
        lambda _: True,
    )
    worker = _legacy_dense_worker(max_tokens, max_num_seqs, num_queries_per_kv)

    fa4_warmup.fa4_cutedsl_warmup(worker)

    assert [
        call_args.args[0] for call_args in worker.model_runner._dummy_run.call_args_list
    ] == expected_tokens


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
    fa4_warmup.fa4_cutedsl_warmup(worker)
    kernel.warmup.assert_called_once_with(config)
    runner._dummy_run.assert_not_called()

    kernel.reset_mock()
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
