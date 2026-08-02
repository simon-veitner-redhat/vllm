# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the max_num_reqs gate on the V2 mixed prefill+decode warmup."""

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

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
