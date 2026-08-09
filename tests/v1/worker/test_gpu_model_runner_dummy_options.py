# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def test_explicit_randomization_restores_input_ids(monkeypatch):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1)
    )
    runner.model_config = SimpleNamespace(get_vocab_size=lambda: 32)
    runner.input_ids = SimpleNamespace(gpu=torch.zeros(4, dtype=torch.int64))
    input_ids = torch.zeros(3, dtype=torch.int64)

    monkeypatch.setattr(
        torch,
        "randint_like",
        lambda tensor, **kwargs: torch.full_like(tensor, 7),
    )

    with runner.maybe_randomize_inputs(
        input_ids, inputs_embeds=None, randomize_inputs=True
    ):
        assert input_ids.tolist() == [7, 7, 7]

    assert input_ids.tolist() == [0, 0, 0]


def test_new_dummy_run_options_are_keyword_only():
    parameters = inspect.signature(GPUModelRunner._dummy_run).parameters

    assert (
        parameters["profile_seq_lens"].kind
        is inspect.Parameter.POSITIONAL_OR_KEYWORD
    )
    assert parameters["randomize_inputs"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["num_reqs"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize(
    "incompatible_option", ["create_mixed_batch", "uniform_decode"]
)
def test_num_reqs_rejects_incompatible_batch_shapes(incompatible_option):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(multimodal_config=None)
    )
    runner.max_num_tokens = 8
    runner.scheduler_config = SimpleNamespace(max_num_seqs=8)
    runner.uniform_decode_query_len = 1

    with pytest.raises(AssertionError):
        runner._dummy_run(4, num_reqs=2, **{incompatible_option: True})
