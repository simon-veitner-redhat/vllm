# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

import pytest

from tests.utils import multi_gpu_test
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler

MODEL = "hmellor/tiny-random-LlamaForCausalLM"


def _sampling_state(worker):
    model_runner = worker.model_runner
    return (
        isinstance(model_runner.sampler, GPUWatermarkSampler),
        model_runner.batch_sharder is not None,
    )


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "parallel_kwargs",
    [
        {"tensor_parallel_size": 2, "enable_batch_sharded_sampling": True},
        {"pipeline_parallel_size": 2},
    ],
    ids=["tp2-sharded-sampling", "pp2"],
)
def test_watermarked_generation_with_model_parallelism(monkeypatch, parallel_kwargs):
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    llm = LLM(
        MODEL,
        dtype="half",
        enforce_eager=True,
        max_model_len=128,
        max_num_seqs=4,
        distributed_executor_backend="mp",
        watermark_config={"algorithm": "gumbel", "key": 42},
        **parallel_kwargs,
    )
    try:
        sampling_states = llm.llm_engine.collective_rpc(_sampling_state)
        if parallel_kwargs.get("enable_batch_sharded_sampling"):
            assert all(watermarked for watermarked, _ in sampling_states)
            assert all(sharded for _, sharded in sampling_states)
        else:
            assert sum(watermarked for watermarked, _ in sampling_states) == 1
            assert not any(sharded for _, sharded in sampling_states)

        first = llm.generate(
            ["Parallel watermark smoke A", "Parallel watermark smoke B"],
            SamplingParams(
                temperature=0.8,
                max_tokens=8,
                ignore_eos=True,
            ),
        )
        second = llm.generate(
            ["Parallel watermark smoke A", "Parallel watermark smoke B"],
            SamplingParams(
                temperature=0.8,
                max_tokens=8,
                ignore_eos=True,
            ),
        )

        assert len(first) == len(second) == 2
        assert [output.outputs[0].token_ids for output in first] == [
            output.outputs[0].token_ids for output in second
        ]
    finally:
        del llm
        cleanup_dist_env_and_memory()


@multi_gpu_test(num_gpus=2)
def test_watermarked_generation_with_data_parallelism():
    async def run() -> None:
        engine = AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=MODEL,
                dtype="half",
                data_parallel_size=2,
                data_parallel_backend="mp",
                enforce_eager=True,
                max_model_len=128,
                max_num_seqs=4,
                watermark_config={"algorithm": "gumbel", "key": 42},
            )
        )

        async def generate(index: int) -> tuple[int, ...]:
            final_output = None
            async for output in engine.generate(
                "Data-parallel watermark smoke",
                SamplingParams(
                    temperature=0.8,
                    max_tokens=8,
                    ignore_eos=True,
                ),
                request_id=f"watermark-dp-{index}",
                data_parallel_rank=index,
            ):
                final_output = output
            assert final_output is not None
            return tuple(final_output.outputs[0].token_ids)

        try:
            token_ids = await asyncio.gather(generate(0), generate(1))
            assert len(token_ids[0]) == 8
            assert token_ids[0] == token_ids[1]
        finally:
            engine.shutdown()

    try:
        asyncio.run(run())
    finally:
        cleanup_dist_env_and_memory()
