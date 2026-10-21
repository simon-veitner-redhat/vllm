# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepGEMM MegaMoE with EP over two SM90 nodes against the dequantized
reference of test_deep_gemm_mega_moe.py, with unequal token counts on rail peers.

The pytest test runs two nodes of 4 GPUs on one host. Two nodes of 8 GPUs, from
the repository root:

    torchrun --nnodes 2 --node-rank <n> --nproc-per-node 8 \\
        --master-addr <node 0> -m tests.kernels.moe.test_deep_gemm_mega_moe_ep
"""

import os

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from tests.kernels.moe.test_deep_gemm_mega_moe import (
    _calc_diff,
    _expert,
    _quantize,
    _round_trip,
)
from tests.utils import multi_gpu_test
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import parallel_state
from vllm.distributed.parallel_state import (
    get_ep_group,
    get_ep_node_group,
    get_ep_rail_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import DPMetadata, ForwardContext, override_forward_context
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    MEGA_MOE_MXFP4,
    DeepGemmMegaMoEExperts,
    MegaMoEActivation,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

NUM_EXPERTS, TOP_K, HIDDEN, INTER, MAX_TOKENS = 64, 6, 512, 256, 256


def _make_experts(vllm_config, num_ranks: int, rank: int) -> tuple:
    """Experts of this rank, loaded from one checkpoint generated on every rank,
    and the dequantized weights of all experts."""
    num_local = NUM_EXPERTS // num_ranks
    experts = DeepGemmMegaMoEExperts(
        vllm_config,
        num_experts=NUM_EXPERTS,
        num_local_experts=num_local,
        experts_start_idx=rank * num_local,
        top_k=TOP_K,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        activation=MegaMoEActivation("swiglu", clamp=10.0),
        weight_format=MEGA_MOE_MXFP4,
        prefix="model.layers.0.experts",
    )
    torch.manual_seed(0)
    w13 = torch.empty(NUM_EXPERTS, 2 * INTER, HIDDEN)
    w2 = torch.empty(NUM_EXPERTS, HIDDEN, INTER)
    for e in range(NUM_EXPERTS):
        w13_ckpt = _quantize(
            (torch.randn(2 * INTER, HIDDEN) / HIDDEN**0.5).bfloat16(),
            experts.weight_format.routed,
        )
        w2_ckpt = _quantize(
            (torch.randn(HIDDEN, INTER) / INTER**0.5).bfloat16(),
            experts.weight_format.routed,
        )
        w13[e], w2[e] = w13_ckpt[3], w2_ckpt[3]
        for shard_id, param, rows, (weight, scale, _, _) in (
            ("w1", "w13", slice(0, INTER), w13_ckpt),
            ("w3", "w13", slice(INTER, None), w13_ckpt),
            ("w2", "w2", slice(None), w2_ckpt),
        ):
            for suffix, tensor in (("weight", weight), ("weight_scale", scale)):
                name = f"{param}_{suffix}"
                experts.weight_loader(
                    getattr(experts, name),
                    tensor[rows],
                    f"experts.{name}",
                    shard_id=shard_id,
                    expert_id=e,
                )
    return experts, w13, w2


def _run_step(experts, w13, w2, num_tokens: list[int], num_padding: int):
    """One launch on every rank with ``num_tokens[rank]`` tokens, the last
    ``num_padding`` of them padding; returns this rank's output and reference."""
    rank = get_ep_group().rank_in_group
    generator = torch.Generator(device="cuda").manual_seed(sum(num_tokens))
    inputs = []
    for n in num_tokens:  # every rank's inputs, so each rank checks its own
        hidden_states = torch.randn(
            n, HIDDEN, device="cuda", generator=generator
        ).bfloat16()
        scores = torch.randn(n, NUM_EXPERTS, device="cuda", generator=generator)
        inputs.append((hidden_states, *torch.topk(scores.softmax(-1), TOP_K)))
    hidden_states, topk_weights, topk_ids = inputs[rank]
    is_padding = torch.zeros(num_tokens[rank], dtype=torch.bool, device="cuda")
    is_padding[num_tokens[rank] - num_padding :] = True

    forward_context = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        dp_metadata=DPMetadata(torch.tensor(num_tokens, dtype=torch.int32)),
        is_padding=is_padding,
    )
    with override_forward_context(forward_context):
        output = experts(hidden_states, topk_weights, topk_ids)

    x = _round_trip(hidden_states.float(), experts.weight_format.hidden_quant)
    expected = torch.zeros_like(x)
    for e in range(NUM_EXPERTS):
        token, slot = ((topk_ids == e) & ~is_padding[:, None]).nonzero(as_tuple=True)
        expected.index_add_(
            0,
            token,
            _expert(
                x[token],
                w13[e],
                w2[e],
                topk_weights[token, slot],
                experts.weight_format,
                experts.activation,
                experts.l2_activation_scale,
            ),
        )
    return output, expected


def main() -> None:
    num_ranks, rank = int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    vllm_config = VllmConfig()
    vllm_config.parallel_config.enable_expert_parallel = True
    vllm_config.kernel_config.moe_backend = "deep_gemm_mega_moe"
    vllm_config.scheduler_config.max_num_batched_tokens = MAX_TOKENS

    with set_current_vllm_config(vllm_config), torch.device("cuda"):
        init_distributed_environment(num_ranks, rank, "env://", local_rank, "nccl")
        vllm_config.parallel_config.data_parallel_size = num_ranks
        initialize_model_parallel(1, 1)
        rail, node = get_ep_rail_group(), get_ep_node_group()
        assert rail is not None and node is not None
        assert rail.world_size == 2 and node.world_size == num_ranks // 2
        assert rail.ranks == [
            rank % node.world_size + i * node.world_size for i in (0, 1)
        ]

        experts, w13, w2 = _make_experts(vllm_config, num_ranks, rank)
        assert experts.rail_group is rail

        half = num_ranks // 2
        steps = [
            ([MAX_TOKENS] * num_ranks, 0),
            ([1 + 37 * r % MAX_TOKENS for r in range(num_ranks)], 0),
            # Unequal counts on every rail, one empty rank, and padding.
            ([0 if r == 1 else 5 + 13 * r for r in range(num_ranks)], 3),
            ([7] * half + [MAX_TOKENS] * half, 2),
        ]
        for num_tokens, num_padding in steps:
            num_padding = min(num_padding, num_tokens[rank])
            output, expected = _run_step(experts, w13, w2, num_tokens, num_padding)
            again, _ = _run_step(experts, w13, w2, num_tokens, num_padding)
            assert output.shape == (num_tokens[rank], HIDDEN)
            assert torch.equal(output, again), "MegaMoE reruns differ"
            num_real = num_tokens[rank] - num_padding
            if num_real:
                diff = _calc_diff(output, expected)
                cos = F.cosine_similarity(output.float(), expected, dim=-1)
                assert diff < 1e-3, (num_tokens, rank, diff)
                assert cos[:num_real].min() > 0.99, (num_tokens, rank)
            assert not output[num_real:].any(), "padding was routed"


def _two_node_worker(local_rank: int, num_ranks: int, port: int) -> None:
    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(num_ranks),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(port),
        }
    )
    parallel_state.get_node_count = lambda: 2
    main()


@multi_gpu_test(num_gpus=8)
@pytest.mark.skipif(
    not (current_platform.is_device_capability(90) and is_deep_gemm_supported()),
    reason="DeepGEMM MegaMoE over nodes requires SM90 and DeepGEMM",
)
def test_deep_gemm_mega_moe_ep_on_two_nodes(monkeypatch):
    """Two nodes of 4 GPUs, with NCCL kept off NVLink and shared memory."""
    monkeypatch.setenv("NCCL_P2P_DISABLE", "1")
    monkeypatch.setenv("NCCL_SHM_DISABLE", "1")
    mp.spawn(_two_node_worker, args=(8, get_open_port()), nprocs=8)


if __name__ == "__main__":
    main()
