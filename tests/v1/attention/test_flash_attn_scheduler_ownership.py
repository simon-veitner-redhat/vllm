# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import vllm


class _CloneablePrefillBackend:
    def clone(self):
        return self


def test_fa4_hopper_split_policy_is_owned_by_flash_attention():
    package_root = Path(vllm.__file__).parent
    sources = [
        package_root / "v1/attention/backends/flash_attn.py",
        package_root / "v1/attention/backends/mla/flashattn_mla.py",
    ]
    combined = "\n".join(path.read_text() for path in sources)

    assert "SplitSchedulerPlanner" in combined
    for policy_fragment in (
        "HopperSplit",
        "plan_hopper_split_schedule",
        "_fa4_hopper_dynamic_splits",
        "_fa4_hopper_mla_dynamic_splits",
        "blocks_per_sm",
        "is_device_capability_family(90",
        "num_compute_units(",
    ):
        assert policy_fragment not in combined


def test_batch1_graph_replay_updates_fa_owned_split_schedule(monkeypatch):
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability()[0] != 9
    ):
        pytest.skip("Hopper-only CUDA graph lifecycle test")

    import vllm.vllm_flash_attn
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends import flash_attn
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    monkeypatch.setattr(flash_attn, "get_flash_attn_version", lambda: 4)

    num_heads, num_kv_heads, head_dim = 16, 8, 256
    block_size, max_seqlen_k = 128, 4096
    model_config = SimpleNamespace(
        get_num_attention_heads=lambda _parallel_config: num_heads,
        get_num_kv_heads=lambda _parallel_config: num_kv_heads,
        get_head_size=lambda: head_dim,
        rswa_window=None,
    )
    parallel_config = SimpleNamespace(cp_kv_cache_interleave_size=1)
    compilation_config = SimpleNamespace(
        cudagraph_mode=SimpleNamespace(
            has_full_cudagraphs=lambda: True
        ),
        max_cudagraph_capture_size=256,
    )
    config = SimpleNamespace(
        model_config=model_config,
        parallel_config=parallel_config,
        cache_config=SimpleNamespace(cache_dtype="auto"),
        compilation_config=compilation_config,
        attention_config=SimpleNamespace(
            flash_attn_max_num_splits_for_cuda_graph=32
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=256),
    )
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_dim,
        dtype=torch.bfloat16,
    )
    builder = flash_attn.FlashAttentionMetadataBuilder(
        spec, ["test"], config, torch.device("cuda")
    )

    page_table = torch.arange(
        max_seqlen_k // block_size,
        dtype=torch.int32,
        device="cuda",
    ).unsqueeze(0)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device="cuda")

    def common_metadata(seq_len):
        return CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc.cpu(),
            seq_lens=torch.tensor(
                [seq_len], dtype=torch.int32, device="cuda"
            ),
            seq_lens_cpu_upper_bound=torch.tensor(
                [seq_len], dtype=torch.int32
            ),
            num_reqs=1,
            num_actual_tokens=1,
            max_query_len=1,
            max_seq_len=seq_len,
            block_table_tensor=page_table,
            slot_mapping=torch.tensor(
                [-1], dtype=torch.int64, device="cuda"
            ),
            causal=True,
        )

    capture_metadata = builder.build_for_cudagraph_capture(
        common_metadata(1)
    )
    assert capture_metadata.max_num_splits == 15
    assert capture_metadata.scheduler_metadata is not None
    assert capture_metadata.scheduler_metadata.item() == -1
    split_ptr = capture_metadata.scheduler_metadata.data_ptr()

    torch.manual_seed(0)
    q = torch.randn(
        1, num_heads, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    k = torch.randn(
        page_table.shape[1],
        block_size,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v = torch.randn_like(k)
    seqused_k = torch.ones(1, dtype=torch.int32, device="cuda")

    def run(num_splits, dynamic_splits):
        return _flash_attn_fwd(
            q,
            k,
            v,
            cu_seqlens_q=query_start_loc,
            seqused_k=seqused_k,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen_k,
            page_table=page_table,
            num_splits=num_splits,
            num_splits_dynamic_ptr=dynamic_splits,
        )[0]

    run(
        capture_metadata.max_num_splits,
        capture_metadata.scheduler_metadata,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = run(
            capture_metadata.max_num_splits,
            capture_metadata.scheduler_metadata,
        )

    replay_metadata = builder.build(0, common_metadata(max_seqlen_k))
    assert replay_metadata.scheduler_metadata is not None
    assert replay_metadata.scheduler_metadata.data_ptr() == split_ptr
    assert replay_metadata.scheduler_metadata.item() == 13
    seqused_k.fill_(max_seqlen_k)
    graph.replay()
    torch.cuda.synchronize()

    expected = run(13, None)
    torch.testing.assert_close(
        graph_out.float(), expected.float(), atol=3e-2, rtol=3e-2
    )


_HETEROGENEOUS_128_LENGTHS = (
    (512,) * 32 + (4096,) * 32 + (8192,) * 32 + (16384,) * 32
)
_HETEROGENEOUS_128_SPLITS = (
    (1,) * 64 + (2,) * 32 + (3,) * 32
)


@pytest.mark.parametrize(
    (
        "batch_size",
        "num_heads",
        "head_dim_v",
        "seq_lens_spec",
        "max_num_splits",
        "split_count",
    ),
    (
        (16, 64, 512, 512, 8, 4),
        (1, 16, 512, 8192, 32, 32),
        (32, 128, 512, 32768, 4, 2),
        (64, 32, 512, 512, 4, 2),
        (128, 16, 512, 1024, 4, 1),
        (
            128,
            16,
            512,
            _HETEROGENEOUS_128_LENGTHS,
            4,
            _HETEROGENEOUS_128_SPLITS,
        ),
        (1, 16, 256, 4096, 32, 32),
    ),
    ids=(
        "dv512-h64-b16-s512",
        "dv512-h16-b1-s8192",
        "dv512-h128-b32-s32768",
        "dv512-h32-b64-s512",
        "dv512-h16-b128-s1024",
        "dv512-h16-b128-heterogeneous",
        "dv256-h16-b1-s4096",
    ),
)
def test_mla_graph_capture_at_one_replays_fa_owned_schedule(
    monkeypatch,
    batch_size,
    num_heads,
    head_dim_v,
    seq_lens_spec,
    max_num_splits,
    split_count,
):
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability()[0] != 9
    ):
        pytest.skip("Hopper-only MLA CUDA graph lifecycle test")

    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonMetadataBuilder,
    )
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.mla import flashattn_mla
    from vllm.v1.kv_cache_interface import MLAAttentionSpec
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd

    monkeypatch.setattr(
        flashattn_mla, "_get_mla_fa_version", lambda _config=None: 4
    )
    monkeypatch.setattr(
        MLACommonMetadataBuilder,
        "determine_prefill_query_data_type",
        staticmethod(lambda _config, dtype: dtype),
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16
    head_dim = 64
    block_size = 128
    target_seq_lens_cpu = torch.tensor(
        (
            [seq_lens_spec] * batch_size
            if isinstance(seq_lens_spec, int)
            else seq_lens_spec
        ),
        dtype=torch.int32,
    )
    seq_len = target_seq_lens_cpu.max().item()
    max_batch_size = max(batch_size, 16)
    model_config = SimpleNamespace(
        dtype=dtype,
        max_model_len=seq_len,
        hf_text_config=SimpleNamespace(
            kv_lora_rank=head_dim_v,
            qk_nope_head_dim=128,
            qk_rope_head_dim=head_dim,
            v_head_dim=128,
        ),
        get_num_attention_heads=lambda _parallel_config: num_heads,
        get_head_size=lambda: head_dim + head_dim_v,
    )
    parallel_config = SimpleNamespace(
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        cp_kv_cache_interleave_size=1,
    )
    compilation_config = SimpleNamespace(
        cudagraph_mode=SimpleNamespace(
            has_full_cudagraphs=lambda: True
        ),
        max_cudagraph_capture_size=max_batch_size,
        static_forward_context={
            "test": SimpleNamespace(
                prefill_backend=_CloneablePrefillBackend()
            )
        },
    )
    config = SimpleNamespace(
        model_config=model_config,
        parallel_config=parallel_config,
        cache_config=SimpleNamespace(
            block_size=block_size, cache_dtype="auto"
        ),
        compilation_config=compilation_config,
        attention_config=SimpleNamespace(
            flash_attn_max_num_splits_for_cuda_graph=32,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_batch_size),
        speculative_config=None,
        additional_config={"requested_decode_fa_version": 4},
    )
    spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_dim + head_dim_v,
        dtype=dtype,
    )
    builder = flashattn_mla.FlashAttnMLAMetadataBuilder(
        spec, ["test"], config, device
    )

    blocks_per_seq = (seq_len + block_size - 1) // block_size
    query_start_loc = torch.arange(
        batch_size + 1, dtype=torch.int32, device=device
    )
    seq_lens = torch.ones(batch_size, dtype=torch.int32, device=device)
    page_table = torch.arange(
        batch_size * blocks_per_seq, dtype=torch.int32, device=device
    ).view(batch_size, blocks_per_seq)

    def common(seq_lens_cpu):
        return CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc.cpu(),
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu,
            num_reqs=batch_size,
            num_actual_tokens=batch_size,
            max_query_len=1,
            max_seq_len=seq_len,
            block_table_tensor=page_table,
            slot_mapping=torch.full(
                (batch_size,), -1, dtype=torch.int64, device=device
            ),
            causal=True,
        )

    capture = builder.build_for_cudagraph_capture(
        common(torch.ones(batch_size, dtype=torch.int32))
    )
    assert capture.decode is not None
    capture_decode = capture.decode
    assert capture_decode.max_num_splits == max_num_splits
    assert capture_decode.scheduler_metadata is not None
    assert capture_decode.scheduler_metadata.tolist() == [
        -1,
        *([1] * (batch_size - 1)),
    ]
    scheduler_ptr = capture_decode.scheduler_metadata.data_ptr()

    torch.manual_seed(0)
    q = torch.randn(
        batch_size, num_heads, head_dim, dtype=dtype, device=device
    )
    qv = torch.randn(
        batch_size, num_heads, head_dim_v, dtype=dtype, device=device
    )
    k = torch.randn(
        batch_size * blocks_per_seq,
        block_size,
        1,
        head_dim,
        dtype=dtype,
        device=device,
    )
    v = torch.randn(
        batch_size * blocks_per_seq,
        block_size,
        1,
        head_dim_v,
        dtype=dtype,
        device=device,
    )

    def run(dynamic_splits, num_splits):
        return _flash_attn_fwd(
            q,
            k,
            v,
            qv=qv,
            cu_seqlens_q=query_start_loc,
            seqused_k=seq_lens,
            max_seqlen_q=1,
            max_seqlen_k=seq_len,
            min_seqlen_k=1,
            page_table=page_table,
            num_splits=num_splits,
            num_splits_dynamic_ptr=dynamic_splits,
        )[0]

    run(capture_decode.scheduler_metadata, capture_decode.max_num_splits)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = run(
            capture_decode.scheduler_metadata,
            capture_decode.max_num_splits,
        )

    seq_lens.copy_(target_seq_lens_cpu.to(device))
    replay = builder.build(
        0, common(target_seq_lens_cpu)
    )
    assert replay.decode is not None
    replay_decode = replay.decode
    assert replay_decode.max_num_splits == capture_decode.max_num_splits
    assert replay_decode.scheduler_metadata is not None
    assert replay_decode.scheduler_metadata.data_ptr() == scheduler_ptr
    expected_split_counts = (
        [split_count] * batch_size
        if isinstance(split_count, int)
        else list(split_count)
    )
    if max(expected_split_counts) == 1:
        expected_split_counts[0] = -1
    assert replay_decode.scheduler_metadata.tolist() == expected_split_counts

    graph.replay()
    torch.cuda.synchronize()
    expected = run(None, 1)
    torch.testing.assert_close(
        graph_out.float(), expected.float(), atol=4e-2, rtol=4e-2
    )
