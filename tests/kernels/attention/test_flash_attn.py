# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

try:
    from vllm.vllm_flash_attn import (
        fa_version_unsupported_reason,
        flash_attn_interface,
        flash_attn_varlen_func,
        is_fa_version_supported,
    )
except ImportError:
    if current_platform.is_rocm():
        pytest.skip(
            "vllm_flash_attn is not supported for vLLM on ROCm.",
            allow_module_level=True,
        )


NUM_HEADS = [(4, 4), (8, 2)]
HEAD_SIZES = [40, 72, 80, 128, 256]
BLOCK_SIZES = [16]
DTYPES = [torch.bfloat16]
QDTYPES = [None, torch.float8_e4m3fn]
# one value large enough to test overflow in index calculation.
# one value small enough to test the schema op check
NUM_BLOCKS = [32768, 2048]
SOFT_CAPS = [None]
SLIDING_WINDOWS = [None, 256]


@pytest.fixture
def fa4_dispatch(monkeypatch: pytest.MonkeyPatch):
    flash_attn_interface._is_sm90_device.cache_clear()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (9, 0))
    forwarded: dict = {}
    calls: list[dict] = []

    def fake_flash_attn_fwd(*args, **kwargs):
        forwarded.update(kwargs)
        calls.append(kwargs)
        result = kwargs.get("out")
        return (args[0] if result is None else result), None, None, None

    fake_interface = ModuleType("vllm.vllm_flash_attn.cute.interface")
    fake_interface._flash_attn_fwd = fake_flash_attn_fwd  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules, "vllm.vllm_flash_attn.cute.interface", fake_interface
    )
    try:
        yield forwarded, calls
    finally:
        flash_attn_interface._is_sm90_device.cache_clear()


def _native_fa4_inputs(
    *,
    device: str = "cpu",
    num_heads: tuple[int, int] = (8, 2),
    out_dtype: torch.dtype = torch.bfloat16,
    descale_layout: str = "scalar",
) -> dict:
    num_query_heads, num_kv_heads = num_heads
    tensor = lambda *args, **kwargs: torch.empty(*args, device=device, **kwargs)
    if descale_layout == "scalar":
        scale_shape = ()
    else:
        scale_shape = (2, num_kv_heads)
    return {
        "q": tensor((2, num_query_heads, 192), dtype=torch.float8_e4m3fn),
        "k": tensor((4, 16, num_kv_heads, 192), dtype=torch.float8_e4m3fn),
        "v": tensor((4, 16, num_kv_heads, 128), dtype=torch.float8_e4m3fn),
        "out": tensor((2, num_query_heads, 128), dtype=out_dtype),
        "out_dtype": out_dtype,
        "cu_seqlens_q": torch.tensor([0, 1, 2], dtype=torch.int32, device=device),
        "seqused_k": torch.tensor([31, 23], dtype=torch.int32, device=device),
        "max_seqlen_q": 1,
        "max_seqlen_k": 31,
        "block_table": torch.tensor(
            [[0, 1], [2, 3]], dtype=torch.int32, device=device
        ),
        "num_splits": 2,
        "fa_version": 4,
        "q_descale": torch.full(scale_shape, 0.25, dtype=torch.float32, device=device),
        "k_descale": torch.full(scale_shape, 0.40, dtype=torch.float32, device=device),
        "v_descale": torch.full(scale_shape, 0.30, dtype=torch.float32, device=device),
    }


def test_fa4_dcp_forwards_native_cp_args(fa4_dispatch) -> None:
    forwarded, _ = fa4_dispatch
    q = torch.empty((1, 1, 64))
    cp_tot_seqused_k = torch.tensor([2], dtype=torch.int32)
    dynamic_scheduler_counter = torch.zeros((1,), dtype=torch.int32)
    result = flash_attn_varlen_func(
        q=q,
        k=torch.empty_like(q),
        v=torch.empty_like(q),
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        seqused_k=torch.tensor([1], dtype=torch.int32),
        max_seqlen_q=1,
        max_seqlen_k=1,
        fa_version=4,
        cp_world_size=2,
        cp_rank=1,
        cp_tot_seqused_k=cp_tot_seqused_k,
        dynamic_scheduler_counter=dynamic_scheduler_counter,
    )

    assert result is q
    assert forwarded["cp_world_size"] == 2
    assert forwarded["cp_rank"] == 1
    assert forwarded["cp_tot_seqused_k"] is cp_tot_seqused_k
    assert forwarded["dynamic_scheduler_counter"] is dynamic_scheduler_counter


@pytest.mark.parametrize("descale_layout", ["scalar", "rank2"])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_heads", [(4, 4), (8, 2), (8, 1)])
def test_fa4_native_fp8_forwards_diffkv_dtype_descales_and_static_splits(
    fa4_dispatch,
    num_heads: tuple[int, int],
    out_dtype: torch.dtype,
    descale_layout: str,
) -> None:
    forwarded, _ = fa4_dispatch
    inputs = _native_fa4_inputs(
        num_heads=num_heads, out_dtype=out_dtype, descale_layout=descale_layout
    )
    flash_attn_varlen_func(**inputs)

    assert forwarded["out"] is inputs["out"]
    assert forwarded["out_dtype"] is out_dtype
    for name in ("q_descale", "k_descale", "v_descale"):
        assert forwarded[name] is inputs[name]
    assert forwarded["num_splits"] == 2
    assert forwarded["num_splits_dynamic_ptr"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fa4_native_fp8_forwards_dynamic_split_counts(fa4_dispatch) -> None:
    forwarded, _ = fa4_dispatch
    inputs = _native_fa4_inputs(device="cuda")
    scheduler_metadata = torch.tensor([2, 3], dtype=torch.int32, device="cuda")
    inputs.update(scheduler_metadata=scheduler_metadata, num_splits=3)
    flash_attn_varlen_func(**inputs)

    assert forwarded["num_splits"] == 3
    assert forwarded["num_splits_dynamic_ptr"] is scheduler_metadata


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fa4_rejects_invalid_dynamic_split_shape_before_dispatch(
    fa4_dispatch,
) -> None:
    _, calls = fa4_dispatch
    inputs = _native_fa4_inputs(device="cuda")
    inputs.update(
        scheduler_metadata=torch.empty((2, 1), dtype=torch.int32, device="cuda"),
        num_splits=3,
    )
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        flash_attn_varlen_func(**inputs)
    assert not calls


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fa4_native_fp8_diffkv_reaches_real_flash_attn_fwd() -> None:
    if not is_fa_version_supported(4):
        pytest.skip(fa_version_unsupported_reason(4))

    q = torch.randn((2, 8, 192), device="cuda").clamp(-2, 2).to(torch.float8_e4m3fn)
    k = torch.randn((16, 2, 192), device="cuda").clamp(-2, 2).to(torch.float8_e4m3fn)
    v = torch.randn((16, 2, 128), device="cuda").clamp(-2, 2).to(torch.float8_e4m3fn)
    scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")

    out = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        out_dtype=torch.float16,
        cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda"),
        cu_seqlens_k=torch.tensor([0, 8, 16], dtype=torch.int32, device="cuda"),
        max_seqlen_q=1,
        max_seqlen_k=8,
        fa_version=4,
        q_descale=scale,
        k_descale=scale,
        v_descale=scale,
        num_splits=1,
    )

    assert out.dtype == torch.float16
    assert out.shape == (2, 8, 128)
    assert torch.isfinite(out).all()


def test_fa4_native_fp8_rejects_output_dtype_mismatch_before_dispatch(
    fa4_dispatch,
) -> None:
    _, calls = fa4_dispatch
    inputs = _native_fa4_inputs(out_dtype=torch.float16)
    inputs["out_dtype"] = torch.bfloat16
    with pytest.raises(ValueError, match="does not match"):
        flash_attn_varlen_func(**inputs)
    assert not calls


def test_fa4_cutedsl_probe_imports_interface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import_module = MagicMock(side_effect=ModuleNotFoundError("no cutlass"))
    monkeypatch.setattr(flash_attn_interface, "import_module", import_module)

    assert (
        flash_attn_interface.fa4_cutedsl_import_error()
        == "ModuleNotFoundError: no cutlass"
    )
    import_module.assert_called_once_with("vllm.vllm_flash_attn.cute.interface")


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(
                    empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                )
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask
        if soft_cap is not None:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


@pytest.mark.parametrize("use_out", [True, False])
@pytest.mark.parametrize(
    "seq_lens", [[(1, 1328), (5, 18), (129, 463)], [(1, 523), (1, 37), (1, 2011)]]
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", SOFT_CAPS)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("fa_version", [2, 3])
@pytest.mark.parametrize("q_dtype", QDTYPES)
@torch.inference_mode()
def test_varlen_with_paged_kv(
    use_out: bool,
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    fa_version: int,
    q_dtype: torch.dtype | None,
) -> None:
    torch.set_default_device("cuda")
    if not is_fa_version_supported(fa_version):
        pytest.skip(
            f"Flash attention version {fa_version} not supported due "
            f'to: "{fa_version_unsupported_reason(fa_version)}"'
        )
    if q_dtype is not None and (dtype != torch.bfloat16 or fa_version == 2):
        pytest.skip(
            "Flash attention with quantized inputs is only "
            "supported on version 3 with bfloat16 base type"
        )
    set_random_seed(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    out = torch.empty_like(query) if use_out else None

    maybe_quantized_query = query
    maybe_quantized_key_cache = key_cache
    maybe_quantized_value_cache = value_cache
    q_descale = None
    k_descale = None
    v_descale = None
    if q_dtype is not None:
        # QKV are drawn from N(0, 1): no need for a fp8 scaling factor
        maybe_quantized_query = query.to(q_dtype)
        maybe_quantized_key_cache = key_cache.to(q_dtype)
        maybe_quantized_value_cache = value_cache.to(q_dtype)

        scale_shape = (num_seqs, num_kv_heads)
        q_descale = torch.ones(scale_shape, dtype=torch.float32)
        k_descale = torch.ones(scale_shape, dtype=torch.float32)
        v_descale = torch.ones(scale_shape, dtype=torch.float32)

    output = flash_attn_varlen_func(
        q=maybe_quantized_query,
        k=maybe_quantized_key_cache,
        v=maybe_quantized_value_cache,
        out=out,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        fa_version=fa_version,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    output = output if not use_out else out

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )
    atol, rtol = 1.5e-2, 1e-2
    if q_dtype is not None:
        atol, rtol = 1.5e-1, 1.5e-1
    (
        torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol),
        f"{torch.max(torch.abs(output - ref_output))}",
    )
