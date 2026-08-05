# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hopper FA4 FP8 cache, descale, and whole-server routing tests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends import fa_utils
from vllm.v1.attention.backends.registry import AttentionBackendEnum


HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
PAGE_SIZE = 16
SEED_LAYERS = {17: 0, 2027: 29, 65537: 31}
PREFILL_ROWS = (
    (19, 3),
    (17, 5, 31, 2, 23, 11, 29, 7, 13, 37, 41, 43, 47, 53, 59, 61, 67),
)
DECODE_ROWS = (
    (3,),
    (17, 11),
    (43, 29, 37, 23, 41, 31, 47, 19),
    tuple(100 + ((17 * index + 7) % 65) for index in range(65)),
)


@dataclass(frozen=True)
class NativeCase:
    name: str
    seed: int
    q_lens: tuple[int, ...]
    k_lens: tuple[int, ...]
    rows: tuple[tuple[int, ...], ...]
    pages: int


NATIVE_CASES = tuple(
    NativeCase(f"prefill-seed-{seed}", seed, (17, 257), (17, 257), PREFILL_ROWS, 96)
    for seed in SEED_LAYERS
) + tuple(
    NativeCase(f"decode-seed-{seed}", seed, (1, 1, 1, 1), (16, 17, 128, 1025), DECODE_ROWS, 192)
    for seed in SEED_LAYERS
)


def _scales(manifest, seed):
    layers = manifest["layers"]
    assert len(layers) == 32
    index = SEED_LAYERS[seed]
    assert layers[index]["layer"] == index
    k_scale = float(layers[index]["k_scale"])
    v_scale = float(layers[index]["v_scale"])
    assert math.isfinite(k_scale) and 0.0 < k_scale != 1.0
    assert math.isfinite(v_scale) and 0.0 < v_scale != 1.0
    return k_scale, v_scale


def _values(generator, shape, scale):
    return (
        torch.randn(shape, dtype=torch.float32, generator=generator)
        .mul_(64.0 * scale)
        .clamp_(-192.0 * scale, 192.0 * scale)
        .to(torch.bfloat16)
    )


def _quantize(values, scale):
    op = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR)
    with set_current_vllm_config(VllmConfig()):
        result, returned_scale = op(
            values.reshape(-1, HEAD_DIM).cuda(), scale.reshape(1)
        )
    assert returned_scale.numel() == 1
    assert returned_scale.data_ptr() == scale.data_ptr()
    return result.reshape(values.shape)


def _expected(values, scale):
    limit = torch.finfo(torch.float8_e4m3fn).max
    return values.float().div(scale).clamp(-limit, limit).to(torch.float8_e4m3fn)


def _table_and_slots(case):
    width = max(len(row) for row in case.rows)
    table = torch.full((len(case.rows), width), -1, dtype=torch.int32)
    flat = []
    per_sequence = []
    for batch, (length, row) in enumerate(zip(case.k_lens, case.rows, strict=True)):
        assert len(row) == math.ceil(length / PAGE_SIZE)
        table[batch, : len(row)] = torch.tensor(row, dtype=torch.int32)
        slots = tuple(row[pos // PAGE_SIZE] * PAGE_SIZE + pos % PAGE_SIZE for pos in range(length))
        flat.extend(slots)
        per_sequence.append(slots)
    return table, torch.tensor(flat, dtype=torch.int64), tuple(per_sequence)


def _hash(tensor):
    import hashlib

    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _current_config(enabled=True):
    return SimpleNamespace(
        attention_config=SimpleNamespace(_hopper_fa4_fp8=enabled),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )


def _implementation():
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    from vllm.v1.attention.backends.utils import AttentionType

    impl = object.__new__(FlashAttentionImpl)
    impl.num_heads = HEADS
    impl.head_size = HEAD_DIM
    impl.scale = 1.0 / math.sqrt(HEAD_DIM)
    impl.num_kv_heads = KV_HEADS
    impl.alibi_slopes = None
    impl.sliding_window = (-1, -1)
    impl.kv_cache_dtype = "fp8"
    impl.logits_soft_cap = 0.0
    impl.kv_sharing_target_layer_name = None
    impl.num_queries_per_kv = HEADS // KV_HEADS
    impl.attn_type = AttentionType.DECODER
    impl.vllm_flash_attn_version = 4
    impl.batch_invariant_enabled = False
    impl.supports_quant_query_input = True
    impl.dcp_world_size = 1
    impl.dcp_rank = 0
    impl.sinks = None
    return impl


def _metadata(case, table, slots):
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

    q_cumulative = (0, *torch.tensor(case.q_lens).cumsum(0).tolist())
    return FlashAttentionMetadata(
        num_actual_tokens=sum(case.q_lens),
        max_query_len=max(case.q_lens),
        query_start_loc=torch.tensor(q_cumulative, dtype=torch.int32, device="cuda"),
        max_seq_len=max(case.k_lens),
        seq_lens=torch.tensor(case.k_lens, dtype=torch.int32, device="cuda"),
        block_table=table.cuda(),
        slot_mapping=slots.cuda(),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        max_num_splits=0,
    )


def _expected_cache(case, values, scale, per_sequence):
    result = torch.zeros(case.pages * PAGE_SIZE, KV_HEADS, HEAD_DIM, dtype=torch.float8_e4m3fn)
    quantized = _expected(values, scale)
    offset = 0
    for length, slots in zip(case.k_lens, per_sequence, strict=True):
        result[list(slots)] = quantized[offset : offset + length]
        offset += length
    return result.reshape(case.pages, PAGE_SIZE, KV_HEADS, HEAD_DIM)


def _reference(query, key_cache, value_cache, case, per_sequence, k_scale, v_scale):
    q = query.float().double().cpu().mul(k_scale)
    key = key_cache.float().double().cpu().reshape(-1, KV_HEADS, HEAD_DIM).mul(k_scale)
    value = value_cache.float().double().cpu().reshape(-1, KV_HEADS, HEAD_DIM).mul(v_scale)
    output = torch.zeros_like(q)
    q_base = 0
    for q_len, k_len, slots in zip(case.q_lens, case.k_lens, per_sequence, strict=True):
        keys = key[list(slots)]
        values = value[list(slots)]
        for row in range(q_len):
            last_key = k_len - q_len + row
            for head in range(HEADS):
                kv_head = head // 4
                scores = torch.mv(
                    keys[: last_key + 1, kv_head], q[q_base + row, head]
                ) / math.sqrt(HEAD_DIM)
                output[q_base + row, head] = (
                    torch.softmax(scores, dim=0)
                    @ values[: last_key + 1, kv_head]
                )
        q_base += q_len
    return output


@pytest.mark.parametrize("case", NATIVE_CASES, ids=lambda case: case.name)
def test_native_writer_reader(case, fp8_scale_manifest, fp8_record, monkeypatch):
    from vllm.v1.attention.backends import flash_attn as backend

    k_value, v_value = _scales(fp8_scale_manifest, case.seed)
    generator = torch.Generator().manual_seed(case.seed)
    query_cpu = _values(generator, (sum(case.q_lens), HEADS, HEAD_DIM), k_value)
    key_cpu = _values(generator, (sum(case.k_lens), KV_HEADS, HEAD_DIM), k_value)
    value_cpu = _values(generator, (sum(case.k_lens), KV_HEADS, HEAD_DIM), v_value)
    k_scale = torch.tensor(k_value, dtype=torch.float32, device="cuda")
    v_scale = torch.tensor(v_value, dtype=torch.float32, device="cuda")
    query = _quantize(query_cpu, k_scale)
    expected_query = _expected(query_cpu, k_value)
    assert torch.equal(
        query.cpu().view(torch.uint8), expected_query.view(torch.uint8)
    )
    table, slots, per_sequence = _table_and_slots(case)
    kv_cache = torch.zeros(
        case.pages,
        KV_HEADS,
        PAGE_SIZE,
        2 * HEAD_DIM,
        dtype=torch.uint8,
        device="cuda",
    )
    layer = SimpleNamespace(_q_scale=k_scale, _k_scale=k_scale, _v_scale=v_scale)
    impl = _implementation()

    q1_final = []
    ordinary = []
    offset = 0
    for q_len, k_len in zip(case.q_lens, case.k_lens, strict=True):
        indices = list(range(offset, offset + k_len))
        if q_len == 1 and k_len > 1:
            ordinary.extend(indices[:-1])
            q1_final.append(indices[-1])
        else:
            ordinary.extend(indices)
        offset += k_len
    for indices in (ordinary, q1_final):
        if indices:
            index = torch.tensor(indices, dtype=torch.int64, device="cuda")
            impl.do_kv_cache_update(
                layer,
                key_cpu.cuda().index_select(0, index),
                value_cpu.cuda().index_select(0, index),
                kv_cache,
                slots.cuda().index_select(0, index),
            )

    raw_key, raw_value = kv_cache.transpose(1, 2).split(HEAD_DIM, dim=-1)
    key_cache = raw_key.view(torch.float8_e4m3fn)
    value_cache = raw_value.view(torch.float8_e4m3fn)
    expected_key = _expected_cache(case, key_cpu, k_value, per_sequence)
    expected_value = _expected_cache(case, value_cpu, v_value, per_sequence)
    assert torch.equal(key_cache.cpu().view(torch.uint8), expected_key.view(torch.uint8))
    assert torch.equal(value_cache.cpu().view(torch.uint8), expected_value.view(torch.uint8))

    observed = []
    original = backend.flash_attn_varlen_func

    def spy(**kwargs):
        observed.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(backend, "flash_attn_varlen_func", spy)
    output = torch.empty(sum(case.q_lens), HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    metadata = _metadata(case, table, slots)
    cache_before = kv_cache.clone()
    with set_current_vllm_config(_current_config()):
        returned = impl.forward(
            layer,
            query,
            torch.empty(0, device="cuda"),
            torch.empty(0, device="cuda"),
            kv_cache,
            metadata,
            output,
        )
    assert returned is output
    assert len(observed) == 1
    for name, owner in (("q_descale", k_scale), ("k_descale", k_scale), ("v_descale", v_scale)):
        descale = observed[0][name]
        assert descale.shape == (len(case.q_lens), KV_HEADS)
        assert descale.stride() == (0, 0)
        assert descale.data_ptr() == owner.data_ptr()
    assert torch.equal(kv_cache, cache_before)
    reference = _reference(query, key_cache, value_cache, case, per_sequence, k_value, v_value)
    error = output.float().double().cpu() - reference
    nrms = float(error.square().mean().sqrt() / max(float(reference.square().mean().sqrt()), 0.001))
    assert float(error.abs().max()) <= 0.16
    assert nrms <= 0.025
    fp8_record(
        {
            "id": case.name,
            "plane": "vllm-native",
            "seed": case.seed,
            "scale_layer": SEED_LAYERS[case.seed],
            "q_lengths": case.q_lens,
            "k_lengths": case.k_lens,
            "block_table": table.tolist(),
            "slot_mapping": per_sequence,
            "hashes": {
                "q": _hash(query),
                "k": _hash(key_cache),
                "v": _hash(value_cache),
                "output": _hash(output),
            },
            "expected_hashes": {
                "q": _hash(expected_query),
                "logical_k": _hash(_expected(key_cpu, k_value)),
                "logical_v": _hash(_expected(value_cpu, v_value)),
                "k_cache": _hash(expected_key),
                "v_cache": _hash(expected_value),
            },
            "output_maxabs": float(error.abs().max()),
            "output_nrms": nrms,
            "requested_backend": "FLASH_ATTN",
            "requested_version": 4,
            "effective_backend": "FLASH_ATTN",
            "effective_version": 4,
        }
    )


def _routing_config():
    attention = SimpleNamespace(
        backend=AttentionBackendEnum.FLASH_ATTN,
        backend_per_kind={},
        flash_attn_version=4,
        _flash_attn_version_fallback=False,
        _flash_attn_version_required=False,
        _hopper_fa4_fp8=False,
    )
    cache = SimpleNamespace(
        cache_dtype="fp8",
        _checkpoint_implied_fp8=False,
        block_size=16,
        sliding_window=None,
        enable_prefix_caching=False,
        calculate_kv_scales=False,
        kv_cache_dtype_skip_layers=[],
    )
    parallel = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
    )
    quantization = {
        "quant_method": "compressed-tensors",
        "quantization_status": "frozen",
        "kv_cache_scheme": {
            "dynamic": False,
            "num_bits": 8,
            "strategy": "tensor",
            "symmetric": True,
            "type": "float",
        },
    }
    model = SimpleNamespace(
        dtype=torch.bfloat16,
        hf_text_config=SimpleNamespace(),
        model_arch_config=SimpleNamespace(quantization_config=quantization),
        use_mla=False,
        is_diffusion=False,
        architecture="LlamaForCausalLM",
        is_multimodal_model=False,
        is_encoder_decoder=False,
        disable_cascade_attn=True,
        get_head_size=lambda: 128,
        get_num_attention_heads=lambda parallel: 32,
        get_num_kv_heads=lambda parallel: 8,
    )
    return SimpleNamespace(
        attention_config=attention,
        cache_config=cache,
        parallel_config=parallel,
        model_config=model,
        speculative_config=None,
    )


@pytest.fixture
def hopper(monkeypatch):
    monkeypatch.setattr(
        fa_utils.current_platform,
        "get_device_capability",
        lambda: DeviceCapability(major=9, minor=0),
    )
    monkeypatch.setattr(fa_utils.envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(fa_utils, "_fa4_cute_import_error", lambda: None)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)


ROUTING_CASES = (
    "admitted",
    "default-version",
    "explicit-fa3",
    "checkpoint-implied",
    "head-scales",
    "runtime-scales",
    "fp16-output",
    "head-dim-64",
    "gqa-8",
    "parallel-and-page",
    "e5m2",
    "head-dim-512",
)


@pytest.mark.parametrize("route", ROUTING_CASES)
def test_routing(route, hopper, monkeypatch, fp8_record):
    config = _routing_config()
    warning = MagicMock()
    info = MagicMock()
    monkeypatch.setattr(fa_utils.logger, "warning_once", warning)
    monkeypatch.setattr(fa_utils.logger, "info_once", info)

    if route == "default-version":
        config.attention_config.flash_attn_version = None
    elif route == "explicit-fa3":
        config.attention_config.flash_attn_version = 3
    elif route == "checkpoint-implied":
        config.cache_config._checkpoint_implied_fp8 = True
    elif route == "head-scales":
        scheme = config.model_config.model_arch_config.quantization_config
        scheme["kv_cache_scheme"]["strategy"] = "head"
    elif route == "runtime-scales":
        config.cache_config.calculate_kv_scales = True
    elif route == "fp16-output":
        config.model_config.dtype = torch.float16
    elif route == "head-dim-64":
        config.model_config.get_head_size = lambda: 64
    elif route == "gqa-8":
        config.model_config.get_num_kv_heads = lambda parallel: 4
    elif route == "parallel-and-page":
        config.cache_config.block_size = 32
        config.parallel_config.tensor_parallel_size = 2
        config.parallel_config.pipeline_parallel_size = 2
        config.parallel_config.decode_context_parallel_size = 2
    elif route == "e5m2":
        config.cache_config.cache_dtype = "fp8_e5m2"
    elif route == "head-dim-512":
        config.model_config.get_head_size = lambda: 512

    if route == "head-dim-512":
        with pytest.raises(ValueError, match="requires FA4"):
            fa_utils.resolve_flash_attn_version(config)
        effective = "startup-error"
    else:
        effective = fa_utils.resolve_flash_attn_version(config)
        if route == "admitted":
            assert effective == 4 and config.attention_config._hopper_fa4_fp8
            assert warning.call_count == 0 and info.call_count == 1
        elif route in ("default-version", "explicit-fa3"):
            assert effective == config.attention_config.flash_attn_version
            assert warning.call_count == info.call_count == 0
        elif route == "e5m2":
            assert effective == 4 and not config.attention_config._hopper_fa4_fp8
            assert warning.call_count == 0 and info.call_count == 1
            assert not fa_utils.flash_attn_supports_kv_cache_dtype(
                "fp8_e5m2",
                requires_alibi=False,
                head_size=128,
                head_size_v=128,
                has_sinks=False,
            )
        else:
            assert effective == 3 and config.attention_config._flash_attn_version_fallback
            assert warning.call_count == 1
            reason = warning.call_args.args[1]
            if route == "parallel-and-page":
                assert reason == "page, TP, PP, DCP"
            elif route == "fp16-output":
                assert reason == "activation/output dtype"
                assert not config.attention_config._hopper_fa4_fp8
    fp8_record(
        {
            "id": route,
            "plane": "vllm-routing",
            "requested_version": 4 if route not in ("default-version", "explicit-fa3") else route,
            "effective_version": effective,
            "fallback_warnings": warning.call_count,
            "admitted": config.attention_config._hopper_fa4_fp8,
        }
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("form", ["dense", "paged-prefill", "paged-decode"])
@pytest.mark.parametrize("return_lse", [False, True], ids=["output", "output-lse"])
def test_routing_and_descales(dtype, form, return_lse, monkeypatch):
    from vllm.vllm_flash_attn import flash_attn_interface as interface
    from vllm.vllm_flash_attn.cute import interface as cute_interface

    if form == "dense":
        q = torch.empty(4, HEADS, HEAD_DIM, dtype=dtype)
        k = torch.empty(6, KV_HEADS, HEAD_DIM, dtype=dtype)
        v = torch.empty_like(k)
        cu_q = torch.tensor([0, 4], dtype=torch.int32)
        cu_k = torch.tensor([0, 6], dtype=torch.int32)
        seqused_k = None
        table = None
        max_q, max_k, causal = 4, 6, True
    elif form == "paged-prefill":
        q = torch.empty(4, HEADS, HEAD_DIM, dtype=dtype)
        k = torch.empty(8, PAGE_SIZE, KV_HEADS, HEAD_DIM, dtype=dtype)
        v = torch.empty_like(k)
        cu_q = torch.tensor([0, 4], dtype=torch.int32)
        cu_k = None
        seqused_k = torch.tensor([6], dtype=torch.int32)
        table = torch.tensor([[3]], dtype=torch.int32)
        max_q, max_k, causal = 4, 6, True
    else:
        q = torch.empty(1, HEADS, HEAD_DIM, dtype=dtype)
        k = torch.empty(8, PAGE_SIZE, KV_HEADS, HEAD_DIM, dtype=dtype)
        v = torch.empty_like(k)
        cu_q = torch.tensor([0, 1], dtype=torch.int32)
        cu_k = None
        seqused_k = torch.tensor([16], dtype=torch.int32)
        table = torch.tensor([[3]], dtype=torch.int32)
        max_q, max_k, causal = 1, 16, True

    observed = []
    expected_out = torch.empty_like(q)
    expected_lse = torch.empty(HEADS, q.shape[0], dtype=torch.float32)

    def fake_fwd(*args, **kwargs):
        observed.append((args, kwargs))
        return expected_out, expected_lse if return_lse else None, None, None

    monkeypatch.setattr(cute_interface, "_flash_attn_fwd", fake_fwd)
    bogus = torch.tensor(3.0)
    with set_current_vllm_config(_current_config(enabled=False)):
        result = interface.flash_attn_varlen_func(
            q,
            k,
            v,
            max_seqlen_q=max_q,
            cu_seqlens_q=cu_q,
            max_seqlen_k=max_k,
            cu_seqlens_k=cu_k,
            seqused_k=seqused_k,
            causal=causal,
            window_size=[-1, -1],
            block_table=table,
            return_softmax_lse=return_lse,
            q_descale=bogus,
            k_descale=bogus,
            v_descale=bogus,
            num_splits=1,
            fa_version=4,
        )
    assert len(observed) == 1
    args, kwargs = observed[0]
    assert args[:3] == (q, k, v)
    assert kwargs["cu_seqlens_q"] is cu_q
    assert kwargs["cu_seqlens_k"] is cu_k
    assert kwargs["seqused_k"] is seqused_k
    assert kwargs["page_table"] is table
    assert kwargs["causal"] is causal
    assert kwargs["num_splits"] == 1
    assert kwargs["q_descale"] is None
    assert kwargs["k_descale"] is None
    assert kwargs["v_descale"] is None
    if return_lse:
        assert result[0] is expected_out and result[1] is expected_lse
    else:
        assert result is expected_out
