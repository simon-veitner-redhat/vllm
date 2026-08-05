# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graph, stream-safety, and wake-up tests for Hopper FA4 FP8."""

from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm.config import set_current_vllm_config


def _load_attention_helpers():
    path = Path(__file__).resolve().parents[1] / "attention" / "test_hopper_fa4_fp8.py"
    spec = importlib.util.spec_from_file_location("hopper_fa4_fp8_attention_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


helpers = _load_attention_helpers()
HEADS = helpers.HEADS
KV_HEADS = helpers.KV_HEADS
HEAD_DIM = helpers.HEAD_DIM
PAGE_SIZE = helpers.PAGE_SIZE


@dataclass(frozen=True)
class GraphCase:
    name: str
    capacity: int
    request_count: int
    uniform: bool
    seed: int
    q_lens: tuple[int, ...]
    k_lens: tuple[int, ...]
    rows: tuple[tuple[int, ...], ...]
    pages: int


def _graph_cases():
    descriptors = (
        ("capacity-1", 1, 1, True, (1,), (16,), ((3,),), 64),
        ("capacity-2", 2, 2, False, (2,), (2,), ((17,),), 64),
        ("capacity-4", 4, 4, False, (1, 3), (16, 3), ((3,), (17,)), 64),
        ("capacity-8", 8, 8, False, (1, 7), (16, 7), ((3,), (17,)), 64),
        ("capacity-16", 16, 16, False, (1, 15), (16, 15), ((3,), (17,)), 64),
        ("capacity-24", 24, 16, False, (1, 23), (16, 23), ((3,), (17, 11)), 64),
        ("capacity-32", 32, 16, False, (1, 31), (16, 31), ((3,), (17, 11)), 64),
    )
    return tuple(
        GraphCase(
            f"{name}-seed-{seed}",
            capacity,
            requests,
            uniform,
            seed,
            q_lens,
            k_lens,
            rows,
            pages,
        )
        for name, capacity, requests, uniform, q_lens, k_lens, rows, pages in descriptors
        for seed in helpers.SEED_LAYERS
    )


GRAPH_CASES = _graph_cases()
assert len(GRAPH_CASES) == 21


def _padded_case(case):
    inactive = case.request_count - len(case.q_lens)
    q_lens = case.q_lens + (0,) * inactive
    k_lens = case.k_lens + (0,) * inactive
    rows = case.rows + ((),) * inactive
    native = helpers.NativeCase(case.name, case.seed, q_lens, k_lens, rows, case.pages)
    width = max(len(row) for row in case.rows)
    table = torch.full((case.request_count, width), -1, dtype=torch.int32)
    flat = []
    per_sequence = []
    for batch, (length, row) in enumerate(zip(q_lens, rows, strict=True)):
        if length:
            table[batch, : len(row)] = torch.tensor(row, dtype=torch.int32)
        slots = tuple(
            row[pos // PAGE_SIZE] * PAGE_SIZE + pos % PAGE_SIZE
            for pos in range(k_lens[batch])
        )
        flat.extend(slots)
        per_sequence.append(slots)
    return native, table, torch.tensor(flat, dtype=torch.int64), tuple(per_sequence)


def _layer(q_scale, k_scale, v_scale):
    return SimpleNamespace(_q_scale=q_scale, _k_scale=k_scale, _v_scale=v_scale)


def _write(impl, layer, native, key, value, kv_cache, slots):
    q1_final = []
    ordinary = []
    offset = 0
    for q_len, k_len in zip(native.q_lens, native.k_lens, strict=True):
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
                key.index_select(0, index),
                value.index_select(0, index),
                kv_cache,
                slots.cuda().index_select(0, index),
            )


def _prepare_graph(case, manifest):
    native, table, slots, per_sequence = _padded_case(case)
    k_value, v_value = helpers._scales(manifest, case.seed)
    generator = torch.Generator().manual_seed(case.seed)
    query_cpu = helpers._values(generator, (case.capacity, HEADS, HEAD_DIM), k_value)
    key_cpu = helpers._values(generator, (sum(case.k_lens), KV_HEADS, HEAD_DIM), k_value)
    value_cpu = helpers._values(generator, (sum(case.k_lens), KV_HEADS, HEAD_DIM), v_value)
    k_scale = torch.tensor(k_value, dtype=torch.float32, device="cuda")
    v_scale = torch.tensor(v_value, dtype=torch.float32, device="cuda")
    query = helpers._quantize(query_cpu, k_scale)
    key = key_cpu.cuda()
    value = value_cpu.cuda()
    kv_cache = torch.zeros(
        case.pages,
        KV_HEADS,
        PAGE_SIZE,
        2 * HEAD_DIM,
        dtype=torch.uint8,
        device="cuda",
    )
    impl = helpers._implementation()
    layer = _layer(k_scale, k_scale, v_scale)
    _write(impl, layer, native, key, value, kv_cache, slots)
    metadata = helpers._metadata(native, table, slots)
    metadata.max_num_splits = 32
    output = torch.empty(case.capacity, HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    return SimpleNamespace(
        case=case,
        native=native,
        table=table,
        slots=slots,
        per_sequence=per_sequence,
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache,
        impl=impl,
        layer=layer,
        metadata=metadata,
        output=output,
        k_value=k_value,
        v_value=v_value,
    )


def _forward(prepared):
    return prepared.impl.forward(
        prepared.layer,
        prepared.query,
        torch.empty(0, device="cuda"),
        torch.empty(0, device="cuda"),
        prepared.kv_cache,
        prepared.metadata,
        prepared.output,
    )


def _capture(prepared):
    with set_current_vllm_config(helpers._current_config()):
        _forward(prepared)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with set_current_vllm_config(helpers._current_config()):
            _forward(prepared)
    return graph


def _reset_compile_monitor():
    from vllm.vllm_flash_attn.flash_attn_interface import (
        disarm_flash_attn_compile_monitor,
    )

    disarm_flash_attn_compile_monitor()


@pytest.fixture(autouse=True)
def _isolated_compile_monitor():
    _reset_compile_monitor()
    yield
    _reset_compile_monitor()


@pytest.mark.parametrize("case", GRAPH_CASES, ids=lambda case: case.name)
def test_graph_replay(case, fp8_scale_manifest, fp8_record):
    from vllm.vllm_flash_attn.flash_attn_interface import (
        arm_flash_attn_compile_monitor,
        get_flash_attn_compile_cache_keys,
    )

    prepared = _prepare_graph(case, fp8_scale_manifest)
    cache_hash = helpers._hash(prepared.kv_cache)
    graph = _capture(prepared)
    inventory = arm_flash_attn_compile_monitor()
    outputs = [prepared.output.clone()]
    for _ in range(10):
        graph.replay()
        torch.cuda.synchronize()
        outputs.append(prepared.output.clone())
        assert helpers._hash(prepared.kv_cache) == cache_hash
    assert all(torch.equal(outputs[0], output) for output in outputs[1:])
    assert torch.isfinite(outputs[0]).all()
    raw_key, raw_value = prepared.kv_cache.transpose(1, 2).split(HEAD_DIM, dim=-1)
    reference = helpers._reference(
        prepared.query,
        raw_key.view(torch.float8_e4m3fn),
        raw_value.view(torch.float8_e4m3fn),
        prepared.native,
        prepared.per_sequence,
        prepared.k_value,
        prepared.v_value,
    )
    error = outputs[0].float().double().cpu() - reference
    vector_maxabs = error.abs().amax(dim=-1)
    vector_nrms = error.square().mean(dim=-1).sqrt() / reference.square().mean(
        dim=-1
    ).sqrt().clamp_min(0.001)
    output_nrms = float(
        error.square().mean().sqrt()
        / max(float(reference.square().mean().sqrt()), 0.001)
    )
    assert float(vector_maxabs.max()) <= 0.16
    assert float(vector_nrms.max()) <= 0.08
    assert output_nrms <= 0.025
    sequence_nrms = []
    q_base = 0
    for q_len in prepared.native.q_lens:
        if q_len:
            sequence_error = error[q_base : q_base + q_len]
            sequence_reference = reference[q_base : q_base + q_len]
            nrms = float(
                sequence_error.square().mean().sqrt()
                / max(
                    float(sequence_reference.square().mean().sqrt()),
                    0.001,
                )
            )
            assert nrms <= 0.025
            sequence_nrms.append(nrms)
        q_base += q_len
    post_warmup_keys = get_flash_attn_compile_cache_keys()
    assert post_warmup_keys == inventory
    fp8_record(
        {
            "id": case.name,
            "plane": "vllm-graph",
            "seed": case.seed,
            "scale_layer": helpers.SEED_LAYERS[case.seed],
            "descriptor": {
                "capacity": case.capacity,
                "requests": case.request_count,
                "uniform": case.uniform,
            },
            "q_lengths": prepared.native.q_lens,
            "k_lengths": prepared.native.k_lens,
            "block_table": prepared.table.tolist(),
            "cache_hash": cache_hash,
            "output_hash": helpers._hash(outputs[0]),
            "output_maxabs": float(vector_maxabs.max()),
            "output_vector_nrms": float(vector_nrms.max()),
            "output_nrms": output_nrms,
            "sequence_output_nrms": sequence_nrms,
            "capture_count": 1,
            "replay_count": 10,
            "warmup_compile_cache_keys": inventory,
            "post_warmup_compile_cache_keys": post_warmup_keys,
            "requested_version": 4,
            "effective_version": 4,
        }
    )


def test_stream_concurrency(fp8_scale_manifest, fp8_record):
    from vllm.vllm_flash_attn.flash_attn_interface import (
        arm_flash_attn_compile_monitor,
        get_flash_attn_compile_cache_keys,
    )

    prefill_case = GraphCase(
        "stream-prefill",
        274,
        2,
        False,
        17,
        (17, 257),
        (17, 257),
        helpers.PREFILL_ROWS,
        96,
    )
    decode_case = GraphCase(
        "stream-decode",
        4,
        4,
        True,
        2027,
        (1, 1, 1, 1),
        (16, 17, 128, 1025),
        helpers.DECODE_ROWS,
        192,
    )
    prefill = _prepare_graph(prefill_case, fp8_scale_manifest)
    decode = _prepare_graph(decode_case, fp8_scale_manifest)
    prefill.metadata.max_num_splits = 1
    decode.metadata.max_num_splits = 0
    with set_current_vllm_config(helpers._current_config()):
        prefill_serial = _forward(prefill).clone()
        decode_serial = _forward(decode).clone()
    warmup_keys = arm_flash_attn_compile_monitor()
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    prefill_outputs = []
    decode_outputs = []
    for _ in range(100):
        with torch.cuda.stream(streams[0]), set_current_vllm_config(
            helpers._current_config()
        ):
            _forward(prefill)
            prefill_outputs.append(prefill.output.clone())
        with torch.cuda.stream(streams[1]), set_current_vllm_config(
            helpers._current_config()
        ):
            _forward(decode)
            decode_outputs.append(decode.output.clone())
    torch.cuda.synchronize()
    assert all(torch.equal(output, prefill_serial) for output in prefill_outputs)
    assert all(torch.equal(output, decode_serial) for output in decode_outputs)
    post_warmup_keys = get_flash_attn_compile_cache_keys()
    assert post_warmup_keys == warmup_keys
    fp8_record(
        {
            "id": "two-stream-overlap",
            "plane": "vllm-stream-concurrency",
            "iterations": 100,
            "stream_count": 2,
            "prefill_output_hash": helpers._hash(prefill.output),
            "decode_output_hash": helpers._hash(decode.output),
            "independent_caches": prefill.kv_cache.data_ptr() != decode.kv_cache.data_ptr(),
            "warmup_compile_cache_keys": warmup_keys,
            "post_warmup_compile_cache_keys": post_warmup_keys,
        }
    )


def _attention_module(q_value, k_value, v_value):
    from vllm.model_executor.layers.attention import Attention

    module = Attention.__new__(Attention)
    torch.nn.Module.__init__(module)
    module._q_scale = torch.tensor(q_value, dtype=torch.float32, device="cuda")
    module._k_scale = torch.tensor(k_value, dtype=torch.float32, device="cuda")
    module._v_scale = torch.tensor(v_value, dtype=torch.float32, device="cuda")
    module._q_scale_float = q_value
    module._k_scale_float = k_value
    module._v_scale_float = v_value
    module._k_scale_cpu = torch.tensor(k_value, dtype=torch.float32)
    module._v_scale_cpu = torch.tensor(v_value, dtype=torch.float32)
    module._authoritative_qkv_scales = (q_value, k_value, v_value)
    return module


def test_sleep_wake(fp8_scale_manifest, fp8_record):
    from vllm.vllm_flash_attn.flash_attn_interface import (
        arm_flash_attn_compile_monitor,
        get_flash_attn_compile_cache_keys,
    )
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    modules = {}
    for layer in fp8_scale_manifest["layers"]:
        index = int(layer["layer"])
        k_value = float(layer["k_scale"])
        v_value = float(layer["v_scale"])
        module = _attention_module(k_value, k_value, v_value)
        modules[f"layer-{index}"] = module

    case = GraphCase(
        "post-wake-decode",
        4,
        4,
        True,
        2027,
        (1, 1, 1, 1),
        (16, 17, 128, 1025),
        helpers.DECODE_ROWS,
        192,
    )
    prepared = _prepare_graph(case, fp8_scale_manifest)
    prepared.layer = modules[f"layer-{helpers.SEED_LAYERS[case.seed]}"]
    graph = _capture(prepared)
    warmup_keys = arm_flash_attn_compile_monitor()
    before = prepared.output.clone()
    cache_before = helpers._hash(prepared.kv_cache)

    snapshots = {}
    for layer in fp8_scale_manifest["layers"]:
        index = int(layer["layer"])
        module = modules[f"layer-{index}"]
        snapshots[index] = {
            name: (
                id(getattr(module, name)),
                getattr(module, name).data_ptr(),
                float(getattr(module, name)),
            )
            for name in ("_q_scale", "_k_scale", "_v_scale")
        }
        module._q_scale.fill_(1.0)
        module._k_scale.fill_(1.0)
        module._v_scale.fill_(1.0)

    runner = object.__new__(GPUModelRunner)
    runner.cache_config = SimpleNamespace(cache_dtype="fp8")
    runner.vllm_config = helpers._current_config()
    runner.compilation_config = SimpleNamespace(static_forward_context=modules)
    runner.kv_caches = [prepared.kv_cache] + [
        torch.full((64,), 0xA5, dtype=torch.uint8, device="cuda")
        for _ in range(len(modules) - 1)
    ]
    GPUModelRunner.init_fp8_kv_scales(runner)
    assert all(torch.count_nonzero(cache) == 0 for cache in runner.kv_caches)
    scale_records = []
    for layer in fp8_scale_manifest["layers"]:
        index = int(layer["layer"])
        module = modules[f"layer-{index}"]
        expected = (float(layer["k_scale"]), float(layer["k_scale"]), float(layer["v_scale"]))
        for name, value in zip(("_q_scale", "_k_scale", "_v_scale"), expected, strict=True):
            old_id, old_ptr, _ = snapshots[index][name]
            tensor = getattr(module, name)
            assert id(tensor) == old_id and tensor.data_ptr() == old_ptr
            assert float(tensor) == value
            assert tensor.dtype == torch.float32
        scale_record = {"layer": index}
        for name in ("_q_scale", "_k_scale", "_v_scale"):
            tensor = getattr(module, name)
            scale_record[name] = {
                "python_id": id(tensor),
                "storage_address": tensor.data_ptr(),
                "dtype": str(tensor.dtype),
                "value": float(tensor),
            }
        scale_records.append(scale_record)

    _write(
        prepared.impl,
        prepared.layer,
        prepared.native,
        prepared.key,
        prepared.value,
        prepared.kv_cache,
        prepared.slots,
    )
    assert helpers._hash(prepared.kv_cache) == cache_before
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(prepared.output, before)
    post_warmup_keys = get_flash_attn_compile_cache_keys()
    assert post_warmup_keys == warmup_keys
    fp8_record(
        {
            "id": "sleep-wake",
            "plane": "vllm-lifecycle",
            "layer_count": len(modules),
            "scale_identity_preserved": True,
            "scale_storage_preserved": True,
            "scale_records": scale_records,
            "cache_zeroed": True,
            "cache_rebuilt": True,
            "graph_reused": True,
            "output_hash": helpers._hash(prepared.output),
            "warmup_compile_cache_keys": warmup_keys,
            "post_warmup_compile_cache_keys": post_warmup_keys,
        }
    )


def _prepare_non_fp8(dtype, form):
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd

    if form == "prefill":
        q_lens, k_lens, rows, pages, tile_m = (17, 257), (17, 257), helpers.PREFILL_ROWS, 96, 128
    else:
        q_lens = (1, 1, 1, 1)
        k_lens = (16, 17, 128, 1025)
        rows, pages, tile_m = helpers.DECODE_ROWS, 192, 64
    case = helpers.NativeCase(f"non-fp8-{form}", 424242, q_lens, k_lens, rows, pages)
    generator = torch.Generator().manual_seed(case.seed)
    query = (
        torch.randn((sum(q_lens), HEADS, HEAD_DIM), generator=generator)
        .mul_(2)
        .clamp_(-6, 6)
        .to(dtype)
        .cuda()
    )
    logical_k = (
        torch.randn((sum(k_lens), KV_HEADS, HEAD_DIM), generator=generator)
        .mul_(2)
        .clamp_(-6, 6)
        .to(dtype)
    )
    logical_v = (
        torch.randn((sum(k_lens), KV_HEADS, HEAD_DIM), generator=generator)
        .mul_(2)
        .clamp_(-6, 6)
        .to(dtype)
    )
    table, _, per_sequence = helpers._table_and_slots(case)
    key = torch.zeros(pages * PAGE_SIZE, KV_HEADS, HEAD_DIM, dtype=dtype)
    value = torch.zeros_like(key)
    offset = 0
    for length, slots in zip(k_lens, per_sequence, strict=True):
        key[list(slots)] = logical_k[offset : offset + length]
        value[list(slots)] = logical_v[offset : offset + length]
        offset += length
    key = key.reshape(pages, PAGE_SIZE, KV_HEADS, HEAD_DIM)
    value = value.reshape_as(key)
    cache = torch.zeros(
        pages,
        KV_HEADS,
        PAGE_SIZE,
        2 * HEAD_DIM,
        dtype=dtype,
        device="cuda",
    )
    key_cache, value_cache = cache.transpose(1, 2).split(HEAD_DIM, dim=-1)
    key_cache.copy_(key.cuda())
    value_cache.copy_(value.cuda())
    key = key_cache
    value = value_cache
    expected_key = key.clone()
    expected_value = value.clone()
    table_cuda = table.cuda()
    cu_q = torch.tensor(
        (0, *torch.tensor(q_lens).cumsum(0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )
    used_k = torch.tensor(k_lens, dtype=torch.int32, device="cuda")
    output = torch.empty_like(query)
    lse = torch.empty(HEADS, sum(q_lens), dtype=torch.float32, device="cuda")

    def run():
        _flash_attn_fwd(
            query,
            key,
            value,
            cu_seqlens_q=cu_q,
            seqused_k=used_k,
            max_seqlen_q=max(q_lens),
            max_seqlen_k=max(k_lens),
            page_table=table_cuda,
            causal=True,
            tile_mn=(tile_m, 128),
            num_splits=32,
            pack_gqa=True,
            return_lse=True,
            out=output,
            lse=lse,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            _arch=90,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    return SimpleNamespace(
        case=case,
        query=query,
        logical_k=logical_k,
        logical_v=logical_v,
        per_sequence=per_sequence,
        key=key,
        value=value,
        expected_key=expected_key,
        expected_value=expected_value,
        graph_inputs=(table_cuda, cu_q, used_k),
        output=output,
        lse=lse,
        graph=graph,
    )


def _assert_non_fp8_bounds(prepared, dtype):
    query = prepared.query.float().double().cpu()
    key = (
        prepared.key.float()
        .double()
        .cpu()
        .reshape(-1, KV_HEADS, HEAD_DIM)
    )
    value = (
        prepared.value.float()
        .double()
        .cpu()
        .reshape(-1, KV_HEADS, HEAD_DIM)
    )
    reference = torch.zeros_like(query)
    reference_lse = torch.empty(query.shape[0], HEADS, dtype=torch.float64)
    q_base = 0
    for q_len, k_len, slots in zip(
        prepared.case.q_lens,
        prepared.case.k_lens,
        prepared.per_sequence,
        strict=True,
    ):
        keys = key[list(slots)]
        values = value[list(slots)]
        for row in range(q_len):
            last_key = k_len - q_len + row
            for head in range(HEADS):
                kv_head = head // 4
                scores = torch.mv(
                    keys[: last_key + 1, kv_head], query[q_base + row, head]
                ) / math.sqrt(HEAD_DIM)
                reference_lse[q_base + row, head] = torch.logsumexp(scores, 0)
                reference[q_base + row, head] = (
                    torch.softmax(scores, 0) @ values[: last_key + 1, kv_head]
                )
        q_base += q_len

    output = prepared.output.float().double().cpu()
    lse = prepared.lse.transpose(0, 1).double().cpu()
    if dtype == torch.bfloat16:
        vector_bounds = (0.03, 0.02, 0.03)
        sequence_bounds = (0.01, 0.01)
    else:
        vector_bounds = (0.015, 0.01, 0.02)
        sequence_bounds = (0.005, 0.008)
    q_base = 0
    for q_len in prepared.case.q_lens:
        got = output[q_base : q_base + q_len]
        ref = reference[q_base : q_base + q_len]
        error = got - ref
        seq_nrms = float(
            error.square().mean().sqrt()
            / max(float(ref.square().mean().sqrt()), 0.001)
        )
        seq_lse = float(
            (
                lse[q_base : q_base + q_len]
                - reference_lse[q_base : q_base + q_len]
            )
            .square()
            .mean()
            .sqrt()
        )
        assert seq_nrms <= sequence_bounds[0]
        assert seq_lse <= sequence_bounds[1]
        for row in range(q_len):
            for head in range(HEADS):
                vector_error = error[row, head]
                maxabs = float(vector_error.abs().max())
                nrms = float(
                    vector_error.square().mean().sqrt()
                    / max(float(ref[row, head].square().mean().sqrt()), 0.001)
                )
                lse_abs = float(
                    abs(
                        lse[q_base + row, head]
                        - reference_lse[q_base + row, head]
                    )
                )
                assert maxabs <= vector_bounds[0]
                assert nrms <= vector_bounds[1]
                assert lse_abs <= vector_bounds[2]
        q_base += q_len


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("form", ["prefill", "decode"])
def test_non_fp8_graph_replay(dtype, form):
    prepared = _prepare_non_fp8(dtype, form)
    cache_hash = (helpers._hash(prepared.key), helpers._hash(prepared.value))
    reference_out, reference_lse = prepared.output.clone(), prepared.lse.clone()
    _assert_non_fp8_bounds(prepared, dtype)
    for _ in range(10):
        prepared.graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(prepared.output, reference_out)
        assert torch.equal(prepared.lse, reference_lse)
        assert (helpers._hash(prepared.key), helpers._hash(prepared.value)) == cache_hash


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_non_fp8_scale_lifecycle(dtype, fp8_record):
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    module = Attention.__new__(Attention)
    torch.nn.Module.__init__(module)
    for name in ("_q_scale", "_k_scale", "_v_scale"):
        setattr(module, name, torch.ones((), dtype=torch.float32, device="cuda"))
    before = {
        name: (
            id(getattr(module, name)),
            getattr(module, name).data_ptr(),
            float(getattr(module, name)),
        )
        for name in ("_q_scale", "_k_scale", "_v_scale")
    }
    runner = object.__new__(GPUModelRunner)
    runner.cache_config = SimpleNamespace(cache_dtype="auto")
    runner.vllm_config = helpers._current_config(enabled=False)
    runner.compilation_config = SimpleNamespace(static_forward_context={"layer": module})
    prepared = _prepare_non_fp8(dtype, "decode")
    output_before = prepared.output.clone()
    lse_before = prepared.lse.clone()
    runner.kv_caches = [prepared.key, prepared.value]
    GPUModelRunner.init_fp8_kv_scales(runner)
    for name, snapshot in before.items():
        tensor = getattr(module, name)
        assert (id(tensor), tensor.data_ptr(), float(tensor)) == snapshot
    prepared.key.zero_()
    prepared.value.zero_()
    assert torch.count_nonzero(prepared.key) == 0
    assert torch.count_nonzero(prepared.value) == 0
    prepared.key.copy_(prepared.expected_key)
    prepared.value.copy_(prepared.expected_value)
    assert torch.equal(prepared.key, prepared.expected_key)
    assert torch.equal(prepared.value, prepared.expected_value)
    prepared.graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(prepared.output, output_before)
    assert torch.equal(prepared.lse, lse_before)
    _assert_non_fp8_bounds(prepared, dtype)
    fp8_record(
        {
            "id": f"non-fp8-lifecycle-{dtype}",
            "plane": "vllm-non-fp8-lifecycle",
            "scale_identity_preserved": True,
            "scale_storage_preserved": True,
            "scale_value": 1.0,
            "cache_rebuilt": True,
            "graph_reused": True,
            "output_hash": helpers._hash(prepared.output),
            "lse_hash": helpers._hash(prepared.lse),
        }
    )
