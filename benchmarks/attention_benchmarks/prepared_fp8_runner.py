# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prepared-cache FP8 attention benchmarking.

The input and scale manifests are supplied by the caller.  This module owns
only regeneration, byte checks, cache preparation, and timing; acceptance
policy belongs to the caller that creates the manifests and YAML config.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from batch_spec import parse_batch_spec
from common import BenchmarkConfig, MockLayer

_TRITON_CUDAGRAPH_RETRIES = 10


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _sha_tensor(tensor: torch.Tensor) -> str:
    data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return _sha_bytes(data)


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


@dataclass(frozen=True)
class PreparedConfig:
    config_path: Path
    base_config_sha256: str
    effective_config_sha256: str
    device: str
    model_path: str
    tokenizer_path: str
    prepared_manifest_path: Path
    prepared_manifest_sha256: str
    scale_manifest_path: Path
    scale_manifest_sha256: str
    arm_order: tuple[str, ...]
    workloads: tuple[str, ...]
    rng_seed: int
    timing_rep_ms: int
    untimed_closure_replays: int
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    block_size: int
    max_model_len: int
    input_scale_multiplier: float
    input_clamp_multiplier: float
    preflight_layer: int


def apply_prepared_cli_overrides(
    yaml_config: dict[str, Any], args: Any
) -> dict[str, Any]:
    """Overlay explicitly supplied prepared-cache policy on a standard config."""
    required = {
        "prepared_manifest": args.prepared_manifest,
        "prepared_manifest_sha256": args.prepared_manifest_sha256,
        "scale_manifest": args.scale_manifest,
        "scale_manifest_sha256": args.scale_manifest_sha256,
        "model_path": args.model_path,
        "rng_seed": args.rng_seed,
        "timing_rep_ms": args.timing_rep_ms,
        "untimed_closure_replays": args.untimed_closure_replays,
        "arm_order": args.arm_order,
        "preflight_layer": args.preflight_layer,
        "input_scale_multiplier": args.input_scale_multiplier,
        "input_clamp_multiplier": args.input_clamp_multiplier,
    }
    missing = [name for name, value in required.items() if value is None]
    _require(not missing, f"missing prepared-FP8 options: {', '.join(missing)}")
    backends = args.backends or ([args.backend] if args.backend else None)
    _require(args.prepared_backends_explicit and backends is not None,
             "prepared-FP8 requires explicit backends")
    _require(args.prepared_batch_specs_explicit and args.batch_specs is not None,
             "prepared-FP8 requires explicit batch specs")

    merged = dict(yaml_config)
    merged.update({
        "mode": "prepared_fp8",
        "batch_specs": list(args.batch_specs),
        "backends": list(backends),
        "device": args.device,
        "prepared_manifest": args.prepared_manifest,
        "prepared_manifest_sha256": args.prepared_manifest_sha256,
        "scale_manifest": args.scale_manifest,
        "scale_manifest_sha256": args.scale_manifest_sha256,
        "model_path": args.model_path,
        "tokenizer_path": args.tokenizer_path or args.model_path,
        "rng_seed": args.rng_seed,
        "timing_rep_ms": args.timing_rep_ms,
        "untimed_closure_replays": args.untimed_closure_replays,
        "arm_order": list(args.arm_order),
        "preflight_layer": args.preflight_layer,
        "dtype": "bfloat16",
        "kv_cache_dtype": "fp8",
        "cuda_graphs": True,
        "input_generation": {
            "distribution": "scaled_clamped_normal",
            "q_scale_source": "k",
            "scale_multiplier": args.input_scale_multiplier,
            "clamp_multiplier": args.input_clamp_multiplier,
        },
    })
    return merged


def validate_prepared_config(
    yaml_config: dict[str, Any], config_path: str
) -> PreparedConfig:
    """Validate a prepared-cache config and its manifests without CUDA."""
    path = Path(config_path).resolve()
    model = yaml_config.get("model", {})
    generation = yaml_config.get("input_generation", {})
    prepared_path = Path(yaml_config["prepared_manifest"]).resolve()
    scale_path = Path(yaml_config["scale_manifest"]).resolve()
    prepared_sha = str(yaml_config["prepared_manifest_sha256"])
    scale_sha = str(yaml_config["scale_manifest_sha256"])
    backends = tuple(yaml_config.get("backends", ()))
    arm_order = tuple(yaml_config.get("arm_order", ()))
    workloads = tuple(yaml_config.get("batch_specs", ()))

    _require(yaml_config.get("mode") == "prepared_fp8",
             "mode must be prepared_fp8")
    _require(yaml_config.get("dtype") == "bfloat16", "dtype must be bfloat16")
    _require(yaml_config.get("kv_cache_dtype") == "fp8", "cache must be fp8")
    _require(yaml_config.get("cuda_graphs") is True, "CUDA graphs must be enabled")
    _require(bool(workloads), "at least one workload is required")
    _require(bool(arm_order), "arm_order must not be empty")
    _require(set(backends) == set(arm_order),
             "every configured backend must occur in arm_order")
    _require(all(name in {"FLASH_ATTN_FA3", "FLASH_ATTN_FA4"}
                 for name in backends), "prepared FP8 supports FA3/FA4 aliases")
    _require(int(yaml_config.get("timing_rep_ms", 0)) > 0,
             "timing_rep_ms must be positive")
    _require(int(yaml_config.get("untimed_closure_replays", -1)) >= 0,
             "untimed_closure_replays must be nonnegative")
    _require(generation.get("distribution") == "scaled_clamped_normal",
             "unsupported input distribution")
    _require(generation.get("q_scale_source") == "k",
             "only q_scale_source=k is supported")
    for key in ("num_layers", "num_q_heads", "num_kv_heads", "head_dim",
                "block_size", "max_model_len"):
        _require(int(model.get(key, 0)) > 0, f"model.{key} must be positive")
    preflight_layer = int(yaml_config.get("preflight_layer", -1))
    _require(0 <= preflight_layer < int(model["num_layers"]),
             "preflight_layer is outside the configured model")
    _require(prepared_path.is_file(), f"missing prepared manifest: {prepared_path}")
    _require(scale_path.is_file(), f"missing scale manifest: {scale_path}")
    _require(_sha_file(prepared_path) == prepared_sha,
             "prepared-manifest byte hash mismatch")
    _require(_sha_file(scale_path) == scale_sha,
             "scale-manifest byte hash mismatch")

    prepared = json.loads(prepared_path.read_text())
    scales = json.loads(scale_path.read_text())
    constants = prepared["constants"]
    _require(
        (
            constants["layers"], constants["q_heads"], constants["kv_heads"],
            constants["head_dim"], constants["page_size"],
        ) == (
            model["num_layers"], model["num_q_heads"], model["num_kv_heads"],
            model["head_dim"], model["block_size"],
        ),
        "prepared-manifest geometry does not match config",
    )
    _require([row["workload"] for row in prepared["workloads"]]
             == list(workloads), "prepared workloads/order do not match config")
    _require(len(scales["layers"]) == int(model["num_layers"]),
             "scale-manifest layer count does not match config")

    return PreparedConfig(
        config_path=path,
        base_config_sha256=_sha_file(path),
        effective_config_sha256=_sha_bytes(
            json.dumps(yaml_config, sort_keys=True, separators=(",", ":")).encode()
        ),
        device=str(yaml_config.get("device", "cuda:0")),
        model_path=str(yaml_config["model_path"]),
        tokenizer_path=str(
            yaml_config.get("tokenizer_path", yaml_config["model_path"])
        ),
        prepared_manifest_path=prepared_path,
        prepared_manifest_sha256=prepared_sha,
        scale_manifest_path=scale_path,
        scale_manifest_sha256=scale_sha,
        arm_order=arm_order,
        workloads=workloads,
        rng_seed=int(yaml_config["rng_seed"]),
        timing_rep_ms=int(yaml_config["timing_rep_ms"]),
        untimed_closure_replays=int(yaml_config["untimed_closure_replays"]),
        num_layers=int(model["num_layers"]),
        num_q_heads=int(model["num_q_heads"]),
        num_kv_heads=int(model["num_kv_heads"]),
        head_dim=int(model["head_dim"]),
        block_size=int(model["block_size"]),
        max_model_len=int(model["max_model_len"]),
        input_scale_multiplier=float(generation["scale_multiplier"]),
        input_clamp_multiplier=float(generation["clamp_multiplier"]),
        preflight_layer=preflight_layer,
    )


def _static_bf16(
    count: int,
    heads: int,
    scale: float,
    generator: torch.Generator,
    config: PreparedConfig,
) -> torch.Tensor:
    return (
        torch.randn(
            (count, heads, config.head_dim),
            generator=generator,
            dtype=torch.float32,
        )
        * (config.input_scale_multiplier * scale)
    ).clamp(
        -config.input_clamp_multiplier * scale,
        config.input_clamp_multiplier * scale,
    ).to(torch.bfloat16)


def _workload_layout(
    workload: str, rows: list[list[int]], block_size: int
) -> tuple[list[int], list[int], torch.Tensor, int]:
    requests = parse_batch_spec(workload)
    q_lens = [request.q_len for request in requests]
    kv_lens = [request.kv_len for request in requests]
    _require(len(rows) == len(requests),
             f"{workload}: block-table request count mismatch")
    _require(len({len(row) for row in rows}) == 1,
             f"{workload}: block table must be rectangular")
    slots = torch.tensor(
        [
            row[token // block_size] * block_size + token % block_size
            for row, kv_len in zip(rows, kv_lens, strict=True)
            for token in range(kv_len)
        ],
        dtype=torch.long,
    )
    pages = max(page for row in rows for page in row) + 1
    return q_lens, kv_lens, slots, pages


def _logical_cache(
    cache: torch.Tensor,
    rows: list[list[int]],
    kv_lens: list[int],
    block_size: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_cache, value_cache = cache.transpose(1, 2).split(head_dim, dim=-1)
    indices = torch.tensor(
        [
            row[token // block_size] * block_size + token % block_size
            for row, kv_len in zip(rows, kv_lens, strict=True)
            for token in range(kv_len)
        ],
        dtype=torch.long,
        device=cache.device,
    )
    return (
        key_cache[indices // block_size, indices % block_size],
        value_cache[indices // block_size, indices % block_size],
    )


@dataclass
class PreparedWorkload:
    name: str
    rows: list[list[int]]
    q_lens: list[int]
    kv_lens: list[int]
    q_list: list[torch.Tensor]
    cache_list: list[torch.Tensor]
    scales: list[tuple[float, float]]
    manifest_row: dict[str, Any]


def prepare_workload(config: PreparedConfig, workload: str) -> PreparedWorkload:
    """Regenerate and byte-verify one workload before constructing any arm."""
    from vllm._custom_ops import reshape_and_cache_flash
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    from vllm.platforms import current_platform

    device = torch.device(config.device)
    manifest = json.loads(config.prepared_manifest_path.read_text())
    manifest_row = next(
        row for row in manifest["workloads"] if row["workload"] == workload
    )
    scale_rows = json.loads(config.scale_manifest_path.read_text())["layers"]
    rows = manifest_row["block_table"]
    q_lens, kv_lens, slots_cpu, pages = _workload_layout(
        workload, rows, config.block_size
    )
    _require(
        _sha_bytes(json.dumps(rows, separators=(",", ":")).encode())
        == manifest_row["block_table_sha256"],
        "block-table hash mismatch",
    )
    _require(_sha_tensor(slots_cpu) == manifest_row["slot_mapping_sha256"],
             "slot-mapping hash mismatch")
    _require(slots_cpu.numel() == manifest_row["expected_slot_count"],
             "slot-mapping count mismatch")

    generator = torch.Generator().manual_seed(config.rng_seed)
    slots = slots_cpu.to(device)
    q_list: list[torch.Tensor] = []
    cache_list: list[torch.Tensor] = []
    scales: list[tuple[float, float]] = []
    for layer_index, (scale_row, expected) in enumerate(
        zip(scale_rows, manifest_row["layers"], strict=True)
    ):
        _require(scale_row["layer"] == layer_index == expected["layer"],
                 "non-contiguous layer manifest")
        k_scale = float(scale_row["k_scale"])
        v_scale = float(scale_row["v_scale"])
        q_cpu = _static_bf16(sum(q_lens), config.num_q_heads, k_scale,
                             generator, config)
        k_cpu = _static_bf16(sum(kv_lens), config.num_kv_heads, k_scale,
                             generator, config)
        v_cpu = _static_bf16(sum(kv_lens), config.num_kv_heads, v_scale,
                             generator, config)
        for label, tensor in (("q", q_cpu), ("k", k_cpu), ("v", v_cpu)):
            _require(_sha_tensor(tensor) == expected[f"{label}_bf16_sha256"],
                     f"layer {layer_index} {label.upper()} input hash mismatch")

        with set_current_vllm_config(VllmConfig()):
            q_fp8, q_scale = QuantFP8(
                static=True, group_shape=GroupShape.PER_TENSOR
            )(q_cpu.to(device), torch.tensor(k_scale, device=device))
        _require(q_scale.numel() == 1, f"layer {layer_index} Q scale is not scalar")
        _require(_sha_tensor(q_fp8) == expected["actual_q_sha256"],
                 f"layer {layer_index} quantized-Q hash mismatch")

        cache = torch.zeros(
            pages,
            config.num_kv_heads,
            config.block_size,
            2 * config.head_dim,
            dtype=current_platform.fp8_dtype(),
            device=device,
        )
        key_cache, value_cache = cache.transpose(1, 2).split(config.head_dim, dim=-1)
        reshape_and_cache_flash(
            k_cpu.to(device),
            v_cpu.to(device),
            key_cache,
            value_cache,
            slots,
            "fp8",
            torch.tensor(k_scale, device=device),
            torch.tensor(v_scale, device=device),
        )
        actual_k, actual_v = _logical_cache(
            cache, rows, kv_lens, config.block_size, config.head_dim
        )
        _require(
            _sha_tensor(actual_k) == expected["actual_k_sha256"]
            == expected["expected_k_sha256"],
            f"layer {layer_index} writer K bytes mismatch",
        )
        _require(
            _sha_tensor(actual_v) == expected["actual_v_sha256"]
            == expected["expected_v_sha256"],
            f"layer {layer_index} writer V bytes mismatch",
        )
        _require(_sha_tensor(cache) == expected["full_zero_initialized_cache_sha256"],
                 f"layer {layer_index} full-cache hash mismatch")
        q_list.append(q_fp8)
        cache_list.append(cache)
        scales.append((k_scale, v_scale))

    return PreparedWorkload(
        workload, rows, q_lens, kv_lens, q_list, cache_list, scales, manifest_row
    )


@dataclass
class BackendState:
    backend: str
    version: int
    vllm_config: Any
    impl: Any
    layers: list[MockLayer]
    metadata: Any


def _benchmark_config(
    config: PreparedConfig, backend: str, workload: str
) -> BenchmarkConfig:
    return BenchmarkConfig(
        backend=backend,
        batch_spec=workload,
        num_layers=config.num_layers,
        head_dim=config.head_dim,
        num_q_heads=config.num_q_heads,
        num_kv_heads=config.num_kv_heads,
        block_size=config.block_size,
        device=config.device,
        max_model_len=config.max_model_len,
        dtype=torch.bfloat16,
        kv_cache_dtype="fp8",
        use_cuda_graphs=True,
    )


def initialize_backend(
    config: PreparedConfig, prepared: PreparedWorkload, backend: str
) -> BackendState:
    from runner import (
        _build_common_attn_metadata,
        _create_backend_impl,
        _create_metadata_builder,
        _create_vllm_config,
        _get_backend_config,
    )
    from vllm.config import set_current_vllm_config
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    version = {"FLASH_ATTN_FA3": 3, "FLASH_ATTN_FA4": 4}[backend]
    bench_config = _benchmark_config(config, backend, prepared.name)
    max_num_blocks = max(page for row in prepared.rows for page in row) + 1
    vllm_config = _create_vllm_config(
        bench_config,
        max_num_blocks,
        model_path=config.model_path,
        tokenizer_path=config.tokenizer_path,
        model_max_model_len=config.max_model_len,
    )
    vllm_config.attention_config.flash_attn_version = version
    device = torch.device(config.device)
    with set_current_vllm_config(vllm_config):
        backend_class, impl, template_layer = _create_backend_impl(
            _get_backend_config("FLASH_ATTN"), bench_config, device, torch.bfloat16
        )
        actual_version = getattr(impl, "vllm_flash_attn_version", None)
        _require(actual_version == version,
                 f"{backend} resolved FA{actual_version}, expected FA{version}")
        _require(getattr(impl, "supports_quant_query_input", False),
                 f"{backend} did not enable FP8 query input")
        common = _build_common_attn_metadata(
            prepared.q_lens, prepared.kv_lens, config.block_size, device
        )
        common.block_table_tensor = torch.tensor(
            prepared.rows, dtype=torch.int32, device=device
        )
        cache_spec = FullAttentionSpec(
            block_size=config.block_size,
            num_kv_heads=config.num_kv_heads,
            head_size=config.head_dim,
            dtype=torch.bfloat16,
        )
        builder = _create_metadata_builder(
            backend_class, cache_spec, vllm_config, device, backend
        )
        metadata = builder.build(0, common)
        layers = []
        for k_scale, v_scale in prepared.scales:
            layer = MockLayer(device, kv_cache_spec=template_layer.get_kv_cache_spec())
            layer._q_scale = torch.tensor(k_scale, dtype=torch.float32, device=device)
            layer._k_scale = torch.tensor(k_scale, dtype=torch.float32, device=device)
            layer._v_scale = torch.tensor(v_scale, dtype=torch.float32, device=device)
            layer._q_scale_float = k_scale
            layer._k_scale_float = k_scale
            layer._v_scale_float = v_scale
            layers.append(layer)
    return BackendState(backend, version, vllm_config, impl, layers, metadata)


def clone_prepared_inputs(
    prepared: PreparedWorkload,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    q_list = [q.clone() for q in prepared.q_list]
    caches = [cache.clone() for cache in prepared.cache_list]
    for layer_index, (q, source_q, cache, source_cache) in enumerate(
        zip(q_list, prepared.q_list, caches, prepared.cache_list, strict=True)
    ):
        _require(torch.equal(q, source_q), f"Q clone mismatch at layer {layer_index}")
        _require(torch.equal(cache, source_cache),
                 f"cache clone mismatch at layer {layer_index}")
    return q_list, caches


def assert_prepared_unchanged(
    prepared: PreparedWorkload,
    q_list: list[torch.Tensor],
    caches: list[torch.Tensor],
) -> None:
    """Fail if a supposedly read-only arm mutated its prepared tensors."""
    for layer_index, (q, source_q, cache, source_cache) in enumerate(
        zip(q_list, prepared.q_list, caches, prepared.cache_list, strict=True)
    ):
        _require(torch.equal(q, source_q), f"Q changed at layer {layer_index}")
        _require(torch.equal(cache, source_cache),
                 f"cache changed at layer {layer_index}")


def run_preflight(
    state: BackendState, q: torch.Tensor, cache: torch.Tensor, layer_index: int
) -> dict[str, Any]:
    from vllm.v1.attention.backends.flash_attn import flash_attn_varlen_func

    metadata = state.metadata
    layer = state.layers[layer_index]
    key_cache, value_cache = cache.transpose(1, 2).split(q.shape[-1], dim=-1)
    descale_shape = (metadata.query_start_loc.shape[0] - 1, key_cache.shape[-2])
    output, lse = flash_attn_varlen_func(
        q=q,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=metadata.query_start_loc,
        seqused_k=metadata.seq_lens,
        max_seqlen_q=metadata.max_query_len,
        max_seqlen_k=metadata.max_seq_len,
        softmax_scale=state.impl.scale,
        causal=metadata.causal,
        window_size=(list(state.impl.sliding_window)
                     if state.impl.sliding_window is not None else None),
        block_table=metadata.block_table,
        softcap=state.impl.logits_soft_cap,
        return_softmax_lse=True,
        scheduler_metadata=metadata.scheduler_metadata,
        fa_version=state.version,
        q_descale=layer._q_scale.expand(descale_shape),
        k_descale=layer._k_scale.expand(descale_shape),
        v_descale=layer._v_scale.expand(descale_shape),
        num_splits=metadata.max_num_splits,
        s_aux=state.impl.sinks,
    )
    torch.accelerator.synchronize()
    _require(
        torch.isfinite(output.float()).all().item(),
        "preflight output is non-finite",
    )
    _require(torch.isfinite(lse.float()).all().item(), "preflight LSE is non-finite")
    return {
        "layer": layer_index,
        "output_sha256": _sha_tensor(output),
        "lse_sha256": _sha_tensor(lse),
    }


def make_closure(
    state: BackendState,
    q_list: list[torch.Tensor],
    cache_list: list[torch.Tensor],
):
    output = torch.empty_like(q_list[0], dtype=torch.bfloat16)
    unused_kv = torch.empty(
        0,
        state.impl.num_kv_heads,
        q_list[0].shape[-1],
        dtype=torch.bfloat16,
        device=q_list[0].device,
    )

    def closure() -> None:
        for layer, query, cache in zip(
            state.layers, q_list, cache_list, strict=True
        ):
            state.impl.forward(
                layer, query, unused_kv, unused_kv, cache, state.metadata,
                output=output,
            )

    return closure


def run_untimed_closures(closure, count: int) -> None:
    for _ in range(count):
        closure()
        torch.accelerator.synchronize()


def _git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance(config: PreparedConfig) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    try:
        smi = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,pstate,clocks.sm,"
                "clocks.mem,power.draw,power.limit",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip().splitlines()
    except (OSError, subprocess.CalledProcessError):
        smi = []
    return {
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "CUTE_DSL_LINEINFO",
                "VLLM_FLASH_ATTN_VERSION", "VLLM_BATCH_INVARIANT",
            )
        },
        "config_path": str(config.config_path),
        "base_config_sha256": config.base_config_sha256,
        "effective_config_sha256": config.effective_config_sha256,
        "prepared_manifest_path": str(config.prepared_manifest_path),
        "prepared_manifest_sha256": config.prepared_manifest_sha256,
        "scale_manifest_path": str(config.scale_manifest_path),
        "scale_manifest_sha256": config.scale_manifest_sha256,
        "runner_sha256": _sha_file(Path(__file__)),
        "entrypoint_path": str(Path(sys.argv[0]).resolve()),
        "entrypoint_sha256": (
            _sha_file(Path(sys.argv[0]).resolve())
            if Path(sys.argv[0]).is_file()
            else None
        ),
        "vllm_commit": _git_head(root),
        "flash_attention_commit": _git_head(root.parent / "flash-attention"),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu_snapshot": smi,
    }


def run_campaign(config: PreparedConfig, output_json: str) -> None:
    from vllm.config import set_current_vllm_config
    from vllm.triton_utils import triton

    output_path = Path(output_json)
    _require(not output_path.exists(), f"refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.accelerator.set_device_index(torch.device(config.device))
    result: dict[str, Any] = {
        "schema": "prepared-fp8-performance-v1",
        "started_utc": datetime.now(UTC).isoformat(),
        "provenance": provenance(config),
        "workloads": [],
    }
    for workload in config.workloads:
        prepared = prepare_workload(config, workload)
        states = {
            backend: initialize_backend(config, prepared, backend)
            for backend in dict.fromkeys(config.arm_order)
        }
        workload_result = {"workload": workload, "arms": []}
        for arm_index, backend in enumerate(config.arm_order):
            arm_started_utc = datetime.now(UTC).isoformat()
            state = states[backend]
            q_list, caches = clone_prepared_inputs(prepared)
            with set_current_vllm_config(state.vllm_config):
                preflight = run_preflight(
                    state,
                    q_list[config.preflight_layer],
                    caches[config.preflight_layer],
                    config.preflight_layer,
                )
                closure = make_closure(state, q_list, caches)
                run_untimed_closures(closure, config.untimed_closure_replays)
                samples_ms = triton.testing.do_bench_cudagraph(
                    closure, rep=config.timing_rep_ms, return_mode="all"
                )
                torch.accelerator.synchronize()
            assert_prepared_unchanged(prepared, q_list, caches)
            samples_ms = [float(sample) for sample in samples_ms]
            _require(len(samples_ms) == _TRITON_CUDAGRAPH_RETRIES,
                     "timing helper returned an unexpected sample count")
            per_layer_us = [
                sample * 1000.0 / config.num_layers for sample in samples_ms
            ]
            workload_result["arms"].append({
                "arm_index": arm_index,
                "backend": backend,
                "started_utc": arm_started_utc,
                "finished_utc": datetime.now(UTC).isoformat(),
                "preflight": preflight,
                "untimed_closure_replays": config.untimed_closure_replays,
                "closure_forward_count": config.num_layers,
                "timing_rep_ms": config.timing_rep_ms,
                "raw_closure_ms": samples_ms,
                "raw_per_layer_us": per_layer_us,
                "median_per_layer_us": statistics.median(per_layer_us),
            })
        result["workloads"].append(workload_result)
    result["finished_utc"] = datetime.now(UTC).isoformat()
    with output_path.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
