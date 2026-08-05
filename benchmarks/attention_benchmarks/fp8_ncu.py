#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Emit one NVTX-scoped closure for a prepared-cache FP8 NCU capture."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import torch
import yaml

from prepared_fp8_runner import (
    assert_prepared_unchanged,
    clone_prepared_inputs,
    initialize_backend,
    make_closure,
    prepare_workload,
    provenance,
    run_preflight,
    run_untimed_closures,
    validate_prepared_config,
)
from vllm.config import set_current_vllm_config
from vllm.v1.worker.workspace import init_workspace_manager


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--backend", required=True, choices=("FLASH_ATTN_FA3", "FLASH_ATTN_FA4")
    )
    parser.add_argument("--workload", required=True)
    parser.add_argument("--batch-specs", nargs="+", required=True)
    parser.add_argument("--arm-order", nargs="+", required=True)
    parser.add_argument("--order-label", required=True)
    parser.add_argument("--nvtx-label", default="fp8_attention_closure")
    parser.add_argument("--one-closure", action="store_true", required=True)
    parser.add_argument("--validate-config-only", action="store_true")
    parser.add_argument("--prepared-manifest", required=True)
    parser.add_argument("--prepared-manifest-sha256", required=True)
    parser.add_argument("--scale-manifest", required=True)
    parser.add_argument("--scale-manifest-sha256", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--rng-seed", type=int, required=True)
    parser.add_argument("--timing-rep-ms", type=int, required=True)
    parser.add_argument("--untimed-closure-replays", type=int, required=True)
    parser.add_argument("--preflight-layer", type=int, required=True)
    parser.add_argument("--input-scale-multiplier", type=float, required=True)
    parser.add_argument("--input-clamp-multiplier", type=float, required=True)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    yaml_config = yaml.safe_load(config_path.read_text())
    yaml_config.update({
        "mode": "prepared_fp8",
        "batch_specs": args.batch_specs,
        "backends": list(dict.fromkeys(args.arm_order)),
        "arm_order": args.arm_order,
        "dtype": "bfloat16",
        "kv_cache_dtype": "fp8",
        "cuda_graphs": True,
        "prepared_manifest": args.prepared_manifest,
        "prepared_manifest_sha256": args.prepared_manifest_sha256,
        "scale_manifest": args.scale_manifest,
        "scale_manifest_sha256": args.scale_manifest_sha256,
        "model_path": args.model_path,
        "tokenizer_path": args.tokenizer_path or args.model_path,
        "rng_seed": args.rng_seed,
        "timing_rep_ms": args.timing_rep_ms,
        "untimed_closure_replays": args.untimed_closure_replays,
        "preflight_layer": args.preflight_layer,
        "input_generation": {
            "distribution": "scaled_clamped_normal",
            "q_scale_source": "k",
            "scale_multiplier": args.input_scale_multiplier,
            "clamp_multiplier": args.input_clamp_multiplier,
        },
    })
    config = validate_prepared_config(yaml_config, str(config_path))
    if args.backend not in config.arm_order:
        parser.error("backend is absent from the configured arm order")
    if args.workload not in config.workloads:
        parser.error("workload is absent from the prepared manifest")
    if args.validate_config_only:
        print("Prepared-FP8 NCU config and manifests are valid.")
        return

    init_workspace_manager(config.device)
    torch.accelerator.set_device_index(torch.device(config.device))
    prepared = prepare_workload(config, args.workload)
    state = initialize_backend(config, prepared, args.backend)
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
        torch.cuda.nvtx.range_push(args.nvtx_label)
        try:
            closure()
        finally:
            torch.cuda.nvtx.range_pop()
        torch.accelerator.synchronize()
    assert_prepared_unchanged(prepared, q_list, caches)

    print(json.dumps({
        "schema": "prepared-fp8-ncu-driver-v1",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "backend": args.backend,
        "workload": args.workload,
        "order_label": args.order_label,
        "nvtx_label": args.nvtx_label,
        "untimed_closure_replays": config.untimed_closure_replays,
        "profiled_closures": 1,
        "closure_forward_count": config.num_layers,
        "preflight": preflight,
        "provenance": provenance(config),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
