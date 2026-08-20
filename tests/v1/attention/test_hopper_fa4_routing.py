# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import AttentionConfig, replace
from vllm.model_executor.models.config import Gemma4Config
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends import fa_utils
from vllm.v1.attention.backends import flash_attn
from vllm.v1.attention.backends import flash_attn_diffkv
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.vllm_flash_attn import flash_attn_interface as hopper_interface
from vllm.vllm_flash_attn.cute.split_scheduler import plan_hopper_split_schedule


def _config(
    *,
    version: int | None = 4,
    kv_cache_dtype: str = "auto",
    backend: AttentionBackendEnum | None = None,
    sparse_mla: bool = False,
    dtype: torch.dtype = torch.bfloat16,
):
    hf_text_config = SimpleNamespace()
    if sparse_mla:
        hf_text_config.index_topk = 2048
    return SimpleNamespace(
        attention_config=AttentionConfig(
            flash_attn_version=version,
            backend=backend,
        ),
        cache_config=SimpleNamespace(cache_dtype=kv_cache_dtype),
        model_config=SimpleNamespace(
            use_mla=sparse_mla,
            is_diffusion=False,
            architecture="DeepseekV3ForCausalLM",
            hf_text_config=hf_text_config,
            get_head_size=lambda: 128,
            dtype=dtype,
        ),
    )


@pytest.fixture
def hopper(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        fa_utils.current_platform,
        "get_device_capability",
        lambda: DeviceCapability(major=9, minor=0),
    )
    monkeypatch.setattr(fa_utils.envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(fa_utils, "has_cutedsl", lambda: True)
    monkeypatch.setattr(
        fa_utils, "is_fa_version_supported", lambda version: version in (3, 4)
    )
    fake_interface = SimpleNamespace(fa4_cutedsl_import_error=lambda: None)
    monkeypatch.setattr(
        fa_utils,
        "import_module",
        lambda name: fake_interface,
    )
    return fake_interface


def _patch_cute_compile(monkeypatch, compile_specs):
    fake_interface = types.ModuleType("vllm.vllm_flash_attn.cute.interface")
    fake_interface.compile_flash_attn_varlen_func_from_specs = compile_specs
    monkeypatch.setitem(sys.modules, fake_interface.__name__, fake_interface)


def test_compile_only_wrapper_forwards_fake_tensor_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compile_specs = MagicMock()
    _patch_cute_compile(monkeypatch, compile_specs)

    hopper_interface.compile_flash_attn_varlen_func_from_specs(
        q_shape=(9, 16, 128),
        k_shape=(512, 16, 2, 128),
        v_shape=(512, 16, 2, 128),
        q_dtype=torch.float8_e4m3fn,
        out_dtype=torch.bfloat16,
        v_stride=(8192, 512, 256, 1),
        cu_seqlens_q_shape=(2,),
        seqused_q_shape=(1,),
        seqused_q_stride=(1,),
        seqused_k_shape=(1,),
        seqused_k_stride=(1,),
        page_table_shape=(1, 512),
        page_table_stride=(512, 1),
        num_splits_dynamic_ptr_shape=(1,),
        num_splits_dynamic_ptr_stride=(1,),
        s_aux_shape=(16,),
        s_aux_stride=(1,),
        dynamic_scheduler_counter_shape=(1,),
        dynamic_scheduler_counter_stride=(1,),
        q_descale_shape=(),
        q_descale_stride=(),
        k_descale_shape=(),
        k_descale_stride=(),
        v_descale_shape=(),
        v_descale_stride=(),
        max_seqlen_q=9,
        max_seqlen_k=8192,
        softmax_scale=0.125,
        causal=True,
        window_size=[127, 0],
        return_softmax_lse=True,
        num_splits=17,
        fa_version=4,
    )

    forwarded = compile_specs.call_args.kwargs
    expected_passthrough = {
        "q_shape": (9, 16, 128),
        "k_shape": (512, 16, 2, 128),
        "v_shape": (512, 16, 2, 128),
        "q_dtype": torch.float8_e4m3fn,
        "out_dtype": torch.bfloat16,
        "v_stride": (8192, 512, 256, 1),
        "cu_seqlens_q_shape": (2,),
        "seqused_q_shape": (1,),
        "seqused_q_stride": (1,),
        "seqused_k_shape": (1,),
        "seqused_k_stride": (1,),
        "page_table_shape": (1, 512),
        "page_table_stride": (512, 1),
        "num_splits_dynamic_ptr_shape": (1,),
        "num_splits_dynamic_ptr_stride": (1,),
        "q_descale_shape": (),
        "q_descale_stride": (),
        "k_descale_shape": (),
        "k_descale_stride": (),
        "v_descale_shape": (),
        "v_descale_stride": (),
        "max_seqlen_q": 9,
        "max_seqlen_k": 8192,
        "softmax_scale": 0.125,
        "causal": True,
        "num_splits": 17,
    }
    assert {
        key: forwarded[key] for key in expected_passthrough
    } == expected_passthrough
    assert forwarded["cu_seqlens_k_shape"] is None
    assert forwarded["learnable_sink_shape"] == (16,)
    assert forwarded["learnable_sink_stride"] == (1,)
    assert forwarded["dynamic_scheduler_counter_shape"] == (1,)
    assert forwarded["dynamic_scheduler_counter_stride"] == (1,)
    assert forwarded["window_size"] == (127, 0)
    assert forwarded["return_lse"] is True


@pytest.mark.parametrize(
    ("window_size", "expected"),
    [
        pytest.param(None, (None, None), id="none"),
        pytest.param([-1, -1], (None, None), id="negative"),
        pytest.param([-1, 0], (None, 0), id="left-unbounded"),
        pytest.param([127, 0], (127, 0), id="bounded"),
    ],
)
def test_compile_only_wrapper_normalizes_window_endpoints(
    monkeypatch: pytest.MonkeyPatch,
    window_size: list[int] | None,
    expected: tuple[int | None, int | None],
) -> None:
    compile_specs = MagicMock()
    _patch_cute_compile(monkeypatch, compile_specs)

    hopper_interface.compile_flash_attn_varlen_func_from_specs(
        q_shape=(1, 8, 128),
        k_shape=(1, 16, 1, 128),
        v_shape=(1, 16, 1, 128),
        q_dtype=torch.float8_e4m3fn,
        out_dtype=torch.bfloat16,
        window_size=window_size,
        fa_version=4,
    )

    assert compile_specs.call_args.kwargs["window_size"] == expected


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"fa_version": 3}, ValueError, "only supported for FA4"),
        (
            {"fa_version": 4, "out_dtype": torch.float32},
            ValueError,
            "out_dtype",
        ),
        (
            {"fa_version": 4, "dropout_p": 0.1},
            NotImplementedError,
            "dropout",
        ),
        (
            {"fa_version": 4, "s_aux_shape": (1, 1)},
            AssertionError,
            "learnable_sink must be rank 1",
        ),
        (
            {"fa_version": 4, "dynamic_scheduler_counter_shape": (2,)},
            AssertionError,
            "dynamic_scheduler_counter must have shape",
        ),
    ],
)
def test_compile_only_wrapper_preserves_validation(kwargs, error, match) -> None:
    with pytest.raises(error, match=match):
        hopper_interface.compile_flash_attn_varlen_func_from_specs(
            q_shape=(1, 8, 128),
            k_shape=(1, 16, 1, 128),
            v_shape=(1, 16, 1, 128),
            q_dtype=torch.float8_e4m3fn,
            **kwargs,
        )


@pytest.mark.parametrize(
    (
        "fa_version",
        "cache_dtype",
        "major",
        "head_size",
        "head_size_v",
        "out_dtype",
        "expected",
    ),
    [
        pytest.param(4, "fp8_e4m3", 9, 128, None, None, "native", id="native"),
        pytest.param(
            4, "fp8_e4m3", 9, 64, 64, None, "native", id="native-64"
        ),
        pytest.param(
            4, "fp8_e4m3", 9, 96, 96, None, "native", id="native-96"
        ),
        pytest.param(
            4, "fp8_e4m3", 9, 192, 192, None, "native", id="native-192"
        ),
        pytest.param(
            4, "fp8_e4m3", 9, 256, 256, None, "native", id="native-256"
        ),
        pytest.param(
            4, "fp8_e4m3", 9, 192, 128, None, "native", id="native-diffkv"
        ),
        pytest.param(
            4, "fp8_e4m3", 9, 128, 192, None, None, id="reversed-diffkv"
        ),
        pytest.param(
            4, "fp8_e4m3", 9, 192, 256, None, None, id="unsupported-diffkv"
        ),
        pytest.param(3, "fp8", 9, 128, None, None, None, id="fa3"),
        pytest.param(4, "auto", 9, 128, None, None, None, id="non-fp8-cache"),
        pytest.param(4, "fp8_e5m2", 9, 128, None, None, None, id="e5m2"),
        pytest.param(4, "fp8", 8, 128, None, None, None, id="pre-hopper"),
        pytest.param(4, "fp8", 10, 128, None, None, None, id="blackwell"),
        pytest.param(
            4, "fp8", 9, 128, None, torch.bfloat16, "native", id="bf16-output"
        ),
        pytest.param(
            4, "fp8", 9, 128, None, torch.float32, None, id="fp32-output"
        ),
    ],
)
def test_get_sm90_fa4_fp8_attention_mode_cells(
    monkeypatch: pytest.MonkeyPatch,
    fa_version: int,
    cache_dtype: str,
    major: int,
    head_size: int,
    head_size_v: int | None,
    out_dtype: torch.dtype | None,
    expected: str | None,
) -> None:
    monkeypatch.setattr(
        fa_utils.current_platform,
        "get_device_capability",
        lambda: DeviceCapability(major=major, minor=0),
    )
    assert fa_utils.get_sm90_fa4_fp8_attention_mode(
        fa_version=fa_version,
        kv_cache_dtype=cache_dtype,
        head_size=head_size,
        head_size_v=head_size_v,
        out_dtype=out_dtype,
    ) == expected


@pytest.mark.parametrize(
    ("head_size", "spec_head_size_v"),
    [
        pytest.param(128, None, id="dense-128-128"),
        pytest.param(192, 128, id="diffkv-192-128"),
    ],
)
def test_metadata_builder_forwards_authoritative_v_head_geometry(
    monkeypatch: pytest.MonkeyPatch,
    head_size: int,
    spec_head_size_v: int | None,
) -> None:
    model_config = SimpleNamespace(
        get_num_attention_heads=lambda _: 8,
        get_num_kv_heads=lambda _: 2,
        get_head_size=lambda: head_size,
        dtype=torch.bfloat16,
        rswa_window=None,
        is_mm_prefix_lm=False,
        uses_alibi=False,
    )
    config = SimpleNamespace(
        model_config=model_config,
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
        cache_config=SimpleNamespace(cache_dtype="fp8"),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False),
            max_cudagraph_capture_size=None,
        ),
        attention_config=SimpleNamespace(
            flash_attn_max_num_splits_for_cuda_graph=32
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=8,
            max_num_batched_tokens=64,
        ),
    )
    spec_kwargs = {
        "dtype": torch.float8_e4m3fn,
        "block_size": 16,
        "head_size": head_size,
        "sliding_window": None,
    }
    if spec_head_size_v is not None:
        spec_kwargs["head_size_v"] = spec_head_size_v
    kv_cache_spec = SimpleNamespace(**spec_kwargs)
    expected_head_size_v = (
        head_size if spec_head_size_v is None else spec_head_size_v
    )

    version_selector = MagicMock(return_value=4)
    planner = MagicMock(
        return_value=SimpleNamespace(num_splits=3, scheduler_metadata=None)
    )
    planner_factory = MagicMock(return_value=planner)
    monkeypatch.setattr(flash_attn, "get_flash_attn_version", version_selector)
    monkeypatch.setattr(
        flash_attn.current_platform, "is_device_capability", lambda _: True
    )
    monkeypatch.setattr(
        flash_attn.current_platform,
        "fp8_dtype",
        lambda: torch.float8_e4m3fn,
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_dcp_group",
        MagicMock(side_effect=AssertionError),
    )
    monkeypatch.setattr(
        "vllm.vllm_flash_attn.cute.split_scheduler.SplitSchedulerPlanner",
        planner_factory,
    )

    builder = flash_attn.FlashAttentionMetadataBuilder(
        kv_cache_spec,
        ["layer.0"],
        config,
        torch.device("cpu"),
    )

    assert builder.headdim_v == expected_head_size_v
    version_selector.assert_called_once_with(
        requires_alibi=False,
        requires_local_attention=False,
        head_size=head_size,
        head_size_v=expected_head_size_v,
    )

    scheduler = MagicMock(return_value=torch.tensor([1], dtype=torch.int32))
    monkeypatch.setattr(flash_attn, "get_scheduler_metadata", scheduler)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
    seq_lens = torch.tensor([16], dtype=torch.int32)
    builder.aot_sliding_window = (-1, -1)
    builder._get_scheduler_metadata(
        aot_schedule=True,
        batch_size=1,
        cu_query_lens=query_start_loc,
        max_query_len=1,
        seqlens=seq_lens,
        max_seq_len=16,
        causal=True,
        max_num_splits=7,
    )
    scheduler_kwargs = scheduler.call_args.kwargs
    assert scheduler_kwargs.pop("cache_seqlens") is seq_lens
    assert scheduler_kwargs.pop("cu_seqlens_q") is query_start_loc
    assert scheduler_kwargs == {
        "batch_size": 1,
        "max_seqlen_q": 1,
        "max_seqlen_k": 16,
        "num_heads_q": 8,
        "num_heads_kv": 2,
        "headdim": head_size,
        "headdim_v": expected_head_size_v,
        "qkv_dtype": torch.float8_e4m3fn,
        "page_size": 16,
        "causal": True,
        "window_size": (-1, -1),
        "num_splits": 7,
    }

    builder.aot_schedule = False
    builder._get_scheduler_metadata = MagicMock(return_value=None)
    common_metadata = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=1,
        max_query_len=1,
        max_seq_len=16,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        block_table_tensor=torch.zeros((1, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(1, dtype=torch.int64),
        causal=True,
        query_start_loc_cpu=query_start_loc,
        seq_lens_cpu_upper_bound=seq_lens,
        mm_req_doc_ranges=None,
        rswa_prefix_lens=None,
    )
    builder.build(common_prefix_len=0, common_attn_metadata=common_metadata)

    planner_args, planner_kwargs = planner.call_args
    assert torch.equal(planner_args[0], query_start_loc)
    assert torch.equal(planner_args[1], seq_lens)
    assert planner_kwargs == {
        "num_heads_q": 8,
        "num_heads_kv": 2,
        "head_dim": head_size,
        "head_dim_v": expected_head_size_v,
        "has_qv": False,
        "cp_world_size": 1,
        "window_size": (-1, -1),
        "cuda_graph_max_num_splits": None,
        "fast_build": False,
    }




@pytest.mark.parametrize(
    ("model_dtype", "effective_version", "fallback"),
    [
        (torch.bfloat16, 4, False),
        (torch.float16, 4, False),
        (torch.float32, 3, True),
    ],
)
def test_native_fp8_output_dtype_resolution(
    hopper,
    model_dtype: torch.dtype,
    effective_version: int,
    fallback: bool,
) -> None:
    config = _config(kv_cache_dtype="fp8", dtype=model_dtype)
    assert fa_utils.resolve_flash_attn_version(config) == effective_version
    assert config.attention_config._flash_attn_version_fallback is fallback


@pytest.mark.parametrize(
    ("head_dim", "head_dim_v"),
    [(96, 96), (192, 128), (192, 192)],
)
def test_split_schedule_rejects_unsupported_geometry_before_device_or_tensors(
    monkeypatch: pytest.MonkeyPatch,
    head_dim: int,
    head_dim_v: int,
) -> None:
    capability_lookup = MagicMock(
        side_effect=AssertionError("capability lookup must not run")
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability_lookup)
    query_start_loc_cpu = MagicMock()
    query_start_loc_cpu.__getitem__.side_effect = AssertionError(
        "query tensor must not be accessed"
    )
    seq_lens_cpu = MagicMock()
    seq_lens_cpu.tolist.side_effect = AssertionError(
        "sequence tensor must not be accessed"
    )

    plan = plan_hopper_split_schedule(
        query_start_loc_cpu,
        seq_lens_cpu,
        device=torch.device("cuda"),
        num_heads_q=32,
        num_heads_kv=8,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        has_qv=False,
        cp_world_size=1,
        window_size=None,
    )

    assert plan is None
    capability_lookup.assert_not_called()
    assert query_start_loc_cpu.mock_calls == []
    assert seq_lens_cpu.mock_calls == []


def test_mixed_graph_schedule_ignores_padded_query_rows(
    monkeypatch: pytest.MonkeyPatch,
):
    device = torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=132),
    )

    plan = plan_hopper_split_schedule(
        torch.tensor([0, 1, 64, 64, 64], dtype=torch.int32),
        torch.tensor([1024, 1024, 1024, 1024], dtype=torch.int32),
        device=device,
        num_heads_q=32,
        num_heads_kv=8,
        head_dim=128,
        head_dim_v=128,
        has_qv=False,
        cp_world_size=1,
        window_size=None,
        cuda_graph_max_num_splits=32,
    )

    assert plan is not None
    assert plan.split_counts is not None
    assert len(plan.split_counts) == 4



def test_default_hopper_flash_attn_version_is_unchanged(hopper):
    config = _config(version=None)

    assert fa_utils.resolve_flash_attn_version(config) is None
    assert config.attention_config.flash_attn_version is None


def test_fa4_request_without_kv_cache_config_does_not_crash(hopper):
    config = _config()
    config.cache_config = None

    assert fa_utils.resolve_flash_attn_version(config) == 4
    assert config.attention_config.flash_attn_version == 4


def test_non_fa3_fp8_cache_format_does_not_fallback(hopper):
    config = _config(kv_cache_dtype="fp8_e5m2")

    assert fa_utils.resolve_flash_attn_version(config) == 4
    assert not config.attention_config._flash_attn_version_fallback


def test_internal_resolution_state_does_not_change_config_hash():
    config = AttentionConfig(flash_attn_version=3)
    original_hash = config.compute_hash()

    config._flash_attn_version_fallback = True
    config._flash_attn_version_required = True

    assert config.compute_hash() == original_hash


def test_internal_resolution_state_survives_config_replace():
    config = AttentionConfig(flash_attn_version=3)
    config._flash_attn_version_fallback = True
    config._flash_attn_version_required = True

    replacement = replace(config, use_non_causal=True)

    assert replacement is not config
    assert replacement._flash_attn_version_fallback
    assert replacement._flash_attn_version_required


@pytest.mark.parametrize("required_by_model", [False, True])
def test_fa4_only_route_rejects_incompatible_fallback(hopper, required_by_model: bool):
    config = _config(kv_cache_dtype="fp8")
    config.attention_config._flash_attn_version_required = required_by_model
    config.model_config.get_head_size = lambda: 160 if required_by_model else 512

    with pytest.raises(ValueError, match="model requires FA4"):
        fa_utils.resolve_flash_attn_version(config)
    assert not config.attention_config._flash_attn_version_fallback


def test_model_required_fa4_accepts_native_fp8_cells(
    hopper, monkeypatch: pytest.MonkeyPatch
):
    config = _config(version=None, kv_cache_dtype="fp8")
    config.model_config.hf_text_config = SimpleNamespace(
        layer_types=["sliding_attention", "full_attention"],
    )
    config.model_config.model_arch_config = MagicMock(total_num_hidden_layers=2)
    config.model_config.model_arch_config.__getitem__.side_effect = [
        SimpleNamespace(head_size=128),
        SimpleNamespace(head_size=256),
    ]
    monkeypatch.setattr(fa_utils, "is_fa_version_supported", lambda version: True)

    Gemma4Config.verify_and_update_config(config)

    assert config.attention_config.flash_attn_version == 4
    assert config.attention_config._flash_attn_version_required
    assert fa_utils.resolve_flash_attn_version(config) == 4
    assert not config.attention_config._flash_attn_version_fallback


@pytest.mark.parametrize("version", (3, 4))
def test_explicit_fa_version_is_frozen_and_logged(
    hopper, monkeypatch: pytest.MonkeyPatch, version
):
    config = _config(version=version)
    info_once = MagicMock()
    monkeypatch.setattr(fa_utils.logger, "info_once", info_once)

    assert fa_utils.resolve_flash_attn_version(config) == version
    assert config.attention_config.flash_attn_version == version
    message = f"requested=FA{version}, effective=FA{version}"
    assert message in info_once.call_args.args[0]
    assert info_once.call_args.kwargs == {"scope": "global"}


@pytest.mark.parametrize(
    ("config_kwargs", "batch_invariant", "import_error", "reason"),
    [
        ({}, True, None, "batch-invariant serving"),
        (
            {"backend": AttentionBackendEnum.FLASH_ATTN_MLA_SPARSE},
            False,
            None,
            "generic sparse-MLA FA3 route",
        ),
        ({}, False, "ModuleNotFoundError: no cutlass", "failed to import"),
    ],
)
def test_fa4_gap_falls_back_the_whole_server(
    hopper,
    monkeypatch: pytest.MonkeyPatch,
    config_kwargs,
    batch_invariant,
    import_error,
    reason,
):
    config = _config(**config_kwargs)
    warning_once = MagicMock()
    monkeypatch.setattr(fa_utils.envs, "VLLM_BATCH_INVARIANT", batch_invariant)
    monkeypatch.setattr(
        hopper,
        "fa4_cutedsl_import_error",
        lambda: import_error,
    )
    monkeypatch.setattr(fa_utils.logger, "warning_once", warning_once)

    assert fa_utils.resolve_flash_attn_version(config) == 3
    assert config.attention_config.flash_attn_version == 3
    assert config.attention_config._flash_attn_version_fallback
    assert reason in warning_once.call_args.args[1]
    assert "whole server is using FA3" in warning_once.call_args.args[0]
    assert warning_once.call_args.kwargs == {"scope": "global"}


def test_auto_selected_generic_sparse_mla_falls_back(
    hopper, monkeypatch: pytest.MonkeyPatch
):
    config = _config(sparse_mla=True)
    warning_once = MagicMock()
    monkeypatch.setattr(fa_utils.logger, "warning_once", warning_once)

    assert fa_utils.resolve_flash_attn_version(config) == 3
    assert config.attention_config.flash_attn_version == 3
    assert "generic sparse-MLA FA3 route" in warning_once.call_args.args[1]


def test_explicit_non_fa_sparse_mla_route_does_not_fallback(hopper):
    config = _config(
        sparse_mla=True,
        backend=AttentionBackendEnum.FLASHMLA_SPARSE,
    )

    assert fa_utils.resolve_flash_attn_version(config) == 4
    assert config.attention_config.flash_attn_version == 4


def test_deepseek_v4_sparse_route_is_outside_fa3_policy(hopper):
    config = _config(sparse_mla=True)
    config.model_config.architecture = "DeepseekV4ForCausalLM"

    assert fa_utils.resolve_flash_attn_version(config) == 4
    assert config.attention_config.flash_attn_version == 4


def test_each_distinct_fallback_reason_uses_warning_once(
    hopper, monkeypatch: pytest.MonkeyPatch
):
    config = _config(kv_cache_dtype="fp8")
    warning_once = MagicMock()
    monkeypatch.setattr(fa_utils.envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(
        hopper,
        "fa4_cutedsl_import_error",
        lambda: "ModuleNotFoundError: no cutlass",
    )
    monkeypatch.setattr(fa_utils.logger, "warning_once", warning_once)

    fa_utils.resolve_flash_attn_version(config)

    reasons = [call.args[1] for call in warning_once.call_args_list]
    assert len(reasons) == len(set(reasons)) == 2
    assert all(
        call.kwargs == {"scope": "global"} for call in warning_once.call_args_list
    )


def test_fallback_reasons_are_collected_from_all_ranks(
    hopper, monkeypatch: pytest.MonkeyPatch
):
    config = _config()
    warning_once = MagicMock()
    monkeypatch.setattr(fa_utils.logger, "warning_once", warning_once)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    cpu_group = object()
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_world_group",
        lambda: SimpleNamespace(cpu_group=cpu_group, world_size=2),
    )

    def all_gather_object(gathered, local, *, group):
        assert group is cpu_group
        gathered[:] = [
            local,
            fa_utils._FA4FallbackState(["remote FA4 dependency failure"], False, True),
        ]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    assert fa_utils.resolve_flash_attn_version(config) == 3
    assert config.attention_config._flash_attn_version_fallback
    assert "remote FA4 dependency failure" in warning_once.call_args.args[1]


def test_non_hopper_rank_joins_and_rejects_unsafe_fallback(
    hopper, monkeypatch: pytest.MonkeyPatch
):
    config = _config()
    monkeypatch.setattr(
        fa_utils.current_platform, "get_device_capability", lambda: None
    )
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    cpu_group = object()
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_world_group",
        lambda: SimpleNamespace(cpu_group=cpu_group, world_size=2),
    )
    gathered_local = None

    def all_gather_object(gathered, local, *, group):
        nonlocal gathered_local
        assert group is cpu_group
        gathered_local = local
        gathered[:] = [
            local,
            fa_utils._FA4FallbackState(["remote FA4 dependency failure"], False, True),
        ]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    with pytest.raises(ValueError, match="FA3 is not supported on every rank"):
        fa_utils.resolve_flash_attn_version(config)
    assert gathered_local == fa_utils._FA4FallbackState([], False, False)
    assert config.attention_config.flash_attn_version == 4
    assert not config.attention_config._flash_attn_version_fallback


@pytest.mark.parametrize(
    ("configured_version", "head_size", "effective_version"),
    [
        (None, 128, 3),
        (3, 512, 4),
        (4, 512, 4),
    ],
)
def test_standard_selector_consumes_frozen_version(
    hopper,
    monkeypatch: pytest.MonkeyPatch,
    configured_version: int | None,
    head_size: int,
    effective_version: int,
):
    config = _config(version=configured_version)
    fake_interface = types.ModuleType("vllm.vllm_flash_attn.flash_attn_interface")
    fake_interface.is_fa_version_supported = lambda version: version in (3, 4)
    fake_interface.fa_version_unsupported_reason = lambda version: None
    monkeypatch.setitem(
        sys.modules,
        "vllm.vllm_flash_attn.flash_attn_interface",
        fake_interface,
    )
    monkeypatch.setattr(fa_utils.current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(fa_utils.current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config_or_none",
        lambda: config,
    )

    assert fa_utils.get_flash_attn_version(head_size=head_size) == effective_version


def test_fallback_rejects_layer_that_requires_fa4(
    hopper, monkeypatch: pytest.MonkeyPatch
):
    config = _config(version=3)
    config.attention_config._flash_attn_version_fallback = True
    fake_interface = types.ModuleType("vllm.vllm_flash_attn.flash_attn_interface")
    fake_interface.is_fa_version_supported = lambda version: version in (3, 4)
    fake_interface.fa_version_unsupported_reason = lambda version: None
    monkeypatch.setitem(
        sys.modules,
        "vllm.vllm_flash_attn.flash_attn_interface",
        fake_interface,
    )
    monkeypatch.setattr(fa_utils.current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(fa_utils.current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config_or_none",
        lambda: config,
    )

    with pytest.raises(ValueError, match="resolved FA4 to FA3"):
        fa_utils.get_flash_attn_version(head_size=512)


@pytest.mark.parametrize("version", (3, 4))
def test_mla_decode_consumes_frozen_version(
    hopper, monkeypatch: pytest.MonkeyPatch, version: int
):
    from vllm.v1.attention.backends.mla import flashattn_mla

    config = _config(version=version)
    monkeypatch.setattr(flashattn_mla, "is_fa_version_supported", lambda version: True)

    assert flashattn_mla._get_mla_fa_version(config) == version


def test_flash_attn_mla_backend_supports_hopper():
    from vllm.v1.attention.backends.mla.flashattn_mla import FlashAttnMLABackend

    assert FlashAttnMLABackend.supports_compute_capability(DeviceCapability(9, 0))
    assert not FlashAttnMLABackend.supports_compute_capability(DeviceCapability(8, 9))
    assert not FlashAttnMLABackend.supports_compute_capability(DeviceCapability(10, 0))


def _forward_case(
    monkeypatch: pytest.MonkeyPatch,
    *,
    module,
    impl_cls,
    head_size: int,
    value_head_size: int,
    model_dtype: torch.dtype,
    output_dtype: torch.dtype,
    native: bool,
    scheduler_metadata: torch.Tensor | None,
) -> SimpleNamespace:
    impl = impl_cls.__new__(impl_cls)
    impl.__dict__.update(
        num_heads=2,
        num_kv_heads=1,
        head_size=head_size,
        scale=0.125,
        alibi_slopes=None,
        sliding_window=(-1, -1),
        kv_cache_dtype="fp8",
        logits_soft_cap=0.0,
        attn_type=flash_attn.AttentionType.DECODER,
        vllm_flash_attn_version=4 if native else 3,
        supports_quant_query_input=True,
        dcp_world_size=1,
        sinks=None,
        model_dtype=model_dtype,
        sm90_fa4_fp8_mode="native" if native else None,
        _dynamic_scheduler_counter=None,
    )
    if module is flash_attn:
        impl.batch_invariant_enabled = False
    else:
        impl.native_fp8_out_dtype = model_dtype if native else None
    metadata = SimpleNamespace(
        num_actual_tokens=1,
        use_cascade=False,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
        max_query_len=1,
        max_seq_len=1,
        block_table=torch.tensor([[0]], dtype=torch.int32),
        scheduler_metadata=scheduler_metadata,
        sliding_window=None,
        causal=True,
        mm_prefix_query_range_tensor=None,
        rswa_prefix_lens=None,
        max_num_splits=2,
    )
    layer = SimpleNamespace(
        _q_scale=torch.tensor(0.5),
        _k_scale=torch.tensor(0.75),
        _v_scale=torch.tensor(1.25),
    )
    fp8_dtype = torch.float8_e4m3fn
    query = torch.empty((1, 2, head_size), dtype=fp8_dtype)
    key = torch.empty((1, 1, head_size), dtype=fp8_dtype)
    value = torch.empty((1, 1, value_head_size), dtype=fp8_dtype)
    kv_cache = torch.empty(
        (1, 1, 1, head_size + value_head_size), dtype=fp8_dtype
    )
    output = torch.empty((1, 2, value_head_size), dtype=output_dtype)
    calls: list[dict] = []
    allocations: list[torch.Tensor] = []
    torch_zeros = torch.zeros

    def record_counter_allocation(*args, **kwargs):
        tensor = torch_zeros(*args, **kwargs)
        if (
            args == ((1,),)
            and kwargs.get("dtype") == torch.int32
            and kwargs.get("device") == query.device
        ):
            allocations.append(tensor)
        return tensor

    monkeypatch.setattr(module.torch, "zeros", record_counter_allocation)
    monkeypatch.setattr(
        module,
        "flash_attn_varlen_func",
        lambda **kwargs: calls.append(kwargs),
        raising=False,
    )
    monkeypatch.setattr(
        module, "canonicalize_singleton_dim_strides", lambda tensor: tensor
    )
    monkeypatch.setattr(module.current_platform, "fp8_dtype", lambda: fp8_dtype)

    impl.forward(layer, query, key, value, kv_cache, metadata, output)
    impl.forward(layer, query, key, value, kv_cache, metadata, output)
    return SimpleNamespace(
        calls=calls,
        allocations=allocations,
        counter_slot=impl._dynamic_scheduler_counter,
        output=output,
        metadata=metadata,
    )


def _dense_forward_case(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_dtype: torch.dtype,
    output_dtype: torch.dtype,
    native: bool,
) -> tuple[dict, torch.Tensor, SimpleNamespace]:
    scheduler_metadata = torch.tensor([2], dtype=torch.int32)
    result = _forward_case(
        monkeypatch,
        module=flash_attn,
        impl_cls=flash_attn.FlashAttentionImpl,
        head_size=64,
        value_head_size=64,
        model_dtype=model_dtype,
        output_dtype=output_dtype,
        native=native,
        scheduler_metadata=scheduler_metadata,
    )
    forwarded = result.calls[-1]
    forwarded["observed_dynamic_scheduler_counters"] = [
        call.get("dynamic_scheduler_counter") for call in result.calls
    ]
    forwarded["dynamic_scheduler_counter_slot"] = result.counter_slot
    forwarded["dynamic_scheduler_counter_allocations"] = result.allocations
    return forwarded, result.output, result.metadata


@pytest.mark.parametrize("model_dtype", (torch.bfloat16, torch.float16))
def test_dense_native_fp8_forwards_preserved_output_dtype_and_scalar_descales(
    monkeypatch: pytest.MonkeyPatch, model_dtype: torch.dtype
):
    forwarded, output, metadata = _dense_forward_case(
        monkeypatch,
        model_dtype=model_dtype,
        output_dtype=model_dtype,
        native=True,
    )

    assert forwarded["out"].data_ptr() == output.data_ptr()
    assert forwarded["out_dtype"] == model_dtype
    assert forwarded["scheduler_metadata"] is metadata.scheduler_metadata
    assert forwarded["fa_version"] == 4
    assert forwarded["q_descale"].ndim == 0
    assert forwarded["k_descale"].ndim == 0
    assert forwarded["v_descale"].ndim == 0
    counters = forwarded["observed_dynamic_scheduler_counters"]
    assert len(forwarded["dynamic_scheduler_counter_allocations"]) == 1
    assert len(counters) == 2
    assert counters[0] is counters[1]
    assert counters[0] is forwarded["dynamic_scheduler_counter_slot"]
    assert counters[0].shape == (1,)
    assert counters[0].dtype == torch.int32
    assert counters[0].device == output.device
    assert counters[0].storage_offset() == 0


def test_dense_native_fp8_rejects_preallocated_output_dtype_mismatch(
    monkeypatch: pytest.MonkeyPatch,
):
    with pytest.raises(AssertionError, match="must match the configured model dtype"):
        _dense_forward_case(
            monkeypatch,
            model_dtype=torch.float16,
            output_dtype=torch.bfloat16,
            native=True,
        )


def test_dense_fa3_preserves_expanded_descales_and_no_native_output_request(
    monkeypatch: pytest.MonkeyPatch,
):
    forwarded, _, _ = _dense_forward_case(
        monkeypatch,
        model_dtype=torch.float16,
        output_dtype=torch.float16,
        native=False,
    )

    assert forwarded["out_dtype"] is None
    assert forwarded["fa_version"] == 3
    assert forwarded["q_descale"].shape == (1, 1)
    assert forwarded["k_descale"].shape == (1, 1)
    assert forwarded["v_descale"].shape == (1, 1)
    assert forwarded["observed_dynamic_scheduler_counters"] == [None, None]
    assert forwarded["dynamic_scheduler_counter_allocations"] == []
    assert forwarded["dynamic_scheduler_counter_slot"] is None


def _diffkv_forward_case(
    monkeypatch: pytest.MonkeyPatch, *, native: bool
) -> tuple[
    list[torch.Tensor | None],
    list[torch.Tensor],
    torch.Tensor | None,
]:
    result = _forward_case(
        monkeypatch,
        module=flash_attn_diffkv,
        impl_cls=flash_attn_diffkv.FlashAttentionDiffKVImpl,
        head_size=192,
        value_head_size=128,
        model_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        native=native,
        scheduler_metadata=None,
    )
    return (
        [call.get("dynamic_scheduler_counter") for call in result.calls],
        result.allocations,
        result.counter_slot,
    )


def test_diffkv_native_fp8_reuses_dynamic_scheduler_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counters, allocations, counter_slot = _diffkv_forward_case(
        monkeypatch, native=True
    )

    assert len(allocations) == 1
    assert len(counters) == 2
    assert counters[0] is counters[1]
    assert counters[0] is counter_slot
    assert counters[0].shape == (1,)
    assert counters[0].dtype == torch.int32
    assert counters[0].device == torch.device("cpu")
    assert counters[0].is_contiguous()
    assert counters[0].storage_offset() == 0


def test_diffkv_fa3_does_not_allocate_dynamic_scheduler_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counters, allocations, counter_slot = _diffkv_forward_case(
        monkeypatch, native=False
    )

    assert counters == [None, None]
    assert allocations == []
    assert counter_slot is None
