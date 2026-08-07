# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import AttentionConfig, replace
from vllm.config.cache import CacheConfig
from vllm.model_executor.models.config import Gemma4Config
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends import fa_utils
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _config(
    *,
    version: int | None = 4,
    kv_cache_dtype: str = "auto",
    backend: AttentionBackendEnum | None = None,
    sparse_mla: bool = False,
    admitted_fp8: bool = False,
):
    hf_text_config = SimpleNamespace()
    if sparse_mla:
        hf_text_config.index_topk = 2048
    if admitted_fp8:
        kv_cache_dtype = "fp8"
        backend = AttentionBackendEnum.FLASH_ATTN
    quantization_config = None
    if admitted_fp8:
        quantization_config = {
            "quant_method": "compressed-tensors",
            "quantization_status": "frozen",
            "kv_cache_scheme": {
                "dynamic": False,
                "num_bits": 8,
                "strategy": "tensor",
                "symmetric": True,
                "type": "float",
            }
        }
    return SimpleNamespace(
        attention_config=AttentionConfig(
            flash_attn_version=version,
            backend=backend,
        ),
        cache_config=SimpleNamespace(
            cache_dtype=kv_cache_dtype,
            _checkpoint_implied_fp8=False,
            block_size=16,
            sliding_window=None,
            enable_prefix_caching=False,
            calculate_kv_scales=False,
            kv_cache_dtype_skip_layers=None,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        kernel_config=SimpleNamespace(enable_jit_warmup=True),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            use_mla=sparse_mla,
            is_diffusion=False,
            is_multimodal_model=False,
            is_encoder_decoder=False,
            is_mm_prefix_lm=False,
            runner_type="generate",
            rswa_window=None,
            disable_cascade_attn=True,
            architecture="DeepseekV3ForCausalLM",
            hf_text_config=hf_text_config,
            model_arch_config=SimpleNamespace(
                quantization_config=quantization_config
            ),
            get_head_size=lambda: 128,
            get_sliding_window=lambda: None,
            get_num_attention_heads=lambda parallel: 32,
            get_num_kv_heads=lambda parallel: 8,
        ),
        speculative_config=None,
    )


@pytest.fixture
def hopper(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        fa_utils.current_platform,
        "get_device_capability",
        lambda: DeviceCapability(major=9, minor=0),
    )
    monkeypatch.setattr(fa_utils.envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(fa_utils, "_fa4_cute_import_error", lambda: None)


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

    cache_config = CacheConfig()
    cache_config._checkpoint_implied_fp8 = True
    assert "_checkpoint_implied_fp8" not in cache_config.metrics_info()


def test_internal_resolution_state_survives_config_replace():
    config = AttentionConfig(flash_attn_version=3)
    config._flash_attn_version_fallback = True
    config._flash_attn_version_required = True

    replacement = replace(config, use_non_causal=True)

    assert replacement is not config
    assert replacement._flash_attn_version_fallback
    assert replacement._flash_attn_version_required


@pytest.mark.parametrize("required_by_model", [False, True])
def test_fa4_only_route_rejects_incompatible_fallback(
    hopper, required_by_model: bool
):
    config = _config(kv_cache_dtype="fp8")
    config.attention_config._flash_attn_version_required = required_by_model
    if not required_by_model:
        config.model_config.get_head_size = lambda: 512

    with pytest.raises(ValueError, match="model requires FA4"):
        fa_utils.resolve_flash_attn_version(config)
    assert not config.attention_config._flash_attn_version_fallback


def test_model_required_fa4_rejects_fallback(
    hopper, monkeypatch: pytest.MonkeyPatch
):
    config = _config(version=None, kv_cache_dtype="fp8")
    config.model_config.hf_text_config = SimpleNamespace(
        head_dim=128,
        global_head_dim=256,
    )
    monkeypatch.setattr(fa_utils, "is_fa_version_supported", lambda version: True)

    Gemma4Config.verify_and_update_config(config)

    assert config.attention_config.flash_attn_version == 4
    assert config.attention_config._flash_attn_version_required
    with pytest.raises(ValueError, match="model requires FA4"):
        fa_utils.resolve_flash_attn_version(config)


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
        ({"kv_cache_dtype": "fp8"}, False, None, "backend"),
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
    monkeypatch.setattr(fa_utils, "_fa4_cute_import_error", lambda: import_error)
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


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("admitted", "fa4"),
        ("missing-backend", "fallback"),
        ("checkpoint-implied", "fallback"),
        ("head-scales", "fallback"),
        ("runtime-scales", "fallback"),
        ("page32", "fallback"),
        ("tp2", "fallback"),
        ("pp2", "fallback"),
        ("dcp2", "fallback"),
        ("head-dim64", "fallback"),
        ("gqa8", "fallback"),
        ("value-dim64", "fallback"),
        ("qheads16", "fallback"),
        ("per-kind-backend", "fallback"),
        ("mla", "fallback"),
        ("diffusion", "fallback"),
        ("multimodal", "fallback"),
        ("encoder-decoder", "fallback"),
        ("non-generate", "fallback"),
        ("non-causal", "fallback"),
        ("cache-window", "fallback"),
        ("model-window", "fallback"),
        ("rswa", "fallback"),
        ("mm-prefix", "fallback"),
        ("softcap", "fallback"),
        ("sinks", "fallback"),
        ("prefix-cache", "fallback"),
        ("mixed-cache", "fallback"),
        ("speculative", "fallback"),
        ("cascade", "fallback"),
        ("bad-scheme", "fallback"),
        ("fp16", "error"),
    ],
)
def test_hopper_fa4_fp8_policy(hopper, case, expected):
    config = _config(admitted_fp8=True)
    if case == "missing-backend":
        config.attention_config.backend = None
    elif case == "checkpoint-implied":
        config.cache_config._checkpoint_implied_fp8 = True
    elif case == "head-scales":
        scheme = config.model_config.model_arch_config.quantization_config
        scheme["kv_cache_scheme"]["strategy"] = "head"
    elif case == "runtime-scales":
        config.cache_config.calculate_kv_scales = True
    elif case == "page32":
        config.cache_config.block_size = 32
    elif case == "tp2":
        config.parallel_config.tensor_parallel_size = 2
    elif case == "pp2":
        config.parallel_config.pipeline_parallel_size = 2
    elif case == "dcp2":
        config.parallel_config.decode_context_parallel_size = 2
    elif case == "head-dim64":
        config.model_config.get_head_size = lambda: 64
    elif case == "gqa8":
        config.model_config.get_num_kv_heads = lambda parallel: 4
    elif case == "value-dim64":
        config.model_config.hf_text_config.linear_value_head_dim = 64
    elif case == "qheads16":
        config.model_config.get_num_attention_heads = lambda parallel: 16
    elif case == "per-kind-backend":
        config.attention_config.backend_per_kind["attention"] = (
            AttentionBackendEnum.FLASH_ATTN
        )
    elif case == "mla":
        config.model_config.use_mla = True
    elif case == "diffusion":
        config.model_config.is_diffusion = True
    elif case == "multimodal":
        config.model_config.is_multimodal_model = True
    elif case == "encoder-decoder":
        config.model_config.is_encoder_decoder = True
    elif case == "non-generate":
        config.model_config.runner_type = "pooling"
    elif case == "non-causal":
        config.attention_config.use_non_causal = True
    elif case == "cache-window":
        config.cache_config.sliding_window = 128
    elif case == "model-window":
        config.model_config.get_sliding_window = lambda: 128
    elif case == "rswa":
        config.model_config.rswa_window = 128
    elif case == "mm-prefix":
        config.model_config.is_mm_prefix_lm = True
    elif case == "softcap":
        config.model_config.hf_text_config.attn_logit_softcapping = 30.0
    elif case == "sinks":
        config.model_config.hf_text_config.attention_sink = True
    elif case == "prefix-cache":
        config.cache_config.enable_prefix_caching = True
    elif case == "mixed-cache":
        config.cache_config.kv_cache_dtype_skip_layers = [0]
    elif case == "speculative":
        config.speculative_config = SimpleNamespace()
    elif case == "cascade":
        config.model_config.disable_cascade_attn = False
    elif case == "bad-scheme":
        scheme = config.model_config.model_arch_config.quantization_config
        scheme["quant_method"] = "other"
    elif case == "fp16":
        config.model_config.dtype = torch.float16

    if expected == "error":
        with pytest.raises(ValueError, match="requires BF16"):
            fa_utils.resolve_flash_attn_version(config)
        return

    effective = fa_utils.resolve_flash_attn_version(config)
    assert effective == (4 if expected == "fa4" else 3)
    assert config.attention_config._hopper_fa4_fp8 is (expected == "fa4")
    assert config.attention_config._flash_attn_version_fallback is (
        expected == "fallback"
    )
    if case == "admitted":
        disabled = _config(admitted_fp8=True)
        disabled.kernel_config.enable_jit_warmup = False
        assert fa_utils.resolve_flash_attn_version(disabled) == 3


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
        fa_utils,
        "_fa4_cute_import_error",
        lambda: "ModuleNotFoundError: no cutlass",
    )
    monkeypatch.setattr(fa_utils.logger, "warning_once", warning_once)

    fa_utils.resolve_flash_attn_version(config)

    warning_once.assert_called_once()
    reasons = warning_once.call_args.args[1]
    assert "failed to import" in reasons
    assert "batch-invariant serving" in reasons
    assert "backend" in reasons
    assert warning_once.call_args.kwargs == {"scope": "global"}


def test_fallback_reasons_are_collected_from_all_ranks(
    hopper, monkeypatch: pytest.MonkeyPatch
):
    config = _config()
    warning_once = MagicMock()
    monkeypatch.setattr(fa_utils.logger, "warning_once", warning_once)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def all_gather_object(gathered, local):
        gathered[:] = [local, ["remote FA4 dependency failure"]]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    assert fa_utils.resolve_flash_attn_version(config) == 3
    assert config.attention_config._flash_attn_version_fallback
    assert "remote FA4 dependency failure" in warning_once.call_args.args[1]


def test_cute_probe_imports_kernel_entrypoint(monkeypatch: pytest.MonkeyPatch):
    import_module = MagicMock(side_effect=ModuleNotFoundError("no cutlass"))
    monkeypatch.setattr(fa_utils, "import_module", import_module)

    assert fa_utils._fa4_cute_import_error() == "ModuleNotFoundError: no cutlass"
    import_module.assert_called_once_with("vllm.vllm_flash_attn.cute.interface")


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
    fake_interface = types.ModuleType(
        "vllm.vllm_flash_attn.flash_attn_interface"
    )
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
    fake_interface = types.ModuleType(
        "vllm.vllm_flash_attn.flash_attn_interface"
    )
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


def test_mla_decode_consumes_frozen_version(hopper, monkeypatch: pytest.MonkeyPatch):
    from vllm.v1.attention.backends.mla import flashattn_mla

    config = _config(version=3)
    monkeypatch.setattr(flashattn_mla, "is_fa_version_supported", lambda version: True)

    assert flashattn_mla._get_mla_fa_version(config) == 3

    monkeypatch.setattr(
        flashattn_mla.current_platform,
        "is_device_capability_family",
        lambda capability: capability == 90,
    )
    monkeypatch.setattr(
        flashattn_mla.current_platform,
        "is_device_capability",
        lambda capability: capability == 90,
    )
    config.attention_config.flash_attn_version = 4
    assert flashattn_mla._get_mla_fa_version(config) == 4

    monkeypatch.setattr(
        flashattn_mla.current_platform, "is_device_capability", lambda _: False
    )
    assert flashattn_mla._get_mla_fa_version(config) == 3

    monkeypatch.setattr(
        flashattn_mla.current_platform,
        "is_device_capability_family",
        lambda _: False,
    )
    assert flashattn_mla._get_mla_fa_version(config) == 4
