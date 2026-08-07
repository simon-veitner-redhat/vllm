# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Track whether upstream flash-attn is available on ROCm.
# Set during module initialization and never modified afterwards.
# This module-level flag avoids repeated import attempts and ensures
# consistent behavior (similar to IS_AITER_FOUND in _aiter_ops.py).
_ROCM_FLASH_ATTN_AVAILABLE = False


def _model_uses_attention_sinks(model_config: Any) -> bool:
    text_config = model_config.hf_text_config
    dflash_config = getattr(text_config, "dflash_config", None)
    return bool(
        getattr(text_config, "attention_sink", False)
        or getattr(text_config, "swa_attention_sink_enabled", False)
        or getattr(text_config, "add_swa_attention_sink_bias", False)
        or (
            isinstance(dflash_config, dict)
            and dflash_config.get("attention_sink_bias", False)
        )
    )


def _hopper_fa4_fp8_fallback_reasons(vllm_config: "VllmConfig") -> list[str]:
    """Return ordered failures of the supported Hopper FA4 FP8 shape."""
    attention = vllm_config.attention_config
    cache = vllm_config.cache_config
    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    reasons: list[str] = []

    backend = attention.backend
    if backend is None or getattr(backend, "name", str(backend)) != "FLASH_ATTN":
        reasons.append("backend")
    if attention.backend_per_kind:
        reasons.append("per-kind backend")
    if cache._checkpoint_implied_fp8:
        reasons.append("cache dtype auto/checkpoint intent")
    if cache.block_size != 16:
        reasons.append("page")
    if parallel.tensor_parallel_size != 1:
        reasons.append("TP")
    if parallel.pipeline_parallel_size != 1:
        reasons.append("PP")
    if parallel.decode_context_parallel_size != 1:
        reasons.append("DCP")
    if model.dtype != torch.bfloat16:
        reasons.append("activation/output dtype")
    if model.get_head_size() != 128:
        reasons.append("geometry")
    if getattr(model.hf_text_config, "linear_value_head_dim", 128) != 128:
        reasons.append("value geometry")
    if model.get_num_attention_heads(parallel) != 32:
        reasons.append("query heads")
    if model.get_num_kv_heads(parallel) != 8:
        reasons.append("GQA")
    if (
        model.use_mla
        or model.is_diffusion
        or model.is_multimodal_model
        or model.is_encoder_decoder
        or model.runner_type != "generate"
    ):
        reasons.append("attention kind")
    if attention.use_non_causal:
        reasons.append("non-causal attention")
    if cache.sliding_window is not None or model.get_sliding_window() is not None:
        reasons.append("sliding window")
    if model.rswa_window is not None:
        reasons.append("mask modification")
    if model.is_mm_prefix_lm:
        reasons.append("MM-prefix mask")
    if getattr(model.hf_text_config, "attn_logit_softcapping", None) not in (
        None,
        0.0,
    ):
        reasons.append("attention softcap")
    if _model_uses_attention_sinks(model):
        reasons.append("attention sinks")
    if cache.enable_prefix_caching:
        reasons.append("prefix caching")
    if cache.calculate_kv_scales:
        reasons.append("runtime scales")
    if cache.kv_cache_dtype_skip_layers:
        reasons.append("mixed cache dtype")
    if vllm_config.speculative_config is not None:
        reasons.append("speculative decoding")
    if not model.disable_cascade_attn:
        reasons.append("cascade")
    if not vllm_config.kernel_config.enable_jit_warmup:
        reasons.append("JIT warmup disabled")

    scheme = model.model_arch_config.quantization_config or {}
    kv_scheme = scheme.get("kv_cache_scheme") or {}
    expected_scheme = {
        "dynamic": False,
        "num_bits": 8,
        "strategy": "tensor",
        "symmetric": True,
        "type": "float",
    }
    if (
        scheme.get("quant_method") != "compressed-tensors"
        or scheme.get("quantization_status") != "frozen"
        or any(kv_scheme.get(key) != value for key, value in expected_scheme.items())
    ):
        reasons.append("quantization scheme")

    return reasons


def _fa4_cute_import_error() -> str | None:
    """Import the real CuTeDSL entry point used by FA4 kernels."""
    try:
        import_module("vllm.vllm_flash_attn.cute.interface")
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None


def _uses_generic_sparse_mla_fa3(vllm_config: "VllmConfig") -> bool:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    attention_config = vllm_config.attention_config
    configured_backends = (
        attention_config.backend,
        *attention_config.backend_per_kind.values(),
    )
    if AttentionBackendEnum.FLASH_ATTN_MLA_SPARSE in configured_backends:
        return True
    if attention_config.backend is not None:
        return False
    if attention_config.backend_per_kind.get("mla_attention") is not None:
        return False

    model_config = vllm_config.model_config
    if model_config is None or not getattr(model_config, "use_mla", False):
        return False
    if getattr(model_config, "architecture", None) in (
        "DeepseekV4ForCausalLM",
        "DeepSeekV4MTPModel",
    ):
        return False
    return hasattr(model_config.hf_text_config, "index_topk")


def _requires_fa4(vllm_config: "VllmConfig") -> bool:
    attention_config = vllm_config.attention_config
    if attention_config._flash_attn_version_required:
        return True

    model_config = vllm_config.model_config
    if model_config is None:
        return False
    get_head_size = getattr(model_config, "get_head_size", None)
    return get_head_size is not None and get_head_size() > 256


def _collect_fallback_reasons(reasons: list[str]) -> list[str]:
    """Return the ordered union of fallback reasons from every worker rank."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return reasons

    world_size = torch.distributed.get_world_size()
    if world_size == 1:
        return reasons
    gathered_reasons: list[list[str] | None] = [None] * world_size
    torch.distributed.all_gather_object(gathered_reasons, reasons)
    return list(
        dict.fromkeys(
            reason
            for rank_reasons in gathered_reasons
            if rank_reasons is not None
            for reason in rank_reasons
        )
    )


def resolve_flash_attn_version(vllm_config: "VllmConfig") -> int | None:
    """Resolve an explicit Hopper FA4 request before model construction.

    Known FA4 gaps that FA3 supports freeze the whole server to FA3. Defaults
    and routes owned by other backends retain their existing selectors.
    """
    requested = vllm_config.attention_config.flash_attn_version
    capability = current_platform.get_device_capability()
    if capability is None:
        return requested
    if requested == 3:
        if capability.major == 9 and capability.minor == 0:
            logger.info_once(
                "FlashAttention version resolved for the whole server: "
                "requested=FA3, effective=FA3.",
                scope="global",
            )
        return 3
    if (capability.major, capability.minor) != (9, 0) or requested != 4:
        return requested
    reasons: list[str] = []
    if import_error := _fa4_cute_import_error():
        reasons.append(f"the FA4 CuTeDSL interface failed to import ({import_error})")
    if envs.VLLM_BATCH_INVARIANT:
        reasons.append("FA4 does not support batch-invariant serving on Hopper")

    cache_config = vllm_config.cache_config
    is_hopper_fp8 = cache_config is not None and cache_config.cache_dtype in (
        "fp8",
        "fp8_e4m3",
    )
    if is_hopper_fp8:
        reasons.extend(_hopper_fa4_fp8_fallback_reasons(vllm_config))
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise ValueError(
                "Hopper FA4 FP8 requires BF16 model activations and output; "
                "FP16 activation output adaptation is not supported"
            )
    if _uses_generic_sparse_mla_fa3(vllm_config):
        reasons.append("FA4 does not support the generic sparse-MLA FA3 route")

    reasons = _collect_fallback_reasons(reasons)
    if reasons:
        if _requires_fa4(vllm_config):
            raise ValueError(
                "The model requires FA4, but the requested Hopper FA4 "
                f"configuration cannot use it: {'; '.join(reasons)}"
            )
        vllm_config.attention_config.flash_attn_version = 3
        vllm_config.attention_config._flash_attn_version_fallback = True
        if is_hopper_fp8:
            logger.warning_once(
                "Hopper FA4 FP8 capability check failed fields: %s; the whole "
                "server is using FA3 (requested=FA4, effective=FA3).",
                ", ".join(reasons),
                scope="global",
            )
        else:
            for reason in reasons:
                logger.warning_once(
                    "%s; the whole server is using FA3 "
                    "(requested=FA4, effective=FA3).",
                    reason,
                    scope="global",
                )
        return 3

    vllm_config.attention_config._hopper_fa4_fp8 = is_hopper_fp8
    logger.info_once(
        "FlashAttention version resolved for the whole server: "
        "requested=FA4, effective=FA4.",
        scope="global",
    )
    return 4

if current_platform.is_cuda():
    from vllm._custom_ops import reshape_and_cache_flash
    from vllm.vllm_flash_attn import (  # type: ignore[attr-defined]
        compile_flash_attn_varlen_func_from_specs,
        flash_attn_varlen_func,
        get_scheduler_metadata,
    )

elif current_platform.is_xpu():
    from vllm import _custom_ops as ops
    from vllm._xpu_ops import xpu_ops

    reshape_and_cache_flash = ops.reshape_and_cache_flash
    flash_attn_varlen_func = xpu_ops.flash_attn_varlen_func  # type: ignore[assignment]
    compile_flash_attn_varlen_func_from_specs = None  # type: ignore[assignment]
    get_scheduler_metadata = xpu_ops.get_scheduler_metadata  # type: ignore[assignment]
elif current_platform.is_rocm():
    try:
        from flash_attn import flash_attn_varlen_func  # type: ignore[no-redef]

        compile_flash_attn_varlen_func_from_specs = None  # type: ignore[assignment]

        # Mark that upstream flash-attn is available on ROCm
        _ROCM_FLASH_ATTN_AVAILABLE = True
    except ImportError:

        def flash_attn_varlen_func(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef,misc]
            raise ImportError(
                "ROCm platform requires upstream flash-attn "
                "to be installed. Please install flash-attn first."
            )

        compile_flash_attn_varlen_func_from_specs = None  # type: ignore[assignment]

    # ROCm doesn't use scheduler metadata (FA3 feature), provide stub
    def get_scheduler_metadata(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        return None

    # ROCm uses the C++ custom op for reshape_and_cache
    from vllm import _custom_ops as ops

    reshape_and_cache_flash = ops.reshape_and_cache_flash


@dataclass(frozen=True)
class FlashAttentionCuTeDSLCompileSpec:
    """High-level FA4 compile-only request used by vLLM warmup.

    This is not the CuTeDSL cache key. FA4 owns the selector that maps these
    serving inputs to the actual compile-static fields: tile sizes, q_stage,
    Split-KV, scheduler choice, layout-presence booleans, dtype/head dims,
    arch, and related fields.
    """

    q_shape: tuple[int, ...]
    k_shape: tuple[int, ...]
    v_shape: tuple[int, ...]
    q_dtype: torch.dtype
    max_seqlen_q: int
    max_seqlen_k: int
    softmax_scale: float
    causal: bool
    fa_version: int
    v_stride: tuple[int, ...] | None = None
    cu_seqlens_q_shape: tuple[int, ...] | None = None
    cu_seqlens_k_shape: tuple[int, ...] | None = None
    window_size: tuple[int, int] | None = None
    return_softmax_lse: bool = False
    num_splits: int = 0

    def compile(self) -> None:
        assert compile_flash_attn_varlen_func_from_specs is not None
        window_size = list(self.window_size) if self.window_size is not None else None
        compile_flash_attn_varlen_func_from_specs(
            q_shape=self.q_shape,
            k_shape=self.k_shape,
            v_shape=self.v_shape,
            q_dtype=self.q_dtype,
            v_stride=self.v_stride,
            cu_seqlens_q_shape=self.cu_seqlens_q_shape,
            cu_seqlens_k_shape=self.cu_seqlens_k_shape,
            max_seqlen_q=self.max_seqlen_q,
            max_seqlen_k=self.max_seqlen_k,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
            window_size=window_size,
            return_softmax_lse=self.return_softmax_lse,
            fa_version=self.fa_version,
            num_splits=self.num_splits,
        )

    def request_key(self) -> tuple[object, ...]:
        return (
            self.q_shape,
            self.k_shape,
            self.v_shape,
            self.q_dtype,
            self.max_seqlen_q,
            self.max_seqlen_k,
            self.softmax_scale,
            self.causal,
            self.fa_version,
            self.v_stride,
            self.cu_seqlens_q_shape,
            self.cu_seqlens_k_shape,
            self.window_size,
            self.return_softmax_lse,
            self.num_splits,
        )


def get_flash_attn_version(
    requires_alibi: bool = False,
    head_size: int | None = None,
    head_size_v: int | None = None,
    has_sinks: bool = False,
    requires_local_attention: bool = False,
) -> int | None:
    if current_platform.is_xpu():
        return 2
    if current_platform.is_rocm():
        # ROCm doesn't use vllm_flash_attn; return None to skip fa_version arg
        return None
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import (
            fa_version_unsupported_reason,
            is_fa_version_supported,
        )

        device_capability = current_platform.get_device_capability()

        assert device_capability is not None

        # 1. default version depending on platform
        if device_capability.major == 9 and is_fa_version_supported(3):
            # Hopper (SM90): prefer FA3
            fa_version = 3
        elif device_capability.major == 10 and is_fa_version_supported(4):
            # Blackwell (SM100+, restrict to SM100 for now): prefer FA4
            fa_version = 4
        else:
            # Fallback to FA2
            fa_version = 2

        # 2. override if passed by environment or config
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        if (
            vllm_config is not None
            and vllm_config.attention_config.flash_attn_version is not None
        ):
            fa_version = vllm_config.attention_config.flash_attn_version

        # 3. fallback for unsupported combinations
        if device_capability.major >= 10 and fa_version == 3:
            logger.warning_once(
                "Cannot use FA version 3 on Blackwell platform, "
                "defaulting to FA version 4 if supported, otherwise FA2."
            )
            fa_version = 4 if is_fa_version_supported(4) else 2

        if requires_alibi and fa_version == 3:
            logger.warning_once(
                "Cannot use FA version 3 with ALiBi, defaulting to FA version 2."
            )
            fa_version = 2

        if requires_alibi and fa_version == 4:
            logger.warning_once(
                "Cannot use FA version 4 with ALiBi, defaulting to FA version 2."
            )
            fa_version = 2

        # Some FA3 unsupported SM90 cases can use FA4 when available.
        if (
            fa_version == 3
            and device_capability.major == 9
            and is_fa_version_supported(4)
        ):
            upgrade_reason = None
            if head_size is not None and head_size > 256:
                upgrade_reason = f"FA3 does not support head_size={head_size} on SM90"
            elif (
                has_sinks
                and head_size is not None
                and head_size_v is not None
                and head_size != head_size_v
            ):
                upgrade_reason = "Diff-KV with sinks"
            elif (
                vllm_config is not None
                and vllm_config.model_config is not None
                and vllm_config.model_config.is_diffusion
            ):
                upgrade_reason = "Per-sequence causal (dynamic_causal) requires FA4"
            if upgrade_reason:
                if (
                    vllm_config is not None
                    and vllm_config.attention_config._flash_attn_version_fallback
                ):
                    raise ValueError(
                        "The server resolved FA4 to FA3, but this attention layer "
                        f"requires FA4: {upgrade_reason}"
                    )
                logger.info_once(
                    "%s: upgrading FlashAttention 3 -> 4",
                    upgrade_reason,
                    scope="local",
                )
                fa_version = 4

        # FA4 currently uses batch-shape-dependent scheduling
        # heuristics on SM100+, which breaks batch invariance.
        if envs.VLLM_BATCH_INVARIANT and fa_version == 4:
            logger.warning_once(
                "Cannot use FA version 4 with batch invariance, "
                "defaulting to FA version 2.",
            )
            fa_version = 2

        if (
            fa_version == 4
            and device_capability.major >= 10
            and head_size == 256
            and requires_local_attention
        ):
            logger.warning_once(
                "FA4 on Blackwell does not support local attention with "
                "head_size=256, defaulting to FA version 2."
            )
            fa_version = 2

        # FA4 on SM100 (Blackwell) has TMEM capacity limits that restrict
        # supported head dimensions to ≤128, with exceptions for 256 and 192/128 (MLA
        # prefill). Development of symmetric 192, 384, and 512 support is being tracked
        # in https://github.com/Dao-AILab/flash-attention/issues/2456
        if (
            fa_version == 4
            and device_capability.major >= 10
            and head_size is not None
            and head_size > 128
            and not (head_size == 256 or (head_size == 192 and head_size_v == 128))
        ):
            logger.warning_once(
                "FA4 on Blackwell does not support head_size=%d due to TMEM "
                "capacity limits, defaulting to FA version 2.",
                head_size,
            )
            fa_version = 2

        if not is_fa_version_supported(fa_version):
            logger.error(
                "Cannot use FA version %d is not supported due to %s",
                fa_version,
                fa_version_unsupported_reason(fa_version),
            )

        assert is_fa_version_supported(fa_version)
        return fa_version
    except (ImportError, AssertionError):
        return None


def is_fa_version_supported(fa_version: int) -> bool:
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import (
            is_fa_version_supported as _is_fa_version_supported,
        )

        return _is_fa_version_supported(fa_version)
    except ImportError:
        return False


def flash_attn_supports_kv_cache_dtype(
    kv_cache_dtype: str = "fp8_e4m3",
    *,
    requires_alibi: bool = False,
    head_size: int | None = None,
    head_size_v: int | None = None,
    has_sinks: bool = False,
) -> bool:
    if kv_cache_dtype == "fp8_e5m2":
        return False
    if current_platform.is_xpu():
        return True
    fa_version = get_flash_attn_version(
        requires_alibi=requires_alibi,
        head_size=head_size,
        head_size_v=head_size_v,
        has_sinks=has_sinks,
    )
    if fa_version == 3 and current_platform.is_device_capability_family(90):
        return True
    if fa_version == 4 and current_platform.is_device_capability_family(100):
        return True
    if fa_version == 4 and current_platform.is_device_capability(90):
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        return bool(
            vllm_config is not None and vllm_config.attention_config._hopper_fa4_fp8
        )
    return False


def flash_attn_supports_quant_query_input() -> bool:
    return not current_platform.is_xpu()


def flash_attn_supports_sinks() -> bool:
    if current_platform.is_xpu():
        return True
    return get_flash_attn_version() in (3, 4)


def flash_attn_supports_mla():
    from vllm.platforms import current_platform

    if current_platform.is_cuda():
        try:
            from vllm.vllm_flash_attn.flash_attn_interface import (
                is_fa_version_supported,
            )

            if current_platform.is_device_capability(90):
                return is_fa_version_supported(3) or is_fa_version_supported(4)
            return (
                current_platform.is_device_capability_family(90)
                and is_fa_version_supported(3)
            )

        except (ImportError, AssertionError):
            pass
    return False


def is_flash_attn_varlen_func_available() -> bool:
    """Check if flash_attn_varlen_func is available.

    This function determines whether the flash_attn_varlen_func imported at module
    level is a working implementation or a stub.

    Platform-specific sources:
    - CUDA: vllm.vllm_flash_attn.flash_attn_varlen_func
    - XPU: xpu_ops.flash_attn_varlen_func
    - ROCm: upstream flash_attn.flash_attn_varlen_func (if available)

    Note: This is separate from the AITER flash attention backend (rocm_aiter_fa.py)
    which uses rocm_aiter_ops.flash_attn_varlen_func. The condition to use AITER is
    handled separately via _aiter_ops.is_aiter_found_and_supported().

    Returns:
        bool: True if a working flash_attn_varlen_func implementation is available.
    """
    if current_platform.is_cuda() or current_platform.is_xpu():
        # CUDA and XPU always have flash_attn_varlen_func available
        return True

    if current_platform.is_rocm():
        # Use the flag set during module import to check if
        # upstream flash-attn was successfully imported
        return _ROCM_FLASH_ATTN_AVAILABLE

    return False
