# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash adapter for the DeepGEMM MegaMoE kernel."""

from typing import Any

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    MEGA_MOE_FP8_BLOCK_SM90,
    MEGA_MOE_MXFP8,
    DeepGemmMegaMoEExperts,
    MegaMoEWeightSpec,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    requant_block_fp8_to_ue8m0,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import _import_deep_gemm

logger = init_logger(__name__)

# (local expert id, shard id); the fused shared expert has expert id None.
Shard = tuple[int | None, str]


class Glm5NextMegaMoEExperts(DeepGemmMegaMoEExperts):
    """MegaMoE experts loaded from GLM-5.3-Flash's block-FP8 checkpoint.

    The checkpoint stores E4M3 weights with one float32 scale per
    ``weight_block_size`` block. Loading writes each weight shard straight into
    the kernel layout and stages its block scale in ``*_weight_scale_inv``.
    Once a load is complete, the model's ``process_weights_after_loading``
    calls :meth:`finalize_block_fp8_weights`. The shared expert is fused the
    same way.

    SM100 takes power-of-two UE8M0 scales per 1x32, so
    :meth:`requantize_block_fp8` rounds every newly loaded shard in place. SM90
    takes the FP32 block scales as they are.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        weight_block_size: tuple[int, int],
        num_shared_experts: int = 0,
        **kwargs,
    ):
        self.weight_block_size = weight_block_size
        self._num_fused_shared_experts = num_shared_experts
        # Shards whose weight / block scale arrived since the last finalize.
        self._new_weights: set[Shard] = set()
        self._new_scales: set[Shard] = set()
        weight_format = MEGA_MOE_MXFP8
        if current_platform.is_device_capability(90):
            if tuple(weight_block_size) != (128, 128):
                raise NotImplementedError(
                    "GLM-5.3 MegaMoE on SM90 takes 128x128 block-FP8 weights, "
                    f"not {tuple(weight_block_size)} blocks."
                )
            weight_format = MEGA_MOE_FP8_BLOCK_SM90
        super().__init__(vllm_config, weight_format=weight_format, **kwargs)

    def _fusable_shared_experts(
        self, shared_experts: nn.Module | None
    ) -> tuple[int, MegaMoEWeightSpec | None]:
        n = self._num_fused_shared_experts
        return n, self.weight_format.routed if n else None

    def _create_weights(self) -> None:
        super()._create_weights()
        e, h, i = self.num_local_experts, self.hidden_size, self.intermediate_size
        self._register_block_scale("w13", (e, 2 * i, h))
        self._register_block_scale("w2", (e, h, i))
        if self.has_fused_shared_experts:
            s = i * self.num_shared_experts
            self._register_block_scale("shared_l1", (2 * s, h))
            self._register_block_scale("shared_l2", (h, s))

    def _register_block_scale(self, name: str, shape: tuple[int, ...]) -> None:
        """Stage the checkpoint's float32 block scales of ``name``. The kernel's
        ``{name}_weight_scale`` is derived from them and never loaded."""
        *outer, n, k = shape
        block_m, block_k = self.weight_block_size
        scale = torch.zeros(*outer, n // block_m, k // block_k, dtype=torch.float32)
        self.register_parameter(
            f"{name}_weight_scale_inv", nn.Parameter(scale, requires_grad=False)
        )
        getattr(self, f"{name}_weight_scale").weight_loader_numel = 0

    @staticmethod
    def _shard_rows(t: torch.Tensor, shard_id: str) -> torch.Tensor:
        """Rows of ``t`` (gate rows, then up rows, for w13) that hold ``shard_id``."""
        if shard_id == "w2":
            return t
        half = t.shape[0] // 2
        return t[:half] if shard_id == "w1" else t[half:]

    @staticmethod
    def _shared_shard_id(name: str) -> str:
        return "w1" if ".gate_proj." in name else "w3" if ".up_proj." in name else "w2"

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        local_expert_ids = self._map_global_expert_id(expert_id)
        if weight_name.endswith("weight_scale_inv"):
            param_prefix = "w2_" if shard_id == "w2" else "w13_"
            loaded = bool(local_expert_ids) and param_prefix in weight_name
            for local_expert_id in local_expert_ids if loaded else ():
                self._shard_rows(param.data[local_expert_id], shard_id).copy_(
                    loaded_weight
                )
            new = self._new_scales
        else:
            loaded = bool(
                super().weight_loader(
                    param,
                    loaded_weight,
                    weight_name,
                    shard_id,
                    expert_id,
                    return_success=True,
                )
            )
            new = self._new_weights
        if loaded:
            new.update(
                (local_expert_id, shard_id) for local_expert_id in local_expert_ids
            )
        return loaded if return_success else None

    def load_shared_expert_weight(self, name: str, loaded_weight: torch.Tensor) -> str:
        """Load one shared-expert checkpoint tensor; return the parameter name."""
        layer = "shared_l2" if self._shared_shard_id(name) == "w2" else "shared_l1"
        suffix = "weight_scale_inv" if "weight_scale" in name else "weight"
        param = getattr(self, f"{layer}_{suffix}")
        param.weight_loader(param, loaded_weight, name)
        return f"{name.partition('shared_experts.')[0]}experts.{layer}_{suffix}"

    def shared_weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, weight_name: str
    ) -> None:
        shard_id = self._shared_shard_id(weight_name)
        if "weight_scale" in weight_name:
            self._shard_rows(param.data, shard_id).copy_(loaded_weight)
            self._new_scales.add((None, shard_id))
        else:
            self._load_weight(param.data, loaded_weight, shard_id, False)
            self._new_weights.add((None, shard_id))

    def _check_paired_shards(self) -> None:
        unpaired = self._new_weights ^ self._new_scales
        if unpaired:
            raise ValueError(
                f"{self.prefix}: block-FP8 shards (local expert, shard; None is "
                "the shared expert) loaded without their weight or block scale: "
                f"{sorted(unpaired, key=str)}"
            )

    def finalize_block_fp8_weights(self) -> None:
        """Turn every shard loaded since the last call into kernel weights."""
        if self.weight_format == MEGA_MOE_FP8_BLOCK_SM90:
            self._transform_block_fp8_sm90()
        else:
            self.requantize_block_fp8()

    @torch.no_grad()
    def _transform_block_fp8_sm90(self, chunk: int = 8) -> None:
        """Build the SM90 kernel weights and scales with DeepGEMM, at most
        ``chunk`` experts per call, without requantizing.

        A load that skipped ``weight_loader`` (e.g. dummy weights) leaves the
        kernel scales unbuilt, so then every expert is transformed.
        """
        self._check_paired_shards()
        experts = {e for e, _ in self._new_weights if e is not None}
        if self.w13_weight_scale.numel() == 0:
            experts = set(range(self.num_local_experts))
        runs: list[list[int]] = []
        for e in sorted(experts):
            if runs and runs[-1][1] == e and e - runs[-1][0] < chunk:
                runs[-1][1] = e + 1
            else:
                runs.append([e, e + 1])
        groups: list[tuple[str, str, Any]] = [
            ("w13", "w2", slice(start, end)) for start, end in runs
        ]
        if self.has_fused_shared_experts and (
            any(e is None for e, _ in self._new_weights)
            or self.shared_l1_weight_scale.numel() == 0
        ):
            groups.append(("shared_l1", "shared_l2", ...))

        deep_gemm = _import_deep_gemm()
        for l1, l2, index in groups:
            w13 = getattr(self, f"{l1}_weight").data
            w2 = getattr(self, f"{l2}_weight").data
            # Back to checkpoint row order: undo the 8-row gate/up interleave.
            w13_rows = w13[index].unflatten(-2, (-1, 2, 8))
            (w13_kernel, sf13), (w2_kernel, sf2) = (
                deep_gemm.transform_weights_for_mega_moe(
                    (
                        w13_rows.transpose(-4, -3).flatten(-4, -2),
                        getattr(self, f"{l1}_weight_scale_inv").data[index],
                    ),
                    (w2[index], getattr(self, f"{l2}_weight_scale_inv").data[index]),
                    activation=self.activation.name,
                    arch_major=9,
                )
            )
            w13[index].copy_(w13_kernel)
            w2[index].copy_(w2_kernel)
            self._store_kernel_scale(l1, index, sf13)
            self._store_kernel_scale(l2, index, sf2)
        self._new_weights.clear()
        self._new_scales.clear()

    def _store_kernel_scale(self, name: str, index: Any, sf: torch.Tensor) -> None:
        """Write DeepGEMM's kernel scales of experts ``index`` (``...`` for the
        shared expert), keeping their strides (MN-major, TMA-aligned)."""
        param = getattr(self, f"{name}_weight_scale")
        if index is ...:
            param.data = sf
            return
        if param.numel() == 0:
            # Groups are contiguous, so every expert takes sf.stride(0) words.
            storage = torch.zeros(
                self.num_local_experts * sf.stride(0), dtype=sf.dtype, device=sf.device
            )
            param.data = storage.as_strided(
                (self.num_local_experts, *sf.shape[1:]), sf.stride()
            )
        param.data[index].copy_(sf)

    @torch.no_grad()
    def requantize_block_fp8(self, chunk: int = 8) -> None:
        """Round every shard loaded since the last call to UE8M0 1x32 scales,
        in place.

        Runs once a load is complete. A weight update may send a weight and its
        block scale in separate ``load_weights`` calls, but both have to be in
        by then.
        """
        self._check_paired_shards()
        for shard_id in ("w1", "w3", "w2"):
            experts = sorted(
                e for e, s in self._new_weights if s == shard_id and e is not None
            )
            for start in range(0, len(experts), chunk):
                self._requantize_shards(
                    [
                        self._shard_tensors(e, shard_id)
                        for e in experts[start : start + chunk]
                    ],
                    shard_id,
                )
            if (None, shard_id) in self._new_weights:
                self._requantize_shards([self._shard_tensors(None, shard_id)], shard_id)
        self._new_weights.clear()
        self._new_scales.clear()

    def _shard_tensors(
        self, local_expert_id: int | None, shard_id: str
    ) -> tuple[torch.Tensor, ...]:
        """Kernel weight, kernel scale and staged block scale of one routed
        expert, or of the shared expert for ``local_expert_id`` None."""
        if local_expert_id is None:
            name = "shared_l2" if shard_id == "w2" else "shared_l1"
        else:
            name = "w2" if shard_id == "w2" else "w13"
        tensors = tuple(
            getattr(self, f"{name}_{suffix}").data
            for suffix in ("weight", "weight_scale", "weight_scale_inv")
        )
        if local_expert_id is None:
            return tensors
        return tuple(t[local_expert_id] for t in tensors)

    def _requantize_shards(
        self, tensors: list[tuple[torch.Tensor, ...]], shard_id: str
    ) -> None:
        """Requantize ``shard_id`` of each kernel-layout weight in place."""
        # A shard's layout destination is its checkpoint rows; the source is unused.
        raw = torch.stack(
            [
                self._kernel_layout_shard(w, w, shard_id, False)[0].reshape(
                    -1, w.shape[-1]
                )
                for w, _, _ in tensors
            ]
        )
        block_scale = torch.stack(
            [self._shard_rows(s, shard_id) for _, _, s in tensors]
        )
        # Power-of-two block scales requantize exactly; any other is rounded.
        if not torch.equal(block_scale, block_scale.log2().round().exp2()):
            logger.warning_once(
                "GLM-5.3-Flash on the deep_gemm_mega_moe backend: the checkpoint's "
                "%dx%d block-FP8 MoE scales are not all powers of two, so "
                "requantizing them to MXFP8 (UE8M0 scales per 1x32) changes the "
                "expert weights. Use another moe_backend to run the block-FP8 "
                "weights as they are.",
                *self.weight_block_size,
            )
        weight, sf = requant_block_fp8_to_ue8m0(raw, block_scale)
        for (dst, dst_scale, _), w, s in zip(tensors, weight, sf):
            self._load_weight(dst, w, shard_id, False)
            self._load_weight(dst_scale, s, shard_id, True)


Glm5NextMegaMoEExperts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
