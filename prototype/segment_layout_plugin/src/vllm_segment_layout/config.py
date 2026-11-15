"""Startup checks for the segment-layout model (the 63114-style config hook)."""

from vllm.model_executor.models.config import VerifyAndUpdateConfig


class SegmentLayoutConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config) -> None:
        model_config = vllm_config.model_config
        if model_config.runner_type != "pooling":
            raise ValueError("BertSegmentLayoutModel is a pooling model")
        kind = getattr(model_config.hf_config, "segment_layout_backend", "auto")
        if kind not in ("auto", "flex", "varlen"):
            raise ValueError(
                f"segment_layout_backend must be auto, flex or varlen, got {kind!r}"
            )
        # No CUDA graph check here: vLLM already forces PIECEWISE for pooling
        # models (config/vllm.py:1997-2007). The backends read the step's rows
        # outside the captured pieces, so PIECEWISE and eager are both fine.
