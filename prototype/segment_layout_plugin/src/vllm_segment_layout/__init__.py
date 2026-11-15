"""vLLM plugin: segment attention layout (one-pass multi-candidate scoring) for
BERT-family encoders, with no change to vLLM itself.

Select it for a stock checkpoint with
``hf_overrides={"architectures": ["BertSegmentLayoutModel"]}``.
"""

ARCH = "BertSegmentLayoutModel"


def register() -> None:
    """``vllm.general_plugins`` entry point. Runs in every vLLM process, so it
    only registers names; the model module is imported lazily."""
    from vllm import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    from .config import SegmentLayoutConfig

    if ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            ARCH, "vllm_segment_layout.model:BertSegmentLayoutModel"
        )
    # MISSING HOOK: MODELS_CONFIG_MAP is a plain dict with no register call.
    MODELS_CONFIG_MAP.setdefault(ARCH, SegmentLayoutConfig)
