"""HTTP surface: an IO processor plugin for ``POST /pooling``.

Request body: ``{"data": {"prefix": <text or token ids>, "candidates": [<text
or token ids>, ...]}}``. The plugin packs one prompt, sets the layout in
``PoolingParams.extra_kwargs`` and returns one mean row per candidate. No
field is added to a vLLM request type.
"""

from vllm.inputs import TokensPrompt
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.pooling_params import PoolingParams

from .layout import EXTRA_KEY, check_lengths


def get_io_processor() -> str:
    return "vllm_segment_layout.io_processor.SegmentLayoutIOProcessor"


def _is_ids(value) -> bool:
    return isinstance(value, list) and all(type(t) is int for t in value)


class SegmentLayoutIOProcessor(IOProcessor[dict, dict]):
    def __init__(self, vllm_config, renderer):
        super().__init__(vllm_config, renderer)
        self.renderer = renderer
        hf_config = vllm_config.model_config.hf_config
        self.max_positions = getattr(hf_config, "max_position_embeddings", None)
        # MISSING HOOK: merge_pooling_params() gets no request data. vLLM calls
        # pre_process() and then merge_pooling_params() with no await between
        # them, so the lengths of the request in flight are parked here.
        self._lengths: list[int] | None = None

    def parse_data(self, data: object) -> dict:
        if not isinstance(data, dict) or set(data) != {"prefix", "candidates"}:
            raise ValueError("data must be {'prefix': ..., 'candidates': [...]}")
        candidates = data["candidates"]
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("candidates must be a non-empty list")
        for item in (data["prefix"], *candidates):
            if not (isinstance(item, str) and item) and not (_is_ids(item) and item):
                raise ValueError("prefix and candidates are text or token id lists")
        return data

    def pre_process(self, prompt: dict, request_id=None, **kwargs):
        tokenizer = self.renderer.get_tokenizer()

        def ids(item, first: bool) -> list[int]:
            if not isinstance(item, str):
                return list(item)
            token_ids = tokenizer.encode(item, add_special_tokens=True)
            # Text prefix: [CLS] prefix [SEP]. Text candidate: candidate [SEP].
            return token_ids if first else token_ids[1:]

        prefix = ids(prompt["prefix"], True)
        candidates = [ids(c, False) for c in prompt["candidates"]]
        token_ids = prefix + [t for c in candidates for t in c]
        self._lengths = check_lengths(
            [len(prefix)] + [len(c) for c in candidates],
            len(token_ids),
            self.max_positions,
        )
        return TokensPrompt(prompt_token_ids=token_ids)

    def merge_pooling_params(self, params: PoolingParams | None = None):
        params = params or PoolingParams()
        params.task = "token_embed"
        params.extra_kwargs = {EXTRA_KEY: self._lengths}
        return params

    def post_process(self, model_output, request_id=None, **kwargs) -> dict:
        rows = model_output[0].outputs.data
        return {"num_candidates": len(rows), "data": rows.float().tolist()}
