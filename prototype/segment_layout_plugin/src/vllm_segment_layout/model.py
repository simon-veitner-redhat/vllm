"""BERT embedding model with a segment attention layout, in model-owned code.

Everything the feature needs sits behind three hooks vLLM already has:
``get_model_state_cls()`` (per-request state, positions), ``Attention(
attn_backend=...)`` (the mask) and ``_build_pooler`` (the readout). A request
opts in with ``PoolingParams(extra_kwargs={"segment_lengths": [P, C1, ...]})``
on the ``token_embed`` task.
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.pooler.seqwise import pooler_for_embed
from vllm.model_executor.layers.pooler.special import DispatchPooler
from vllm.model_executor.layers.pooler.tokwise import (
    TokenPooler,
    pooler_for_token_embed,
)
from vllm.model_executor.models.bert import BertEmbeddingModel
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.worker.gpu.model_states.encoder_only import EncoderOnlyModelState

from .backends import StepContext, segment_backend
from .layout import EXTRA_KEY, SegmentLayout, build_step, parse_lengths

# Under the "vllm" logger so the lines reach vLLM's handlers.
logger = init_logger("vllm.segment_layout")


def _lengths_of(pooling_params, prompt_len: int | None) -> list[int] | None:
    extra = pooling_params.extra_kwargs if pooling_params is not None else None
    if not extra or EXTRA_KEY not in extra:
        return None
    return parse_lengths(extra[EXTRA_KEY], prompt_len)


class SegmentLayoutModelState(EncoderOnlyModelState):
    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        self._step: StepContext = model.segment_step
        self._varlen: bool = model.segment_backend_kind == "varlen"
        self._max_positions: int = model.config.max_position_embeddings
        self._layouts: dict[str, SegmentLayout] = {}
        # The compiled BertModel reads `positions` inside a CUDA graph, which
        # replays the address it saw at capture. So the override lives in one
        # persistent buffer that is also handed out for capture
        # (prepare_dummy_inputs), never in a fresh tensor.
        self._positions = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )

    def add_request(self, req_index, new_req_data):
        super().add_request(req_index, new_req_data)
        params = new_req_data.pooling_params
        if params is None or not params.extra_kwargs:
            return
        if EXTRA_KEY not in params.extra_kwargs:
            return
        token_ids = new_req_data.prompt_token_ids
        lengths = _lengths_of(params, None if token_ids is None else len(token_ids))
        if lengths is None or lengths[0] + max(lengths[1:]) > self._max_positions:
            # A raise here would take the engine down. The request is served as
            # an ordinary read; the pooler applies the same rule.
            logger.warning(
                "Request %s: unusable %s, served without a segment layout",
                new_req_data.req_id,
                EXTRA_KEY,
            )
            params.extra_kwargs.pop(EXTRA_KEY)
            return
        self._layouts[new_req_data.req_id] = SegmentLayout.from_lengths(lengths)

    def remove_request(self, req_id):
        super().remove_request(req_id)
        self._layouts.pop(req_id, None)

    def prepare_attn(
        self,
        input_batch,
        cudagraph_mode,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=False,
        ubatch_idx=0,
        model_specific_attn_metadata=None,
    ):
        rows = None
        if self._layouts and not for_capture:
            rows = build_step(
                input_batch.query_start_loc_np,
                [self._layouts.get(req_id) for req_id in input_batch.req_ids],
            )
        self._step.set(rows, self.device, self._varlen)
        return super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture,
            ubatch_idx,
            model_specific_attn_metadata,
        )

    def prepare_inputs(self, input_batch, req_states):
        model_inputs = super().prepare_inputs(input_batch, req_states)
        positions = self._positions[: input_batch.num_tokens_after_padding]
        positions.copy_(model_inputs.get("positions", input_batch.positions))
        rows = self._step.rows
        if rows is not None and rows.num_tokens == input_batch.num_tokens:
            for start, layout in rows.hits:
                positions[start : start + layout.total] = torch.from_numpy(
                    layout.positions
                ).to(self.device)
        model_inputs["positions"] = positions
        return model_inputs

    def prepare_dummy_inputs(self, num_reqs, num_tokens):
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        model_inputs["positions"] = self._positions[:num_tokens]
        return model_inputs


class SegmentTokenPooler(TokenPooler):
    """``token_embed`` pooler. A layout request returns one row per candidate:
    the mean of its token states (FP32 accumulation). Other requests, and
    batches without a layout, take the stock path."""

    def forward(self, hidden_states, pooling_metadata):
        params = pooling_metadata.pooling_params
        layouts = [
            _lengths_of(param, int(prompt_len))
            for param, prompt_len in zip(params, pooling_metadata.prompt_lens)
        ]
        if not any(lengths is not None for lengths in layouts):
            return super().forward(hidden_states, pooling_metadata)

        outputs = []
        rows = self.pooling(hidden_states, pooling_metadata)
        for data, param, lengths in zip(rows, params, layouts, strict=True):
            if data is not None and lengths is not None:
                candidates = data[lengths[0] :].split(lengths[1:])
                means = torch.stack(
                    [c.mean(dim=0, dtype=torch.float32) for c in candidates]
                )
                data = means if self.head.head_dtype is not None else means.to(data.dtype)
            outputs.append(self.head.forward_chunk(data, param))
        return outputs


class BertSegmentLayoutModel(BertEmbeddingModel):
    """Stock BERT weights and layers. Only the attention backend, the model
    state and the ``token_embed`` pooler are its own."""

    def _build_model(self, vllm_config, prefix=""):
        model = super()._build_model(vllm_config, prefix)

        kind = getattr(self.config, "segment_layout_backend", "auto")
        if kind == "auto":
            fa_dtypes = FlashAttentionBackend.supported_dtypes
            kind = "varlen" if vllm_config.model_config.dtype in fa_dtypes else "flex"
        self.segment_backend_kind = kind
        self.segment_step = StepContext()
        backend = segment_backend(kind)

        # MISSING HOOK: BertSelfAttention builds EncoderOnlyAttention, which
        # picks its backend itself (encoder_only_attention.py:70-77) and takes
        # no `attn_backend`. So each layer is rebuilt as a plain Attention with
        # the model's backend, under the same layer name.
        layers = vllm_config.compilation_config.static_forward_context
        for bert_layer in model.encoder.layer:
            self_attn = bert_layer.attention.self
            name = self_attn.attn.layer_name
            del layers[name]
            self_attn.attn = Attention(
                num_heads=self_attn.num_heads,
                head_size=self_attn.head_dim,
                scale=self_attn.scaling,
                num_kv_heads=self_attn.num_kv_heads,
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
                prefix=name,
                attn_type=AttentionType.ENCODER_ONLY,
                attn_backend=backend,
            )
            self_attn.attn.segment_step = self.segment_step
        logger.info(
            "Segment layout: %s backend on %d layers", kind, len(model.encoder.layer)
        )
        return model

    def _build_pooler(self, pooler_config):
        stock = pooler_for_token_embed(pooler_config)
        return DispatchPooler(
            {
                "token_embed": SegmentTokenPooler(stock.pooling, stock.head),
                "embed": pooler_for_embed(pooler_config),
            }
        )

    def get_model_state_cls(self):
        return SegmentLayoutModelState
