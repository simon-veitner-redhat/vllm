# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BERT embedding model with a segment attention layout: one forward for a
shared prefix and several candidates.

A request opts in with ``PoolingParams(extra_kwargs={"segment_lengths": [P, C1,
...]})`` on the ``token_embed`` task. The prefix sees itself, each candidate
sees the prefix and itself, candidate positions restart at P, and the output
has one mean row per candidate.

Everything is owned by this file: the model state (``get_model_state_cls``),
the attention backends (``Attention(attn_backend=...)``) and the pooler.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vllm.config import get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.encoder_only_attention import (
    create_encoder_only_attention_backend,
)
from vllm.model_executor.layers.pooler.seqwise import pooler_for_embed
from vllm.model_executor.layers.pooler.special import DispatchPooler
from vllm.model_executor.layers.pooler.tokwise import (
    TokenPooler,
    pooler_for_token_embed,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionType,
    subclass_attention_backend,
    subclass_attention_backend_with_overrides,
)
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
)
from vllm.v1.attention.backends.flex_attention import (
    FlexAttentionBackend,
    FlexAttentionMetadataBuilder,
)
from vllm.v1.worker.gpu.model_states.encoder_only import EncoderOnlyModelState

from .bert import BertEmbeddingModel

logger = init_logger(__name__)

EXTRA_KEY = "segment_lengths"
MAX_SEGMENTS = 4096


def parse_lengths(value: Any, prompt_len: int | None = None) -> list[int] | None:
    """Return the validated lengths, or None when ``value`` is not a usable layout.

    The worker must not raise on a bad request, so every consumer (model state
    and pooler) calls this and treats None as "ordinary request".
    """
    if (
        not isinstance(value, (list, tuple))
        or not 2 <= len(value) <= MAX_SEGMENTS
        or any(type(n) is not int or n <= 0 for n in value)
    ):
        return None
    if prompt_len is not None and sum(value) != prompt_len:
        return None
    return list(value)


@dataclass
class SegmentLayout:
    """Per-request rows, built once when the request is added."""

    lengths: list[int]
    positions: np.ndarray  # [L] int64, candidates restart at P
    segment: np.ndarray  # [L] int32, 0 for the prefix, k for candidate k
    q_lens: np.ndarray  # [N + 1] int32, query rows per segment
    k_lens: np.ndarray  # [N + 1] int32, key rows per segment (P, P + Ck)
    kv_index: np.ndarray  # [sum(k_lens)] int64, request-local key rows per segment

    @property
    def total(self) -> int:
        return int(self.q_lens.sum())

    @classmethod
    def from_lengths(cls, lengths: list[int]) -> "SegmentLayout":
        q_lens = np.asarray(lengths, dtype=np.int32)
        prefix = int(q_lens[0])
        starts = np.concatenate(([0], np.cumsum(q_lens)[:-1]))
        segment = np.repeat(np.arange(len(lengths), dtype=np.int32), q_lens)
        local = np.arange(int(q_lens.sum()), dtype=np.int64) - np.repeat(starts, q_lens)
        positions = np.where(segment == 0, local, prefix + local)
        k_lens = q_lens.copy()
        k_lens[1:] += prefix
        prefix_rows = np.arange(prefix, dtype=np.int64)
        kv_index = [prefix_rows]
        for start, length in zip(starts[1:], q_lens[1:]):
            kv_index += [prefix_rows, np.arange(start, start + length, dtype=np.int64)]
        return cls(
            lengths, positions, segment, q_lens, k_lens, np.concatenate(kv_index)
        )


@dataclass
class StepRows:
    """One engine step with at least one layout request, as CPU arrays over the
    packed batch of ``num_tokens`` tokens."""

    num_tokens: int
    # (start, layout) of each layout request in the packed batch
    hits: list[tuple[int, SegmentLayout]]
    # Mask form: q sees kv iff own[kv] == own[q] or own[kv] == vis[q].
    own: np.ndarray  # [T] int32, id of the token's segment, unique in the batch
    vis: np.ndarray  # [T] int32, id of the segment the token also sees (its prefix)
    # Varlen form: segment s has q rows cu_q[s]:cu_q[s+1] and the key rows
    # kv_index[cu_k[s]:cu_k[s+1]].
    cu_q: np.ndarray  # [S + 1] int32
    cu_k: np.ndarray  # [S + 1] int32
    kv_index: np.ndarray  # [cu_k[-1]] int64
    max_q: int
    max_k: int


def build_step(
    query_start_loc: np.ndarray, layouts: list[SegmentLayout | None]
) -> StepRows | None:
    """``query_start_loc`` is [num_reqs + 1]; ``layouts[i]`` is request i's layout
    or None. A request without a layout is one segment that sees itself."""
    if not any(layout is not None for layout in layouts):
        return None
    num_reqs = len(layouts)
    starts = query_start_loc[: num_reqs + 1].astype(np.int64)
    counts = np.diff(starts)
    num_tokens = int(starts[-1])
    own = np.repeat(np.arange(num_reqs, dtype=np.int32), counts)
    vis = own.copy()
    hits = []
    q_lens, k_lens, kv_index = [], [], []
    next_id = num_reqs
    for i, layout in enumerate(layouts):
        start, count = int(starts[i]), int(counts[i])
        if layout is None:
            q_lens.append(counts[i : i + 1].astype(np.int32))
            k_lens.append(q_lens[-1])
            kv_index.append(np.arange(start, start + count, dtype=np.int64))
            continue
        if layout.total != count:
            raise ValueError(
                f"layout covers {layout.total} tokens, request has {count} scheduled"
            )
        hits.append((start, layout))
        own[start : start + count] = layout.segment + next_id
        vis[start : start + count] = next_id
        next_id += len(layout.lengths)
        q_lens.append(layout.q_lens)
        k_lens.append(layout.k_lens)
        kv_index.append(layout.kv_index + start)
    q_all, k_all = np.concatenate(q_lens), np.concatenate(k_lens)
    zero = np.zeros(1, dtype=np.int32)
    return StepRows(
        num_tokens=num_tokens,
        hits=hits,
        own=own,
        vis=vis,
        cu_q=np.concatenate((zero, np.cumsum(q_all, dtype=np.int32))),
        cu_k=np.concatenate((zero, np.cumsum(k_all, dtype=np.int32))),
        kv_index=np.concatenate(kv_index),
        max_q=int(q_all.max()),
        max_k=int(k_all.max()),
    )


@dataclass
class VarlenRows:
    num_tokens: int
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    kv_index: torch.Tensor
    max_q: int
    max_k: int


class StepContext:
    """Rows of the current engine step. Written by the model state in
    ``prepare_attn``, read by the backends of the same model."""

    def __init__(self) -> None:
        self.rows: StepRows | None = None
        self.varlen: VarlenRows | None = None

    def set(self, rows: StepRows | None, device: torch.device, varlen: bool) -> None:
        self.rows = rows
        self.varlen = None
        if rows is not None and varlen:
            self.varlen = VarlenRows(
                num_tokens=rows.num_tokens,
                cu_q=torch.from_numpy(rows.cu_q).to(device),
                cu_k=torch.from_numpy(rows.cu_k).to(device),
                kv_index=torch.from_numpy(rows.kv_index).to(device),
                max_q=rows.max_q,
                max_k=rows.max_k,
            )


def _context_of(vllm_config, layer_names) -> StepContext:
    layers = get_layers_from_vllm_config(vllm_config, Attention, layer_names)
    return next(iter(layers.values())).segment_step


class SegmentFlexBuilder(FlexAttentionMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.segment_step = _context_of(vllm_config, layer_names)
        # Full capacity and persistent: the mask closure keeps one shape, so a
        # new batch size does not recompile it.
        num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self._own = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        self._vis = torch.zeros(num_tokens, dtype=torch.int32, device=device)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        rows = self.segment_step.rows
        if rows is None or rows.num_tokens != metadata.num_actual_tokens:
            return metadata  # the stock mask, untouched

        own, vis = self._own, self._vis
        own[: rows.num_tokens].copy_(torch.from_numpy(rows.own))
        vis[: rows.num_tokens].copy_(torch.from_numpy(rows.vis))

        def segment_mask_mod(b, h, q_idx, kv_idx):
            kv_own = own[kv_idx]
            return (kv_own == own[q_idx]) | (kv_own == vis[q_idx])

        # COST: super().build() has already built the stock block mask, because
        # FlexAttentionMetadataBuilder.build names FlexAttentionMetadata and
        # builds the mask in one go. On a layout step the mask is built twice.
        metadata.get_mask_mod = lambda: segment_mask_mod
        metadata.mask_mod = segment_mask_mod
        metadata.block_mask = metadata.build_block_mask()
        return metadata


class SegmentVarlenImpl(FlashAttentionImpl):
    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        rows = layer.segment_step.varlen
        if (
            rows is None
            or attn_metadata is None
            or rows.num_tokens != attn_metadata.num_actual_tokens
        ):
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )
        num_tokens = rows.num_tokens
        descale_shape = (rows.cu_q.shape[0] - 1, self.num_kv_heads)
        # COPY of the call in FlashAttentionImpl._forward_encoder_attention
        # (flash_attn.py:1705-1731), with the key rows of each segment gathered
        # and their own cu_seqlens. The stock method takes one query_start_loc
        # for both sides, so it cannot be reused.
        flash_attn_varlen_func(
            q=query[:num_tokens],
            k=key.index_select(0, rows.kv_index),
            v=value.index_select(0, rows.kv_index),
            out=output[:num_tokens],
            cu_seqlens_q=rows.cu_q,
            cu_seqlens_k=rows.cu_k,
            max_seqlen_q=rows.max_q,
            max_seqlen_k=rows.max_k,
            softmax_scale=self.scale,
            causal=False,
            alibi_slopes=self.alibi_slopes,
            window_size=list(self.sliding_window),
            softcap=self.logits_soft_cap,
            fa_version=self.vllm_flash_attn_version,
            q_descale=layer._q_scale.expand(descale_shape)
            if self.supports_quant_query_input
            else None,
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            num_splits=1 if self.batch_invariant_enabled or self.fa4_hd256 else 0,
            s_aux=self.sinks,
        )
        return output


def segment_backend(kind: str) -> type[AttentionBackend]:
    """``flex`` or ``varlen``, wrapped like EncoderOnlyAttention wraps its backend
    (the wrapper's builder forces ``causal=False``)."""
    if kind == "flex":
        backend = subclass_attention_backend(
            "SegmentLayout_", FlexAttentionBackend, SegmentFlexBuilder
        )
    elif kind == "varlen":
        backend = subclass_attention_backend_with_overrides(
            "SegmentLayout_",
            FlashAttentionBackend,
            {"get_impl_cls": staticmethod(lambda: SegmentVarlenImpl)},
        )
    else:
        raise ValueError(f"segment_layout_backend must be flex or varlen, got {kind!r}")
    return create_encoder_only_attention_backend(backend)


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
