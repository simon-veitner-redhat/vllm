# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.v1.watermarking.prfs.philox import (
    _CONTEXT_DOMAIN as _CONTEXT_DOMAIN_VALUE,
)
from vllm.v1.watermarking.prfs.philox import (
    _PHILOX_M0 as _PHILOX_M0_VALUE,
)
from vllm.v1.watermarking.prfs.philox import (
    _PHILOX_M1 as _PHILOX_M1_VALUE,
)
from vllm.v1.watermarking.prfs.philox import (
    _PHILOX_W0 as _PHILOX_W0_VALUE,
)
from vllm.v1.watermarking.prfs.philox import (
    _PHILOX_W1 as _PHILOX_W1_VALUE,
)
from vllm.v1.watermarking.prfs.philox import (
    _TOKEN_DOMAIN as _TOKEN_DOMAIN_VALUE,
)
from vllm.v1.watermarking.prfs.philox import (
    _UINT32_MASK as _UINT32_MASK_VALUE,
)
from vllm.v1.worker.gpu.sample.gumbel import gumbel_block_argmax

if HAS_TRITON:
    from triton.language import math as tl_math
else:
    tl_math = tl

_UINT32_MASK = tl.constexpr(_UINT32_MASK_VALUE) if HAS_TRITON else _UINT32_MASK_VALUE
_PHILOX_M0 = tl.constexpr(_PHILOX_M0_VALUE) if HAS_TRITON else _PHILOX_M0_VALUE
_PHILOX_M1 = tl.constexpr(_PHILOX_M1_VALUE) if HAS_TRITON else _PHILOX_M1_VALUE
_PHILOX_W0 = tl.constexpr(_PHILOX_W0_VALUE) if HAS_TRITON else _PHILOX_W0_VALUE
_PHILOX_W1 = tl.constexpr(_PHILOX_W1_VALUE) if HAS_TRITON else _PHILOX_W1_VALUE
_CONTEXT_DOMAIN = (
    tl.constexpr(_CONTEXT_DOMAIN_VALUE) if HAS_TRITON else _CONTEXT_DOMAIN_VALUE
)
_TOKEN_DOMAIN = tl.constexpr(_TOKEN_DOMAIN_VALUE) if HAS_TRITON else _TOKEN_DOMAIN_VALUE


@triton.jit
def _repeated_context_mask_kernel(
    output_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    req_indices_ptr,
    prompt_lens_ptr,
    total_lens_ptr,
    contexts_ptr,
    context_stride,
    CONTEXT_WIDTH: tl.constexpr,
    MAX_HISTORY: tl.constexpr,
    INCLUDE_PROMPT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    req_idx = tl.load(req_indices_ptr + row).to(tl.int64)
    if req_idx < 0:
        tl.store(output_ptr + row, 0)
        return
    prompt_len = tl.load(prompt_lens_ptr + req_idx)
    total_len = tl.load(total_lens_ptr + req_idx)
    sequence_start = 0 if INCLUDE_PROMPT else prompt_len
    history_len = total_len - sequence_start

    offsets = tl.arange(0, BLOCK)
    repeated = tl.full((), 0, tl.int32)
    scan_start = 0
    if MAX_HISTORY > 0:
        scan_start = tl.maximum(history_len - MAX_HISTORY, 0)
    # Aligned loop bounds let the per-lane offsets fold into immediates.
    aligned_start = (scan_start // BLOCK) * BLOCK
    for block_start in tl.range(aligned_start, history_len, BLOCK):
        previous_pos = block_start + offsets
        in_window = (previous_pos >= scan_start) & (previous_pos < history_len)
        matches = in_window
        for offset in range(CONTEXT_WIDTH):
            context_token = tl.load(contexts_ptr + row * context_stride + offset)
            historical_pos = previous_pos + offset - CONTEXT_WIDTH
            historical_token = tl.load(
                all_token_ids_ptr
                + req_idx * all_token_ids_stride
                + sequence_start
                + tl.maximum(historical_pos, 0),
                mask=in_window & (historical_pos >= 0),
                other=-1,
            )
            matches &= historical_token == context_token
        repeated |= tl.max(matches.to(tl.int32), axis=0)
    tl.store(output_ptr + row, repeated)


@triton.jit
def _watermark_prep_kernel(
    contexts_ptr,
    context_stride,
    skip_mask_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    req_indices_ptr,
    prompt_lens_ptr,
    total_lens_ptr,
    watermarking_ptr,
    temperatures_ptr,
    CONTEXT_WIDTH: tl.constexpr,
    MAX_HISTORY: tl.constexpr,
    INCLUDE_PROMPT: tl.constexpr,
    SCAN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per sampled row: write its context and whether to skip it.

    Fuses the context gather, the repeated-context scan and the skip-mask
    reduction so the sampler issues one launch instead of ~25 per step.
    """
    row = tl.program_id(0).to(tl.int64)
    req_idx = tl.load(req_indices_ptr + row).to(tl.int64)
    valid_req = req_idx >= 0
    safe_req_idx = tl.maximum(req_idx, 0)
    prompt_len = tl.load(prompt_lens_ptr + safe_req_idx, mask=valid_req, other=0)
    total_len = tl.load(total_lens_ptr + safe_req_idx, mask=valid_req, other=0)
    sequence_start = 0 if INCLUDE_PROMPT else prompt_len
    history_len = total_len - sequence_start

    watermarked = valid_req
    if watermarking_ptr is not None:
        enabled = tl.load(watermarking_ptr + safe_req_idx, mask=valid_req, other=0)
        temperature = tl.load(
            temperatures_ptr + safe_req_idx, mask=valid_req, other=0.0
        )
        watermarked = valid_req & (enabled != 0) & (temperature != 0.0)

    # Positions before `sequence_start` and unused rows read as -1.
    row_base = all_token_ids_ptr + safe_req_idx * all_token_ids_stride
    # The scan reloads the context from `context_row`; the barrier below orders
    # the single-lane stores before the all-lane loads.
    context_row = contexts_ptr + row * context_stride
    for offset in tl.static_range(CONTEXT_WIDTH):
        position = total_len - CONTEXT_WIDTH + offset
        context_token = tl.load(
            row_base + tl.maximum(position, 0),
            mask=valid_req & (position >= sequence_start),
            other=-1,
        )
        tl.store(context_row + offset, context_token)
    tl.debug_barrier()

    repeated = tl.full((), 0, tl.int32)
    if SCAN:
        offsets = tl.arange(0, BLOCK)
        # No scan for unwatermarked rows or histories shorter than the context.
        do_scan = watermarked & (history_len >= CONTEXT_WIDTH)
        scan_end = tl.where(do_scan, history_len, 0)
        scan_start = 0
        if MAX_HISTORY > 0:
            scan_start = tl.maximum(scan_end - MAX_HISTORY, 0)
        aligned_start = (scan_start // BLOCK) * BLOCK
        for block_start in tl.range(aligned_start, scan_end, BLOCK):
            previous_pos = block_start + offsets
            in_window = (previous_pos >= scan_start) & (previous_pos < scan_end)
            matches = in_window
            for offset in tl.static_range(CONTEXT_WIDTH):
                context_token = tl.load(context_row + offset)
                historical_pos = previous_pos + offset - CONTEXT_WIDTH
                historical_token = tl.load(
                    row_base + sequence_start + tl.maximum(historical_pos, 0),
                    mask=in_window & (historical_pos >= 0),
                    other=-1,
                )
                matches &= historical_token == context_token
            repeated |= tl.max(matches.to(tl.int32), axis=0)

    if skip_mask_ptr is not None:
        skip = tl.where(watermarked, repeated, 1)
        tl.store(skip_mask_ptr + row, skip != 0)


def _watermark_prep_cpu(
    all_token_ids: torch.Tensor,
    req_indices: torch.Tensor,
    prompt_lens: torch.Tensor,
    total_lens: torch.Tensor,
    context_width: int,
    watermarking: torch.Tensor | None,
    temperatures: torch.Tensor | None,
    max_history: int | None,
    include_prompt: bool,
    scan: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference for `watermark_prep`, and the path taken on CPU."""
    device = all_token_ids.device
    req_indices_64 = req_indices.to(torch.int64)
    valid_reqs = req_indices_64 >= 0
    safe_req_indices = req_indices_64.clamp_min(0)
    total = total_lens[safe_req_indices].to(torch.int64)
    prompt = prompt_lens[safe_req_indices].to(torch.int64)
    sequence_starts = torch.zeros_like(prompt) if include_prompt else prompt
    offsets = torch.arange(-context_width, 0, dtype=torch.int64, device=device)
    positions = total.unsqueeze(-1) + offsets
    valid_positions = valid_reqs.unsqueeze(-1) & (
        positions >= sequence_starts.unsqueeze(-1)
    )
    contexts = all_token_ids[safe_req_indices.unsqueeze(-1), positions.clamp_min(0)]
    contexts = torch.where(valid_positions, contexts, -1)

    if watermarking is None:
        watermarked = valid_reqs.clone()
    else:
        assert temperatures is not None
        watermarked = (
            valid_reqs
            & watermarking[safe_req_indices]
            & (temperatures[safe_req_indices] != 0)
        )
    if scan:
        repeated = _repeated_context_mask_cpu(
            all_token_ids,
            req_indices,
            prompt_lens,
            total_lens,
            contexts,
            max_history,
            include_prompt,
        ).to(device)
        watermarked = watermarked & ~repeated
    return contexts, ~watermarked


def watermark_prep(
    all_token_ids: torch.Tensor,
    req_indices: torch.Tensor,
    prompt_lens: torch.Tensor,
    total_lens: torch.Tensor,
    context_width: int,
    *,
    watermarking: torch.Tensor | None = None,
    temperatures: torch.Tensor | None = None,
    max_history: int | None = None,
    include_prompt: bool = False,
    scan: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(contexts, skip_mask)` for the rows in `req_indices`.

    `contexts` holds the last `context_width` tokens of each row, -1 padded
    before the scope start. `skip_mask` marks rows sampled without the
    watermark: unused, opted out, temperature 0, or (with `scan`) repeated
    context. With `watermarking=None` only contexts are computed.
    """
    num_rows = len(req_indices)
    if all_token_ids.device.type == "cpu":
        return _watermark_prep_cpu(
            all_token_ids,
            req_indices,
            prompt_lens,
            total_lens,
            context_width,
            watermarking,
            temperatures,
            max_history,
            include_prompt,
            scan,
        )

    contexts = torch.empty(
        (num_rows, context_width),
        dtype=all_token_ids.dtype,
        device=all_token_ids.device,
    )
    skip_mask = torch.empty(num_rows, dtype=torch.bool, device=all_token_ids.device)
    _watermark_prep_kernel[(num_rows,)](
        contexts,
        contexts.stride(0),
        skip_mask,
        all_token_ids,
        all_token_ids.stride(0),
        req_indices,
        prompt_lens,
        total_lens,
        watermarking,
        temperatures,
        CONTEXT_WIDTH=context_width,
        MAX_HISTORY=0 if max_history is None else max_history,
        INCLUDE_PROMPT=include_prompt,
        SCAN=scan,
        BLOCK=512,
    )
    return contexts, skip_mask


def _repeated_context_mask_cpu(
    all_token_ids: torch.Tensor,
    req_indices: torch.Tensor,
    prompt_lens: torch.Tensor,
    total_lens: torch.Tensor,
    contexts: torch.Tensor,
    max_history: int | None = None,
    include_prompt: bool = False,
) -> torch.Tensor:
    """Reference implementation; the parity tests check the Triton kernel against it."""
    repeated = torch.zeros(len(req_indices), dtype=torch.bool)
    for row, req_idx_tensor in enumerate(req_indices):
        req_idx = int(req_idx_tensor)
        if req_idx < 0:
            continue
        prompt_len = int(prompt_lens[req_idx])
        total_len = int(total_lens[req_idx])
        sequence_start = 0 if include_prompt else prompt_len
        history_tokens = all_token_ids[req_idx, sequence_start:total_len].tolist()
        prefix = [-1] * contexts.shape[-1]
        current_context = tuple(contexts[row].tolist())
        history_start = (
            max(0, len(history_tokens) - max_history) if max_history is not None else 0
        )
        for history_pos, token_id in enumerate(history_tokens):
            if history_pos >= history_start and (
                tuple(prefix[-contexts.shape[-1] :]) == current_context
            ):
                repeated[row] = True
                break
            prefix.append(token_id)
    return repeated


def repeated_context_mask(
    all_token_ids: torch.Tensor,
    req_indices: torch.Tensor,
    prompt_lens: torch.Tensor,
    total_lens: torch.Tensor,
    contexts: torch.Tensor,
    max_history: int | None = None,
    include_prompt: bool = False,
) -> torch.Tensor:
    """Return, per row, whether the row's context already occurred in its history.

    Args:
        all_token_ids: `[max_num_reqs, max_model_len]` token ids of every request.
        req_indices: Request slot per sampled row; -1 marks a padding row, which
            is reported as not repeated.
        prompt_lens: Prompt length per request slot.
        total_lens: Prompt plus generated length per request slot.
        contexts: `[num_rows, context_width]` context per row, padded with -1
            before the start of the scanned history.
        max_history: Number of most recent history positions searched, or
            ``None`` for all of them. The window compared at each position
            reaches `context_width` tokens further back.
        include_prompt: Search the prompt as well as the generated tokens.
    """
    if all_token_ids.device.type == "cpu":
        return _repeated_context_mask_cpu(
            all_token_ids,
            req_indices,
            prompt_lens,
            total_lens,
            contexts,
            max_history,
            include_prompt,
        )

    if contexts.stride(-1) != 1:
        contexts = contexts.contiguous()
    repeated = torch.empty(len(req_indices), dtype=torch.bool, device=contexts.device)
    _repeated_context_mask_kernel[(len(req_indices),)](
        repeated,
        all_token_ids,
        all_token_ids.stride(0),
        req_indices,
        prompt_lens,
        total_lens,
        contexts,
        contexts.stride(0),
        CONTEXT_WIDTH=contexts.shape[-1],
        MAX_HISTORY=0 if max_history is None else max_history,
        INCLUDE_PROMPT=include_prompt,
        BLOCK=512,
    )
    return repeated


@triton.jit
def _mulhilo32(multiplier: tl.constexpr, value):
    high = tl_math.umulhi(multiplier, value)
    low = tl.mul(multiplier, value, sanitize_overflow=False)
    return high, low


@triton.jit
def _philox4x32_10(counter_0, counter_1, counter_2, counter_3, key_0, key_1):
    for round_index in range(10):
        high_0, low_0 = _mulhilo32(_PHILOX_M0, counter_0)
        high_1, low_1 = _mulhilo32(_PHILOX_M1, counter_2)
        counter_0, counter_1, counter_2, counter_3 = (
            (high_1 ^ counter_1 ^ key_0) & _UINT32_MASK,
            low_1,
            (high_0 ^ counter_3 ^ key_1) & _UINT32_MASK,
            low_0,
        )
        if round_index != 9:
            key_0 = (key_0 + _PHILOX_W0) & _UINT32_MASK
            key_1 = (key_1 + _PHILOX_W1) & _UINT32_MASK
    return counter_0, counter_1, counter_2, counter_3


@triton.jit
def _uint32_to_uniform(value):
    mantissa = (value & _UINT32_MASK) >> 8
    scaled = (mantissa + 1).to(tl.float32) * 2**-24
    return (scaled.to(tl.uint32, bitcast=True) - 1).to(tl.float32, bitcast=True)


@triton.jit
def _gumbel_value(logits_ptr, output, mask):
    logits = tl.load(logits_ptr, mask=mask, other=float("-inf")).to(tl.float32)
    logits = tl.where(logits != logits, float("-inf"), logits)
    uniform = _uint32_to_uniform(output)
    return logits - tl.log(-tl.log(uniform))


@triton.jit
def _select_max(best_value, best_token, candidate_value, candidate):
    take_candidate = candidate_value > best_value
    return (
        tl.where(take_candidate, candidate_value, best_value),
        tl.where(take_candidate, candidate, best_token),
    )


@triton.jit(do_not_specialize=["key_0_value", "key_1_value"])
def _philox_gumbel_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    logits_ptr,
    logits_stride,
    contexts_ptr,
    context_stride,
    skip_mask_ptr,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    key_0_value,
    key_1_value,
    vocab_size,
    CONTEXT_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block_index = tl.program_id(1)
    groups = block_index * (BLOCK_SIZE // 4) + tl.arange(0, BLOCK_SIZE // 4)

    if skip_mask_ptr is not None:
        skip_watermark = tl.load(skip_mask_ptr + row)
        if skip_watermark:
            candidate = block_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = candidate < vocab_size
            logits_row = logits_ptr + row * logits_stride
            logits = tl.load(logits_row + candidate, mask=mask, other=float("-inf")).to(
                tl.float32
            )
            # Skipped rows must match gumbel_sample bit for bit, so read the
            # sampling state through the same helper it uses.
            value, index = gumbel_block_argmax(
                logits,
                candidate,
                mask,
                row,
                expanded_idx_mapping_ptr,
                temp_ptr,
                seeds_ptr,
                pos_ptr,
                None,
                0,
                0,
                None,
                vocab_size,
                IS_DRAFTING=False,
                APPLY_TEMPERATURE=False,
                USE_FP64=USE_FP64,
            )
            token_id = block_index * BLOCK_SIZE + index
            tl.store(
                local_argmax_ptr + row * local_argmax_stride + block_index,
                token_id,
            )
            tl.store(local_max_ptr + row * local_max_stride + block_index, value)
            return

    key_0 = key_0_value.to(tl.uint32)
    key_1 = key_1_value.to(tl.uint32)
    state_0 = tl.full((), _CONTEXT_DOMAIN, tl.uint32)
    state_1 = tl.full((), CONTEXT_WIDTH, tl.uint32)
    state_2 = tl.full((), 0, tl.uint32)
    state_3 = tl.full((), 0, tl.uint32)
    for offset in range(0, CONTEXT_WIDTH, 4):
        context_0 = tl.load(contexts_ptr + row * context_stride + offset).to(tl.uint32)
        if offset + 1 < CONTEXT_WIDTH:
            context_1 = tl.load(contexts_ptr + row * context_stride + offset + 1).to(
                tl.uint32
            )
        else:
            context_1 = tl.full((), _UINT32_MASK - 1, tl.uint32)
        if offset + 2 < CONTEXT_WIDTH:
            context_2 = tl.load(contexts_ptr + row * context_stride + offset + 2).to(
                tl.uint32
            )
        else:
            context_2 = tl.full((), _UINT32_MASK - 2, tl.uint32)
        if offset + 3 < CONTEXT_WIDTH:
            context_3 = tl.load(contexts_ptr + row * context_stride + offset + 3).to(
                tl.uint32
            )
        else:
            context_3 = tl.full((), _UINT32_MASK - 3, tl.uint32)
        state_0, state_1, state_2, state_3 = _philox4x32_10(
            (state_0 ^ context_0) & _UINT32_MASK,
            (state_1 ^ context_1) & _UINT32_MASK,
            (state_2 ^ context_2) & _UINT32_MASK,
            (state_3 ^ context_3) & _UINT32_MASK,
            (key_0 ^ offset) & _UINT32_MASK,
            key_1,
        )

    candidate_words = groups.to(tl.uint32)
    vector_zero = candidate_words * 0
    output_0, output_1, output_2, output_3 = _philox4x32_10(
        candidate_words & _UINT32_MASK,
        state_0 + vector_zero,
        state_1 + vector_zero,
        state_2 + vector_zero,
        ((key_0 ^ state_3) & _UINT32_MASK) + vector_zero,
        ((key_1 ^ _TOKEN_DOMAIN) & _UINT32_MASK) + vector_zero,
    )
    candidate_0 = groups * 4
    candidate_1 = candidate_0 + 1
    candidate_2 = candidate_0 + 2
    candidate_3 = candidate_0 + 3
    logits_row = logits_ptr + row * logits_stride
    value_0 = _gumbel_value(
        logits_row + candidate_0, output_0, candidate_0 < vocab_size
    )
    value_1 = _gumbel_value(
        logits_row + candidate_1, output_1, candidate_1 < vocab_size
    )
    value_2 = _gumbel_value(
        logits_row + candidate_2, output_2, candidate_2 < vocab_size
    )
    value_3 = _gumbel_value(
        logits_row + candidate_3, output_3, candidate_3 < vocab_size
    )
    best_value = value_0
    best_token = candidate_0
    best_value, best_token = _select_max(best_value, best_token, value_1, candidate_1)
    best_value, best_token = _select_max(best_value, best_token, value_2, candidate_2)
    best_value, best_token = _select_max(best_value, best_token, value_3, candidate_3)
    value = tl.max(best_value, axis=0)
    token_id = tl.min(tl.where(best_value == value, best_token, vocab_size), axis=0)
    tl.store(local_argmax_ptr + row * local_argmax_stride + block_index, token_id)
    tl.store(local_max_ptr + row * local_max_stride + block_index, value)


def philox_gumbel_sample(
    logits: torch.Tensor,
    contexts: torch.Tensor,
    key: int,
    *,
    skip_mask: torch.Tensor | None = None,
    expanded_idx_mapping: torch.Tensor | None = None,
    temperatures: torch.Tensor | None = None,
    seeds: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    use_fp64: bool = False,
) -> torch.Tensor:
    sampling_state = (expanded_idx_mapping, temperatures, seeds, positions)
    if skip_mask is None and any(value is not None for value in sampling_state):
        raise ValueError("sampling state requires skip_mask")
    if skip_mask is not None and any(value is None for value in sampling_state):
        raise ValueError("skip_mask requires complete sampling state")

    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    if contexts.stride(-1) != 1:
        contexts = contexts.contiguous()
    if skip_mask is not None:
        skip_mask = skip_mask.contiguous()
    if expanded_idx_mapping is not None:
        expanded_idx_mapping = expanded_idx_mapping.contiguous()
    if positions is not None:
        positions = positions.contiguous()
    num_tokens, vocab_size = logits.shape
    block_size = 1024
    num_blocks = triton.cdiv(vocab_size, block_size)
    local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
    local_max = logits.new_empty(
        num_tokens,
        num_blocks,
        dtype=torch.float64 if use_fp64 else torch.float32,
    )
    _philox_gumbel_kernel[(num_tokens, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        logits,
        logits.stride(0),
        contexts,
        contexts.stride(0),
        skip_mask,
        expanded_idx_mapping,
        seeds,
        positions,
        temperatures,
        key & _UINT32_MASK_VALUE,
        key >> 32,
        vocab_size,
        CONTEXT_WIDTH=contexts.shape[-1],
        BLOCK_SIZE=block_size,
        USE_FP64=use_fp64,
    )
    max_block_index = local_max.argmax(dim=-1, keepdim=True)
    return local_argmax.gather(dim=-1, index=max_block_index).view(-1)
