# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-supplied attention blocks for block-bidirectional checkpoints.

Some checkpoints (Base Labs mammoth-1 and elephant-1) were trained with a
``block_bidirectional`` mask on their full-attention layers. The prompt is cut
into consecutive blocks, and a query sees a key when the key comes earlier
(causal) or when both sit in the same block. With a prefix block and a
question block, the prefix is bidirectional over itself and never sees the
question, and the question sees the prefix and all of itself.

A request carries its blocks in ``SamplingParams.extra_args["attention_blocks"]``
as inclusive ``[start, end]`` token ranges. Three places act on them:

- The model runners pass them to the attention backends as
  ``CommonAttentionMetadata.mm_req_doc_ranges``, the PrefixLM ranges that the
  Triton and FlexAttention backends already OR into the causal mask.
- The scheduler never ends a step inside a block, because the keys of the
  block's later tokens must exist when its earlier tokens are computed.
- The prefix cache keys every hash block by the tokens of each attention block
  it overlaps, so a cached block is reused only when its whole attention block
  is identical.

Everything here is pure Python so the scheduler, the frontend and the tests can
use it without a GPU.
"""

import hashlib
from array import array
from collections.abc import Mapping, Sequence
from typing import Any

ATTENTION_BLOCKS_KEY = "attention_blocks"
"""The ``SamplingParams.extra_args`` key that holds a request's blocks."""

RANGE_BACKENDS = frozenset(
    {"TRITON_ATTN", "FLEX_ATTENTION", "TRITON_FLASHINFER", "TRITON_FLASH_ATTN"}
)
"""Attention backends that apply ``mm_req_doc_ranges`` to every prefill row.

FLASH_ATTN also declares PrefixLM support, but it sizes its range buffers from
``is_mm_prefix_lm`` only, so it would drop request ranges without a word. The
composites route every batch with a range to their Triton child."""

AttentionBlocks = tuple[tuple[int, int], ...]


class AttentionBlocksError(ValueError):
    """Invalid ``attention_blocks`` on a request."""


def parse_attention_blocks(value: Any) -> AttentionBlocks:
    """Validate the shape of a request's blocks and return them as tuples.

    Blocks are inclusive ``[start, end]`` token ranges, sorted and
    non-overlapping. Raises AttentionBlocksError otherwise.
    """
    if not isinstance(value, (list, tuple)) or not value:
        raise AttentionBlocksError(
            f"{ATTENTION_BLOCKS_KEY} must be a non-empty list of [start, end] "
            "token ranges"
        )
    blocks: list[tuple[int, int]] = []
    for block in value:
        if (
            not isinstance(block, (list, tuple))
            or len(block) != 2
            or not all(isinstance(i, int) and not isinstance(i, bool) for i in block)
        ):
            raise AttentionBlocksError(
                f"{ATTENTION_BLOCKS_KEY}: {block!r} is not an inclusive "
                "[start, end] pair of token positions"
            )
        start, end = block
        if start < 0 or end < start:
            raise AttentionBlocksError(
                f"{ATTENTION_BLOCKS_KEY}: [{start}, {end}] is not a range of "
                "token positions"
            )
        if blocks and start <= blocks[-1][1]:
            raise AttentionBlocksError(
                f"{ATTENTION_BLOCKS_KEY} must be sorted and must not overlap, "
                f"got [{start}, {end}] after {list(blocks[-1])}"
            )
        blocks.append((start, end))
    return tuple(blocks)


def get_attention_blocks(sampling_params: Any) -> AttentionBlocks | None:
    """The blocks of a request's sampling params, or None if it has none."""
    extra_args: Mapping[str, Any] | None = getattr(sampling_params, "extra_args", None)
    if not extra_args:
        return None
    value = extra_args.get(ATTENTION_BLOCKS_KEY)
    if value is None:
        return None
    return parse_attention_blocks(value)


def check_attention_blocks(
    blocks: AttentionBlocks,
    prompt_len: int,
    max_num_batched_tokens: int,
) -> None:
    """Request-level checks: the blocks lie in the prompt, and the prompt fits
    one step, so the scheduler can always finish it without ending a step
    inside a block."""
    if blocks[-1][1] >= prompt_len:
        raise AttentionBlocksError(
            f"{ATTENTION_BLOCKS_KEY}: block {list(blocks[-1])} ends past the "
            f"prompt of {prompt_len} tokens"
        )
    if prompt_len > max_num_batched_tokens:
        raise AttentionBlocksError(
            f"a prompt with {ATTENTION_BLOCKS_KEY} must fit one scheduler step, "
            f"but it has {prompt_len} tokens and --max-num-batched-tokens is "
            f"{max_num_batched_tokens}"
        )


def ends_inside_block(end: int, blocks: AttentionBlocks) -> bool:
    """Whether a step that computes up to token ``end`` (exclusive) stops
    after the first token of a block and before its last."""
    return any(start < end <= stop for start, stop in blocks)


def doc_ranges(blocks: AttentionBlocks) -> list[tuple[int, int]]:
    """The blocks as ``mm_req_doc_ranges`` entries for one request."""
    return list(blocks)


def block_aligned_chunk_end(
    blocks: AttentionBlocks,
    start: int,
    budget_end: int,
    prefill_end: int,
    align_end: int | None = None,
    mamba_block_size: int | None = None,
) -> int:
    """The furthest end a prefill step of a request with attention blocks may
    stop at, in ``(start, budget_end]``. Returns ``start`` when no end is
    allowed and the request must wait.

    A step never ends inside a block (see ``ends_inside_block``). This holds
    for a step that starts inside a block too: its earlier tokens came from a
    cache hit of the identical block, and its later tokens still need every key
    of the block.

    With Mamba "align" prefix caching (``mamba_block_size`` set),
    ``align_end`` is where ``Scheduler._mamba_block_aligned_split`` would stop.
    Its stops at block boundaries only save states for the cache, except one:
    a step that starts inside a Mamba block must not run past that block's end
    unless the split itself allows it (an in-chunk checkpoint), or the block's
    stale state would be cached. This function keeps that rule, prefers the
    split's own end whenever it is allowed, and only stops a step inside a
    Mamba block when the next step can then reach the block's end or the end of
    the prompt, so a request never gets stuck.
    """
    if align_end is None:
        align_end = budget_end
    if start >= prefill_end:
        return align_end
    bs = mamba_block_size

    def next_boundary(pos: int) -> int:
        assert bs is not None
        return (pos // bs + 1) * bs

    def allowed(end: int) -> bool:
        if ends_inside_block(end, blocks):
            return False
        if bs is None:
            return True
        # A step from inside a Mamba block stays in it unless the split says
        # otherwise.
        if end != align_end and start % bs and end > next_boundary(start):
            return False
        # A step that stops inside a Mamba block leaves the next step unable to
        # cross the block's end; that step must still have somewhere to stop.
        if end < prefill_end and end % bs:
            target = min(next_boundary(end), prefill_end)
            if ends_inside_block(target, blocks):
                return False
        return True

    if start < align_end <= budget_end and allowed(align_end):
        return align_end
    candidates = {budget_end, prefill_end}
    for block_start, block_end in blocks:
        candidates.add(block_start)
        candidates.add(block_end + 1)
    if bs is not None:
        candidates.add(next_boundary(start))
        candidates.add(budget_end // bs * bs)
    ok = [e for e in candidates if start < e <= budget_end and allowed(e)]
    return max(ok, default=start)


def block_hash_extra_keys(
    token_ids: Sequence[int],
    blocks: AttentionBlocks,
    start: int,
    end: int,
    digests: dict[tuple[int, int], bytes],
) -> list[tuple[str, bytes, int]]:
    """Prefix-cache extra keys for the hash block ``[start, end)``.

    One ``("blk", digest, offset)`` key per attention block the hash block
    overlaps: the digest covers all of the attention block's tokens and the
    offset is the attention block's start relative to the hash block. Two
    prompts then share a cached block only when every attention block it
    touches is identical, which is when its KV and recurrent state are.
    ``digests`` caches the digest of each attention block per request.
    """
    keys: list[tuple[str, bytes, int]] = []
    for block_start, block_end in blocks:
        if block_end < start:
            continue
        if block_start >= end:
            break
        digest = digests.get((block_start, block_end))
        if digest is None:
            tokens = array("q", token_ids[block_start : block_end + 1])
            digest = hashlib.sha256(tokens.tobytes()).digest()
            digests[(block_start, block_end)] = digest
        keys.append(("blk", digest, block_start - start))
    return keys
