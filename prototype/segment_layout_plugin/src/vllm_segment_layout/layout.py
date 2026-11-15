"""Segment layout arithmetic. No vLLM import, CPU only.

A layout is ``[P, C1, ..., CN]``: a shared prefix of P tokens, then N candidates.
The prefix sees itself. Candidate k sees the prefix and itself. Candidate
positions restart at P.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

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


def check_lengths(value: Any, prompt_len: int, max_positions: int | None) -> list[int]:
    """Front-end validation with messages (IO processor, clients)."""
    lengths = parse_lengths(value)
    if lengths is None:
        raise ValueError(
            f"{EXTRA_KEY} must list the prefix length and 1 to {MAX_SEGMENTS - 1} "
            "candidate lengths, all positive integers"
        )
    if sum(lengths) != prompt_len:
        raise ValueError(
            f"{EXTRA_KEY} sum to {sum(lengths)}, but the prompt has {prompt_len} tokens"
        )
    longest = lengths[0] + max(lengths[1:])
    if max_positions is not None and longest > max_positions:
        raise ValueError(
            f"prefix + longest candidate is {longest} tokens, the model has "
            f"{max_positions} positions"
        )
    return lengths


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


def dense_mask(step: StepRows) -> np.ndarray:
    """[T, T] bool reference of the mask form (tests only)."""
    return (step.own[None, :] == step.own[:, None]) | (
        step.own[None, :] == step.vis[:, None]
    )


def dense_mask_from_varlen(step: StepRows) -> np.ndarray:
    """[T, T] bool reference of the varlen form (tests only)."""
    mask = np.zeros((step.num_tokens, step.num_tokens), dtype=bool)
    for s in range(len(step.cu_q) - 1):
        keys = step.kv_index[step.cu_k[s] : step.cu_k[s + 1]]
        mask[step.cu_q[s] : step.cu_q[s + 1], keys] = True
    return mask
