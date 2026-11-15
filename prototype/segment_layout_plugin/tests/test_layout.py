"""CPU: layout arithmetic. Needs numpy only (no vLLM, no GPU)."""

import numpy as np
import pytest

from vllm_segment_layout.layout import (
    SegmentLayout,
    build_step,
    check_lengths,
    dense_mask,
    dense_mask_from_varlen,
    parse_lengths,
)


def test_rows_match_the_pr_unit_test():
    layout = SegmentLayout.from_lengths([3, 2, 4])
    assert layout.segment.tolist() == [0, 0, 0, 1, 1, 2, 2, 2, 2]
    assert layout.positions.tolist() == [0, 1, 2, 3, 4, 3, 4, 5, 6]
    assert layout.q_lens.tolist() == [3, 2, 4]
    assert layout.k_lens.tolist() == [3, 5, 7]
    assert layout.kv_index.tolist() == [0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 5, 6, 7, 8]


def test_mask_matches_the_pr_unit_test():
    # Request 0: prefix 2, candidates of 1 and 2 tokens. Request 1: 2 tokens, no layout.
    step = build_step(np.array([0, 5, 7]), [SegmentLayout.from_lengths([2, 1, 2]), None])
    expected = [
        [1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 0, 0],
        [1, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 1],
    ]
    assert dense_mask(step).astype(int).tolist() == expected
    assert dense_mask_from_varlen(step).astype(int).tolist() == expected
    assert step.cu_q.tolist() == [0, 2, 3, 5, 7]
    assert step.cu_k.tolist() == [0, 2, 5, 9, 11]
    assert (step.max_q, step.max_k) == (2, 4)


def test_no_layout_in_the_batch_is_none():
    assert build_step(np.array([0, 5, 7]), [None, None]) is None


@pytest.mark.parametrize("seed", range(20))
def test_mask_and_varlen_forms_agree_on_random_batches(seed):
    rng = np.random.default_rng(seed)
    layouts, counts = [], []
    for _ in range(int(rng.integers(1, 6))):
        if rng.random() < 0.6:
            lengths = rng.integers(1, 9, size=int(rng.integers(2, 7))).tolist()
            layouts.append(SegmentLayout.from_lengths(lengths))
            counts.append(sum(lengths))
        else:
            layouts.append(None)
            counts.append(int(rng.integers(1, 12)))
    if not any(layouts):
        layouts[0], counts[0] = SegmentLayout.from_lengths([2, 3]), 5
    step = build_step(np.concatenate(([0], np.cumsum(counts))), layouts)
    mask = dense_mask(step)
    assert (mask == dense_mask_from_varlen(step)).all()
    # Brute force from the definition.
    offset = 0
    for layout, count in zip(layouts, counts):
        block = mask[offset : offset + count]
        assert not block[:, :offset].any() and not block[:, offset + count :].any()
        inner = block[:, offset : offset + count]
        if layout is None:
            assert inner.all()
        else:
            seg = layout.segment
            assert (inner == ((seg[None, :] == 0) | (seg[None, :] == seg[:, None]))).all()
        offset += count


@pytest.mark.parametrize("bad", [[4], [0, 2], [3, -1], [2, 1.5], "ab", None, [True, 2]])
def test_bad_lengths(bad):
    assert parse_lengths(bad) is None
    with pytest.raises(ValueError):
        check_lengths(bad, 4, 512)


def test_front_end_checks():
    assert check_lengths([2, 3], 5, 512) == [2, 3]
    with pytest.raises(ValueError, match="sum to 5"):
        check_lengths([2, 3], 6, 512)
    with pytest.raises(ValueError, match="positions"):
        check_lengths([500, 20, 3], 523, 512)
    assert parse_lengths([2, 3], 6) is None
