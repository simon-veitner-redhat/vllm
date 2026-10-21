# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ``modeling_overrides`` block of Base Labs checkpoints (mammoth-1,
elephant-1).

It records how a checkpoint was trained and must be served: the mask of the
full-attention and linear-attention layers, and how answers are read. For
example::

    "modeling_overrides": {
        "attention": {
            "full_attention": "block_bidirectional",
            "linear_attention": "causal"
        },
        "readout": "slot_lm",
        "head_dim": 256,
        "n_slots": 256
    }

Any value vLLM cannot serve faithfully is rejected at load instead of being
served causally. ``VLLM_MODELING_OVERRIDES`` selects how the block is applied:

- ``apply`` (default): block-bidirectional attention and the slot readout.
- ``causal``: the slot readout and its prompt, with a causal mask. This is
  the ablation control for the mask on the same build.
- ``ignore``: stock serving. The checkpoint loads as a plain
  ``Qwen3_5ForConditionalGeneration`` with a causal mask and vLLM's own
  structured decisions prompt.
"""

from dataclasses import dataclass
from typing import Any

FULL_ATTENTION_MASKS = ("causal", "block_bidirectional")
"""Full-attention masks vLLM can serve."""

READOUTS = ("slot_lm",)
"""Readouts vLLM can serve. ``head`` (a learned pointer head) is not."""

MAX_SLOTS = 256
"""Slot labels the prompt defines (``SlotLMStrategy.SLOT_LABELS``)."""

_KNOWN_KEYS = frozenset({"attention", "readout", "head_dim", "n_slots"})
_KNOWN_ATTENTION_KEYS = frozenset({"full_attention", "linear_attention"})


@dataclass(frozen=True)
class ModelingOverrides:
    full_attention: str
    """``causal`` or ``block_bidirectional``."""
    readout: str
    """``slot_lm``: the LM-head logits of the slot labels at the last token."""
    n_slots: int
    """How many slot labels the checkpoint was trained with."""

    @property
    def block_attention(self) -> bool:
        return self.full_attention == "block_bidirectional"


OVERRIDES_KEYS = ("modeling_overrides", "baseten_modeling_overrides")
"""Where checkpoints keep the block. Baseten's checkpoints use the second
name; both are read, and they must agree when both are set."""


def get_raw_modeling_overrides(config: Any) -> Any:
    """The overrides block of a config, under either key, or None."""
    found = [getattr(config, key) for key in OVERRIDES_KEYS if hasattr(config, key)]
    found = [value for value in found if value is not None]
    if len(found) == 2 and found[0] != found[1]:
        raise ValueError(
            "modeling_overrides and baseten_modeling_overrides disagree: "
            f"{found[0]!r} vs {found[1]!r}"
        )
    return found[0] if found else None


def parse_modeling_overrides(raw: Any) -> ModelingOverrides | None:
    """The overrides of a checkpoint's ``config.json``, or None without any.

    Raises ValueError for anything vLLM would otherwise ignore, so a checkpoint
    is never served differently from how it was trained without notice.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"modeling_overrides must be an object, got {raw!r}")
    if unknown := raw.keys() - _KNOWN_KEYS:
        raise ValueError(f"unsupported modeling_overrides keys {sorted(unknown)}")

    attention = raw.get("attention") or {}
    if not isinstance(attention, dict):
        raise ValueError(
            f"modeling_overrides.attention must be an object, got {attention!r}"
        )
    if unknown := attention.keys() - _KNOWN_ATTENTION_KEYS:
        raise ValueError(
            f"unsupported modeling_overrides.attention keys {sorted(unknown)}"
        )
    full_attention = attention.get("full_attention", "causal")
    if full_attention not in FULL_ATTENTION_MASKS:
        raise ValueError(
            f"unsupported full_attention mask {full_attention!r}; vLLM serves "
            f"{list(FULL_ATTENTION_MASKS)}"
        )
    linear_attention = attention.get("linear_attention", "causal")
    if linear_attention != "causal":
        raise ValueError(
            f"unsupported linear_attention mask {linear_attention!r}; linear "
            "attention layers are recurrent and only 'causal' is served"
        )

    readout = raw.get("readout", "slot_lm")
    if readout not in READOUTS:
        raise ValueError(
            f"unsupported readout {readout!r}; vLLM serves {list(READOUTS)}"
        )
    n_slots = raw.get("n_slots", MAX_SLOTS)
    if (
        not isinstance(n_slots, int)
        or isinstance(n_slots, bool)
        or not 1 <= n_slots <= MAX_SLOTS
    ):
        raise ValueError(f"modeling_overrides.n_slots must be 1 to {MAX_SLOTS}")
    # `head_dim` is the width of the optional pointer head. slot_lm has none.
    return ModelingOverrides(
        full_attention=full_attention, readout=readout, n_slots=n_slots
    )
