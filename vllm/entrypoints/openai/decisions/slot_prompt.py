# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The prompt of slot-readout decision checkpoints (Base Labs mammoth-1 and
elephant-1).

A port of ``qev/prompt.py`` (prompt version ``qev-labels-v2``, OneJev at
81ce62f), restricted to the production template the checkpoints were trained
on. Every byte matters: the prefix length, the block boundary and the slot a
label lands on all follow from this text, so keep it in step with the
reference rather than with vLLM's other prompts.

One question is one chat::

    system: SYSTEM_PROMPT
    user: "<state>\\n{state}\\n</state>\\n\\n" + suffix

where the suffix is the question, its options under slot labels, and the
answer line. Every question of a request shares the prefix up to the end of the
state block.
"""

import json
from typing import Any

PROMPT_VERSION = "qev-labels-v2"

SYSTEM_PROMPT = (
    "Apply the question to the state. Choose exactly one of the listed options. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#: Two-letter labels that are not one token in the Qwen3.5 vocabulary.
_NOT_ONE_TOKEN = frozenset(
    {
        "BQ", "BZ", "CJ", "CQ", "CZ", "DQ", "DZ", "EJ", "EY", "FJ", "FQ", "FV",
        "FZ", "GJ", "GK", "GQ", "GZ", "HJ", "IY", "JF", "JG", "JH", "JL", "JN",
        "JQ",
    }
)  # fmt: skip
#: Slot i labels option i: A to Z, then the two-letter labels in order.
SLOT_LABELS: tuple[str, ...] = tuple(
    list(LETTERS)
    + [a + b for a in LETTERS for b in LETTERS if a + b not in _NOT_ONE_TOKEN][
        : 256 - len(LETTERS)
    ]
)
MAX_SLOTS = len(SLOT_LABELS)

QUESTION_HEADER = "Question:"
SCORE_HEADER = "Rate the state:"
ANSWER_LINE = "Answer with one letter: {letters}."
ANSWER_LINE_MANY = "Answer with one label: {letters}."

CHOICE_DEFAULT_INSTRUCTIONS = "Which option applies to the state?"
SCORE_DEFAULT_INSTRUCTIONS = "Which level describes the state?"
NOUL_DEFAULT_INSTRUCTIONS = (
    "Is the statement true, or is the answer to the question yes?"
)
NOUL_DEFAULT_TRUE = "the statement is true / the answer is yes"
NOUL_DEFAULT_FALSE = "the statement is false / the answer is no"

#: Placed where a suffix goes to find where the shared prefix ends.
PREFIX_MARK = "QEV_PREFIX_MARK"


def render_entry(entry: Any) -> str:
    """Render an instructions or criteria entry (string, object, array or null)
    as text."""
    if entry is None:
        return ""
    if isinstance(entry, str):
        return entry.strip()
    return json.dumps(entry, ensure_ascii=False, indent=2)


def render_state(state: Any) -> str:
    if state is None:
        return ""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2)


def options_lines(names: list[str], descriptions: list[Any]) -> list[str]:
    lines = []
    for letter, name, desc in zip(SLOT_LABELS, names, descriptions):
        text = render_entry(desc)
        lines.append(f"{letter}. {name}: {text}" if text else f"{letter}. {name}")
    return lines


def question_options(type_name: str, criteria: Any) -> tuple[list[str], list[Any]]:
    """The option names and descriptions of a question, in slot order: a
    choice's criteria keys, yes then no, or the score levels from 0."""
    if type_name == "noul":
        crit = criteria or {}
        true, false = crit.get("true"), crit.get("false")
        return ["yes", "no"], [
            render_entry(true) if true is not None else NOUL_DEFAULT_TRUE,
            render_entry(false) if false is not None else NOUL_DEFAULT_FALSE,
        ]
    if type_name == "choice":
        names = list(criteria.keys())
        return names, [criteria[k] for k in names]
    if type_name == "score":
        return [f"level {i}" for i in range(len(criteria))], list(criteria)
    raise ValueError(f"unknown question type {type_name!r}")


def render_suffix(type_name: str, instructions: Any, criteria: Any) -> str:
    """One question's suffix: the question, its options and the answer line."""
    if type_name == "noul":
        default = NOUL_DEFAULT_INSTRUCTIONS
    elif type_name == "choice":
        default = CHOICE_DEFAULT_INSTRUCTIONS
    else:
        default = SCORE_DEFAULT_INSTRUCTIONS
    text = render_entry(instructions) or default
    names, descriptions = question_options(type_name, criteria)
    if len(names) > MAX_SLOTS:
        raise ValueError(
            f"{len(names)} options exceed the {MAX_SLOTS} slots of prompt "
            f"{PROMPT_VERSION}"
        )
    letters = SLOT_LABELS[: len(names)]
    header = SCORE_HEADER if type_name == "score" else QUESTION_HEADER
    answer = ANSWER_LINE if len(letters) <= len(LETTERS) else ANSWER_LINE_MANY
    return (
        f"{header} {text}\n\nOptions:\n"
        + "\n".join(options_lines(names, descriptions))
        + "\n\n"
        + answer.format(letters=", ".join(letters))
    )


def state_block(state: Any) -> str:
    return f"<state>\n{render_state(state)}\n</state>\n\n"


def build_messages(state: Any, suffix: str) -> list[dict[str, str]]:
    """The chat of one question. Every question of a request shares the prefix
    up to the end of the state block."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": state_block(state) + suffix},
    ]
