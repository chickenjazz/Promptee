"""Integration tests for the three reported rewriter failure modes.

Failure 1: rewriter ANSWERS the pasted task instead of rewriting it.
Failure 2: core task replaced with generic "create a prompt" wording.
Failure 3: rewriter outputs a code block when the user did not ask for code.

These tests exercise the deterministic guardrails (archetype classifier,
post-generation validator, deterministic diagnostics) without loading the
generation model. End-to-end smoke tests requiring the LoRA adapter live
in the verification section of the implementation plan, not here.

History note: an earlier version of this file relied on `detect_answer_shape`
and `_score_specificity_recall`, both of which have been removed. The new
validator is more conservative by design — it falls back to the raw prompt
only on high-confidence failure shapes (meta-prompt drift, leading "the answer
is" / "sure" / "certainly", empty output, unexpected code blocks for
non-coding prompts). Subtler answer shapes (e.g., a "Yes, X is..." preamble
to a paragraph answer) are not caught here; the heuristic scorer's semantic-
preservation gate is the second line of defense.
"""

from __future__ import annotations

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset_builder.prompt_templates import Archetype, detect_archetype
from tools.prompt_diagnostics import find_prompt_issues
from tools.prompt_validator import validate_rewrite


# ---------------------------------------------------------------------
# Failure mode 1: rewriter answers the prompt
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, candidate",
    [
        ("What is gravity?", "The answer is gravity is the force that attracts objects."),
        ("Explain photosynthesis.", "Sure, photosynthesis is the process by which plants convert sunlight into energy."),
        ("Compare TCP and UDP.", "Certainly! TCP is a connection-oriented protocol while UDP is connectionless."),
        ("Summarize the Mamba paper.", "Here is the explanation: Mamba is a state-space model architecture."),
    ],
)
def test_validator_flags_answer_shaped_outputs(raw, candidate):
    result = validate_rewrite(raw, candidate)
    assert result["status"] == "invalid"
    assert any(
        issue["type"] == "answer_instead_of_rewrite" for issue in result["issues"]
    )


def test_validator_flags_empty_output():
    result = validate_rewrite("Anything", "")
    assert result["status"] == "invalid"
    assert any(issue["type"] == "empty_output" for issue in result["issues"])


# ---------------------------------------------------------------------
# Failure mode 2: rewriter drifts into "create a prompt for this task"
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, candidate",
    [
        (
            "Write a Python function for fibonacci",
            "Create a prompt that asks an AI to write a fibonacci function.",
        ),
        (
            "Explain the CAP theorem",
            "Write a prompt for an AI to explain the CAP theorem to a beginner.",
        ),
        (
            "Summarize this paper",
            "Generate a prompt that summarizes academic papers in three bullets.",
        ),
    ],
)
def test_validator_flags_meta_prompt_drift(raw, candidate):
    result = validate_rewrite(raw, candidate)
    assert result["status"] == "invalid"
    assert any(issue["type"] == "meta_prompt_drift" for issue in result["issues"])


def test_diagnostics_warn_on_meta_prompt_phrasing_in_input():
    """If the user *types* meta-prompt phrasing, surface it as an issue —
    the rewriter is otherwise likely to mirror it back."""
    issues = find_prompt_issues("Create a prompt that explains gravity")
    assert any(issue["type"] == "meta_prompt_drift" for issue in issues)


# ---------------------------------------------------------------------
# Failure mode 3: code block emitted for non-coding prompts
# ---------------------------------------------------------------------

def test_validator_flags_unexpected_code_block_for_creative_prompt():
    raw = "Write a haiku about autumn"
    candidate = "```\nLeaves drift down softly\nWhispers of the dying year\nWinter's breath approaches\n```"
    result = validate_rewrite(raw, candidate)
    assert result["status"] == "invalid"
    assert any(issue["type"] == "unexpected_code_output" for issue in result["issues"])


def test_validator_allows_code_block_for_coding_prompt():
    raw = "Write a python function that adds two numbers"
    candidate = (
        "ROLE: Senior Python engineer.\n"
        "TASK:\nImplement an `add(a, b)` function.\n"
        "```python\n# scaffold only — no implementation\n```"
    )
    result = validate_rewrite(raw, candidate)
    assert result["status"] == "valid"


# ---------------------------------------------------------------------
# Archetype-routing drift control
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Write a Python function for fibonacci", Archetype.CODING),
        ("Compare TCP and UDP for low-latency video streaming", Archetype.ANALYTICAL),
        ("Write a haiku about autumn leaves", Archetype.CREATIVE),
        ("Help me through a difficult conversation with my coach", Archetype.CONVERSATIONAL),
        ("Build a lesson plan template for grade 7 science", Archetype.STRUCTURED),
        ("write a quick tldr summary", Archetype.CONCISE),
    ],
)
def test_archetype_routing_for_standard_cases(raw, expected):
    """Drift control: standard prompts must keep their archetype routing
    so the LoRA adapter receives the same scaffold-shape it was trained on."""
    assert detect_archetype(raw) == expected


def test_proper_structured_rewrite_passes_validator():
    raw = "Write a Python function for fibonacci"
    candidate = (
        "ROLE: Senior Python engineer.\n"
        "TASK:\nWrite a function that returns fib(n).\n"
        "INPUTS:\n- n: int\nOUTPUTS:\n- int\n"
        "CONSTRAINTS:\n- O(n) time"
    )
    result = validate_rewrite(raw, candidate)
    assert result["status"] == "valid"
    assert result["issues"] == []
