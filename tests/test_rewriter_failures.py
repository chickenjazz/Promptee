"""Regression tests for the three reported rewriter failure modes.

Failure 1: rewriter ANSWERS the pasted task instead of rewriting it.
Failure 2: core task replaced with generic "create a prompt" wording — specific
           constraints (language, framework, entity names, counts) lost.
Failure 3: analysis-style inputs ("Do you know X?", "What is X?") collapse to
           Concise / Minimal Modular and either answer or get one-line rewrites.

These tests exercise the deterministic guardrails (classifier, answer-shape
detector, specificity-recall gate) in isolation. They do NOT load the
model — that's covered by the end-to-end smoke tests in the verification
section of the plan, which require GPU + adapter on disk.
"""
from __future__ import annotations

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset_builder.prompt_templates import Archetype, detect_archetype
from tools.prompt_optimizer import detect_answer_shape


# ─────────────────────────────────────────────────────────────────────
# Failure mode 3: analysis-style inputs must NOT collapse to CONCISE
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "prompt",
    [
        "Do you know what causal attention is?",
        "Are you familiar with the CAP theorem?",
        "Can you explain how RoPE works?",
        "What is gradient checkpointing?",
        "How do plants grow?",
        "How does backpropagation work?",
        "Do you understand the halting problem?",
    ],
)
def test_analysis_questions_route_to_analytical(prompt):
    assert detect_archetype(prompt) == Archetype.ANALYTICAL, (
        f"Expected ANALYTICAL routing for analysis-style question; "
        f"old classifier collapsed these to CONCISE."
    )


@pytest.mark.parametrize(
    "prompt",
    [
        "What is gradient checkpointing?",
        "How do plants grow?",
        "Are you familiar with X?",
    ],
)
def test_short_questions_never_route_to_concise(prompt):
    assert detect_archetype(prompt) != Archetype.CONCISE, (
        f"Short question must not fall into the Concise/Minimal Modular "
        f"path which collapses the rewrite to one sentence."
    )


def test_short_imperative_still_concise():
    """Drift-control: non-question short imperatives must keep their old
    Concise routing so the LoRA adapter's training distribution is preserved."""
    assert detect_archetype("write a quick tldr summary") == Archetype.CONCISE
    # Plain commands without keywords keep falling through length heuristic
    assert detect_archetype("Write about garden tools.") == Archetype.CONCISE


# ─────────────────────────────────────────────────────────────────────
# Failure mode 1: answer-shape detector must catch code/answer outputs
# ─────────────────────────────────────────────────────────────────────

def test_detector_catches_code_answer():
    raw = "Write a Python function that returns the nth Fibonacci number."
    cand = "def fib(n):\n    if n<2: return n\n    return fib(n-1)+fib(n-2)"
    flagged, reasons = detect_answer_shape(raw, cand)
    assert flagged, f"Should flag code answer; reasons={reasons}"


def test_detector_catches_sql_answer():
    raw = "Give me a SQL query joining users and orders on user_id."
    cand = "SELECT u.id, u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id;"
    flagged, _ = detect_answer_shape(raw, cand)
    assert flagged


def test_detector_catches_yes_preamble_answer():
    raw = "Do you know what causal attention is?"
    cand = (
        "Yes, causal attention is a mechanism where each token can only "
        "attend to previous tokens in the sequence. It is implemented by "
        "masking future positions during the attention computation."
    )
    flagged, _ = detect_answer_shape(raw, cand)
    assert flagged


def test_detector_passes_proper_rewrite():
    raw = "Write a Python function that returns the nth Fibonacci number."
    cand = (
        "ROLE: Senior Python engineer.\n"
        "TASK:\nWrite a function that returns fib(n) for [Insert n range].\n"
        "INPUTS:\n- n: int\nOUTPUTS:\n- int\n"
        "CONSTRAINTS:\n- O(n) time"
    )
    flagged, reasons = detect_answer_shape(raw, cand)
    assert not flagged, f"Should not flag a proper rewrite; reasons={reasons}"


def test_detector_no_false_positive_when_raw_quotes_code():
    """Detector subtracts substrings present in raw, so quoting user code
    in the rewrite must not trigger the code-line signal."""
    raw = "Help me debug this code: def foo(): pass"
    cand = (
        "ROLE: Senior engineer.\nTASK:\nDebug the snippet below.\n"
        "INPUTS:\n- def foo(): pass\nOUTPUTS:\n- diagnosis"
    )
    flagged, _ = detect_answer_shape(raw, cand)
    assert not flagged


def test_detector_passes_short_conversational_rewrite():
    raw = "write a haiku about autumn"
    cand = (
        "Write a haiku about autumn that captures the essence of the "
        "season in three lines following the 5-7-5 syllable structure."
    )
    flagged, _ = detect_answer_shape(raw, cand)
    assert not flagged


# ─────────────────────────────────────────────────────────────────────
# Failure mode 2: specificity-recall gate must catch entity loss
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def scorer():
    """Lazy-import so we don't pay model-loading cost when running just the
    classifier/detector tests."""
    from tools.heuristic_scorer import HeuristicScorer
    return HeuristicScorer()


def test_specificity_recall_passes_when_entities_preserved(scorer):
    raw = (
        "Write a TypeScript function in Next.js 14 App Router that streams "
        "OpenAI responses with SSE."
    )
    cand = (
        "ROLE: Senior TypeScript engineer.\n"
        "TASK:\nBuild a Next.js 14 App Router endpoint that streams OpenAI "
        "completions over SSE.\n"
        "INPUTS:\n- An OpenAI API key\nOUTPUTS:\n- An SSE stream"
    )
    recall = scorer._score_specificity_recall(raw, cand)
    assert recall >= 0.80, f"All entities present, recall={recall}"


def test_specificity_recall_catches_entity_loss(scorer):
    raw = (
        "Write a TypeScript function in Next.js 14 App Router that streams "
        "OpenAI responses with SSE."
    )
    # Generic rewrite that drops every named entity
    cand = (
        "ROLE: Senior engineer.\nTASK:\nBuild a function that streams "
        "responses to the client.\nINPUTS:\n- An API key\nOUTPUTS:\n- A stream"
    )
    recall = scorer._score_specificity_recall(raw, cand)
    assert recall < 0.80, (
        f"Generic rewrite drops TypeScript/Next.js/OpenAI/SSE; "
        f"recall should fall below floor, got {recall}"
    )


def test_specificity_recall_preserves_counts_and_audience(scorer):
    raw = "Summarize this 2024 paper on Mamba in 3 bullets for a senior ML engineer."
    cand_good = (
        "ROLE: Senior ML engineer.\n"
        "TASK:\nSummarize the 2024 paper on Mamba in 3 bullets for a "
        "senior ML engineer audience."
    )
    cand_bad = "Summarize the paper briefly."
    assert scorer._score_specificity_recall(raw, cand_good) >= 0.80
    assert scorer._score_specificity_recall(raw, cand_bad) < 0.80


def test_specificity_recall_one_when_raw_has_no_entities(scorer):
    """Recall is undefined when raw has no specific tokens; should pass."""
    raw = "help me write something fun"
    cand = "ROLE: Creative writer.\nTASK:\nWrite something fun for the user."
    recall = scorer._score_specificity_recall(raw, cand)
    assert recall == 1.0
