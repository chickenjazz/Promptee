"""
White-box tests for technical reliability.

Rubric item: 7. Technical Reliability — exception paths, fallbacks, edge
cases, and graceful degradation under malformed inputs.
"""

import pytest

from tools.heuristic_scorer import HeuristicScorer
from tools.prompt_optimizer import PromptOptimizer
from tools.prompt_diagnostics import find_prompt_issues
from tools.prompt_validator import validate_rewrite


# ── HeuristicScorer edge cases ────────────────────────────────────────────

def test_evaluate_empty_string_does_not_crash(scorer):
    result = scorer.evaluate("")
    # Empty input collapses to zero-information; final_score should be a
    # valid float, not NaN, and the call must not raise.
    assert result["final_score"] == 0.0 or 0.0 <= result["final_score"] <= 1.0
    assert result["rejected"] is False  # raw-only evaluation can't be rejected


def test_evaluate_whitespace_only_does_not_crash(scorer):
    result = scorer.evaluate("    \n\n   \t  ")
    assert isinstance(result["final_score"], float)


def test_evaluate_very_long_input_completes(scorer):
    long = "Summarise the report in three bullet points. " * 200  # ~9k chars
    result = scorer.evaluate(long)
    assert isinstance(result["final_score"], float)


def test_evaluate_unicode_input_does_not_crash(scorer):
    weird = "Résumé étiqueté 你好 🎉 — explain this in two bullet points."
    result = scorer.evaluate(weird)
    assert isinstance(result["final_score"], float)


def test_evaluate_with_only_punctuation(scorer):
    result = scorer.evaluate("!!! ??? ... ;;; ")
    assert isinstance(result["final_score"], float)


def test_evaluate_pair_with_drastically_different_meaning_rejects(scorer):
    """End-to-end smoke that a totally unrelated 'rewrite' is caught by the
    semantic gate, not silently scored as an improvement."""
    raw = "Write a Python function that reverses a linked list."
    bad = "Compose a haiku about autumn leaves."
    result = scorer.evaluate(raw, bad)
    # Either rejected outright OR final_score is dragged near zero by the
    # soft-band scaling. Both are acceptable safety behaviours.
    if not result["rejected"]:
        assert result["final_score"] < result["quality_improvement"], (
            f"unrelated rewrite should be down-scaled or rejected, got "
            f"final_score={result['final_score']} for "
            f"improvement={result['quality_improvement']}"
        )


# ── prompt_diagnostics edge cases ─────────────────────────────────────────

def test_find_prompt_issues_handles_empty_string():
    issues = find_prompt_issues("")
    assert isinstance(issues, list)


def test_find_prompt_issues_handles_very_long_input():
    long = "Summarise the report. " * 1000  # ~22k chars
    issues = find_prompt_issues(long)
    assert isinstance(issues, list)
    # No assertion on content — we just need it not to raise or hang.


def test_find_prompt_issues_handles_unicode():
    issues = find_prompt_issues("Résumé étiqueté 你好 🎉")
    assert isinstance(issues, list)


# ── PromptOptimizer reliability ───────────────────────────────────────────

def test_optimizer_rewrite_without_load_raises_runtime_error():
    """White-box guard: the public API must raise (not silently return) when
    rewrite() is called before load_model()."""
    optimizer = PromptOptimizer(adapter_path="/nonexistent")
    with pytest.raises(RuntimeError):
        optimizer.rewrite("anything")


def test_clean_output_handles_empty_string():
    assert PromptOptimizer._clean_output("") == ""


def test_clean_output_handles_whitespace_only():
    assert PromptOptimizer._clean_output("   \n\t  ") == ""


def test_resolve_base_model_falls_back_when_metadata_missing():
    """When adapter_metadata.json doesn't exist, the resolver returns the
    caller-supplied default — not a crash, not a silent change."""
    resolved = PromptOptimizer._resolve_base_model(
        "DEFAULT/model-id", "/path/that/does/not/exist"
    )
    assert resolved == "DEFAULT/model-id"


# ── prompt_validator failure-mode coverage ────────────────────────────────

def test_validator_flags_meta_prompt_drift():
    raw = "summarize this article"
    bad = "Create a prompt that summarizes the article concisely."
    result = validate_rewrite(raw, bad)
    assert result["status"] == "invalid"
    assert any(i["type"] == "meta_prompt_drift" for i in result["issues"])


def test_validator_flags_answer_instead_of_rewrite():
    raw = "what is gravity"
    bad = "The answer is: gravity is a force that pulls objects toward Earth."
    result = validate_rewrite(raw, bad)
    assert result["status"] == "invalid"
    assert any(i["type"] == "answer_instead_of_rewrite" for i in result["issues"])


def test_validator_flags_empty_output():
    result = validate_rewrite("anything", "")
    assert result["status"] == "invalid"
    assert any(i["type"] == "empty_output" for i in result["issues"])


def test_validator_flags_unexpected_code_when_raw_was_not_coding():
    raw = "Write a haiku about rain."
    bad = "```python\ndef haiku(): pass\n```"
    result = validate_rewrite(raw, bad)
    assert result["status"] == "invalid"
    assert any(i["type"] == "unexpected_code_output" for i in result["issues"])


def test_validator_allows_code_when_raw_was_coding_request():
    raw = "Write a Python function that reverses a list."
    good = "```python\ndef reverse_list(items): return items[::-1]\n```"
    result = validate_rewrite(raw, good)
    # Coding raw + code output → no "unexpected_code_output" issue.
    assert not any(i["type"] == "unexpected_code_output" for i in result["issues"])


def test_validator_passes_clean_rewrite():
    raw = "explain photosynthesis"
    good = (
        "Explain photosynthesis to a beginner student in three short bullet "
        "points, in plain language without jargon."
    )
    result = validate_rewrite(raw, good)
    assert result["status"] == "valid"
    assert result["issues"] == []


def test_validator_handles_none_inputs_gracefully():
    """Defensive: the validator strips with `or ""`, so None should not raise."""
    result = validate_rewrite(None, None)
    # Empty optimized_prompt → at least empty_output issue.
    assert result["status"] == "invalid"
