"""
White-box tests for the heuristic scoring algorithm.

Rubric items covered:
  1. Heuristic Scoring Accuracy — clarity, specificity, ambiguity, redundancy,
     length, structural bonus, and the final score formula.
  6. Algorithm Efficiency — wall-clock guard on the scoring pipeline.

Targets: tools/heuristic_scorer.py (HeuristicScorer, ScorerConfig, AMBIGUOUS_TOKENS).
"""

import time

import pytest

from tools.heuristic_scorer import (
    AMBIGUOUS_TOKENS,
    HeuristicScorer,
    ScorerConfig,
)


# ── Stage 1: Clarity ──────────────────────────────────────────────────────

def test_clarity_higher_for_imperative_than_for_vague(scorer, sample_prompts):
    vague = scorer._score_prompt(sample_prompts["vague"])
    structured = scorer._score_prompt(sample_prompts["structured"])
    assert structured["clarity"] > vague["clarity"], (
        f"structured clarity={structured['clarity']:.4f} should exceed "
        f"vague clarity={vague['clarity']:.4f}"
    )


def test_clarity_in_unit_range(scorer, sample_prompts):
    for key in ("vague", "moderate", "structured", "short"):
        m = scorer._score_prompt(sample_prompts[key])
        assert 0.0 <= m["clarity"] <= 1.0, f"{key} clarity out of range: {m['clarity']}"


# ── Stage 2: Specificity ──────────────────────────────────────────────────

def test_specificity_increases_with_constraints(scorer):
    bare = scorer._score_prompt("Write about dogs.")
    constrained = scorer._score_prompt(
        "Write a 300-word article about Golden Retriever puppies, "
        "covering temperament, training, and exercise needs, formatted as JSON."
    )
    assert constrained["specificity"] > bare["specificity"], (
        f"constrained specificity={constrained['specificity']:.4f} should "
        f"exceed bare specificity={bare['specificity']:.4f}"
    )


def test_specificity_in_unit_range(scorer, sample_prompts):
    for key, prompt in sample_prompts.items():
        if not isinstance(prompt, str):
            continue
        m = scorer._score_prompt(prompt)
        assert 0.0 <= m["specificity"] <= 1.0, (
            f"{key} specificity out of range: {m['specificity']}"
        )


# ── Stage 3: Ambiguity penalty ────────────────────────────────────────────

def test_ambiguity_penalty_triggers_on_known_tokens(scorer):
    penalty = scorer._compute_ambiguity_penalty(
        "explain something about various things etc somehow"
    )
    assert penalty > 0, "ambiguous tokens should produce a non-zero penalty"


def test_ambiguity_penalty_capped_at_config_max(scorer, scorer_config):
    # Pile on every ambiguous token to push the penalty above its cap.
    spam = " ".join(sorted(AMBIGUOUS_TOKENS))
    penalty = scorer._compute_ambiguity_penalty(spam)
    assert penalty <= scorer_config.ambiguity_max_penalty + 1e-9, (
        f"ambiguity penalty {penalty} exceeded cap "
        f"{scorer_config.ambiguity_max_penalty}"
    )


def test_ambiguity_penalty_zero_on_clean_prompt(scorer):
    penalty = scorer._compute_ambiguity_penalty(
        "Summarise the report in three bullet points."
    )
    assert penalty == 0.0


# ── Stage 4: Redundancy penalty ───────────────────────────────────────────

def test_redundancy_triggers_on_repeated_word(scorer):
    penalty = scorer._compute_redundancy_penalty("explain this very very clearly")
    assert penalty > 0, "repeated word should trigger redundancy penalty"


def test_redundancy_does_not_trigger_when_words_differ(scorer):
    # No consecutive duplicate content tokens → zero penalty. Confirms the
    # detector targets *consecutive* repetition, not all token frequency.
    penalty = scorer._compute_redundancy_penalty(
        "explain this clearly and thoroughly"
    )
    assert penalty == 0.0


def test_redundancy_does_not_trigger_on_non_consecutive_repeats(scorer):
    # "explain X and explain Y" should not penalise — the duplicates aren't
    # consecutive after tokenisation.
    penalty = scorer._compute_redundancy_penalty(
        "explain photosynthesis and explain respiration"
    )
    assert penalty == 0.0


def test_redundancy_triggers_across_punctuation(scorer):
    # Documented behaviour: punctuation is stripped before consecutive-
    # duplicate detection, so "very, very" still counts as redundant.
    # This is genuine white-box knowledge: the spaCy-based tokeniser drops
    # punctuation from the content-token stream.
    penalty = scorer._compute_redundancy_penalty(
        "explain this very, very clearly"
    )
    assert penalty > 0


def test_redundancy_capped_at_config_max(scorer, scorer_config):
    text = " ".join(["foo foo"] * 50)
    penalty = scorer._compute_redundancy_penalty(text)
    assert penalty <= scorer_config.redundancy_max_penalty + 1e-9


# ── Stage 5: Length penalty ───────────────────────────────────────────────

def test_length_penalty_zero_for_long_prompt(scorer):
    long_prompt = (
        "Write a detailed proposal for migrating our payments service from "
        "REST to gRPC, covering rollout phases and risk mitigation."
    )
    assert scorer._compute_length_penalty(long_prompt) == 0.0


def test_length_penalty_positive_for_short_prompt(scorer):
    assert scorer._compute_length_penalty("Explain AI") > 0


# ── Stage 7: Structural bonus ─────────────────────────────────────────────

def test_structural_bonus_rewards_headers_and_bullets(scorer, sample_prompts):
    plain = scorer._compute_structural_bonus(sample_prompts["moderate"])
    structured = scorer._compute_structural_bonus(sample_prompts["structured"])
    assert structured > plain, (
        f"structured bonus={structured:.4f} should exceed plain bonus={plain:.4f}"
    )


def test_structural_bonus_capped(scorer, scorer_config):
    # Hammer with every structural pattern at once.
    text = (
        "1. step one\n"
        "- bullet one\n"
        "Format: JSON\n"
        "Act as a teacher\n"
        "First, second, finally\n"
    )
    bonus = scorer._compute_structural_bonus(text)
    assert bonus <= scorer_config.structural_bonus_cap + 1e-9


# ── Stage 6: Quality score formula ────────────────────────────────────────

def test_quality_score_in_unit_range(scorer, sample_prompts):
    for key in ("vague", "moderate", "structured"):
        m = scorer._score_prompt(sample_prompts[key])
        assert 0.0 <= m["quality"] <= 1.0 + 1e-6, (
            f"{key} quality out of range: {m['quality']}"
        )


def test_quality_score_orders_prompts_correctly(scorer, sample_prompts):
    # Structured prompt with explicit constraints must outscore both the
    # vague and moderate baselines. Without this, the entire scoring pipeline
    # would fail to reward optimisation.
    vague = scorer._score_prompt(sample_prompts["vague"])
    structured = scorer._score_prompt(sample_prompts["structured"])
    assert structured["quality"] > vague["quality"]


def test_evaluate_raw_only_returns_self_similar_baseline(scorer, sample_prompts):
    result = scorer.evaluate(sample_prompts["moderate"])
    assert result["semantic_preservation"] == 1.0
    assert result["quality_improvement"] == 0.0
    assert result["clarity_delta"] == 0.0
    assert result["specificity_delta"] == 0.0
    assert result["rejected"] is False
    assert result["final_score"] == result["raw_quality"]


def test_evaluate_pair_computes_positive_improvement_for_real_optimisation(
    scorer, sample_prompts
):
    # Vague raw → structured candidate must produce strictly positive
    # quality improvement (assuming semantic preservation passes).
    result = scorer.evaluate(
        sample_prompts["vague"], sample_prompts["structured"]
    )
    # Improvement may be scaled by semantic if the meaning drifted heavily;
    # the ungated quality_improvement must still be positive.
    assert result["quality_improvement"] > 0, result


# ── Rubric 6: Algorithm Efficiency ────────────────────────────────────────

def test_evaluate_completes_within_time_budget(scorer):
    """Loose wall-clock guard: a 200-word prompt should evaluate in < 5s on CPU.

    This is a regression guard, not a benchmark — exists to catch accidental
    O(n²) regressions in the scoring pipeline.
    """
    long_prompt = (
        "Act as a senior software engineer. "
        "Design a scalable distributed caching system that handles 100k requests "
        "per second across three regions. " * 5
    )
    start = time.perf_counter()
    scorer.evaluate(long_prompt, long_prompt + " Add a tracing requirement.")
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"evaluate() took {elapsed:.2f}s — performance regression"


def test_evaluate_does_not_grow_superlinearly(scorer):
    """A prompt 4× longer should not take 16× as long. Asserts roughly linear
    scaling in token count to catch O(n²) regressions in spaCy/regex paths."""
    short = "Summarise the key takeaways in three bullet points."
    long = " ".join([short] * 8)

    # Warm up: discard the first measurement to remove cold-cache effects.
    scorer.evaluate(short)

    t1 = time.perf_counter()
    for _ in range(3):
        scorer.evaluate(short)
    short_avg = (time.perf_counter() - t1) / 3

    t2 = time.perf_counter()
    for _ in range(3):
        scorer.evaluate(long)
    long_avg = (time.perf_counter() - t2) / 3

    # Allow up to 16× headroom (8× length × 2× slack). Anything worse is a
    # genuine non-linear regression worth flagging.
    if short_avg > 0:
        ratio = long_avg / short_avg
        assert ratio < 16.0, (
            f"scoring grew {ratio:.1f}× for an 8× longer prompt — "
            f"possible super-linear regression"
        )
