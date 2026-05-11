"""
White-box tests for the semantic-preservation gate.

Rubric item: 3. Semantic Preservation.

Two layers of verification:
  - Real-embedding tests on crafted prompt pairs (paraphrase vs unrelated)
    to check that cosine similarity behaves as expected end-to-end.
  - Threshold-boundary tests that bypass the encoder and call the gating
    arithmetic directly with synthetic similarity scalars, so the assertions
    don't depend on exact embedding values.
"""

import pytest

from tools.heuristic_scorer import HeuristicScorer, ScorerConfig


# ── End-to-end semantic similarity (real model) ───────────────────────────

def test_paraphrase_pair_scores_high_similarity(scorer, sample_prompts):
    sim = scorer._score_semantic_preservation(
        sample_prompts["near_paraphrase_a"],
        sample_prompts["near_paraphrase_b"],
    )
    # all-MiniLM-L6-v2 puts true paraphrases comfortably above 0.7.
    # Loose bound to absorb minor model-version drift.
    assert sim > 0.70, f"paraphrase similarity unexpectedly low: {sim:.4f}"


def test_unrelated_pair_scores_below_soft_threshold(scorer, sample_prompts):
    sim = scorer._score_semantic_preservation(
        sample_prompts["topically_unrelated_a"],
        sample_prompts["topically_unrelated_b"],
    )
    # Topically unrelated prompts should fall well under the 0.7 soft threshold.
    assert sim < 0.50, f"unrelated similarity unexpectedly high: {sim:.4f}"


def test_self_similarity_is_one(scorer, sample_prompts):
    sim = scorer._score_semantic_preservation(
        sample_prompts["moderate"], sample_prompts["moderate"]
    )
    assert sim == pytest.approx(1.0, abs=1e-3)


# ── Gating-threshold branch coverage ──────────────────────────────────────

class _FixedScorer(HeuristicScorer):
    """Subclass that injects a synthetic semantic similarity, leaving every
    other code path untouched. Used to deterministically hit the hard-floor,
    soft-band, and pass-through branches without depending on the encoder."""

    forced_similarity: float = 1.0

    def _score_semantic_preservation(self, raw_prompt, optimized_prompt):
        return self.forced_similarity


@pytest.fixture(scope="module")
def fixed_scorer():
    return _FixedScorer()


# Use prompts that produce a non-trivial quality_improvement so we can
# distinguish "scaled down" from "rejected" from "passed through".
RAW_PROMPT = "tell me something about stuff"
CANDIDATE = (
    "Act as a senior data scientist.\n"
    "## Objective\n"
    "Summarise supervised learning in 200 words.\n"
    "## Constraints\n"
    "- Cover 3 algorithms: linear regression, decision trees, SVMs.\n"
    "- Include one numeric example per algorithm.\n"
    "## Output Format\n"
    "Markdown with H2 sections."
)


def test_hard_floor_rejects_below_threshold(fixed_scorer):
    fixed_scorer.forced_similarity = 0.39  # just under default 0.40 floor
    result = fixed_scorer.evaluate(RAW_PROMPT, CANDIDATE)
    assert result["rejected"] is True
    assert result["final_score"] == 0.0


def test_hard_floor_inclusive_at_exactly_threshold(fixed_scorer):
    # The check is `if semantic < hard_floor`, so similarity == floor must NOT
    # reject. (Inclusive boundary on the not-rejected side.)
    fixed_scorer.forced_similarity = 0.40
    result = fixed_scorer.evaluate(RAW_PROMPT, CANDIDATE)
    assert result["rejected"] is False


def test_soft_band_scales_improvement(fixed_scorer):
    fixed_scorer.forced_similarity = 0.55  # in soft band [0.40, 0.70)
    result = fixed_scorer.evaluate(RAW_PROMPT, CANDIDATE)
    assert result["rejected"] is False
    # Scaled: final_score = improvement * 0.55. The reference for "ungated"
    # is the same evaluation at sim=1.0 (no scaling). Within rounding tolerance.
    fixed_scorer.forced_similarity = 1.0
    ungated = fixed_scorer.evaluate(RAW_PROMPT, CANDIDATE)
    expected = round(ungated["quality_improvement"] * 0.55, 4)
    assert result["final_score"] == pytest.approx(expected, abs=1e-3)


def test_soft_threshold_inclusive_passes_through(fixed_scorer):
    # The check is `elif semantic < soft_threshold`, so similarity == 0.70
    # should NOT trigger scaling — full improvement passes through.
    fixed_scorer.forced_similarity = 0.70
    result = fixed_scorer.evaluate(RAW_PROMPT, CANDIDATE)
    assert result["final_score"] == pytest.approx(
        result["quality_improvement"], abs=1e-3
    )


def test_above_soft_threshold_no_scaling(fixed_scorer):
    fixed_scorer.forced_similarity = 0.95
    result = fixed_scorer.evaluate(RAW_PROMPT, CANDIDATE)
    assert result["final_score"] == pytest.approx(
        result["quality_improvement"], abs=1e-3
    )


# ── Custom config thresholds are respected ────────────────────────────────

def test_custom_thresholds_applied():
    """Override the gating thresholds and verify the gate respects them."""
    custom = ScorerConfig(semantic_hard_floor=0.20, semantic_soft_threshold=0.50)
    scorer = _FixedScorer(config=custom)

    # 0.30 is above the new 0.20 floor but inside the new soft band.
    scorer.forced_similarity = 0.30
    result = scorer.evaluate(RAW_PROMPT, CANDIDATE)
    assert result["rejected"] is False, "0.30 sits above the custom 0.20 floor"
    expected_scaled = round(result["quality_improvement"] * 0.30, 4)
    # Improvement gets scaled by 0.30 inside the soft band.
    # Recompute from the same final state — improvement in result is rounded.
    # We accept ±1e-3 to absorb rounding.
    # Alternative: compare to the unscaled run.
    scorer.forced_similarity = 1.0
    unscaled = scorer.evaluate(RAW_PROMPT, CANDIDATE)
    expected = round(unscaled["quality_improvement"] * 0.30, 4)
    assert result["final_score"] == pytest.approx(expected, abs=1e-3)
