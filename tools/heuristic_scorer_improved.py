"""
Improved Heuristic Scorer — Deterministic Prompt Quality Evaluation Pipeline.

Implements the full scoring pipeline from:
  - architecture/heuristic_scorer.md (base specification)
  - architecture/heuristic_scorer_improv_checklist.md (all 12 improvements)

Pipeline flow order:
  1. Clarity scoring (per-sentence actionability)
  2. Specificity scoring (constraint density)
  3. Ambiguity penalty (vague token detection)
  4. Redundancy penalty (repeated token detection)
  5. Length normalization (short-prompt adjustment)
  6. Quality score = dynamic-weight-normalized combination of clarity + specificity
  7. Structural bonus (optional, additive reward for structured prompts)
  8. Semantic constraint gating (hard reject below floor, soft scaling above)
  9. Metric delta computation (raw vs candidate)
  10. Rejection logic (semantic floor OR negative improvement)

Design decisions and assumptions are documented inline.
"""

import os
import re
import sys
import math
import logging
from typing import TypedDict, Optional, Dict, Any, Set, List
from collections import Counter

# Fix SSL_CERT_FILE pointing to non-existent path (known env issue)
if 'SSL_CERT_FILE' in os.environ and not os.path.exists(os.environ['SSL_CERT_FILE']):
    del os.environ['SSL_CERT_FILE']

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spacy
from sentence_transformers import SentenceTransformer, util

# Configure structured logging
logger = logging.getLogger("promptee.heuristic_scorer")


# ═══════════════════════════════════════════════════════════════════════════
# Output Schema
# ═══════════════════════════════════════════════════════════════════════════

class ScoreResult(TypedDict):
    """
    Full scoring output as specified in the Recommended Output Format (§10).

    Contains both raw and candidate quality scores, deltas, penalties,
    and the final gated improvement score.
    """
    # ── Per-prompt quality metrics ────────────────────────────────────
    raw_quality: float             # Quality score of the original prompt
    candidate_quality: float       # Quality score of the candidate (or raw if no candidate)

    # ── Component scores (computed on the target prompt) ──────────────
    clarity: float                 # 0.0–1.0, actionability ratio
    specificity: float             # 0.0–1.0, constraint density
    ambiguity_penalty: float       # 0.0–1.0, vague token ratio penalty
    redundancy_penalty: float      # 0.0–1.0, repeated token penalty
    length_penalty: float          # 0.0–1.0, short-prompt adjustment factor
    structural_bonus: float        # 0.0–0.10, reward for structured formatting

    # ── Improvement metrics ───────────────────────────────────────────
    quality_improvement: float     # candidate_quality - raw_quality
    clarity_delta: float           # candidate_clarity - raw_clarity
    specificity_delta: float       # candidate_specificity - raw_specificity

    # ── Semantic constraint ───────────────────────────────────────────
    semantic_preservation: float   # 0.0–1.0, cosine similarity
    rejected: bool                 # True if semantic < hard rejection floor

    # ── Final output ──────────────────────────────────────────────────
    final_score: float             # Gated improvement score (reward signal)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration Dataclass
# ═══════════════════════════════════════════════════════════════════════════

class ScorerConfig:
    """
    Centralised, configurable thresholds for the scoring pipeline.

    All defaults are tuned for English-language prompt optimisation and
    can be overridden at instantiation time for experimentation.
    """

    def __init__(
        self,
        # ── Core weights (clarity + specificity only) ─────────────────
        w_c: float = 0.5,
        w_s: float = 0.5,

        # ── Semantic gating thresholds ────────────────────────────────
        # Hard floor: below this → immediate rejection (§3 + §12)
        semantic_hard_floor: float = 0.40,
        # Soft threshold: between floor and this → improvement is scaled
        # by the semantic score itself (§12 soft penalty)
        semantic_soft_threshold: float = 0.70,

        # ── Specificity tuning ────────────────────────────────────────
        specificity_ideal_density: float = 0.25,

        # ── Penalty tuning ────────────────────────────────────────────
        ambiguity_max_penalty: float = 0.15,
        redundancy_max_penalty: float = 0.15,

        # ── Length normalization ───────────────────────────────────────
        # Prompts shorter than this token count receive a length penalty
        min_tokens_for_full_score: int = 8,

        # ── Structural bonus ──────────────────────────────────────────
        structural_bonus_cap: float = 0.10,
    ):
        # ── Validate weight normalization assumption ──────────────────
        # Weights must be positive; they will be dynamically normalised
        # at scoring time, but negative weights are nonsensical.
        if w_c < 0 or w_s < 0:
            raise ValueError("Weights w_c and w_s must be non-negative.")

        self.w_c = w_c
        self.w_s = w_s
        self.semantic_hard_floor = semantic_hard_floor
        self.semantic_soft_threshold = semantic_soft_threshold
        self.specificity_ideal_density = specificity_ideal_density
        self.ambiguity_max_penalty = ambiguity_max_penalty
        self.redundancy_max_penalty = redundancy_max_penalty
        self.min_tokens_for_full_score = min_tokens_for_full_score
        self.structural_bonus_cap = structural_bonus_cap


# ═══════════════════════════════════════════════════════════════════════════
# Ambiguity Token Set
# ═══════════════════════════════════════════════════════════════════════════

# Checklist §5: vague tokens that should be penalised.
# Kept as a module-level frozenset for immutability and O(1) lookup.
AMBIGUOUS_TOKENS: frozenset = frozenset({
    "some", "something", "thing", "things", "stuff",
    "various", "etc", "certain", "whatever", "somehow",
    "somewhat", "somewhere", "anywhere", "anything",
    "probably", "maybe", "kind", "sort", "lot",
    "basically", "generally", "usually", "approximately",
})

# ═══════════════════════════════════════════════════════════════════════════
# Structural Pattern Detectors
# ═══════════════════════════════════════════════════════════════════════════

# Checklist §11: patterns that indicate structured prompts.
_STRUCTURAL_PATTERNS: List = [
    re.compile(r"^\s*\d+[\.\)]\s", re.MULTILINE),             # Numbered steps
    re.compile(r"^\s*[-•*]\s", re.MULTILINE),                  # Bullet points
    re.compile(r"\b(?:act|role|persona)\s+(?:as|of)\b", re.I), # Role definitions
    re.compile(r"\b(?:format|output)\s*:", re.I),              # Output format instructions
    re.compile(r"\b(?:step\s+\d+|first|second|third|finally)\b", re.I),  # Sequential markers
]


# ═══════════════════════════════════════════════════════════════════════════
# Improved Heuristic Scorer
# ═══════════════════════════════════════════════════════════════════════════

class HeuristicScorer:
    """
    Deterministic heuristic scorer for prompt quality evaluation.

    Implements the full pipeline from the architecture specification and
    all 12 improvements from the improvement checklist:

      1. Separate quality score (clarity + specificity only)
      2. Quality improvement metric (candidate - raw)
      3. Semantic preservation as constraint gate
      4. Dynamic weight normalization
      5. Ambiguity penalty
      6. Improved actionability detection
      7. Redundancy penalty
      8. Length normalization
      9. Metric deltas
      10. Raw vs candidate quality output
      11. Structural bonus
      12. Soft semantic penalty
    """

    def __init__(self, config: Optional[ScorerConfig] = None):
        """
        Args:
            config: Scoring configuration. Uses sensible defaults if None.
        """
        self.config = config or ScorerConfig()

        # Instance-level model references (no global mutable state)
        self._nlp: Optional[Any] = None
        self._st_model: Optional[Any] = None
        self._load_models()

    # ── Model Loading ─────────────────────────────────────────────────

    def _load_models(self) -> None:
        """Load spaCy and sentence-transformer models with structured error logging."""
        try:
            self._nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model 'en_core_web_sm' loaded successfully.")
        except OSError:
            logger.error(
                "spaCy model 'en_core_web_sm' not found. "
                "Install with: python -m spacy download en_core_web_sm"
            )

        try:
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("SentenceTransformer 'all-MiniLM-L6-v2' loaded successfully.")
        except Exception as e:
            logger.error(f"SentenceTransformer failed to load: {e}")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 1: Clarity Scoring
    # ═══════════════════════════════════════════════════════════════════

    def _score_clarity(self, prompt: str) -> float:
        """
        SOP §3A — Clarity Score (0.0–1.0).

        Per-sentence analysis:
          1. A sentence is "actionable" if:
             a) It contains a ROOT verb with at least one dobj child (strict), OR
             b) It contains a ROOT verb in imperative mood — no subject required
                (Checklist §6: allows "Explain clearly", "Be concise", "Act as an expert").
          2. Ratio = actionable_sentences / total_sentences.
          3. Penalty: noun chunks that have no verb correlation reduce the score.

        Assumption: We detect imperative mood heuristically by checking whether
        the ROOT verb has no explicit subject (nsubj/nsubjpass). This is an
        approximation; a true imperative detector would require morphological
        analysis not available in en_core_web_sm.
        """
        if not self._nlp or not prompt.strip():
            return 0.0

        doc = self._nlp(prompt)
        sentences = list(doc.sents)
        if not sentences:
            return 0.0

        actionable_count: int = 0
        total_noun_chunks: int = 0
        correlated_noun_chunks: int = 0

        for sent in sentences:
            root_verb = None
            has_dobj: bool = False
            has_subject: bool = False
            sent_verbs: Set = set()

            for token in sent:
                # Find the ROOT of the sentence and check if it's a verb
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    root_verb = token
                    sent_verbs.add(token)
                # Track all verbs in the sentence for noun-chunk correlation
                if token.pos_ == "VERB":
                    sent_verbs.add(token)
                # Check for direct objects
                if token.dep_ == "dobj":
                    has_dobj = True
                # Check for explicit subjects (to detect imperative mood)
                if token.dep_ in ("nsubj", "nsubjpass"):
                    has_subject = True

            # Improvement §6: Accept imperative verbs without dobj.
            # A sentence is actionable if:
            #   (a) ROOT verb + direct object (original strict rule), OR
            #   (b) ROOT verb with no explicit subject (imperative heuristic)
            if root_verb and (has_dobj or not has_subject):
                actionable_count += 1

            # Noun-chunk penalty: chunks whose head is NOT a verb are "orphaned"
            for chunk in sent.noun_chunks:
                total_noun_chunks += 1
                if chunk.root.head in sent_verbs:
                    correlated_noun_chunks += 1

        # Base ratio: actionable sentences / total sentences
        ratio: float = actionable_count / len(sentences)

        # Penalty for excessive uncorrelated noun chunks
        if total_noun_chunks > 0:
            orphan_ratio: float = 1.0 - (correlated_noun_chunks / total_noun_chunks)
            penalty: float = orphan_ratio * 0.2  # Max 20% penalty
            ratio = max(ratio - penalty, 0.0)

        return min(ratio, 1.0)

    # ═══════════════════════════════════════════════════════════════════
    # Stage 2: Specificity Scoring
    # ═══════════════════════════════════════════════════════════════════

    def _score_specificity(self, prompt: str) -> float:
        """
        SOP §3B — Specificity Score (0.0–1.0).

        Measures constraint density using dependency relations:
          - amod: adjectival modifiers (e.g., "detailed report", "blue sky")
          - nummod: numeric modifiers (e.g., "5 paragraphs", "3 examples")
          - Named entities (ENT): specific references (e.g., "Python", "2024")

        Density = (amod_count + nummod_count + entity_count) / token_count
        Score = density / ideal_density (capped at 1.0)
        """
        if not self._nlp or not prompt.strip():
            return 0.0

        doc = self._nlp(prompt)
        if len(doc) == 0:
            return 0.0

        # Count dependency-based modifiers (SOP specifies amod and nummod)
        modifiers = [token for token in doc if token.dep_ in ("amod", "nummod")]
        ents = list(doc.ents)

        density: float = (len(modifiers) + len(ents)) / max(len(doc), 1.0)
        score: float = density / self.config.specificity_ideal_density
        return min(score, 1.0)

    # ═══════════════════════════════════════════════════════════════════
    # Stage 3: Ambiguity Penalty
    # ═══════════════════════════════════════════════════════════════════

    def _compute_ambiguity_penalty(self, prompt: str) -> float:
        """
        Checklist §5 — Ambiguity Penalty (0.0–max_penalty).

        Penalizes vague tokens that reduce prompt precision.
        The penalty is proportional to the ratio of ambiguous tokens
        to total tokens, scaled by the configured max penalty.

        Assumption: We match against lowercased lemmas to catch inflected
        forms (e.g., "things" → "thing"). Punctuation tokens are excluded
        from the denominator to avoid deflating the ratio.
        """
        if not self._nlp or not prompt.strip():
            return 0.0

        doc = self._nlp(prompt)
        # Exclude punctuation from the token count for a fairer ratio
        content_tokens = [t for t in doc if not t.is_punct and not t.is_space]
        if not content_tokens:
            return 0.0

        ambiguous_count: int = sum(
            1 for t in content_tokens if t.lemma_.lower() in AMBIGUOUS_TOKENS
        )
        ratio: float = ambiguous_count / len(content_tokens)
        return min(ratio * self.config.ambiguity_max_penalty / 0.15, self.config.ambiguity_max_penalty)

    # ═══════════════════════════════════════════════════════════════════
    # Stage 4: Redundancy Penalty
    # ═══════════════════════════════════════════════════════════════════

    def _compute_redundancy_penalty(self, prompt: str) -> float:
        """
        Checklist §7 — Redundancy Penalty (0.0–max_penalty).

        Penalizes repeated consecutive tokens (e.g., "very very detailed",
        "explain explain explain"). This prevents score gaming through
        token repetition.

        Algorithm:
          1. Extract lowercased content tokens (excluding punctuation/space).
          2. Count consecutive duplicate pairs (bigrams where both tokens match).
          3. Penalty = (duplicate_pairs / total_pairs) * max_penalty.

        Assumption: Only consecutive duplicates are penalised. Non-consecutive
        repetition (e.g., "explain X and explain Y") is considered legitimate
        and is not penalised.
        """
        if not self._nlp or not prompt.strip():
            return 0.0

        doc = self._nlp(prompt)
        content_tokens = [t.lower_ for t in doc if not t.is_punct and not t.is_space]

        if len(content_tokens) < 2:
            return 0.0

        # Count consecutive duplicate pairs
        duplicate_pairs: int = sum(
            1 for i in range(len(content_tokens) - 1)
            if content_tokens[i] == content_tokens[i + 1]
        )
        total_pairs: int = len(content_tokens) - 1
        ratio: float = duplicate_pairs / total_pairs
        return min(ratio * self.config.redundancy_max_penalty / 0.15, self.config.redundancy_max_penalty)

    # ═══════════════════════════════════════════════════════════════════
    # Stage 5: Length Normalization
    # ═══════════════════════════════════════════════════════════════════

    def _compute_length_penalty(self, prompt: str) -> float:
        """
        Checklist §8 — Length Normalization Factor (0.0–1.0).

        Returns a scaling factor that penalises extremely short prompts.
        Prompts with fewer tokens than `min_tokens_for_full_score` receive
        a proportionally reduced score.

        Formula:
          factor = min(token_count / min_tokens_for_full_score, 1.0)

        The returned value is the *reduction factor* (1.0 - factor), i.e.,
        how much penalty is applied. A factor of 0.0 means no penalty.

        Assumption: We count spaCy tokens (excluding punctuation) as the
        token count. This is more linguistically meaningful than whitespace
        splitting.
        """
        if not self._nlp or not prompt.strip():
            return 1.0  # Maximum penalty for empty prompts

        doc = self._nlp(prompt)
        content_tokens = [t for t in doc if not t.is_punct and not t.is_space]
        token_count: int = len(content_tokens)

        if token_count >= self.config.min_tokens_for_full_score:
            return 0.0  # No penalty

        # Linear scaling: fewer tokens → more penalty
        factor: float = token_count / self.config.min_tokens_for_full_score
        return round(1.0 - factor, 4)

    # ═══════════════════════════════════════════════════════════════════
    # Stage 6: Quality Score Calculation
    # ═══════════════════════════════════════════════════════════════════

    def _compute_quality(
        self,
        clarity: float,
        specificity: float,
        ambiguity_penalty: float,
        redundancy_penalty: float,
        length_penalty: float,
        structural_bonus: float,
    ) -> float:
        """
        Compute the composite quality score using ONLY clarity and specificity.

        Checklist §1 — Separate Quality Score:
          quality = (w_c' * clarity) + (w_s' * specificity)
          where w_c' and w_s' are dynamically normalised (§4).

        Checklist §4 — Dynamic Weight Normalization:
          w_c' = w_c / (w_c + w_s)
          w_s' = w_s / (w_c + w_s)

        This ensures the quality score stays in [0.0, 1.0] regardless of
        how weights are configured, and prevents score inflation from
        semantic preservation being mixed in.

        After computing the base quality, penalties are subtracted and
        the structural bonus is added (capped at 1.0).
        """
        w_sum: float = self.config.w_c + self.config.w_s
        if w_sum == 0:
            # Edge case: both weights are zero — degenerate config
            logger.warning("Both w_c and w_s are 0. Quality defaults to 0.0.")
            return 0.0

        # Dynamic weight normalization (§4)
        w_c_norm: float = self.config.w_c / w_sum
        w_s_norm: float = self.config.w_s / w_sum

        # Base quality from clarity + specificity
        base_quality: float = (w_c_norm * clarity) + (w_s_norm * specificity)

        # Apply penalties (subtractive, clamped to 0)
        penalised: float = base_quality - ambiguity_penalty - redundancy_penalty

        # Apply length penalty (multiplicative reduction)
        length_factor: float = 1.0 - length_penalty
        penalised *= length_factor

        # Apply structural bonus (additive, capped at 1.0)
        final: float = penalised + structural_bonus

        return max(min(final, 1.0), 0.0)

    # ═══════════════════════════════════════════════════════════════════
    # Stage 7: Structural Bonus
    # ═══════════════════════════════════════════════════════════════════

    def _compute_structural_bonus(self, prompt: str) -> float:
        """
        Checklist §11 — Structural Bonus (0.0–structural_bonus_cap).

        Rewards prompts that use structured formatting such as:
          - Numbered steps (e.g., "1. First step")
          - Bullet points (e.g., "- Item")
          - Role definitions (e.g., "Act as an expert")
          - Output format instructions (e.g., "Format: JSON")

        Each detected pattern contributes equally. The bonus is capped
        at the configured maximum.

        Assumption: We check for pattern *presence* (binary), not count.
        A prompt with 10 bullet points gets the same bonus as one with 1.
        This prevents gaming through excessive structure.
        """
        if not prompt.strip():
            return 0.0

        matched_patterns: int = sum(
            1 for pattern in _STRUCTURAL_PATTERNS if pattern.search(prompt)
        )

        # Each pattern contributes an equal fraction of the cap
        per_pattern: float = self.config.structural_bonus_cap / len(_STRUCTURAL_PATTERNS)
        return min(matched_patterns * per_pattern, self.config.structural_bonus_cap)

    # ═══════════════════════════════════════════════════════════════════
    # Stage 8: Semantic Constraint Gating
    # ═══════════════════════════════════════════════════════════════════

    def _score_semantic_preservation(self, raw_prompt: str, optimized_prompt: str) -> float:
        """
        SOP §3C — Semantic Preservation (0.0–1.0).

        Cosine similarity between contextual embeddings of raw and optimised prompts.
        This metric is used as a CONSTRAINT GATE (not mixed into quality).

        Gating logic (§3 + §12):
          - If sim < hard_floor (0.40): REJECT immediately
          - If sim < soft_threshold (0.70): scale improvement by sim
          - If sim >= soft_threshold: no penalty, full improvement score
        """
        if not self._st_model:
            logger.warning(
                "SentenceTransformer not available. Cannot compute semantic preservation."
            )
            return 0.0

        if not raw_prompt.strip() or not optimized_prompt.strip():
            return 0.0

        emb_raw = self._st_model.encode(raw_prompt)
        emb_opt = self._st_model.encode(optimized_prompt)
        sim: float = util.cos_sim(emb_raw, emb_opt).item()
        return max(sim, 0.0)

    # ═══════════════════════════════════════════════════════════════════
    # Stage 9 & 10: Metric Deltas + Rejection Logic
    # ═══════════════════════════════════════════════════════════════════
    # (Implemented within the public evaluate() method below)

    # ═══════════════════════════════════════════════════════════════════
    # Internal: Score a single prompt for quality components
    # ═══════════════════════════════════════════════════════════════════

    def _score_prompt(self, prompt: str) -> Dict[str, float]:
        """
        Compute all quality components for a single prompt.

        Returns a dict with: clarity, specificity, ambiguity_penalty,
        redundancy_penalty, length_penalty, structural_bonus, quality.
        """
        clarity: float = self._score_clarity(prompt)
        specificity: float = self._score_specificity(prompt)
        ambiguity_penalty: float = self._compute_ambiguity_penalty(prompt)
        redundancy_penalty: float = self._compute_redundancy_penalty(prompt)
        length_penalty: float = self._compute_length_penalty(prompt)
        structural_bonus: float = self._compute_structural_bonus(prompt)

        quality: float = self._compute_quality(
            clarity, specificity,
            ambiguity_penalty, redundancy_penalty,
            length_penalty, structural_bonus,
        )

        return {
            "clarity": clarity,
            "specificity": specificity,
            "ambiguity_penalty": ambiguity_penalty,
            "redundancy_penalty": redundancy_penalty,
            "length_penalty": length_penalty,
            "structural_bonus": structural_bonus,
            "quality": quality,
        }

    # ═══════════════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════════════

    def evaluate(self, raw_prompt: str, candidate_prompt: Optional[str] = None) -> ScoreResult:
        """
        Evaluate a prompt (or a raw/optimised pair) and return structured scores.

        Pipeline:
          1. Score the raw prompt for quality components.
          2. If candidate provided, score the candidate separately.
          3. Compute quality_improvement = candidate_quality - raw_quality.
          4. Compute semantic preservation between raw and candidate.
          5. Apply semantic gating (§3 + §12):
             - Below hard floor → reject + zero improvement
             - Below soft threshold → scale improvement by semantic score
             - Above soft threshold → full improvement
          6. Compute metric deltas.
          7. Return full ScoreResult.

        When only `raw_prompt` is provided:
          - All metrics are computed on the raw prompt.
          - Semantic preservation defaults to 1.0 (self-similarity).
          - quality_improvement, deltas are all 0.0.
          - Rejected is always False.

        Backward compatibility: The returned ScoreResult is a superset of
        the original interface. The 'total' key is replaced by 'final_score'
        to avoid confusion with the old semantics.
        """
        # ── Step 1: Score the raw prompt ──────────────────────────────
        raw_metrics: Dict[str, float] = self._score_prompt(raw_prompt)

        if candidate_prompt is None:
            # Raw-only evaluation: return baseline scores
            return ScoreResult(
                raw_quality=round(raw_metrics["quality"], 4),
                candidate_quality=round(raw_metrics["quality"], 4),
                clarity=round(raw_metrics["clarity"], 4),
                specificity=round(raw_metrics["specificity"], 4),
                ambiguity_penalty=round(raw_metrics["ambiguity_penalty"], 4),
                redundancy_penalty=round(raw_metrics["redundancy_penalty"], 4),
                length_penalty=round(raw_metrics["length_penalty"], 4),
                structural_bonus=round(raw_metrics["structural_bonus"], 4),
                quality_improvement=0.0,
                clarity_delta=0.0,
                specificity_delta=0.0,
                semantic_preservation=1.0,
                rejected=False,
                final_score=round(raw_metrics["quality"], 4),
            )

        # ── Step 2: Score the candidate prompt ────────────────────────
        cand_metrics: Dict[str, float] = self._score_prompt(candidate_prompt)

        # ── Step 3: Quality improvement (§2) ──────────────────────────
        quality_improvement: float = cand_metrics["quality"] - raw_metrics["quality"]

        # ── Step 4: Semantic preservation (§3C) ───────────────────────
        semantic: float = self._score_semantic_preservation(raw_prompt, candidate_prompt)

        # ── Step 5: Semantic gating (§3 + §12) ────────────────────────
        rejected: bool = False
        final_score: float = quality_improvement

        if semantic < self.config.semantic_hard_floor:
            # Hard rejection: meaning has drifted too far
            rejected = True
            final_score = 0.0
            logger.warning(
                f"Semantic preservation below hard floor: {semantic:.4f} < "
                f"{self.config.semantic_hard_floor}. Rewrite REJECTED."
            )
        elif semantic < self.config.semantic_soft_threshold:
            # Soft penalty: scale improvement by semantic similarity (§12)
            # This provides smoother optimisation behaviour and better
            # training stability than a hard cutoff at the soft threshold.
            final_score = quality_improvement * semantic
            logger.info(
                f"Semantic preservation below soft threshold: {semantic:.4f} < "
                f"{self.config.semantic_soft_threshold}. "
                f"Improvement scaled: {quality_improvement:.4f} → {final_score:.4f}"
            )
        # else: semantic >= soft_threshold → full improvement, no penalty

        # ── Step 6: Metric deltas (§9) ────────────────────────────────
        clarity_delta: float = cand_metrics["clarity"] - raw_metrics["clarity"]
        specificity_delta: float = cand_metrics["specificity"] - raw_metrics["specificity"]

        # ── Step 7: Assemble result ───────────────────────────────────
        return ScoreResult(
            raw_quality=round(raw_metrics["quality"], 4),
            candidate_quality=round(cand_metrics["quality"], 4),
            clarity=round(cand_metrics["clarity"], 4),
            specificity=round(cand_metrics["specificity"], 4),
            ambiguity_penalty=round(cand_metrics["ambiguity_penalty"], 4),
            redundancy_penalty=round(cand_metrics["redundancy_penalty"], 4),
            length_penalty=round(cand_metrics["length_penalty"], 4),
            structural_bonus=round(cand_metrics["structural_bonus"], 4),
            quality_improvement=round(quality_improvement, 4),
            clarity_delta=round(clarity_delta, 4),
            specificity_delta=round(specificity_delta, 4),
            semantic_preservation=round(semantic, 4),
            rejected=rejected,
            final_score=round(final_score, 4),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Standalone Test / Demo
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    scorer = HeuristicScorer()

    print("=" * 70)
    print("DEMO: Heuristic Scorer — Improved Pipeline")
    print("=" * 70)

    # ── Test Case 1: Weak prompt ──────────────────────────────────────
    weak = "tell me something about stuff"
    weak_optimised = "explain some things"
    print("\n[WEAK PROMPT]")
    print(f"  Raw: {weak!r}")
    print(f"  Candidate: {weak_optimised!r}")
    result = scorer.evaluate(weak, weak_optimised)
    for k, v in result.items():
        print(f"  {k}: {v}")

    # ── Test Case 2: Moderate prompt ──────────────────────────────────
    moderate = "Write a summary of machine learning"
    moderate_optimised = "Write a concise 200-word summary of supervised machine learning, covering key algorithms and their applications."
    print("\n[MODERATE PROMPT]")
    print(f"  Raw: {moderate!r}")
    print(f"  Candidate: {moderate_optimised!r}")
    result = scorer.evaluate(moderate, moderate_optimised)
    for k, v in result.items():
        print(f"  {k}: {v}")

    # ── Test Case 3: Optimised prompt ─────────────────────────────────
    optimised_raw = "Explain neural networks"
    optimised_cand = (
        "Act as a computer science professor. Explain the architecture of "
        "feedforward neural networks in 3 sections:\n"
        "1. Input layer and feature representation\n"
        "2. Hidden layers and activation functions (ReLU, sigmoid)\n"
        "3. Output layer and loss computation\n"
        "Format: Use bullet points for key concepts. Limit to 500 words."
    )
    print("\n[OPTIMISED PROMPT]")
    print(f"  Raw: {optimised_raw!r}")
    print(f"  Candidate: {optimised_cand!r}")
    result = scorer.evaluate(optimised_raw, optimised_cand)
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 70)
