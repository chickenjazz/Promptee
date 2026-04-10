import os
import sys
import logging
from typing import TypedDict, Optional

# Fix SSL_CERT_FILE pointing to non-existent path (known env issue)
if 'SSL_CERT_FILE' in os.environ and not os.path.exists(os.environ['SSL_CERT_FILE']):
    del os.environ['SSL_CERT_FILE']

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spacy
from sentence_transformers import SentenceTransformer, util

# Configure structured logging
logger = logging.getLogger("promptee.heuristic_scorer")


class ScoreResult(TypedDict):
    clarity: float
    specificity: float
    semantic_preservation: float
    total: float
    rejected: bool


class HeuristicScorer:
    """
    Deterministic heuristic scorer for prompt quality evaluation.
    
    Implements the scoring logic defined in architecture/heuristic_scorer.md:
      - Clarity: per-sentence actionability (ROOT verb + dobj), penalized for orphan noun chunks
      - Specificity: density of constraint modifiers (amod, nummod) + named entities
      - Semantic Preservation: cosine similarity gating via sentence-transformers
    """

    def __init__(
        self,
        w_c: float = 0.4,
        w_s: float = 0.4,
        w_p: float = 0.2,
        sim_threshold: float = 0.70,
        specificity_ideal_density: float = 0.25,
    ):
        """
        Args:
            w_c: Weight for clarity in composite score.
            w_s: Weight for specificity in composite score.
            w_p: Weight for semantic preservation in composite score (when available).
            sim_threshold: Minimum cosine similarity for semantic preservation gate.
            specificity_ideal_density: Target modifier+entity density (denominator for normalization).
        """
        self.w_c = w_c
        self.w_s = w_s
        self.w_p = w_p
        self.sim_threshold = sim_threshold
        self.specificity_ideal_density = specificity_ideal_density

        # Instance-level model references (no global mutable state)
        self._nlp = None
        self._st_model = None
        self._load_models()

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

    # ── Clarity ──────────────────────────────────────────────────────────

    def _score_clarity(self, prompt: str) -> float:
        """
        SOP §3A — Clarity Score (0.0–1.0).

        Per-sentence analysis:
          1. A sentence is "actionable" if it contains a ROOT verb with at least one dobj child.
          2. Ratio = actionable_sentences / total_sentences.
          3. Penalty: noun chunks that have no verb correlation reduce the score.
        """
        if not self._nlp or not prompt.strip():
            return 0.0

        doc = self._nlp(prompt)
        sentences = list(doc.sents)
        if not sentences:
            return 0.0

        actionable_count = 0
        total_noun_chunks = 0
        correlated_noun_chunks = 0

        for sent in sentences:
            root_verb = None
            has_dobj = False
            sent_verbs = set()

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

            if root_verb and has_dobj:
                actionable_count += 1

            # Noun-chunk penalty: chunks whose head is NOT a verb are "orphaned"
            for chunk in sent.noun_chunks:
                total_noun_chunks += 1
                if chunk.root.head in sent_verbs:
                    correlated_noun_chunks += 1

        # Base ratio: actionable sentences / total sentences
        ratio = actionable_count / len(sentences)

        # Penalty for excessive uncorrelated noun chunks
        if total_noun_chunks > 0:
            orphan_ratio = 1.0 - (correlated_noun_chunks / total_noun_chunks)
            penalty = orphan_ratio * 0.2  # Max 20% penalty
            ratio = max(ratio - penalty, 0.0)

        return min(ratio, 1.0)

    # ── Specificity ──────────────────────────────────────────────────────

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

        density = (len(modifiers) + len(ents)) / max(len(doc), 1.0)
        score = density / self.specificity_ideal_density
        return min(score, 1.0)

    # ── Semantic Preservation ────────────────────────────────────────────

    def _score_semantic_preservation(self, raw_prompt: str, optimized_prompt: str) -> float:
        """
        SOP §3C — Semantic Preservation (0.0–1.0).

        Cosine similarity between contextual embeddings of raw and optimized prompts.
        Acts as a gating threshold: if score < sim_threshold, rewrite is REJECTED.
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
        sim = util.cos_sim(emb_raw, emb_opt).item()
        return max(sim, 0.0)

    # ── Public API ───────────────────────────────────────────────────────

    def evaluate(self, raw_prompt: str, candidate_prompt: str = None) -> ScoreResult:
        """
        Evaluate a prompt (or a raw/optimized pair) and return structured scores.

        When `candidate_prompt` is provided:
          - Clarity and specificity are computed on the candidate.
          - Semantic preservation is computed between raw and candidate.
          - If semantic preservation < threshold, the result is marked as rejected.
          - Semantic preservation is incorporated into the total score.

        When only `raw_prompt` is provided:
          - Clarity and specificity are computed on the raw prompt.
          - Semantic preservation defaults to 1.0 (baseline self-similarity).
          - Rejected is always False.
        """
        target = candidate_prompt if candidate_prompt else raw_prompt
        clarity = self._score_clarity(target)
        specificity = self._score_specificity(target)

        rejected = False

        if candidate_prompt:
            sim = self._score_semantic_preservation(raw_prompt, candidate_prompt)
            # Gate: reject if below threshold
            if sim < self.sim_threshold:
                rejected = True
                logger.warning(
                    f"Semantic preservation below threshold: {sim:.4f} < {self.sim_threshold}. "
                    f"Rewrite rejected."
                )
            # Composite includes semantic preservation weight
            total = (self.w_c * clarity) + (self.w_s * specificity) + (self.w_p * sim)
        else:
            sim = 1.0  # Self-similarity baseline
            # No semantic weight for raw-only scoring (2 components only)
            total = (self.w_c * clarity) + (self.w_s * specificity) + (self.w_p * sim)

        return ScoreResult(
            clarity=round(clarity, 4),
            specificity=round(specificity, 4),
            semantic_preservation=round(sim, 4),
            total=round(total, 4),
            rejected=rejected,
        )
