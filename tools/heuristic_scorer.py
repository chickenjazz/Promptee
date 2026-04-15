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
from typing import TypedDict, Optional, Dict, Any, List
from collections import Counter

# Fix SSL_CERT_FILE pointing to non-existent path (known env issue)
if 'SSL_CERT_FILE' in os.environ and not os.path.exists(os.environ['SSL_CERT_FILE']):
    del os.environ['SSL_CERT_FILE']

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spacy
from sentence_transformers import SentenceTransformer, util
from spellchecker import SpellChecker

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

        # ── Clarity scoring tuning ────────────────────────────────────
        weak_verb_weight: float = 0.5,
        noun_orphan_penalty_factor: float = 0.10,
        typo_penalty_per_token: float = 0.005,
        enable_typo_penalty: bool = False,

        # ── Clarity component weights (v2 structure-aware scoring) ────
        clarity_actionability_weight: float = 0.35,
        clarity_structure_weight: float = 0.30,
        clarity_completeness_weight: float = 0.25,
        clarity_max_fragment_penalty: float = 0.10,
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
        self.weak_verb_weight = weak_verb_weight
        self.noun_orphan_penalty_factor = noun_orphan_penalty_factor
        self.typo_penalty_per_token = typo_penalty_per_token
        self.enable_typo_penalty = enable_typo_penalty
        self.clarity_actionability_weight = clarity_actionability_weight
        self.clarity_structure_weight = clarity_structure_weight
        self.clarity_completeness_weight = clarity_completeness_weight
        self.clarity_max_fragment_penalty = clarity_max_fragment_penalty


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
# Clarity — Verb Classification Sets
# ═══════════════════════════════════════════════════════════════════════════

# Modal verbs that indicate instructional intent even in declarative form.
MODAL_VERBS: frozenset = frozenset({
    "should", "must", "shall", "ought", "need",
})

# Strong action verbs — full weight (1.0) in clarity scoring.
STRONG_VERBS: frozenset = frozenset({
    "explain", "generate", "create", "compare", "analyze", "analyse",
    "implement", "define", "list", "describe", "summarize", "summarise",
    "classify", "evaluate", "compute", "build", "design", "optimize",
    "optimise", "write", "produce", "develop", "construct", "extract",
    "identify", "transform", "convert", "calculate", "derive", "outline",
    "specify", "translate", "rewrite", "revise", "edit", "format",
    "organize", "organise", "categorize", "categorise", "rank", "sort",
    "filter", "validate", "verify", "test", "debug", "refactor",
    "diagram", "illustrate", "demonstrate", "prove", "solve",
})

# Weak action verbs — reduced weight in clarity scoring.
WEAK_VERBS: frozenset = frozenset({
    "discuss", "talk", "consider", "think", "wonder", "mention",
    "touch", "look", "see", "try", "feel", "seem", "appear",
    "note", "recall", "remember", "ponder", "reflect",
})

# ═══════════════════════════════════════════════════════════════════════════
# Clarity — Implicit Command Detection
# ═══════════════════════════════════════════════════════════════════════════

# Detects structural implicit commands like "Step 1: Data preprocessing"
IMPLICIT_COMMAND_PATTERN = re.compile(
    r"^\s*(?:step|phase|task|part|stage)\s+\d+\s*[:.\-\u2013\u2014]\s*",
    re.IGNORECASE,
)

# ═══════════════════════════════════════════════════════════════════════════
# Clarity — Sentence Fragment Indicators
# ═══════════════════════════════════════════════════════════════════════════

# Tokens/patterns that indicate a sentence is a fragment to be attached
# to the previous sentence rather than counted independently.
FRAGMENT_INDICATORS: frozenset = frozenset({
    # Single-word fragment starters
    "using", "including", "especially", "particularly",
    "namely", "specifically",
    # Two-word fragment starters (checked as "word1 word2")
    "such as", "for example", "for instance",
    # Abbreviation forms
    "e.g.", "i.e.",
})


# ═══════════════════════════════════════════════════════════════════════════
# Clarity — Structure-Aware Detection Patterns
# ═══════════════════════════════════════════════════════════════════════════

# Markdown header detection (# Heading, ## Subheading, etc.)
_HEADER_PATTERN = re.compile(r"^\s*#{1,6}\s+\S", re.MULTILINE)

# Labeled section detection (Role:, Objective:, Requirements:, etc.)
_LABELED_SECTION_PATTERN = re.compile(
    r"^\s*(?:role|objective|goal|task|purpose|requirements?"
    r"|constraints?|limitations?|output|format|context"
    r"|background|instructions?|deliverables?|criteria"
    r"|scope|audience|tone|style|examples?)\s*[:]\s*",
    re.IGNORECASE | re.MULTILINE,
)

# Bullet item detection (- item, * item, bullet item)
_BULLET_ITEM_PATTERN = re.compile(r"^\s*[-*\u2022]\s+", re.MULTILINE)

# Numbered item detection (1. item, 2) item)
_NUMBERED_ITEM_PATTERN = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)

# ═══════════════════════════════════════════════════════════════════════════
# Clarity — Completeness Component Patterns
# ═══════════════════════════════════════════════════════════════════════════

# Each pattern detects a complementary instruction component.
# Score = detected_count / total_count (5 components).
_COMPLETENESS_PATTERNS: Dict[str, re.Pattern] = {
    "role": re.compile(
        r"\b(?:act\s+as|role|persona|you\s+are|imagine\s+you(?:'re|\s+are))\b",
        re.IGNORECASE,
    ),
    "objective": re.compile(
        r"\b(?:objective|goal|task|purpose|aim|mission)\b",
        re.IGNORECASE,
    ),
    "requirements": re.compile(
        r"\b(?:requirements?|must|should|need\s+to|shall|ensure|include)\b",
        re.IGNORECASE,
    ),
    "constraints": re.compile(
        r"\b(?:constraints?|limit(?:ation)?s?|restrict(?:ion)?s?|within"
        r"|maximum|minimum|at\s+most|at\s+least|no\s+more\s+than"
        r"|no\s+fewer\s+than|avoid|do\s+not|don't)\b",
        re.IGNORECASE,
    ),
    "output_format": re.compile(
        r"\b(?:output|format|return|respond|deliver(?:able)?|provide|present|display)"
        r"\s*(?:as|in|using|:)?\b",
        re.IGNORECASE,
    ),
}


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
        self._spellchecker: Optional[Any] = None
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

        # Load pyspellchecker for optional typo detection
        try:
            self._spellchecker = SpellChecker()
            logger.info("SpellChecker loaded successfully.")
        except Exception as e:
            logger.warning(f"SpellChecker failed to load: {e}")
            self._spellchecker = None

    # ═══════════════════════════════════════════════════════════════════
    # Stage 1: Clarity Scoring (v3 — Structure-Aware)
    # ═══════════════════════════════════════════════════════════════════

    def _extract_instruction_units(self, prompt: str) -> List[str]:
        """
        Split a prompt into instruction units: individual sentences,
        bullet items, numbered items, headers, and labeled sections.

        Bullet/numbered/header/labeled lines become their own units.
        Prose lines are grouped and split into sentences via spaCy,
        with fragment attachment applied.

        Returns:
            List of non-empty instruction unit strings.
        """
        lines = prompt.split("\n")
        units: List[str] = []
        prose_block: List[str] = []

        structural_line = re.compile(
            r"^\s*(?:[-*\u2022]\s+|\d+[.)]\s+|#{1,6}\s+\S)", re.UNICODE
        )

        def flush_prose(block: List[str]) -> None:
            """Parse accumulated prose lines into sentence units."""
            if not block:
                return
            text = " ".join(block)
            if not text.strip():
                return
            doc = self._nlp(text)
            raw_sents = list(doc.sents)
            # Fragment attachment
            merged: List = []
            for sent in raw_sents:
                first_token = sent[0].text.lower() if len(sent) > 0 else ""
                first_two = (
                    f"{sent[0].text.lower()} {sent[1].text.lower()}"
                    if len(sent) > 1 else first_token
                )
                is_frag = (
                    first_token in FRAGMENT_INDICATORS
                    or first_two in FRAGMENT_INDICATORS
                )
                if is_frag and merged:
                    prev = merged[-1]
                    merged[-1] = doc[prev.start:sent.end]
                else:
                    merged.append(sent)
            for sent in merged:
                txt = sent.text.strip()
                if txt:
                    units.append(txt)

        for line in lines:
            stripped = line.strip()
            if not stripped:
                flush_prose(prose_block)
                prose_block = []
            elif structural_line.match(line) or _LABELED_SECTION_PATTERN.match(line):
                flush_prose(prose_block)
                prose_block = []
                units.append(stripped)
            else:
                prose_block.append(stripped)

        flush_prose(prose_block)
        return units

    def _compute_actionability(
        self, units: List[str]
    ) -> tuple:
        """
        Compute per-unit actionability score (0.0–1.0).

        Each instruction unit is checked for actionable verbs, modal verbs,
        passive constructions, implicit commands, or labeled section patterns.
        Score = actionable_units / total_units.

        Returns:
            (score, diagnostics_dict)
        """
        if not units:
            return 0.0, {
                'verb_count': 0, 'modal_count': 0, 'passive_count': 0,
                'implicit_commands': 0, 'actionable_units': 0, 'total_units': 0,
            }

        total_verb_count = 0
        total_modal_count = 0
        total_passive_count = 0
        total_implicit_commands = 0
        actionable_units = 0

        for unit_text in units:
            unit_is_actionable = False

            # Check for labeled section (e.g., "Role:", "Constraints:")
            if _LABELED_SECTION_PATTERN.match(unit_text):
                unit_is_actionable = True

            # Check for implicit command (e.g., "Step 1: Data preprocessing")
            if IMPLICIT_COMMAND_PATTERN.match(unit_text):
                total_implicit_commands += 1
                unit_is_actionable = True

            # spaCy analysis for verbs
            doc = self._nlp(unit_text)
            for token in doc:
                if token.pos_ == "VERB":
                    has_subject = any(
                        child.dep_ in ("nsubj", "nsubjpass")
                        for child in token.children
                    )
                    has_dobj = any(
                        child.dep_ == "dobj" for child in token.children
                    )
                    is_actionable_verb = (
                        has_dobj or not has_subject or token.dep_ == "ROOT"
                    )
                    if is_actionable_verb:
                        total_verb_count += 1
                        unit_is_actionable = True

                if token.pos_ == "AUX" and token.lemma_.lower() in MODAL_VERBS:
                    total_modal_count += 1
                    has_governed = any(
                        child.pos_ == "VERB" for child in token.children
                    ) or token.head.pos_ == "VERB"
                    if has_governed:
                        unit_is_actionable = True

                if token.dep_ == "nsubjpass" and token.head.pos_ == "VERB":
                    total_passive_count += 1

            if unit_is_actionable:
                actionable_units += 1

        score = actionable_units / len(units)
        return min(score, 1.0), {
            'verb_count': total_verb_count,
            'modal_count': total_modal_count,
            'passive_count': total_passive_count,
            'implicit_commands': total_implicit_commands,
            'actionable_units': actionable_units,
            'total_units': len(units),
        }

    def _compute_structure_score(self, prompt: str) -> tuple:
        """
        Compute structure diversity score (0.0–1.0).

        Checks for presence of 5 structural element types:
          headers, bullets, numbered lists, labeled sections, step patterns.
        More diverse structure → higher score.

        Returns:
            (score, diagnostics_dict)
        """
        checks = {
            'has_headers': bool(_HEADER_PATTERN.search(prompt)),
            'has_bullets': bool(_BULLET_ITEM_PATTERN.search(prompt)),
            'has_numbered': bool(_NUMBERED_ITEM_PATTERN.search(prompt)),
            'has_labeled_sections': bool(_LABELED_SECTION_PATTERN.search(prompt)),
            'has_step_patterns': bool(re.search(
                r"\b(?:step\s+\d+|first|second|third|finally)\b",
                prompt, re.IGNORECASE,
            )),
        }
        count = sum(checks.values())
        # Graduated scoring: more diverse structure types → higher score
        score_map = [0.0, 0.30, 0.55, 0.75, 0.90, 1.0]
        score = score_map[min(count, 5)]
        return score, checks

    def _compute_completeness(self, prompt: str) -> tuple:
        """
        Compute instruction completeness score (0.0–1.0).

        Checks for presence of 5 complementary instruction components:
          role, objective, requirements, constraints, output_format.
        Score = detected / 5.

        Returns:
            (score, diagnostics_dict)
        """
        detected: Dict[str, bool] = {}
        for name, pattern in _COMPLETENESS_PATTERNS.items():
            detected[name] = bool(pattern.search(prompt))
        count = sum(detected.values())
        score = count / len(_COMPLETENESS_PATTERNS)
        return min(score, 1.0), detected

    def _compute_weak_fragment_penalty(
        self, prompt: str, units: List[str]
    ) -> tuple:
        """
        Compute fragment penalty (0.0–max_penalty).

        Only applies to unstructured prompts. If the prompt contains any
        structural elements (bullets, headers, labeled sections), no
        penalty is applied. For unstructured text, penalises units that
        lack both verbs and structural context.

        Returns:
            (penalty, diagnostics_dict)
        """
        # Check if prompt has structural context
        is_structured = (
            bool(_HEADER_PATTERN.search(prompt))
            or bool(_BULLET_ITEM_PATTERN.search(prompt))
            or bool(_NUMBERED_ITEM_PATTERN.search(prompt))
            or bool(_LABELED_SECTION_PATTERN.search(prompt))
        )

        if is_structured or not units:
            return 0.0, {'structured': is_structured, 'fragment_count': 0}

        # Count verbless fragments in unstructured text
        fragment_count = 0
        for unit_text in units:
            doc = self._nlp(unit_text)
            has_verb = any(t.pos_ == "VERB" for t in doc)
            has_modal = any(
                t.pos_ == "AUX" and t.lemma_.lower() in MODAL_VERBS
                for t in doc
            )
            is_implicit = bool(IMPLICIT_COMMAND_PATTERN.match(unit_text))
            if not has_verb and not has_modal and not is_implicit:
                fragment_count += 1

        ratio = fragment_count / len(units)
        penalty = ratio * self.config.clarity_max_fragment_penalty
        return min(penalty, self.config.clarity_max_fragment_penalty), {
            'structured': False,
            'fragment_count': fragment_count,
        }

    def _score_clarity(self, prompt: str) -> Dict[str, Any]:
        """
        Structure-Aware Clarity Score (0.0–1.0) with diagnostics.

        Measures instructional completeness and execution readiness using
        a multi-component formula:

          clarity = actionability * w_a + structure * w_s + completeness * w_c
                  - weak_fragment_penalty

        Components:
          - actionability: per-unit verb/command detection (not per-token)
          - structure: diversity of formatting elements (headers, bullets, etc.)
          - completeness: presence of instruction components (role, objective, etc.)
          - fragment_penalty: only applied to unstructured verbless text

        Returns:
            dict with 'score' (float 0.0–1.0) and diagnostic counters.
        """
        empty_result: Dict[str, Any] = {
            'score': 0.0,
            'actionability': 0.0, 'structure': 0.0, 'completeness': 0.0,
            'fragment_penalty': 0.0,
            'verb_count': 0, 'modal_count': 0, 'passive_count': 0,
            'implicit_commands': 0, 'actionable_units': 0, 'total_units': 0,
            'detected_components': {}, 'detected_structures': {},
        }

        if not self._nlp or not prompt.strip():
            return empty_result

        # Step 1: Extract instruction units
        units = self._extract_instruction_units(prompt)
        if not units:
            return empty_result

        # Step 2: Compute each component
        actionability, act_diag = self._compute_actionability(units)
        structure, struct_diag = self._compute_structure_score(prompt)
        completeness, comp_diag = self._compute_completeness(prompt)
        frag_penalty, frag_diag = self._compute_weak_fragment_penalty(prompt, units)

        # Step 3: Weighted combination
        cfg = self.config
        score = (
            actionability * cfg.clarity_actionability_weight
            + structure * cfg.clarity_structure_weight
            + completeness * cfg.clarity_completeness_weight
            - frag_penalty
        )
        score = max(min(score, 1.0), 0.0)

        logger.debug(
            f"Clarity analysis: score={score:.4f}, "
            f"actionability={actionability:.4f}, structure={structure:.4f}, "
            f"completeness={completeness:.4f}, frag_penalty={frag_penalty:.4f}, "
            f"verbs={act_diag['verb_count']}, modals={act_diag['modal_count']}, "
            f"passive={act_diag['passive_count']}, "
            f"implicit_cmds={act_diag['implicit_commands']}"
        )

        return {
            'score': round(score, 4),
            'actionability': round(actionability, 4),
            'structure': round(structure, 4),
            'completeness': round(completeness, 4),
            'fragment_penalty': round(frag_penalty, 4),
            # Legacy diagnostic keys (backward compat)
            'verb_count': act_diag['verb_count'],
            'modal_count': act_diag['modal_count'],
            'passive_count': act_diag['passive_count'],
            'implicit_commands': act_diag['implicit_commands'],
            # New diagnostic keys
            'actionable_units': act_diag['actionable_units'],
            'total_units': act_diag['total_units'],
            'detected_components': comp_diag,
            'detected_structures': struct_diag,
        }

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

    def _score_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Compute all quality components for a single prompt.

        Returns a dict with: clarity, specificity, ambiguity_penalty,
        redundancy_penalty, length_penalty, structural_bonus, quality,
        and clarity diagnostics (verb_count, modal_count, passive_count,
        implicit_commands).
        """
        clarity_result = self._score_clarity(prompt)
        clarity: float = clarity_result['score']
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
            # Clarity diagnostics (legacy)
            "clarity_verb_count": clarity_result['verb_count'],
            "clarity_modal_count": clarity_result['modal_count'],
            "clarity_passive_count": clarity_result['passive_count'],
            "clarity_implicit_commands": clarity_result['implicit_commands'],
            # Clarity diagnostics (v3 structure-aware)
            "clarity_actionability": clarity_result.get('actionability', 0.0),
            "clarity_structure": clarity_result.get('structure', 0.0),
            "clarity_completeness": clarity_result.get('completeness', 0.0),
            "clarity_fragment_penalty": clarity_result.get('fragment_penalty', 0.0),
            "clarity_actionable_units": clarity_result.get('actionable_units', 0),
            "clarity_total_units": clarity_result.get('total_units', 0),
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

    # ── Test Case 4: Mandatory validation — structured must beat vague ─
    raw_vague = "build a website for my car wash business"
    structured_candidate = (
        "Act as a senior full-stack web developer.\n"
        "\n"
        "## Objective\n"
        "Build a responsive business website for a car wash service.\n"
        "\n"
        "## Requirements\n"
        "- Hero section with call-to-action button\n"
        "- Service listing with prices\n"
        "- Online booking form\n"
        "- Mobile-first, high-contrast design\n"
        "- Contact section with embedded Google Maps\n"
        "\n"
        "## Constraints\n"
        "- Output a single HTML file with inline CSS and JS\n"
        "- Must load under 3 seconds on mobile\n"
        "- No external frameworks or CDN dependencies\n"
        "\n"
        "## Output Format\n"
        "Deliver the complete HTML file with comments explaining each section."
    )
    print("\n[MANDATORY VALIDATION: Structured vs Vague]")
    print(f"  Raw:       {raw_vague!r}")
    print(f"  Candidate: {structured_candidate[:80]!r}...")
    result = scorer.evaluate(raw_vague, structured_candidate)
    for k, v in result.items():
        print(f"  {k}: {v}")

    # Compute individual clarity scores for comparison
    raw_metrics = scorer._score_prompt(raw_vague)
    cand_metrics = scorer._score_prompt(structured_candidate)
    print(f"\n  >> Raw clarity:       {raw_metrics['clarity']:.4f}")
    print(f"  >> Candidate clarity: {cand_metrics['clarity']:.4f}")
    print(f"  >> Raw quality:       {raw_metrics['quality']:.4f}")
    print(f"  >> Candidate quality: {cand_metrics['quality']:.4f}")
    assert cand_metrics['clarity'] > raw_metrics['clarity'], \
        f"FAIL: candidate clarity ({cand_metrics['clarity']}) must exceed raw clarity ({raw_metrics['clarity']})"
    assert cand_metrics['quality'] > raw_metrics['quality'], \
        f"FAIL: candidate quality ({cand_metrics['quality']}) must exceed raw quality ({raw_metrics['quality']})"
    assert result['final_score'] > 0, \
        f"FAIL: final_score ({result['final_score']}) must be positive"
    print("  >> All mandatory assertions PASSED")

    print("\n" + "=" * 70)
