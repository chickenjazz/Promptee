"""
Improved Heuristic Scorer — Deterministic Prompt Quality Evaluation Pipeline.

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

    # ── Diagnostics (Target Prompt) ───────────────────────────────────
    clarity_actionability: float
    clarity_structure: float
    clarity_completeness: float
    clarity_fragment_penalty: float
    specificity_modifiers: float
    specificity_entities: float
    specificity_ranges: float
    specificity_formats: float
    specificity_tools: float
    specificity_negation: float
    specificity_persona: float
    specificity_coverage: float
    specificity_intensity: float

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
        # Raised from 0.25 to 0.50 to account for weighted constraint
        # signals (entities 1.2×, ranges 1.5×, etc.) in the redesigned
        # specificity scorer.  The old value caused short prompts with
        # even one signal to saturate at 1.0.
        specificity_ideal_density: float = 0.50,

        # Length floor for specificity: prompts with fewer informative
        # tokens than this are linearly down-scaled. Prevents tiny prompts
        # anchored on a single named entity (e.g. "Write an essay about
        # Leonardo Da Vinci") from pinning specificity at 1.0 — density
        # alone is not a reliable signal on small samples.
        specificity_length_floor: int = 12,

        # ── Penalty tuning ────────────────────────────────────────────
        ambiguity_max_penalty: float = 0.15,
        redundancy_max_penalty: float = 0.05,

        # ── Length normalization ───────────────────────────────────────
        # Prompts shorter than this token count receive a length penalty
        min_tokens_for_full_score: int = 8,

        # ── Structural bonus ──────────────────────────────────────────
        structural_bonus_cap: float = 0.10,

        # ── Clarity scoring tuning ────────────────────────────────────
        weak_verb_weight: float = 0.5,
        modal_verb_weight: float = 0.75,
        noun_orphan_penalty_factor: float = 0.10,
        typo_penalty_per_token: float = 0.005,
        enable_typo_penalty: bool = False,

        # ── Clarity component weights (v2 structure-aware scoring) ────
        clarity_actionability_weight: float = 0.55,
        clarity_structure_weight: float = 0.30,
        clarity_completeness_weight: float = 0.15,
        clarity_max_fragment_penalty: float = 0.10,

        # ── Actionability verb-density tuning ─────────────────────────
        # Soft cap on weighted verb score per instruction unit.
        # Prevents verbosity gaming (stuffing 8 verbs into one sentence).
        # A unit with verb_score >= cap is treated as maximally actionable.
        actionability_verb_cap: float = 1.0,
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
        self.specificity_length_floor = specificity_length_floor
        self.ambiguity_max_penalty = ambiguity_max_penalty
        self.redundancy_max_penalty = redundancy_max_penalty
        self.min_tokens_for_full_score = min_tokens_for_full_score
        self.structural_bonus_cap = structural_bonus_cap
        self.weak_verb_weight = weak_verb_weight
        self.modal_verb_weight = modal_verb_weight
        self.noun_orphan_penalty_factor = noun_orphan_penalty_factor
        self.typo_penalty_per_token = typo_penalty_per_token
        self.enable_typo_penalty = enable_typo_penalty
        self.clarity_actionability_weight = clarity_actionability_weight
        self.clarity_structure_weight = clarity_structure_weight
        self.clarity_completeness_weight = clarity_completeness_weight
        self.clarity_max_fragment_penalty = clarity_max_fragment_penalty
        self.actionability_verb_cap = actionability_verb_cap


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
    "diagram", "illustrate", "demonstrate", "prove", "solve", "include", "exclude",
    "recommend", "suggest", "propose", "argue", "justify", "support", "oppose",
    "follow"
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
# Expanded with synonyms so prompts using Goal/Instructions/Guidelines
# are recognised identically to Objective/Requirements.
_LABELED_SECTION_PATTERN = re.compile(
    r"^\s*(?:role|objective|goal|task|purpose|requirements?"
    r"|constraints?|limitations?|output|format|context"
    r"|background|instructions?|deliverables?|criteria"
    r"|scope|audience|tone|style|examples?"
    r"|guidelines?|specifications?|rules|directions?"
    r"|expectations?|boundaries|guardrails"
    r"|target|intent|mission|aim"
    r"|expected\s+(?:response|result|output|format)"
    r"|response\s+format|answer\s+format|desired\s+(?:format|outcome)"
    r")\s*[:]\s*",
    re.IGNORECASE | re.MULTILINE,
)

# Content-free labeled section header — matches lines where the label
# has NO instruction content after the colon (e.g. "ROLE:\n" or "Requirements:  ").
# Used to detect pure metadata headers that should not count as instruction units.
_CONTENT_FREE_LABEL_RE = re.compile(
    r"^\s*(?:role|objective|goal|task|purpose|requirements?"
    r"|constraints?|limitations?|output|format|context"
    r"|background|instructions?|deliverables?|criteria"
    r"|scope|audience|tone|style|examples?"
    r"|guidelines?|specifications?|rules|directions?"
    r"|expectations?|boundaries|guardrails"
    r"|target|intent|mission|aim"
    r"|expected\s+(?:response|result|output|format)"
    r"|response\s+format|answer\s+format|desired\s+(?:format|outcome)"
    r")\s*[:]\s*$",
    re.IGNORECASE,
)

# Content-free markdown header — matches lines like "# Role", "## Requirements"
# where the heading text is purely a section label with no instruction content.
_CONTENT_FREE_MD_HEADER_RE = re.compile(
    r"^\s*#{1,6}\s+(?:role|objective|goal|task|purpose|requirements?"
    r"|constraints?|limitations?|output|format|context"
    r"|background|instructions?|deliverables?|criteria"
    r"|scope|audience|tone|style|examples?"
    r"|guidelines?|specifications?|rules|directions?"
    r"|expectations?|boundaries|guardrails"
    r"|target|intent|mission|aim"
    r")\s*$",
    re.IGNORECASE,
)

# Declarative section labels — sections whose content is context-setting,
# not directive. Content under these headers is excluded from the
# actionability denominator (it contributes to specificity/completeness
# instead). Everything else is treated as directive.
_DECLARATIVE_SECTION_LABELS: frozenset = frozenset({
    "role", "persona", "context", "background", "audience",
    "tone", "style", "scope", "examples", "example",
})

# Bullet item detection (- item, * item, bullet item)
_BULLET_ITEM_PATTERN = re.compile(r"^\s*[-*\u2022]\s+", re.MULTILINE)

# Numbered item detection (1. item, 2) item)
_NUMBERED_ITEM_PATTERN = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)

# ═══════════════════════════════════════════════════════════════════════════
# Clarity — Completeness Component Patterns
# ═══════════════════════════════════════════════════════════════════════════

# Each pattern detects a complementary instruction component.
# Score = detected_count / total_count (5 components).
# Expanded with synonyms so "Goal" matches "Objective", "Instructions"
# matches "Requirements", "Expected Response" matches "Output Format", etc.
_COMPLETENESS_PATTERNS: Dict[str, re.Pattern] = {
    "role": re.compile(
        r"\b(?:act\s+as|role|persona|you\s+are|imagine\s+you(?:'re|\s+are)"
        r"|as\s+a(?:n)?\s+\w+|pretend|behave\s+as|play\s+the\s+role)\b",
        re.IGNORECASE,
    ),
    "objective": re.compile(
        r"\b(?:objective|goal|task|purpose|aim|mission"
        r"|target|intent|desired\s+outcome|expected\s+result"
        r"|what\s+I\s+want|what\s+I\s+need|what\s+you\s+should)\b",
        re.IGNORECASE,
    ),
    "requirements": re.compile(
        r"\b(?:requirements?|must|should|need\s+to|shall|ensure|include"
        r"|instructions?|guidelines?|specifications?"
        r"|rules|directions?|criteria|expectations?)\b",
        re.IGNORECASE,
    ),
    "constraints": re.compile(
        r"\b(?:constraints?|limit(?:ation)?s?|restrict(?:ion)?s?|within"
        r"|maximum|minimum|at\s+most|at\s+least|no\s+more\s+than"
        r"|no\s+fewer\s+than|avoid|do\s+not|don't"
        r"|boundar(?:y|ies)|guardrails?|prohibited|forbidden"
        r"|disallowed|not\s+allowed|out\s+of\s+scope)\b",
        re.IGNORECASE,
    ),
    "output_format": re.compile(
        r"\b(?:output|format|return|respond|deliver(?:able)?|provide|present|display"
        r"|expected\s+response|desired\s+format|response\s+format"
        r"|answer\s+format|reply|result|render)"
        r"\s*(?:as|in|using|:)?\b",
        re.IGNORECASE,
    ),
    "context": re.compile(
        r"\b(?:context|background|audience|scenario|situation)\b",
        re.IGNORECASE,
    ),
    "examples": re.compile(
        r"\b(?:examples?|samples?|reference|few-shot)\b",
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
        import torch
        logger.info(f"System Check | PyTorch CUDA available: {torch.cuda.is_available()}")

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

    @staticmethod
    def _extract_section_label(line: str) -> Optional[str]:
        """
        Extract the section label from a labeled-section line.

        Given a line like "Requirements:" or "Role: You are an expert",
        returns the lowercase label (e.g. "requirements", "role").
        Returns None if the line is not a labeled section.
        """
        m = _LABELED_SECTION_PATTERN.match(line)
        if not m:
            return None
        # The matched text is "  Label:  " — strip and remove the colon.
        return m.group().strip().rstrip(":").strip().lower()

    def _extract_instruction_units(self, prompt: str) -> List[tuple]:
        """
        Split a prompt into instruction units with section context.

        Each unit is a (text, is_declarative) tuple where:
          - text: the instruction unit string
          - is_declarative: True if this unit falls under a declarative
            section (Role, Context, Background, Audience, Tone, Style,
            Scope, Examples) whose content is context-setting, not
            directive. Declarative units are excluded from the
            actionability denominator.

        Content-free section headers (e.g. a line containing only
        "REQUIREMENTS:" or "# Constraints") are dropped entirely —
        they are structural metadata, not instruction units.

        Returns:
            List of (text, is_declarative) tuples.
        """
        lines = prompt.split("\n")
        units: List[tuple] = []
        prose_block: List[str] = []
        # Track the current section context (None = top-level / directive)
        current_section_declarative: bool = False

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
                    units.append((txt, current_section_declarative))

        for line in lines:
            stripped = line.strip()
            if not stripped:
                flush_prose(prose_block)
                prose_block = []
            elif structural_line.match(line) or _LABELED_SECTION_PATTERN.match(line):
                flush_prose(prose_block)
                prose_block = []

                # Update section context if this is a labeled section
                label = self._extract_section_label(line)
                if label is not None:
                    current_section_declarative = (
                        label in _DECLARATIVE_SECTION_LABELS
                    )

                # Drop content-free headers — they're structural metadata
                is_content_free = (
                    _CONTENT_FREE_LABEL_RE.match(line)
                    or _CONTENT_FREE_MD_HEADER_RE.match(line)
                )
                if not is_content_free:
                    units.append((stripped, current_section_declarative))
            else:
                prose_block.append(stripped)

        flush_prose(prose_block)
        return units

    def _compute_actionability(
        self, units: List[tuple]
    ) -> tuple:
        """
        Compute verb-density actionability score (0.0–1.0).

        Instead of binary per-unit scoring (actionable_units / total_units),
        this computes a weighted verb score per directive unit:

          - Strong verbs contribute 1.0 each
          - Weak verbs contribute weak_verb_weight (default 0.5)
          - Modal-governed verbs contribute modal_verb_weight (default 0.75)
          - Implicit commands contribute 1.0

        Per-unit score is capped at actionability_verb_cap (default 2.0)
        to prevent verbosity gaming.  Final score =
        sum(capped_unit_scores) / (directive_units * cap), clamped to [0, 1].

        Declarative units (under Role, Context, Background, Audience,
        Tone, Style, Scope sections) are excluded from both the
        numerator and denominator — they contribute to specificity
        and completeness, not actionability.

        Args:
            units: List of (text, is_declarative) tuples from
                   _extract_instruction_units.

        Returns:
            (score, diagnostics_dict)
        """
        if not units:
            return 0.0, {
                'verb_count': 0, 'modal_count': 0, 'passive_count': 0,
                'implicit_commands': 0, 'actionable_units': 0,
                'total_units': 0, 'declarative_units': 0,
            }

        cap = self.config.actionability_verb_cap
        total_verb_count = 0
        total_modal_count = 0
        total_passive_count = 0
        total_implicit_commands = 0
        actionable_units = 0
        declarative_count = 0
        sum_unit_scores: float = 0.0

        for unit_text, is_declarative in units:
            # Skip declarative units (Role, Context, etc.) — they
            # don't belong in the actionability measurement.
            if is_declarative:
                declarative_count += 1
                continue

            unit_score: float = 0.0

            # Check for implicit command (e.g., "Step 1: Data preprocessing")
            if IMPLICIT_COMMAND_PATTERN.match(unit_text):
                total_implicit_commands += 1
                unit_score += 1.0

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
                        # Strong vs weak verb weighting
                        lemma = token.lemma_.lower()
                        if lemma in WEAK_VERBS:
                            unit_score += self.config.weak_verb_weight
                        else:
                            unit_score += 1.0

                if token.pos_ == "AUX" and token.lemma_.lower() in MODAL_VERBS:
                    total_modal_count += 1
                    has_governed = any(
                        child.pos_ == "VERB" for child in token.children
                    ) or token.head.pos_ == "VERB"
                    if has_governed:
                        unit_score += self.config.modal_verb_weight

                if token.dep_ == "nsubjpass" and token.head.pos_ == "VERB":
                    total_passive_count += 1

            # Apply soft cap per unit
            capped_score = min(unit_score, cap)
            sum_unit_scores += capped_score

            if unit_score > 0:
                actionable_units += 1

        # Directive units = total minus declarative
        directive_count = len(units) - declarative_count
        if directive_count <= 0:
            # Prompt is entirely declarative (rare edge case)
            return 0.0, {
                'verb_count': total_verb_count,
                'modal_count': total_modal_count,
                'passive_count': total_passive_count,
                'implicit_commands': total_implicit_commands,
                'actionable_units': actionable_units,
                'total_units': len(units),
                'declarative_units': declarative_count,
            }

        # Verb-density: normalise by (directive_units * cap)
        score = sum_unit_scores / (directive_count * cap)
        return min(score, 1.0), {
            'verb_count': total_verb_count,
            'modal_count': total_modal_count,
            'passive_count': total_passive_count,
            'implicit_commands': total_implicit_commands,
            'actionable_units': actionable_units,
            'total_units': len(units),
            'declarative_units': declarative_count,
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
        score_map = [0.0, 0.50, 0.85, 1.0, 1.0, 1.0]
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
        score = count / 3.0
        return min(score, 1.0), detected

    def _compute_weak_fragment_penalty(
        self, prompt: str, units: List[tuple]
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
        for unit_text, _is_declarative in units:
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
    # Stage 2: Specificity Scoring — Helpers
    # ═══════════════════════════════════════════════════════════════════

    # Section headers that act as scaffolding in structured prompt templates.
    # Excluded from the informative-token denominator so a ROLE/TASK/CONSTRAINTS
    # rewrite is not penalised for its own structure.
    _SCAFFOLD_HEADER_TOKENS: frozenset = frozenset({
        "ROLE", "TASK", "INPUTS", "OUTPUTS", "OUTPUT", "CONSTRAINTS",
        "REQUIREMENTS", "OBJECTIVE", "SECTIONS", "FORMAT", "LANGUAGE",
        "STACK", "EDGE", "CASES", "BEST", "PRACTICES", "NOTES", "CRITERIA",
        "QUESTION", "SUBJECT", "DELIVERABLES", "DETAIL", "LEVEL", "ORDER",
        "RECOMMENDATION", "ANALYSIS", "DEPTH", "GOAL", "GOALS", "CONTEXT",
        "BACKGROUND",
    })

    # Detects "[Insert ...]" placeholder spans whose content is scaffolding,
    # not user-supplied specificity. Used to exclude tokens inside such spans
    # from the informative-token denominator.
    _RE_INSERT_PLACEHOLDER = re.compile(
        r"\[\s*Insert\b[^\]]*\]", re.IGNORECASE
    )

    @classmethod
    def _get_informative_tokens(cls, doc) -> list:
        """
        Return tokens excluding punctuation, whitespace, stopwords, symbols,
        scaffold-section headers (ROLE/TASK/...), and tokens inside
        ``[Insert ...]`` placeholder spans. Produces a clean denominator
        for constraint density so that template structure does not
        deflate the ratio.

        Numbers are kept — they are informative for specificity.
        """
        text = doc.text
        insert_spans = [
            (m.start(), m.end())
            for m in cls._RE_INSERT_PLACEHOLDER.finditer(text)
        ]

        def _in_insert_span(tok) -> bool:
            tok_start = tok.idx
            tok_end = tok.idx + len(tok.text)
            return any(s <= tok_start and tok_end <= e for s, e in insert_spans)

        return [
            t for t in doc
            if not t.is_punct
            and not t.is_space
            and not t.is_stop
            and t.pos_ != "SYM"
            and t.text.upper() not in cls._SCAFFOLD_HEADER_TOKENS
            and not _in_insert_span(t)
        ]

    # ── Regex patterns for constraint detection (compiled once) ──────

    _RE_RANGE_NUMERIC = re.compile(
        r"\b\d+\s*[-\u2013\u2014]\s*\d+\b"
    )
    _RE_BOUND_PHRASE = re.compile(
        r"\b(?:at\s+least|at\s+most|no\s+more\s+than|no\s+fewer\s+than"
        r"|between\s+\d+\s+and\s+\d+|up\s+to\s+\d+)\b",
        re.IGNORECASE,
    )
    # Verb + (optional modifiers, up to ~4 words) + format/deliverable noun.
    # Modifiers allow patterns like "Output ONLY the JSON code block",
    # "provide a complete YAML manifest", "Generate strictly valid JSON".
    _RE_FORMAT_INSTRUCTION = re.compile(
        r"\b(?:return|output|provide|give|deliver|generate|produce|"
        r"create|write|format)"
        r"(?:\s+(?:only|strictly|exactly|a|an|the|complete|valid|"
        r"well[- ]formed|following|all))*"
        r"\s+"
        r"(?:json|csv|markdown|xml|yaml|table|html|plain\s+text"
        r"|bullet\s+list|structured\s+outline|code\s+block|manifest|"
        r"payload|schema|deliverables?|outline|report|response|template)\b",
        re.IGNORECASE,
    )
    # "Provide the following deliverables:" / "Deliver the following:" pattern,
    # ubiquitous in `chosen` rewrites. Treated as a single format-instruction
    # signal (weight 1.2) since it commits the response shape.
    _RE_DELIVERABLES_LIST = re.compile(
        r"\b(?:provide|deliver|return|produce|include)\s+"
        r"the\s+following(?:\s+\w+){0,3}\s*:",
        re.IGNORECASE,
    )
    _RE_TOOL_REQUIREMENT = re.compile(
        r"\b(?:using|in|with|use|implement(?:ed)?\s+in|written\s+in|"
        r"build(?:\s+with)?|leverage)\s+"
        r"(?:python|javascript|typescript|sql|bash|r|go|java|c\+\+|c#|"
        r"ruby|rust|kotlin|swift|scala|php|perl|powershell|lua|dart|"
        r"elixir|haskell|nodejs|node\.js)\b",
        re.IGNORECASE,
    )
    # Frameworks: scored at reduced weight (0.6) inside
    # _detect_tool_and_deliverable_constraints. Distinct regex so the bare
    # framework name doesn't need a prefix verb (a framework alone is a
    # weaker specificity signal than an explicit "using <lang>" directive).
    _RE_FRAMEWORK_REQUIREMENT = re.compile(
        r"\b(?:react|vue|angular|svelte|next\.?js|django|flask|fastapi|"
        r"spring|express|laravel|rails|\.net|tailwind|bootstrap|"
        r"pytorch|tensorflow|jax|pandas|numpy|scikit[- ]?learn)\b",
        re.IGNORECASE,
    )
    _RE_FILE_FORMAT = re.compile(
        r"\.(?:json|csv|xml|yaml|yml|html|md|txt|pdf|png|jpg)\b",
        re.IGNORECASE,
    )
    _RE_STRUCTURAL_EXECUTION = re.compile(
        r"\b(?:step[- ]by[- ]step|comparison\s+table|numbered\s+(?:list|explanation)"
        r"|code\s+block|worked\s+example|side[- ]by[- ]side)\b",
        re.IGNORECASE,
    )
    # Negation constraints bound the output space ("Do not use deprecated
    # libraries", "Never include personal data"). Real specificity signal.
    _RE_NEGATION_CONSTRAINT = re.compile(
        r"\b(?:do\s+not|don['’]t|must\s+not|should\s+not|shouldn['’]t"
        r"|never|avoid|refrain\s+from|refuse\s+to|strictly\s+refuse"
        r"|without\s+(?:using|including|adding))\b",
        re.IGNORECASE,
    )
    # Persona / role declaration: constrains tone, expertise, voice.
    _RE_PERSONA_DECLARATION = re.compile(
        r"(?:^|\n|\.\s+)\s*(?:you\s+are|act\s+as|assume\s+the\s+role\s+of"
        r"|as\s+(?:a|an)\s+expert|role\s*:\s*you\s+are)\b",
        re.IGNORECASE,
    )

    def _score_entities_attached(self, doc) -> float:
        """
        Score named entities by attachment validation and span length.

        Only entities syntactically connected to a verb or constraint
        phrase contribute.  Each attached entity token contributes 1.2
        (higher weight than adjective modifiers at 0.6).

        Unattached entities (pure spam like "Python JSON HTML") score 0.
        """
        total: float = 0.0
        verbal_deps = frozenset({"dobj", "pobj", "attr", "nsubj", "nsubjpass", "compound", "appos"})
        # Skip numeric entity labels — these are already captured by
        # nummod (scalar) or range-constraint detectors.  Counting them
        # again as entities would double-count a single number token.
        numeric_labels = frozenset({"CARDINAL", "ORDINAL"})

        for ent in doc.ents:
            if ent.label_ in numeric_labels:
                continue
            attached = False
            for token in ent:
                # Direct verbal head
                if token.head.pos_ in ("VERB", "AUX"):
                    attached = True
                    break
                # Dependency relation implying verbal governance
                if token.dep_ in verbal_deps:
                    attached = True
                    break
                # Walk up to check for a verbal ancestor (max 3 hops)
                ancestor = token.head
                for _ in range(3):
                    if ancestor.pos_ in ("VERB", "AUX"):
                        attached = True
                        break
                    if ancestor == ancestor.head:
                        break
                    ancestor = ancestor.head
                if attached:
                    break
            if attached:
                total += len(ent) * 1.2
        return total

    def _detect_format_instructions(self, text: str) -> float:
        """
        Detect output format requirement phrases.

        Examples: "Return JSON", "output as markdown", "provide a table",
        "Output ONLY the JSON code block", "Provide the following
        deliverables:".  Each match contributes weight 1.2.
        """
        matches = self._RE_FORMAT_INSTRUCTION.findall(text)
        deliverables = self._RE_DELIVERABLES_LIST.findall(text)
        return (len(matches) + len(deliverables)) * 1.2

    def _detect_range_constraints(self, text: str) -> tuple:
        """
        Detect bounded numeric constraints and return (score, range_token_positions).

        Ranges ("3-5") and bound phrases ("at least 3") contribute weight
        1.5 each.  Also returns character spans of range numbers so the
        caller can exclude them from the scalar nummod count.
        """
        range_matches = self._RE_RANGE_NUMERIC.findall(text)
        bound_matches = self._RE_BOUND_PHRASE.findall(text)
        count = len(range_matches) + len(bound_matches)

        # Collect character-level start positions of numbers inside ranges
        # so the caller can de-duplicate against nummod tokens.
        range_char_spans: set = set()
        for m in self._RE_RANGE_NUMERIC.finditer(text):
            range_char_spans.add((m.start(), m.end()))

        return count * 1.5, range_char_spans

    def _detect_tool_and_deliverable_constraints(self, text: str) -> float:
        """
        Detect tool requirements, file format mentions, structural
        execution constraints, and named frameworks.  Tools/files/exec
        contribute 1.0 each; framework names contribute 0.6 each (weaker
        signal than an explicit language directive).
        """
        tool_count = len(self._RE_TOOL_REQUIREMENT.findall(text))
        file_count = len(self._RE_FILE_FORMAT.findall(text))
        exec_count = len(self._RE_STRUCTURAL_EXECUTION.findall(text))
        framework_count = len(self._RE_FRAMEWORK_REQUIREMENT.findall(text))
        return (tool_count + file_count + exec_count) * 1.0 + framework_count * 0.6

    def _detect_negation_constraints(self, text: str) -> float:
        """
        Detect negation-style constraints ("Do not", "Never", "Avoid",
        "Strictly refuse", "Without using"). Each match contributes
        weight 1.0 — same scale as a scalar nummod, since a negation
        clause meaningfully bounds the response.
        """
        return len(self._RE_NEGATION_CONSTRAINT.findall(text)) * 1.0

    def _detect_persona_declarations(self, text: str) -> float:
        """
        Detect persona / role declarations at the start of a line or
        sentence ("You are an expert...", "Act as a senior engineer",
        "ROLE: You are..."). Each match contributes weight 0.8 — slightly
        below entity weight, since persona constrains style/tone rather
        than the substantive output.
        """
        return len(self._RE_PERSONA_DECLARATION.findall(text)) * 0.8

    def _compute_local_ambiguity_ratio(self, informative_tokens: list) -> float:
        """
        Compute the ratio of ambiguous tokens to informative tokens.

        Reuses the module-level AMBIGUOUS_TOKENS set.  Returns a raw
        ratio (0.0–1.0) used for the ambiguity dampening interaction
        inside specificity scoring.
        """
        if not informative_tokens:
            return 0.0
        ambig_count = sum(
            1 for t in informative_tokens
            if t.lemma_.lower() in AMBIGUOUS_TOKENS
        )
        return ambig_count / len(informative_tokens)

    # ═══════════════════════════════════════════════════════════════════
    # Stage 2: Specificity Scoring
    # ═══════════════════════════════════════════════════════════════════

    def _score_specificity(self, prompt: str) -> Dict[str, Any]:
        """
        SOP §3B — Specificity Score (0.0–1.0) with diagnostics.

        Hybrid coverage + intensity scoring across constraint categories:
          - modifiers (amod + nummod, combined)
          - Entities: attachment-validated, span-length-weighted
          - Range constraints: "3-5", "at least X"
          - Format instructions: "Return JSON", "output table"
          - Tool/deliverable constraints: "using Python", ".csv"
          - Negation constraints: "Do not / Never / Avoid ..."
          - Persona declarations: "You are an expert ..."

        Score = 0.85·coverage + 0.15·intensity, where coverage is the
        fraction of categories represented and intensity is a saturating
        function of total signal weight. This makes specificity
        length-invariant: a tight, well-specified short prompt and a
        verbose, well-specified long prompt receive comparable scores.

        Ambiguity dampening (exponential decay) still applies to penalise
        vague-but-numeric prompts. The length floor still protects against
        unreliable signals on very short samples.

        Signal separation: this function never rewards formatting structure
        (headers, bullets, numbered lists, layout) — those belong to clarity.

        Returns:
            dict with 'score' (float 0.0–1.0) and diagnostic sub-scores.
        """
        empty_result: Dict[str, Any] = {
            'score': 0.0,
            'modifiers': 0.0, 'entities': 0.0, 'ranges': 0.0,
            'formats': 0.0, 'tools': 0.0, 'negation': 0.0,
            'persona': 0.0, 'coverage': 0.0, 'intensity': 0.0,
        }

        if not self._nlp or not prompt.strip():
            return empty_result

        doc = self._nlp(prompt)
        if len(doc) == 0:
            return empty_result

        # ── Informative token denominator (Issues 1, 6) ──────────────
        informative = self._get_informative_tokens(doc)
        n = len(informative)
        if n == 0:
            return empty_result

        # ── Range constraints (Issue 9) — detect before nummod ───────
        range_score, range_char_spans = self._detect_range_constraints(prompt)

        # ── Modifier scoring with diminishing returns (Issues 5, 10) ─
        #   amod: group by head noun, apply log2 scaling per head, weight 0.6
        #   nummod: weight 1.0, but exclude tokens inside detected ranges
        amod_by_head: Dict[int, int] = {}
        nummod_score: float = 0.0

        for token in doc:
            if token.dep_ == "amod":
                # Skip vague adjectives — they belong to ambiguity, not specificity
                if token.lemma_.lower() in AMBIGUOUS_TOKENS:
                    continue
                head_idx = token.head.i
                amod_by_head[head_idx] = amod_by_head.get(head_idx, 0) + 1
            elif token.dep_ == "nummod":
                # Check if this nummod token falls inside a range span
                token_start = token.idx
                token_end = token.idx + len(token.text)
                in_range = any(
                    rs <= token_start and token_end <= re
                    for rs, re in range_char_spans
                )
                if not in_range:
                    nummod_score += 1.0

        # Diminishing returns: log2(count+1) per head, capped at 1.5
        # (effectively rewards at most ~2 adjectives per noun).
        # Weight 0.4 — adjectives are the weakest specificity signal.
        amod_score: float = sum(
            min(math.log2(c + 1), 1.5) for c in amod_by_head.values()
        ) * 0.4

        # ── Entity scoring — attachment-validated (Issues 2, 4, 8) ───
        entity_score: float = self._score_entities_attached(doc)

        # ── Format instructions (Issue 7) ─────────────────────────────
        format_score: float = self._detect_format_instructions(prompt)

        # ── Tool & deliverable constraints (Issues 3, 11) ─────────────
        tool_deliverable_score: float = self._detect_tool_and_deliverable_constraints(prompt)

        # ── Negation & persona constraints ────────────────────────────
        # Real specificity signals that previously scored zero: negative
        # constraints bound the output space, persona declarations bound
        # tone/voice.
        negation_score: float = self._detect_negation_constraints(prompt)
        persona_score: float = self._detect_persona_declarations(prompt)

        # ── Hybrid coverage + intensity scoring ───────────────────────
        # Length-invariant by design. Each constraint *category* gets
        # binary credit, so a 30-token prompt with all dimensions
        # specified scores the same as a 300-token one. A small
        # intensity term breaks ties (e.g. "1 negation" vs "5 negations")
        # without letting any single category dominate.
        modifier_score: float = amod_score + nummod_score
        category_scores: tuple = (
            modifier_score,   # modifiers (combined — weakest signal pair)
            entity_score,
            range_score,
            format_score,
            tool_deliverable_score,
            negation_score,
            persona_score,
        )
        present: int = sum(1 for s in category_scores if s > 0)
        coverage: float = present / len(category_scores)

        total_signal: float = sum(category_scores)
        # k = 0.2 → intensity ≈ 0.5 at signal weight ~3.5,
        # saturates near 1.0 by signal ≥ ~15. Prevents stuffing.
        intensity: float = 1.0 - math.exp(-0.2 * total_signal)

        # Rebalanced from 0.85/0.15 → 0.65/0.35. The original 85% weight
        # on binary category presence made it structurally impossible for
        # prompts covering 3–4 of 7 categories to exceed ~0.60 specificity,
        # even with strong signal intensity. The new blend rewards depth
        # over breadth: a prompt with persona + negation + entities + formats
        # (4/7 coverage = 0.57) and high intensity (≈0.90) now scores
        # 0.65*0.57 + 0.35*0.90 ≈ 0.69 → after structural bonus and clarity,
        # the quality ceiling rises from ~88% to ~95%.
        score: float = 0.65 * coverage + 0.35 * intensity

        # ── Ambiguity dampening (Issue 12) ────────────────────────────
        # Exponential decay: mild ambiguity → mild reduction,
        # heavy ambiguity → steep reduction.
        ambig_ratio: float = self._compute_local_ambiguity_ratio(informative)
        dampening: float = math.exp(-2.0 * ambig_ratio)
        score *= dampening

        # ── Length floor ──────────────────────────────────────────────
        # Density on tiny samples is unreliable. A 5-token prompt anchored
        # on a single named entity can hit density ≥ ideal trivially. Linearly
        # down-scale specificity for prompts below `specificity_length_floor`
        # informative tokens; full credit at or above the floor.
        floor: int = self.config.specificity_length_floor
        if floor > 0 and n < floor:
            score *= n / floor

        score = max(min(score, 1.0), 0.0)

        return {
            'score': round(score, 4),
            'modifiers': round(modifier_score, 4),
            'entities': round(entity_score, 4),
            'ranges': round(range_score, 4),
            'formats': round(format_score, 4),
            'tools': round(tool_deliverable_score, 4),
            'negation': round(negation_score, 4),
            'persona': round(persona_score, 4),
            'coverage': round(coverage, 4),
            'intensity': round(intensity, 4),
        }

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
        specificity_result = self._score_specificity(prompt)
        specificity: float = specificity_result['score']
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
            # Specificity diagnostics
            "specificity_modifiers": specificity_result.get('modifiers', 0.0),
            "specificity_entities": specificity_result.get('entities', 0.0),
            "specificity_ranges": specificity_result.get('ranges', 0.0),
            "specificity_formats": specificity_result.get('formats', 0.0),
            "specificity_tools": specificity_result.get('tools', 0.0),
            "specificity_negation": specificity_result.get('negation', 0.0),
            "specificity_persona": specificity_result.get('persona', 0.0),
            "specificity_coverage": specificity_result.get('coverage', 0.0),
            "specificity_intensity": specificity_result.get('intensity', 0.0),
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
                clarity_actionability=raw_metrics.get("clarity_actionability", 0.0),
                clarity_structure=raw_metrics.get("clarity_structure", 0.0),
                clarity_completeness=raw_metrics.get("clarity_completeness", 0.0),
                clarity_fragment_penalty=raw_metrics.get("clarity_fragment_penalty", 0.0),
                specificity_modifiers=raw_metrics.get("specificity_modifiers", 0.0),
                specificity_entities=raw_metrics.get("specificity_entities", 0.0),
                specificity_ranges=raw_metrics.get("specificity_ranges", 0.0),
                specificity_formats=raw_metrics.get("specificity_formats", 0.0),
                specificity_tools=raw_metrics.get("specificity_tools", 0.0),
                specificity_negation=raw_metrics.get("specificity_negation", 0.0),
                specificity_persona=raw_metrics.get("specificity_persona", 0.0),
                specificity_coverage=raw_metrics.get("specificity_coverage", 0.0),
                specificity_intensity=raw_metrics.get("specificity_intensity", 0.0),
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
            clarity_actionability=cand_metrics.get("clarity_actionability", 0.0),
            clarity_structure=cand_metrics.get("clarity_structure", 0.0),
            clarity_completeness=cand_metrics.get("clarity_completeness", 0.0),
            clarity_fragment_penalty=cand_metrics.get("clarity_fragment_penalty", 0.0),
            specificity_modifiers=cand_metrics.get("specificity_modifiers", 0.0),
            specificity_entities=cand_metrics.get("specificity_entities", 0.0),
            specificity_ranges=cand_metrics.get("specificity_ranges", 0.0),
            specificity_formats=cand_metrics.get("specificity_formats", 0.0),
            specificity_tools=cand_metrics.get("specificity_tools", 0.0),
            specificity_negation=cand_metrics.get("specificity_negation", 0.0),
            specificity_persona=cand_metrics.get("specificity_persona", 0.0),
            specificity_coverage=cand_metrics.get("specificity_coverage", 0.0),
            specificity_intensity=cand_metrics.get("specificity_intensity", 0.0),
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
    weak = "Build a website for my car wash business."
    weak_optimised = """Role:
Act as a senior full-stack web developer.

Objective:
Design a responsive website for a car wash business.

Requirements:
1. Create a homepage, services page, pricing page, booking form, and contact page.
2. Include a mobile-friendly layout and modern UI.
3. Use HTML5, CSS3, and JavaScript.

Constraints:
- Do not use external frameworks.
- Generate between 400 and 600 words.
- Include exactly 5 sections.

Output Format:
Return the response in Markdown using headings, bullet lists, and code blocks."""

    print("\n[WEAK PROMPT]")
    print(f"  Raw: {weak!r}")
    print(f"  Candidate: {weak_optimised!r}")
    result = scorer.evaluate(weak, weak_optimised)
    for k, v in result.items():
        print(f"  {k}: {v}")

    # Compute individual clarity scores for comparison
    raw_metrics = scorer._score_prompt(weak)
    cand_metrics = scorer._score_prompt(weak_optimised)
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
