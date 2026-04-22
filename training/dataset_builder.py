"""
Dataset Builder — Scorer-Guided Atomic Rewrite Pipeline for DPO Training

SOP Reference: architecture/dpo_training.md §3

Reads raw prompts from dataset/RAW_prompts.csv, filters for instructional tasks,
detects missing structural components via the HeuristicScorer, builds a single
atomic rewrite instruction per prompt, generates 1–3 rewrite candidates (retrying
only on rejection), scores them, and outputs preference pairs to
datasets/preference_pairs.jsonl.

Pipeline flow:
  1. Baseline scoring of the raw prompt
  2. Detect missing structural components (role, objective, requirements,
     constraints, output_format, context)
  3. Build a single atomic rewrite instruction with section-token hints
  4. Generate up to max_attempts rewrite candidates (varied temperature)
  5. Validate each candidate (no task-answering, no examples, no intent drift)
  6. Score each valid candidate immediately with HeuristicScorer
  7. Track and select the best-scoring candidate across all attempts
  8. Accept if quality improvement exceeds MIN_GAIN threshold
  9. Construct DPO pair (x=raw, y_w=best_candidate, y_l=raw)

Best-candidate selection: all valid candidates are scored immediately during
generation. The highest-quality candidate is selected, not just the first valid
one. This improves DPO signal quality at no additional generation cost.

This script is OFFLINE-ONLY. It must never be imported by runtime code.
"""

import os
import sys
import csv
import json
import random
import logging
import argparse
import re
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any, Optional, Tuple

# Project root for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from tools.heuristic_scorer import (
    HeuristicScorer,
    AMBIGUOUS_TOKENS,
    _COMPLETENESS_PATTERNS,
)

# Deterministic sampling across runs
random.seed(0)

logger = logging.getLogger("promptee.dataset_builder")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

# Default paths
RAW_PROMPTS_PATH = os.path.join(PROJECT_ROOT, "dataset", "RAW_prompts.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "datasets", "preference_pairs.jsonl")

# ── Pipeline Configuration Constants ────────────────────────────────────
MAX_REWRITE_ATTEMPTS = 3
MIN_GAIN = 0.15
REWRITE_TEMPERATURE_BASE = 0.35
REWRITE_TEMPERATURE_STEP = 0.10
REWRITE_TOP_P = 0.88

# ── Multi-Negative DPO Expansion Constants ──────────────────────────────
TARGET_DATASET_SIZE = 1200
LENGTH_RATIO_MIN = 0.6
LENGTH_RATIO_MAX = 1.4
SEMANTIC_FLOOR = 0.40

# ── Instructional Task Filter ────────────────────────────────────────────

# Verb patterns that indicate an instructional/task-oriented prompt
INSTRUCTIONAL_VERBS = re.compile(
    r"^(write|create|build|make|implement|design|develop|generate|produce|compose|"
    r"explain|describe|analyze|compare|summarize|translate|rewrite|sort|group|"
    r"categorize|classify|help|give|how|what are the steps|what are some ways|"
    r"break down|set up|deploy|optimize|fix|debug|parse|convert|handle|validate|"
    r"monitor|test|configure|calculate|compute|simulate|model|predict|train|"
    r"evaluate|assess|improve|enhance|refine|format|structure|organize|plan|"
    r"outline|list|define|identify|find|solve|prove|derive|show)",
    re.IGNORECASE,
)

# Patterns that indicate NON-instructional prompts (opinions, stories, vague "tell me")
NON_INSTRUCTIONAL_PATTERNS = [
    re.compile(r"^tell me about\b", re.IGNORECASE),
    re.compile(r"^tell me a story\b", re.IGNORECASE),
    re.compile(r"^write a (story|poem|song)\b", re.IGNORECASE),
    re.compile(r"^what is the (meaning|significance|importance)\b", re.IGNORECASE),
    re.compile(r"^what are the (pros and cons|strengths and weaknesses)\b", re.IGNORECASE),
    re.compile(r"^(are|is) .+ (good|bad|useful|effective)\??$", re.IGNORECASE),
    re.compile(r"^what does this code do\??$", re.IGNORECASE),
    re.compile(r"^what are .+ opinions\b", re.IGNORECASE),
    re.compile(r"^give me .+ advice\b", re.IGNORECASE),
    re.compile(r"^what are your thoughts\b", re.IGNORECASE),
]


def _contains_task_object(prompt: str) -> bool:
    """
    Check whether the prompt contains a concrete task object (noun phrase).

    Returns False for prompts that are too short or lack a recognizable
    task-relevant noun, filtering out vague two-word commands like "explain ai".
    """
    words = prompt.lower().split()

    if len(words) < 5:
        return False

    _TASK_NOUN_SUFFIXES = (
        "ion", "ment", "data", "model", "system", "function", "code",
        "plan", "analysis", "script", "file", "list", "table", "report",
        "query", "pipeline", "service", "endpoint", "schema", "config",
        "module", "class", "test", "page", "form", "workflow", "database",
        "server", "interface", "algorithm", "structure", "template",
        "component", "application", "api", "output", "input", "response",
        "request", "format", "chart", "graph", "dashboard", "logic",
    )

    return any(word.endswith(_TASK_NOUN_SUFFIXES) for word in words)


def is_instructional(prompt: str) -> bool:
    """
    Filter: returns True only if the prompt represents an instructional task.

    Requires BOTH an instructional verb AND a concrete task object to pass.
    Rejects vague "tell me about" prompts, stories, poems, opinion questions,
    and short prompts lacking a recognizable task noun.
    """
    prompt = prompt.strip()
    if not prompt or len(prompt) < 10:
        return False

    # Reject known non-instructional patterns
    for pattern in NON_INSTRUCTIONAL_PATTERNS:
        if pattern.search(prompt):
            return False

    has_verb = bool(INSTRUCTIONAL_VERBS.search(prompt)) or bool(
        re.match(r"^how (do|can|should|would|might) (i|we|you)\b", prompt, re.IGNORECASE)
    )

    if not has_verb:
        return False

    return _contains_task_object(prompt)


# ── Indexed Record Structure ─────────────────────────────────────────────

@dataclass
class PromptRecord:
    """
    Indexed record for a single raw prompt and its rewrite result.
    Prevents key collisions from duplicate prompts by using row index as ID.
    """
    idx: int
    raw_prompt: str
    best_candidate: Optional[str] = None
    best_final_score: float = -1.0
    best_candidate_quality: float = -1.0
    baseline_quality: float = 0.0
    accepted: bool = False
    attempt_count: int = 0
    missing_components: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Missing Component Detection ─────────────────────────────────────────

def detect_missing_components(prompt: str) -> List[str]:
    """
    Detect which structural components are absent from the prompt.

    Uses the scorer's completeness patterns (role, objective, requirements,
    constraints, output_format) to identify gaps. Returns a list of missing
    component names.

    Also checks for context signals separately since _COMPLETENESS_PATTERNS
    does not include a 'context' detector.
    """
    missing = []

    # Use the scorer's completeness patterns
    for name, pattern in _COMPLETENESS_PATTERNS.items():
        if not pattern.search(prompt):
            missing.append(name)

    # Check for context signals (not in _COMPLETENESS_PATTERNS)
    context_pattern = re.compile(
        r"\b(?:context|background|scenario|situation|given\s+that"
        r"|assuming|suppose|consider\s+that)\b",
        re.IGNORECASE,
    )
    if not context_pattern.search(prompt):
        missing.append("context")

    return missing


# ── Section Token Map for Rewrite Hints ─────────────────────────────────

SECTION_TOKEN_MAP: Dict[str, str] = {
    "role": "ROLE",
    "objective": "OBJECTIVE",
    "requirements": "REQUIREMENTS",
    "constraints": "CONSTRAINTS",
    "output_format": "OUTPUT FORMAT",
    "context": "CONTEXT",
}


# ── Diverse Rewrite Instruction Templates ───────────────────────────────

_BASE_RULES = (
    "Rules:\n"
    "- Preserve the original task intent and meaning exactly\n"
    "- Only improve structure and clarity\n"
    "- Do NOT solve the task or generate the output the prompt requests\n"
    "- Do NOT include examples, sample inputs, or sample outputs\n"
    "- Do NOT include placeholder content or generic filler\n"
    "- Output ONLY the rewritten prompt — no explanations or commentary\n"
)


def _missing_block(missing_components: List[str]) -> str:
    if not missing_components:
        return (
            "The prompt already contains all major structural sections. "
            "Focus on improving clarity, precision, and organization.\n\n"
        )
    lines = "\n".join(
        f"- Add a {SECTION_TOKEN_MAP.get(c, c.upper())} section"
        for c in missing_components
    )
    return (
        f"Rewrite the prompt by adding the following missing sections:\n"
        f"{lines}\n\n"
    )


def clarity_template(raw_prompt: str, missing: List[str]) -> Dict[str, str]:
    """Original clarity-focused rewrite instruction."""
    system = (
        "You are a prompt refinement engine. Your ONLY task is to rewrite the "
        "user's prompt into a clearer, more specific, and well-structured "
        "version.\n\n"
        f"{_missing_block(missing)}{_BASE_RULES}"
    )
    user = (
        f"Rewrite this prompt by adding the missing structural sections "
        f"to improve clarity and usability:\n\n{raw_prompt}"
    )
    return {"system": system, "user": user}


def production_template(raw_prompt: str, missing: List[str]) -> Dict[str, str]:
    """Frame the rewrite as a production-ready spec for an engineering team."""
    system = (
        "You are a senior prompt engineer producing a production-ready "
        "specification for an engineering team. Rewrite the user's prompt as "
        "a precise, unambiguous brief suitable for handoff.\n\n"
        f"{_missing_block(missing)}{_BASE_RULES}"
    )
    user = (
        f"Convert this prompt into a production-ready specification with "
        f"explicit role, objective, requirements, constraints, and output "
        f"format sections:\n\n{raw_prompt}"
    )
    return {"system": system, "user": user}


def structured_reasoning_template(raw_prompt: str, missing: List[str]) -> Dict[str, str]:
    """Frame the rewrite as ordered reasoning steps with an IO contract."""
    system = (
        "You are a prompt refinement engine that organises prompts as "
        "ordered reasoning steps with a clear input/output contract.\n\n"
        f"{_missing_block(missing)}{_BASE_RULES}"
    )
    user = (
        f"Rewrite this prompt as a numbered sequence of reasoning steps with "
        f"an explicit input contract and output contract:\n\n{raw_prompt}"
    )
    return {"system": system, "user": user}


def api_ready_template(raw_prompt: str, missing: List[str]) -> Dict[str, str]:
    """Frame the rewrite as a structured API-style request."""
    system = (
        "You are a prompt refinement engine that produces API-ready prompts. "
        "Rewrite the user's prompt with explicit labelled fields: ROLE, "
        "OBJECTIVE, REQUIREMENTS, CONSTRAINTS, OUTPUT FORMAT.\n\n"
        f"{_missing_block(missing)}{_BASE_RULES}"
    )
    user = (
        f"Rewrite this prompt as a structured API-ready request with each "
        f"field on its own labelled line:\n\n{raw_prompt}"
    )
    return {"system": system, "user": user}


def deterministic_execution_template(raw_prompt: str, missing: List[str]) -> Dict[str, str]:
    """Tighten ambiguity and add measurable acceptance criteria."""
    system = (
        "You are a prompt refinement engine focused on deterministic "
        "execution. Eliminate vague tokens, replace fuzzy quantifiers with "
        "measurable acceptance criteria, and lock down the output contract.\n\n"
        f"{_missing_block(missing)}{_BASE_RULES}"
    )
    user = (
        f"Rewrite this prompt to remove ambiguity and add measurable "
        f"acceptance criteria so the output is deterministic:\n\n{raw_prompt}"
    )
    return {"system": system, "user": user}


INSTRUCTION_TEMPLATES: List[Callable[[str, List[str]], Dict[str, str]]] = [
    clarity_template,
    production_template,
    structured_reasoning_template,
    api_ready_template,
    deterministic_execution_template,
]


def build_atomic_rewrite_instruction(
    raw_prompt: str, missing_components: List[str]
) -> Dict[str, str]:
    """Backward-compatible alias — defaults to the clarity template."""
    return clarity_template(raw_prompt, missing_components)


# ── Candidate Validation ────────────────────────────────────────────────

# Patterns that suggest the model answered the task instead of rewriting
_TASK_ANSWER_INDICATORS = [
    re.compile(r"^(?:Here(?:'s| is)|Sure|Of course|Certainly|Let me)", re.IGNORECASE),
    re.compile(r"^(?:The answer|The solution|The result)\s", re.IGNORECASE),
    re.compile(r"^```(?:python|javascript|html|css|sql|bash|java|cpp)", re.IGNORECASE),
    re.compile(r"^(?:def |class |import |from |function |const |let |var )", re.IGNORECASE),
]

# Patterns that suggest the candidate contains examples
_EXAMPLE_INDICATORS = re.compile(
    r"\b(?:example\s*(?:input|output|response|result)|"
    r"sample\s*(?:input|output|response|result)|"
    r"for\s+example\s*:|e\.g\.\s*:)\s*\n",
    re.IGNORECASE,
)


def is_valid_rewrite(candidate: str, raw_prompt: str) -> tuple:
    """
    Validate that a candidate is a legitimate prompt rewrite, not a task answer.

    Returns:
        (is_valid: bool, rejection_reason: str or None)
    """
    if not candidate or not candidate.strip():
        return False, "empty_candidate"

    candidate_stripped = candidate.strip()
    raw_stripped = raw_prompt.strip()

    # Reject verbatim or near-verbatim copies
    if candidate_stripped.lower() == raw_stripped.lower():
        return False, "verbatim_copy"

    # Reject trivially short outputs
    if len(candidate_stripped) < 20:
        return False, "too_short"

    # Reject candidates shorter than 50% of raw prompt word count
    raw_words = len(raw_stripped.split())
    cand_words = len(candidate_stripped.split())
    if raw_words > 5 and cand_words < raw_words * 0.5:
        return False, "shorter_than_raw"

    # Reject if the model answered the task instead of rewriting
    for pattern in _TASK_ANSWER_INDICATORS:
        if pattern.match(candidate_stripped):
            return False, "task_answer_detected"

    # Reject if candidate contains example blocks
    if _EXAMPLE_INDICATORS.search(candidate_stripped):
        return False, "contains_examples"

    # Reject candidates with shallow/filler sections
    if not check_section_usefulness(candidate_stripped):
        return False, "shallow_sections"

    return True, None


# ── Section Usefulness Detection ─────────────────────────────────────────

# Pattern to detect section headers like "ROLE:", "## OBJECTIVE", "[CONSTRAINTS]", etc.
_SECTION_HEADER_PATTERN = re.compile(
    r"^(?:#{1,3}\s+|\[)?(?:ROLE|TASK|OBJECTIVE|CONTEXT|INPUT|OUTPUT|CONSTRAINTS|"
    r"STEPS|REQUIREMENTS|ASSUMPTIONS|EDGE\s*CASES|EXAMPLES)(?:\])?[\s:]*$",
    re.IGNORECASE | re.MULTILINE,
)

# Generic filler phrases that indicate a section has no real content.
_FILLER_PHRASES = [
    "as needed", "as appropriate", "as required", "as necessary",
    "follow best practices", "use best practices", "industry best practices",
    "ensure quality", "high quality output", "well-structured output",
    "provide a comprehensive", "deliver a comprehensive",
    "n/a", "not applicable", "none", "tbd", "to be determined",
]


def check_section_usefulness(candidate: str) -> bool:
    """
    Validate that section headers in a candidate have meaningful content.

    Returns False if:
      - A section header is followed by empty or trivially short content
      - A section contains only generic filler phrases
      - More than half the sections are shallow (< 15 chars of real content)

    Returns True if the candidate has no sections (unstructured is fine)
    or if sections are genuinely useful.
    """
    lines = candidate.split("\n")
    sections_found = 0
    shallow_sections = 0

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Check if this line is a section header
        if _SECTION_HEADER_PATTERN.match(line):
            sections_found += 1

            # Collect content until next header or end
            content_lines = []
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if _SECTION_HEADER_PATTERN.match(next_line):
                    break
                if next_line:  # Skip blank lines
                    content_lines.append(next_line)
                j += 1

            # Check if content is meaningful
            content_text = " ".join(content_lines).strip().lower()

            if not content_text:
                # Empty section
                shallow_sections += 1
            elif len(content_text) < 15:
                # Trivially short section content
                shallow_sections += 1
            else:
                # Check for generic filler
                is_filler = any(
                    filler in content_text for filler in _FILLER_PHRASES
                )
                # If the content is ONLY filler phrases (no substance beyond them)
                remaining = content_text
                for filler in _FILLER_PHRASES:
                    remaining = remaining.replace(filler, "").strip()
                if is_filler and len(remaining) < 15:
                    shallow_sections += 1

            i = j  # Skip to next section
        else:
            i += 1

    # If no sections found, the candidate is unstructured — that's fine
    if sections_found == 0:
        return True

    # Reject if more than half the sections are shallow
    if shallow_sections > sections_found * 0.5:
        logger.debug(
            f"  Section usefulness check failed: {shallow_sections}/{sections_found} shallow"
        )
        return False

    return True


# ── Candidate Deduplication ──────────────────────────────────────────────

def _normalize_for_dedup(text: str) -> str:
    """Normalize text for near-duplicate detection: lowercase, collapse whitespace, strip punctuation."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)                 # Collapse whitespace
    text = re.sub(r"[^\w\s]", "", text)               # Strip punctuation
    return text


# ── Atomic Rewriter ─────────────────────────────────────────────────────

class AtomicRewriter:
    """
    Scorer-guided atomic rewriter using the local Qwen2.5-7B-Instruct model.

    Generates a single rewrite candidate per attempt using a dynamically
    built instruction that targets only the missing structural components
    detected by the HeuristicScorer.

    Replaces the previous multi-style 14–21 candidate generation with
    a focused 1–3 attempt pipeline.
    """

    def __init__(self):
        from tools.prompt_optimizer import PromptOptimizer

        self.optimizer = PromptOptimizer()
        self.optimizer.load_model()

    def generate_atomic_candidate(
        self,
        raw_prompt: str,
        rewrite_instruction: Dict[str, str],
        temperature: float = REWRITE_TEMPERATURE_BASE,
    ) -> Optional[str]:
        """
        Generate a single rewrite candidate using the atomic instruction.

        Args:
            raw_prompt: The original prompt text.
            rewrite_instruction: Dict with 'system' and 'user' keys from
                build_atomic_rewrite_instruction().
            temperature: Sampling temperature for this attempt.

        Returns:
            The generated candidate string, or None if generation fails.
        """
        try:
            result = self.optimizer.rewrite(
                raw_prompt,
                sys_prompt_override=rewrite_instruction["system"],
                user_prompt_template=rewrite_instruction["user"].replace(
                    raw_prompt, "{}"
                ),
                temperature=temperature,
                top_p=REWRITE_TOP_P,
            )

            # The optimizer returns raw_prompt as fallback on failure
            if result == raw_prompt:
                return None

            return result

        except Exception as e:
            logger.warning(f"  Generation error: {e}")
            return None


# ── Negative Candidate Generator ────────────────────────────────────────

# Section keywords used by the section-stripping transformations.
_SECTION_KEYWORDS: Dict[str, List[str]] = {
    "constraints": ["constraints", "constraint", "limitations", "restrictions"],
    "context": ["context", "background", "scenario", "situation"],
    "output_format": ["output format", "output", "format", "deliverable", "deliverables"],
}

# Sentence-level patterns used as a fallback when a section header is absent.
_CONTEXT_SENTENCE_PATTERN = re.compile(
    r"\b(?:context|background|scenario|given\s+that|assuming|suppose|consider\s+that)\b",
    re.IGNORECASE,
)

# Filler / hedge banks for `add_noise`.
_NOISE_FILLERS = ("as needed", "somehow", "in some way", "as appropriate")
_NOISE_HEDGES = ("Maybe ", "Probably ", "Kind of ", "Sort of ")


class NegativeCandidateGenerator:
    """
    Synthesises structurally-degraded variants of a chosen rewrite.

    Each transformation preserves task intent (verified later by the scorer's
    semantic floor) but removes one quality dimension — structure, specificity,
    or completeness. Returns None when a transformation is a no-op so the
    caller can skip emitting trivial duplicates.
    """

    def __init__(self, nlp):
        self._nlp = nlp
        self.transformations: Dict[str, Callable[[str], Optional[str]]] = {
            "remove_structure": self.remove_structure,
            "make_vague": self.make_vague,
            "remove_constraints": self.remove_constraints,
            "remove_context": self.remove_context,
            "remove_output_format": self.remove_output_format,
            "shorten": self.shorten,
            "add_noise": self.add_noise,
            "reduce_specificity": self.reduce_specificity,
        }

    # ── Section / structure stripping helpers ────────────────────────

    def _strip_section(self, text: str, keywords: List[str]) -> Optional[str]:
        """Remove markdown sections, labelled lines, or sentences for given keywords."""
        kw_alt = "|".join(re.escape(k) for k in keywords)
        section_header = re.compile(
            r"^\s*(?:#{1,6}\s+|\*\*\s*)?(?:" + kw_alt + r")(?:\s*\*\*)?\s*[:]?\s*$",
            re.IGNORECASE,
        )
        next_header = re.compile(
            r"^\s*(?:#{1,6}\s+\S|\*\*\s*\w+|[A-Z][A-Z _]+:\s*$)"
        )
        inline_label = re.compile(
            r"^\s*(?:" + kw_alt + r")\s*:", re.IGNORECASE
        )

        lines = text.split("\n")
        out: List[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if section_header.match(line):
                i += 1
                while i < len(lines) and not next_header.match(lines[i]):
                    i += 1
                continue
            if inline_label.match(line):
                i += 1
                continue
            out.append(line)
            i += 1

        result = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
        if not result or result == text.strip():
            return None
        return result

    # ── Public transformations ───────────────────────────────────────

    def remove_structure(self, chosen: str) -> Optional[str]:
        text = chosen
        text = re.sub(r"^\s*#{1,6}\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*[-*\u2022]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text if text and text != chosen.strip() else None

    def make_vague(self, chosen: str) -> Optional[str]:
        text = re.sub(r"\b\d+\b", "some", chosen)
        text = re.sub(r"\bspecific\b", "various", text, flags=re.IGNORECASE)
        text = re.sub(r"\bexactly\b", "approximately", text, flags=re.IGNORECASE)
        text = re.sub(r"\bmust\b", "should probably", text, flags=re.IGNORECASE)
        text = re.sub(r"\bprecise(?:ly)?\b", "somewhat", text, flags=re.IGNORECASE)
        return text if text != chosen else None

    def remove_constraints(self, chosen: str) -> Optional[str]:
        return self._strip_section(chosen, _SECTION_KEYWORDS["constraints"])

    def remove_context(self, chosen: str) -> Optional[str]:
        stripped = self._strip_section(chosen, _SECTION_KEYWORDS["context"])
        base = stripped if stripped is not None else chosen
        # Drop sentences containing context cues (fallback).
        doc = self._nlp(base)
        kept = [
            s.text.strip() for s in doc.sents
            if not _CONTEXT_SENTENCE_PATTERN.search(s.text)
        ]
        result = " ".join(kept).strip()
        if not result or result == chosen.strip():
            return None
        return result

    def remove_output_format(self, chosen: str) -> Optional[str]:
        return self._strip_section(chosen, _SECTION_KEYWORDS["output_format"])

    def shorten(self, chosen: str) -> Optional[str]:
        doc = self._nlp(chosen)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        if len(sents) < 2:
            return None
        # Keep ~60% to stay above the length floor; clamp to at least 1.
        keep = max(int(round(len(sents) * 0.6)), 1)
        if keep >= len(sents):
            return None
        result = " ".join(sents[:keep]).strip()
        return result if result and result != chosen.strip() else None

    def add_noise(self, chosen: str) -> Optional[str]:
        noise_type = random.choice(("filler", "hedge", "repetition", "typo"))

        if noise_type == "filler":
            sentences = re.split(r"(?<=[.!?])\s+", chosen)
            if not sentences:
                return None
            idx = random.randrange(len(sentences))
            sentences[idx] = sentences[idx].rstrip(".!?") + " " + random.choice(_NOISE_FILLERS) + "."
            result = " ".join(sentences)

        elif noise_type == "hedge":
            sentences = re.split(r"(?<=[.!?])\s+", chosen)
            if not sentences or not sentences[0]:
                return None
            idx = random.randrange(len(sentences))
            s = sentences[idx]
            if s:
                sentences[idx] = random.choice(_NOISE_HEDGES) + s[0].lower() + s[1:]
            result = " ".join(sentences)

        elif noise_type == "repetition":
            words = chosen.split()
            if len(words) < 4:
                return None
            i = random.randint(1, len(words) - 1)
            words.insert(i, words[i])
            result = " ".join(words)

        else:  # typo
            words = chosen.split()
            long_idx = [i for i, w in enumerate(words) if len(re.sub(r"\W", "", w)) > 4]
            if not long_idx:
                return None
            i = random.choice(long_idx)
            w = words[i]
            pos = random.randint(0, len(w) - 2)
            words[i] = w[:pos] + w[pos + 1] + w[pos] + w[pos + 2:]
            result = " ".join(words)

        return result if result and result != chosen else None

    def reduce_specificity(self, chosen: str) -> Optional[str]:
        doc = self._nlp(chosen)
        drop_indices = set()
        for token in doc:
            if token.dep_ == "nummod":
                drop_indices.add(token.i)
            elif (
                token.dep_ == "amod"
                and token.lemma_.lower() not in AMBIGUOUS_TOKENS
            ):
                drop_indices.add(token.i)
        if not drop_indices:
            return None
        parts: List[str] = []
        for token in doc:
            if token.i in drop_indices:
                continue
            parts.append(token.text + token.whitespace_)
        result = re.sub(r"\s+", " ", "".join(parts)).strip()
        return result if result and result != chosen.strip() else None

    # ── Subtle structural contrast ───────────────────────────────────

    def partial_component_dropout(self, chosen: str) -> Optional[str]:
        """Remove exactly one of {constraints, context, output_format}."""
        choice = random.choice(("constraints", "context", "output_format"))
        if choice == "constraints":
            return self.remove_constraints(chosen)
        if choice == "context":
            return self.remove_context(chosen)
        return self.remove_output_format(chosen)


# ── Length & Rejection Validation ───────────────────────────────────────

def _spacy_token_count(text: str, nlp) -> int:
    """Count informative spaCy tokens (excluding punctuation/whitespace)."""
    doc = nlp(text)
    return sum(1 for t in doc if not t.is_punct and not t.is_space)


def length_ratio_ok(chosen: str, rejected: str, nlp) -> bool:
    """True iff the rejected/chosen length ratio is within [0.6, 1.4]."""
    cn = _spacy_token_count(chosen, nlp)
    if cn == 0:
        return False
    ratio = _spacy_token_count(rejected, nlp) / cn
    return LENGTH_RATIO_MIN <= ratio <= LENGTH_RATIO_MAX


def is_valid_rejection(
    rejected: str, chosen: str, raw: str, scorer: HeuristicScorer
) -> tuple:
    """Validate a synthesised rejection. Returns (ok, reason_or_none)."""
    if not rejected or not rejected.strip():
        return False, "empty"

    rej = rejected.strip()
    rej_n = _normalize_for_dedup(rej)
    if rej_n == _normalize_for_dedup(chosen):
        return False, "duplicate_of_chosen"
    if rej_n == _normalize_for_dedup(raw):
        return False, "duplicate_of_raw"

    for pattern in _TASK_ANSWER_INDICATORS:
        if pattern.match(rej):
            return False, "task_answer_detected"

    if not length_ratio_ok(chosen, rej, scorer._nlp):
        return False, "length_ratio_out_of_bounds"

    score = scorer.evaluate(raw, rej)
    if score["semantic_preservation"] < SEMANTIC_FLOOR:
        return False, "semantic_below_floor"

    return True, None


# ── Main Pipeline ────────────────────────────────────────────────────────

def load_raw_prompts(path: str) -> list[str]:
    """Load raw prompts from CSV (single column 'bad_prompt') or JSONL."""
    prompts = []
    if path.endswith(".csv"):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row.get("bad_prompt", "").strip()
                if prompt:
                    prompts.append(prompt)
    elif path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    prompts.append(json.loads(line)["prompt"])
    else:
        raise ValueError(f"Unsupported file format: {path}")
    return prompts


def build_preference_pairs(
    raw_prompts_path: str = RAW_PROMPTS_PATH,
    output_path: str = OUTPUT_PATH,
    dry_run: bool = False,
    dry_run_limit: int = 5,
    min_gain: float = MIN_GAIN,
    max_attempts: int = MAX_REWRITE_ATTEMPTS,
    target_size: int = TARGET_DATASET_SIZE,
) -> int:
    """
    Generate a multi-negative DPO dataset using the scorer-guided pipeline.

    Pipeline per prompt:
      1. Baseline scoring + missing-component detection.
      2. Per attempt: sample one of 5 instruction templates, generate, score,
         keep highest-quality accepted candidate as `chosen`.
      3. Synthesise structured `rejected` variants from the chosen text using
         `NegativeCandidateGenerator` (8 transformations + partial dropout).
      4. Always emit a wide-margin (chosen vs raw) baseline rejection per
         accepted prompt for anchor signal.
      5. Gate every rejection on length-ratio (0.6–1.4 spaCy tokens) and
         semantic preservation vs raw (≥ 0.40). Emit one DPO row per passing
         (chosen, rejected) pair.
      6. Stop early once `target_size` rows are accumulated.

    Returns:
        Number of preference pairs written.
    """
    import torch
    logger.info(f"System Check | PyTorch CUDA available: {torch.cuda.is_available()}")

    # Step 1: Initialize Rewriter
    logger.info("Initializing AtomicRewriter (requires GPU)")
    rewriter = AtomicRewriter()

    # Load raw prompts
    if not os.path.exists(raw_prompts_path):
        logger.error(f"Raw prompts file not found: {raw_prompts_path}")
        return 0

    all_prompts = load_raw_prompts(raw_prompts_path)
    logger.info(f"Loaded {len(all_prompts)} total prompts from {raw_prompts_path}")

    # Instructional filter intentionally removed — every raw prompt is processed
    # to maximise the pool that feeds the multi-negative expansion. Quality is
    # gated downstream by the scorer + length-ratio + semantic checks.
    if dry_run:
        all_prompts = all_prompts[:dry_run_limit]

    # Build indexed records
    records: List[PromptRecord] = [
        PromptRecord(idx=i, raw_prompt=p)
        for i, p in enumerate(all_prompts)
    ]

    # Initialize scorer for baseline scoring and component detection
    # NOTE: Scorer is loaded alongside the rewriter here because we need
    # it for component detection BEFORE generation. The rewriter model
    # is unloaded after generation to free VRAM for scorer embedding ops.
    scorer = HeuristicScorer()

    # ── PHASE 1: Baseline scoring + component detection ──────────────
    logger.info("\n--- PHASE 1: BASELINE SCORING & COMPONENT DETECTION ---")

    for record in records:
        raw_score = scorer.evaluate(record.raw_prompt)
        record.baseline_quality = raw_score["raw_quality"]
        record.missing_components = detect_missing_components(record.raw_prompt)
        logger.info(
            f"[{record.idx}] Baseline quality: {record.baseline_quality:.4f}, "
            f"Missing: {record.missing_components}"
        )

    # ── PHASE 2: Generate, score immediately, select best candidate ──
    logger.info(
        f"\n--- PHASE 2: ATOMIC REWRITE + IMMEDIATE SCORING "
        f"(max {max_attempts} attempts per prompt) ---"
    )

    total_attempts = 0
    total_valid = 0
    pairs: List[Dict[str, Any]] = []
    skipped = 0
    rejection_type_counts: Dict[str, int] = {}
    accepted_chosen_count = 0
    target_reached = False
    negative_generator = NegativeCandidateGenerator(scorer._nlp)

    for record in records:
        if target_reached:
            break

        logger.info(
            f"Rewriting prompt {record.idx + 1}/{len(records)}: "
            f"{record.raw_prompt[:60]}..."
        )

        # Track the best candidate across all attempts
        best_candidate = None
        best_score = record.baseline_quality
        best_score_result = None

        # Attempt generation with increasing temperature; sample a fresh
        # instruction template per attempt so the rewrites cover multiple
        # framing styles (clarity, production-spec, reasoning, API, deterministic).
        for attempt in range(max_attempts):
            temperature = REWRITE_TEMPERATURE_BASE + (attempt * REWRITE_TEMPERATURE_STEP)
            temperature = min(temperature, 0.60)
            total_attempts += 1
            record.attempt_count = attempt + 1

            template_fn = random.choice(INSTRUCTION_TEMPLATES)
            instruction = template_fn(
                record.raw_prompt, record.missing_components
            )

            candidate = rewriter.generate_atomic_candidate(
                record.raw_prompt, instruction, temperature=temperature
            )

            if candidate is None:
                logger.debug(
                    f"  Attempt {attempt + 1}: generation returned None"
                )
                continue

            # Validate candidate
            is_valid, reason = is_valid_rewrite(candidate, record.raw_prompt)
            if not is_valid:
                logger.debug(
                    f"  Attempt {attempt + 1}: rejected validation ({reason})"
                )
                continue

            total_valid += 1

            # Score immediately — compare against baseline
            score_result = scorer.evaluate(record.raw_prompt, candidate)

            # Hard gate: semantic preservation floor
            if score_result["rejected"]:
                logger.debug(
                    f"  Attempt {attempt + 1}: rejected by scorer "
                    f"(semantic={score_result['semantic_preservation']:.4f})"
                )
                continue

            # Acceptance conditions
            semantic_ok = score_result["semantic_preservation"] >= 0.40
            quality_ok = score_result["candidate_quality"] > record.baseline_quality
            gain_ok = score_result["final_score"] >= min_gain

            if not (semantic_ok and quality_ok and gain_ok):
                logger.debug(
                    f"  Attempt {attempt + 1}: below threshold "
                    f"(quality={score_result['candidate_quality']:.4f}, "
                    f"gain={score_result['final_score']:.4f})"
                )
                continue

            # Keep only if this candidate beats the current best
            candidate_quality = score_result["candidate_quality"]
            if candidate_quality > best_score:
                best_candidate = candidate
                best_score = candidate_quality
                best_score_result = score_result
                logger.debug(
                    f"  Attempt {attempt + 1}: new best "
                    f"(quality={candidate_quality:.4f}, "
                    f"{len(candidate.split())} words)"
                )

        # After all attempts: accept best or skip
        if best_candidate is None or best_score_result is None:
            logger.warning(
                f"[{record.idx}] No candidate passed all thresholds "
                f"after {record.attempt_count} attempts"
            )
            skipped += 1
            continue

        # Accept best candidate for this record
        record.accepted = True
        record.best_candidate = best_candidate
        record.best_final_score = best_score_result["final_score"]
        record.best_candidate_quality = best_score_result["candidate_quality"]
        record.metadata = {
            "final_score": round(best_score_result["final_score"], 4),
            "quality_delta": round(
                best_score_result["candidate_quality"] - record.baseline_quality, 4
            ),
            "clarity_delta": round(best_score_result["clarity_delta"], 4),
            "specificity_delta": round(best_score_result["specificity_delta"], 4),
            "semantic_preservation": round(
                best_score_result["semantic_preservation"], 4
            ),
            "components_added": record.missing_components,
            "attempt_count": record.attempt_count,
        }
        accepted_chosen_count += 1

        # ── Multi-negative emission ──────────────────────────────────
        # Always include the wide-margin (chosen vs raw) baseline first.
        rejected_variants: List[Tuple[str, str]] = [
            (record.raw_prompt, "raw_baseline"),
        ]

        # Run all 8 transformations on the chosen text, gating each result.
        for name, fn in negative_generator.transformations.items():
            try:
                candidate_neg = fn(record.best_candidate)
            except Exception as exc:  # pragma: no cover — defensive
                logger.debug(f"  Transformation {name} raised: {exc}")
                continue
            if candidate_neg is None:
                continue
            ok, reason = is_valid_rejection(
                candidate_neg, record.best_candidate, record.raw_prompt, scorer
            )
            if not ok:
                logger.debug(f"  Rejection '{name}' dropped: {reason}")
                continue
            rejected_variants.append((candidate_neg, name))

        # Subtle structural contrast — drop exactly one component.
        try:
            dropout_candidate = negative_generator.partial_component_dropout(
                record.best_candidate
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug(f"  partial_component_dropout raised: {exc}")
            dropout_candidate = None
        if dropout_candidate is not None:
            ok, reason = is_valid_rejection(
                dropout_candidate, record.best_candidate, record.raw_prompt, scorer
            )
            if ok:
                rejected_variants.append((dropout_candidate, "partial_dropout"))
            else:
                logger.debug(f"  Rejection 'partial_dropout' dropped: {reason}")

        # Emit one DPO row per variant.
        emitted_for_record = 0
        for rejected_text, rejection_type in rejected_variants:
            pairs.append({
                "prompt": record.raw_prompt,
                "chosen": record.best_candidate,
                "rejected": rejected_text,
                "rejection_type": rejection_type,
            })
            rejection_type_counts[rejection_type] = (
                rejection_type_counts.get(rejection_type, 0) + 1
            )
            emitted_for_record += 1

        logger.info(
            f"[{record.idx}] Emitted {emitted_for_record} preference rows "
            f"(running total: {len(pairs)} / target {target_size})"
        )

        if dry_run:
            print(f"\n{'='*70}")
            print(f"[{record.idx + 1}] Raw: {record.raw_prompt}")
            print(f"    Baseline quality:  {record.baseline_quality:.4f}")
            print(f"    Missing:           {record.missing_components}")
            print(f"    Attempts:          {record.attempt_count}")
            print(
                f"    y_w ({len(record.best_candidate.split())} words): "
                f"{record.best_candidate[:200]}..."
            )
            print(f"    y_w quality:       {record.best_candidate_quality:.4f}")
            print(f"    final_score:       {record.best_final_score:.4f}")
            print(f"    Rejections:        {emitted_for_record} variants — "
                  f"{[t for _, t in rejected_variants]}")

        # Early-stop once the target is reached (skipped in dry-run so we
        # always inspect the requested number of prompts).
        if not dry_run and len(pairs) >= target_size:
            target_reached = True
            logger.info(
                f"Target dataset size {target_size} reached — stopping early."
            )

    logger.info(
        f"Generated {total_valid}/{total_attempts} valid chosen candidates "
        f"for {len(records)} prompts ({accepted_chosen_count} accepted)"
    )

    # Unload Qwen to free VRAM
    logger.info("\nUnloading Qwen generation model to free GPU VRAM...")
    del rewriter.optimizer.model
    del rewriter.optimizer.tokenizer
    del rewriter
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    # Build distribution summary for the rejection-type histogram.
    histogram_lines = "\n".join(
        f"    {name:>22}: {count}"
        for name, count in sorted(
            rejection_type_counts.items(), key=lambda kv: -kv[1]
        )
    ) or "    (none)"

    # Write output
    if not dry_run:
        if len(pairs) < target_size:
            logger.warning(
                f"Final dataset size ({len(pairs)}) is below target "
                f"({target_size}). Writing what we have."
            )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        logger.info(f"\nWrote {len(pairs)} preference pairs to {output_path}")
        logger.info(
            f"Rejection-type distribution:\n{histogram_lines}"
        )
    else:
        print(f"\n{'='*70}")
        print(
            f"DRY RUN COMPLETE: {len(pairs)} preference pairs from "
            f"{accepted_chosen_count} accepted chosens, {skipped} skipped"
        )
        acceptance_rate = (
            (accepted_chosen_count / len(records) * 100) if records else 0
        )
        avg_negatives = (
            (len(pairs) / accepted_chosen_count) if accepted_chosen_count else 0
        )
        print(f"Chosen acceptance rate: {acceptance_rate:.1f}%")
        print(f"Avg rejections per chosen: {avg_negatives:.2f}")
        print(f"Total attempts: {total_attempts}")
        print(f"Rejection-type histogram:\n{histogram_lines}")

    logger.info(
        f"Done. Pairs: {len(pairs)}, Accepted chosens: "
        f"{accepted_chosen_count}, Skipped: {skipped}, "
        f"Total processed: {len(records)}, "
        f"Total generation attempts: {total_attempts}"
    )
    return len(pairs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build DPO preference pairs (scorer-guided atomic pipeline)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only 5 prompts and print results without writing",
    )
    parser.add_argument(
        "--dry-run-limit",
        type=int,
        default=5,
        help="Number of prompts to process in dry run (default: 5)",
    )
    parser.add_argument(
        "--input", default=RAW_PROMPTS_PATH, help="Path to raw prompts CSV or JSONL"
    )
    parser.add_argument(
        "--output", default=OUTPUT_PATH, help="Path to write preference pairs JSONL"
    )
    parser.add_argument(
        "--min-gain",
        type=float,
        default=MIN_GAIN,
        help=f"Minimum final_score threshold for acceptance (default: {MIN_GAIN})",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=MAX_REWRITE_ATTEMPTS,
        help=f"Maximum rewrite attempts per prompt (default: {MAX_REWRITE_ATTEMPTS})",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=TARGET_DATASET_SIZE,
        help=(
            f"Target preference-pair count; pipeline early-stops once reached "
            f"(default: {TARGET_DATASET_SIZE})"
        ),
    )
    args = parser.parse_args()

    build_preference_pairs(
        raw_prompts_path=args.input,
        output_path=args.output,
        dry_run=args.dry_run,
        dry_run_limit=args.dry_run_limit,
        min_gain=args.min_gain,
        max_attempts=args.max_attempts,
        target_size=args.target_size,
    )
