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
from collections import defaultdict
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

# NOTE: Global random.seed removed. Pipeline uses RejectionPipelineConfig.random_seed
# via a dedicated random.Random instance for reproducibility without side-effects.

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

# ── Multi-Negative DPO Expansion Constants (legacy — used as fallbacks) ──
TARGET_DATASET_SIZE = 1200
LENGTH_RATIO_MIN = 0.6
LENGTH_RATIO_MAX = 1.4
SEMANTIC_FLOOR = 0.40


# ── Centralized Rejection Pipeline Configuration ────────────────────────

@dataclass
class RejectionPipelineConfig:
    """
    Centralized, configurable parameters for the multi-negative rejection
    pipeline.  All magic numbers live here so tuning never requires editing
    transform or validation code.
    """
    # How many transforms to sample per accepted chosen prompt
    transforms_per_prompt: int = 5

    # Relative selection weights for each transform (higher = more likely)
    transform_weights: Dict[str, float] = field(default_factory=lambda: {
        # Structural
        "remove_structure":         1.0,
        "remove_constraints":       1.0,
        "remove_output_format":     1.0,
        "section_shuffle":          1.2,
        "section_merge":            1.2,
        "partial_component_dropout": 0.8,
        # Semantic
        "weaken_objective":         1.5,
        "reduce_specificity":       1.5,
        "ambiguity_injection":      1.3,
        "remove_key_requirement":   1.3,
        # Coherence
        "contradictory_constraints": 1.5,
        "vague_success_criteria":   1.3,
        # Surface (low weight — deprioritised)
        "make_vague":              0.8,
        "shorten":                 0.6,
        "add_noise":               0.3,
    })

    # Soft category quotas per chosen prompt (targets, not hard limits)
    category_quotas: Dict[str, int] = field(default_factory=lambda: {
        "structural": 2,
        "semantic":   2,
        "coherence":  1,
        "surface":    1,
        "llm_generated": 1,
    })

    # Validation thresholds
    semantic_floor: float = 0.40
    max_similarity_between_negatives: float = 0.85
    min_quality_gap: float = 0.05
    length_ratio_min: float = 0.6
    length_ratio_max: float = 1.4

    # LLM-generated negatives
    llm_negative_ratio: float = 0.20
    llm_negatives_per_prompt: int = 1

    # Difficulty tier mix (targets)
    difficulty_mix: Dict[str, float] = field(default_factory=lambda: {
        "easy":   0.25,
        "medium": 0.50,
        "hard":   0.25,
    })

    # Dataset
    target_dataset_size: int = 1200
    random_seed: int = 42

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
    Scorer-guided atomic rewriter using the local Qwen2.5-3B-Instruct model.

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


# ── Negative Candidate Generator v2 ─────────────────────────────────────
#
# Category-organised, weighted-sampling-ready negative generator.
# 15 transforms across 4 categories: structural, semantic, coherence, surface.

# Strong action verbs that indicate clear, specific prompts.
STRONG_VERBS = {
    "implement", "design", "create", "build", "develop", "generate",
    "analyze", "evaluate", "optimize", "calculate", "define", "specify",
    "construct", "deploy", "integrate", "refactor", "validate", "debug",
    "configure", "extract", "transform", "classify", "benchmark",
    "automate", "orchestrate", "diagnose", "synthesize", "compile",
}

# Weak replacement verbs that make objectives vague.
WEAK_VERBS = {
    "handle", "deal with", "work on", "look at", "do something with",
    "try to", "consider", "think about", "explore", "address",
    "manage", "process", "help with", "use", "touch on",
}

# Section keywords used by the section-stripping transformations.
_SECTION_KEYWORDS: Dict[str, List[str]] = {
    "constraints": ["constraints", "constraint", "limitations", "restrictions"],
    "context": ["context", "background", "scenario", "situation"],
    "output_format": ["output format", "output", "format", "deliverable", "deliverables"],
    "requirements": ["requirements", "requirement", "specs", "specifications"],
    "objective": ["objective", "goal", "purpose", "task", "aim"],
}

# Sentence-level patterns used as a fallback when a section header is absent.
_CONTEXT_SENTENCE_PATTERN = re.compile(
    r"\b(?:context|background|scenario|given\s+that|assuming|suppose|consider\s+that)\b",
    re.IGNORECASE,
)

# Filler / hedge banks for `add_noise`.
_NOISE_FILLERS = ("as needed", "somehow", "in some way", "as appropriate")
_NOISE_HEDGES = ("Maybe ", "Probably ", "Kind of ", "Sort of ")

# LLM degradation system prompts for model-generated negatives.
_LLM_DEGRADATION_PROMPTS: Dict[str, str] = {
    "vague": (
        "You are a prompt degradation engine. Rewrite the user's prompt to be "
        "LESS specific. Remove concrete numbers, replace precise technical "
        "terms with general ones, and weaken measurable criteria into vague "
        "qualifiers. Preserve the same general task/topic.\n\n"
        "Rules:\n"
        "- Output ONLY the degraded prompt — no explanations\n"
        "- Keep the same topic and domain\n"
        "- Make it noticeably vaguer but still a coherent prompt\n"
        "- Do NOT add nonsense or break grammar"
    ),
    "underspecified": (
        "You are a prompt degradation engine. Rewrite the user's prompt by "
        "REMOVING requirements and constraints. Keep only the bare core task "
        "statement. Strip away role definitions, output format specifications, "
        "and detailed requirements. Preserve the same general task/topic.\n\n"
        "Rules:\n"
        "- Output ONLY the degraded prompt — no explanations\n"
        "- Result should feel incomplete but still be a real prompt\n"
        "- Keep the same topic and domain\n"
        "- Do NOT add nonsense or break grammar"
    ),
    "weakly_structured": (
        "You are a prompt degradation engine. Rewrite the user's prompt as a "
        "SINGLE PARAGRAPH without any formatting: no headers, no bullet points, "
        "no numbered lists, no section labels. Merge everything into continuous "
        "prose. Preserve the same general task/topic.\n\n"
        "Rules:\n"
        "- Output ONLY the degraded prompt — no explanations\n"
        "- Remove ALL structural formatting\n"
        "- Keep the same topic and domain\n"
        "- Keep it grammatically correct"
    ),
    "overgeneralized": (
        "You are a prompt degradation engine. Rewrite the user's prompt to be "
        "MORE GENERIC. Replace specific technologies, frameworks, and domain "
        "terms with general equivalents. Replace concrete examples with abstract "
        "references. Preserve the same general task category.\n\n"
        "Rules:\n"
        "- Output ONLY the degraded prompt — no explanations\n"
        "- Keep the same broad task type\n"
        "- Make it less actionable by removing specifics\n"
        "- Do NOT add nonsense or break grammar"
    ),
}

# Patterns for detecting measurable criteria (used by vague_success_criteria).
_MEASURABLE_CRITERIA_PATTERN = re.compile(
    r"(?:at\s+(?:least|most)\s+\d+|"
    r"no\s+(?:more|fewer)\s+than\s+\d+|"
    r"within\s+\d+|"
    r"under\s+\d+|"
    r"\d+\s*(?:seconds?|minutes?|hours?|ms|words?|characters?|lines?|items?|%|percent))",
    re.IGNORECASE,
)

# Vague replacement phrases for measurable criteria.
_VAGUE_CRITERIA_REPLACEMENTS = [
    "in a reasonable time",
    "with good quality",
    "as efficiently as possible",
    "to an acceptable standard",
    "in a suitable manner",
    "to a high degree",
]


class NegativeCandidateGenerator:
    """
    v2 — Category-organised, weighted-sampling-ready negative generator.

    15 transforms across 4 categories:
      - Structural: remove/shuffle/merge/dropout sections
      - Semantic:   weaken objectives, reduce specificity, inject ambiguity
      - Coherence:  contradictory constraints, vague success criteria
      - Surface:    make_vague, shorten, add_noise (deprioritised)

    Each transformation preserves task intent (verified later by the scorer's
    semantic floor) but degrades one quality dimension. Returns None when a
    transformation is a no-op so the caller can skip trivial duplicates.
    """

    # Maps each transform name to its quality category.
    CATEGORY_MAP: Dict[str, str] = {
        # Structural
        "remove_structure":         "structural",
        "remove_constraints":       "structural",
        "remove_output_format":     "structural",
        "section_shuffle":          "structural",
        "section_merge":            "structural",
        "partial_component_dropout": "structural",
        # Semantic
        "weaken_objective":         "semantic",
        "reduce_specificity":       "semantic",
        "ambiguity_injection":      "semantic",
        "remove_key_requirement":   "semantic",
        # Coherence
        "contradictory_constraints": "coherence",
        "vague_success_criteria":   "coherence",
        # Surface (deprioritised)
        "make_vague":              "surface",
        "shorten":                 "surface",
        "add_noise":               "surface",
    }

    def __init__(self, nlp, rng: random.Random = None):
        self._nlp = nlp
        self._rng = rng or random.Random(42)

        # All transforms registered in a single dict (replaces old
        # `transformations` + separate `partial_component_dropout`).
        self.all_transforms: Dict[str, Callable[[str], Optional[str]]] = {
            # Structural
            "remove_structure":         self.remove_structure,
            "remove_constraints":       self.remove_constraints,
            "remove_output_format":     self.remove_output_format,
            "section_shuffle":          self.section_shuffle,
            "section_merge":            self.section_merge,
            "partial_component_dropout": self.partial_component_dropout,
            # Semantic
            "weaken_objective":         self.weaken_objective,
            "reduce_specificity":       self.reduce_specificity,
            "ambiguity_injection":      self.ambiguity_injection,
            "remove_key_requirement":   self.remove_key_requirement,
            # Coherence
            "contradictory_constraints": self.contradictory_constraints,
            "vague_success_criteria":   self.vague_success_criteria,
            # Surface
            "make_vague":              self.make_vague,
            "shorten":                 self.shorten,
            "add_noise":               self.add_noise,
        }

        # Legacy compat: old code referenced `self.transformations`
        self.transformations = self.all_transforms

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

    def _find_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse text into top-level sections (header + body lines).
        Returns list of {'header': str, 'body': list[str], 'start': int, 'end': int}.
        """
        header_pat = re.compile(
            r"^\s*(?:#{1,6}\s+\S|[A-Z][A-Za-z _]+:\s*$|\*\*\s*\w+)", re.MULTILINE
        )
        lines = text.split("\n")
        sections: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None

        for i, line in enumerate(lines):
            if header_pat.match(line):
                if current is not None:
                    current["end"] = i - 1
                    sections.append(current)
                current = {"header": line, "body": [], "start": i, "end": i}
            elif current is not None:
                current["body"].append(line)
            # Lines before the first header are ignored for section operations

        if current is not None:
            current["end"] = len(lines) - 1
            sections.append(current)

        return sections

    # ═══════════════════════════════════════════════════════════════════
    # A. STRUCTURAL TRANSFORMS
    # ═══════════════════════════════════════════════════════════════════

    def remove_structure(self, chosen: str) -> Optional[str]:
        """Flatten all markdown structure into a single paragraph."""
        text = chosen
        text = re.sub(r"^\s*#{1,6}\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*[-*\u2022]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text if text and text != chosen.strip() else None

    def remove_constraints(self, chosen: str) -> Optional[str]:
        """Remove the constraints/limitations section entirely."""
        return self._strip_section(chosen, _SECTION_KEYWORDS["constraints"])

    def remove_output_format(self, chosen: str) -> Optional[str]:
        """Remove the output format/deliverables section entirely."""
        return self._strip_section(chosen, _SECTION_KEYWORDS["output_format"])

    def section_shuffle(self, chosen: str) -> Optional[str]:
        """Randomly reorder top-level sections, breaking logical flow."""
        sections = self._find_sections(chosen)
        if len(sections) < 2:
            return None

        # Build prefix (any lines before the first section header)
        first_start = sections[0]["start"]
        lines = chosen.split("\n")
        prefix_lines = lines[:first_start]

        # Shuffle sections
        shuffled = list(sections)
        self._rng.shuffle(shuffled)

        # Check if order actually changed
        if all(s["start"] == o["start"] for s, o in zip(shuffled, sections)):
            return None

        # Reassemble
        result_lines = list(prefix_lines)
        for sec in shuffled:
            result_lines.append(sec["header"])
            result_lines.extend(sec["body"])

        result = "\n".join(result_lines).strip()
        return result if result and result != chosen.strip() else None

    def section_merge(self, chosen: str) -> Optional[str]:
        """Merge two adjacent sections into a single paragraph, removing structure
        without removing content."""
        sections = self._find_sections(chosen)
        if len(sections) < 2:
            return None

        # Pick a random index to merge section[i] and section[i+1]
        merge_idx = self._rng.randint(0, len(sections) - 2)
        lines = chosen.split("\n")

        # Build the merged content (strip headers, join as prose)
        sec_a = sections[merge_idx]
        sec_b = sections[merge_idx + 1]
        merged_body = " ".join(
            line.strip()
            for line in sec_a["body"] + sec_b["body"]
            if line.strip()
        )
        if not merged_body:
            return None

        # Reassemble: keep sections before merge_idx, insert merged, keep rest
        result_lines = []
        for i, sec in enumerate(sections):
            if i == merge_idx:
                result_lines.append(merged_body)
            elif i == merge_idx + 1:
                continue  # absorbed into merge
            else:
                result_lines.append(sec["header"])
                result_lines.extend(sec["body"])

        result = "\n".join(result_lines).strip()
        return result if result and result != chosen.strip() else None

    def partial_component_dropout(self, chosen: str) -> Optional[str]:
        """Remove exactly one of {constraints, context, output_format}."""
        choice = self._rng.choice(("constraints", "context", "output_format"))
        if choice == "constraints":
            return self.remove_constraints(chosen)
        if choice == "context":
            return self._strip_section(chosen, _SECTION_KEYWORDS["context"])
        return self.remove_output_format(chosen)

    # ═══════════════════════════════════════════════════════════════════
    # B. SEMANTIC TRANSFORMS
    # ═══════════════════════════════════════════════════════════════════

    def weaken_objective(self, chosen: str) -> Optional[str]:
        """Replace strong action verbs with weak verbs, specific goals with
        vague phrases. Targets the verb-driven clarity of the prompt."""
        text = chosen
        changes = 0
        for strong in STRONG_VERBS:
            pattern = re.compile(r"\b" + re.escape(strong) + r"\b", re.IGNORECASE)
            if pattern.search(text):
                # Pick a random weak replacement
                weak = self._rng.choice(list(WEAK_VERBS))
                text = pattern.sub(weak, text, count=1)
                changes += 1
                if changes >= 3:
                    break
        return text if changes > 0 and text != chosen else None

    def reduce_specificity(self, chosen: str) -> Optional[str]:
        """Drop adjectival modifiers and numeric modifiers to reduce precision."""
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

    def ambiguity_injection(self, chosen: str) -> Optional[str]:
        """Insert 2–3 ambiguous tokens from AMBIGUOUS_TOKENS into key sentences."""
        doc = self._nlp(chosen)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        if len(sents) < 2:
            return None

        # Pick 2–3 sentences to inject ambiguity into
        n_inject = min(self._rng.randint(2, 3), len(sents))
        targets = self._rng.sample(range(len(sents)), n_inject)

        ambig_tokens = list(AMBIGUOUS_TOKENS)
        for idx in targets:
            token = self._rng.choice(ambig_tokens)
            words = sents[idx].split()
            if len(words) > 2:
                insert_pos = self._rng.randint(1, len(words) - 1)
                words.insert(insert_pos, token)
                sents[idx] = " ".join(words)

        result = " ".join(sents).strip()
        return result if result != chosen.strip() else None

    def remove_key_requirement(self, chosen: str) -> Optional[str]:
        """Drop 1–2 specific bullet items from requirements-like sections.
        Subtler than full section removal — removes specific requirements
        rather than the entire section."""
        lines = chosen.split("\n")
        bullet_pattern = re.compile(r"^\s*[-*\u2022]\s+.+", re.UNICODE)

        # Find all bullet indices
        bullet_indices = [i for i, line in enumerate(lines) if bullet_pattern.match(line)]
        if len(bullet_indices) < 3:
            # Not enough bullets to remove safely
            return None

        # Remove 1–2 bullets
        n_remove = min(self._rng.randint(1, 2), len(bullet_indices) - 1)
        remove_indices = set(self._rng.sample(bullet_indices, n_remove))

        result_lines = [line for i, line in enumerate(lines) if i not in remove_indices]
        result = "\n".join(result_lines).strip()
        return result if result and result != chosen.strip() else None

    # ═══════════════════════════════════════════════════════════════════
    # C. COHERENCE TRANSFORMS
    # ═══════════════════════════════════════════════════════════════════

    def contradictory_constraints(self, chosen: str) -> Optional[str]:
        """Add a constraint that conflicts with an existing one.
        Introduces logical inconsistency while keeping the prompt readable."""
        # Contradictory pairs: if we detect one, we inject the other
        contradiction_pairs = [
            (r"\bshort\b|\bbrief\b|\bconcise\b",
             "Ensure the response is comprehensive and detailed, covering all aspects thoroughly."),
            (r"\bdetailed\b|\bcomprehensive\b|\bthorough\b",
             "Keep the response brief and under 100 words."),
            (r"\bsimple\b|\bbasic\b|\bminimal\b",
             "Include advanced edge cases and complex scenarios."),
            (r"\bJSON\b|\bjson\b",
             "Format the output as a plain-text narrative paragraph."),
            (r"\btable\b|\btabular\b",
             "Present all data as continuous prose without tables."),
            (r"\bno external\b|\bno dependencies\b|\bstandalone\b",
             "Integrate with at least 3 external libraries for robust functionality."),
            (r"\bstep.by.step\b|\bnumbered\b",
             "Provide the solution as a single holistic paragraph without steps."),
        ]

        injected = False
        result = chosen
        for pattern_str, contradiction in contradiction_pairs:
            if re.search(pattern_str, chosen, re.IGNORECASE):
                # Append the contradictory constraint
                result = result.rstrip() + "\n" + contradiction
                injected = True
                break

        if not injected:
            # Generic contradiction: add a length constraint that likely conflicts
            result = chosen.rstrip() + "\nLimit the entire response to exactly 50 words."

        return result if result != chosen else None

    def vague_success_criteria(self, chosen: str) -> Optional[str]:
        """Replace measurable acceptance criteria with vague qualitative ones."""
        matches = list(_MEASURABLE_CRITERIA_PATTERN.finditer(chosen))
        if not matches:
            return None

        text = chosen
        # Replace up to 2 measurable criteria
        replaced = 0
        for match in reversed(matches[:2]):
            replacement = self._rng.choice(_VAGUE_CRITERIA_REPLACEMENTS)
            text = text[:match.start()] + replacement + text[match.end():]
            replaced += 1

        return text if replaced > 0 and text != chosen else None

    # ═══════════════════════════════════════════════════════════════════
    # D. SURFACE TRANSFORMS (deprioritised)
    # ═══════════════════════════════════════════════════════════════════

    def make_vague(self, chosen: str) -> Optional[str]:
        """Broaden specific terms into vague alternatives."""
        text = re.sub(r"\b\d+\b", "some", chosen)
        text = re.sub(r"\bspecific\b", "various", text, flags=re.IGNORECASE)
        text = re.sub(r"\bexactly\b", "approximately", text, flags=re.IGNORECASE)
        text = re.sub(r"\bmust\b", "should probably", text, flags=re.IGNORECASE)
        text = re.sub(r"\bprecise(?:ly)?\b", "somewhat", text, flags=re.IGNORECASE)
        text = re.sub(r"\brequired\b", "optional", text, flags=re.IGNORECASE)
        text = re.sub(r"\bcritical\b", "nice to have", text, flags=re.IGNORECASE)
        text = re.sub(r"\bessential\b", "helpful", text, flags=re.IGNORECASE)
        return text if text != chosen else None

    def shorten(self, chosen: str) -> Optional[str]:
        """Drop internal sentences (not just tail) to simulate incomplete prompts."""
        doc = self._nlp(chosen)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        if len(sents) < 3:
            return None

        # Drop ~40% of sentences, keeping first and last for coherence
        n_drop = max(int(round(len(sents) * 0.4)), 1)
        # Only drop from internal sentences (index 1 to -1)
        internal_indices = list(range(1, len(sents) - 1))
        if not internal_indices:
            return None

        n_drop = min(n_drop, len(internal_indices))
        drop_set = set(self._rng.sample(internal_indices, n_drop))

        result_sents = [s for i, s in enumerate(sents) if i not in drop_set]
        result = " ".join(result_sents).strip()
        return result if result and result != chosen.strip() else None

    def add_noise(self, chosen: str) -> Optional[str]:
        """Add filler or hedge noise. Typo and repetition modes are deprioritised
        to favour more realistic quality degradations."""
        # Weighted towards filler/hedge (realistic), away from typo/repetition
        noise_type = self._rng.choices(
            ("filler", "hedge", "repetition", "typo"),
            weights=(4, 4, 1, 1),
            k=1,
        )[0]

        if noise_type == "filler":
            sentences = re.split(r"(?<=[.!?])\s+", chosen)
            if not sentences:
                return None
            idx = self._rng.randrange(len(sentences))
            sentences[idx] = sentences[idx].rstrip(".!?") + " " + self._rng.choice(_NOISE_FILLERS) + "."
            result = " ".join(sentences)

        elif noise_type == "hedge":
            sentences = re.split(r"(?<=[.!?])\s+", chosen)
            if not sentences or not sentences[0]:
                return None
            idx = self._rng.randrange(len(sentences))
            s = sentences[idx]
            if s:
                sentences[idx] = self._rng.choice(_NOISE_HEDGES) + s[0].lower() + s[1:]
            result = " ".join(sentences)

        elif noise_type == "repetition":
            words = chosen.split()
            if len(words) < 4:
                return None
            i = self._rng.randint(1, len(words) - 1)
            words.insert(i, words[i])
            result = " ".join(words)

        else:  # typo
            words = chosen.split()
            long_idx = [i for i, w in enumerate(words) if len(re.sub(r"\W", "", w)) > 4]
            if not long_idx:
                return None
            i = self._rng.choice(long_idx)
            w = words[i]
            pos = self._rng.randint(0, len(w) - 2)
            words[i] = w[:pos] + w[pos + 1] + w[pos] + w[pos + 2:]
            result = " ".join(words)

        return result if result and result != chosen else None

    # ═══════════════════════════════════════════════════════════════════
    # E. LLM-GENERATED NEGATIVES
    # ═══════════════════════════════════════════════════════════════════

    def generate_llm_negative(
        self,
        chosen: str,
        raw: str,
        rewriter: "AtomicRewriter",
        mode: str = None,
    ) -> Optional[str]:
        """
        Use the existing Qwen rewriter model to intentionally degrade a prompt.

        Modes: vague, underspecified, weakly_structured, overgeneralized.
        If mode is None, one is selected randomly.

        MUST be called before the rewriter model is unloaded.
        """
        if mode is None:
            mode = self._rng.choice(list(_LLM_DEGRADATION_PROMPTS.keys()))

        sys_prompt = _LLM_DEGRADATION_PROMPTS.get(mode)
        if sys_prompt is None:
            logger.warning(f"Unknown LLM degradation mode: {mode}")
            return None

        user_prompt = f"Degrade this prompt:\n\n{chosen}"

        try:
            result = rewriter.optimizer.rewrite(
                chosen,
                sys_prompt_override=sys_prompt,
                user_prompt_template=user_prompt.replace(chosen, "{}"),
                temperature=0.6,
                top_p=0.9,
            )
            # The optimizer returns the input as fallback on failure
            if result == chosen or result == raw:
                return None
            return result
        except Exception as e:
            logger.debug(f"  LLM negative generation failed: {e}")
            return None


# ── Weighted Transform Sampling ─────────────────────────────────────────

def sample_transforms(
    available: Dict[str, Callable],
    config: RejectionPipelineConfig,
    rng: random.Random,
) -> List[Tuple[str, Callable]]:
    """
    Sample k transforms using weighted random selection with category
    quota enforcement.

    Algorithm:
      1. Build per-category pools from available transforms.
      2. For each category with a quota, sample up to quota items (weighted).
      3. If total < k, fill remaining from all categories (weighted, no dupes).

    Returns list of (name, callable) tuples.
    """
    k = config.transforms_per_prompt
    weights = config.transform_weights
    quotas = config.category_quotas
    category_map = NegativeCandidateGenerator.CATEGORY_MAP

    # Build per-category pools
    category_pools: Dict[str, List[Tuple[str, Callable, float]]] = defaultdict(list)
    for name, fn in available.items():
        cat = category_map.get(name, "surface")
        w = weights.get(name, 1.0)
        if w > 0:
            category_pools[cat].append((name, fn, w))

    selected: List[Tuple[str, Callable]] = []
    selected_names: set = set()

    # Phase 1: fill quotas
    for cat, quota in quotas.items():
        if cat == "llm_generated":
            continue  # LLM negatives are handled separately
        pool = category_pools.get(cat, [])
        if not pool:
            continue
        names_pool = [p[0] for p in pool]
        fns_pool = [p[1] for p in pool]
        weights_pool = [p[2] for p in pool]
        n = min(quota, len(pool), k - len(selected))
        if n <= 0:
            continue
        chosen_indices = []
        remaining_indices = list(range(len(pool)))
        for _ in range(n):
            if not remaining_indices:
                break
            r_weights = [weights_pool[i] for i in remaining_indices]
            picks = rng.choices(remaining_indices, weights=r_weights, k=1)
            idx = picks[0]
            chosen_indices.append(idx)
            remaining_indices.remove(idx)
        for idx in chosen_indices:
            name = names_pool[idx]
            if name not in selected_names:
                selected.append((name, fns_pool[idx]))
                selected_names.add(name)

    # Phase 2: fill remaining up to k from all pools
    if len(selected) < k:
        all_remaining = [
            (name, fn, weights.get(name, 1.0))
            for cat, pool in category_pools.items()
            for name, fn, _ in pool
            if name not in selected_names and cat != "llm_generated"
        ]
        needed = k - len(selected)
        if all_remaining:
            remaining_names = [p[0] for p in all_remaining]
            remaining_fns = [p[1] for p in all_remaining]
            remaining_weights = [p[2] for p in all_remaining]
            n_fill = min(needed, len(all_remaining))
            indices = list(range(len(all_remaining)))
            for _ in range(n_fill):
                if not indices:
                    break
                r_w = [remaining_weights[i] for i in indices]
                picks = rng.choices(indices, weights=r_w, k=1)
                idx = picks[0]
                name = remaining_names[idx]
                if name not in selected_names:
                    selected.append((name, remaining_fns[idx]))
                    selected_names.add(name)
                indices.remove(idx)

    return selected


# ── Length & Rejection Validation (v2) ──────────────────────────────────

def _spacy_token_count(text: str, nlp) -> int:
    """Count informative spaCy tokens (excluding punctuation/whitespace)."""
    doc = nlp(text)
    return sum(1 for t in doc if not t.is_punct and not t.is_space)


def length_ratio_ok(
    chosen: str,
    rejected: str,
    nlp,
    config: RejectionPipelineConfig = None,
) -> bool:
    """True iff the rejected/chosen length ratio is within configured bounds."""
    cn = _spacy_token_count(chosen, nlp)
    if cn == 0:
        return False
    ratio = _spacy_token_count(rejected, nlp) / cn
    lo = config.length_ratio_min if config else LENGTH_RATIO_MIN
    hi = config.length_ratio_max if config else LENGTH_RATIO_MAX
    return lo <= ratio <= hi


def _assign_difficulty_tier(quality_gap: float) -> str:
    """Assign easy/medium/hard tier based on quality gap magnitude."""
    if quality_gap >= 0.20:
        return "easy"
    elif quality_gap >= 0.08:
        return "medium"
    else:
        return "hard"


def _grammar_sanity_check(text: str, nlp) -> bool:
    """Lightweight check that text has at least one complete dependency
    parse with a root. Caps input at 500 chars for performance."""
    doc = nlp(text[:500])
    roots = [t for t in doc if t.dep_ == "ROOT"]
    return len(roots) >= 1


def is_valid_rejection(
    rejected: str,
    chosen: str,
    raw: str,
    scorer: HeuristicScorer,
    chosen_quality: float = None,
    existing_negatives: List[str] = None,
    config: RejectionPipelineConfig = None,
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Enhanced rejection validation with quality-gap enforcement,
    inter-negative deduplication, and difficulty tier assignment.

    Returns: (ok, reason_or_none, metadata_or_none)

    Backward compatible: if chosen_quality / existing_negatives / config
    are not provided, falls back to legacy behaviour.
    """
    if not rejected or not rejected.strip():
        return False, "empty", None

    rej = rejected.strip()
    rej_n = _normalize_for_dedup(rej)
    if rej_n == _normalize_for_dedup(chosen):
        return False, "duplicate_of_chosen", None
    if rej_n == _normalize_for_dedup(raw):
        return False, "duplicate_of_raw", None

    for pattern in _TASK_ANSWER_INDICATORS:
        if pattern.match(rej):
            return False, "task_answer_detected", None

    # Length ratio check
    if not length_ratio_ok(chosen, rej, scorer._nlp, config):
        return False, "length_ratio_out_of_bounds", None

    # Semantic preservation vs raw
    score = scorer.evaluate(raw, rej)
    sem_floor = config.semantic_floor if config else SEMANTIC_FLOOR
    if score["semantic_preservation"] < sem_floor:
        return False, "semantic_below_floor", None

    # Grammar sanity
    if not _grammar_sanity_check(rej, scorer._nlp):
        return False, "grammar_broken", None

    # ── New v2 quality gates ──────────────────────────────────────────

    rejected_quality = score["candidate_quality"]
    quality_gap = None
    difficulty_tier = None

    if chosen_quality is not None:
        # Reject negatives scoring ABOVE chosen
        if rejected_quality >= chosen_quality:
            return False, "rejected_beats_chosen", None

        # Quality-gap minimum enforcement
        min_gap = config.min_quality_gap if config else 0.05
        quality_gap = chosen_quality - rejected_quality
        if quality_gap < min_gap:
            return False, "insufficient_quality_gap", None

        # Difficulty tier assignment
        difficulty_tier = _assign_difficulty_tier(quality_gap)

    # Inter-negative semantic deduplication
    if existing_negatives and scorer._st_model is not None:
        max_sim = config.max_similarity_between_negatives if config else 0.85
        try:
            from sentence_transformers import util as st_util
            rej_emb = scorer._st_model.encode(rej)
            for existing in existing_negatives:
                existing_emb = scorer._st_model.encode(existing)
                sim = st_util.cos_sim(rej_emb, existing_emb).item()
                if sim > max_sim:
                    return False, "too_similar_to_existing_negative", None
        except Exception as e:
            logger.debug(f"  Dedup embedding failed: {e}")

    metadata = {
        "rejected_quality": round(rejected_quality, 4),
        "quality_gap": round(quality_gap, 4) if quality_gap is not None else None,
        "difficulty_tier": difficulty_tier,
        "semantic_preservation": round(score["semantic_preservation"], 4),
    }
    return True, None, metadata


# ── Pipeline Metrics Tracker ────────────────────────────────────────────

@dataclass
class PipelineMetrics:
    """Tracks per-transform and aggregate pipeline metrics for observability."""

    # Per-transform tracking
    transform_attempts: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    transform_accepted: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    transform_rejected_reasons: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )

    # Quality metrics
    quality_gaps: List[float] = field(default_factory=list)

    # Distribution tracking
    tier_counts: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    category_counts: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    dedup_prevented: int = 0

    # Per-chosen tracking
    negatives_per_chosen: List[int] = field(default_factory=list)

    def log_attempt(
        self,
        transform_name: str,
        accepted: bool,
        reason: str = None,
        metadata: Dict[str, Any] = None,
    ) -> None:
        """Record a single transform attempt and its outcome."""
        self.transform_attempts[transform_name] += 1
        if accepted:
            self.transform_accepted[transform_name] += 1
            if metadata:
                gap = metadata.get("quality_gap")
                if gap is not None:
                    self.quality_gaps.append(gap)
                tier = metadata.get("difficulty_tier")
                if tier:
                    self.tier_counts[tier] += 1
                cat = NegativeCandidateGenerator.CATEGORY_MAP.get(
                    transform_name, "surface"
                )
                self.category_counts[cat] += 1
        else:
            if reason == "too_similar_to_existing_negative":
                self.dedup_prevented += 1
            self.transform_rejected_reasons[transform_name][reason or "unknown"] += 1

    def log_summary(self) -> None:
        """Log a comprehensive summary of pipeline metrics."""
        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE METRICS SUMMARY")
        logger.info("=" * 70)

        # Per-transform acceptance rates
        logger.info("\n  Per-Transform Acceptance Rates:")
        for name in sorted(self.transform_attempts.keys()):
            attempts = self.transform_attempts[name]
            accepted = self.transform_accepted.get(name, 0)
            rate = (accepted / attempts * 100) if attempts > 0 else 0
            logger.info(f"    {name:>30}: {accepted}/{attempts} ({rate:.1f}%)")

        # Category distribution
        logger.info("\n  Category Distribution:")
        for cat, count in sorted(self.category_counts.items(), key=lambda x: -x[1]):
            logger.info(f"    {cat:>20}: {count}")

        # Difficulty tier distribution
        logger.info("\n  Difficulty Tier Distribution:")
        for tier, count in sorted(self.tier_counts.items()):
            logger.info(f"    {tier:>10}: {count}")

        # Quality gap stats
        if self.quality_gaps:
            avg = sum(self.quality_gaps) / len(self.quality_gaps)
            mn = min(self.quality_gaps)
            mx = max(self.quality_gaps)
            logger.info(
                f"\n  Quality Gap: avg={avg:.4f}, min={mn:.4f}, max={mx:.4f}"
            )

        # Dedup stats
        logger.info(f"\n  Duplicate negatives prevented: {self.dedup_prevented}")

        # Negatives per chosen
        if self.negatives_per_chosen:
            avg_neg = sum(self.negatives_per_chosen) / len(self.negatives_per_chosen)
            logger.info(f"  Avg negatives per chosen: {avg_neg:.2f}")

        logger.info("=" * 70)


# ── Main Pipeline ────────────────────────────────────────────────────────

def load_raw_prompts(path: str) -> list[str]:
    """Load raw prompts from CSV (single column 'bad_prompt') or JSONL."""
    prompts = []
    if path.endswith(".csv"):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row.get("x", row.get("bad_prompt", "")).strip()
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
    config: RejectionPipelineConfig = None,
) -> int:
    """
    Generate a multi-negative DPO dataset using the v2 scorer-guided pipeline.

    Pipeline per prompt:
      1. Baseline scoring + missing-component detection.
      2. Per attempt: sample one of 5 instruction templates, generate, score,
         keep highest-quality accepted candidate as `chosen`.
      3. (While Qwen is loaded) Generate LLM-degraded negatives for each
         accepted chosen prompt.
      4. Unload Qwen to free VRAM.
      5. Sample k rule-based transforms (weighted, category-quota'd) and
         generate rejected variants.
      6. Validate every rejection: quality-gap ≥ min, rejected < chosen,
         inter-negative dedup, grammar sanity, difficulty tier assignment.
      7. Emit one DPO row per passing (chosen, rejected) pair.
      8. Stop early once `target_size` rows are accumulated.

    Returns:
        Number of preference pairs written.
    """
    import torch
    logger.info(f"System Check | PyTorch CUDA available: {torch.cuda.is_available()}")

    # Initialize config
    if config is None:
        config = RejectionPipelineConfig(
            target_dataset_size=target_size,
        )

    # Deterministic RNG
    rng = random.Random(config.random_seed)

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

    # ── PHASE 2: Generate chosen + LLM negatives (Qwen loaded) ───────
    logger.info(
        f"\n--- PHASE 2: ATOMIC REWRITE + IMMEDIATE SCORING "
        f"(max {max_attempts} attempts per prompt) ---"
    )

    total_attempts = 0
    total_valid = 0
    pairs: List[Dict[str, Any]] = []
    skipped = 0
    accepted_chosen_count = 0
    target_reached = False
    negative_generator = NegativeCandidateGenerator(scorer._nlp, rng=rng)
    metrics = PipelineMetrics()

    # Accumulate LLM negatives while the model is loaded
    llm_negatives_map: Dict[int, List[Tuple[str, str]]] = {}

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

            template_fn = rng.choice(INSTRUCTION_TEMPLATES)
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

        # ── Generate LLM-degraded negatives while Qwen is loaded ─────
        if config.llm_negative_ratio > 0:
            n_llm = config.llm_negatives_per_prompt
            llm_negs: List[Tuple[str, str]] = []
            for _ in range(n_llm):
                llm_neg = negative_generator.generate_llm_negative(
                    record.best_candidate, record.raw_prompt, rewriter
                )
                if llm_neg is not None:
                    llm_negs.append((llm_neg, "llm_negative"))
            if llm_negs:
                llm_negatives_map[record.idx] = llm_negs
                logger.debug(
                    f"  Generated {len(llm_negs)} LLM negative(s) for [{record.idx}]"
                )

        # Check early stopping (approximate — final count determined after
        # rule-based negatives are generated post-Qwen-unload).
        estimated_negatives = (
            1 + config.transforms_per_prompt + len(llm_negatives_map.get(record.idx, []))
        )
        if not dry_run and (len(pairs) + accepted_chosen_count * estimated_negatives) >= config.target_dataset_size:
            # Don't stop yet — we need the model for more LLM negatives.
            # But stop generating new chosens.
            pass

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

    # ── PHASE 3: Rule-based negatives + validation (post-Qwen) ───────
    logger.info(
        f"\n--- PHASE 3: MULTI-NEGATIVE GENERATION "
        f"(k={config.transforms_per_prompt} weighted sampling) ---"
    )

    for record in records:
        if not record.accepted:
            continue
        if target_reached:
            break

        # Collect accepted negative texts for inter-negative dedup
        accepted_neg_texts: List[str] = []

        # Start with wide-margin (chosen vs raw) baseline
        rejected_variants: List[Tuple[str, str, Optional[Dict]]] = []

        # Raw baseline always passes — it's the anchor signal, but still
        # validate quality gap.
        raw_ok, raw_reason, raw_meta = is_valid_rejection(
            record.raw_prompt, record.best_candidate, record.raw_prompt,
            scorer, record.best_candidate_quality, accepted_neg_texts, config,
        )
        if raw_ok:
            rejected_variants.append((record.raw_prompt, "raw_baseline", raw_meta))
            accepted_neg_texts.append(record.raw_prompt)
            metrics.log_attempt("raw_baseline", True, metadata=raw_meta)
        else:
            # Raw baseline should almost always pass; log if it doesn't
            logger.debug(f"  raw_baseline rejected: {raw_reason}")
            metrics.log_attempt("raw_baseline", False, reason=raw_reason)

        # ── Add LLM-generated negatives ──────────────────────────────
        for llm_text, llm_type in llm_negatives_map.get(record.idx, []):
            metrics.log_attempt("llm_negative", True)  # Track attempt
            ok, reason, meta = is_valid_rejection(
                llm_text, record.best_candidate, record.raw_prompt,
                scorer, record.best_candidate_quality, accepted_neg_texts, config,
            )
            if ok:
                rejected_variants.append((llm_text, "llm_negative", meta))
                accepted_neg_texts.append(llm_text)
                metrics.log_attempt("llm_negative", True, metadata=meta)
            else:
                logger.debug(f"  LLM rejection dropped: {reason}")
                metrics.log_attempt("llm_negative", False, reason=reason)

        # ── Weighted transform sampling ──────────────────────────────
        selected = sample_transforms(
            negative_generator.all_transforms, config, rng
        )

        for name, fn in selected:
            try:
                candidate_neg = fn(record.best_candidate)
            except Exception as exc:  # pragma: no cover — defensive
                logger.debug(f"  Transformation {name} raised: {exc}")
                metrics.log_attempt(name, False, reason="exception")
                continue
            if candidate_neg is None:
                metrics.log_attempt(name, False, reason="no_op")
                continue

            ok, reason, meta = is_valid_rejection(
                candidate_neg, record.best_candidate, record.raw_prompt,
                scorer, record.best_candidate_quality, accepted_neg_texts, config,
            )
            if not ok:
                logger.debug(f"  Rejection '{name}' dropped: {reason}")
                metrics.log_attempt(name, False, reason=reason)
                continue

            rejected_variants.append((candidate_neg, name, meta))
            accepted_neg_texts.append(candidate_neg)
            metrics.log_attempt(name, True, metadata=meta)

        # Emit one DPO row per variant.
        emitted_for_record = 0
        for rejected_text, rejection_type, meta in rejected_variants:
            row = {
                "prompt": record.raw_prompt,
                "chosen": record.best_candidate,
                "rejected": rejected_text,
                "rejection_type": rejection_type,
                "rejection_category": NegativeCandidateGenerator.CATEGORY_MAP.get(
                    rejection_type, "other"
                ),
                "difficulty_tier": meta.get("difficulty_tier") if meta else None,
                "quality_gap": meta.get("quality_gap") if meta else None,
                "chosen_quality": round(record.best_candidate_quality, 4),
                "rejected_quality": meta.get("rejected_quality") if meta else None,
            }
            pairs.append(row)
            emitted_for_record += 1

        metrics.negatives_per_chosen.append(emitted_for_record)

        logger.info(
            f"[{record.idx}] Emitted {emitted_for_record} preference rows "
            f"(running total: {len(pairs)} / target {config.target_dataset_size})"
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
            print(f"    Rejections:        {emitted_for_record} variants:")
            for rej_text, rej_type, rej_meta in rejected_variants:
                tier = rej_meta.get("difficulty_tier", "?") if rej_meta else "?"
                gap = rej_meta.get("quality_gap", "?") if rej_meta else "?"
                cat = NegativeCandidateGenerator.CATEGORY_MAP.get(rej_type, "other")
                clean_rej_text = rej_text.replace('\n', ' ')
                print(
                    f"      - [{rej_type}] ({cat}/{tier}, gap={gap}): "
                    f"{clean_rej_text[:120]}..."
                )

        # Early-stop once the target is reached (skipped in dry-run so we
        # always inspect the requested number of prompts).
        if not dry_run and len(pairs) >= config.target_dataset_size:
            target_reached = True
            logger.info(
                f"Target dataset size {config.target_dataset_size} reached — stopping early."
            )

    # ── PHASE 4: Write output + metrics ──────────────────────────────

    if not dry_run:
        if len(pairs) < config.target_dataset_size:
            logger.warning(
                f"Final dataset size ({len(pairs)}) is below target "
                f"({config.target_dataset_size}). Writing what we have."
            )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            # Schema version header
            header = {
                "_schema_version": 2,
                "_generator": "dataset_builder_v2",
                "_config": {
                    "transforms_per_prompt": config.transforms_per_prompt,
                    "min_quality_gap": config.min_quality_gap,
                    "semantic_floor": config.semantic_floor,
                    "max_similarity_between_negatives": config.max_similarity_between_negatives,
                    "llm_negative_ratio": config.llm_negative_ratio,
                    "random_seed": config.random_seed,
                },
            }
            f.write(json.dumps(header, ensure_ascii=False) + "\n")
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        logger.info(f"\nWrote {len(pairs)} preference pairs to {output_path}")
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

    # Log comprehensive metrics
    metrics.log_summary()

    logger.info(
        f"Done. Pairs: {len(pairs)}, Accepted chosens: "
        f"{accepted_chosen_count}, Skipped: {skipped}, "
        f"Total processed: {len(records)}, "
        f"Total generation attempts: {total_attempts}"
    )
    return len(pairs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build DPO preference pairs (v2 scorer-guided multi-negative pipeline)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only a few prompts and print results without writing",
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
    # ── v2 config arguments ──────────────────────────────────────────
    parser.add_argument(
        "--transforms-per-prompt",
        type=int,
        default=5,
        help="Number of transforms to sample per chosen prompt (default: 5)",
    )
    parser.add_argument(
        "--min-quality-gap",
        type=float,
        default=0.05,
        help="Minimum quality gap between chosen and rejected (default: 0.05)",
    )
    parser.add_argument(
        "--max-negative-similarity",
        type=float,
        default=0.85,
        help="Maximum cosine similarity between negatives (default: 0.85)",
    )
    parser.add_argument(
        "--llm-negative-ratio",
        type=float,
        default=0.20,
        help="Fraction of negatives generated by LLM degradation (default: 0.20)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling (default: 42)",
    )
    args = parser.parse_args()

    # Build config from CLI args
    pipeline_config = RejectionPipelineConfig(
        transforms_per_prompt=args.transforms_per_prompt,
        min_quality_gap=args.min_quality_gap,
        max_similarity_between_negatives=args.max_negative_similarity,
        llm_negative_ratio=args.llm_negative_ratio,
        target_dataset_size=args.target_size,
        random_seed=args.random_seed,
    )

    build_preference_pairs(
        raw_prompts_path=args.input,
        output_path=args.output,
        dry_run=args.dry_run,
        dry_run_limit=args.dry_run_limit,
        min_gain=args.min_gain,
        max_attempts=args.max_attempts,
        target_size=args.target_size,
        config=pipeline_config,
    )

