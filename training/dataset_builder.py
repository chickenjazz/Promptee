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
import logging
import argparse
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Project root for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from tools.heuristic_scorer import HeuristicScorer, _COMPLETENESS_PATTERNS

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


# ── Atomic Rewrite Instruction Builder ──────────────────────────────────

def build_atomic_rewrite_instruction(
    raw_prompt: str, missing_components: List[str]
) -> Dict[str, str]:
    """
    Build a single atomic rewrite instruction targeting all missing components.

    Returns a dict with 'system' and 'user' prompts ready for the optimizer.
    The instruction adapts dynamically based on which components are missing.
    """
    # Build the section-token hint list
    if missing_components:
        missing_sections_text = "\n".join(
            f"- Add a {SECTION_TOKEN_MAP.get(comp, comp.upper())} section"
            for comp in missing_components
        )
        component_block = (
            f"Rewrite the prompt by adding the following missing sections:\n"
            f"{missing_sections_text}\n\n"
        )
    else:
        component_block = (
            "The prompt already contains all major structural sections. "
            "Focus on improving clarity, precision, and organization.\n\n"
        )

    system_prompt = (
        "You are a prompt refinement engine. Your ONLY task is to rewrite the "
        "user's prompt into a clearer, more specific, and well-structured "
        "version.\n\n"
        f"{component_block}"
        "Rules:\n"
        "- Preserve the original task intent and meaning exactly\n"
        "- Only improve structure and clarity\n"
        "- Do NOT solve the task or generate the output the prompt requests\n"
        "- Do NOT include examples, sample inputs, or sample outputs\n"
        "- Do NOT include placeholder content or generic filler\n"
        "- Output ONLY the rewritten prompt — no explanations or commentary\n"
    )

    user_prompt = (
        f"Rewrite this prompt by adding the missing structural sections "
        f"to improve clarity and usability:\n\n{raw_prompt}"
    )

    return {"system": system_prompt, "user": user_prompt}


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
) -> int:
    """
    Generate preference pairs using the scorer-guided atomic rewrite pipeline.

    Pipeline per prompt:
      1. Baseline scoring → detect missing components
      2. Build atomic rewrite instruction with section-token hints
      3. Generate up to max_attempts candidates (varied temperature)
      4. Validate → Score immediately → Track best candidate
      5. Select highest-quality candidate → Construct DPO pair

    Args:
        raw_prompts_path: Path to raw prompts CSV or JSONL
        output_path: Path to write preference pairs JSONL
        dry_run: If True, process only dry_run_limit prompts and print results
        dry_run_limit: Number of prompts to process in dry run
        min_gain: Minimum final_score threshold for acceptance
        max_attempts: Maximum rewrite attempts per prompt

    Returns:
        Number of valid preference pairs generated
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

    # Filter for instructional tasks only
    instructional = [p for p in all_prompts if is_instructional(p)]
    logger.info(
        f"Filtered to {len(instructional)} instructional prompts "
        f"({len(all_prompts) - len(instructional)} rejected as non-instructional)"
    )

    if dry_run:
        instructional = instructional[:dry_run_limit]

    # Build indexed records
    records: List[PromptRecord] = [
        PromptRecord(idx=i, raw_prompt=p)
        for i, p in enumerate(instructional)
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
    pairs = []
    skipped = 0

    for record in records:
        logger.info(
            f"Rewriting prompt {record.idx + 1}/{len(records)}: "
            f"{record.raw_prompt[:60]}..."
        )

        # Build a single rewrite instruction for this prompt
        instruction = build_atomic_rewrite_instruction(
            record.raw_prompt, record.missing_components
        )

        # Track the best candidate across all attempts
        best_candidate = None
        best_score = record.baseline_quality
        best_score_result = None

        # Attempt generation with increasing temperature, score each immediately
        for attempt in range(max_attempts):
            temperature = REWRITE_TEMPERATURE_BASE + (attempt * REWRITE_TEMPERATURE_STEP)
            temperature = min(temperature, 0.60)
            total_attempts += 1
            record.attempt_count = attempt + 1

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

        pair = {
            "prompt": record.raw_prompt,
            "chosen": record.best_candidate,
            "rejected": record.raw_prompt,
        }
        pairs.append(pair)

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
            print(f"    Metadata:          {json.dumps(record.metadata, indent=6)}")

    logger.info(
        f"Generated {total_valid}/{total_attempts} valid candidates "
        f"for {len(records)} prompts"
    )

    # Unload Qwen to free VRAM
    logger.info("\nUnloading Qwen generation model to free GPU VRAM...")
    del rewriter.optimizer.model
    del rewriter.optimizer.tokenizer
    del rewriter
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    # Write output
    if not dry_run:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        logger.info(f"\nWrote {len(pairs)} preference pairs to {output_path}")
    else:
        print(f"\n{'='*70}")
        print(f"DRY RUN COMPLETE: {len(pairs)} valid pairs, {skipped} skipped")
        acceptance_rate = (
            (len(pairs) / len(records) * 100) if records else 0
        )
        print(f"Acceptance rate: {acceptance_rate:.1f}%")
        print(f"Total attempts: {total_attempts}")
        print(f"Avg attempts per accepted: {total_attempts / max(len(pairs), 1):.1f}")

    logger.info(
        f"Done. Accepted: {len(pairs)}, Skipped: {skipped}, "
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
    args = parser.parse_args()

    build_preference_pairs(
        raw_prompts_path=args.input,
        output_path=args.output,
        dry_run=args.dry_run,
        dry_run_limit=args.dry_run_limit,
        min_gain=args.min_gain,
        max_attempts=args.max_attempts,
    )
