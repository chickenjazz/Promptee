"""
Dataset Builder — Preference Pair Generator for DPO Training

SOP Reference: architecture/dpo_training.md §3

Reads raw prompts from dataset/RAW_prompts.csv, filters for instructional tasks,
generates diverse modular rewrite candidates using dynamic section selection from
12 allowed sections (ROLE, TASK, OBJECTIVE, CONTEXT, INPUT, OUTPUT, CONSTRAINTS,
STEPS, REQUIREMENTS, ASSUMPTIONS, EDGE CASES, EXAMPLES), scores them with the
HeuristicScorer, and outputs preference pairs to datasets/preference_pairs.jsonl.

Multi-candidate generation produces 14–21 candidates per raw prompt (2–3 samples
across 7 rewriting styles) with diversity sampling (varied temperatures, nucleus
sampling, and perturbed decoding settings). Candidates are deduplicated, validated
for section usefulness, and ranked by the scorer's final reward signal. The best
candidate with positive improvement and valid semantic preservation is selected as
the preferred rewrite (y_w) for each DPO preference pair.

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

from tools.heuristic_scorer import HeuristicScorer

logger = logging.getLogger("promptee.dataset_builder")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

# Default paths
RAW_PROMPTS_PATH = os.path.join(PROJECT_ROOT, "dataset", "RAW_prompts.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "datasets", "preference_pairs.jsonl")

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


def is_instructional(prompt: str) -> bool:
    """
    Filter: returns True only if the prompt represents an instructional task.
    Rejects vague "tell me about" prompts, stories, poems, opinion questions.
    """
    prompt = prompt.strip()
    if not prompt or len(prompt) < 10:
        return False

    # Reject known non-instructional patterns
    for pattern in NON_INSTRUCTIONAL_PATTERNS:
        if pattern.search(prompt):
            return False

    # Accept if starts with an instructional verb
    if INSTRUCTIONAL_VERBS.search(prompt):
        return True

    # Accept "How do/can/should I..." patterns
    if re.match(r"^how (do|can|should|would|might) (i|we|you)\b", prompt, re.IGNORECASE):
        return True

    return False


# ── Indexed Record Structure ─────────────────────────────────────────────

@dataclass
class PromptRecord:
    """
    Indexed record for a single raw prompt and its candidates/scores.
    Prevents key collisions from duplicate prompts by using row index as ID.
    """
    idx: int
    raw_prompt: str
    candidates: List[str] = field(default_factory=list)
    score_results: List[Dict[str, Any]] = field(default_factory=list)
    best_candidate: Optional[str] = None
    best_final_score: float = -1.0
    best_candidate_quality: float = -1.0
    baseline_quality: float = 0.0
    accepted: bool = False


# ── Rewriting Style Definitions ──────────────────────────────────────────

# Base instruction block shared by all rewriting styles.
# Intentionally relaxes "preserve exactly" — model must preserve core intent
# but may enrich with missing constraints, format specs, assumptions, etc.
_BASE_SYSTEM_INSTRUCTION = (
    "You are an expert prompt engineer. Rewrite the user's prompt into a clearer, "
    "more specific, and well-structured prompt.\n\n"
    "Rules:\n"
    "- Choose only the sections that are genuinely useful for this specific task\n"
    "- Do NOT force a fixed template — the structure should fit the task, not the other way around\n"
    "- Preserve the core task intent, but you may enrich the prompt with:\n"
    "  * missing constraints or boundaries\n"
    "  * output format specifications\n"
    "  * reasonable assumptions\n"
    "  * task-specific detail and context\n"
    "  * concrete examples relevant to the task\n"
    "  * clarity improvements and disambiguation\n"
    "- Do NOT add generic filler — every section and sentence must add real value\n"
    "- Do NOT include examples that are not relevant to the specific task\n"
    "- Keep a natural, professional tone with high readability\n"
    "- Output ONLY the rewritten prompt — no explanations, preamble, or meta-commentary\n\n"
    "Available sections (use only what genuinely helps this task):\n"
    "ROLE, TASK, OBJECTIVE, CONTEXT, INPUT, OUTPUT, CONSTRAINTS, "
    "STEPS, REQUIREMENTS, ASSUMPTIONS, EDGE CASES, EXAMPLES\n\n"
)

# 7 rewriting styles with distinct information strategies (not just wording variants).
# Each style has a base temperature and top_p that will be perturbed for multi-sampling.
REWRITING_STYLES = [
    {
        "name": "concise_complete",
        "addendum": (
            "Style directive: Write a CONCISE but COMPLETE rewrite. "
            "Prioritize brevity and density — every sentence must carry information. "
            "Use short labeled sections and tight bullet points only where they add value. "
            "Strip all redundancy. Keep under 200 words if possible. "
            "If the task is simple, a single well-crafted paragraph may be better than sections."
        ),
        "temperature": 0.35,
        "top_p": 0.82,
        "user_template": "Rewrite this prompt into a concise, information-dense version:\n\n{}",
    },
    {
        "name": "constraint_rich",
        "addendum": (
            "Style directive: Write a CONSTRAINT-RICH rewrite that maximises specificity. "
            "Focus on what to include, what to exclude, formatting rules, length expectations, "
            "quality criteria, and error handling requirements. "
            "Add 4–8 concrete, task-specific constraints that the original prompt is missing. "
            "Constraints must be actionable and measurable, not vague."
        ),
        "temperature": 0.45,
        "top_p": 0.88,
        "user_template": "Rewrite this prompt with strong emphasis on specific, measurable constraints:\n\n{}",
    },
    {
        "name": "example_guided",
        "addendum": (
            "Style directive: Write an EXAMPLE-GUIDED rewrite where concrete examples "
            "clarify what the output should look like. "
            "Include 1–2 task-specific input/output examples or mock responses. "
            "Do NOT use generic placeholder examples — every example must be relevant to "
            "the actual task described in the prompt. "
            "Use examples to implicitly define quality expectations."
        ),
        "temperature": 0.50,
        "top_p": 0.90,
        "user_template": "Rewrite this prompt with task-specific examples that clarify the expected output:\n\n{}",
    },
    {
        "name": "procedural",
        "addendum": (
            "Style directive: Write a PROCEDURAL rewrite that breaks the task into clear steps. "
            "Use 3–7 numbered steps in logical execution order. "
            "Each step should be a single actionable instruction. "
            "Include expected deliverable or checkpoint after key steps. "
            "Only use steps if the task genuinely benefits from sequential breakdown."
        ),
        "temperature": 0.38,
        "top_p": 0.84,
        "user_template": "Rewrite this prompt as a clear step-by-step procedure:\n\n{}",
    },
    {
        "name": "expert_brief",
        "addendum": (
            "Style directive: Write an EXPERT BRIEF — a professional task assignment. "
            "Define a specific expert persona with relevant domain expertise. "
            "Provide background context and audience assumptions. "
            "Frame the prompt as a professional consulting request or technical brief. "
            "Balance detail with clarity — detailed but not verbose."
        ),
        "temperature": 0.55,
        "top_p": 0.90,
        "user_template": "Rewrite this prompt as an expert-level professional task brief:\n\n{}",
    },
    {
        "name": "output_focused",
        "addendum": (
            "Style directive: Write an OUTPUT-FOCUSED rewrite that prioritizes defining "
            "what the deliverable should look like. "
            "Specify output format (JSON, markdown, table, code, prose, etc.), "
            "structure requirements, length expectations, and quality criteria. "
            "The rewrite should make it impossible to misunderstand what the response format should be. "
            "Work backwards from the desired output to define the task."
        ),
        "temperature": 0.42,
        "top_p": 0.86,
        "user_template": "Rewrite this prompt with precise output format and deliverable specifications:\n\n{}",
    },
    {
        "name": "assumption_expanding",
        "addendum": (
            "Style directive: Write an ASSUMPTION-EXPANDING rewrite that surfaces hidden assumptions. "
            "Identify what the original prompt takes for granted and make it explicit: "
            "target audience, skill level, technology stack, scope boundaries, environment. "
            "Add reasonable assumptions as explicit statements so the responder doesn't have to guess. "
            "Reduce ambiguity by being specific about what is in-scope and out-of-scope."
        ),
        "temperature": 0.52,
        "top_p": 0.90,
        "user_template": "Rewrite this prompt by surfacing hidden assumptions and reducing ambiguity:\n\n{}",
    },
]

# Temperature perturbation offsets for multi-sampling within each style.
# Each style generates len(SAMPLE_PERTURBATIONS) candidates with varied decoding.
SAMPLE_PERTURBATIONS = [
    {"temp_offset": 0.00, "top_p_offset": 0.00},   # Base settings
    {"temp_offset": 0.10, "top_p_offset": -0.03},   # Slightly warmer, tighter nucleus
    {"temp_offset": -0.05, "top_p_offset": 0.05},   # Slightly cooler, wider nucleus
]


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


def deduplicate_candidates(candidates: List[str], similarity_threshold: float = 0.85) -> List[str]:
    """
    Remove exact and near-duplicate candidates.

    Uses normalized text comparison. Two candidates are considered near-duplicates
    if their normalized word overlap (Jaccard similarity) exceeds the threshold.

    Returns a list with duplicates removed, preserving order of first occurrence.
    """
    if len(candidates) <= 1:
        return candidates

    unique = []
    seen_normalized = []

    for candidate in candidates:
        norm = _normalize_for_dedup(candidate)
        norm_words = set(norm.split())

        is_duplicate = False
        for seen_norm in seen_normalized:
            seen_words = set(seen_norm.split())

            # Exact normalized match
            if norm == seen_norm:
                is_duplicate = True
                break

            # Jaccard similarity for near-duplicates
            if norm_words and seen_words:
                intersection = len(norm_words & seen_words)
                union = len(norm_words | seen_words)
                jaccard = intersection / union if union > 0 else 0.0
                if jaccard >= similarity_threshold:
                    is_duplicate = True
                    break

        if not is_duplicate:
            unique.append(candidate)
            seen_normalized.append(norm)

    removed = len(candidates) - len(unique)
    if removed > 0:
        logger.debug(f"  Deduplication removed {removed} near-duplicate candidates")

    return unique


# ── Modularized Rewrite Strategy ─────────────────────────────────────────

class RewriteStrategy:
    """
    Base class for rewrite candidate generation.
    """
    def generate_candidates(self, raw_prompt: str) -> list[str]:
        raise NotImplementedError


class DynamicLLMRewriter(RewriteStrategy):
    """
    Multi-candidate diverse rewriter using the local Qwen2.5-7B-Instruct model.

    Generates 14–21 candidate rewrites per raw prompt by iterating over 7 distinct
    rewriting styles with 2–3 temperature-perturbed samples per style:

      1. concise_complete      — tight, information-dense, minimal filler
      2. constraint_rich       — maximises constraints, boundaries, requirements
      3. example_guided        — leads with task-specific examples
      4. procedural            — step-by-step numbered breakdown
      5. expert_brief          — expert persona with professional framing
      6. output_focused        — deliverable-first, format-driven
      7. assumption_expanding  — surfaces hidden assumptions, reduces ambiguity

    Each style uses a distinct information strategy (not just wording variants).
    Dynamic section selection allows the model to choose only useful sections.

    Diversity is achieved through:
      - 7 styles with distinct structural and informational approaches
      - 2–3 samples per style with perturbed temperature and top_p
      - Different user prompt framings per style

    Candidates are:
      - Validated for basic quality (non-empty, non-copy, sufficient length)
      - Checked for section usefulness (no empty/filler sections)
      - Deduplicated using Jaccard similarity before scoring
    """

    def __init__(self):
        from tools.prompt_optimizer import PromptOptimizer

        self.optimizer = PromptOptimizer()
        self.optimizer.load_model()

        # Build full system prompts for each style
        self.styles = []
        for style_def in REWRITING_STYLES:
            self.styles.append({
                "name": style_def["name"],
                "sys_prompt": _BASE_SYSTEM_INSTRUCTION + style_def["addendum"],
                "temperature": style_def["temperature"],
                "top_p": style_def["top_p"],
                "user_template": style_def["user_template"],
            })

    def _validate_candidate(self, candidate: str, raw_prompt: str) -> bool:
        """
        Validate a generated candidate for basic quality and section usefulness.

        Rejects:
          - Empty or trivially short outputs (< 20 chars)
          - Verbatim or near-verbatim copies of the raw prompt
          - Outputs shorter than 50% of the raw prompt word count
          - Candidates with shallow/empty/filler sections
        """
        if not candidate or not candidate.strip():
            return False

        candidate_stripped = candidate.strip()
        raw_stripped = raw_prompt.strip()

        # Reject verbatim copies
        if candidate_stripped == raw_stripped:
            return False

        # Reject near-verbatim copies (case-insensitive)
        if candidate_stripped.lower() == raw_stripped.lower():
            return False

        # Reject trivially short outputs
        if len(candidate_stripped) < 20:
            return False

        # Reject candidates shorter than 50% of raw prompt word count
        raw_words = len(raw_stripped.split())
        cand_words = len(candidate_stripped.split())
        if raw_words > 5 and cand_words < raw_words * 0.5:
            return False

        # Reject candidates with shallow/filler sections
        if not check_section_usefulness(candidate_stripped):
            return False

        return True

    def generate_candidates(self, raw_prompt: str) -> list[str]:
        """
        Generate 14–21 diverse rewrite candidates for a single raw prompt.

        Iterates over all 7 rewriting styles, generating 2–3 samples per style
        with perturbed temperature and top_p values for diversity. Results are
        validated, then deduplicated before returning.

        Returns:
            Deduplicated list of valid candidate strings (may be empty if all fail).
        """
        raw_candidates = []
        total_attempts = 0

        for style in self.styles:
            for perturb in SAMPLE_PERTURBATIONS:
                # Compute perturbed decoding settings (clamp to safe ranges)
                temp = max(0.1, min(1.0, style["temperature"] + perturb["temp_offset"]))
                top_p = max(0.5, min(0.99, style["top_p"] + perturb["top_p_offset"]))
                total_attempts += 1

                try:
                    result = self.optimizer.rewrite(
                        raw_prompt,
                        sys_prompt_override=style["sys_prompt"],
                        user_prompt_template=style["user_template"],
                        temperature=temp,
                        top_p=top_p,
                    )

                    # Validate the generated candidate
                    if not self._validate_candidate(result, raw_prompt):
                        logger.debug(
                            f"  [{style['name']}|t={temp:.2f}] Rejected: failed validation"
                        )
                        continue

                    raw_candidates.append(result)
                    logger.debug(
                        f"  [{style['name']}|t={temp:.2f}] Accepted "
                        f"({len(result.split())} words)"
                    )

                except Exception as e:
                    logger.warning(
                        f"  [{style['name']}|t={temp:.2f}] Generation error: {e}"
                    )

        # Deduplicate before returning
        unique_candidates = deduplicate_candidates(raw_candidates)

        logger.info(
            f"  Generated {len(raw_candidates)}/{total_attempts} valid → "
            f"{len(unique_candidates)} unique after dedup"
        )
        return unique_candidates


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
    strategy: str = "dynamic",
    dry_run: bool = False,
    dry_run_limit: int = 5,
) -> int:
    """
    Generate preference pairs from raw prompts.

    Uses an indexed record structure (PromptRecord) to prevent key collisions
    from duplicate prompts. Candidates are ranked by the scorer's final_score
    (reward signal) first, with candidate_quality as a tiebreaker.

    All score comparisons use unrounded internal float values for accuracy.
    Rounding is applied only for display output.

    Args:
        raw_prompts_path: Path to raw prompts CSV or JSONL
        output_path: Path to write preference pairs JSONL
        strategy: Rewrite strategy (only "dynamic" supported)
        dry_run: If True, process only dry_run_limit prompts and print results
        dry_run_limit: Number of prompts to process in dry run

    Returns:
        Number of valid preference pairs generated
    """
    import torch
    logger.info(f"System Check | PyTorch CUDA available: {torch.cuda.is_available()}")

    # Step 1: Initialize Rewriter
    if strategy == "dynamic":
        logger.info("Using DynamicLLMRewriter (requires GPU)")
        rewriter = DynamicLLMRewriter()
    else:
        logger.warning(f"Strategy {strategy} is deprecated/unknown. Using dynamic LLM default.")
        rewriter = DynamicLLMRewriter()

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

    # Build indexed records (prevents key collisions from duplicate prompts)
    records: List[PromptRecord] = [
        PromptRecord(idx=i, raw_prompt=p)
        for i, p in enumerate(instructional)
    ]

    # BATCH PROCESS 1: Generation
    num_styles = len(REWRITING_STYLES)
    num_samples = len(SAMPLE_PERTURBATIONS)
    logger.info(
        f"\n--- STEP 1: BATCH GENERATING CANDIDATES "
        f"({num_styles} styles × {num_samples} samples) ---"
    )

    for record in records:
        logger.info(
            f"Generating candidates for prompt {record.idx + 1}/{len(records)} "
            f"({num_styles} styles × {num_samples} samples)"
        )
        record.candidates = rewriter.generate_candidates(record.raw_prompt)
        if not record.candidates:
            logger.warning(
                f"[{record.idx}] No candidates generated for: "
                f"{record.raw_prompt[:60]}..."
            )

    # Unload Qwen to free VRAM for the Scorer
    logger.info("\nUnloading Qwen Generation model to free GPU VRAM...")
    del rewriter.optimizer.model
    del rewriter.optimizer.tokenizer
    del rewriter
    import torch, gc
    torch.cuda.empty_cache()
    gc.collect()

    # BATCH PROCESS 2: Scoring
    logger.info("\n--- STEP 2: BATCH SCORING CANDIDATES ---")
    scorer = HeuristicScorer()

    pairs = []
    skipped = 0
    total_candidates_generated = 0
    total_unique_candidates = 0

    for record in records:
        total_candidates_generated += len(record.candidates)
        total_unique_candidates += len(record.candidates)

        if not record.candidates:
            skipped += 1
            continue

        # Score the original raw prompt as a baseline guarantee
        # Use unrounded internal value for comparison accuracy
        raw_score = scorer.evaluate(record.raw_prompt)
        record.baseline_quality = raw_score["raw_quality"]

        # Score all candidates and collect results
        scored_candidates = []
        candidates_rejected_semantic = 0

        for candidate in record.candidates:
            score_result = scorer.evaluate(record.raw_prompt, candidate)

            # Hard gate 1: reject if semantic preservation fails
            if score_result["rejected"]:
                candidates_rejected_semantic += 1
                continue

            scored_candidates.append({
                "candidate": candidate,
                "final_score": score_result["final_score"],
                "candidate_quality": score_result["candidate_quality"],
                "semantic_preservation": score_result["semantic_preservation"],
                "quality_improvement": score_result["quality_improvement"],
            })

        # Rank by final_score (reward signal) first, candidate_quality as tiebreaker
        scored_candidates.sort(
            key=lambda x: (x["final_score"], x["candidate_quality"]),
            reverse=True,
        )

        # Find best candidate with positive improvement
        best = None
        for sc in scored_candidates:
            # Use unrounded values for acceptance decision
            if sc["final_score"] > 0 and sc["candidate_quality"] > record.baseline_quality:
                best = sc
                break

        if best is None:
            best_info = scored_candidates[0] if scored_candidates else None
            best_fs = best_info["final_score"] if best_info else -1.0
            best_cq = best_info["candidate_quality"] if best_info else -1.0
            logger.warning(
                f"[{record.idx}] Rejected: No candidate with positive reward "
                f"(Best final_score: {best_fs:.6f}, best quality: {best_cq:.6f}, "
                f"baseline: {record.baseline_quality:.6f}, "
                f"evaluated: {len(scored_candidates)}, "
                f"semantic_rejected: {candidates_rejected_semantic})"
            )
            skipped += 1
            continue

        # Accept this record
        record.accepted = True
        record.best_candidate = best["candidate"]
        record.best_final_score = best["final_score"]
        record.best_candidate_quality = best["candidate_quality"]

        pair = {
            "x": record.raw_prompt,
            "y_w": record.best_candidate,
            "y_l": record.raw_prompt,
        }
        pairs.append(pair)

        if dry_run:
            print(f"\n{'='*70}")
            print(f"[{record.idx + 1}] Raw: {record.raw_prompt}")
            print(f"    Raw quality:       {record.baseline_quality:.4f}")
            print(f"    Candidates scored: {len(scored_candidates)}")
            print(f"    Semantic rejected: {candidates_rejected_semantic}")
            print(
                f"    y_w ({len(record.best_candidate.split())} words): "
                f"{record.best_candidate[:200]}..."
            )
            print(f"    y_w quality:       {record.best_candidate_quality:.4f}")
            print(f"    y_w final_score:   {record.best_final_score:.4f}")
            print(
                f"    Improvement:       "
                f"{record.best_candidate_quality - record.baseline_quality:.4f}"
            )

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
        print(f"Total unique candidates: {total_unique_candidates}")
        acceptance_rate = (
            (len(pairs) / len(records) * 100) if records else 0
        )
        print(f"Acceptance rate: {acceptance_rate:.1f}%")

    logger.info(
        f"Done. Accepted: {len(pairs)}, Skipped: {skipped}, "
        f"Total processed: {len(records)}, "
        f"Total unique candidates: {total_unique_candidates}"
    )
    return len(pairs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build DPO preference pairs")
    parser.add_argument(
        "--strategy",
        choices=["dynamic"],
        default="dynamic",
        help="Rewrite strategy: 'dynamic' (GPU required, dynamic inference using local model)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only 5 prompts and print results without writing",
    )
    parser.add_argument(
        "--input", default=RAW_PROMPTS_PATH, help="Path to raw prompts CSV or JSONL"
    )
    parser.add_argument(
        "--output", default=OUTPUT_PATH, help="Path to write preference pairs JSONL"
    )
    args = parser.parse_args()

    build_preference_pairs(
        raw_prompts_path=args.input,
        output_path=args.output,
        strategy=args.strategy,
        dry_run=args.dry_run,
    )
