"""
Dataset Builder — Preference Pair Generator for DPO Training

SOP Reference: architecture/dpo_training.md §3

Reads raw prompts from dataset/RAW_prompts.csv, filters for instructional tasks,
generates diverse modular rewrite candidates using dynamic section selection from
12 allowed sections (ROLE, TASK, OBJECTIVE, CONTEXT, INPUT, OUTPUT, CONSTRAINTS,
STEPS, REQUIREMENTS, ASSUMPTIONS, EDGE CASES, EXAMPLES), scores them with the
HeuristicScorer, and outputs preference pairs to datasets/preference_pairs.jsonl.

Multi-candidate generation produces 3–5 candidates per raw prompt using varied
rewriting styles (concise structured, constraint-heavy, example-driven, procedural
steps, expert roleplay) with diversity sampling (varied temperatures and nucleus
sampling). The best-scoring candidate that beats the raw prompt baseline is selected
as the preferred rewrite (y_w) for each DPO preference pair.

This script is OFFLINE-ONLY. It must never be imported by runtime code.
"""

import os
import sys
import csv
import json
import logging
import argparse
import re

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


# ── Rewriting Style Definitions ──────────────────────────────────────────

# Base instruction block shared by all rewriting styles.
# Each style appends a style-specific addendum to guide toward a different structure.
_BASE_SYSTEM_INSTRUCTION = (
    "You are an expert prompt engineer. Rewrite the user's prompt into a clearer, "
    "more specific, modular prompt.\n\n"
    "Rules:\n"
    "- Use 3–6 helpful sections chosen dynamically based on the task\n"
    "- Sections are optional and task-dependent — only include what is useful\n"
    "- Avoid rigid templates or repetitive block structures\n"
    "- Improve clarity, specificity, and actionability\n"
    "- Add missing details: clarify ambiguity, define output format, add constraints\n"
    "- Specify level of detail, assumptions, and examples when beneficial\n"
    "- Preserve the original intent and meaning exactly\n"
    "- Keep a natural, professional tone with high readability\n"
    "- Output ONLY the rewritten prompt — no explanations, preamble, or meta-commentary\n\n"
    "Allowed sections (use only what helps):\n"
    "ROLE, TASK, OBJECTIVE, CONTEXT, INPUT, OUTPUT, CONSTRAINTS, "
    "STEPS, REQUIREMENTS, ASSUMPTIONS, EDGE CASES, EXAMPLES\n\n"
)

# Each rewriting style: (name, system_prompt_addendum, temperature, top_p, user_template)
REWRITING_STYLES = [
    {
        "name": "concise_structured",
        "addendum": (
            "Style directive: Write a CONCISE, tightly structured rewrite. "
            "Prioritize brevity — use short labeled sections and bullet points. "
            "Focus on ROLE, TASK, CONSTRAINTS, and OUTPUT sections. "
            "Keep the total rewrite compact (shorter than 250 words). "
            "Every sentence must add value — no filler."
        ),
        "temperature": 0.4,
        "top_p": 0.85,
        "user_template": "Rewrite this prompt into a concise, structured version:\n\n{}",
    },
    {
        "name": "constraint_heavy",
        "addendum": (
            "Style directive: Write a CONSTRAINT-HEAVY rewrite that maximises specificity. "
            "Focus on OBJECTIVE, CONSTRAINTS, REQUIREMENTS, and EDGE CASES sections. "
            "Add explicit boundaries: what to include, what to exclude, formatting rules, "
            "length expectations, quality criteria, and error handling requirements. "
            "Include at least 4–6 concrete constraints."
        ),
        "temperature": 0.5,
        "top_p": 0.90,
        "user_template": "Rewrite this prompt with heavy emphasis on constraints and requirements:\n\n{}",
    },
    {
        "name": "example_driven",
        "addendum": (
            "Style directive: Write an EXAMPLE-DRIVEN rewrite that leads with concrete examples. "
            "Focus on TASK, EXAMPLES, OUTPUT, and CONTEXT sections. "
            "Include at least one input/output example or mock response. "
            "When applicable, show expected JSON structure, code patterns, or text formatting. "
            "Use the examples to implicitly define constraints rather than listing rules."
        ),
        "temperature": 0.5,
        "top_p": 0.90,
        "user_template": "Rewrite this prompt with examples that clarify the expected output:\n\n{}",
    },
    {
        "name": "procedural_steps",
        "addendum": (
            "Style directive: Write a PROCEDURAL, step-by-step rewrite. "
            "Focus on ROLE, STEPS, REQUIREMENTS, and OUTPUT sections. "
            "Break the task into 3–7 numbered steps in logical order. "
            "Each step should be a clear, actionable instruction. "
            "Include deliverable expectations for each step when appropriate."
        ),
        "temperature": 0.4,
        "top_p": 0.85,
        "user_template": "Rewrite this prompt as a step-by-step procedure:\n\n{}",
    },
    {
        "name": "expert_roleplay",
        "addendum": (
            "Style directive: Write an EXPERT ROLEPLAY rewrite with rich contextual framing. "
            "Focus on ROLE, CONTEXT, OBJECTIVE, and ASSUMPTIONS sections. "
            "Define a specific expert persona with relevant domain expertise. "
            "Provide detailed background context and audience assumptions. "
            "Make the prompt feel like a professional brief or consulting request. "
            "Use a balanced length — detailed but not verbose."
        ),
        "temperature": 0.6,
        "top_p": 0.92,
        "user_template": "Rewrite this prompt as an expert-level professional brief:\n\n{}",
    },
]


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

    Generates 3–5 candidate rewrites per raw prompt by iterating over distinct
    rewriting styles, each with a unique system prompt guiding the model toward
    a different structural approach:

      1. concise_structured  — tight, section-heavy, minimal filler
      2. constraint_heavy    — maximises constraints, edge cases, requirements
      3. example_driven      — leads with examples and expected output
      4. procedural_steps    — step-by-step numbered breakdown
      5. expert_roleplay     — expert persona with rich contextual framing

    Dynamic section selection allows the model to choose 3–6 relevant sections
    from 12 allowed options (ROLE, TASK, OBJECTIVE, CONTEXT, INPUT, OUTPUT,
    CONSTRAINTS, STEPS, REQUIREMENTS, ASSUMPTIONS, EDGE CASES, EXAMPLES).

    Diversity is achieved through:
      - Style-specific system prompts with distinct structural guidance
      - Varied temperatures (0.4–0.6) and nucleus sampling (top_p 0.85–0.92)
      - Different user prompt framings per style

    Validation is lightweight (no rigid regex parsing):
      - Reject empty/trivial outputs
      - Reject verbatim copies of the raw prompt
      - Reject outputs shorter than 50% of raw prompt length
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
        Lightweight validation for a generated candidate.

        Rejects:
          - Empty or trivially short outputs
          - Verbatim copies of the raw prompt
          - Outputs shorter than 50% of the raw prompt token count

        Returns True if the candidate passes validation.
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

        # Reject trivially short outputs (less than 20 characters)
        if len(candidate_stripped) < 20:
            return False

        # Reject candidates shorter than 50% of raw prompt length
        # (a good rewrite should generally be equal or longer)
        raw_words = len(raw_stripped.split())
        cand_words = len(candidate_stripped.split())
        if raw_words > 5 and cand_words < raw_words * 0.5:
            return False

        return True

    def generate_candidates(self, raw_prompt: str) -> list[str]:
        """
        Generate 3–5 diverse rewrite candidates for a single raw prompt.

        Iterates over all rewriting styles, calling the local model with
        style-specific system prompts, temperatures, and user framings.
        Failed or invalid generations are logged and skipped.

        Returns:
            List of valid candidate strings (may be empty if all fail).
        """
        candidates = []

        for style in self.styles:
            try:
                result = self.optimizer.rewrite(
                    raw_prompt,
                    sys_prompt_override=style["sys_prompt"],
                    user_prompt_template=style["user_template"],
                    temperature=style["temperature"],
                    top_p=style["top_p"],
                )

                # Validate the generated candidate
                if not self._validate_candidate(result, raw_prompt):
                    logger.debug(
                        f"  [{style['name']}] Rejected: failed validation "
                        f"(empty, copy, or too short)"
                    )
                    continue

                candidates.append(result)
                logger.debug(
                    f"  [{style['name']}] Accepted candidate "
                    f"({len(result.split())} words)"
                )

            except Exception as e:
                logger.warning(
                    f"  [{style['name']}] Generation error: {e}"
                )

        logger.info(
            f"  Generated {len(candidates)}/{len(self.styles)} valid candidates"
        )
        return candidates


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
    strategy: str = "modular",
    dry_run: bool = False,
    dry_run_limit: int = 5,
) -> int:
    """
    Generate preference pairs from raw prompts.

    Args:
        raw_prompts_path: Path to raw prompts CSV or JSONL
        output_path: Path to write preference pairs JSONL
        strategy: "modular" for ModularRewriter, "model" for ModelBasedRewriter
        dry_run: If True, process only dry_run_limit prompts and print results
        dry_run_limit: Number of prompts to process in dry run

    Returns:
        Number of valid preference pairs generated
    """
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

    # BATCH PROCESS 1: Generation
    logger.info("\n--- STEP 1: BATCH GENERATING CANDIDATES ---")
    candidates_map = {}
    for i, raw_prompt in enumerate(instructional):
        logger.info(f"Generating candidates for prompt {i+1}/{len(instructional)} (5 styles)")
        candidates = rewriter.generate_candidates(raw_prompt)
        if not candidates:
            logger.warning(f"[{i}] No candidates generated for: {raw_prompt[:60]}...")
        candidates_map[raw_prompt] = candidates

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
    total_candidates_accepted = 0

    for i, raw_prompt in enumerate(instructional):
        candidates = candidates_map.get(raw_prompt, [])
        total_candidates_generated += len(candidates)
        if not candidates:
            skipped += 1
            continue

        # Score the original raw prompt as a baseline guarantee
        raw_score = scorer.evaluate(raw_prompt)
        baseline_score = raw_score["raw_quality"]

        # Score all candidates against the raw prompt — pick the best
        best_candidate = None
        best_score = -1.0
        candidates_evaluated = 0
        candidates_rejected_semantic = 0

        for candidate in candidates:
            score_result = scorer.evaluate(raw_prompt, candidate)
            candidates_evaluated += 1

            # Hard gate 1: reject if semantic preservation fails
            if score_result["rejected"]:
                candidates_rejected_semantic += 1
                continue

            if score_result["candidate_quality"] > best_score:
                best_score = score_result["candidate_quality"]
                best_candidate = candidate

        # Hard gate 2: The mathematically enforced DPO guarantee
        # y_w MUST score strictly higher than the original raw prompt (y_l)
        if best_candidate is None or best_score <= baseline_score:
            logger.warning(
                f"[{i}] Rejected: No candidate beat baseline score "
                f"(Best: {best_score:.4f} vs Baseline: {baseline_score:.4f}, "
                f"evaluated: {candidates_evaluated}, semantic_rejected: {candidates_rejected_semantic})"
            )
            skipped += 1
            continue

        total_candidates_accepted += 1

        # y_l = original raw prompt from the dataset (per Discovery answer)
        pair = {
            "x": raw_prompt,
            "y_w": best_candidate,
            "y_l": raw_prompt,
        }
        pairs.append(pair)

        if dry_run:
            print(f"\n{'='*70}")
            print(f"[{i+1}] Raw: {raw_prompt}")
            print(f"    Raw quality:  {baseline_score:.4f}")
            print(f"    Candidates evaluated: {candidates_evaluated}")
            print(f"    y_w ({len(best_candidate.split())} words): {best_candidate[:200]}...")
            print(f"    y_w quality:  {best_score:.4f}")
            print(f"    Improvement:  {best_score - baseline_score:.4f}")

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
        print(f"Total candidates generated: {total_candidates_generated}")
        acceptance_rate = (len(pairs) / len(instructional) * 100) if instructional else 0
        print(f"Acceptance rate: {acceptance_rate:.1f}%")

    logger.info(
        f"Done. Accepted: {len(pairs)}, Skipped: {skipped}, "
        f"Total processed: {len(instructional)}, "
        f"Total candidates generated: {total_candidates_generated}"
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
