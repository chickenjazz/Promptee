"""
Dataset Builder — Preference Pair Generator for DPO Training

SOP Reference: architecture/dpo_training.md §3

Reads raw prompts from dataset/RAW_prompts.csv, filters for instructional tasks,
generates modularized rewrite candidates using the [ROLE][OBJECTIVE][CONTEXT]
[CONSTRAINTS][EXAMPLES] structure, scores them with the HeuristicScorer, and
outputs preference pairs to datasets/preference_pairs.jsonl.

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


# ── Modularized Rewrite Strategy ─────────────────────────────────────────

class RewriteStrategy:
    """
    Base class for rewrite candidate generation.
    """
    def generate_candidates(self, raw_prompt: str) -> list[str]:
        raise NotImplementedError


class DynamicLLMRewriter(RewriteStrategy):
    """
    Uses the local Qwen2.5-7B-Instruct model to dynamically infer the role,
    context, and constraints for the prompt, then formats it into the strict
    [ROLE][OBJECTIVE][CONTEXT][CONSTRAINTS][EXAMPLES] block.
    
    Includes regex-based parsing to reject hallucinations and enforce clean DPO data.
    """

    def __init__(self):
        from tools.prompt_optimizer import PromptOptimizer

        self.optimizer = PromptOptimizer()
        self.optimizer.load_model()
        
        self.sys_prompt = (
            "You are an expert prompt engineering dataset builder. "
            "Your task is to analyze the user's raw prompt and extract exactly 4 fields "
            "to construct a highly modularized, professional prompt.\n\n"
            "Rules:\n"
            "1. You MUST output EXACTLY and ONLY the following fields in this format:\n"
            "ROLE: <expert role inferred from task>\n"
            "OBJECTIVE: <the core task clarified>\n"
            "CONTEXT: <background and audience assumptions>\n"
            "CONSTRAINTS: <specific output rules, formatting, and quality requirements>\n\n"
            "2. DO NOT include markdown, explanations, introductory text, OR the EXAMPLES field (it will be auto-generated).\n"
            "3. The OBJECTIVE must accurately preserve the user's original semantic intent."
        )
        
        self.user_prompt_template = "Analyze and extract fields for this prompt: {}"

    def generate_candidates(self, raw_prompt: str) -> list[str]:
        candidates = []
        prompt_clean = raw_prompt.strip().rstrip(".")

        # Generate a candidate by calling the local model
        try:
            result = self.optimizer.rewrite(
                raw_prompt, 
                sys_prompt_override=self.sys_prompt, 
                user_prompt_template=self.user_prompt_template
            )
            
            if not result or result == raw_prompt:
                logger.warning("Dynamic LLM generation failed or safely fell back to raw.")
                return []

            # Use regex to strictly parse the structured output
            role_match = re.search(r"ROLE:\s*(.*?)(?=\n(?:OBJECTIVE):|\Z)", result, re.DOTALL | re.IGNORECASE)
            obj_match = re.search(r"OBJECTIVE:\s*(.*?)(?=\n(?:CONTEXT):|\Z)", result, re.DOTALL | re.IGNORECASE)
            ctx_match = re.search(r"CONTEXT:\s*(.*?)(?=\n(?:CONSTRAINTS):|\Z)", result, re.DOTALL | re.IGNORECASE)
            con_match = re.search(r"CONSTRAINTS:\s*(.*?)(?=\n(?:ROLE|OBJECTIVE|CONTEXT|CONSTRAINTS):|\Z)", result, re.DOTALL | re.IGNORECASE)

            # If the model hallucinates or drops fields, we reject this candidate (noise reduction)
            if not (role_match and obj_match and ctx_match and con_match):
                logger.warning(f"LLM hallucinated formatting. Rejected output: {result[:100]}...")
                return []

            role = role_match.group(1).strip()
            objective = obj_match.group(1).strip()
            context = ctx_match.group(1).strip()
            constraints = con_match.group(1).strip()
            
            # Format the properties into lists if they aren't already
            if "-" not in constraints:
                constraints = f"- {constraints}"

            # Assemble the structured rewrite cleanly
            clean_rewrite = (
                f"[ROLE]\n"
                f"You are a {role}. Your output is clear, well-structured, "
                f"educational, and follows industry best practices.\n\n"
                f"[OBJECTIVE]\n"
                f"{objective}\n\n"
                f"[CONTEXT]\n"
                f"{context}\n\n"
                f"[CONSTRAINTS]\n"
                f"{constraints}\n"
                f"- Output ONLY the requested content. Do not include meta-commentary, "
                f"introductory text, or concluding remarks.\n\n"
                f"[EXAMPLES]\n"
                f"If the task involves code, follow this pattern:\n"
                f"  step_result = perform_action(input)\n"
                f"If the task is text-based, provide a fully formatted mock response first."
            )
            
            candidates.append(clean_rewrite)

        except Exception as e:
            logger.warning(f"Model generation threw an error: {e}")

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
        logger.info(f"Generating rewrite for prompt {i+1}/{len(instructional)}")
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

    for i, raw_prompt in enumerate(instructional):
        candidates = candidates_map.get(raw_prompt, [])
        if not candidates:
            skipped += 1
            continue

        # Score the original raw prompt as a baseline guarantee
        raw_score = scorer.evaluate(raw_prompt)
        baseline_score = raw_score["total"]

        # Score all candidates against the raw prompt
        best_candidate = None
        best_score = -1.0

        for candidate in candidates:
            score_result = scorer.evaluate(raw_prompt, candidate)

            # Hard gate 1: reject if semantic preservation fails
            if score_result["rejected"]:
                continue

            if score_result["total"] > best_score:
                best_score = score_result["total"]
                best_candidate = candidate

        # Hard gate 2: The mathematically enforced DPO guarantee
        # y_w MUST score strictly higher than the original raw prompt (y_l)
        if best_candidate is None or best_score <= baseline_score:
            logger.warning(
                f"[{i}] Rejected: No candidate beat baseline score "
                f"(Best: {best_score:.4f} vs Baseline: {baseline_score:.4f})"
            )
            skipped += 1
            continue

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
            print(f"    Raw score:  {raw_score}")
            print(f"    y_w: {best_candidate[:200]}...")
            print(f"    y_w score:  total={best_score:.4f}")

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

    logger.info(
        f"Done. Generated: {len(pairs)}, Skipped: {skipped}, "
        f"Total processed: {len(instructional)}"
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
