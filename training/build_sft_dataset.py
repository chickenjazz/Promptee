"""
SFT Dataset Builder — Convert DPO preference pairs into ChatML SFT records
with system-prompt dropout.

Reads datasets/preference_pairs.jsonl ({x, y_w, y_l}), drops y_l, filters
to y_w examples that contain both ROLE: and TASK: headers, and assigns one
of three system-prompt variants per row:

    r < 0.30 : full strong meta-prompt
    r < 0.70 : weak generic prompt ("Rewrite this prompt to make it better.")
    r >= 0.70: no system message at all

This decouples output quality from system-prompt presence so the model
internalises the rewrite shape rather than depending on a specific prompt
at inference time.

Output: datasets/sft_dataset.jsonl with one ChatML record per line:
    {"messages": [{"role": "system", "content": "..."}, ...]}

OFFLINE-ONLY. Not imported by runtime code.
"""

import os
import sys
import json
import random
import logging
import argparse
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from training._prompts import STRONG_PROMPT, WEAK_PROMPT, USER_TEMPLATE

logger = logging.getLogger("promptee.build_sft_dataset")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

DEFAULT_INPUT = os.path.join(PROJECT_ROOT, "datasets", "preference_pairs.jsonl")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "datasets", "sft_dataset.jsonl")
DEFAULT_SEED = 42

# Dropout probability boundaries
P_STRONG = 0.30
P_WEAK = 0.70  # cumulative — anything in [P_STRONG, P_WEAK) gets weak

_ROLE_RE = re.compile(r"\bROLE\s*:", re.IGNORECASE)
_TASK_RE = re.compile(r"\bTASK\s*:", re.IGNORECASE)
_PERSONA_RE = re.compile(r"\bYou are an?\b")


def _passes_structure_filter(y_w: str) -> bool:
    """Keep rewrites that show at least one structural signal: a ROLE: header,
    a TASK: header, or a 'You are a/an ...' persona opener. Any of the three
    is enough — the SFT target only needs to demonstrate *some* deliberate
    structure relative to the raw prompt."""
    return (
        bool(_ROLE_RE.search(y_w))
        or bool(_TASK_RE.search(y_w))
        or bool(_PERSONA_RE.search(y_w))
    )


def _build_record(x: str, y_w: str, rng: random.Random) -> dict:
    r = rng.random()
    messages = []
    if r < P_STRONG:
        messages.append({"role": "system", "content": STRONG_PROMPT})
        bucket = "strong"
    elif r < P_WEAK:
        messages.append({"role": "system", "content": WEAK_PROMPT})
        bucket = "weak"
    else:
        bucket = "none"

    messages.append({"role": "user", "content": USER_TEMPLATE.format(raw_prompt=x)})
    messages.append({"role": "assistant", "content": y_w})
    return {"messages": messages}, bucket


def build(
    input_path: str = DEFAULT_INPUT,
    output_path: str = DEFAULT_OUTPUT,
    seed: int = DEFAULT_SEED,
) -> dict:
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Preference dataset not found: {input_path}. "
            f"Run training/dataset_builder.py first."
        )

    rng = random.Random(seed)

    total = 0
    kept = 0
    counts = {"strong": 0, "weak": 0, "none": 0}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as src, \
         open(output_path, "w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSON at line {total}: {e}")
                continue

            x = row.get("x")
            y_w = row.get("y_w")
            if not isinstance(x, str) or not isinstance(y_w, str):
                continue

            if not _passes_structure_filter(y_w):
                continue

            record, bucket = _build_record(x, y_w, rng)
            counts[bucket] += 1
            kept += 1
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Read {total} pairs; kept {kept} after structure filter.")
    if kept:
        logger.info(
            "Dropout mix — strong: %d (%.1f%%), weak: %d (%.1f%%), none: %d (%.1f%%)",
            counts["strong"], 100 * counts["strong"] / kept,
            counts["weak"],   100 * counts["weak"]   / kept,
            counts["none"],   100 * counts["none"]   / kept,
        )
    logger.info(f"Wrote SFT dataset to: {output_path}")

    if kept < 300:
        logger.warning(
            "Only %d examples passed the ROLE:+TASK: filter. "
            "Consider relaxing the filter or regenerating preference pairs "
            "before SFT to avoid overfitting.", kept
        )

    return {"total": total, "kept": kept, **counts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SFT dataset from DPO preference pairs.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    build(input_path=args.input, output_path=args.output, seed=args.seed)
