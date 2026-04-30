"""Convert the populated CSV into a DPO preference JSONL.

Reads a CSV that already contains `prompt`, `rewritten_prompt`, and
`losing_prompts` columns (the output of build_rewritten_dataset.py followed
by build_losing_prompts.py) and writes one JSON object per row with the
keys expected by training/dpo_trainer.py:

    {"x": <raw prompt>, "y_w": <chosen rewrite>, "y_l": <losing rewrite>}

Rows missing any of the three fields are skipped and reported.

Usage:
    python -m dataset_builder.build_preference_jsonl \\
        --input dataset/RAW_prompts.csv \\
        --output datasets/preference_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from dataset_builder.config import (
    LOSING_COLUMN,
    PROMPT_COLUMN,
    REWRITE_COLUMN,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "RAW_prompts.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "preference_pairs.jsonl"

logger = logging.getLogger("dataset_builder.build_preference_jsonl")


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dataset_builder.build_preference_jsonl",
        description="Convert a CSV of (prompt, rewritten_prompt, losing_prompts) "
                    "into the JSONL format consumed by training/dpo_trainer.py.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Input CSV path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output JSONL path.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def convert(input_csv: Path, output_jsonl: Path) -> tuple[int, int]:
    """Read CSV, write JSONL. Returns (written, skipped)."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    missing = [c for c in (PROMPT_COLUMN, REWRITE_COLUMN, LOSING_COLUMN) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input CSV is missing required column(s): {missing}. "
            f"Expected: {PROMPT_COLUMN}, {REWRITE_COLUMN}, {LOSING_COLUMN}."
        )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for i, row in df.iterrows():
            x = str(row[PROMPT_COLUMN]).strip()
            y_w = str(row[REWRITE_COLUMN]).strip()
            y_l = str(row[LOSING_COLUMN]).strip()

            if not (x and y_w and y_l):
                skipped += 1
                logger.debug("Row %d skipped (empty field): x=%d y_w=%d y_l=%d",
                             i, len(x), len(y_w), len(y_l))
                continue

            fh.write(json.dumps({"x": x, "y_w": y_w, "y_l": y_l}, ensure_ascii=False))
            fh.write("\n")
            written += 1

    return written, skipped


def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    logger.info("Reading: %s", args.input)
    written, skipped = convert(args.input, args.output)
    logger.info("Wrote %d pairs to %s (%d rows skipped for missing fields).",
                written, args.output, skipped)
    return 0 if written > 0 else 1


if __name__ == "__main__":
    sys.path.append(str(PROJECT_ROOT))
    sys.exit(main())
