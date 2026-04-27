"""CLI entrypoint for the automatic prompt-rewrite dataset builder.

Reads a CSV with a `prompt` column, rewrites each prompt with Qwen2.5-7B-Instruct,
and writes the improved version to the `rewritten_prompt` column. Resume-safe:
non-empty `rewritten_prompt` cells are skipped unless --overwrite is passed.
Crash-safe: a checkpoint is written every --save-every rows via atomic rename.

Usage:
    python -m dataset_builder.build_rewritten_dataset \\
        --input dataset/test_dataset.csv \\
        --output dataset/test_dataset.csv \\
        --save-every 25
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional but strongly recommended
    def tqdm(iterable, **_kwargs):  # type: ignore
        return iterable

from dataset_builder.cleaners import clean_output
from dataset_builder.config import (
    DEFAULT_GENERATION,
    DEFAULT_INPUT_CSV,
    DEFAULT_OUTPUT_CSV,
    ERROR_COLUMN,
    PROMPT_COLUMN,
    REWRITE_COLUMN,
    STRICT_GENERATION,
)
from dataset_builder.model_loader import format_chat_prompt, generate, load_model_and_tokenizer
from dataset_builder.prompt_templates import build_plan
from dataset_builder.validators import validate

# Fix SSL_CERT_FILE pointing to non-existent path (known env issue)
if 'SSL_CERT_FILE' in os.environ and not os.path.exists(os.environ['SSL_CERT_FILE']):
    del os.environ['SSL_CERT_FILE']

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


logger = logging.getLogger("dataset_builder.build")


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dataset_builder.build_rewritten_dataset",
        description="Rewrite prompts in a CSV using Qwen/Qwen2.5-7B-Instruct.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_CSV,
                        help="Input CSV path (must contain a 'prompt' column).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV,
                        help="Output CSV path. Same as input is allowed.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate even rows that already have a non-empty rewritten_prompt.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most this many rows after start-row. 0 = no limit.")
    parser.add_argument("--start-row", type=int, default=0,
                        help="0-based row offset to start processing from.")
    parser.add_argument("--save-every", type=int, default=25,
                        help="Checkpoint the output CSV every N successfully processed rows.")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_GENERATION["max_new_tokens"],
                        help="Override generation max_new_tokens (useful for failure-injection tests).")
    parser.add_argument("--no-quant", action="store_true",
                        help="Disable 4-bit quantization (requires significantly more VRAM).")
    parser.add_argument("--log-level", default="INFO",
                        help="Python logging level (DEBUG, INFO, WARNING, ERROR).")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Number of times to retry rewriting if an error occurs.")
    return parser.parse_args(argv)


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    if PROMPT_COLUMN not in df.columns:
        raise ValueError(f"Input CSV is missing the required '{PROMPT_COLUMN}' column.")
    if REWRITE_COLUMN not in df.columns:
        df[REWRITE_COLUMN] = ""
    if ERROR_COLUMN not in df.columns:
        df[ERROR_COLUMN] = ""
    # Coerce to strings so downstream comparisons are predictable.
    for col in (PROMPT_COLUMN, REWRITE_COLUMN, ERROR_COLUMN):
        df[col] = df[col].fillna("").astype(str)
    return df


def _atomic_save(df: pd.DataFrame, output: Path) -> None:
    """Write CSV via tmp file + os.replace so a crash mid-write can't corrupt the output."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=output.name, suffix=".tmp", dir=str(output.parent))
    os.close(fd)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, output)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _select_row_indices(df: pd.DataFrame, start: int, limit: int) -> range:
    end = len(df) if limit <= 0 else min(len(df), start + limit)
    start = max(0, min(start, len(df)))
    return range(start, end)


def _rewrite_one(model, tokenizer, raw_prompt: str, max_new_tokens: int) -> Tuple[str, str, str]:
    """Run the two-attempt rewrite pipeline.

    Returns (rewritten, error_reason, archetype). On success, error_reason is "".
    On failure, rewritten is "" and error_reason describes why.
    """
    plan = build_plan(raw_prompt)
    chat_prompt = format_chat_prompt(tokenizer, plan.system_instruction, plan.user_message)

    # Attempt 1 — sampled, spec defaults.
    raw_out = generate(
        model,
        tokenizer,
        chat_prompt,
        max_new_tokens=max_new_tokens,
        temperature=DEFAULT_GENERATION["temperature"],
        top_p=DEFAULT_GENERATION["top_p"],
        do_sample=True,
        repetition_penalty=DEFAULT_GENERATION["repetition_penalty"],
    )
    cleaned = clean_output(raw_out)
    result = validate(cleaned, raw_prompt, plan.archetype)
    if result.ok:
        return cleaned, "", plan.archetype.value

    first_reason = result.reason or "unknown validation failure"
    logger.debug("Attempt 1 failed (%s); retrying deterministically.", first_reason)

    # Attempt 2 — deterministic, no sampling. Spec §Model Requirement allows this for stricter consistency.
    raw_out = generate(
        model,
        tokenizer,
        chat_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=STRICT_GENERATION["repetition_penalty"],
    )
    cleaned = clean_output(raw_out)
    result = validate(cleaned, raw_prompt, plan.archetype)
    if result.ok:
        return cleaned, "", plan.archetype.value

    return "", f"validation failed: {first_reason}; deterministic retry: {result.reason}", plan.archetype.value


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    if not args.input.exists():
        logger.error("Input CSV not found: %s", args.input)
        return 2

    logger.info("Loading dataset: %s", args.input)
    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    df = _ensure_columns(df)
    logger.info("Loaded %d rows. Columns: %s", len(df), list(df.columns))

    indices = _select_row_indices(df, args.start_row, args.limit)
    if not indices:
        logger.warning("No rows selected (start_row=%d, limit=%d, total=%d).",
                       args.start_row, args.limit, len(df))
        _atomic_save(df, args.output)
        return 0

    # Decide which rows actually need work, before paying the model load cost.
    pending = []
    for i in indices:
        raw = df.at[i, PROMPT_COLUMN].strip()
        existing = df.at[i, REWRITE_COLUMN].strip()
        if not raw:
            df.at[i, ERROR_COLUMN] = "empty prompt"
            continue
        if existing and not args.overwrite:
            continue
        pending.append(i)

    if not pending:
        logger.info("Nothing to do — all selected rows already have rewritten_prompt. Saving and exiting.")
        _atomic_save(df, args.output)
        return 0

    logger.info("Rows pending rewrite: %d / %d selected", len(pending), len(indices))

    logger.info("Loading Qwen model (this can take a minute)...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(load_in_4bit=not args.no_quant)
    logger.info("Model ready in %.1fs.", time.time() - t0)

    processed_since_save = 0
    successes = 0
    failures = 0

    try:
        for row_idx in tqdm(pending, desc="rewriting", unit="row"):
            raw_prompt = df.at[row_idx, PROMPT_COLUMN].strip()
            success = False
            rewritten, err, archetype = "", "", ""

            for attempt in range(1, args.max_retries + 1):
                try:
                    rewritten, err, archetype = _rewrite_one(
                        model, tokenizer, raw_prompt, max_new_tokens=args.max_new_tokens
                    )
                    success = True
                    break
                except RuntimeError as exc:
                    msg = str(exc)
                    if "out of memory" in msg.lower():
                        logger.warning("CUDA OOM on row %d (attempt %d/%d); clearing cache.", row_idx, attempt, args.max_retries)
                        try:
                            import torch
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        err = f"cuda oom: {msg[:200]}"
                    else:
                        logger.exception("Generation runtime error on row %d (attempt %d/%d)", row_idx, attempt, args.max_retries)
                        err = f"runtime error: {msg[:200]}"
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Unexpected error on row %d (attempt %d/%d)", row_idx, attempt, args.max_retries)
                    err = f"unexpected error: {str(exc)[:200]}"
            
            if success:
                if rewritten:
                    df.at[row_idx, REWRITE_COLUMN] = rewritten
                    df.at[row_idx, ERROR_COLUMN] = ""
                    successes += 1
                    logger.debug("row %d ok (archetype=%s, len=%d)", row_idx, archetype, len(rewritten))
                else:
                    df.at[row_idx, ERROR_COLUMN] = err
                    failures += 1
                    logger.debug("row %d failed: %s", row_idx, err)
            else:
                df.at[row_idx, ERROR_COLUMN] = err
                failures += 1

            processed_since_save += 1

            if processed_since_save >= args.save_every:
                _atomic_save(df, args.output)
                logger.info("Checkpoint saved at row %d (ok=%d, fail=%d).",
                            row_idx, successes, failures)
                processed_since_save = 0
    finally:
        _atomic_save(df, args.output)
        logger.info("Final save complete: %s", args.output)

    logger.info("Done. ok=%d fail=%d total_pending=%d", successes, failures, len(pending))
    return 0 if failures == 0 else 1


def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
