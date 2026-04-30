"""CLI entrypoint for generating `losing_prompts` (DPO rejected samples).

Reads a CSV that already has both `prompt` and `rewritten_prompt` populated,
generates a weaker-but-plausible rewrite for each row, and writes it to the
`losing_prompts` column. Resume-safe: non-empty `losing_prompts` cells are
skipped unless --overwrite is passed. Crash-safe: a checkpoint is written every
--save-every rows via atomic rename.

The losing rewrite must be *better than the raw prompt* but *clearly worse than
the chosen rewritten_prompt* — that gap is the DPO preference signal.

Usage:
    python -m dataset_builder.build_losing_prompts \\
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
except ImportError:
    def tqdm(iterable, **_kwargs):  # type: ignore
        return iterable

from dataset_builder.cleaners import clean_output
from dataset_builder.config import (
    DEFAULT_GENERATION,
    DEFAULT_INPUT_CSV,
    DEFAULT_OUTPUT_CSV,
    LOSING_COLUMN,
    LOSING_ERROR_COLUMN,
    PROMPT_COLUMN,
    REWRITE_COLUMN,
    STRICT_GENERATION,
)
from dataset_builder.losing_prompt_template import build_losing_plan
from dataset_builder.model_loader import format_chat_prompt, generate, load_model_and_tokenizer
from dataset_builder.validators import validate

# Fix SSL_CERT_FILE pointing to non-existent path (known env issue)
if 'SSL_CERT_FILE' in os.environ and not os.path.exists(os.environ['SSL_CERT_FILE']):
    del os.environ['SSL_CERT_FILE']

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


logger = logging.getLogger("dataset_builder.build_losing")


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dataset_builder.build_losing_prompts",
        description="Generate weaker rewrites (losing_prompts) for DPO preference training.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_CSV,
                        help="Input CSV (must contain 'prompt' and 'rewritten_prompt' columns).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV,
                        help="Output CSV path. Same as input is allowed.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate even rows that already have a non-empty losing_prompts.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most this many rows after start-row. 0 = no limit.")
    parser.add_argument("--start-row", type=int, default=0,
                        help="0-based row offset to start processing from.")
    parser.add_argument("--save-every", type=int, default=25,
                        help="Checkpoint the output CSV every N successfully processed rows.")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_GENERATION["max_new_tokens"],
                        help="Override generation max_new_tokens.")
    parser.add_argument("--no-quant", action="store_true",
                        help="Disable 4-bit quantization (requires significantly more VRAM).")
    parser.add_argument("--log-level", default="INFO",
                        help="Python logging level (DEBUG, INFO, WARNING, ERROR).")
    parser.add_argument("--max-retries", type=int, default=5,
                        help="Per-row retry count for both exceptions and validation failures. "
                             "Each retry uses a slightly different temperature to escape stuck samples.")
    parser.add_argument("--max-passes", type=int, default=5,
                        help="Outer-loop passes over still-empty rows. The script keeps re-sweeping "
                             "failed rows until every row has a losing_prompts entry or this cap is hit.")
    return parser.parse_args(argv)


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in (PROMPT_COLUMN, REWRITE_COLUMN) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input CSV is missing required column(s): {missing}. "
            f"Both '{PROMPT_COLUMN}' and '{REWRITE_COLUMN}' must be present."
        )
    if LOSING_COLUMN not in df.columns:
        df[LOSING_COLUMN] = ""
    if LOSING_ERROR_COLUMN not in df.columns:
        df[LOSING_ERROR_COLUMN] = ""
    for col in (PROMPT_COLUMN, REWRITE_COLUMN, LOSING_COLUMN, LOSING_ERROR_COLUMN):
        df[col] = df[col].fillna("").astype(str)
    return df


def _atomic_save(df: pd.DataFrame, output: Path) -> None:
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


def _normalize(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _generate_losing(
    model,
    tokenizer,
    raw_prompt: str,
    chosen_rewrite: str,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool = True,
) -> Tuple[str, str]:
    """Single generation attempt with the given sampling settings.

    Returns (losing_text, error_reason). On success, error_reason is "".
    The outer per-row loop is responsible for varying temperature across attempts.
    """
    plan = build_losing_plan(raw_prompt, chosen_rewrite)
    chat_prompt = format_chat_prompt(tokenizer, plan.system_instruction, plan.user_message)

    raw_norm = _normalize(raw_prompt)
    chosen_norm = _normalize(chosen_rewrite)

    raw_out = generate(
        model,
        tokenizer,
        chat_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=DEFAULT_GENERATION["top_p"],
        do_sample=do_sample,
        repetition_penalty=DEFAULT_GENERATION["repetition_penalty"]
        if do_sample
        else STRICT_GENERATION["repetition_penalty"],
    )
    cleaned = clean_output(raw_out)

    result = validate(cleaned, raw_prompt, plan.archetype)
    if not result.ok:
        return "", result.reason or "validation failed"
    cleaned_norm = _normalize(cleaned)
    if cleaned_norm == chosen_norm:
        return "", "identical to chosen rewrite"
    if cleaned_norm == raw_norm:
        return "", "identical to raw prompt"
    return cleaned, ""


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

    def _collect_pending() -> list:
        out = []
        for i in indices:
            raw = df.at[i, PROMPT_COLUMN].strip()
            chosen = df.at[i, REWRITE_COLUMN].strip()
            existing = df.at[i, LOSING_COLUMN].strip()
            if not raw:
                df.at[i, LOSING_ERROR_COLUMN] = "empty prompt"
                continue
            if not chosen:
                df.at[i, LOSING_ERROR_COLUMN] = "empty rewritten_prompt"
                continue
            # On the first pass, --overwrite governs filled rows. On later passes
            # we only revisit rows that are still empty, regardless of --overwrite,
            # so we don't churn rows that already succeeded.
            if existing:
                continue
            out.append(i)
        return out

    # First pass also honors --overwrite: rebuild the initial pending list with that flag.
    pending = []
    for i in indices:
        raw = df.at[i, PROMPT_COLUMN].strip()
        chosen = df.at[i, REWRITE_COLUMN].strip()
        existing = df.at[i, LOSING_COLUMN].strip()
        if not raw:
            df.at[i, LOSING_ERROR_COLUMN] = "empty prompt"
            continue
        if not chosen:
            df.at[i, LOSING_ERROR_COLUMN] = "empty rewritten_prompt"
            continue
        if existing and not args.overwrite:
            continue
        pending.append(i)

    if not pending:
        logger.info("Nothing to do — all selected rows already have losing_prompts. Saving and exiting.")
        _atomic_save(df, args.output)
        return 0

    logger.info("Rows pending losing-rewrite: %d / %d selected", len(pending), len(indices))

    logger.info("Loading Qwen model (this can take a minute)...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(load_in_4bit=not args.no_quant)
    logger.info("Model ready in %.1fs.", time.time() - t0)

    # Temperature ladder used across per-row retries. Slightly different sampling
    # each attempt is what lets a row escape a stuck validation failure.
    base_temp = float(DEFAULT_GENERATION["temperature"])
    temp_ladder = [base_temp, base_temp + 0.2, base_temp + 0.4, base_temp + 0.6]

    def _attempt_row(row_idx: int, attempt: int, total_attempts: int) -> Tuple[str, str]:
        """Single per-row attempt with a temperature/strategy chosen from `attempt`.

        Returns (losing_text, error_reason). Empty losing_text means failure.
        """
        raw_prompt = df.at[row_idx, PROMPT_COLUMN].strip()
        chosen_rewrite = df.at[row_idx, REWRITE_COLUMN].strip()

        # Last attempt of the row falls back to deterministic decoding.
        if attempt == total_attempts:
            do_sample = False
            temperature = base_temp  # ignored when do_sample=False
        else:
            do_sample = True
            temperature = temp_ladder[min(attempt - 1, len(temp_ladder) - 1)]

        try:
            return _generate_losing(
                model, tokenizer, raw_prompt, chosen_rewrite,
                max_new_tokens=args.max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )
        except RuntimeError as exc:
            msg = str(exc)
            if "out of memory" in msg.lower():
                logger.warning("CUDA OOM on row %d (attempt %d/%d); clearing cache.",
                               row_idx, attempt, total_attempts)
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                return "", f"cuda oom: {msg[:200]}"
            logger.exception("Generation runtime error on row %d (attempt %d/%d)",
                             row_idx, attempt, total_attempts)
            return "", f"runtime error: {msg[:200]}"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error on row %d (attempt %d/%d)",
                             row_idx, attempt, total_attempts)
            return "", f"unexpected error: {str(exc)[:200]}"

    processed_since_save = 0
    total_successes = 0

    try:
        for pass_num in range(1, args.max_passes + 1):
            if pass_num > 1:
                pending = _collect_pending()
                if not pending:
                    logger.info("All selected rows now have losing_prompts. Stopping after pass %d.",
                                pass_num - 1)
                    break
                logger.info("Pass %d/%d — %d rows still empty, retrying.",
                            pass_num, args.max_passes, len(pending))

            pass_successes = 0
            pass_failures = 0

            for row_idx in tqdm(pending, desc=f"losing-rewriting (pass {pass_num})", unit="row"):
                losing, err = "", ""

                for attempt in range(1, args.max_retries + 1):
                    losing, err = _attempt_row(row_idx, attempt, args.max_retries)
                    if losing:
                        break
                    logger.debug("row %d pass %d attempt %d/%d failed: %s",
                                 row_idx, pass_num, attempt, args.max_retries, err)

                if losing:
                    df.at[row_idx, LOSING_COLUMN] = losing
                    df.at[row_idx, LOSING_ERROR_COLUMN] = ""
                    pass_successes += 1
                    total_successes += 1
                else:
                    df.at[row_idx, LOSING_ERROR_COLUMN] = err or "generation failed"
                    pass_failures += 1

                processed_since_save += 1

                if processed_since_save >= args.save_every:
                    _atomic_save(df, args.output)
                    logger.info("Checkpoint saved at row %d (pass %d ok=%d, fail=%d).",
                                row_idx, pass_num, pass_successes, pass_failures)
                    processed_since_save = 0

            logger.info("Pass %d complete — ok=%d, fail=%d (still empty).",
                        pass_num, pass_successes, pass_failures)
            _atomic_save(df, args.output)

            if pass_failures == 0:
                logger.info("All rows filled after pass %d. Stopping early.", pass_num)
                break
    finally:
        _atomic_save(df, args.output)
        logger.info("Final save complete: %s", args.output)

    still_empty = sum(
        1 for i in indices
        if not df.at[i, LOSING_COLUMN].strip()
        and df.at[i, PROMPT_COLUMN].strip()
        and df.at[i, REWRITE_COLUMN].strip()
    )
    logger.info("Done. successes=%d still_empty=%d (passes used up to %d)",
                total_successes, still_empty, args.max_passes)
    return 0 if still_empty == 0 else 1


def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
