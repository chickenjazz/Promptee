"""
Normalize prompt↔completion whitespace seam in preference_pairs.jsonl.

Enforces a single convention so DPOTrainer's tokenization sanity check does
not fire (see TRL warning: "Mismatch between tokenized prompt and the start
of tokenized prompt+chosen").

Convention:
  - `x`   (prompt)   MUST end with exactly one "\n"
  - `y_w` (chosen)   MUST NOT start with whitespace
  - `y_l` (rejected) MUST NOT start with whitespace

Usage:
  # Report only (dry run):
  python -m dataset_builder.normalize_seam_whitespace

  # Rewrite the file in place (creates .bak alongside):
  python -m dataset_builder.normalize_seam_whitespace --apply
"""

import argparse
import json
import os
import shutil
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(PROJECT_ROOT, "datasets", "test_preference_pairs.jsonl")

PROMPT_KEY = "x"
COMPLETION_KEYS = ("y_w", "y_l")


def normalize_record(rec: dict) -> tuple[dict, dict]:
    """Return (normalized_record, change_flags)."""
    flags = {"prompt_newline_added": False, "completion_lstripped": {}}

    prompt = rec.get(PROMPT_KEY, "")
    if not isinstance(prompt, str):
        return rec, flags

    new_prompt = prompt.rstrip(" \t\r\n") + "\n"
    if new_prompt != prompt:
        flags["prompt_newline_added"] = True
    rec[PROMPT_KEY] = new_prompt

    for key in COMPLETION_KEYS:
        val = rec.get(key, "")
        if not isinstance(val, str):
            continue
        new_val = val.lstrip()
        if new_val != val:
            flags["completion_lstripped"][key] = True
        rec[key] = new_val

    return rec, flags


def audit(path: str) -> dict:
    """Scan and report violations without modifying the file."""
    stats = {
        "total": 0,
        "prompt_missing_newline": 0,
        "prompt_trailing_space_then_newline": 0,
        "y_w_leading_ws": 0,
        "y_l_leading_ws": 0,
    }
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            rec = json.loads(line)
            prompt = rec.get(PROMPT_KEY, "")
            if isinstance(prompt, str):
                if not prompt.endswith("\n"):
                    stats["prompt_missing_newline"] += 1
                elif len(prompt) >= 2 and prompt[-2] in " \t":
                    stats["prompt_trailing_space_then_newline"] += 1
            for key, stat_key in (("y_w", "y_w_leading_ws"), ("y_l", "y_l_leading_ws")):
                val = rec.get(key, "")
                if isinstance(val, str) and val and val[0].isspace():
                    stats[stat_key] += 1
    return stats


def apply(path: str) -> dict:
    """Rewrite file in place after backing up to <path>.bak."""
    backup = path + ".bak"
    shutil.copy2(path, backup)

    changed = {"records_modified": 0, "prompt_newline_added": 0,
               "y_w_lstripped": 0, "y_l_lstripped": 0, "total": 0}

    tmp = path + ".tmp"
    with open(path, "r", encoding="utf-8") as fin, \
         open(tmp, "w", encoding="utf-8", newline="\n") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            changed["total"] += 1
            rec = json.loads(line)
            new_rec, flags = normalize_record(rec)
            modified = (
                flags["prompt_newline_added"]
                or bool(flags["completion_lstripped"])
            )
            if modified:
                changed["records_modified"] += 1
            if flags["prompt_newline_added"]:
                changed["prompt_newline_added"] += 1
            if flags["completion_lstripped"].get("y_w"):
                changed["y_w_lstripped"] += 1
            if flags["completion_lstripped"].get("y_l"):
                changed["y_l_lstripped"] += 1
            fout.write(json.dumps(new_rec, ensure_ascii=False) + "\n")

    os.replace(tmp, path)
    changed["backup"] = backup
    return changed


def print_stats(label: str, stats: dict) -> None:
    print(f"\n=== {label} ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--path", default=DEFAULT_PATH,
                        help=f"JSONL path (default: {DEFAULT_PATH})")
    parser.add_argument("--apply", action="store_true",
                        help="Rewrite the file in place (creates .bak).")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"ERROR: file not found: {args.path}", file=sys.stderr)
        return 2

    before = audit(args.path)
    print_stats("BEFORE", before)

    if not args.apply:
        violations = (before["prompt_missing_newline"]
                      + before["prompt_trailing_space_then_newline"]
                      + before["y_w_leading_ws"]
                      + before["y_l_leading_ws"])
        print(f"\nDry run. Total violations: {violations}")
        print("Re-run with --apply to fix.")
        return 0

    result = apply(args.path)
    print_stats("APPLIED", result)
    print(f"\nBackup written to: {result['backup']}")

    after = audit(args.path)
    print_stats("AFTER", after)

    remaining = (after["prompt_missing_newline"]
                 + after["prompt_trailing_space_then_newline"]
                 + after["y_w_leading_ws"]
                 + after["y_l_leading_ws"])
    if remaining != 0:
        print(f"\nWARNING: {remaining} violations remain after normalization.",
              file=sys.stderr)
        return 1
    print("\nAll records conform to the seam-whitespace convention.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
