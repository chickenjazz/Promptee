"""Diagnostic-only audit of datasets/preference_pairs.jsonl.

Reports the actual length and structure-marker distribution of `y_w` (chosen)
and `y_l` (rejected) so we can decide whether the runtime adapter expects
structured output, concise output, or some mix. NOT used to gate runtime
behavior — it only informs future decisions about retraining for adaptive
generation styles.

Schema note: this dataset uses DPO-style keys `x` (raw), `y_w` (chosen), `y_l`
(rejected) — not `chosen` / `rejected` as in some other DPO toolchains.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median


DATASET_PATH = Path(__file__).resolve().parent.parent / "datasets" / "preference_pairs.jsonl"

STRUCTURE_MARKERS = [
    "ROLE:",
    "TASK:",
    "INPUTS:",
    "OUTPUTS:",
    "CONSTRAINTS:",
    "REQUIREMENTS:",
]


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pct(count: int, total: int) -> float:
    return round((count / total) * 100, 2) if total else 0.0


def main() -> None:
    rows = load_rows(DATASET_PATH)
    total = len(rows)
    chosen_values = [row.get("y_w", "") for row in rows]
    rejected_values = [row.get("y_l", "") for row in rows]

    chosen_lengths = [len(text.split()) for text in chosen_values]
    rejected_lengths = [len(text.split()) for text in rejected_values]

    print(f"Dataset: {DATASET_PATH}")
    print(f"Total rows: {total}")
    print()
    print("CHOSEN (y_w) length stats")
    print(f"  Average: {mean(chosen_lengths):.2f} words")
    print(f"  Median:  {median(chosen_lengths):.2f} words")
    print(f"  Under 40 words: {pct(sum(n < 40 for n in chosen_lengths), total)}%")
    print(f"  Under 80 words: {pct(sum(n < 80 for n in chosen_lengths), total)}%")
    print()
    print("CHOSEN (y_w) structure markers")
    for marker in STRUCTURE_MARKERS:
        count = sum(marker in text for text in chosen_values)
        print(f"  Contains '{marker}': {pct(count, total)}%")

    all_structured = sum(any(marker in text for marker in STRUCTURE_MARKERS) for text in chosen_values)
    print(f"  Contains any structure marker: {pct(all_structured, total)}%")
    print()
    print("REJECTED (y_l) length stats")
    print(f"  Average: {mean(rejected_lengths):.2f} words")
    print(f"  Median:  {median(rejected_lengths):.2f} words")
    print()
    print("Interpretation note: do NOT use these numbers to gate runtime behavior.")
    print("They inform the dataset-expansion decision before any future retraining")
    print("for adaptive concise generation.")


if __name__ == "__main__":
    main()
