"""One-off: measure archetype-classifier shift between current and candidate.

Reads dataset/RAW_prompts.csv, runs both the existing detect_archetype and a
candidate (question-aware) classifier over the `prompt` column, and reports:
  - per-archetype distribution under each
  - overall shift rate
  - old -> new confusion matrix
  - a sample of shifted prompts per (old, new) cell

Run: python measure_archetype_drift.py
Decision rule: if shift > 10%, gate the new classifier behind a runtime flag
or regenerate training data before relying on it.
"""
from __future__ import annotations

import csv
import os
import re
import sys
from collections import Counter, defaultdict

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset_builder.prompt_templates import (
    Archetype,
    _KEYWORDS,
    _archetype_index,
    detect_archetype as detect_old,
)


# Candidate classifier ---------------------------------------------------------
# Same shape as detect_archetype, but:
#   1. Adds question-form patterns to ANALYTICAL.
#   2. Routes inputs ending in `?` or starting with an auxiliary verb to
#      ANALYTICAL regardless of length.
#   3. Only routes to CONCISE when an explicit Concise keyword matches.

_QUESTION_FORM_PATTERNS = (
    r"\b(?:do you know|do you understand|are you familiar|"
    r"have you (?:ever|seen|heard|read)|can you explain|"
    r"what is|what are|how does|how do|why does|why do)\b"
)
_AUX_VERB_START = re.compile(
    r"^\s*(?:do|does|did|is|are|was|were|am|"
    r"can|could|would|will|should|shall|may|might|"
    r"have|has|had)\b",
    re.IGNORECASE,
)


def detect_new(raw_prompt: str) -> Archetype:
    """Surgical change: only re-route question-shaped inputs.

    Non-questions follow the original behavior exactly. Questions get a single
    nudge toward ANALYTICAL via question-form keywords + a question-aware
    fallback. Short imperative commands (e.g. "build a chatbot...") remain
    CONCISE under the original length fallback.
    """
    text = (raw_prompt or "").strip()
    if not text:
        return Archetype.CONCISE

    lowered = text.lower()
    is_question = (
        text.endswith("?")
        or bool(_AUX_VERB_START.match(text))
        or bool(re.search(_QUESTION_FORM_PATTERNS, lowered))
    )

    scores = {arch: 0 for arch, _ in _KEYWORDS}
    for archetype, patterns in _KEYWORDS:
        for pat in patterns:
            if re.search(pat, lowered):
                scores[archetype] += 1

    # Add a single ANALYTICAL signal for question-shaped inputs.
    if is_question:
        scores[Archetype.ANALYTICAL] += 1

    if not any(scores.values()):
        # Original length-based fallback, but questions go to ANALYTICAL
        # regardless of length (so "Do you know X?" never collapses to Concise).
        if is_question:
            return Archetype.ANALYTICAL
        return Archetype.CONCISE if len(text.split()) <= 12 else Archetype.ANALYTICAL

    best = max(scores, key=lambda a: (scores[a], -_archetype_index(a)))
    return best


# Driver -----------------------------------------------------------------------

def main() -> int:
    csv_path = os.path.join(PROJECT_ROOT, "dataset", "RAW_prompts.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found", file=sys.stderr)
        return 1

    old_dist: Counter[str] = Counter()
    new_dist: Counter[str] = Counter()
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    shifted_samples: dict[tuple[str, str], list[str]] = defaultdict(list)
    total = 0

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "prompt" not in (reader.fieldnames or []):
            print(f"ERROR: 'prompt' column missing in {csv_path}", file=sys.stderr)
            return 1
        for row in reader:
            p = (row.get("prompt") or "").strip()
            if not p:
                continue
            old = detect_old(p).value
            new = detect_new(p).value
            old_dist[old] += 1
            new_dist[new] += 1
            confusion[(old, new)] += 1
            total += 1
            if old != new and len(shifted_samples[(old, new)]) < 5:
                shifted_samples[(old, new)].append(p[:120])

    if total == 0:
        print("No prompts read.", file=sys.stderr)
        return 1

    shifted = sum(c for (o, n), c in confusion.items() if o != n)
    shift_rate = shifted / total

    archetypes = [a.value for a in Archetype]

    print(f"\n=== Archetype drift report ({total} prompts) ===\n")

    print("Distribution OLD -> NEW:")
    print(f"  {'archetype':<16} {'old':>8} {'new':>8} {'delta':>8}")
    for a in archetypes:
        o, n = old_dist.get(a, 0), new_dist.get(a, 0)
        print(f"  {a:<16} {o:>8} {n:>8} {n - o:>+8}")

    print(f"\nShifted: {shifted}/{total} = {shift_rate:.2%}")
    print(f"Decision: {'SAFE to ship' if shift_rate < 0.10 else 'CONSIDER GATING'} (threshold 10%)\n")

    print("Confusion (rows = old, cols = new). Diagonal omitted.")
    header = "  old\\new        " + " ".join(f"{a[:7]:>8}" for a in archetypes)
    print(header)
    for o in archetypes:
        cells = []
        for n in archetypes:
            if o == n:
                cells.append(f"{'.':>8}")
            else:
                cells.append(f"{confusion.get((o, n), 0):>8}")
        print(f"  {o:<14}  " + " ".join(cells))

    print("\nSample shifted prompts (up to 5 per cell):")
    for (o, n), samples in sorted(shifted_samples.items(), key=lambda kv: -confusion[kv[0]]):
        print(f"\n  [{o} -> {n}] ({confusion[(o, n)]} prompts)")
        for s in samples:
            print(f"    - {s}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
