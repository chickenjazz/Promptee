"""Static configuration: model id, paths, generation params, cleaning sets.

Anything tunable at runtime lives on the CLI (see build_rewritten_dataset.py);
this module holds defaults and constants the spec calls out by name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

MODEL_NAME: str = "Qwen/Qwen2.5-3B-Instruct"

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_CSV: Path = PROJECT_ROOT / "dataset" / "test_dataset.csv"
DEFAULT_OUTPUT_CSV: Path = PROJECT_ROOT / "dataset" / "test_dataset.csv"

PROMPT_COLUMN: str = "prompt"
REWRITE_COLUMN: str = "rewritten_prompt"
LOSING_COLUMN: str = "losing_prompts"
ERROR_COLUMN: str = "rewrite_error"
LOSING_ERROR_COLUMN: str = "losing_error"

DEFAULT_GENERATION: Dict[str, object] = {
    "temperature": 0.3,
    "top_p": 0.9,
    "max_new_tokens": 512,
    "do_sample": True,
    "repetition_penalty": 1.05,
}

# Deterministic retry — used after a sampled generation fails validation.
STRICT_GENERATION: Dict[str, object] = {
    "do_sample": False,
    "max_new_tokens": 512,
    "repetition_penalty": 1.05,
}

# Lowercased; matched against the start of the cleaned output.
FILLER_OPENERS: Tuple[str, ...] = (
    "sure",
    "sure!",
    "sure,",
    "here is",
    "here's",
    "here you go",
    "of course",
    "certainly",
    "absolutely",
    "the answer is",
    "as an ai",
    "i'll",
    "i will",
    "let me",
    "okay",
    "ok,",
)

# Spec §Validation Rules item 4 — these labels must never appear in the cell.
DIAGNOSTIC_LABELS: Tuple[str, ...] = (
    "Archetype:",
    "Weaknesses Found:",
    "Improvement Summary:",
    "Rewritten Prompt:",
    "Diagnosis:",
    "Analysis:",
    "Notes:",
)
