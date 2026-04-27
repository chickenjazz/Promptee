"""Output validation per spec §Validation Rules.

Returns a structured result so the caller can record the failure reason in the
`rewrite_error` column. Each rule is independently testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from dataset_builder.config import DIAGNOSTIC_LABELS, FILLER_OPENERS
from dataset_builder.prompt_templates import Archetype


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: Optional[str] = None


_ANSWER_SHAPED_PATTERNS = (
    re.compile(r"^\s*```(?:python|javascript|js|typescript|ts|java|c\+\+|sql|bash|shell|go|rust)\b", re.IGNORECASE),
    re.compile(r"^\s*(?:def |class |function |const |let |var |import |from |#include)\b"),
    re.compile(r"^\s*Once upon a time\b", re.IGNORECASE),
    re.compile(r"^\s*(?:Step 1[:.]|First,)\s+", re.IGNORECASE),
)

_INSTRUCTION_LIKE_PATTERNS = (
    re.compile(r"^\s*(?:[A-Z][A-Z /]+:)", re.MULTILINE),  # ROLE:, TASK:, OBJECTIVE:, etc.
    re.compile(
        r"^\s*(?:write|create|generate|produce|design|build|develop|draft|compose|"
        r"explain|describe|summari[sz]e|list|outline|compare|analy[sz]e|evaluate|"
        r"plan|propose|help|guide|coach|advise|teach|act|assume|imagine|pretend|"
        r"give|provide|show|recommend|suggest|review|critique|translate|"
        r"rewrite|refactor|debug|implement|return|output)\b",
        re.IGNORECASE,
    ),
)


def _is_instruction_like(text: str) -> bool:
    return any(p.search(text) for p in _INSTRUCTION_LIKE_PATTERNS)


def _starts_with_filler(text: str) -> bool:
    lowered = text.lstrip().lower()
    return any(lowered.startswith(opener) for opener in FILLER_OPENERS)


def _contains_diagnostic_labels(text: str) -> bool:
    return any(label in text for label in DIAGNOSTIC_LABELS)


def _length_sane(rewritten: str, raw: str, archetype: Archetype) -> bool:
    """Spec §Validation Rules item 7 — bound runaway expansions on simple prompts."""
    long_arch = {Archetype.CODING, Archetype.STRUCTURED, Archetype.ANALYTICAL}
    cap = max(800, 6 * max(len(raw), 50))
    if archetype in long_arch:
        cap = max(cap, 3000)
    return len(rewritten) <= cap


def validate(rewritten: str, raw: str, archetype: Archetype) -> ValidationResult:
    if not rewritten or not rewritten.strip():
        return ValidationResult(False, "empty output")

    if _starts_with_filler(rewritten):
        return ValidationResult(False, "starts with assistant filler")

    if _contains_diagnostic_labels(rewritten):
        return ValidationResult(False, "contains diagnostic label")

    raw_norm = raw.strip()
    rewritten_norm = rewritten.strip()
    if rewritten_norm == raw_norm and len(raw_norm.split()) > 6:
        # Identical to raw is only acceptable for very short prompts where
        # there is genuinely nothing to improve.
        return ValidationResult(False, "identical to raw prompt")

    for pat in _ANSWER_SHAPED_PATTERNS:
        if pat.match(rewritten_norm) and archetype != Archetype.CREATIVE:
            return ValidationResult(False, "output looks like a task answer")

    if not _is_instruction_like(rewritten_norm):
        return ValidationResult(False, "output is not instruction-shaped")

    if not _length_sane(rewritten_norm, raw_norm, archetype):
        return ValidationResult(False, "output length exceeds sanity cap")

    return ValidationResult(True, None)
