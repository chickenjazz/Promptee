"""Deterministic post-generation guard for the prompt rewriter.

The rewriter is a black box: it sometimes answers the prompt instead of rewriting
it, drifts into "create a prompt for this task", or emits a code block when the
raw prompt did not ask for code. This module catches those failure modes with
plain regex so the optimizer can fall back to the raw prompt instead of shipping
a bad rewrite to the user.
"""

from __future__ import annotations

import re
from typing import List, Literal, TypedDict


ValidationStatus = Literal["valid", "invalid"]


class ValidationIssue(TypedDict):
    type: str
    severity: Literal["low", "medium", "high"]
    message: str


class ValidationResult(TypedDict):
    status: ValidationStatus
    issues: List[ValidationIssue]


META_PROMPT_PATTERNS = [
    r"\bcreate a prompt (for|that|to)\b",
    r"\bwrite a prompt (for|that|to)\b",
    r"\bgenerate a prompt (for|that|to)\b",
    r"\bthis prompt asks\b",
    r"\bthe prompt should ask\b",
]


ANSWER_PATTERNS = [
    r"^the answer is\b",
    r"^here'?s (the )?answer\b",
    r"^here is (the )?(code|solution|explanation)\b",
    r"^sure[,! ]+",
    r"^certainly[,! ]+",
    r"\bin conclusion\b",
]


CODE_FENCE_PATTERN = r"```"


CODING_REQUEST_PATTERNS = [
    r"\bcode\b",
    r"\bimplement\b",
    r"\bdebug\b",
    r"\brefactor\b",
    r"\bfunction\b",
    r"\bclass\b",
    r"\bpython\b",
    r"\bjavascript\b",
    r"\btypescript\b",
    r"\bjava\b",
    r"\bsql\b",
]


def _matches_any(patterns: List[str], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def validate_rewrite(raw_prompt: str, optimized_prompt: str) -> ValidationResult:
    """Validate that `optimized_prompt` is a rewrite, not an answer or meta-prompt."""
    issues: List[ValidationIssue] = []

    raw = (raw_prompt or "").strip()
    output = (optimized_prompt or "").strip()
    lowered = output.lower()

    if not output:
        issues.append({
            "type": "empty_output",
            "severity": "high",
            "message": "The model returned an empty rewrite.",
        })

    if _matches_any(META_PROMPT_PATTERNS, lowered):
        issues.append({
            "type": "meta_prompt_drift",
            "severity": "high",
            "message": "The rewrite appears to create a prompt about the task instead of directly rewriting the original task.",
        })

    if _matches_any(ANSWER_PATTERNS, lowered):
        issues.append({
            "type": "answer_instead_of_rewrite",
            "severity": "high",
            "message": "The output appears to answer the prompt instead of rewriting it.",
        })

    raw_is_coding_related = _matches_any(CODING_REQUEST_PATTERNS, raw.lower())
    output_contains_code_fence = bool(re.search(CODE_FENCE_PATTERN, output))

    if output_contains_code_fence and not raw_is_coding_related:
        issues.append({
            "type": "unexpected_code_output",
            "severity": "high",
            "message": "The output contains code formatting even though the raw prompt was not clearly a coding request.",
        })

    high_severity_exists = any(issue["severity"] == "high" for issue in issues)

    return {
        "status": "invalid" if high_severity_exists else "valid",
        "issues": issues,
    }
