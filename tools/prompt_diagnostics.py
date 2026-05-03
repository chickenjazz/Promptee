"""Deterministic prompt-quality diagnostics.

Returns frontend-ready issue spans for ambiguity, weak action phrases, redundancy,
missing prompt components, answer-generation risk, and meta-prompt drift risk.
The output drives both inline highlighting and the issue panel.

Intentionally not LLM-backed: keeping diagnostics deterministic makes the system
explainable and reproducible, and prevents the model from co-mingling diagnostic
output with the rewrite.
"""

from __future__ import annotations

import re
from typing import List, Literal, Optional, TypedDict

from rules.prompt_quality_rules import (
    AMBIGUOUS_TOKENS,
    CONTEXT_TERMS,
    OUTPUT_FORMAT_TERMS,
    QUESTION_STARTERS,
    WEAK_PHRASES,
)


IssueType = Literal[
    "ambiguity",
    "weak_action",
    "redundancy",
    "too_short",
    "missing_output_format",
    "missing_context",
    "missing_constraints",
    "meta_prompt_drift",
    "answering_risk",
]


Severity = Literal["low", "medium", "high"]


class PromptIssue(TypedDict):
    id: str
    type: IssueType
    severity: Severity
    span: Optional[str]
    start: Optional[int]
    end: Optional[int]
    message: str
    suggestion: str


def find_prompt_issues(prompt: str) -> List[PromptIssue]:
    """Return frontend-ready prompt issue metadata for highlighting and recommendations."""
    text = prompt or ""
    issues: List[PromptIssue] = []

    issues.extend(_find_ambiguity(text))
    issues.extend(_find_weak_actions(text))
    issues.extend(_find_redundancy(text))
    issues.extend(_find_too_short(text))
    issues.extend(_find_missing_output_format(text))
    issues.extend(_find_missing_context(text))
    issues.extend(_find_answering_risk(text))
    issues.extend(_find_meta_prompt_drift_risk(text))

    return issues


def _find_ambiguity(text: str) -> List[PromptIssue]:
    issues: List[PromptIssue] = []

    for match in re.finditer(r"\b\w+\b", text):
        token = match.group(0)
        if token.lower() in AMBIGUOUS_TOKENS:
            issues.append({
                "id": f"ambiguity-{match.start()}",
                "type": "ambiguity",
                "severity": "medium",
                "span": token,
                "start": match.start(),
                "end": match.end(),
                "message": f"'{token}' is vague and may weaken prompt specificity.",
                "suggestion": "Replace it with a concrete topic, object, requirement, or constraint.",
            })

    return issues


def _find_weak_actions(text: str) -> List[PromptIssue]:
    issues: List[PromptIssue] = []
    lowered = text.lower()

    for phrase, suggestion in WEAK_PHRASES.items():
        for match in re.finditer(re.escape(phrase), lowered):
            issues.append({
                "id": f"weak-action-{match.start()}",
                "type": "weak_action",
                "severity": "medium",
                "span": text[match.start():match.end()],
                "start": match.start(),
                "end": match.end(),
                "message": "This phrase does not clearly define the action expected from the AI.",
                "suggestion": suggestion,
            })

    return issues


def _find_redundancy(text: str) -> List[PromptIssue]:
    issues: List[PromptIssue] = []

    for match in re.finditer(r"\b(\w+)\s+\1\b", text, flags=re.IGNORECASE):
        issues.append({
            "id": f"redundancy-{match.start()}",
            "type": "redundancy",
            "severity": "low",
            "span": match.group(0),
            "start": match.start(),
            "end": match.end(),
            "message": "This repeated word may reduce prompt clarity.",
            "suggestion": "Remove the duplicate word.",
        })

    return issues


def _find_too_short(text: str) -> List[PromptIssue]:
    stripped = text.strip()
    if not stripped:
        return []

    if len(stripped.split()) < 5:
        return [{
            "id": "too-short",
            "type": "too_short",
            "severity": "medium",
            "span": stripped,
            "start": 0,
            "end": len(text),
            "message": "The prompt may be too short to provide enough context.",
            "suggestion": "Add audience level, scope, examples, or expected depth.",
        }]

    return []


def _find_missing_output_format(text: str) -> List[PromptIssue]:
    lowered = text.lower()

    if not any(term in lowered for term in OUTPUT_FORMAT_TERMS):
        return [{
            "id": "missing-output-format",
            "type": "missing_output_format",
            "severity": "medium",
            "span": None,
            "start": None,
            "end": None,
            "message": "The prompt does not specify the desired output format.",
            "suggestion": "Add how the answer should be presented, such as bullets, table, JSON, or paragraph.",
        }]

    return []


def _find_missing_context(text: str) -> List[PromptIssue]:
    lowered = text.lower()

    if len(text.split()) >= 4 and not any(term in lowered for term in CONTEXT_TERMS):
        return [{
            "id": "missing-context",
            "type": "missing_context",
            "severity": "medium",
            "span": None,
            "start": None,
            "end": None,
            "message": "The prompt does not specify the audience, learner level, purpose, or context.",
            "suggestion": "Add context such as beginner level, Grade 7 student, professional audience, or intended purpose.",
        }]

    return []


def _find_answering_risk(text: str) -> List[PromptIssue]:
    lowered = text.lower().strip()

    if any(lowered.startswith(starter) for starter in QUESTION_STARTERS):
        return [{
            "id": "answering-risk",
            "type": "answering_risk",
            "severity": "low",
            "span": None,
            "start": None,
            "end": None,
            "message": "This is a direct question, so the rewriter must avoid answering it.",
            "suggestion": "Rewrite the question into a clearer prompt while preserving the original intent.",
        }]

    return []


def _find_meta_prompt_drift_risk(text: str) -> List[PromptIssue]:
    issues: List[PromptIssue] = []
    lowered = text.lower()
    patterns = [
        r"\bcreate a prompt\b",
        r"\bwrite a prompt\b",
        r"\bgenerate a prompt\b",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, lowered):
            issues.append({
                "id": f"meta-prompt-drift-{match.start()}",
                "type": "meta_prompt_drift",
                "severity": "medium",
                "span": text[match.start():match.end()],
                "start": match.start(),
                "end": match.end(),
                "message": "This wording may cause the rewriter to produce a prompt about a prompt.",
                "suggestion": "Only keep this phrasing if the user explicitly wants to create a prompt.",
            })

    return issues
