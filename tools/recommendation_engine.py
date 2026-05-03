"""Turn diagnostic issues + heuristic scores into educational guidance.

This is the tutor layer. It explains *why* a prompt was weak and produces an
institutional guideline aligned with the research question on responsible AI use
in educational settings. Lives outside the model so the rewriter is never asked
to generate diagnostics or recommendations alongside the rewrite itself.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set


def build_recommendations(
    raw_prompt: str,
    issues: List[Dict[str, Any]],
    score_result: Optional[Dict[str, Any]] = None,
    archetype: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert prompt issues and scores into educational recommendations."""
    recommendations: List[str] = []
    issue_types = {issue.get("type") for issue in issues}

    if "ambiguity" in issue_types:
        recommendations.append(
            "Replace vague words with specific topics, objects, requirements, or constraints."
        )

    if "weak_action" in issue_types:
        recommendations.append(
            "Use stronger action verbs such as explain, compare, analyze, summarize, evaluate, or generate."
        )

    if "redundancy" in issue_types:
        recommendations.append(
            "Remove repeated words or phrases to improve clarity and readability."
        )

    if "missing_output_format" in issue_types:
        recommendations.append(
            "Specify the desired output format, such as bullets, table, JSON, short paragraph, or step-by-step list."
        )

    if "missing_context" in issue_types:
        recommendations.append(
            "Add learner level, purpose, role, or context to make the AI response more appropriate."
        )

    if "too_short" in issue_types:
        recommendations.append(
            "Add enough detail for the AI to understand the expected scope, depth, and audience."
        )

    if "answering_risk" in issue_types:
        recommendations.append(
            "For question-based prompts, rewrite the question into a clearer instruction instead of answering it directly."
        )

    if _is_simple_prompt(raw_prompt):
        recommendations.append(
            "This appears to be a simple learning prompt. A concise prompt may be enough in normal classroom use, even if the optimizer returns a structured rewrite."
        )

    if not recommendations:
        recommendations.append(
            "The prompt is generally understandable. Minor improvements may focus on audience, format, or expected depth."
        )

    guideline = _build_institutional_guideline(raw_prompt, archetype, issue_types)

    return {
        "recommendations": recommendations,
        "institutional_guideline": guideline,
    }


def _is_simple_prompt(raw_prompt: str) -> bool:
    text = (raw_prompt or "").strip().lower()
    return len(text.split()) <= 12 or text.startswith((
        "what is",
        "what are",
        "who is",
        "why is",
        "how does",
        "do you know",
    ))


def _build_institutional_guideline(
    raw_prompt: str,
    archetype: Optional[str],
    issue_types: Set[Optional[str]],
) -> str:
    if _is_simple_prompt(raw_prompt):
        return (
            "For simple factual or educational prompts, students should prioritize clarity, audience level, "
            "and expected depth. Rigid prompt templates are optional and should not be required for every AI interaction."
        )

    if archetype in {"Coding", "Structured"}:
        return (
            "For complex academic, technical, or assessment-related prompts, students should use structured prompts "
            "with explicit task, output format, constraints, and evaluation criteria."
        )

    if "missing_context" in issue_types or "missing_output_format" in issue_types:
        return (
            "In educational AI use, prompts should provide sufficient context and output expectations so AI-generated "
            "responses remain useful, transparent, and aligned with learning goals."
        )

    return (
        "Students should match prompt structure to task complexity, using concise prompts for simple questions and "
        "structured prompts for complex or high-stakes academic tasks."
    )
