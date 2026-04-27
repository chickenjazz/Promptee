"""Archetype detection, modularity selection, and the LLM instruction template.

Every public string here either lifts directly from architecture/auto-rewriter.md
or implements a rule the spec defines explicitly. Keep them in sync if the spec
changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Archetype(str, Enum):
    CREATIVE = "Creative"
    CODING = "Coding"
    CONVERSATIONAL = "Conversational"
    STRUCTURED = "Structured"
    ANALYTICAL = "Analytical"
    CONCISE = "Concise"


class Modularity(str, Enum):
    FULL = "Full Modular"
    SEMI = "Semi Modular"
    MINIMAL = "Minimal Modular"
    NATURAL = "Natural Language Modular"


# Order matters: more specific archetypes are checked first so that, for example,
# "explain recursion in 3 bullets" hits Concise before Analytical.
_KEYWORDS = (
    (
        Archetype.CONCISE,
        (
            r"\bin (?:one|1|two|2|three|3|five|5|\d+) (?:sentence|sentences|words|bullets?|lines?)\b",
            r"\b(?:quick|short|brief|simple|one[- ]line|one[- ]sentence|tl;?dr)\b",
            r"\bsummari[sz]e\b",
            r"\bdirect steps\b",
        ),
    ),
    (
        Archetype.CODING,
        (
            r"\b(?:code|coding|debug|debugging|refactor|implement|implementation|function|method|class|module)\b",
            r"\b(?:python|javascript|typescript|java|c\+\+|c#|golang|rust|sql|bash|powershell)\b",
            r"\b(?:react|vue|angular|svelte|next\.?js|node|django|flask|fastapi|spring|express|\.net|laravel)\b",
            r"\b(?:api|endpoint|controller|backend|frontend|database|schema|migration|unit test|stack trace)\b",
            r"\b(?:script|cli|regex|algorithm|data structure)\b",
        ),
    ),
    (
        Archetype.STRUCTURED,
        (
            r"\b(?:plan|checklist|template|roadmap|framework|matrix|lesson plan|manual|survey|flashcards?)\b",
            r"\b(?:outline|curriculum|syllabus|itinerary|agenda|spec|specification)\b",
            r"\b(?:table of contents|step[- ]by[- ]step guide)\b",
        ),
    ),
    (
        Archetype.CREATIVE,
        (
            r"\b(?:story|poem|haiku|sonnet|limerick|verse|song|novel|fiction|essay|caption|tagline|slogan|jingle|lyrics|script|screenplay|monologue|dialogue)\b",
            r"\b(?:character|protagonist|antagonist|world[- ]building|setting|narrative)\b",
            r"\b(?:branding|brand voice|ad copy|tagline|marketing copy|instagram|tiktok|tweet)\b",
            r"\b(?:write (?:a|an) (?:engaging|catchy|fun|witty))\b",
        ),
    ),
    (
        Archetype.CONVERSATIONAL,
        (
            r"\b(?:help me|guide me|coach me|advise me|advice|how (?:do|should|can) i)\b",
            r"\b(?:emotional|relationship|interpersonal|breakup|conflict|anxiety|stress|grief)\b",
            r"\b(?:role[- ]?play|pretend you are|act as my (?:coach|mentor|therapist|friend))\b",
        ),
    ),
    (
        Archetype.ANALYTICAL,
        (
            r"\b(?:compare|contrast|analy[sz]e|evaluate|assess|discuss|explain|reason about|argue|critique)\b",
            r"\b(?:pros and cons|trade[- ]offs?|implications?|differences?|similarit(?:y|ies))\b",
        ),
    ),
)


_DEFAULT_MODULARITY = {
    Archetype.CREATIVE: Modularity.SEMI,
    Archetype.CODING: Modularity.FULL,
    Archetype.CONVERSATIONAL: Modularity.NATURAL,
    Archetype.STRUCTURED: Modularity.FULL,
    Archetype.ANALYTICAL: Modularity.SEMI,
    Archetype.CONCISE: Modularity.MINIMAL,
}


def detect_archetype(raw_prompt: str) -> Archetype:
    """Return the best-fit archetype for `raw_prompt`.

    Pure keyword/regex classifier — fast, deterministic, and good enough for
    routing modularity. The actual rewrite is done by the LLM, which has
    final say on phrasing.
    """
    text = (raw_prompt or "").strip()
    if not text:
        return Archetype.CONCISE

    lowered = text.lower()
    scores = {arch: 0 for arch, _ in _KEYWORDS}
    for archetype, patterns in _KEYWORDS:
        for pat in patterns:
            if re.search(pat, lowered):
                scores[archetype] += 1

    if not any(scores.values()):
        # Heuristic fallback: short prompts default to Concise, long ones to Analytical.
        return Archetype.CONCISE if len(text.split()) <= 12 else Archetype.ANALYTICAL

    # Honor _KEYWORDS ordering when scores tie (Concise > Coding > Structured > ...).
    best = max(scores, key=lambda a: (scores[a], -_archetype_index(a)))
    return best


def _archetype_index(archetype: Archetype) -> int:
    for i, (arch, _) in enumerate(_KEYWORDS):
        if arch == archetype:
            return i
    return len(_KEYWORDS)


def modularity_for(archetype: Archetype) -> Modularity:
    return _DEFAULT_MODULARITY[archetype]


# ----------------------------------------------------------------------------
# Prompt templates (lifted from spec §Required LLM Instruction Template + §Archetype-Specific Rewrite Rules).
# ----------------------------------------------------------------------------

_BASE_SYSTEM = """You are an expert Prompt Rewriter, Prompt Architect, and Prompt Quality Optimizer.

Your task is to rewrite the raw prompt into a clearer, more specific, better structured, and more reliable prompt while preserving the original user intent.

You must first internally detect the prompt archetype and choose the correct modularity style, but you must NOT output the archetype, diagnosis, explanation, or improvement summary.

Supported archetypes:
- Creative
- Coding
- Conversational
- Structured
- Analytical
- Concise

Default modularity rules:
- Creative: Semi Modular
- Coding: Full Modular
- Conversational: Natural Language Modular
- Structured: Full Modular
- Analytical: Semi Modular or Full Modular
- Concise: Minimal Modular

Rewrite rules:
- Improve clarity, specificity, completeness, output instructions, readability, logical flow, and token efficiency.
- Preserve the original intent, topic, constraints, and expected task.
- Do not answer the prompt.
- Do not generate the requested output.
- Do not add irrelevant requirements.
- Do not overcomplicate simple prompts.
- Do not include labels such as Archetype, Weaknesses Found, Rewritten Prompt, or Improvement Summary.
- Return only the final rewritten prompt."""


_ARCHETYPE_GUIDANCE = {
    Archetype.CREATIVE: (
        "This prompt is Creative. Use Semi Modular style: grouped natural language with light structure. "
        "Improve tone, audience, style, mood, and output constraints. Avoid rigid section labels."
    ),
    Archetype.CODING: (
        "This prompt is Coding. Use Full Modular style with explicit labeled sections such as TASK, "
        "LANGUAGE/STACK, INPUTS, OUTPUT, CONSTRAINTS, EDGE CASES. The rewritten prompt must ASK for the code "
        "or explanation. It must NOT contain the actual implementation. Clarify language, framework, expected "
        "files or functions, input/output behavior, validation, and edge cases."
    ),
    Archetype.CONVERSATIONAL: (
        "This prompt is Conversational. Use Natural Language Modular style: a conversational instruction flow "
        "without rigid labels. Improve warmth, empathy, interaction flow, practical guidance, and user-centered "
        "phrasing."
    ),
    Archetype.STRUCTURED: (
        "This prompt is Structured. Use Full Modular style with sections such as OBJECTIVE, SECTIONS, FORMAT, "
        "DETAIL LEVEL, ORDER, CONSTRAINTS. Improve expected structure, section order, completeness, formatting "
        "instructions, and output usability."
    ),
    Archetype.ANALYTICAL: (
        "This prompt is Analytical. Use Semi Modular for simple comparisons or Full Modular for complex analysis. "
        "For complex analysis, sections may include QUESTION, SUBJECT, CRITERIA, ANALYSIS DEPTH, OUTPUT FORMAT, "
        "FINAL RECOMMENDATION. Improve criteria, reasoning depth, conceptual scope, and final synthesis."
    ),
    Archetype.CONCISE: (
        "This prompt is Concise. Use Minimal Modular style: a compact direct prompt with only the most important "
        "constraints. Improve directness, brevity, clear output limit, and simple wording."
    ),
}


@dataclass(frozen=True)
class PromptPlan:
    archetype: Archetype
    modularity: Modularity
    system_instruction: str
    user_message: str


def build_plan(raw_prompt: str) -> PromptPlan:
    """Return the full prompt plan (archetype, modularity, system + user messages)."""
    archetype = detect_archetype(raw_prompt)
    modularity = modularity_for(archetype)
    guidance = _ARCHETYPE_GUIDANCE[archetype]
    system = f"{_BASE_SYSTEM}\n\nArchetype hint for the current row (do not echo this back):\n{guidance}"
    user = (
        "Raw prompt:\n"
        f"{raw_prompt.strip()}\n\n"
        "Final rewritten prompt only:"
    )
    return PromptPlan(archetype, modularity, system, user)
