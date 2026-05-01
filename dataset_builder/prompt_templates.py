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
            # Question-form patterns: catches "Do you know X?", "What is X?",
            # etc. so analysis-style questions don't collapse to Concise.
            r"\b(?:do you know|do you understand|are you familiar|"
            r"have you (?:ever|seen|heard|read)|can you explain|"
            r"what is|what are|how does|how do|why does|why do)\b",
        ),
    ),
)


# Pattern matched against the raw prompt's first token to detect inputs that
# start with an auxiliary verb (yes/no question form). Used by detect_archetype's
# fallback so questions never default to CONCISE on length alone.
_AUX_VERB_START = re.compile(
    r"^\s*(?:do|does|did|is|are|was|were|am|"
    r"can|could|would|will|should|shall|may|might|"
    r"have|has|had)\b",
    re.IGNORECASE,
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
        # Heuristic fallback: short prompts default to Concise, long ones to
        # Analytical — but questions never default to Concise (otherwise
        # "Do you know X?" collapses to a one-line Minimal Modular rewrite).
        is_question = text.endswith("?") or bool(_AUX_VERB_START.match(text))
        if is_question:
            return Archetype.ANALYTICAL
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


# Per-archetype guidance lifted from architecture/auto-rewriter.md §Archetype-Specific Rewrite Rules.
# Injected into the system prompt as a hint so the model emphasizes the right axes for each archetype.
_ARCHETYPE_GUIDANCE = {
    Archetype.CREATIVE: (
        "This is a Creative prompt. Use Semi Modular rewriting.\n"
        "Improve: tone, audience, style, mood, originality, output constraints.\n"
        "Avoid overly rigid section labels unless the original prompt is complex."
    ),
    Archetype.CODING: (
        "This is a Coding prompt. Use Full Modular rewriting with sections like:\n"
        "TASK / LANGUAGE/STACK / INPUTS / OUTPUT / CONSTRAINTS / EDGE CASES.\n"
        "Improve: language clarity, framework/stack details, expected files or functions, "
        "input/output behavior, validation, edge cases.\n"
        "Ask for the code or explanation — do not write the implementation yourself."
    ),
    Archetype.CONVERSATIONAL: (
        "This is a Conversational prompt. Use Natural Language Modular rewriting.\n"
        "Improve: warmth, empathy, interaction flow, practical guidance, user-centered phrasing.\n"
        "Output one or two flowing paragraphs — no labeled sections."
    ),
    Archetype.STRUCTURED: (
        "This is a Structured prompt. Use Full Modular rewriting with sections like:\n"
        "OBJECTIVE / SECTIONS / FORMAT / DETAIL LEVEL / ORDER / CONSTRAINTS.\n"
        "Improve: expected structure, section order, completeness, formatting instructions, "
        "output usability."
    ),
    Archetype.ANALYTICAL: (
        "This is an Analytical prompt. Use Semi Modular or Full Modular rewriting depending on complexity.\n"
        "For complex prompts use sections like: QUESTION / SUBJECT / CRITERIA / ANALYSIS DEPTH / "
        "OUTPUT FORMAT / FINAL RECOMMENDATION.\n"
        "Improve: comparison criteria, reasoning depth, conceptual scope, final synthesis, examples."
    ),
    Archetype.CONCISE: (
        "This is a Concise prompt. Use Minimal Modular rewriting.\n"
        "Improve: directness, brevity, clear output limit, simple wording.\n"
        "Output a single sharpened command or question — no labels, no scaffolding."
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
