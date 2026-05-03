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

Your task is to rewrite the raw prompt into a clearer, more specific, better-structured, and more reliable prompt while preserving the original user intent.

Internally judge what kind of task the prompt represents and pick a rewrite shape that fits the content. Do NOT output any archetype label, diagnosis, explanation, or improvement summary.

Choose the rewrite shape that fits the task:
- Coding, technical specifications, or build-this-thing requests: use modular sections drawn from ROLE, TASK, INPUTS, OUTPUT, CONSTRAINTS, EDGE CASES, TESTING, CONTEXT — and only the ones the request actually warrants.
- Structured deliverables (plans, checklists, lesson plans, frameworks): use modular sections drawn from OBJECTIVE, SECTIONS, FORMAT, DETAIL LEVEL, ORDER, CONSTRAINTS — and only the ones that fit.
- Analytical, comparative, or evaluative requests: use light modular structure (e.g. ROLE, TASK, REQUIREMENTS, or QUESTION, SUBJECT, CRITERIA, OUTPUT FORMAT) only when sections add clarity. Otherwise keep it as a clear paragraph instruction.
- Creative writing: prefer a clear single-paragraph instruction; use lightly-labeled sections (ROLE, TASK, REQUIREMENTS) only when the request is genuinely complex.
- Conversational, advisory, or interpersonal requests: write one or two flowing paragraphs of natural language. Do NOT impose ROLE / TASK / OUTPUT scaffolding.

Section header rules:
- Only use section headers that GENUINELY FIT the task. Do NOT use coding-style headers (LANGUAGE/STACK, INPUTS, EDGE CASES, FILES/FUNCTIONS, TESTING) for non-coding tasks.
- Use section names as real headings (each on its own line, followed by content).
- Do not add empty or generic sections just to complete a template. Use the smallest structure that preserves the task and improves clarity.

Rewrite rules:
- Improve clarity, specificity, completeness, output instructions, readability, logical flow, and token efficiency.
- Preserve the original intent, topic, constraints, and expected task.
- DO NOT ANSWER the prompt.
- Do NOT generate the requested output.
- Do NOT add irrelevant requirements.
- Do not transform the task into "create a prompt for this task" unless the original user explicitly asks for prompt creation.
- If the raw prompt is a question, rewrite it as a clearer prompt. Do not answer the question.
- If the raw prompt asks for code, rewrite the coding request. Do not provide the code implementation.
- If the raw prompt asks for an explanation, rewrite the explanation request. Do not provide the explanation.
- Keep the rewritten prompt as a direct instruction to the assistant, not as a meta-instruction about prompt writing.
- Do not include labels such as Archetype, Weaknesses Found, Rewritten Prompt, or Improvement Summary.
- Return only the final rewritten prompt."""


# Per-archetype guidance lifted from architecture/auto-rewriter.md §Archetype-Specific Rewrite Rules.
#
# REFERENCE ONLY — no longer injected into the runtime system prompt.
#
# The runtime instruction is now a single stable prompt (_BASE_SYSTEM) so a
# misclassified prompt cannot drag the rewriter into a wrong-shape scaffold —
# e.g., an HR question that hits the Coding regex on the word "function" used
# to be force-fit into TASK / LANGUAGE/STACK / EDGE CASES sections, which
# tanked semantic preservation and tripped the API's improvement gate.
#
# Archetype detection is preserved for metadata, diagnostics, highlighting,
# recommendations, analytics, and Chapter 4 reporting. This dictionary keeps
# the canonical per-archetype guidance text so those layers can consult it
# without re-deriving it.
_ARCHETYPE_GUIDANCE = {
    Archetype.CREATIVE: (
        "This is a Creative prompt. Use Semi Modular rewriting.\n"
        "Improve: tone, audience, style, mood, originality, output constraints.\n"
        "Avoid overly rigid section labels unless the original prompt is complex."
    ),
    Archetype.CODING: (
        "This is a Coding prompt. Use structured rewriting.\n"
        "Improve clarity, specificity, completeness, expected behavior, constraints, and output usability.\n"
        "Preserve the original coding task exactly. Do not answer it or write the implementation.\n\n"

        "Use clear section headings. Include these core sections when applicable:\n"
        "TASK:\n"
        "LANGUAGE/STACK:\n"
        "OUTPUT:\n"
        "CONSTRAINTS:\n\n"

        "Optional section headings. Include only when clearly relevant or implied by the raw prompt:\n"
        "INPUTS:\n"
        "FILES/FUNCTIONS:\n"
        "EDGE CASES:\n"
        "TESTING:\n"
        "CONTEXT:\n\n"

        "Rules for Coding prompts:\n"
        "- Use section names as actual headings, not as a slash-separated list.\n"
        "- Use the smallest structured format that preserves the original task and improves clarity.\n"
        "- Do not add empty or generic sections just to complete a template.\n"
        "- Do not phrase the rewrite as 'create a prompt that asks...' or 'ask the assistant to...'.\n"
        "- Rewrite the user's coding request directly."
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
    """Return the full prompt plan (archetype, modularity, system + user messages).

    The system prompt is a single stable instruction that does NOT vary by
    archetype. Archetype + modularity are returned as metadata for
    diagnostics, highlighting, recommendations, and analytics — they are
    never injected into the model-facing prompt. This protects the rewriter
    from an archetype misclassification pulling it into a wrong-shape scaffold.
    """
    archetype = detect_archetype(raw_prompt)
    modularity = modularity_for(archetype)
    system = _BASE_SYSTEM
    # User wrapper kept identical to the form used when the DPO `chosen` pairs
    # were generated. Anti-answer / anti-meta rules live in _BASE_SYSTEM so the
    # adapter sees the same user-role shell it was trained against — duplicating
    # them in the user role caused the adapter to echo this wrapper verbatim
    # instead of producing a rewrite (observed on "Create a C++ code about oop
    # and games" — distribution shift, not a generation parameter issue).
    user = (
        "Raw prompt:\n"
        f"{raw_prompt.strip()}\n\n"
        "Final rewritten prompt only:"
    )
    return PromptPlan(archetype, modularity, system, user)
