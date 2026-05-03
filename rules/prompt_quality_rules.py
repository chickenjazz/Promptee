"""Constants used by tools/prompt_diagnostics.py and tools/recommendation_engine.py.

Centralized so we have one place to tune the deterministic prompt-quality rules
instead of duplicating literal sets across modules.
"""

AMBIGUOUS_TOKENS = {
    "some",
    "something",
    "stuff",
    "things",
    "thing",
    "etc",
    "whatever",
    "maybe",
    "probably",
    "kind",
    "sort",
    "basically",
    "generally",
    "nice",
    "good",
    "bad",
    "better",
}


WEAK_PHRASES = {
    "talk about": "Use a stronger action verb such as explain, compare, analyze, or summarize.",
    "look into": "Specify whether the AI should analyze, summarize, investigate, or explain.",
    "do something": "State the exact task the AI should perform.",
    "make it better": "Specify what should improve, such as clarity, tone, structure, accuracy, or conciseness.",
    "make this good": "Specify the quality target, such as professional, concise, persuasive, or beginner-friendly.",
    "fix this": "Specify what kind of fix is needed, such as grammar, logic, code behavior, formatting, or clarity.",
}


OUTPUT_FORMAT_TERMS = {
    "format",
    "output",
    "return",
    "respond",
    "table",
    "bullet",
    "bullets",
    "json",
    "paragraph",
    "list",
    "essay",
    "markdown",
    "csv",
}


CONTEXT_TERMS = {
    "beginner",
    "intermediate",
    "advanced",
    "student",
    "teacher",
    "grade",
    "college",
    "audience",
    "context",
    "purpose",
    "for my",
    "for a",
}


QUESTION_STARTERS = (
    "what is",
    "what are",
    "who is",
    "who are",
    "why is",
    "why are",
    "how does",
    "how do",
    "do you know",
    "can you explain",
)
