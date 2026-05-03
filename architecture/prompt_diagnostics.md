# Prompt Diagnostics and Highlighting

## Purpose

The prompt diagnostics module identifies prompt-level weaknesses that reduce clarity, specificity, structure, or reliability.

Unlike the prompt rewriter, this module does not use a language model. It uses deterministic rules so that the system remains explainable and repeatable.

## Issue Categories

- Ambiguity
- Weak action phrase
- Redundancy
- Missing output format
- Missing context
- Missing constraints
- Too short
- Answering risk
- Meta-prompt drift risk

## Output Schema

Each issue contains:

- `id`
- `type`
- `severity` (low / medium / high)
- `span` (the matched substring, or null)
- `start` / `end` (character offsets, or null)
- `message`
- `suggestion`

Issues with `start` and `end` are rendered as inline highlights in `Frontend/components/PromptHighlighter.tsx`. Issues without spans are shown in the issue panel and feed the recommendation engine.

## Configuration

Constants live in `rules/prompt_quality_rules.py`:

- `AMBIGUOUS_TOKENS` — vague tokens like "something", "etc", "stuff"
- `WEAK_PHRASES` — phrases like "make it better", "fix this"
- `OUTPUT_FORMAT_TERMS` — words signaling format intent (table, bullet, JSON, …)
- `CONTEXT_TERMS` — audience or purpose hints (beginner, student, audience, …)
- `QUESTION_STARTERS` — direct-question prefixes used to flag answering risk

## Relationship to Heuristic Scoring

The heuristic scorer produces numerical quality metrics. The diagnostics module explains *which parts* of the prompt caused weakness. Together, these allow the system to provide actionable prompt-engineering feedback.
