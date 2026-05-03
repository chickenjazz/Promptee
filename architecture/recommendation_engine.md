# Recommendation Engine

## Purpose

The recommendation engine converts scoring results and diagnostic issues into actionable educational guidance.

This supports the research question:

> How can the system's evaluation results be utilized to formulate actionable recommendations and institutional guidelines for the responsible AI use in educational settings?

## Inputs

- Raw prompt
- Diagnostic issues (from `tools/prompt_diagnostics.py`)
- Heuristic scores (from `tools/heuristic_scorer.py`)
- Archetype metadata

## Outputs

- `recommendations` — list of plain-English suggestions tailored to the issue mix
- `institutional_guideline` — single statement appropriate to the prompt's complexity and archetype

## Adapter-Safe Principle

The recommendation engine may explain that a concise prompt is enough for simple classroom use, but it does not require the adapter to generate concise rewrites. Concise generation is gated on dataset expansion + retraining (see `architecture/adapter_safety.md`).

## Example

Raw prompt: `What is gravity?`

Recommendation includes:

> This appears to be a simple learning prompt. A concise prompt may be enough in normal classroom use, even if the optimizer returns a structured rewrite.

Institutional guideline:

> For simple factual or educational prompts, students should prioritize clarity, audience level, and expected depth. Rigid prompt templates are optional and should not be required for every AI interaction.
