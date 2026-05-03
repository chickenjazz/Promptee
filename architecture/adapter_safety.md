# Adapter Safety and Runtime Alignment

## Purpose

This document explains how Promptee avoids invalidating its trained LoRA/DPO adapters during runtime refactors.

## Core Principle

The runtime rewriter should remain aligned with the output style found in `datasets/preference_pairs.jsonl`.

If the preference pairs mostly contain structured chosen prompts, the runtime rewriter should continue requesting structured prompt rewrites. The empirical audit (`tools/audit_preference_pairs.py`) reports a median chosen length of 151 words and 50.15% of chosen examples containing at least one structure marker (ROLE, TASK, INPUTS, OUTPUTS, CONSTRAINTS, REQUIREMENTS) — so adapter alignment is about *length and verbosity* more than strict marker presence.

## Single Source of Truth

`dataset_builder.prompt_templates.build_plan()` is the only place that constructs the system + user prompt at runtime. The optimizer routes every `rewrite()` call through it.

The system prompt it returns is **a single stable instruction that does not vary by archetype**. Archetype detection is preserved, but archetype/modularity are returned only as metadata for diagnostics, highlighting, recommendations, analytics, and Chapter 4 reporting. They are never injected into the model-facing prompt.

This protects the rewriter from a single classifier false-positive (e.g., the word "function" in "make the company *function* from one day to the next") pulling the rewrite into a wrong-shape scaffold (TASK / LANGUAGE/STACK / EDGE CASES) for an HR/organizational prompt. Such mismatches previously tanked semantic preservation enough to trip the API's `improvement < 0` gate and force fallback to raw.

## Safe Changes

The following changes do not invalidate the adapter:

- deterministic diagnostics
- text highlighting
- post-generation validation of model output
- recommendation generation outside the model
- frontend display changes
- stricter rewrite-only boundaries inside the user message

## Risky Changes

The following changes may weaken adapter behavior unless the dataset is expanded and the adapter is retrained:

- forcing concise one-line rewrites at runtime
- making rewrite style fully adaptive at runtime
- removing structured output expectations
- changing the system prompt far away from the DPO training distribution

## Future Adaptive Rewrite Policy

Adaptive minimal/semi/full modular rewriting should only be activated after auditing the preference dataset and adding enough concise chosen examples. The current dataset has only 9.8% of chosen outputs under 80 words, so any concise-routing threshold is non-binding on this distribution.
