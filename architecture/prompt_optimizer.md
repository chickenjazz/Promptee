# Prompt Optimizer SOP

## 1. Goal
To transform a raw, sub-optimal user prompt into a highly structured, clear, and specific prompt using parameter-efficient fine-tuning (QLoRA).

## 2. Inputs
- `raw_prompt`: The original user query.

## 3. Tool Logic & Processing

### A. Model Loading (Runtime)
- **Base Model**: `Qwen2.5-3B-Instruct`
- **Quantization**: 4-bit NormalFloat (NF4) via `bitsandbytes` to ensure lightweight inference.
- **Adapter**: Load pre-trained LoRA parameters from `models/adapters/` via `peft`.

### B. Execution Flow
1. The `raw_prompt` is injected into a strict system meta-prompt instructing the model to act as a prompt optimization refinement engine.
2. The tokenizer prepares the inputs.
3. The forward pass computes the optimal rewrite without altering the model states (gradient calculation disabled).
4. The output is decoded and cleaned of any meta-commentary (the model must ONLY return the optimized text, not a conversational response).

## 4. Edge Cases
- **Over-verbosity**: The system must enforce generation parameters (e.g., `max_new_tokens`) to prevent runaway outputs.
- **Hallucination**: Mitigated by the post-processing pipeline (Heuristic Scorer) which checks the Semantic Preservation threshold.
- **Fallback**: If the model fails to generate a response, or if the output is rejected by the Heuristic Scorer or by the post-generation validator, the optimizer must return the original `raw_prompt` and log the failure.

## 5. Separation of Responsibilities

The prompt optimizer must only rewrite the prompt.

It must not generate diagnostics, recommendations, explanations, or institutional guidelines. Those are handled by separate deterministic modules:

- `tools/prompt_diagnostics.py` — finds prompt-level weaknesses and produces highlight spans.
- `tools/prompt_validator.py` — catches answer-shaped, meta-prompt, and unexpected-code outputs after generation.
- `tools/recommendation_engine.py` — produces actionable recommendations and institutional guidelines from issues + scores.

This separation prevents the model from answering the user task or drifting into meta-prompt generation.

## 6. Adapter-Safe Runtime Behavior

The runtime prompt is constructed by `dataset_builder.prompt_templates.build_plan()`. The system prompt it returns is a **single stable instruction** — it does not change based on the detected archetype. The model is given the rules for choosing rewrite shape and section headers itself, and the same instruction is sent for every prompt regardless of archetype.

Archetype detection is preserved for metadata, diagnostics, highlighting, recommendations, and analytics — never for runtime generation control. This avoids the previous failure mode where a single regex false-positive (e.g., a non-coding prompt hitting the Coding classifier on the word "function") would force the model into a `TASK / LANGUAGE/STACK / EDGE CASES` scaffold that was never appropriate for the request.

If the current adapter was trained mostly on structured chosen prompts, the runtime keeps structured prompt rewriting as the default behavior because the stable instruction tells the model to choose modular sections when the task warrants them. Concise rewrite recommendations are surfaced through the tutor layer (recommendation engine + diagnostics), but concise generation is **not** forced at runtime. See `architecture/adapter_safety.md` for the dataset audit and the criteria that gate any future adaptive concise generation.
