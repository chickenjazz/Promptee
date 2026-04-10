# Prompt Optimizer SOP

## 1. Goal
To transform a raw, sub-optimal user prompt into a highly structured, clear, and specific prompt using parameter-efficient fine-tuning (QLoRA).

## 2. Inputs
- `raw_prompt`: The original user query.

## 3. Tool Logic & Processing

### A. Model Loading (Runtime)
- **Base Model**: `Qwen2.5-7B-Instruct`
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
- **Fallback**: If the model fails to generate a response, or if the output is rejected by the Heuristic Scorer, the optimizer must return the original `raw_prompt` and log the failure.
