# DPO Training Pipeline SOP

## 1. Goal
Train LoRA adapters on Qwen2.5-3B-Instruct using Direct Preference Optimization (DPO) so the model learns to produce higher-quality prompt rewrites as ranked by the HeuristicScorer.

## 2. Inputs
- `datasets/raw_prompts.jsonl` — Source raw prompts
- `datasets/preference_pairs.jsonl` — Generated preference dataset `{x, y_w, y_l}`

## 3. Preference Pair Generation (dataset_builder.py)

### Workflow
1. Load raw prompts from `datasets/raw_prompts.jsonl`
2. For each raw prompt `x`:
   a. Generate multiple rewrite candidates using the base Qwen model
   b. Score each candidate with `HeuristicScorer.evaluate(x, candidate)`
   c. Filter: reject candidates where `semantic_preservation < threshold`
   d. Select `y_w` = candidate with highest `total` score that passes the gate
   e. Set `y_l` = original raw prompt `x` (per Discovery answer)
3. Write `{x, y_w, y_l}` to `datasets/preference_pairs.jsonl`

### Constraints
- Rewrite candidates must be modularized (allow swapping generation strategies)
- Semantic similarity threshold applies as a hard gate
- If no candidate passes the gate, skip that prompt (do not generate a broken pair)

## 4. QLoRA Configuration

| Parameter | Value |
|---|---|
| Quantization | 4-bit NormalFloat (NF4) |
| Double quantization | Enabled |
| Compute dtype | bfloat16 |
| Base weights | Frozen |
| Trainable params | LoRA adapter matrices only |
| Optimizer | Paged AdamW 8-bit |
| Gradient checkpointing | Enabled |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |

## 5. DPO Training (dpo_trainer.py)

### Workflow
1. Load base Qwen2.5-3B-Instruct with 4-bit quantization
2. Attach LoRA adapters to attention layers
3. Load `datasets/preference_pairs.jsonl`
4. Initialize `trl.DPOTrainer` with:
   - `beta=0.1` (KL penalty coefficient)
   - `max_length=512`
   - `max_prompt_length=256`
5. Train for configured epochs
6. Save adapter checkpoints to `training/checkpoints/`

### Optimization Objective
```
maximize log P(y_w | x) > log P(y_l | x)
```

## 6. Export (export_adapter.py)

### Workflow
1. Load best checkpoint from `training/checkpoints/`
2. Save inference-ready adapter weights to `models/adapters/`
3. Verify adapter loads correctly with `PeftModel.from_pretrained()`

## 7. Edge Cases
- **GPU memory overflow**: Use gradient checkpointing + paged optimizer. Target: RTX 3070 (8GB VRAM).
- **Empty preference pairs**: If dataset_builder produces 0 valid pairs, abort training with clear error.
- **Divergent training**: Monitor loss; if loss plateaus or increases, reduce learning rate.
