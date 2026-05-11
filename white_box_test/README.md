# White-Box Tests

Tests that exercise the *internal* logic of Promptee's algorithms — branches, decision points, and edge cases — rather than treating each component as a black box.

## What's covered

| Script | Rubric item | Targets |
|---|---|---|
| `test_heuristic_scoring.py` | Heuristic Scoring Accuracy + Algorithm Efficiency | `tools/heuristic_scorer.py` — clarity / specificity / penalties / final score / timing |
| `test_prompt_optimization.py` | Prompt Optimization Effectiveness | `tools/prompt_optimizer.py` — preamble stripping, fallback paths, generation kwargs (model is stubbed) |
| `test_semantic_preservation.py` | Semantic Preservation | `tools/heuristic_scorer.py` — semantic gating thresholds (real embeddings + boundary stubs) |
| `test_training_config.py` | DPO + QLoRA Implementation | `training/dpo_trainer.py`, `training/sft_trainer.py` — LoRA / 4-bit / dropout config (static AST inspection, no GPU) |
| `test_technical_reliability.py` | Technical Reliability | Edge cases: empty, None, very long inputs, exception fallbacks |
| `test_model_integration.py` | Model Integration | `tools/api.py` `/optimize_prompt` orchestration with stubbed optimizer |

## What's intentionally not covered

- **Actual DPO / QLoRA training runs.** They require a GPU and multi-hour runs. The config-shape tests here verify the static configuration that the trainers consume, which is the only part with deterministic logic.
- **Real LLM inference** in the optimizer suite. Loading Qwen2.5-3B + 4-bit quantization is environment-specific; the stubbed-model tests cover the deterministic pre/post-processing logic instead.

## Running

From the repository root:

```bash
pytest white_box_test/ -v
```

First run will take ~5–10s loading spaCy and `all-MiniLM-L6-v2`. Subsequent test files share the same scorer instance via a session-scoped fixture.

To run a single rubric item:

```bash
pytest white_box_test/test_heuristic_scoring.py -v
pytest white_box_test/test_semantic_preservation.py -v
```

## Notes

- The suite never imports `training/dpo_trainer.py` or `training/sft_trainer.py` directly — those modules contain a top-level `raise RuntimeError` if CUDA is missing. We parse them with the `ast` module instead.
- `test_model_integration.py` patches the optimizer with a stub before constructing the FastAPI `TestClient`, so no GPU model is loaded.
