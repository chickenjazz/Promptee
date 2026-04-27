# Auto Prompt Rewrite Dataset Builder

Implements [`architecture/auto-rewriter.md`](../architecture/auto-rewriter.md). Reads a CSV with a `prompt` column, rewrites each prompt using `Qwen/Qwen2.5-3B-Instruct`, and writes the improved version to `rewritten_prompt`.

## Install

The repo's top-level `requirements.txt` already covers the heavy dependencies (`torch`, `transformers`, `accelerate`, `bitsandbytes`). For a minimal install scoped to this builder:

```bash
pip install -r dataset_builder/requirements.txt
```

## Run

Smoke-test on 20 rows into a separate file:

```bash
python -m dataset_builder.build_rewritten_dataset \
  --input dataset/test_dataset.csv \
  --output dataset/test_dataset.smoke.csv \
  --overwrite --limit 20 --start-row 0
```

Resume a real run (skips rows that already have a non-empty `rewritten_prompt`):

```bash
python -m dataset_builder.build_rewritten_dataset \
  --input dataset/test_dataset.csv \
  --output dataset/test_dataset.csv \
  --save-every 25
```

## CLI Reference

| Flag | Default | Purpose |
|---|---|---|
| `--input` | `dataset/test_dataset.csv` | Input CSV. Must contain a `prompt` column. |
| `--output` | `dataset/test_dataset.csv` | Output CSV. Same as input is allowed (atomic write). |
| `--overwrite` | off | Regenerate even rows that already have a non-empty rewrite. |
| `--limit N` | `0` (no limit) | Process at most N rows after `--start-row`. |
| `--start-row N` | `0` | 0-based row offset to start from. |
| `--save-every N` | `25` | Checkpoint every N processed rows. |
| `--max-new-tokens N` | `512` | Override generation length (use small N for failure-injection tests). |
| `--no-quant` | off | Disable 4-bit quantization (full bfloat16; needs ~16GB VRAM). |
| `--log-level` | `INFO` | `DEBUG` for per-row archetype + length diagnostics. |

## Behavior

- **Resume-safe.** Rows where `rewritten_prompt` is non-empty are skipped unless `--overwrite` is passed.
- **Crash-safe.** Output is checkpointed every `--save-every` rows via temp-file + atomic rename. A SIGINT or hardware crash leaves the last checkpoint intact; a final save runs in `finally:`.
- **Per-row failures don't halt the run.** They're recorded in a `rewrite_error` column. Subsequent runs (without `--overwrite`) automatically retry rows whose `rewritten_prompt` is still empty.
- **Two attempts per row.** Sampled (`temperature=0.3`) first; on validation failure, deterministic retry (`do_sample=False`) per spec §Model Requirement.
- **CUDA OOM is non-fatal.** The error is recorded and the loop moves on after `torch.cuda.empty_cache()`.

## Output guarantees (spec §Validation Rules + §Cleaning Rules)

A row's `rewritten_prompt` cell never contains:

- assistant filler openers ("Sure, ...", "Here is ...", "Of course")
- diagnostic labels ("Archetype:", "Weaknesses Found:", "Improvement Summary:", "Rewritten Prompt:")
- wrapping markdown fences or wrapping quote pairs
- raw model output identical to the input prompt (for non-trivial inputs)
- output that looks like a task answer rather than a prompt (rejects leading `def `, ` ```python`, etc.)

Failures of any rule send the row to `rewrite_error` instead of writing bad data.

## File Layout

| File | Purpose |
|---|---|
| `build_rewritten_dataset.py` | CLI entrypoint + main loop + checkpointing |
| `config.py` | Constants: model id, paths, generation params, filler/label sets |
| `model_loader.py` | Qwen 4-bit load, chat-template formatter, `generate()` helper |
| `prompt_templates.py` | Archetype detector, modularity selector, LLM instruction builder |
| `cleaners.py` | Spec §Cleaning Rules pipeline |
| `validators.py` | Spec §Validation Rules with structured `ValidationResult` |
| `tests/` | Unit tests for cleaner / validator / archetype detection (no GPU required) |

## Tests

```bash
python -m pytest dataset_builder/tests -q
```

These tests cover the deterministic pieces (cleaning, validation, archetype routing) and require no GPU or model download.
