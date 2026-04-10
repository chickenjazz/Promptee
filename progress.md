# Progress and Maintenance Log

## 2026-04-07
- Project Initialization (Protocol 0)
- Analyzed `claude.md` and manuscript PDF.
- Initialized `task_plan.md`, `findings.md`, `progress.md`, `gemini.md`.
- Received answers to 5 Discovery Questions.
- Updated `gemini.md` with strict Data Schemas, Behavioral Rules, and Architectural Invariants.
- Phase 2 (Link): Created `.env.example`, `test_llm_links.py`, `test_nlp_links.py`, `config/settings.py`.
- Phase 3 (Architect): Created SOPs, `heuristic_scorer.py`, `prompt_optimizer.py`, `external_llm.py`, `api.py`.
- Fixed Gemini model reference: `gemini-1.5-flash` → `gemini-2.0-flash` (404 error).
- Gemini API rate-limited (429) — key valid, daily quota exhausted.

## 2026-04-08
- Phase 4 (Stylize): Built premium dark-mode React-equivalent frontend dashboard.
  - `frontend/index.html`: Full semantic layout with header, prompt input, score rings, comparison views.
  - `frontend/style.css`: Glassmorphism design system, animated SVG score rings, responsive grid.
  - `frontend/app.js`: API integration, ring animations, copy-to-clipboard, loading states, health check.
  - Mounted frontend as static files through FastAPI (`/static` + `/` route).
  - Added CORS middleware to `tools/api.py`.
- Offline DPO Pipeline: Built complete training infrastructure in `training/`.
  - `architecture/dpo_training.md`: Layer 1 SOP with QLoRA config, DPO objective, edge cases.
  - `training/dataset_builder.py`: Reads `dataset/RAW_prompts.csv`, filters for instructional tasks,
    generates modularized [ROLE][OBJECTIVE][CONTEXT][CONSTRAINTS][EXAMPLES] rewrites,
    scores with HeuristicScorer, outputs `datasets/preference_pairs.jsonl`.
  - `training/dpo_trainer.py`: QLoRA + DPO training via `trl.DPOTrainer`.
  - `training/export_adapter.py`: Exports trained adapters to `models/adapters/`.
  - Dry-run test: 2/5 valid pairs generated (3 rejected by semantic gate) — pipeline functional.
- Updated `gemini.md` y_l schema: now uses original raw prompt from dataset (per user Discovery answer).
