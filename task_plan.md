# Project Phases & Goals

## Phase 1: B - Blueprint (Review Pending)
- [x] Answer Discovery Questions
- [x] Define JSON Data Schema in `gemini.md`
- [ ] Confirm Payload shape
- [x] Project Research

## Phase 2: L - Link
- [x] Verify API connections and `.env` credentials (LLM providers)
- [x] Build minimal scripts in `tools/` for connectivity verification

## Phase 3: A - Architect (The 3-Layer Build)
- [x] Create technical SOPs in `architecture/`
- [x] Develop Navigation / Decision Making logic
- [x] Build Python engines in `tools/`
  - [x] `HeuristicScorer` module (Clarity, Specificity, Semantic Preservation via sentence-transformers)
  - [x] `PromptOptimizer` module (Load fine-tuned Qwen2.5-3B-Instruct + LoRA)
  - [x] `ExternalLLMService` (OpenAI/Claude/Gemini API calls)
  - [x] FastAPI `PromptController` and `PromptService` (tools/api.py)

## Phase 4: S - Stylize (Refinement & UI)
- [x] Format outputs and payloads
- [x] Build React frontend for interactive Evaluation Dashboard
- [x] Build Offline DPO Dataset Pipeline & Training Infrastructure

## Phase 5: T - Trigger (Deployment)
- [ ] Cloud Transfer
- [ ] Set up execution triggers
- [ ] Finalize Maintenance Log in `gemini.md`
