# Research, Discoveries, and Constraints

## 2026-04-07 Discovery
- Project aligns with the architecture detailed in "Prompt Optimization Pipeline Using Multi-Criteria Heuristic Evaluation and Transformer Fine-Tuning Algorithms" (`QuadCore _MANUSCRIPT (3).pdf` and `claude.md`).
- A fully automated prompt optimization pipeline using a fine-tuned Qwen2.5-7B-Instruct.
- Evaluation via Multi-Criteria Decision-Making (MCDM) heuristic engine (Clarity, Specificity, Semantic Preservation).
- Model alignment via Direct Preference Optimization (DPO) and QLoRA.
- Architecture consists of Presentation Layer (React), Application Layer (FastAPI), Model Layer, and Offline Training Layer.

## 2026-04-07 Architectural Deep-Dive
- **Runtime Integration**: External LLMs (OpenAI/Claude/Gemini) are strictly for *evaluation comparison* (raw vs optimized). They require API keys stored in `.env`.
- **NLP Integrations**: `sentence-transformers` for cosine similarity/BERTScore recall.
- **Model details**: Qwen2.5-7B-Instruct is the base model. LoRA adapters live in `models/adapters/`.
- **Data rules**: Strict separation between runtime inference and offline training. Runtime does not train.
- **Constraints**: Optimization fails and falls back to original if semantic similarity drops below threshold or if heuristic improvement is <= 0.
