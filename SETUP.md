# Collaborator Setup

This guide gets you to a running Promptee instance with the same model the maintainer uses, without any local training. The merged model is hosted on Hugging Face Hub at [chickenjazz/promptee-3b](https://huggingface.co/chickenjazz/promptee-3b) and downloads automatically on first run (~6 GB, cached after).

## Prerequisites

- **Python 3.10–3.12**
- **NVIDIA GPU with ≥6 GB VRAM** (4-bit NF4 quantization at runtime). CPU-only works but inference will be slow.
- **CUDA toolkit** matching your PyTorch build
- **Node.js 20+** (for the frontend)
- **~10 GB free disk** for the HF cache + dependencies

## 1. Clone and install

```bash
git clone https://github.com/chickenjazz/Promptee.git
cd Promptee

# Python deps
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Frontend deps
cd Frontend
npm install
cd ..
```

## 2. (Optional) Hugging Face authentication

The merged model is public, so no token is required. If you ever switch the repo to private, run:

```bash
huggingface-cli login
```

## 3. Run the backend

```bash
# from project root, with venv active
uvicorn tools.api:app --reload --host 0.0.0.0 --port 8000
```

First startup takes a few minutes — Transformers downloads `chickenjazz/promptee-3b` into `~/.cache/huggingface/hub/`. Subsequent starts are fast (cached).

You should see:

```
Loading tokenizer: chickenjazz/promptee-3b
Loading model: chickenjazz/promptee-3b (4-bit NF4)
LoRA adapters merged into base weights.
Optimizer engine loaded successfully.
```

If you see `No adapters found in models/adapters. Using base model weights.` that is **expected** for the merged model — the DPO LoRA is already baked in.

## 4. Run the frontend

In a separate terminal:

```bash
cd Frontend
npm run dev
```

Open http://localhost:3000.

## 5. Inference programmatically

If you want to use the model outside the API, three lines:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("chickenjazz/promptee-3b")
model = AutoModelForCausalLM.from_pretrained("chickenjazz/promptee-3b", device_map="auto")
```

Or use the `PromptOptimizer` class directly, which adds 4-bit quantization, the system meta-prompt, and output cleaning:

```python
from tools.prompt_optimizer import PromptOptimizer

opt = PromptOptimizer()
opt.load_model()
print(opt.rewrite("write a python function that reverses a list"))
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` at load | Lower the batch size or run on CPU: `device_map="cpu"`. 4-bit NF4 needs ~3–4 GB of VRAM for inference. |
| `bitsandbytes` import error on Windows | Install the prebuilt wheel: `pip install bitsandbytes --upgrade --force-reinstall --no-deps`. |
| First run hangs at "Loading tokenizer" | HF download in progress. Watch `~/.cache/huggingface/` — file should be growing. |
| Output looks like vanilla Qwen, not refined | Confirm the log line `Loading model: chickenjazz/promptee-3b`. If it loaded `Qwen/Qwen2.5-3B-Instruct` instead, the default in `tools/prompt_optimizer.py` was overridden. |
| Want to use a different base model | Pass it explicitly: `PromptOptimizer(base_model_id="my-org/my-model")`. |

## For maintainers: pushing a new release

After retraining the DPO adapter, regenerate and re-upload the merged model:

```bash
huggingface-cli login                                                # once
python training/merge_for_release.py --push chickenjazz/promptee-3b
```

This folds the new `models/adapters/` into `models/sft_baseline/` and pushes the result.
