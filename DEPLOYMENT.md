# Promptee — Deployment Runbook

Two separate deployments:

| Component | Where | Notes |
|---|---|---|
| Next.js frontend (`Frontend/`) | **Vercel** | Free Hobby tier is enough |
| FastAPI backend (`tools/api.py`) | **RunPod GPU Pod** (or Modal) | Needs ≥16 GB VRAM (L4 / A10 / RTX 4090) |

---

## What was already wired up for you

- `Frontend/components/tabs/DemoTab.tsx` now reads `process.env.NEXT_PUBLIC_API_URL`. Falls back to `http://127.0.0.1:8000` for local dev.
- `Frontend/.env.local` — local dev value, **do not commit**.
- `Frontend/.env.example` — checked in so others know which vars to set.
- `tools/api.py` CORS now appends extra origins from `FRONTEND_ORIGINS` env var (comma-separated).
- `Dockerfile` + `.dockerignore` at repo root. Image excludes the entire `models/` directory and `Frontend/` (deployed separately). The merged release model is pulled from Hugging Face Hub at runtime — see Part A.

---

## Part A — Backend on RunPod

The runtime model is the public Hugging Face release [chickenjazz/promptee-3b](https://huggingface.co/chickenjazz/promptee-3b) — SFT + DPO already merged into a single ~6 GB checkpoint. No local weights, no Network Volume upload, no `scp`.

### A1. Build and push the backend image

```bash
# from repo root
docker build -t <your-dockerhub-user>/promptee-api:latest .
docker push <your-dockerhub-user>/promptee-api:latest
```

If you don't have Docker Hub, GitHub Container Registry (`ghcr.io/<user>/promptee-api`) works the same.

### A2. (Recommended) Create a small HF cache volume

The first request triggers a ~6 GB download from HF into `$HF_HOME`. Without a persistent volume, every cold start pays that cost. With one, you pay it once.

1. RunPod dashboard → **Storage → Network Volumes → New Network Volume**.
   - Region: same one you'll deploy the pod in.
   - Size: **10 GB** (covers the merged model + tokenizer + headroom).
2. No upload step. The volume starts empty; the first model load fills it.

If you want to skip this, the pod still works — every cold start just re-downloads 6 GB.

### A3. Deploy the GPU Pod

1. RunPod → **Pods → Deploy** → pick a GPU (RTX 4090 or L4 are the cheapest that fit ≥16 GB VRAM).
2. **Container Image**: `<your-dockerhub-user>/promptee-api:latest`.
3. **Volume Disk** (optional but recommended): attach the volume from A2, mount path `/app/.cache/huggingface`.
4. **Expose HTTP Ports**: `8000`.
5. **Environment Variables**:
   - `FRONTEND_ORIGINS` = `https://<your-vercel-app>.vercel.app` (you'll get this URL after Part B; redeploy the pod once you have it).
   - `GEMINI_API_KEY` = your key (only if you want the benchmark path).
   - `HF_HOME` = `/app/.cache/huggingface` (matches the mount path in step 3).
   - `HF_TOKEN` = only needed if you switched `chickenjazz/promptee-3b` to private.
6. Deploy. RunPod gives you a URL like `https://<pod-id>-8000.proxy.runpod.net`.
7. Smoke test:
   ```bash
   curl -X POST https://<pod-id>-8000.proxy.runpod.net/optimize_prompt \
        -H "Content-Type: application/json" \
        -d '{"prompt":"write me an essay"}'
   ```
   First call is slow (HF download + model load into VRAM, several minutes). Subsequent calls are fast. With the cache volume attached, only the very first ever cold start pays the download cost.

You can confirm the right model loaded by checking pod logs for:
```
Loading tokenizer: chickenjazz/promptee-3b
Loading model: chickenjazz/promptee-3b (4-bit NF4)
```
The line `No adapters found in models/adapters. Using base model weights.` is **expected** in production — the DPO LoRA is already merged into the HF release.

### A4. (Optional) Switch to RunPod Serverless later

Serverless is cheaper for low traffic but requires wrapping the API in a `runpod.serverless.start` handler instead of a plain HTTP server. Skip until traffic justifies it.

---

## Part B — Frontend on Vercel

### B1. Push the repo to GitHub

If you haven't already:
```bash
git add .
git commit -m "prep deployment configs"
git push
```

### B2. Import to Vercel

1. [vercel.com/new](https://vercel.com/new) → **Import** your GitHub repo.
2. **Root Directory**: click **Edit** and set to `Frontend`. (Critical — without this, Vercel tries to build the Python repo root.)
3. **Framework Preset**: Next.js (auto-detected).
4. **Environment Variables** — add:
   - `NEXT_PUBLIC_API_URL` = `https://<pod-id>-8000.proxy.runpod.net` (the URL from A3, no trailing slash).
5. **Deploy**.
6. After deploy succeeds, copy the `*.vercel.app` URL.

### B3. Wire CORS back to Vercel

1. Go back to RunPod → your pod → **Edit env vars**.
2. Set `FRONTEND_ORIGINS` to your Vercel URL (e.g. `https://promptee-xyz.vercel.app`).
3. Restart the pod.
4. Open the Vercel URL in a browser → run a prompt → should work end to end.

---

## Local development (unchanged)

```bash
# terminal 1 — backend
.venv\Scripts\activate
uvicorn tools.api:app --reload

# terminal 2 — frontend
cd Frontend
npm run dev
```

`Frontend/.env.local` already points at `http://127.0.0.1:8000`.

---

## Cost expectation

- **Vercel**: $0 (Hobby tier).
- **RunPod GPU Pod, always-on RTX 4090**: ~$0.34/hr × 730 hr = **~$250/mo**.
- **RunPod GPU Pod, stop when idle**: pay only for hours running. Stopping/starting takes ~30 s (plus a one-time HF download on first start unless the cache volume is attached).
- **Network Volume (HF cache, 10 GB)**: ~$0.07/GB/mo = **$0.70/mo**.

Stop the pod when you're not actively demoing. That single habit is the difference between a $5 month and a $250 month.

---

## Common gotchas

- **CORS error in browser console** → `FRONTEND_ORIGINS` on the pod doesn't match the exact Vercel URL (check protocol, no trailing slash).
- **First-ever request takes 5+ minutes** → Expected. The pod is downloading `chickenjazz/promptee-3b` (~6 GB) from HF on first load. Attach the cache volume from A2 to amortize this across restarts.
- **HF download fails with 401** → The repo is private and `HF_TOKEN` isn't set on the pod, or the token lacks read access.
- **`No adapters found in models/adapters` log line** → Expected on every production start. The DPO adapter is already merged into `chickenjazz/promptee-3b`; the runtime correctly falls through to use the merged weights.
- **Vercel build fails with "module not found" for backend imports** → Root Directory wasn't set to `Frontend`. Fix in Vercel project settings.
- **First request after pod start times out from the frontend** → Increase frontend fetch timeout, or hit `/optimize_prompt` once with a throwaway curl after starting the pod to warm the model.
