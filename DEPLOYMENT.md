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

Tag every build with the current git SHA. **Do not deploy `:latest`** — it's a moving target with no rollback path, and RunPod can't reliably tell when it changed.

Bash (Git Bash / WSL / Linux / macOS):
```bash
# from repo root
TAG=$(git rev-parse --short HEAD)
docker build \
  -t <your-dockerhub-user>/promptee-api:$TAG \
  -t <your-dockerhub-user>/promptee-api:latest \
  .
docker push <your-dockerhub-user>/promptee-api:$TAG
docker push <your-dockerhub-user>/promptee-api:latest
echo "Deploy this tag in RunPod: $TAG"
```

PowerShell (Windows):
```powershell
$TAG = git rev-parse --short HEAD
docker build -t chickenjazz/promptee-api:$TAG -t chickenjazz/promptee-api:latest .
docker push chickenjazz/promptee-api:$TAG
docker push chickenjazz/promptee-api:latest
"Deploy this tag in RunPod: $TAG"
```

The `:latest` tag is pushed alongside the SHA only as a convenience for `docker pull` on dev machines — RunPod itself should always reference the immutable SHA tag (see A3 step 2).

If you don't have Docker Hub, GitHub Container Registry (`ghcr.io/<user>/promptee-api`) works the same.

**For defense day specifically**, freeze a known-good build under a named tag and never overwrite it:
```bash
docker tag <user>/promptee-api:<verified-sha> <user>/promptee-api:defense-locked
docker push <user>/promptee-api:defense-locked
```
Then point the RunPod template at `:defense-locked`. Future builds on the SHA tag won't affect this frozen image.

### A2. (Recommended) Create a small HF cache volume

The first request triggers a ~6 GB download from HF into `$HF_HOME`. Without a persistent volume, every cold start pays that cost. With one, you pay it once.

1. RunPod dashboard → **Storage → Network Volumes → New Network Volume**.
   - Region: same one you'll deploy the pod in.
   - Size: **10 GB** (covers the merged model + tokenizer + headroom).
2. No upload step. The volume starts empty; the first model load fills it.

If you want to skip this, the pod still works — every cold start just re-downloads 6 GB.

### A3. Deploy the GPU Pod

1. RunPod → **Pods → Deploy** → pick a GPU (RTX 4090 or L4 are the cheapest that fit ≥16 GB VRAM).
2. **Container Image**: `<your-dockerhub-user>/promptee-api:<sha>` — the immutable SHA tag from A1, **not** `:latest`. To deploy a new version later, edit the template's image tag to the new SHA and restart the pod; RunPod sees the tag change and pulls cleanly.
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

## Pushing updates

### Frontend changes
Vercel auto-deploys from the GitHub branch.
```bash
git add Frontend/...
git commit -m "ui: ..."
git push
```
Build runs in ~60 s, production URL updates automatically.

### Backend changes
Three steps: rebuild → push → restart pod with the new SHA.
```bash
TAG=$(git rev-parse --short HEAD)
docker build -t <user>/promptee-api:$TAG -t <user>/promptee-api:latest .
docker push <user>/promptee-api:$TAG
```
Then in RunPod → your pod → **Edit** → change **Container Image** to `<user>/promptee-api:$TAG` → **Save** → **Restart**. The new SHA forces a clean pull. Roll back by editing the template back to the previous SHA — old image versions stay in the registry.

### What does NOT need a rebuild
- Editing `*.md` docs (excluded from the image by `.dockerignore`).
- Changing env vars (`FRONTEND_ORIGINS`, `HF_TOKEN`, etc.) — edit in RunPod, restart pod.
- Frontend changes — those go through Vercel only.

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
- **Restarted the pod but it's still running old code** → You're on `:latest` (or any non-immutable tag) and RunPod served a cached image. Switch the template to the explicit SHA tag from A1 and restart — that forces a clean pull.
