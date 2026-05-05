# Promptee FastAPI backend image.
#
# Image strategy:
#   - Bake CUDA + python deps + source code only.
#   - No model weights baked in. The runtime pulls the fully merged release
#     `chickenjazz/promptee-3b` (SFT + DPO baked in) from Hugging Face Hub
#     on first request. Mount a small persistent volume at $HF_HOME so the
#     ~6 GB download survives container restarts.
#
# Runtime env vars expected:
#   FRONTEND_ORIGINS   comma-separated list of allowed CORS origins
#                      (e.g. https://your-app.vercel.app)
#   GEMINI_API_KEY     if you want the external-LLM benchmark path enabled
#   HF_HOME            cache dir for Hugging Face downloads. Point at a
#                      mounted volume (e.g. /app/.cache/huggingface) so the
#                      6 GB merged model isn't re-downloaded on every cold start.
#   HF_TOKEN           only needed if you switch the HF repo to private.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip git \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so source edits don't bust the layer cache.
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Project code (the .dockerignore excludes node_modules, .venv, datasets,
# logs, and the heavy models/sft_baseline directory).
COPY . .

EXPOSE 8000

CMD ["uvicorn", "tools.api:app", "--host", "0.0.0.0", "--port", "8000"]
