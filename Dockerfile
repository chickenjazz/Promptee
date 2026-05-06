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
#                      WITHOUT trailing slashes (browsers never send one in
#                      the Origin header, so a trailing slash here causes the
#                      CORS preflight to 400 and the frontend silently fails).
#                      e.g. https://promptee-fawn.vercel.app
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

# Ubuntu 22.04 ships Python 3.10 by default; the `python3-pip` apt package
# binds to it. Keep them aligned so `pip install` and `python -m ...` use the
# same interpreter — installing python3.11 alongside breaks this and produces
# "No module named X" errors at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so source edits don't bust the layer cache.
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install -r requirements.txt \
 && python -m spacy download en_core_web_sm

# Project code (the .dockerignore excludes node_modules, .venv, datasets,
# logs, and the heavy models/sft_baseline directory).
COPY . .

EXPOSE 8000

CMD ["uvicorn", "tools.api:app", "--host", "0.0.0.0", "--port", "8000"]
