# ---- base image ----
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PORT=8000 \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential curl ca-certificates tini fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# spaCy model (small)
RUN python -m spacy download en_core_web_sm

# Pre-cache SBERT so runtime doesn't download
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")
PY

# App code
COPY . .
RUN mkdir -p /app/uploads

# ✅ HEALTHCHECK without heredoc (uses curl)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl -fsS "http://localhost:${PORT:-8000}/healthz" || exit 1

ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["sh","-c","uvicorn app_main:app --host 0.0.0.0 --port ${PORT:-8000}"]
