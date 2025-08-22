# ---- base image ----
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PORT=8000

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential curl ca-certificates tini fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python deps ----
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# spaCy model
RUN python -m spacy download en_core_web_sm

# Pre-cache the SBERT model so first request is fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ---- app code ----
COPY . .
RUN mkdir -p /app/uploads

ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["sh","-c","uvicorn app_main:app --host 0.0.0.0 --port ${PORT:-8000}"]
