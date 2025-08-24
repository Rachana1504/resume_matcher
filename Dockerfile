# Use a slim Python base
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Build tools for wheels that need compiling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better caching)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    python -m spacy download en_core_web_sm

# Copy app
COPY . .

# Render provides $PORT; bind to it
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app_main:app --host 0.0.0.0 --port ${PORT:-8000}"]
