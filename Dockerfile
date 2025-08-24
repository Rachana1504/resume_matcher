# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# system deps (spacy wheels etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt && \
    python -m spacy download en_core_web_sm

COPY . .
ENV PORT=8000
EXPOSE 8000
CMD ["uvicorn", "app_main:app", "--host", "0.0.0.0", "--port", "8000"]
