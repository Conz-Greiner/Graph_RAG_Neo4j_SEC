# syntax=docker/dockerfile:1.6
# Multi-stage build for the Agentic Graph-RAG application service.

# ---------- Stage 1: builder ----------
FROM python:3.11.9-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# System deps needed to compile some Python wheels (e.g. sentence-transformers)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --prefix=/install -r requirements.txt

# ---------- Stage 2: runtime ----------
FROM python:3.11.9-slim AS runtime

LABEL maintainer="mlops-team <mlops@example.com>"
LABEL org.opencontainers.image.title="agentic-graph-rag"
LABEL org.opencontainers.image.description="Production Agentic Graph-RAG service"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8080

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        poppler-utils \
        tesseract-ocr \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd --gid 1000 agr && \
    useradd --uid 1000 --gid agr --shell /bin/bash --create-home agr

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY config ./config
COPY src   ./src
COPY tests ./tests

# Data & logs directories owned by non-root user
RUN mkdir -p /app/data /app/logs /app/.cache && \
    chown -R agr:agr /app

USER agr

EXPOSE 8080 8501

HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=10 \
    CMD curl -fsS http://localhost:8080/health || exit 1

# Default command: run FastAPI. Override in compose / k8s to launch Streamlit.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]