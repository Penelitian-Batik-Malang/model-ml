# syntax=docker/dockerfile:1.4

# ==========================================
# Stage 1: Builder - Install dependencies
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install only runtime libraries needed by Pillow/OpenCV wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements dan install Python packages
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt


# ==========================================
# Stage 2: Runtime - Minimal production image
# ==========================================
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages dari builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY main.py .
COPY app/ ./app/
COPY tpu/ ./tpu/

# Create mount points for external volumes
RUN mkdir -p /app/models /app/checkpoints /app/data /app/sessions

# Set PYTHONPATH untuk tpu/models dan subfoldernya
ENV PYTHONPATH=/app:/app/tpu:/app/tpu/models:/app/tpu/models/official/efficientnet:/app/tpu/models/hyperparameters
ENV PYTHONUNBUFFERED=1

# Non-root user untuk security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/health', timeout=5)"

# Expose port
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "30", "--log-level", "info"]
