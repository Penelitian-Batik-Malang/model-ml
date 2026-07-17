FROM python:3.12-slim

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# ── Optimasi CPU untuk HF Spaces (2 core) ──
# Set thread count sesuai jumlah core yang tersedia
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2
ENV OPENBLAS_NUM_THREADS=2
ENV NUMEXPR_NUM_THREADS=2

# Kurangi noise log TensorFlow dan paksa CPU mode
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV CUDA_VISIBLE_DEVICES=""

# Matikan telemetry yang tidak perlu
ENV TRANSFORMERS_OFFLINE=0
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV TOKENIZERS_PARALLELISM=false

# Update dan install OS dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" dengan user ID 1000 (HF Spaces requirement)
RUN useradd -m -u 1000 user

# Switch ke user
USER user

# Set home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory
WORKDIR $HOME/app

# Copy requirements file dulu (leverage Docker cache)
COPY --chown=user:user requirements.txt $HOME/app/

# Install python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy seluruh aplikasi
COPY --chown=user:user . $HOME/app/

# HF Spaces port
ENV PORT=7860
EXPOSE 7860

# Start: download models dulu lalu jalankan server
# --timeout-keep-alive 300: penting untuk request colorizer yang bisa 2+ menit
# --workers 1: WAJIB 1 worker untuk shared model state & RAM terbatas
CMD ["sh", "-c", "python scripts/download_models.py && uvicorn main:app --host 0.0.0.0 --port 7860 --workers 1 --timeout-keep-alive 300"]
