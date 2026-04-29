FROM python:3.7-slim

# Install system dependencies untuk OpenCV, pycocotools, dan OpenGL
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip setuptools wheel

# requirements.txt dari pip freeze — simpan sebagai UTF-8 sebelum build
COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download ConvNeXt Small weights agar tidak diunduh saat container start
RUN python -c "from torchvision import models; models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)"

# PYTHONPATH untuk fashion inference script (tpu/models)
ENV PYTHONPATH="/app/tpu/models:/app/tpu/models/official/efficientnet:/app/tpu/models/hyperparameters"

EXPOSE 8000
