# Batik & Fashion ML API

This repository contains the Machine Learning services for Batik Classification, Fashion Segmentation, and Batik Recommendations.

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.9+
- Docker & Docker Compose (Recommended)
- Large Model Files (see `models/README.md` and `checkpoints/README.md`)

### 2. Installation (Local)
```bash
# Clone the repository
git clone https://github.com/Penelitian-Batik-Malang/model-ml.git
cd model-ml

# Install dependencies (choose service)
pip install -r requirements-batik.txt
# OR
pip install -r requirements-fashion.txt
```

### 3. Running with Docker
```bash
docker-compose up --build
```

---

## 📦 Large Files Management (Models & Checkpoints)

Due to their size, models and checkpoints are **not included** in this repository. 

### Where to store them?
- **Development**: Keep them in the `models/`, `checkpoints/`, and `data/` folders as specified in `config.py`.
- **VPS/Production**: 
    1. Store the files in a secure Cloud Storage (e.g., S3, Google Drive, or a dedicated storage server).
    2. On the VPS, create these directories manually in the project root.
    3. Use `wget` or `curl` to download them directly to the VPS.
    4. **Docker Tip**: Always mount these directories as **Volumes** so you don't have to include 1GB+ files in your Docker images.

See [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) for detailed VPS instructions.

---

## 🛠 Services
- **Batik Service (Port 8001)**: Search, Motif Classification, Type Classification.
- **Fashion Service (Port 8002)**: Segmentation, Blending, Recommendation.

---

## 📄 Documentation
- [Models Requirements](./models/README.md)
- [Checkpoints Requirements](./checkpoints/README.md)
- [Deployment Guide](./DEPLOYMENT_GUIDE.md)
