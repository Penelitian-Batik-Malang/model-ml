# VPS Deployment Guide

This guide explains how to deploy the Batik ML API to a VPS (Virtual Private Server) using Docker.

## 🏗 Directory Structure on VPS

We recommend placing the project in `/var/www/batik-ml-api`.

```text
/var/www/batik-ml-api/
├── .git/
├── models/         <-- Manually upload/download large .h5, .pt, .npy files here
├── checkpoints/    <-- Manually upload/download segmentation checkpoints here
├── data/           <-- Store .npz feature files here
├── src/            <-- Git repository content
└── docker-compose.yml
```

## 📥 1. Handling Large Files

Do not try to `git push` these files. Instead:

### Option A: Manual Upload (SCP)
```bash
# From your local machine
scp -r ./models user@vps-ip:/var/www/batik-ml-api/
scp -r ./checkpoints user@vps-ip:/var/www/batik-ml-api/
```

### Option B: Direct Download (Recommended)
Store your files on S3 (e.g., IDCloudHost, AWS) or a public link, then on the VPS:
```bash
cd /var/www/batik-ml-api/models
wget https://your-storage-link.com/model_ConvNextTiny_original_all.pt
```

## 🚀 2. Deployment Steps

### Step 1: Clone the Repository
```bash
git clone https://github.com/Penelitian-Batik-Malang/model-ml.git .
```

### Step 2: Prepare Large Files
Ensure `models/`, `checkpoints/`, and `data/` are populated as described in their respective `README.md` files.

### Step 3: Configure Environment
Create a `.env` file if you need to override ports or S3 URLs.

### Step 4: Run with Docker Compose
```bash
docker-compose up -d --build
```

## 🐳 Docker Volume Mapping
Our `docker-compose.yml` uses volume mapping to ensure the container can see the large files on the host:

```yaml
volumes:
  - ./models:/app/models
  - ./checkpoints:/app/checkpoints
  - ./data:/app/data
```

## 🔒 3. Security Notes
- Ensure ports `8001` and `8002` are only accessible to your main web application (use a Firewall like `ufw`).
- If using a Reverse Proxy (Nginx), set up SSL.

## 🛠 Troubleshooting
- **Out of Memory**: ML models require significant RAM. Ensure your VPS has at least 4GB-8GB RAM + Swap.
- **File Not Found**: Double check that the filenames in your `models/` folder match exactly with `config.py`.
