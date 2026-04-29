# Batik & Fashion ML API

Machine Learning service terpadu untuk Klasifikasi Batik, Segmentasi Fashion, dan Rekomendasi Batik — dijalankan sebagai **satu service** berbasis Python 3.7 / TF 1.15 / PyTorch 1.12.

## Prerequisites

- Python 3.7
- Docker & Docker Compose (opsional)
- File model besar — lihat `models/README.md` dan `checkpoints/README.md`

---

## Instalasi Lokal (Virtual Environment)

```bash
# Buat dan aktifkan venv Python 3.7
py -3.7 -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Jalankan server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

> **Catatan**: `requirements.txt` adalah hasil `pip freeze`. Pastikan file disimpan sebagai **UTF-8** sebelum digunakan untuk Docker build.

---

## Menjalankan dengan Docker

```bash
docker-compose up --build
```

Service tersedia di `http://localhost:8000`.

---

## Endpoints

| Prefix | Fungsi |
|--------|--------|
| `POST /fashion/segment` | Segmentasi pakaian (Mask R-CNN, Fashionpedia) |
| `POST /fashion/blend-manual` | Blending batik ke pakaian (upload manual) |
| `POST /fashion/blend-cbir` | Blending batik ke pakaian (rekomendasi warna) |
| `POST /search/general` | Pencarian batik serupa (CBIR ConvNeXt Small) |
| `POST /detection/motif` | Klasifikasi motif batik (CNN parallel) |
| `POST /detection/type` | Klasifikasi jenis batik — Tulis vs Cap (ConvNeXt Tiny) |
| `GET  /health` | Health check |
| `GET  /docs` | Swagger UI (auto-generated) |

---

## Manajemen File Besar (Model & Checkpoint)

Model dan checkpoint **tidak disertakan** di repository karena ukurannya.

**Struktur direktori yang dibutuhkan:**
```
ml-api/
├── models/          # .h5, .pt, .npy, .pkl, .csv, .json
├── checkpoints/     # fashionpedia-r50-fpn/model.ckpt.*
└── data/            # batik_skenario_3_warna.npz
```

**Development**: Letakkan file di folder masing-masing sesuai `config.py`.

**VPS/Production**:
1. Simpan di cloud storage (S3, Google Drive, dsb.)
2. Download langsung ke VPS dengan `wget` atau `curl`
3. Mount folder sebagai Docker Volume agar tidak masuk ke image

Lihat [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) untuk panduan VPS lengkap.

---

## Stack Teknis

| Komponen | Versi |
|----------|-------|
| Python | 3.7 |
| TensorFlow | 1.15.0 (fashion inference + motif CNN) |
| PyTorch | 1.12.1 + torchvision 0.13.1 (batik search & type) |
| FastAPI | 0.68.x |
| Port | 8000 (single service) |
