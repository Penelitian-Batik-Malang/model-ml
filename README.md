# Batik Malang — Text-to-Image Retrieval

Sistem pencarian gambar batik Malang berdasarkan deskripsi teks Bahasa Indonesia.
Pengguna cukup menulis deskripsi seperti *"batik warna coklat dengan motif bunga"*,
lalu sistem mengembalikan gambar batik paling relevan dari koleksi.

## Arsitektur

**Model:** ConvNeXt-Small (image encoder) + IndoBERT (text encoder) + Projection Head 768d,
dilatih dengan contrastive loss agar embedding teks dan citra yang relevan berada berdekatan
dalam satu ruang embedding bersama. Retrieval dilakukan dengan cosine similarity.

Repo ini berisi **service ML (FastAPI)**. Untuk integrasi ke aplikasi Laravel, lihat bagian
[Integrasi Laravel](#integrasi-laravel) di bawah.

---

## Instalasi

```bash
# (opsional) buat virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# install semua dependency
pip install -r requirements.txt
```

## Konfigurasi

Semua konfigurasi ada di bagian atas `main.py`:

```python
MODEL_PATH     = "Modelmultimodal.pt"        # path checkpoint model
NPZ_PATH       = "gallery.npz"               # path file embedding gallery
CSV_PATH       = "dataset_caption.csv"       # CSV caption (fallback build gallery)
IMAGE_ROOT     = r"Data_Untuk_Clustering"    # folder dataset: IMAGE_ROOT/<kategori>/<file>
TOKENIZER_NAME = "indobenchmark/indobert-base-p1"
MAX_TEXT_LEN   = 64
IMG_SIZE       = 224
PROJ_DIM       = 768
BATCH_SIZE     = 32
DEFAULT_TOP_K  = 5
```

Sesuaikan path-path ini dengan lokasi file di komputer kamu sebelum menjalankan service.

## Menjalankan Service

Saat startup, service akan:
1. Memuat model dari `MODEL_PATH` (checkpoint dengan key `model_state_dict`).
2. Memuat tokenizer IndoBERT.
3. Memuat gallery embedding dari `gallery.npz` — **kalau file ini belum ada**, service
   otomatis membangunnya dari `dataset_caption.csv` + gambar di `Data_Untuk_Clustering/`
   (proses ini sekali saja, bisa memakan waktu tergantung jumlah gambar & CPU/GPU —
   lihat [`benchmark_cpu.py`](#inspect_npzpy--benchmark_cpupy) untuk estimasi).

---

## API Endpoints

### `GET /`
Health check dasar.
```json
{ "status": "ok", "service": "Batik Image Retrieval API", "gallery": 1259, "device": "cpu" }
```

### `GET /info`
Statistik gallery (jumlah gambar, kategori, dimensi embedding).
```json
{
  "total_images": 1259,
  "total_categories": 46,
  "embedding_dim": 768,
  "categories": { "Acha Mahakala": 12, "...": "..." }
}
```

### `POST /search-text`
Cari gambar berdasarkan deskripsi teks.

**Request:**
```json
{ "query": "batik parang warna coklat tua", "top_k": 5 }
```

**Response:**
```json
{
  "query": "batik parang warna coklat tua",
  "top_k": 5,
  "results": [
    {
      "rank": 1,
      "score": 0.8421,
      "image_id": 512,
      "category": "Batik Parang",
      "filename": "IMG_1234.JPG",
      "image_url": "http://127.0.0.1:8001/Data_Untuk_Clustering/512"
    }
  ]
}
```

### `GET /Data_Untuk_Clustering/{image_id}`
Mengambil file gambar berdasarkan `image_id` dari hasil pencarian.

---

## Utility: `inspect_npz.py` & `benchmark_cpu.py`

**Cek isi gallery embedding** (jumlah gambar, kategori, validasi normalisasi):
```bash
python inspect_npz.py gallery.npz
python inspect_npz.py gallery.npz --show 15
python inspect_npz.py gallery.npz --category "Batik Sekar Jagad"
```

**Estimasi waktu ekstraksi embedding di CPU lokal** (sebelum menjalankan build gallery
penuh, terutama kalau dataset besar):
```bash
python benchmark_cpu.py --n_samples 20 --batch_size 8 --total_images 1259
```

---

## Troubleshooting

| Gejala | Penyebab | Solusi |
|---|---|---|
| `ModuleNotFoundError: No module named 'pandas'` | Dependency belum ter-install | `pip install -r requirements.txt` |
| Gambar tidak muncul di frontend (broken image) | `image_url` berupa path relatif, bukan absolute URL | Sudah diperbaiki di `main.py` — pastikan pakai versi yang membangun `base_url` dari `request.base_url` |
| `RouteNotFoundException: Route [search] not defined` | Nama route Laravel tidak cocok dengan yang dipanggil di Blade (`route('search')`) | Samakan `->name(...)` di route dengan yang dipakai di Blade |
| Hasil pencarian tidak relevan / skor aneh | Embedding belum di-L2-normalize | Cek dengan `inspect_npz.py` — kolom "Rata-rata L2 norm" harus ≈ 1.0 |
| `KeyError` saat load checkpoint | Nama key checkpoint beda (`model_state` vs `model_state_dict`) | Pastikan checkpoint disimpan dengan key `model_state_dict`, sesuai `model.py` |

---

## Catatan Pengembangan

- Proses build gallery dari CSV berjalan **sinkron saat startup** — untuk dataset besar,
  server akan terlihat "menggantung" sesaat sebelum siap menerima request. Ini normal.
- Setelah `gallery.npz` tersimpan, restart berikutnya jauh lebih cepat karena tinggal load file.
- CORS diaktifkan untuk semua origin (`allow_origins=["*"]`) — sesuaikan ke domain spesifik
  saat deployment production.
