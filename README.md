# ML API — Dokumentasi Final
> Versi final setelah migrasi ke arsitektur microservice Docker.
> Dokumen ini adalah referensi utama untuk integrasi Laravel dan pengembangan selanjutnya.
> Link file yang dibutuhkan (MODELS, DATA, CHECKPOINTS) : https://drive.google.com/drive/folders/1-BeWhnrE5rsIWSqC3D1UuWjFy4COIupF?usp=drive_link
---

## Arsitektur Keseluruhan

```
┌─────────────────────────────────────────────────────────────┐
│                     Laravel (PHP)                           │
│              app/Services/MLService.php                     │
└───────────────────┬─────────────────────┬───────────────────┘
                    │                     │
          HTTP POST │                     │ HTTP POST
                    ▼                     ▼
┌───────────────────────┐   ┌───────────────────────────────┐
│   Batik Service       │   │   Fashion Service             │
│   Port 8001           │   │   Port 8002                   │
│   Python 3.9          │   │   Python 3.7 / TF 1.15        │
│   PyTorch + sklearn   │   │   TensorFlow TPU / Fashionpedia│
└───────────────────────┘   └───────────────────────────────┘
         │                             │
    /search/general              /fashion/segment
    /detection/motif             /fashion/blend-manual
    /detection/type              /fashion/blend-cbir
                                 /fashion/reset-session
                                 /fashion/session/{id}
```

---

## 1. Struktur Folder ML API

```
ml-api/
├── Dockerfile.batik              ← image Python 3.9 (PyTorch)
├── Dockerfile.fashion            ← image Python 3.7 (TF 1.15)
├── docker-compose.yml
├── config.py                     ← semua path & konstanta
├── main_batik.py                 ← entry point Batik Service
├── main_fashion.py               ← entry point Fashion Service
│
├── routes/
│   ├── search_routes.py          ← /search/*
│   ├── detection_routes.py       ← /detection/*
│   └── fashion_segmentation_routes.py  ← /fashion/*
│
├── services/
│   ├── batik_search_engine.py    ← CBIR KMeans (ConvNeXt feature)
│   ├── motif_classification_engine.py  ← CNN motif batik
│   ├── type_classification_engine.py   ← ConvNeXt Tiny cap/tulis
│   ├── fashion_segmentation_engine.py  ← Fashionpedia + ambil_model_busana()
│   ├── fashion_recommendation_engine.py ← CBIR warna (KMeans CIELAB)
│   └── fashion_blending_engine.py      ← multiply_blend()
│
├── models/
│   ├── features_768_features.npy
│   ├── features_768_kmeans_model.pkl
│   ├── features_768_indexed_database.csv   ← kolom: path_gdrive,path_s3,label,label_int,cluster
│   ├── augmentTest_batik_cnn_pararel_elu3.h5
│   ├── label_mapping_pararelEluAugment3.json
│   └── model_ConvNextTiny_original_all.pt
│
├── data/
│   └── batik_skenario_3_warna.npz  ← CBIR warna fashion (1244 item)
│
├── checkpoints/
│   └── fashionpedia-r50-fpn/model.ckpt
│
└── sessions/                       ← auto-created per session UUID
    └── {session_id}/
        ├── fashion.jpg
        ├── current.jpg
        ├── result.npy
        └── meta.json
```

---

## 2. Deploy & Menjalankan Services

```bash
# Pertama kali / setelah perubahan Dockerfile
docker-compose build --no-cache

# Jalankan semua services
docker-compose up -d

# Restart satu service saja
docker-compose restart batik-service
docker-compose restart fashion-service

# Lihat log realtime
docker-compose logs -f batik-service
```

### Catatan `docker-compose.yml`
- Volume `torch_cache:/root/.cache/torch` → ConvNeXt Small (~192MB) tidak download ulang tiap restart
- ConvNeXt juga sudah di-**prebake** ke Docker image saat `build`, volume hanya sebagai safety net
- Jika model baru ditambah ke `models/` — **tidak perlu rebuild** karena volume `.:/app` sudah mount langsung

---

## 3. Batik Service — Port 8001

### `POST /search/general` — Pencarian Batik Serupa (CBIR Gambar)

**Fungsi:** Query gambar batik → temukan batik paling mirip menggunakan ConvNeXt feature + KMeans cluster.

**Input:** `multipart/form-data`
| Field | Tipe | Keterangan |
|-------|------|-----------|
| `file` | `UploadFile` | Gambar batik (JPG/PNG) |

*Alternatif: `application/json` dengan field `image` berisi `data:image/jpeg;base64,...`*

**Response:**
```json
{
  "success": true,
  "cluster_id": 3,
  "message": "Found 10 similar images in cluster 3",
  "results": [
    {
      "path_s3": "original/Topeng Gandring Wirasena/Topeng_1.JPG",
      "label": "Topeng Gandring Wirasena",
      "cluster": 3,
      "similarity": 0.9823
    }
  ]
}
```

> **`path_s3`** adalah path relatif di S3. Gabungkan dengan base URL S3 untuk dapat URL lengkap.

---

### `POST /detection/motif` — Klasifikasi Motif Batik

**Input:** `multipart/form-data` atau JSON base64
**Response:**
```json
{ "label": "Kawung", "confidence": 0.9234, "percentage": "92.34%" }
```

---

### `POST /detection/type` — Klasifikasi Jenis Batik (Cap/Tulis)

**Input:** `multipart/form-data` atau JSON base64
**Response:**
```json
{ "label": "Batik Tulis", "confidence": 0.8711, "percentage": "87.11%" }
```

---

### `GET /detection/motif/labels` & `GET /detection/type/labels`
```json
["Kawung", "Parang", "Mega Mendung", ...]
```

---

## 4. Fashion Service — Port 8002

### `POST /fashion/segment` — Segmentasi + CBIR Warna

**Fungsi:** Upload foto fashion → segmentasi bagian pakaian + rekomendasi batik berdasarkan warna dominan.

**Input:** `multipart/form-data`
| Field | Tipe | Keterangan |
|-------|------|-----------|
| `image` | `UploadFile` | Foto fashion (JPG/PNG) |

**Proses internal:**
1. Simpan ke `sessions/{uuid}/fashion.jpg` + `current.jpg`
2. Jalankan Fashionpedia inference → `result.npy`
3. **`ambil_model_busana()`** — pilih SATU label upper body dengan piksel terbesar *(sesuai skripsi)*
4. Ekstrak warna CIELAB → KMeans k=3 → `query_centroids`
5. CBIR retrieval ke `batik_skenario_3_warna.npz` (1244 item, Euclidean+Hungarian)
6. Encode semua mask bagian pakaian sebagai base64 RGBA PNG

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "image_size": { "w": 640, "h": 480 },
  "parts": {
    "shirt": [
      {
        "index": 0,
        "bbox": { "x": 100, "y": 60, "w": 200, "h": 180 },
        "mask_b64": "iVBORw0KGgo...",
        "area": 14500,
        "score": 0.94
      }
    ],
    "sleeve": [
      { "index": 0, "bbox": {...}, "mask_b64": "...", "area": 3200, "score": 0.87 }
    ]
  },
  "cbir": {
    "selected_label": "shirt",
    "selected_class_id": 1,
    "pixel_count": 14500,
    "query_centroids": [[88.6, -3.1, 4.4], [38.1, 4.8, -1.9], [73.8, -2.9, 3.2]],
    "top_5": [
      {
        "rank": 1,
        "filename": "https://is3.cloudhost.id/color-dominant-batik/biru/IMG_001.jpg",
        "label": "biru",
        "jarak": 2.34,
        "thumbnail_b64": ""
      }
    ],
    "top_10": [...],
    "top_15": [...]
  }
}
```

---

### `POST /fashion/blend-manual` — Tempel Batik dari Upload

**Input:** `multipart/form-data`
| Field | Tipe | Default | Keterangan |
|-------|------|---------|-----------|
| `session_id` | `str` | — | UUID dari `/segment` |
| `part` | `str` | — | `"shirt"`, `"sleeve"`, `"dress"`, dll |
| `instance_index` | `int` | `0` | Jika multi-instance |
| `batik` | `UploadFile` | — | File gambar batik |

**Response:** `{ "image_b64": "base64 JPEG" }`

---

### `POST /fashion/blend-cbir` — Tempel Batik dari Rekomendasi

**Input:** `multipart/form-data`
| Field | Tipe | Default | Keterangan |
|-------|------|---------|-----------|
| `session_id` | `str` | — | UUID dari `/segment` |
| `part` | `str` | — | Part yang mau di-blend |
| `instance_index` | `int` | `0` | — |
| `batik_filename` | `str` | — | URL S3 dari `cbir.top_k[n].filename` |

**Response:** `{ "image_b64": "base64 JPEG" }`

---

### `POST /fashion/reset-session` — Reset ke Foto Asli

**Input:** `{ "session_id": "uuid" }`
**Response:** `{ "image_b64": "base64 JPEG foto asli" }`

---

### `GET /fashion/session/{session_id}` — Status Sesi

```json
{
  "session_id": "uuid",
  "current_image_b64": "base64 JPEG",
  "parts_detected": ["shirt", "sleeve"],
  "parts_blended": ["sleeve"]
}
```

---

## 5. Label Parts yang Dikenali

### Upper Body — untuk CBIR & blending
| Class ID | Label | Class ID | Label |
|----------|-------|----------|-------|
| 1 | `shirt` | 10 | `dress` |
| 2 | `t-shirt` | 11 | `jumpsuit` |
| 3 | `sweater` | 12 | `suit` |
| 4 | `cardigan` | 13 | `coat` |
| 5 | `jacket` | | |
| 6 | `vest` | | |

### Part Detail — untuk blending per-bagian
| Class ID | Label | Class ID | Label |
|----------|-------|----------|-------|
| 28 | `hood` | 32 | `sleeve` |
| 29 | `collar` | 33 | `pocket` |
| 30 | `lapel` | 34 | `neckline` |
| 31 | `epaulette` | | |

---

## 6. Integrasi Laravel

### `config/services.php`
```php
'ml_api' => [
    'batik_url'   => env('ML_BATIK_URL',   'http://localhost:8001'),
    'fashion_url' => env('ML_FASHION_URL', 'http://localhost:8002'),
],
```

### `.env`
```env
ML_BATIK_URL=http://batik-service:8001
ML_FASHION_URL=http://fashion-service:8002
```

> Jika Laravel dan ML API dijalankan di server yang sama dalam satu Docker network,
> gunakan nama service (`batik-service`, `fashion-service`).
> Jika diakses dari host: `http://127.0.0.1:8001`.

### `app/Services/MLService.php`
```php
<?php
namespace App\Services;

use Illuminate\Support\Facades\Http;

class MLService
{
    private string $batikUrl;
    private string $fashionUrl;

    public function __construct()
    {
        $this->batikUrl   = config('services.ml_api.batik_url');
        $this->fashionUrl = config('services.ml_api.fashion_url');
    }

    // ─── BATIK SERVICE ───────────────────────────────────────

    public function searchGeneralBatik(string $imagePath): array
    {
        return Http::timeout(60)
            ->attach('file', file_get_contents($imagePath), basename($imagePath))
            ->post("{$this->batikUrl}/search/general")
            ->json();
    }

    public function detectMotif(string $imagePath): array
    {
        return Http::timeout(60)
            ->attach('file', file_get_contents($imagePath), basename($imagePath))
            ->post("{$this->batikUrl}/detection/motif")
            ->json();
    }

    public function detectType(string $imagePath): array
    {
        return Http::timeout(60)
            ->attach('file', file_get_contents($imagePath), basename($imagePath))
            ->post("{$this->batikUrl}/detection/type")
            ->json();
    }

    // ─── FASHION SERVICE ──────────────────────────────────────

    public function segmentFashion(string $imagePath): array
    {
        return Http::timeout(300)  // inference bisa 60–180 detik
            ->attach('image', file_get_contents($imagePath), basename($imagePath))
            ->post("{$this->fashionUrl}/fashion/segment")
            ->json();
    }

    public function blendManual(string $sessionId, string $part, string $batikPath, int $index = 0): array
    {
        return Http::timeout(60)
            ->attach('batik', file_get_contents($batikPath), basename($batikPath))
            ->post("{$this->fashionUrl}/fashion/blend-manual", [
                'session_id'     => $sessionId,
                'part'           => $part,
                'instance_index' => $index,
            ])
            ->json();
    }

    public function blendFromCbir(string $sessionId, string $part, string $batikS3Url, int $index = 0): array
    {
        return Http::timeout(60)
            ->asForm()
            ->post("{$this->fashionUrl}/fashion/blend-cbir", [
                'session_id'     => $sessionId,
                'part'           => $part,
                'instance_index' => $index,
                'batik_filename' => $batikS3Url,
            ])
            ->json();
    }

    public function resetSession(string $sessionId): array
    {
        return Http::timeout(30)
            ->post("{$this->fashionUrl}/fashion/reset-session", [
                'session_id' => $sessionId,
            ])
            ->json();
    }

    public function getSession(string $sessionId): array
    {
        return Http::timeout(30)
            ->get("{$this->fashionUrl}/fashion/session/{$sessionId}")
            ->json();
    }
}
```

---

## 7. Algoritma Inti (Referensi Skripsi)

### `ambil_model_busana()` — Seleksi Upper Body untuk CBIR
```
Input : raw_result dict (dari load_segmentation_result)
Output: (final_mask, label_upper_body, idx_upper_body) | None

Algoritma:
1. Iterasi semua class yang terdeteksi
2. Jika class_id ∈ {1,2,3,4,5,6,10,11,12,13}:
   - Decode RLE mask → binary_mask (H×W)
   - Hitung pixel_count = np.sum(binary_mask)
   - Jika pixel_count > max_sebelumnya → simpan sebagai kandidat
3. Return mask label dengan piksel TERBESAR
```
> Menggantikan `_get_mask_union()` yang sebelumnya melakukan union outer/inner.
> Versi skripsi lebih sederhana dan fokus: satu garment dominan = representasi pakaian.

### `multiply_blend()` — Multiply Blending
```
Input : mask (H×W), fashion_rgb (H×W×3), batik_rgb (H×W×3)
Output: result (H×W×3)

Algoritma:
1. Hitung bounding box dari piksel mask (tight bbox dari mask pixel)
2. Crop ROI fashion dan mask sesuai bbox
3. Resize batik ke ukuran bbox (INTER_LANCZOS4)
4. Konversi ROI fashion → grayscale
5. shading_map = gray / (mean_brightness + 1e-6), clip [0, 2.0]
6. batik_float[:,:,i] *= shading_map  (per channel RGB)
7. Tempel hanya pada piksel mask == 1
8. Return fashion dengan area batik tertempel
```
> Clip [0, 2.0] adalah peningkatan dari skripsi untuk mencegah overexposure area terang.

### CBIR Warna — Euclidean + Hungarian
```
Input : query_centroids (3×3 LAB), db_centroids (3×3 LAB per batik)
Output: jarak (float)

Algoritma:
1. cost_matrix[i][j] = ||query[i] - db[j]||₂
2. Hungarian algorithm → optimal assignment
3. jarak = sum(cost[row,col]) / 3
→ Sort ascending → ambil top-5, top-10, top-15
```

---

## 8. Panduan Penambahan Fitur

### A. Endpoint Baru di Service yang Sudah Ada

1. Buat `services/nama_engine.py`
2. Tambah route di `routes/nama_routes.py`
3. Register di `main_batik.py` atau `main_fashion.py`
4. Jika ada dependency baru → tambah ke `requirements-batik.txt` / `requirements-fashion.txt`
5. `docker-compose build --no-cache batik-service` (rebuild hanya jika ada dependency baru)

### B. Membuat Service Baru (Environment Terpisah)

Diperlukan jika:
- Butuh Python versi berbeda (saat ini: 3.7 untuk TF1.15, 3.9 untuk PyTorch)
- Ada konflik library yang tidak bisa diselesaikan
- Fitur baru sangat resource-heavy dan tidak mau ganggu service lain

**Langkah:**

1. Buat `ml-api/Dockerfile.newservice`
2. Buat `ml-api/main_newservice.py`
3. Tambah di `docker-compose.yml`:
```yaml
new-service:
  build:
    context: .
    dockerfile: Dockerfile.newservice
  ports:
    - "8003:8003"
  volumes:
    - .:/app
  environment:
    - PORT=8003
  command: uvicorn main_newservice:app --host 0.0.0.0 --port 8003 --reload
```
4. Tambah di `config/services.php` Laravel:
```php
'new_service_url' => env('ML_NEW_SERVICE_URL', 'http://localhost:8003'),
```
5. Tambah method di `MLService.php`

### C. Menambah Model File Baru

1. Taruh file model di `ml-api/models/`
2. Tambah path di `config.py`:
```python
NEW_MODEL_PATH = BASE_DIR / "models" / "my_new_model.pt"
```
3. **Tidak perlu rebuild Docker** — volume `.:/app` sudah mount langsung

---

## 9. Troubleshooting

| Masalah | Penyebab | Solusi |
|---------|----------|--------|
| `400 Bad Request` di `/search/general` | `KeyError: 'filename'` — kolom CSV adalah `path_s3` bukan `filename` | **Sudah diperbaiki** — kode sekarang pakai `row['path_s3']` |
| Download 192MB ConvNeXt tiap restart | Model cache tidak persist | **Sudah diperbaiki** — prebake di Dockerfile + volume `torch_cache` |
| `InconsistentVersionWarning` KMeans | PKL dibuat sklearn 1.6.1, service pakai 1.3.0 | Re-generate PKL dengan sklearn 1.3.0 **atau** ubah `requirements-batik.txt` ke `scikit-learn==1.6.1` |
| CBIR tidak muncul di response `/segment` | `BATIK_DB` None — NPZ tidak ditemukan | Cek path `data/batik_skenario_3_warna.npz` di container |
| Session 404 | Session expire setelah 2 jam | Re-upload foto fashion, session baru dibuat otomatis |
| Fashionpedia timeout | Model besar, CPU only | Timeout sudah 600 detik; gunakan GPU di production |

---

## 10. Versi & Dependency Kunci

### Batik Service (Python 3.9)
```
torch==2.0.1
torchvision==0.15.2
scikit-learn==1.3.0      ← perhatikan kompatibilitas PKL
fastapi==0.103.1
```

### Fashion Service (Python 3.7)
```
tensorflow==1.15.0       ← WAJIB 1.15 untuk Fashionpedia TPU model
pycocotools
opencv-python
scipy, scikit-learn
```
