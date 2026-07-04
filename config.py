from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent

# ==========================================
# 1. FASHIONPEDIA SEGMENTATION CONFIGURATION
# ==========================================
# Using exported saved model under `models/` (no TPU repo required)
FASHION_SAVED_MODEL_DIR = BASE_DIR / "models"

# ==========================================
# 2. FASHION RECOMMENDATION (CBIR) CONFIG
# ==========================================
# URL public S3 IDCloudHost untuk gambar hasil rekomendasi warna
S3_BATIK_FASHION_ROOT_URL = "https://is3.cloudhost.id/color-dominant-batik"
# Updated filename after data refactor
FASHION_CBIR_FEATURES_NPZ = BASE_DIR / "data" / "fitur_warna_dominan_batik_rekomendasi_by_fashion.npz"

# ==========================================
# 3. GENERAL BATIK SEARCH CONFIGURATION
# ==========================================
BATIK_SEARCH_FEATURES_NPY = BASE_DIR / "models" / "features_768_features.npy"
BATIK_SEARCH_KMEANS_MODEL = BASE_DIR / "models" / "features_768_kmeans_model.pkl"
BATIK_SEARCH_INDEXED_DB_CSV = BASE_DIR / "models" / "features_768_indexed_database.csv"

# ==========================================
# 4. BATIK MOTIF CLASSIFICATION CONFIG
# ==========================================
BATIK_MOTIF_MODEL_H5 = BASE_DIR / "models" / "augmentTest_batik_cnn_pararel_elu3.h5"
BATIK_MOTIF_LABEL_JSON = BASE_DIR / "models" / "label_mapping_pararelEluAugment3.json"

# ==========================================
# 5. BATIK TYPE CLASSIFICATION CONFIG
# ==========================================
BATIK_TYPE_MODEL_PT = BASE_DIR / "models" / "model_ConvNextTiny_original_all.pt"

# ==========================================
# 6. SERVER CONFIGURATION
# ==========================================
SERVICE_PORT = int(os.getenv("PORT", 8000))
