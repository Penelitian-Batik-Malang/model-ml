from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(os.getenv("BATIK_MODELS_DIR", BASE_DIR / "models"))
CHECKPOINTS_DIR = Path(os.getenv("BATIK_CHECKPOINTS_DIR", BASE_DIR / "checkpoints"))
DATA_DIR = Path(os.getenv("BATIK_DATA_DIR", BASE_DIR / "data"))

# ==========================================
# 1. FASHIONPEDIA SEGMENTATION CONFIGURATION
# ==========================================
TPU_DIR = BASE_DIR / "tpu"
FASHION_DETECTION_DIR = TPU_DIR / "models" / "official" / "detection"
FASHION_PROJECT_DIR = FASHION_DETECTION_DIR / "projects" / "fashionpedia"

FASHION_CHECKPOINT_PATH = CHECKPOINTS_DIR / "fashionpedia-r50-fpn" / "model.ckpt"
FASHION_LABEL_MAP_PATH = FASHION_PROJECT_DIR / "dataset" / "fashionpedia_label_map.csv"
FASHION_CONFIG_FILE = FASHION_PROJECT_DIR / "configs" / "yaml" / "r50fpn_amrcnn.yaml"
FASHION_INFERENCE_SCRIPT = FASHION_DETECTION_DIR / "inference_fashionpedia.py"

# ==========================================
# 2. FASHION RECOMMENDATION (CBIR) CONFIG
# ==========================================
# URL public S3 IDCloudHost untuk gambar hasil rekomendasi warna
S3_BATIK_FASHION_ROOT_URL = "https://is3.cloudhost.id/color-dominant-batik" 
FASHION_CBIR_FEATURES_NPZ = DATA_DIR / "batik_skenario_3_warna.npz"

# ==========================================
# 3. GENERAL BATIK SEARCH CONFIGURATION
# ==========================================
BATIK_SEARCH_FEATURES_NPY = MODELS_DIR / "features_768_features.npy"
BATIK_SEARCH_KMEANS_MODEL = MODELS_DIR / "features_768_kmeans_model.pkl"
BATIK_SEARCH_INDEXED_DB_CSV = MODELS_DIR / "features_768_indexed_database.csv"

# ==========================================
# 4. BATIK MOTIF CLASSIFICATION CONFIG
# ==========================================
BATIK_MOTIF_MODEL_H5 = MODELS_DIR / "augmentTest_batik_cnn_pararel_elu3.h5"
BATIK_MOTIF_LABEL_JSON = MODELS_DIR / "label_mapping_pararelEluAugment3.json"

# ==========================================
# 5. BATIK TYPE CLASSIFICATION CONFIG
# ==========================================
BATIK_TYPE_MODEL_PT = MODELS_DIR / "model_ConvNextTiny_original_all.pt"

# ==========================================
# 6. SERVER CONFIGURATION
# ==========================================
SERVICE_PORT = int(os.getenv("PORT", 8000))
