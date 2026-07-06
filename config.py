import os
 
# ── Path & file ────────────────────────────────────────────────────────────
MODEL_PATH  = os.getenv("MODEL_PATH", "Modelmultimodal.pt")
NPZ_PATH    = os.getenv("NPZ_PATH", "gallery.npz")
CSV_PATH    = os.getenv("CSV_PATH", "dataset_caption.csv")
IMAGE_ROOT  = os.getenv("IMAGE_ROOT", r"Data_Untuk_Clustering")
 
# ── Model ──────────────────────────────────────────────────────────────────
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", "indobenchmark/indobert-base-p1")
MAX_TEXT_LEN   = int(os.getenv("MAX_TEXT_LEN", 64))
IMG_SIZE       = int(os.getenv("IMG_SIZE", 224))
PROJ_DIM       = int(os.getenv("PROJ_DIM", 768))
 
# ── Inference ──────────────────────────────────────────────────────────────
BATCH_SIZE    = int(os.getenv("BATCH_SIZE", 32))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 5))
 
# ── Image serving ──────────────────────────────────────────────────────────
IMAGE_ENDPOINT_PREFIX = "/Data_Untuk_Clustering"
 
# ── CORS ───────────────────────────────────────────────────────────────────
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
