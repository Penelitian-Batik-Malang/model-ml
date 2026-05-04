from pathlib import Path

from app.config.settings import settings

TPU_DIR = Path(settings.TPU_PATH)
FASHION_DETECTION_DIR = TPU_DIR / "models" / "official" / "detection"
FASHION_PROJECT_DIR = FASHION_DETECTION_DIR / "projects" / "fashionpedia"

FASHION_CHECKPOINT_PATH = Path(settings.CHECKPOINTS_PATH) / "fashionpedia-r50-fpn" / "model.ckpt"
FASHION_LABEL_MAP_PATH = FASHION_PROJECT_DIR / "dataset" / "fashionpedia_label_map.csv"
FASHION_CONFIG_FILE = FASHION_PROJECT_DIR / "configs" / "yaml" / "r50fpn_amrcnn.yaml"
FASHION_INFERENCE_SCRIPT = FASHION_DETECTION_DIR / "inference_fashionpedia.py"

FASHION_CBIR_FEATURES_NPZ = Path(settings.DATA_PATH) / "batik_skenario_3_warna.npz"
