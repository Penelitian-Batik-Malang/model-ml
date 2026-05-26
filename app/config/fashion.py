from pathlib import Path

from app.config.settings import settings

# Saved model directory: use configured MODEL_PATH (falls back to repo models/)
FASHION_SAVED_MODEL_DIR = Path(settings.MODEL_PATH)

# CBIR features file: updated filename located under DATA_PATH
# Previously: batik_skenario_3_warna.npz
FASHION_CBIR_FEATURES_NPZ = Path(settings.DATA_PATH) / "fitur_warna_dominan_batik_rekomendasi_by_fashion.npz"
