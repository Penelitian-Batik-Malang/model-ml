"""
Patch model .h5 yang disimpan dengan TF 2.x / Keras 3.x agar bisa dimuat dengan TF 1.15.

Perubahan:
  1. class_name "Functional" -> "Model"
  2. "batch_shape" -> "batch_input_shape"  (InputLayer, TF 2.11+)
  3. Hapus key TF 2.x-only: "groups" (Conv2D), "module" (layer metadata)
  4. dtype DTypePolicy dict -> string  (Keras 3.x menyimpan dtype sbg objek)

Aman dijalankan berulang (idempotent).
Jalankan dengan: venv\\Scripts\\python patch_model_h5.py
"""
import json
import shutil
import h5py
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models" / "augmentTest_batik_cnn_pararel_elu3.h5"
BACKUP_PATH = MODEL_PATH.with_suffix(".h5.bak")

if not MODEL_PATH.exists():
    raise FileNotFoundError("Model tidak ditemukan: " + str(MODEL_PATH))

if not BACKUP_PATH.exists():
    shutil.copy2(MODEL_PATH, BACKUP_PATH)
    print("Backup dibuat: " + str(BACKUP_PATH))

# Key yang tidak dikenal TF 1.15 dan aman dihapus dari semua layer config
TF2_ONLY_KEYS = {
    "groups",           # Conv2D grouped convolution (TF 2.3+)
    "module",           # layer module path (Keras 3.x)
    "keepdims",         # GlobalPooling2D (Keras 2.x+, tidak ada di TF 1.15)
    "registered_name",  # custom layer registration (Keras 3.x)
    "build_config",     # build state (Keras 3.x)
    "compile_config",   # compile state (Keras 3.x)
}


def _extract_dtype(v):
    """Ekstrak nama dtype string dari berbagai format TF 2.x / Keras 3.x."""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        # Keras 3.x: {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}}
        cfg = v.get("config", {})
        if isinstance(cfg, dict) and "name" in cfg:
            return cfg["name"]
        # TF 2.x sederhana: {"class_name": "float32", "config": {}}
        if "class_name" in v:
            return v["class_name"]
    return "float32"  # fallback aman


def _patch(obj):
    """Traversal rekursif — terapkan semua fix ke setiap node dict/list."""
    if isinstance(obj, list):
        return [_patch(item) for item in obj]
    if not isinstance(obj, dict):
        return obj

    result = {}
    for k, v in obj.items():
        # Fix 3a: hapus key TF 2.x-only
        if k in TF2_ONLY_KEYS:
            continue
        # Fix 4: dtype dict -> string
        if k == "dtype" and isinstance(v, dict):
            result[k] = _extract_dtype(v)
            continue
        result[k] = _patch(v)
    return result


with h5py.File(str(MODEL_PATH), "r+") as f:
    raw = f.attrs["model_config"]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")

    config = json.loads(raw)

    # Fix 1: class_name "Functional" -> "Model"
    if config.get("class_name") == "Functional":
        config["class_name"] = "Model"
        print("Fix 1: Functional -> Model")

    # Fix 2: batch_shape -> batch_input_shape (string replace lebih andal utk nested)
    # Fix 3 & 4: via _patch() traversal rekursif
    config = _patch(config)

    patched = json.dumps(config)
    before = patched
    patched = patched.replace('"batch_shape":', '"batch_input_shape":')
    if patched != before:
        print("Fix 2: batch_shape -> batch_input_shape")

    config = json.loads(patched)

    f.attrs["model_config"] = json.dumps(config).encode("utf-8")

print("Fix 3+4: groups/module dihapus, dtype dinormalisasi ke string")
print("Selesai. File: " + str(MODEL_PATH))
