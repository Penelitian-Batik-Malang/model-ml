import os
import json
import time
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array

# Limit threads for TF API
try:
    cpu = max(1, (os.cpu_count() or 2) // 2)
    tf.config.threading.set_intra_op_parallelism_threads(cpu)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass

from config import BASE_DIR

DEFAULT_MODEL_PATH = str(BASE_DIR / "models" / "augmentTest_batik_cnn_pararel_elu3.h5")
DEFAULT_LABEL_PATH = str(BASE_DIR / "models" / "label_mapping_pararelEluAugment3.json")

model = None
label_map = {}
ENABLE_META = True

def preprocess_image(image_pil: Image.Image, target_size=(224, 224)):
    image_pil = image_pil.convert("RGB")
    arr = img_to_array(image_pil)
    arr = tf.image.resize(arr, target_size).numpy()
    arr = arr / 255.0
    return np.expand_dims(arr.astype("float32", copy=False), axis=0)

def _predict_numpy(inp_bhwc):
    t = tf.convert_to_tensor(inp_bhwc, dtype=tf.float32)
    out = model(t, training=False)
    return out.numpy()[0]

def _topk(preds, k=3):
    idx = preds.argsort()[-k:][::-1]
    out = []
    for i in idx:
        out.append({
            "class_name": label_map.get(int(i), "?"),
            "confidence": float(preds[i]),
            "percentage": f"{round(float(preds[i]) * 100, 2)}%"
        })
    return out

def initialize_model(model_path=None, label_path=None):
    global model, label_map
    model_path = model_path or DEFAULT_MODEL_PATH
    label_path = label_path or DEFAULT_LABEL_PATH

    model = load_model(model_path)

    with open(label_path, "r") as f:
        raw = json.load(f)
        if all(isinstance(v, (int, str)) for v in raw.values()):
            label_map = {int(v): k for k, v in raw.items()}
        else:
            label_map = {int(k): str(v) for k, v in raw.items()}

    dummy = np.zeros((1, 224, 224, 3), dtype="float32")
    _ = _predict_numpy(dummy)
    print(f"[Motif] Model loaded & warmed up from {model_path}")

def predict_motif(image_pil: Image.Image) -> dict:
    if model is None:
        initialize_model()

    t0 = time.time()
    inp = preprocess_image(image_pil)
    t1 = time.time()
    preds = _predict_numpy(inp)
    t2 = time.time()

    result = {"predictions": _topk(preds, 3)}
    if ENABLE_META:
        result["prep_time_ms"] = round((t1 - t0) * 1000, 2)
        result["infer_time_ms"] = round((t2 - t1) * 1000, 2)
        result["total_time_ms"] = round((t2 - t0) * 1000, 2)
    return result

def get_motif_labels() -> list:
    if not label_map:
        initialize_model()
    return list(label_map.values())
