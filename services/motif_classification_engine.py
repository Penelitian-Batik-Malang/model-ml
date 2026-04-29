import os
import json
import time
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Concatenate,
)
from tensorflow.keras.models import Model

# Limit threads — wrapped untuk kompatibilitas TF 1.x dan TF 2.x
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


def _build_motif_model():
    """
    Rebuild arsitektur CNN motif secara manual agar kompatibel dengan TF 1.15.
    Arsitektur dibaca dari model_weights yang ada di h5 (Keras 3.x format).
    Layer name harus identik dengan nama di h5 supaya load_weights() cocok.

    Arsitektur: parallel CNN dua jalur (3x3 dan 5x5) → merge → head.
    Input: (224, 224, 3), Output: 39 kelas softmax.
    """
    inp = Input(shape=(224, 224, 3), name="input_layer_5")

    # Branch 1: 3x3 kernels
    x1 = Conv2D(32,  (3, 3), padding="same", activation="elu", name="conv2d_35")(inp)
    x1 = MaxPooling2D((2, 2), name="max_pooling2d_30")(x1)
    x1 = Conv2D(64,  (3, 3), padding="same", activation="elu", name="conv2d_36")(x1)
    x1 = MaxPooling2D((2, 2), name="max_pooling2d_31")(x1)
    x1 = Conv2D(128, (3, 3), padding="same", activation="elu", name="conv2d_37")(x1)
    x1 = MaxPooling2D((2, 2), name="max_pooling2d_32")(x1)

    # Branch 2: 5x5 kernels
    x2 = Conv2D(32,  (5, 5), padding="same", activation="elu", name="conv2d_38")(inp)
    x2 = MaxPooling2D((2, 2), name="max_pooling2d_33")(x2)
    x2 = Conv2D(64,  (5, 5), padding="same", activation="elu", name="conv2d_39")(x2)
    x2 = MaxPooling2D((2, 2), name="max_pooling2d_34")(x2)
    x2 = Conv2D(128, (5, 5), padding="same", activation="elu", name="conv2d_40")(x2)
    x2 = MaxPooling2D((2, 2), name="max_pooling2d_35")(x2)

    # Merge & head
    x = Concatenate(axis=-1, name="concatenate_5")([x1, x2])
    x = Conv2D(256, (3, 3), padding="same", activation="elu", name="conv2d_41")(x)
    x = GlobalAveragePooling2D(name="global_average_pooling2d_5")(x)
    x = Dense(128, activation="elu",      name="dense_10")(x)
    out = Dense(39, activation="softmax", name="dense_11")(x)

    return Model(inputs=inp, outputs=out)


def preprocess_image(image_pil: Image.Image, target_size=(224, 224)):
    arr = np.array(image_pil.convert("RGB").resize(target_size, Image.BILINEAR), dtype="float32")
    arr = arr / 255.0
    return np.expand_dims(arr, axis=0)


def _predict_numpy(inp_bhwc):
    return model.predict(inp_bhwc)[0]


def _topk(preds, k=3):
    idx = preds.argsort()[-k:][::-1]
    return [
        {
            "class_name": label_map.get(int(i), "?"),
            "confidence": float(preds[i]),
            "percentage": "{}%".format(round(float(preds[i]) * 100, 2)),
        }
        for i in idx
    ]


def initialize_model(model_path=None, label_path=None):
    global model, label_map
    model_path = model_path or DEFAULT_MODEL_PATH
    label_path = label_path or DEFAULT_LABEL_PATH

    # Bypass load_model() — model_config Keras 3.x tidak kompatibel dengan TF 1.15.
    # Rebuild arsitektur manual, lalu load hanya weights dari h5.
    model = _build_motif_model()
    model.load_weights(model_path)

    with open(label_path, "r") as f:
        raw = json.load(f)
        if all(isinstance(v, (int, str)) for v in raw.values()):
            label_map = {int(v): k for k, v in raw.items()}
        else:
            label_map = {int(k): str(v) for k, v in raw.items()}

    dummy = np.zeros((1, 224, 224, 3), dtype="float32")
    _ = _predict_numpy(dummy)
    print("[Motif] Model loaded & warmed up from " + model_path)


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
