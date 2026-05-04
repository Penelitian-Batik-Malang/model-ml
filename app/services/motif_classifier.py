import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array

logger = logging.getLogger(__name__)

# Reduce TensorFlow noise and force CPU usage
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


class MotifClassifier:
    """TensorFlow-based motif classifier."""

    def __init__(self, model_path: str, label_path: str) -> None:
        self.model_path = Path(model_path)
        self.label_path = Path(label_path)
        self.model = None
        self.label_map: Dict[int, str] = {}
        self.loaded = False

    def load(self) -> bool:
        """Load model and label map from disk."""
        if not self.model_path.exists():
            logger.error("Motif model file not found: %s", self.model_path)
            return False
        if not self.label_path.exists():
            logger.error("Motif label map not found: %s", self.label_path)
            return False

        try:
            self.model = load_model(str(self.model_path))

            with open(self.label_path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)

            # If format is {label_name: index}, flip to {index: label}
            if all(isinstance(v, (int, str)) for v in raw.values()):
                self.label_map = {int(v): str(k) for k, v in raw.items()}
            else:
                self.label_map = {int(k): str(v) for k, v in raw.items()}

            # Warm-up
            dummy = np.zeros((1, 224, 224, 3), dtype="float32")
            _ = self._predict_numpy(dummy)

            self.loaded = True
            logger.info("Motif model loaded: %s", self.model_path)
            return True
        except Exception as exc:
            logger.error("Failed to load motif model: %s", exc, exc_info=True)
            self.loaded = False
            return False

    def _preprocess(self, image: Image.Image, target_size=(224, 224)) -> np.ndarray:
        image = image.convert("RGB")
        arr = img_to_array(image)
        arr = tf.image.resize(arr, target_size).numpy()
        arr = arr / 255.0
        return np.expand_dims(arr.astype("float32", copy=False), axis=0)

    def _predict_numpy(self, inp_bhwc: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Motif model not loaded")
        tensor = tf.convert_to_tensor(inp_bhwc, dtype=tf.float32)
        output = self.model(tensor, training=False)
        return output.numpy()[0]

    def _topk(self, preds: np.ndarray, k: int = 3) -> List[dict]:
        idx = preds.argsort()[-k:][::-1]
        out = []
        for i in idx:
            out.append(
                {
                    "class_name": self.label_map.get(int(i), "?"),
                    "confidence": float(preds[i]),
                }
            )
        return out

    def predict(self, image: Image.Image, top_k: int = 3) -> dict:
        """Predict motif from PIL image and return formatted result."""
        if not self.loaded:
            raise RuntimeError("Motif model not loaded")

        inp = self._preprocess(image)
        preds = self._predict_numpy(inp)
        top = self._topk(preds, top_k)

        if not top:
            return {
                "motif": "Unknown",
                "confidence": 0.0,
                "probability_distribution": {},
            }

        probability_distribution = {
            item["class_name"]: item["confidence"] for item in top
        }

        return {
            "motif": top[0]["class_name"],
            "confidence": float(top[0]["confidence"]),
            "probability_distribution": probability_distribution,
        }

    def get_labels(self) -> list:
        return list(self.label_map.values())
