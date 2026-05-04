import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

logger = logging.getLogger(__name__)


class TulisClassifier:
    """PyTorch-based classifier for batik tulis vs cap."""

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.model = None
        self.loaded = False
        self.label_map: Dict[int, str] = {0: "Batik Cap", 1: "Batik Tulis"}

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def load(self) -> bool:
        if not self.model_path.exists():
            logger.error("Tulis model file not found: %s", self.model_path)
            return False

        try:
            model = models.convnext_tiny(weights=None)
            model.classifier = nn.Sequential(
                nn.Flatten(1),
                nn.Linear(model.classifier[2].in_features, 512),
                nn.ELU(),
                nn.Linear(512, 2),
            )
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()

            self.model = model
            self.loaded = True
            logger.info("Tulis model loaded: %s", self.model_path)
            return True
        except Exception as exc:
            logger.error("Failed to load tulis model: %s", exc, exc_info=True)
            self.loaded = False
            return False

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        gray = image.convert("L")
        img = np.array(gray)
        img = self.clahe.apply(img)
        img = np.stack([img, img, img], axis=-1)
        img = self.transform(img)
        return img.unsqueeze(0).to(self.device)

    def _topk(self, probs: np.ndarray, k: int = 3) -> List[dict]:
        idx = probs.argsort()[-k:][::-1]
        out = []
        for i in idx:
            out.append(
                {
                    "class_name": self.label_map.get(int(i), "?"),
                    "confidence": float(probs[i]),
                }
            )
        return out

    def predict(self, image: Image.Image, top_k: int = 3) -> dict:
        if not self.loaded or self.model is None:
            raise RuntimeError("Tulis model not loaded")

        img_tensor = self._preprocess(image)
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]

        top = self._topk(probs, top_k)
        if not top:
            return {
                "label": "Unknown",
                "confidence": 0.0,
                "probability_distribution": {},
            }

        distribution = {item["class_name"]: item["confidence"] for item in top}

        return {
            "label": top[0]["class_name"],
            "confidence": float(top[0]["confidence"]),
            "probability_distribution": distribution,
        }

    def get_labels(self) -> list:
        return list(self.label_map.values())
