import os
import time
import psutil
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2

from config import BASE_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MODEL_PATH = str(BASE_DIR / "models" / "model_ConvNextTiny_original_all.pt")

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

label_map = {0: "Batik Cap", 1: "Batik Tulis"}
model = None

def initialize_model(model_path=None):
    global model
    model_path = model_path or DEFAULT_MODEL_PATH
    
    try:
        model = models.convnext_tiny(weights=None)
        model.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(model.classifier[2].in_features, 512),
            nn.ELU(),
            nn.Linear(512, 2)
        )
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print(f"[Type] Tulis/Cap model loaded successfully from {model_path}")
        return True
    except Exception as e:
        print(f"[Type] Error loading tulis/cap model: {str(e)}")
        return False

def preprocess_image(image_pil: Image.Image):
    gray = image_pil.convert('L')
    img = np.array(gray)
    img = clahe.apply(img)
    img = np.stack([img, img, img], axis=-1)
    img = transform(img)
    return img.unsqueeze(0).to(DEVICE)

def predict_type(image_pil: Image.Image) -> dict:
    if model is None:
        success = initialize_model()
        if not success:
            return {"error": "Model failed to load"}

    start_time = time.time()
    img_tensor = preprocess_image(image_pil)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]

    top_idx = probs.argsort()[-2:][::-1]
    results = [{"class_name": label_map.get(i, "?"),
                "confidence": float(probs[i]),
                "percentage": f"{round(probs[i] * 100, 2)}%"} for i in top_idx]

    return {
        "predictions": results,
        "inference_time_ms": round((time.time() - start_time) * 1000, 2),
        "memory_usage_mb": round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)
    }

def get_type_labels() -> list:
    return list(label_map.values())
