from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

def load_image_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)

def save_image_from_rgb(image_rgb: np.ndarray, path: Path, quality: int = 90) -> None:
    image = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
    image.save(path, format="JPEG", quality=quality)

def resize_mask_to_image(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if mask.shape == target_shape:
        return mask.astype(np.uint8)
    image = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    image = image.resize((target_shape[1], target_shape[0]), resample=Image.NEAREST)
    return np.array(image) > 0

def multiply_blend(mask: np.ndarray, fashion_rgb: np.ndarray, batik_rgb: np.ndarray) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)
    mask_bool = mask > 0
    if not mask_bool.any():
        return fashion_rgb.copy()

    y_indices, x_indices = np.where(mask_bool)
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()
    roi = fashion_rgb[y_min : y_max + 1, x_min : x_max + 1]
    mask_crop = mask_bool[y_min : y_max + 1, x_min : x_max + 1].astype(np.uint8)

    bbox_h, bbox_w = mask_crop.shape

    # Resize batik HANYA seukuran bounding box pakaian, menghindari pola tile yang aneh
    batik_fitted = cv2.resize(batik_rgb, (bbox_w, bbox_h), interpolation=cv2.INTER_LANCZOS4)

    # Grayscale menggunakan cv2 (RGB2GRAY) sesuai standar
    fashion_gray_crop = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    # Rata-rata pencahayaan hanya dari area mask
    rata_pencahayaan = np.mean(fashion_gray_crop[mask_crop == 1])
    shading_map = fashion_gray_crop / (rata_pencahayaan + 1e-6)

    # Clip shading_map agar warna batik tidak ter-overexpose oleh area terang fashion
    shading_map = np.clip(shading_map, 0.0, 2.0)

    # Apply shading per channel (loop) agar warna batik dominan
    batik_float = batik_fitted.astype(float)
    for i in range(3):
        batik_float[:, :, i] *= shading_map

    batik_final = np.clip(batik_float, 0, 255).astype(np.uint8)

    result = fashion_rgb.copy()
    roi_result = result[y_min : y_max + 1, x_min : x_max + 1]
    roi_result[mask_crop == 1] = batik_final[mask_crop == 1]
    result[y_min : y_max + 1, x_min : x_max + 1] = roi_result
    return result
