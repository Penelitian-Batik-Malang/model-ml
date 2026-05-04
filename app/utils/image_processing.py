import base64
import io
from typing import Sequence

import numpy as np
from PIL import Image


def encode_image_to_base64_jpeg(image_rgb: np.ndarray, quality: int = 90) -> str:
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    image = Image.fromarray(image_rgb, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def encode_mask_rgba_base64(mask: np.ndarray, color: Sequence[int]) -> str:
    mask_bool = (mask > 0).astype(np.uint8)
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = np.array(color[:3], dtype=np.uint8)
    rgba[..., 3] = np.where(mask_bool, int(color[3]), 0).astype(np.uint8)
    image = Image.fromarray(rgba, mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
