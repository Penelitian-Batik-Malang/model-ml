import logging

import numpy as np

from app.services.core.model_loader import ModelLoader
from app.services.core.palette import extract_all_palettes, extract_dominant_colors_kmeans, extract_palette_histogram, extract_palette_median_cut
from app.services.core.recolor import recolor_image, recolor_with_white_preserve, prepare_image, prepare_palette
from app.services.core.image_utils import file_to_numpy, numpy_to_base64, numpy_to_file

logger = logging.getLogger(__name__)


def get_model_status() -> dict:
    loader = ModelLoader.get_instance()
    return {"model_loaded": loader.is_ready}


def extract_palette(image_bytes: bytes, method: str = "all", n_colors: int = 6, max_size: int = 1280) -> dict:
    img_np = file_to_numpy(image_bytes, max_width=max_size, max_height=max_size)

    if method == "all":
        return extract_all_palettes(img_np, n_colors=n_colors)
    else:
        fn = {
            "kmeans": extract_dominant_colors_kmeans,
            "histogram": extract_palette_histogram,
            "median_cut": extract_palette_median_cut,
        }[method]
        return {method: fn(img_np, n_final=n_colors)}


def recolor(image_bytes: bytes, palette_hex: list, white_threshold: float = 150.0, max_size: int = 1280) -> np.ndarray:
    img_np = file_to_numpy(image_bytes, max_width=max_size, max_height=max_size)
    return recolor_with_white_preserve(img_np, palette_hex, white_threshold=white_threshold)


def recolor_simple(image_bytes: bytes, palette_hex: list, max_size: int = 1280) -> np.ndarray:
    img_np = file_to_numpy(image_bytes, max_width=max_size, max_height=max_size)
    img_tensor = prepare_image(img_np)
    pal_tensor, _ = prepare_palette(palette_hex)
    return recolor_image(img_tensor, pal_tensor)
