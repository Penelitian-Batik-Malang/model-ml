import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from pycocotools import mask as mask_api

from app.config.fashion import (
    FASHION_DETECTION_DIR,
    FASHION_CHECKPOINT_PATH,
    FASHION_LABEL_MAP_PATH,
    FASHION_CONFIG_FILE,
    FASHION_INFERENCE_SCRIPT,
    TPU_DIR,
)
from app.utils.image_processing import encode_mask_rgba_base64
from app.utils.session_handler import set_detected_parts

PART_IDS_BLENDING = {28, 29, 30, 31, 32, 33, 34}
UPPER_BODY_IDS = {1, 2, 3, 4, 5, 6, 10, 11, 12, 13}

PART_LABELS = {
    1: "shirt",
    2: "t-shirt",
    3: "sweater",
    4: "cardigan",
    5: "jacket",
    6: "vest",
    10: "dress",
    11: "jumpsuit",
    12: "suit",
    13: "coat",
    28: "hood",
    29: "collar",
    30: "lapel",
    31: "epaulette",
    32: "sleeve",
    33: "pocket",
    34: "neckline",
}

PART_COLORS = {
    "shirt": [128, 128, 128, 128],
    "t-shirt": [100, 150, 200, 128],
    "sweater": [200, 150, 100, 128],
    "cardigan": [150, 200, 100, 128],
    "jacket": [200, 100, 150, 128],
    "vest": [150, 100, 200, 128],
    "dress": [100, 200, 150, 128],
    "jumpsuit": [250, 150, 50, 128],
    "suit": [50, 150, 250, 128],
    "coat": [150, 250, 50, 128],
    "sleeve": [255, 80, 80, 128],
    "collar": [80, 160, 255, 128],
    "lapel": [80, 200, 80, 128],
    "hood": [255, 180, 50, 128],
    "pocket": [180, 80, 255, 128],
    "neckline": [255, 255, 80, 128],
    "epaulette": [80, 220, 220, 128],
}


def _build_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(FASHION_DETECTION_DIR),
            str(TPU_DIR / "models"),
            str(TPU_DIR / "models" / "official" / "efficientnet"),
            str(TPU_DIR / "models" / "hyperparameters"),
        ]
    )
    return env


def run_fashion_segmentation(
    image_path: Path, output_npy: Path, output_html: Path, timeout_seconds: int
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(FASHION_INFERENCE_SCRIPT),
        "--model=attribute_mask_rcnn",
        "--image_size=640",
        f"--checkpoint_path={FASHION_CHECKPOINT_PATH}",
        f"--label_map_file={FASHION_LABEL_MAP_PATH}",
        f"--config_file={FASHION_CONFIG_FILE}",
        f"--image_file_pattern={image_path}",
        f"--output_html={output_html}",
        "--max_boxes_to_draw=15",
        "--min_score_threshold=0.05",
        f"--output_file={output_npy}",
    ]
    env = _build_env()
    return subprocess.run(
        cmd,
        cwd=str(FASHION_DETECTION_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def load_segmentation_result(output_npy: Path) -> Dict:
    if not output_npy.exists():
        raise FileNotFoundError(f"Segmentation output not found: {output_npy}")
    data = np.load(str(output_npy), allow_pickle=True)
    if hasattr(data, "tolist"):
        data = data.tolist()
    if isinstance(data, np.ndarray) and data.size == 1:
        data = data[0]
    if isinstance(data, list) and len(data) > 0:
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError("Unexpected segmentation output format")
    return data


def decode_mask(encoded_mask: Dict) -> np.ndarray:
    if encoded_mask is None:
        raise ValueError("Encoded mask is None")
    decoded = mask_api.decode(encoded_mask)
    if decoded.ndim == 3 and decoded.shape[2] == 1:
        decoded = decoded[:, :, 0]
    return decoded.astype(np.uint8)


def _object_bbox(mask: np.ndarray) -> Dict[str, int]:
    positions = np.argwhere(mask > 0)
    if positions.size == 0:
        return {"x": 0, "y": 0, "w": 0, "h": 0}
    y_min, x_min = positions.min(axis=0)
    y_max, x_max = positions.max(axis=0)
    return {
        "x": int(x_min),
        "y": int(y_min),
        "w": int(x_max - x_min + 1),
        "h": int(y_max - y_min + 1),
    }


def _mask_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask > 0))


def ambil_model_busana(raw_result: Dict) -> Optional[Tuple[np.ndarray, int, int]]:
    upper_body_mask = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]
    jumlah_komposisi_piksel = 0
    label_upper_body = None
    idx_upper_body = None
    final_mask = None

    classes = raw_result.get("classes", [])
    masks = raw_result.get("masks", [])

    for idx, class_label in enumerate(classes):
        if int(class_label) not in upper_body_mask:
            continue

        rle_mask_dict = masks[idx]
        if rle_mask_dict is None:
            continue

        rle = dict(rle_mask_dict)
        if isinstance(rle.get("counts"), bytes):
            rle["counts"] = rle["counts"].decode("utf-8")

        binary_mask = mask_api.decode(rle)
        if binary_mask.ndim == 3 and binary_mask.shape[2] == 1:
            binary_mask = binary_mask[:, :, 0]

        pixel_count = int(np.sum(binary_mask))
        if pixel_count > jumlah_komposisi_piksel:
            jumlah_komposisi_piksel = pixel_count
            label_upper_body = int(class_label)
            idx_upper_body = idx
            final_mask = binary_mask.astype(np.uint8)

    if final_mask is None:
        return None

    return final_mask, label_upper_body, idx_upper_body


def build_parts_response(result: Dict, session_id: Optional[str] = None) -> Dict:
    classes = np.asarray(result.get("classes", []), dtype=np.int32)
    scores = np.asarray(result.get("scores", []), dtype=np.float32)
    masks = result.get("masks", [])
    num_instances = min(len(classes), len(scores), len(masks))

    part_masks = {name: [] for name in PART_COLORS.keys()}
    parts = {name: [] for name in PART_COLORS.keys()}
    upper_body_instances = []
    image_shape = None

    for index in range(num_instances):
        class_id = int(classes[index])
        score = float(scores[index])
        if score < 0.3:
            continue

        encoded_mask = masks[index]
        if encoded_mask is None:
            continue
        mask = decode_mask(encoded_mask)

        if image_shape is None:
            image_shape = mask.shape

        if class_id in UPPER_BODY_IDS:
            part_name = PART_LABELS.get(class_id, f"part_{class_id}")
            part_idx = len(parts[part_name])
            upper_body_instances.append((part_name, part_idx, mask))

            parts[part_name].append(
                {
                    "index": part_idx,
                    "bbox": None,
                    "mask_b64": None,
                    "area": 0,
                    "score": round(score, 3),
                    "original_mask": mask,
                }
            )
        elif class_id in PART_IDS_BLENDING:
            part_name = PART_LABELS.get(class_id, f"part_{class_id}")
            part_masks.setdefault(part_name, []).append(mask)
            parts[part_name].append(
                {
                    "index": len(parts[part_name]),
                    "bbox": _object_bbox(mask),
                    "mask_b64": encode_mask_rgba_base64(
                        mask, PART_COLORS.get(part_name, [255, 255, 255, 128])
                    ),
                    "area": _mask_area(mask),
                    "score": round(score, 3),
                }
            )

    for part_name, part_idx, ub_mask in upper_body_instances:
        final_ub_mask = ub_mask.copy()
        for masks_list in part_masks.values():
            for p_mask in masks_list:
                final_ub_mask = np.logical_and(final_ub_mask, np.logical_not(p_mask)).astype(np.uint8)

        parts[part_name][part_idx]["bbox"] = _object_bbox(final_ub_mask)
        parts[part_name][part_idx]["mask_b64"] = encode_mask_rgba_base64(
            final_ub_mask, PART_COLORS.get(part_name, [128, 128, 128, 128])
        )
        parts[part_name][part_idx]["area"] = _mask_area(final_ub_mask)
        del parts[part_name][part_idx]["original_mask"]

    height, width = image_shape if image_shape is not None else (0, 0)
    clean_parts = {k: v for k, v in parts.items() if len(v) > 0}
    detected_keys = list(clean_parts.keys())

    if session_id is not None:
        set_detected_parts(session_id, detected_keys)

    return {
        "image_size": {"w": int(width), "h": int(height)},
        "parts": clean_parts,
    }
