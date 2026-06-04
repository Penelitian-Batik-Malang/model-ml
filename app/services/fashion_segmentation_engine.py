import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
from pycocotools import mask as mask_api

from app.config.fashion import FASHION_SAVED_MODEL_DIR
from app.utils.image_processing import encode_mask_rgba_base64
from app.utils.session_handler import set_detected_parts

_saved_fashionpedia_model = None


def _yxyx_to_xywh(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes)
    if boxes.size == 0:
        return boxes
    ymin = boxes[..., 0]
    xmin = boxes[..., 1]
    ymax = boxes[..., 2]
    xmax = boxes[..., 3]
    width = xmax - xmin
    height = ymax - ymin
    return np.stack([xmin, ymin, width, height], axis=-1)


def _paste_instance_masks(
    masks: np.ndarray, detected_boxes: np.ndarray, image_height: int, image_width: int
) -> np.ndarray:
    _, mask_height, mask_width = masks.shape
    scale = max((mask_width + 2.0) / mask_width, (mask_height + 2.0) / mask_height)
    w_half = detected_boxes[:, 2] * 0.5
    h_half = detected_boxes[:, 3] * 0.5
    x_c = detected_boxes[:, 0] + w_half
    y_c = detected_boxes[:, 1] + h_half
    boxes_exp = np.zeros(detected_boxes.shape, dtype=np.int32)
    boxes_exp[:, 0] = (x_c - w_half * scale).astype(np.int32)
    boxes_exp[:, 2] = (x_c + w_half * scale).astype(np.int32)
    boxes_exp[:, 1] = (y_c - h_half * scale).astype(np.int32)
    boxes_exp[:, 3] = (y_c + h_half * scale).astype(np.int32)

    padded_mask = np.zeros((mask_height + 2, mask_width + 2), dtype=np.float32)
    segms = []
    for mask_ind, mask in enumerate(masks):
        im_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask[:, :]
        ref_box = boxes_exp[mask_ind, :]
        w = max(ref_box[2] - ref_box[0] + 1, 1)
        h = max(ref_box[3] - ref_box[1] + 1, 1)
        mask_resized = Image.fromarray(padded_mask).resize((w, h), resample=Image.BILINEAR)
        mask_bin = (np.array(mask_resized) > 0.5).astype(np.uint8)
        x_0 = min(max(ref_box[0], 0), image_width)
        x_1 = min(max(ref_box[2] + 1, 0), image_width)
        y_0 = min(max(ref_box[1], 0), image_height)
        y_1 = min(max(ref_box[3] + 1, 0), image_height)
        im_mask[y_0:y_1, x_0:x_1] = mask_bin[
            (y_0 - ref_box[1]):(y_1 - ref_box[1]),
            (x_0 - ref_box[0]):(x_1 - ref_box[0]),
        ]
        segms.append(im_mask)
    return np.array(segms, dtype=np.uint8)


def _encode_rle_mask(mask: np.ndarray) -> Dict:
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    mask_uint8 = np.asfortranarray((mask > 0).astype(np.uint8))
    encoded = mask_api.encode(mask_uint8)
    if isinstance(encoded.get("counts"), bytes):
        encoded["counts"] = encoded["counts"].decode("utf-8")
    return encoded


def _load_saved_model():
    global _saved_fashionpedia_model
    if _saved_fashionpedia_model is not None:
        return _saved_fashionpedia_model

    import tensorflow as tf

    if not FASHION_SAVED_MODEL_DIR.exists():
        raise FileNotFoundError(f"Saved model not found: {FASHION_SAVED_MODEL_DIR}")

    _saved_fashionpedia_model = tf.saved_model.load(str(FASHION_SAVED_MODEL_DIR))
    return _saved_fashionpedia_model


def _run_saved_model_segmentation(
    image_path: Path, output_npy: Path, output_html: Path
) -> subprocess.CompletedProcess:
    model = _load_saved_model()
    sig = model.signatures.get("serving_default") or list(model.signatures.values())[0]

    with open(image_path, "rb") as reader:
        image_bytes = reader.read()

    import tensorflow as tf

    try:
        outputs = sig(tf.constant([image_bytes]))
    except Exception:
        image = Image.open(image_path).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.expand_dims(array, axis=0)
        outputs = sig(tf.constant(array))

    outputs_np = {k: v.numpy() for k, v in outputs.items()}
    boxes = outputs_np.get("detection_boxes")
    scores = outputs_np.get("detection_scores")
    classes = outputs_np.get("detection_classes")
    masks = outputs_np.get("detection_masks")
    num = outputs_np.get("num_detections")

    if num is not None:
        n = int(np.asarray(num).reshape(-1)[0])
    else:
        n = boxes.shape[1] if boxes is not None else 0

    def _maybe_first(x):
        if x is None:
            return None
        arr = np.asarray(x)
        if arr.ndim >= 2 and arr.shape[0] == 1:
            arr = arr[0]
        return arr

    boxes = _maybe_first(boxes)
    scores = _maybe_first(scores)
    classes = _maybe_first(classes)
    masks = _maybe_first(masks)

    if boxes is not None:
        boxes = boxes[:n]
    if scores is not None:
        scores = scores[:n]
    if classes is not None:
        classes = classes[:n].astype(np.int32)
    if masks is not None:
        masks = masks[:n]

    image = Image.open(image_path).convert("RGB")
    image_height, image_width = image.size[1], image.size[0]

    image_info = outputs_np.get("image_info")
    if image_info is not None:
        try:
            image_info_arr = np.asarray(image_info)
            if image_info_arr.ndim >= 3:
                scale_y = float(image_info_arr[0][2][0])
                scale_x = float(image_info_arr[0][2][1])
            elif image_info_arr.ndim == 2:
                scale_y = float(image_info_arr[2][0])
                scale_x = float(image_info_arr[2][1])
            else:
                scale_y = 1024.0 / image_height
                scale_x = 1024.0 / image_width
        except Exception:
            scale_y = 1024.0 / image_height
            scale_x = 1024.0 / image_width
    else:
        scale_y = 1024.0 / image_height
        scale_x = 1024.0 / image_width

    if boxes is not None:
        boxes = np.asarray(boxes, dtype=np.float32).copy()
        if boxes.size:
            if boxes.max() <= 1.0:
                boxes[:, [0, 2]] *= image_height
                boxes[:, [1, 3]] *= image_width
            else:
                boxes[:, [0, 2]] /= scale_y
                boxes[:, [1, 3]] /= scale_x

    if boxes is not None and masks is not None:
        boxes_xywh = _yxyx_to_xywh(boxes)
        masks = _paste_instance_masks(masks, boxes_xywh, image_height, image_width)
        encoded_masks = [_encode_rle_mask(mask) for mask in masks]
    else:
        encoded_masks = [None] * n

    if classes is None:
        classes = np.zeros((n,), dtype=np.int32)
    if scores is None:
        scores = np.zeros((n,), dtype=np.float32)

    result = {
        "classes": classes.tolist(),
        "scores": scores.tolist(),
        "masks": encoded_masks,
    }

    np.save(str(output_npy), result, allow_pickle=True)
    try:
        output_html.parent.mkdir(parents=True, exist_ok=True)
        output_html.write_text("")
    except Exception:
        pass

    return subprocess.CompletedProcess(args=[str(FASHION_SAVED_MODEL_DIR)], returncode=0, stdout="saved model inference", stderr="")


def run_fashion_segmentation(
    image_path: Path, output_npy: Path, output_html: Path, timeout_seconds: int
) -> subprocess.CompletedProcess:
    if not FASHION_SAVED_MODEL_DIR.exists():
        raise FileNotFoundError(f"Saved model not found: {FASHION_SAVED_MODEL_DIR}")
    return _run_saved_model_segmentation(image_path, output_npy, output_html)

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
    blending_instances = []
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
            part_idx = len(parts[part_name])
            blending_instances.append((part_name, part_idx, mask, score))
            parts[part_name].append(None) # placeholder

    # Resolve overlaps among blending instances
    blending_instances_with_area = [
        (p_name, p_idx, mask, score, _mask_area(mask))
        for p_name, p_idx, mask, score in blending_instances
    ]
    blending_instances_with_area.sort(key=lambda x: x[4], reverse=True)
    
    claimed_blending = np.zeros(image_shape, dtype=bool) if image_shape else None
    resolved_blending = []
    
    if claimed_blending is not None:
        for p_name, p_idx, mask, score, area in blending_instances_with_area:
            resolved_mask = np.logical_and(mask, np.logical_not(claimed_blending)).astype(np.uint8)
            claimed_blending = np.logical_or(claimed_blending, mask)
            
            new_area = _mask_area(resolved_mask)
            if new_area > 0:
                parts[p_name][p_idx] = {
                    "index": p_idx,
                    "bbox": _object_bbox(resolved_mask),
                    "mask_b64": encode_mask_rgba_base64(
                        resolved_mask, PART_COLORS.get(p_name, [255, 255, 255, 128])
                    ),
                    "area": new_area,
                    "score": round(score, 3),
                }
                resolved_blending.append((p_name, resolved_mask))
            else:
                parts[p_name][p_idx] = None

    # Resolve overlaps among upper body instances (largest proportion wins the pixels)
    upper_body_instances_with_area = [
        (p_name, p_idx, mask, _mask_area(mask))
        for p_name, p_idx, mask in upper_body_instances
    ]
    upper_body_instances_with_area.sort(key=lambda x: x[3], reverse=True)
    
    claimed_upper = np.zeros(image_shape, dtype=bool) if image_shape else None
    resolved_upper_body = []
    
    if claimed_upper is not None:
        for p_name, p_idx, mask, area in upper_body_instances_with_area:
            resolved_mask = np.logical_and(mask, np.logical_not(claimed_upper)).astype(np.uint8)
            claimed_upper = np.logical_or(claimed_upper, mask)
            resolved_upper_body.append((p_name, p_idx, resolved_mask))
        upper_body_instances = resolved_upper_body

    for part_name, part_idx, ub_mask in upper_body_instances:
        final_ub_mask = ub_mask.copy()
        for p_name, p_mask in resolved_blending:
            final_ub_mask = np.logical_and(final_ub_mask, np.logical_not(p_mask)).astype(np.uint8)

        area_final = _mask_area(final_ub_mask)
        if area_final == 0:
            parts[part_name][part_idx] = None
        else:
            parts[part_name][part_idx]["bbox"] = _object_bbox(final_ub_mask)
            parts[part_name][part_idx]["mask_b64"] = encode_mask_rgba_base64(
                final_ub_mask, PART_COLORS.get(part_name, [128, 128, 128, 128])
            )
            parts[part_name][part_idx]["area"] = area_final
            if "original_mask" in parts[part_name][part_idx]:
                del parts[part_name][part_idx]["original_mask"]

    height, width = image_shape if image_shape is not None else (0, 0)
    
    # Filter out None and empty lists
    clean_parts = {}
    for k, v in parts.items():
        valid_items = [item for item in v if item is not None]
        if valid_items:
            clean_parts[k] = valid_items
    detected_keys = list(clean_parts.keys())

    if session_id is not None:
        set_detected_parts(session_id, detected_keys)

    return {
        "image_size": {"w": int(width), "h": int(height)},
        "parts": clean_parts,
    }
