import shutil
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4
import numpy as np

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from pycocotools import mask as mask_api

from services.fashion_segmentation_engine import build_parts_response, load_segmentation_result, run_fashion_segmentation, ambil_model_busana, PART_LABELS, UPPER_BODY_IDS, PART_IDS_BLENDING
from services.fashion_blending_engine import load_image_rgb, save_image_from_rgb, multiply_blend, resize_mask_to_image
from services.fashion_recommendation_engine import extract_query_centroids, retrieve_batik, load_batik_database
from utils.image_processing import encode_image_to_base64_jpeg
from utils.session_handler import (
    init_session, get_session_dir, session_exists, add_blended_part, reset_blended_parts, load_session_meta
)
from config import FASHION_CBIR_FEATURES_NPZ

router = APIRouter()

# Global DB
BATIK_DB = None

class ResetRequest(BaseModel):
    session_id: str

def init_cbir_db():
    global BATIK_DB
    if FASHION_CBIR_FEATURES_NPZ.exists():
        BATIK_DB = load_batik_database(FASHION_CBIR_FEATURES_NPZ)
        print(f"Database batik loaded: {len(BATIK_DB['filenames'])} items")
    else:
        print(f"WARNING: NPZ tidak ditemukan di {FASHION_CBIR_FEATURES_NPZ}")

def _save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    upload_file.file.seek(0)
    with destination.open("wb") as dest_file:
        shutil.copyfileobj(upload_file.file, dest_file)

def _read_current_image_base64(session_dir: Path) -> str:
    image_path = session_dir / "current.jpg"
    if not image_path.exists():
        raise FileNotFoundError("current.jpg not found")
    image_rgb = load_image_rgb(image_path)
    return encode_image_to_base64_jpeg(image_rgb)

# _get_mask_union() digantikan oleh ambil_model_busana() dari fashion_segmentation_engine
# sesuai implementasi skripsi: pilih SATU label upper body dengan piksel terbesar

def _get_blending_mask(raw_result: Dict[str, Any], part: str, instance_index: int) -> np.ndarray:
    classes = raw_result.get("classes", [])
    scores = raw_result.get("scores", [])
    masks = raw_result.get("masks", [])

    target_class_ids = []
    for cid, label in PART_LABELS.items():
        if label == part:
            target_class_ids.append(cid)

    filtered_matched_masks = []
    all_parts_masks = []

    for idx, class_id in enumerate(classes):
        if float(scores[idx]) < 0.3:
            continue
        if int(class_id) in target_class_ids:
            filtered_matched_masks.append(masks[idx])
        if int(class_id) in PART_IDS_BLENDING:
            all_parts_masks.append(masks[idx])

    if len(filtered_matched_masks) == 0:
        raise HTTPException(status_code=404, detail=f"No valid mask found for part '{part}'")
    if instance_index < 0 or instance_index >= len(filtered_matched_masks):
        raise HTTPException(status_code=400, detail="Invalid instance_index")

    encoded_mask = filtered_matched_masks[instance_index]
    decoded = mask_api.decode(encoded_mask)
    if decoded.ndim == 3 and decoded.shape[2] == 1:
        decoded = decoded[:, :, 0]
    base_mask = decoded.astype(bool)

    if any(cid in UPPER_BODY_IDS for cid in target_class_ids):
        for p_enc in all_parts_masks:
            p_dec = mask_api.decode(p_enc)
            if p_dec.ndim == 3 and p_dec.shape[2] == 1:
                p_dec = p_dec[:, :, 0]
            base_mask = np.logical_and(base_mask, np.logical_not(p_dec.astype(bool)))

    return base_mask.astype(np.uint8)

@router.post("/segment")
def fashion_segment(image: UploadFile = File(...)) -> Dict[str, Any]:
    session_id = str(uuid4())
    session_dir = init_session(session_id)
    fashion_path = session_dir / "fashion.jpg"
    current_path = session_dir / "current.jpg"
    result_npy = session_dir / "result.npy"
    output_html = session_dir / "inference.html"

    try:
        _save_upload_file(image, fashion_path)
        shutil.copy2(fashion_path, current_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded image: {exc}")

    proc = run_fashion_segmentation(fashion_path, result_npy, output_html)
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail={"message": "Segmentation failed.", "stderr": proc.stderr})

    raw_result = load_segmentation_result(result_npy)
    parts_response = build_parts_response(raw_result, session_id=session_id)

    cbir_response = {}
    if BATIK_DB is not None:
        # Sesuai skripsi: pilih SATU label upper body dengan piksel terbesar
        busana_result = ambil_model_busana(raw_result)
        if busana_result is not None:
            final_mask, label_upper_body, idx_upper_body = busana_result
            fashion_rgb = load_image_rgb(current_path)
            query_centroids = extract_query_centroids(fashion_rgb, final_mask, kluster=3)
            cbir_data = retrieve_batik(query_centroids, BATIK_DB, top_k_list=[5, 10, 15])
            cbir_response = {
                "selected_label": PART_LABELS.get(label_upper_body, f"class_{label_upper_body}"),
                "selected_class_id": label_upper_body,
                "pixel_count": int(np.sum(final_mask)),
                "query_centroids": query_centroids.tolist(),
                **cbir_data,
            }

    return {"session_id": session_id, "image_size": parts_response["image_size"], "parts": parts_response["parts"], "cbir": cbir_response}

@router.post("/blend-manual")
def blend_manual(session_id: str = Form(...), part: str = Form(...), instance_index: int = Form(0), batik: UploadFile = File(...)) -> Dict[str, Any]:
    if not session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    session_dir = get_session_dir(session_id)
    current_path = session_dir / "current.jpg"
    result_npy = session_dir / "result.npy"

    raw_result = load_segmentation_result(result_npy)
    selected_mask = _get_blending_mask(raw_result, part, instance_index)

    current_rgb = load_image_rgb(current_path)
    batik_path = session_dir / "batik_upload.jpg"
    _save_upload_file(batik, batik_path)
    batik_rgb = load_image_rgb(batik_path)

    if selected_mask.shape != current_rgb.shape[:2]:
        selected_mask = resize_mask_to_image(selected_mask, current_rgb.shape[:2])

    blended_rgb = multiply_blend(selected_mask, current_rgb, batik_rgb)
    save_image_from_rgb(blended_rgb, current_path)
    add_blended_part(session_id, part, instance_index)

    return {"image_b64": encode_image_to_base64_jpeg(blended_rgb)}

@router.post("/blend-cbir")
def blend_from_cbir(session_id: str = Form(...), part: str = Form(...), instance_index: int = Form(0), batik_filename: str = Form(...)) -> Dict[str, Any]:
    if not session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    import requests
    from PIL import Image
    import io
    
    try:
        response = requests.get(batik_filename)
        response.raise_for_status()
        batik_pil = Image.open(io.BytesIO(response.content)).convert("RGB")
        batik_rgb = np.array(batik_pil)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal mengunduh gambar batik dari S3: {str(e)}")

    session_dir = get_session_dir(session_id)
    current_path = session_dir / "current.jpg"
    result_npy = session_dir / "result.npy"

    raw_result = load_segmentation_result(result_npy)
    selected_mask = _get_blending_mask(raw_result, part, instance_index)

    current_rgb = load_image_rgb(current_path)

    if selected_mask.shape != current_rgb.shape[:2]:
        selected_mask = resize_mask_to_image(selected_mask, current_rgb.shape[:2])

    blended_rgb = multiply_blend(selected_mask, current_rgb, batik_rgb)
    save_image_from_rgb(blended_rgb, current_path)
    add_blended_part(session_id, part, instance_index)

    return {"image_b64": encode_image_to_base64_jpeg(blended_rgb)}

@router.post("/reset-session")
def reset_session(payload: ResetRequest) -> Dict[str, str]:
    session_id = payload.session_id
    if not session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    session_dir = get_session_dir(session_id)
    fashion_path = session_dir / "fashion.jpg"
    current_path = session_dir / "current.jpg"
    
    shutil.copy2(fashion_path, current_path)
    reset_blended_parts(session_id)
    return {"image_b64": _read_current_image_base64(session_dir)}

@router.get("/session/{session_id}")
def get_session(session_id: str) -> Dict[str, Any]:
    if not session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    session_dir = get_session_dir(session_id)
    current_image_b64 = _read_current_image_base64(session_dir)
    meta = load_session_meta(session_id)
    return {
        "session_id": session_id,
        "current_image_b64": current_image_b64,
        "parts_detected": meta.get("parts_detected", []),
        "parts_blended": meta.get("parts_blended", []),
    }
