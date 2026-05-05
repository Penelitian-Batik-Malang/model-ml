import asyncio
import io
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

import numpy as np
import requests
from fastapi import APIRouter, File, Form, Request, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image
from pycocotools import mask as mask_api

from app.config.rate_limit import CBIR_LIMIT, CLASSIFY_LIMIT, limiter
from app.config.settings import settings
from app.services.fashion_blending_engine import (
    load_image_rgb,
    multiply_blend,
    resize_mask_to_image,
    save_image_from_rgb,
)
from app.services.fashion_cbir_store import get_fashion_cbir_db
from app.services.fashion_recommendation_engine import extract_query_centroids, retrieve_batik
from app.services.fashion_segmentation_engine import (
    PART_LABELS,
    PART_IDS_BLENDING,
    UPPER_BODY_IDS,
    ambil_model_busana,
    build_parts_response,
    load_segmentation_result,
    run_fashion_segmentation,
)
from app.utils.image_processing import encode_image_to_base64_jpeg
from app.utils.image_validator import ImageValidator
from app.utils.response import ResponseBuilder
from app.utils.session_handler import (
    add_blended_part,
    get_session_dir,
    init_session,
    load_session_meta,
    reset_blended_parts,
    session_exists,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/fashion")


def _save_bytes(content: bytes, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)


def _read_current_image_base64(session_dir: Path) -> str:
    image_path = session_dir / "current.jpg"
    if not image_path.exists():
        raise FileNotFoundError("current.jpg not found")
    image_rgb = load_image_rgb(image_path)
    return encode_image_to_base64_jpeg(image_rgb)


def _get_blending_mask(raw_result: Dict[str, Any], part: str, instance_index: int) -> np.ndarray:
    classes = raw_result.get("classes", [])
    scores = raw_result.get("scores", [])
    masks = raw_result.get("masks", [])

    target_class_ids = []
    for class_id, label in PART_LABELS.items():
        if label == part:
            target_class_ids.append(class_id)

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
        raise ValueError(f"No valid mask found for part '{part}'")
    if instance_index < 0 or instance_index >= len(filtered_matched_masks):
        raise ValueError("Invalid instance_index")

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


async def _read_valid_image(file: UploadFile) -> bytes:
    if not file.filename:
        raise ValueError("Filename is required")
    if not file.content_type:
        raise ValueError("Content-Type header is missing")

    file_content = await file.read()
    is_valid, error_msg = ImageValidator.validate_full(
        file_content=file_content,
        content_type=file.content_type,
    )
    if not is_valid:
        raise ValueError(error_msg)

    return file_content


@router.post(
    "/segment",
    status_code=status.HTTP_200_OK,
    summary="Fashion segmentation",
)
@limiter.limit(CBIR_LIMIT)
async def fashion_segment(request: Request, image: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        file_content = await _read_valid_image(image)
    except ValueError as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ResponseBuilder.error(
                message="Invalid image",
                status=400,
                errors=[str(exc)],
            ).model_dump(),
        )

    session_id = str(uuid4())
    session_dir = init_session(session_id)
    fashion_path = session_dir / "fashion.jpg"
    current_path = session_dir / "current.jpg"
    result_npy = session_dir / "result.npy"
    output_html = session_dir / "inference.html"

    try:
        _save_bytes(file_content, fashion_path)
        shutil.copy2(fashion_path, current_path)
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ResponseBuilder.error(
                message="Failed to save uploaded image",
                status=500,
                errors=["Internal server error"],
            ).model_dump(),
        )

    try:
        proc = await asyncio.to_thread(
            run_fashion_segmentation,
            fashion_path,
            result_npy,
            output_html,
            settings.INFERENCE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        logger.error("Fashion segmentation timeout")
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content=ResponseBuilder.error(
                message="Inference timeout",
                status=504,
                errors=["Segmentation inference exceeded timeout"],
            ).model_dump(),
        )
    except Exception as exc:
        logger.error("Fashion segmentation error: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ResponseBuilder.error(
                message="Internal server error",
                status=500,
                errors=["An unexpected error occurred"],
            ).model_dump(),
        )

    if proc.returncode != 0:
        logger.error("Fashion segmentation failed: %s", proc.stderr)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ResponseBuilder.error(
                message="Segmentation failed",
                status=500,
                errors=["Segmentation failed"],
            ).model_dump(),
        )

    raw_result = load_segmentation_result(result_npy)
    parts_response = build_parts_response(raw_result, session_id=session_id)

    cbir_response = {}
    fashion_db = get_fashion_cbir_db()
    if fashion_db is not None:
        busana_result = ambil_model_busana(raw_result)
        if busana_result is not None:
            final_mask, label_upper_body, _idx_upper_body = busana_result
            fashion_rgb = load_image_rgb(current_path)
            query_centroids = extract_query_centroids(fashion_rgb, final_mask, kluster=3)
            cbir_data = retrieve_batik(query_centroids, fashion_db, top_k_list=[5, 10, 15])
            cbir_response = {
                "selected_label": PART_LABELS.get(label_upper_body, f"class_{label_upper_body}"),
                "selected_class_id": label_upper_body,
                "pixel_count": int(np.sum(final_mask)),
                "query_centroids": query_centroids.tolist(),
                **cbir_data,
            }

    payload = {
        "session_id": session_id,
        "image_size": parts_response["image_size"],
        "parts": parts_response["parts"],
        "cbir": cbir_response,
    }

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=ResponseBuilder.success(
            data=payload,
            message="Segmentation successful",
            status=200,
        ).model_dump(),
    )


@router.post(
    "/blend-manual",
    status_code=status.HTTP_200_OK,
    summary="Blend manual",
)
@limiter.limit(CLASSIFY_LIMIT)
async def blend_manual(
    request: Request,
    session_id: str = Form(...),
    part: str = Form(...),
    instance_index: int = Form(0),
    batik: UploadFile = File(...),
) -> Dict[str, Any]:
    if not session_exists(session_id):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=ResponseBuilder.error(
                message="Session not found",
                status=404,
                errors=["Session not found"],
            ).model_dump(),
        )

    try:
        batik_bytes = await _read_valid_image(batik)
    except ValueError as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ResponseBuilder.error(
                message="Invalid image",
                status=400,
                errors=[str(exc)],
            ).model_dump(),
        )

    session_dir = get_session_dir(session_id)
    current_path = session_dir / "current.jpg"
    result_npy = session_dir / "result.npy"

    raw_result = load_segmentation_result(result_npy)
    try:
        selected_mask = _get_blending_mask(raw_result, part, instance_index)
    except ValueError as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ResponseBuilder.error(
                message="Invalid request",
                status=400,
                errors=[str(exc)],
            ).model_dump(),
        )

    current_rgb = load_image_rgb(current_path)
    batik_path = session_dir / "batik_upload.jpg"
    _save_bytes(batik_bytes, batik_path)
    batik_rgb = load_image_rgb(batik_path)

    if selected_mask.shape != current_rgb.shape[:2]:
        selected_mask = resize_mask_to_image(selected_mask, current_rgb.shape[:2])

    blended_rgb = multiply_blend(selected_mask, current_rgb, batik_rgb)
    save_image_from_rgb(blended_rgb, current_path)
    add_blended_part(session_id, part, instance_index)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=ResponseBuilder.success(
            data={"image_b64": encode_image_to_base64_jpeg(blended_rgb)},
            message="Blend successful",
            status=200,
        ).model_dump(),
    )


def _download_batik_image(url: str) -> np.ndarray:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    batik_pil = Image.open(io.BytesIO(response.content)).convert("RGB")
    return np.array(batik_pil)


@router.post(
    "/blend-cbir",
    status_code=status.HTTP_200_OK,
    summary="Blend from CBIR",
)
@limiter.limit(CBIR_LIMIT)
async def blend_from_cbir(
    request: Request,
    session_id: str = Form(...),
    part: str = Form(...),
    instance_index: int = Form(0),
    batik_filename: str = Form(...),
) -> Dict[str, Any]:
    if not session_exists(session_id):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=ResponseBuilder.error(
                message="Session not found",
                status=404,
                errors=["Session not found"],
            ).model_dump(),
        )

    try:
        batik_rgb = await asyncio.to_thread(_download_batik_image, batik_filename)
    except Exception as exc:
        logger.error("Download batik error: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ResponseBuilder.error(
                message="Gagal mengunduh gambar batik",
                status=400,
                errors=["Failed to download batik image"],
            ).model_dump(),
        )

    session_dir = get_session_dir(session_id)
    current_path = session_dir / "current.jpg"
    result_npy = session_dir / "result.npy"

    raw_result = load_segmentation_result(result_npy)
    try:
        selected_mask = _get_blending_mask(raw_result, part, instance_index)
    except ValueError as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ResponseBuilder.error(
                message="Invalid request",
                status=400,
                errors=[str(exc)],
            ).model_dump(),
        )

    current_rgb = load_image_rgb(current_path)

    if selected_mask.shape != current_rgb.shape[:2]:
        selected_mask = resize_mask_to_image(selected_mask, current_rgb.shape[:2])

    blended_rgb = multiply_blend(selected_mask, current_rgb, batik_rgb)
    save_image_from_rgb(blended_rgb, current_path)
    add_blended_part(session_id, part, instance_index)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=ResponseBuilder.success(
            data={"image_b64": encode_image_to_base64_jpeg(blended_rgb)},
            message="Blend successful",
            status=200,
        ).model_dump(),
    )


@router.post(
    "/reset-session",
    status_code=status.HTTP_200_OK,
    summary="Reset session",
)
@limiter.limit(CLASSIFY_LIMIT)
async def reset_session(request: Request, session_id: str = Form(...)) -> Dict[str, Any]:
    if not session_exists(session_id):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=ResponseBuilder.error(
                message="Session not found",
                status=404,
                errors=["Session not found"],
            ).model_dump(),
        )

    session_dir = get_session_dir(session_id)
    fashion_path = session_dir / "fashion.jpg"
    current_path = session_dir / "current.jpg"

    shutil.copy2(fashion_path, current_path)
    reset_blended_parts(session_id)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=ResponseBuilder.success(
            data={"image_b64": _read_current_image_base64(session_dir)},
            message="Session reset",
            status=200,
        ).model_dump(),
    )


@router.get(
    "/session/{session_id}",
    status_code=status.HTTP_200_OK,
    summary="Get session",
)
@limiter.limit(CLASSIFY_LIMIT)
async def get_session(request: Request, session_id: str) -> Dict[str, Any]:
    if not session_exists(session_id):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=ResponseBuilder.error(
                message="Session not found",
                status=404,
                errors=["Session not found"],
            ).model_dump(),
        )

    session_dir = get_session_dir(session_id)
    current_image_b64 = _read_current_image_base64(session_dir)
    meta = load_session_meta(session_id)
    payload = {
        "session_id": session_id,
        "current_image_b64": current_image_b64,
        "parts_detected": meta.get("parts_detected", []),
        "parts_blended": meta.get("parts_blended", []),
    }

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=ResponseBuilder.success(
            data=payload,
            message="Session loaded",
            status=200,
        ).model_dump(),
    )
