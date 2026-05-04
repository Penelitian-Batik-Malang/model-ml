import asyncio
import io
import logging
from typing import Any, Dict, List

from fastapi import APIRouter, File, Request, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image

from app.config.rate_limit import CLASSIFY_LIMIT, limiter
from app.config.settings import settings
from app.services.model_loader import get_model_loader
from app.utils.image_validator import ImageValidator
from app.utils.response import ResponseBuilder

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/detection")


def _validate_upload(file: UploadFile) -> tuple[bool, str]:
    if not file.filename:
        return False, "Filename is required"
    if not file.content_type:
        return False, "Content-Type header is missing"
    return True, ""


def _build_error_response(status_code: int, message: str, errors: List[str]):
    return JSONResponse(
        status_code=status_code,
        content=ResponseBuilder.error(
            message=message,
            status=status_code,
            errors=errors,
        ).model_dump(),
    )


@router.get(
    "/motif/labels",
    status_code=status.HTTP_200_OK,
    summary="Daftar label motif",
)
async def motif_labels() -> Dict[str, Any]:
    model_loader = get_model_loader()
    motif_classifier = model_loader.get_motif_classifier()
    if not model_loader.is_motif_loaded() or motif_classifier is None:
        return _build_error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Model not loaded",
            ["Motif model loading failed or not initialized"],
        )

    labels = motif_classifier.get_labels()
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=ResponseBuilder.success(
            data=labels,
            message="Labels retrieved",
            status=200,
        ).model_dump(),
    )


@router.post(
    "/motif",
    status_code=status.HTTP_200_OK,
    summary="Deteksi motif batik",
)
@limiter.limit(CLASSIFY_LIMIT)
async def detect_motif(request: Request, file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        valid, error_msg = _validate_upload(file)
        if not valid:
            return _build_error_response(status.HTTP_400_BAD_REQUEST, "Invalid request", [error_msg])

        file_content = await file.read()
        is_valid, error_msg = ImageValidator.validate_full(
            file_content=file_content,
            content_type=file.content_type,
        )
        if not is_valid:
            return _build_error_response(status.HTTP_400_BAD_REQUEST, "Invalid image", [error_msg])

        model_loader = get_model_loader()
        motif_classifier = model_loader.get_motif_classifier()
        if not model_loader.is_motif_loaded() or motif_classifier is None:
            return _build_error_response(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "Model not loaded",
                ["Motif model loading failed or not initialized"],
            )

        image = Image.open(io.BytesIO(file_content))
        result = await asyncio.wait_for(
            asyncio.to_thread(motif_classifier.predict, image, 3),
            timeout=settings.INFERENCE_TIMEOUT_SECONDS,
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=ResponseBuilder.success(
                data=result,
                message="Motif detection successful",
                status=200,
            ).model_dump(),
        )
    except asyncio.TimeoutError:
        logger.error("Motif detection timeout")
        return _build_error_response(
            status.HTTP_504_GATEWAY_TIMEOUT,
            "Inference timeout",
            ["Motif inference exceeded 30 seconds timeout"],
        )
    except Exception as exc:
        logger.error("Motif detection error: %s", exc, exc_info=True)
        return _build_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Internal server error",
            ["An unexpected error occurred"],
        )


@router.get(
    "/type/labels",
    status_code=status.HTTP_200_OK,
    summary="Daftar label jenis batik",
)
async def type_labels() -> Dict[str, Any]:
    model_loader = get_model_loader()
    tulis_classifier = model_loader.get_tulis_classifier()
    if not model_loader.is_tulis_loaded() or tulis_classifier is None:
        return _build_error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Model not loaded",
            ["Tulis model loading failed or not initialized"],
        )

    labels = tulis_classifier.get_labels()
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=ResponseBuilder.success(
            data=labels,
            message="Labels retrieved",
            status=200,
        ).model_dump(),
    )


@router.post(
    "/type",
    status_code=status.HTTP_200_OK,
    summary="Deteksi jenis batik (tulis/cap)",
)
@limiter.limit(CLASSIFY_LIMIT)
async def detect_type(request: Request, file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        valid, error_msg = _validate_upload(file)
        if not valid:
            return _build_error_response(status.HTTP_400_BAD_REQUEST, "Invalid request", [error_msg])

        file_content = await file.read()
        is_valid, error_msg = ImageValidator.validate_full(
            file_content=file_content,
            content_type=file.content_type,
        )
        if not is_valid:
            return _build_error_response(status.HTTP_400_BAD_REQUEST, "Invalid image", [error_msg])

        model_loader = get_model_loader()
        tulis_classifier = model_loader.get_tulis_classifier()
        if not model_loader.is_tulis_loaded() or tulis_classifier is None:
            return _build_error_response(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "Model not loaded",
                ["Tulis model loading failed or not initialized"],
            )

        image = Image.open(io.BytesIO(file_content))
        result = await asyncio.wait_for(
            asyncio.to_thread(tulis_classifier.predict, image, 3),
            timeout=settings.INFERENCE_TIMEOUT_SECONDS,
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=ResponseBuilder.success(
                data=result,
                message="Type detection successful",
                status=200,
            ).model_dump(),
        )
    except asyncio.TimeoutError:
        logger.error("Type detection timeout")
        return _build_error_response(
            status.HTTP_504_GATEWAY_TIMEOUT,
            "Inference timeout",
            ["Type inference exceeded 30 seconds timeout"],
        )
    except Exception as exc:
        logger.error("Type detection error: %s", exc, exc_info=True)
        return _build_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Internal server error",
            ["An unexpected error occurred"],
        )
