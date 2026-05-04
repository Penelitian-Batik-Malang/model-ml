import asyncio
import base64
import io
import logging
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image

from app.config.rate_limit import CBIR_LIMIT, limiter
from app.config.settings import settings
from app.services.batik_search_engine import search_general_batik
from app.utils.image_validator import ImageValidator
from app.utils.response import ResponseBuilder

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search")


def _parse_base64_payload(image_value: str) -> Tuple[bytes, Optional[str]]:
    try:
        if image_value.startswith("data:") and "," in image_value:
            header, b64_data = image_value.split(",", 1)
            content_type = None
            if ";base64" in header:
                content_type = header[5:].split(";base64", 1)[0]
            return base64.b64decode(b64_data), content_type
        return base64.b64decode(image_value), None
    except Exception as exc:
        raise ValueError("Invalid base64 image") from exc


def _validate_image_bytes(image_bytes: bytes, content_type: Optional[str]) -> Tuple[bool, str]:
    if content_type:
        return ImageValidator.validate_full(image_bytes, content_type)

    is_valid, error_msg = ImageValidator.validate_file_size(len(image_bytes))
    if not is_valid:
        return is_valid, error_msg
    return ImageValidator.validate_image_format(image_bytes)


def _load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


async def _get_image_from_request(
    request: Request, file: Optional[UploadFile]
) -> Image.Image:
    if file is not None:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        if not file.content_type:
            raise HTTPException(status_code=400, detail="Content-Type header is missing")

        file_content = await file.read()
        is_valid, error_msg = ImageValidator.validate_full(
            file_content=file_content,
            content_type=file.content_type,
        )
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        return _load_image_from_bytes(file_content)

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="No valid image provided")

    image_value = data.get("image") if isinstance(data, dict) else None
    if not image_value:
        raise HTTPException(status_code=400, detail="No valid image provided")

    try:
        image_bytes, content_type = _parse_base64_payload(image_value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    is_valid, error_msg = _validate_image_bytes(image_bytes, content_type)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    return _load_image_from_bytes(image_bytes)


@router.post(
    "/general",
    status_code=status.HTTP_200_OK,
    summary="Pencarian umum batik",
)
@limiter.limit(CBIR_LIMIT)
async def search_general(request: Request, file: UploadFile = File(None)) -> Dict[str, Any]:
    try:
        image = await _get_image_from_request(request, file)
        result = await asyncio.wait_for(
            asyncio.to_thread(search_general_batik, image, 10),
            timeout=settings.INFERENCE_TIMEOUT_SECONDS,
        )

        if not result.get("success"):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ResponseBuilder.error(
                    message="Search failed",
                    status=400,
                    errors=[result.get("error") or result.get("message") or "Search failed"],
                ).model_dump(),
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=ResponseBuilder.success(
                data=result,
                message="Search successful",
                status=200,
            ).model_dump(),
        )
    except asyncio.TimeoutError:
        logger.error("Search timeout")
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content=ResponseBuilder.error(
                message="Search timeout",
                status=504,
                errors=["Search exceeded timeout"],
            ).model_dump(),
        )
    except HTTPException as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content=ResponseBuilder.error(
                message="Invalid request",
                status=exc.status_code,
                errors=[str(exc.detail)],
            ).model_dump(),
        )
    except Exception as exc:
        logger.error("Search error: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ResponseBuilder.error(
                message="Internal server error",
                status=500,
                errors=["An unexpected error occurred"],
            ).model_dump(),
        )
