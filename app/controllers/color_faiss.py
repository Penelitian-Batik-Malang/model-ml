import asyncio
import io
import logging
from typing import Annotated, List

import cv2
import numpy as np
from PIL import Image
from fastapi import APIRouter, File, Form, Request, UploadFile, status
from fastapi.responses import JSONResponse

from app.config.rate_limit import CBIR_LIMIT, CLASSIFY_LIMIT, limiter
from app.config.settings import settings
from app.services.color_faiss_retriever import get_color_faiss_retriever
from app.services.extract_dominant_color import ExtractDominantColor
from app.services.s3_storage import get_s3_storage
from app.utils.image_validator import ImageValidator
from app.utils.resize import Resize
from app.utils.response import ResponseBuilder

logger = logging.getLogger(__name__)
router = APIRouter()
INVALID_REQUEST_MESSAGE = "Invalid request"


def _validate_num_cluster(num_cluster: int) -> bool:
    return num_cluster in {3, 4, 5}


def _parse_selected_colors(value: str, num_cluster: int) -> List[int]:
    if not value:
        return list(range(num_cluster))

    cleaned = value.replace(";", ",").replace("|", ",")
    items = [item.strip() for item in cleaned.split(",") if item.strip()]

    indices = []
    for item in items:
        idx = int(item)
        if idx < 1 or idx > num_cluster:
            raise ValueError("selected_colors out of range")
        indices.append(idx - 1)

    if not indices:
        return list(range(num_cluster))

    return sorted(set(indices))


def _load_image(file_content: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(file_content)).convert("RGB")
    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


@router.post(
    "/color-palette-faiss",
    response_model=None,
    status_code=status.HTTP_200_OK,
    tags=["Color FAISS"],
    summary="Extract dominant color palette",
)
@limiter.limit(CLASSIFY_LIMIT)
async def color_palette_faiss(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    num_cluster: Annotated[int, Form(...)],
):
    try:
        if not _validate_num_cluster(num_cluster):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ResponseBuilder.error(
                    message="Invalid num_cluster",
                    status=400,
                    errors=["num_cluster must be 3, 4, or 5"],
                ).model_dump(),
            )

        if not file.filename:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ResponseBuilder.error(
                    message=INVALID_REQUEST_MESSAGE,
                    status=400,
                    errors=["Filename is required"],
                ).model_dump(),
            )

        if not file.content_type:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ResponseBuilder.error(
                    message=INVALID_REQUEST_MESSAGE,
                    status=400,
                    errors=["Content-Type header is missing"],
                ).model_dump(),
            )

        file_content = await file.read()
        is_valid, error_msg = ImageValidator.validate_full(
            file_content=file_content,
            content_type=file.content_type,
        )
        if not is_valid:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ResponseBuilder.error(
                    message="Invalid image",
                    status=400,
                    errors=[error_msg],
                ).model_dump(),
            )

        image_bgr = _load_image(file_content)
        resized = Resize.proportional_resize(image_bgr, settings.COLOR_FAISS_MAX_SIZE)
        palette, _ = ExtractDominantColor.extract_palette_and_vector_s1(resized, num_cluster)

        response_payload = {
            "palette": palette,
            "count": len(palette),
        }
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=ResponseBuilder.success(
                data=response_payload,
                message="Palette extracted",
                status=200,
            ).model_dump(),
        )

    except Exception as exc:
        logger.error("Color palette error: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ResponseBuilder.error(
                message="Internal server error",
                status=500,
                errors=["An unexpected error occurred"],
            ).model_dump(),
        )


@router.post(
    "/get-recommendation-faiss",
    response_model=None,
    status_code=status.HTTP_200_OK,
    tags=["Color FAISS"],
    summary="Get recommendations using FAISS dominant colors",
)
@limiter.limit(CBIR_LIMIT)
async def get_recommendation_faiss(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    num_cluster: Annotated[int, Form(...)],
    top_k: Annotated[int, Form()] = 5,
    selected_colors: Annotated[str, Form()] = "",
):
    try:
        if not _validate_num_cluster(num_cluster):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ResponseBuilder.error(
                    message="Invalid num_cluster",
                    status=400,
                    errors=["num_cluster must be 3, 4, or 5"],
                ).model_dump(),
            )

        if top_k <= 0:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ResponseBuilder.error(
                    message="Invalid top_k",
                    status=400,
                    errors=["top_k must be greater than 0"],
                ).model_dump(),
            )

        if not file.filename:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ResponseBuilder.error(
                    message=INVALID_REQUEST_MESSAGE,
                    status=400,
                    errors=["Filename is required"],
                ).model_dump(),
            )

        if not file.content_type:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ResponseBuilder.error(
                    message=INVALID_REQUEST_MESSAGE,
                    status=400,
                    errors=["Content-Type header is missing"],
                ).model_dump(),
            )

        file_content = await file.read()
        is_valid, error_msg = ImageValidator.validate_full(
            file_content=file_content,
            content_type=file.content_type,
        )
        if not is_valid:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ResponseBuilder.error(
                    message="Invalid image",
                    status=400,
                    errors=[error_msg],
                ).model_dump(),
            )

        selected_slots = _parse_selected_colors(selected_colors, num_cluster)

        image_bgr = _load_image(file_content)
        resized = Resize.proportional_resize(image_bgr, settings.COLOR_FAISS_MAX_SIZE)
        feature_vector = ExtractDominantColor.extract_dominant_colors_s1(resized, num_cluster)

        retriever = get_color_faiss_retriever(
            settings.DATA_PATH,
            settings.COLOR_FAISS_SCENARIO,
            settings.COLOR_FAISS_CANDIDATE_MULTIPLIER,
        )

        results = await asyncio.to_thread(
            retriever.search,
            feature_vector,
            num_cluster,
            selected_slots,
            top_k,
        )

        storage = get_s3_storage()
        for item in results:
            image_key = storage.normalize_key(item.get("image_path", ""))
            item["image_path"] = image_key
            item["image_url"] = storage.generate_presigned_url(
                image_key,
                bucket_name=settings.S3_BUCKET_NAME_COLOR_FAISS or None,
            )

        # attach result counts and presigned urls already added above
        response_payload = {
            "results": results,
            "result_count": len(results),
        }
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=ResponseBuilder.success(
                data=response_payload,
                message="Recommendation successful",
                status=200,
            ).model_dump(),
        )

    except ValueError as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ResponseBuilder.error(
                message=INVALID_REQUEST_MESSAGE,
                status=400,
                errors=[str(exc)],
            ).model_dump(),
        )
    except Exception as exc:
        logger.error("Recommendation error: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ResponseBuilder.error(
                message="Internal server error",
                status=500,
                errors=["An unexpected error occurred"],
            ).model_dump(),
        )
