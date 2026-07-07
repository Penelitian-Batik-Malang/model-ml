import asyncio
import io
import logging
from typing import Annotated, List, Optional

import cv2
import numpy as np
from PIL import Image
from fastapi import APIRouter, File, Form, Request, UploadFile, status
from fastapi.responses import JSONResponse

from app.config.rate_limit import CBIR_LIMIT, CLASSIFY_LIMIT, limiter
from app.config.settings import settings
from app.services.color_faiss_retriever import get_color_faiss_retriever
from app.services.extract_dominant_color import ExtractDominantColor, LARGEST_K
from app.services.s3_storage import get_s3_storage
from app.utils.image_validator import ImageValidator
from app.utils.resize import Resize
from app.utils.response import ResponseBuilder

logger = logging.getLogger(__name__)
router = APIRouter()
INVALID_REQUEST_MESSAGE = "Invalid request"
DEFAULT_NUM_CLUSTER = LARGEST_K   # = 14 (sesuai FAISS artifacts)


def _parse_selected_colors(value: str, num_cluster: int = DEFAULT_NUM_CLUSTER) -> Optional[List[int]]:
    if not value:
        return None

    cleaned = value.replace(";", ",").replace("|", ",")
    items = [item.strip() for item in cleaned.split(",") if item.strip()]

    indices = []
    for item in items:
        idx = int(item)
        if idx < 1 or idx > num_cluster:
            raise ValueError(f"selected_colors out of range {num_cluster}")
        indices.append(idx - 1)

    if not indices:
        return None

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
):
    try:
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
        resized   = Resize.proportional_resize(image_bgr, 384)
        result    = ExtractDominantColor.extract_dominant_colors_careful(
            resized,
            max_clusters=DEFAULT_NUM_CLUSTER,
            find_elbow=True,
        )
        if result is None:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ResponseBuilder.error(
                    message="Invalid image",
                    status=400,
                    errors=["Unable to extract dominant palette from the image."],
                ).model_dump(),
            )

        palette_hex = [
            ExtractDominantColor.LABELER.lab_to_hex(f["L"], f["a"], f["b"])
            for f in result["features"]
        ]
        
        colors_array = [
            [f["L"], f["a"], f["b"], f["P"]]
            for f in result["features"]
        ]

        response_payload = {
            "palette": palette_hex,
            "color_names": result["labels"],
            "colors": colors_array,
            "count": len(result["labels"]),
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
    top_k: Annotated[int, Form()] = 15,
    selected_colors: Annotated[str, Form()] = "",
):
    try:
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

        selected_slots = _parse_selected_colors(selected_colors, num_cluster=DEFAULT_NUM_CLUSTER)

        image_bgr = _load_image(file_content)
        resized   = Resize.proportional_resize(image_bgr, 384)
        result    = ExtractDominantColor.extract_dominant_colors_careful(
            resized,
            max_clusters=DEFAULT_NUM_CLUSTER,
            find_elbow=True,
        )
        if result is None:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ResponseBuilder.error(
                    message="Invalid image",
                    status=400,
                    errors=["Unable to extract dominant palette from the image."],
                ).model_dump(),
            )

        # Pastikan hasil ekstraksi dapat diproses FAISS
        if "full_query_vector" in result:
            result["full_query_vector"] = np.array(
                result["full_query_vector"], dtype=np.float32
            ).reshape(-1)
        if "feature_vector" in result:
            result["feature_vector"] = np.array(
                result["feature_vector"], dtype=np.float32
            ).reshape(-1)

        # Kirimkan result dict lengkap ke retriever (berisi full_query_vector)
        retriever = get_color_faiss_retriever(settings.DATA_PATH)

        results = await asyncio.to_thread(
            retriever.search,
            result,
            selected_slots,
            top_k,
            n_candidates_multiplier=10,
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
