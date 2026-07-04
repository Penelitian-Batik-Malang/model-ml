import asyncio
import logging

import numpy as np
from fastapi import APIRouter, File, Request, UploadFile, status
from fastapi.responses import JSONResponse

from app.config.rate_limit import CLASSIFY_LIMIT, limiter
from app.config.settings import settings
from app.schemas.recolor import PaletteExtractRequest, RecolorRequest, RecolorSimpleRequest
from app.services.core.model_loader import ModelLoader
from app.services.core.palette import extract_all_palettes
from app.services.core.recolor import prepare_image, prepare_palette, recolor_image, recolor_with_white_preserve
from app.utils.image_validator import ImageValidator
from app.utils.response import ResponseBuilder

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/recolor")


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


def _bytes_to_numpy(file_bytes: bytes, max_size: int = 1280) -> np.ndarray:
    from app.services.core.image_utils import file_to_numpy
    return file_to_numpy(file_bytes, max_width=max_size, max_height=max_size)


@router.get("/health", status_code=status.HTTP_200_OK, summary="Recolor model health check")
@limiter.limit(CLASSIFY_LIMIT)
async def health(request: Request):
    loader = ModelLoader.get_instance()
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=ResponseBuilder.success(
            data={"model_loaded": loader.is_ready},
            message="Recolor service is running",
            status=200,
        ).model_dump(),
    )


@router.post("/palette/extract", status_code=status.HTTP_200_OK, summary="Extract color palette from image")
@limiter.limit(CLASSIFY_LIMIT)
async def palette_extract(
    request: Request,
    image: UploadFile = File(...),
    method: str = "all",
    n_colors: int = 6,
):
    try:
        file_bytes = await _read_valid_image(image)
    except ValueError as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ResponseBuilder.error(
                message="Invalid image",
                status=400,
                errors=[str(exc)],
            ).model_dump(),
        )

    try:
        img_np = await asyncio.to_thread(_bytes_to_numpy, file_bytes)
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ResponseBuilder.error(
                message="Failed to process image",
                status=400,
                errors=[str(exc)],
            ).model_dump(),
        )

    try:
        if method == "all":
            palette = await asyncio.to_thread(extract_all_palettes, img_np, n_colors=n_colors)
        else:
            from app.services.core.palette import (
                extract_dominant_colors_kmeans,
                extract_palette_histogram,
                extract_palette_median_cut,
            )
            fn = {
                "kmeans": extract_dominant_colors_kmeans,
                "histogram": extract_palette_histogram,
                "median_cut": extract_palette_median_cut,
            }[method]
            palette = {method: await asyncio.to_thread(fn, img_np, n_final=n_colors)}

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=ResponseBuilder.success(
                data={"method": method, "n_colors": n_colors, "palette": palette},
                message="Palette extracted successfully",
                status=200,
            ).model_dump(),
        )
    except Exception as exc:
        logger.error("Palette extraction error: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ResponseBuilder.error(
                message="Palette extraction failed",
                status=500,
                errors=[str(exc)],
            ).model_dump(),
        )


@router.post("/recolor", status_code=status.HTTP_200_OK, summary="Recolor image with white preservation")
@limiter.limit(CLASSIFY_LIMIT)
async def recolor(
    request: Request,
    image: UploadFile = File(...),
    palette: str = '["#FF0000","#00FF00","#0000FF","#FFFF00","#FF00FF","#00FFFF"]',
    white_threshold: float = 150.0,
):
    import json
    try:
        palette_list = json.loads(palette)
        recolor_req = RecolorRequest(palette=palette_list, white_threshold=white_threshold)
    except (json.JSONDecodeError, ValueError) as exc:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ResponseBuilder.error(
                message="Invalid palette",
                status=422,
                errors=[str(exc)],
            ).model_dump(),
        )

    try:
        file_bytes = await _read_valid_image(image)
    except ValueError as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ResponseBuilder.error(
                message="Invalid image",
                status=400,
                errors=[str(exc)],
            ).model_dump(),
        )

    try:
        img_np = await asyncio.to_thread(_bytes_to_numpy, file_bytes)
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ResponseBuilder.error(
                message="Failed to process image",
                status=400,
                errors=[str(exc)],
            ).model_dump(),
        )

    loader = ModelLoader.get_instance()
    if not loader.is_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ResponseBuilder.error(
                message="Model not loaded",
                status=503,
                errors=["Recolor models are not loaded yet"],
            ).model_dump(),
        )

    try:
        result = await asyncio.to_thread(
            recolor_with_white_preserve,
            img_np,
            recolor_req.palette,
            white_threshold=recolor_req.white_threshold,
        )
    except Exception as exc:
        logger.error("Recolor error: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ResponseBuilder.error(
                message="Recoloring failed",
                status=500,
                errors=[str(exc)],
            ).model_dump(),
        )

    from app.services.core.image_utils import numpy_to_base64
    result_b64 = numpy_to_base64(result)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=ResponseBuilder.success(
            data={"image_b64": result_b64},
            message="Recoloring successful",
            status=200,
        ).model_dump(),
    )


@router.post("/recolor/simple", status_code=status.HTTP_200_OK, summary="Recolor image without white preservation")
@limiter.limit(CLASSIFY_LIMIT)
async def recolor_simple(
    request: Request,
    image: UploadFile = File(...),
    palette: str = '["#FF0000","#00FF00","#0000FF","#FFFF00","#FF00FF","#00FFFF"]',
):
    import json
    try:
        palette_list = json.loads(palette)
        RecolorSimpleRequest(palette=palette_list)
    except (json.JSONDecodeError, ValueError) as exc:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ResponseBuilder.error(
                message="Invalid palette",
                status=422,
                errors=[str(exc)],
            ).model_dump(),
        )

    try:
        file_bytes = await _read_valid_image(image)
    except ValueError as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ResponseBuilder.error(
                message="Invalid image",
                status=400,
                errors=[str(exc)],
            ).model_dump(),
        )

    try:
        img_np = await asyncio.to_thread(_bytes_to_numpy, file_bytes)
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ResponseBuilder.error(
                message="Failed to process image",
                status=400,
                errors=[str(exc)],
            ).model_dump(),
        )

    loader = ModelLoader.get_instance()
    if not loader.is_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ResponseBuilder.error(
                message="Model not loaded",
                status=503,
                errors=["Recolor models are not loaded yet"],
            ).model_dump(),
        )

    try:
        def _recolor_simple(img, pal):
            img_tensor = prepare_image(img)
            pal_tensor, _ = prepare_palette(pal)
            return recolor_image(img_tensor, pal_tensor)

        result = await asyncio.to_thread(_recolor_simple, img_np, palette_list)
    except Exception as exc:
        logger.error("Recolor simple error: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ResponseBuilder.error(
                message="Recoloring failed",
                status=500,
                errors=[str(exc)],
            ).model_dump(),
        )

    from app.services.core.image_utils import numpy_to_base64
    result_b64 = numpy_to_base64(result)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=ResponseBuilder.success(
            data={"image_b64": result_b64},
            message="Recoloring successful",
            status=200,
        ).model_dump(),
    )
