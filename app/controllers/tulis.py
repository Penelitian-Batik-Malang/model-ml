import asyncio
import io
import logging
from PIL import Image
from fastapi import APIRouter, UploadFile, File, status, Request
from fastapi.responses import JSONResponse

from app.config.rate_limit import CLASSIFY_LIMIT, limiter
from app.config.settings import settings
from app.services.model_loader import get_model_loader
from app.utils.image_validator import ImageValidator
from app.utils.response import ResponseBuilder

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/tulis",
    response_model=None,
    status_code=status.HTTP_200_OK,
    tags=["Tulis"],
    summary="Classify Batik Tulis vs Cap",
    description=f"Classify batik tulis vs cap. Rate limit: {CLASSIFY_LIMIT}",
)
@limiter.limit(CLASSIFY_LIMIT)
async def classify_tulis(request: Request, file: UploadFile = File(...)):
    """Classify batik tulis vs cap from uploaded image."""
    try:
        model_loader = get_model_loader()
        tulis_classifier = model_loader.get_tulis_classifier()
        if not model_loader.is_tulis_loaded() or tulis_classifier is None:
            logger.error("Tulis model not loaded")
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=ResponseBuilder.error(
                    message="Model not loaded",
                    status=503,
                    errors=["Tulis model loading failed or not initialized"],
                ).model_dump(),
            )

        if not file.filename:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ResponseBuilder.error(
                    message="Invalid request",
                    status=400,
                    errors=["Filename is required"],
                ).model_dump(),
            )

        if not file.content_type:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ResponseBuilder.error(
                    message="Invalid request",
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
            logger.warning("Image validation failed: %s", error_msg)
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ResponseBuilder.error(
                    message="Invalid image",
                    status=400,
                    errors=[error_msg],
                ).model_dump(),
            )

        try:
            image = Image.open(io.BytesIO(file_content))
            result = await asyncio.wait_for(
                asyncio.to_thread(tulis_classifier.predict, image, 3),
                timeout=settings.INFERENCE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error("Tulis inference timeout")
            return JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content=ResponseBuilder.error(
                    message="Inference timeout",
                    status=504,
                    errors=["Tulis inference exceeded 30 seconds timeout"],
                ).model_dump(),
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=ResponseBuilder.success(
                data=result,
                message="Tulis classification successful",
                status=200,
            ).model_dump(),
        )

    except Exception as exc:
        logger.error("Tulis classification error: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ResponseBuilder.error(
                message="Internal server error",
                status=500,
                errors=["An unexpected error occurred"],
            ).model_dump(),
        )
