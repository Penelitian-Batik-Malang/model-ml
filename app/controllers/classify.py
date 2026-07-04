import logging
import asyncio
import io
from PIL import Image
from fastapi import APIRouter, UploadFile, File, status, HTTPException, Request
from fastapi.responses import JSONResponse
from app.schemas.response import APIResponse, ClassificationResponse
from app.utils.response import ResponseBuilder
from app.utils.image_validator import ImageValidator
from app.services.model_loader import get_model_loader
from app.config.rate_limit import CLASSIFY_LIMIT, limiter
from app.config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/classify",
    response_model=APIResponse,
    status_code=status.HTTP_200_OK,
    tags=["Classification"],
    summary="Classify Batik Motif",
    description=f"Classify batik motif dari file gambar. Rate limit: {CLASSIFY_LIMIT}",
)
@limiter.limit(CLASSIFY_LIMIT)
async def classify_motif(request: Request, file: UploadFile = File(...)):
    """
    Classify batik motif dari file gambar yang di-upload.
    
    Request:
    - file: Gambar dalam format JPEG, PNG, atau WebP (max 5MB)
    
    Response:
    - motif: Nama motif batik yang teridentifikasi
    - confidence: Confidence score (0-1)
    - probability_distribution: Distribusi probabilitas semua kelas
    
    Errors:
    - 400: Invalid file (format, size, atau content type)
    - 401: Missing atau invalid API Key
    - 504: Inference timeout (lebih dari 30 detik)
    - 500: Internal server error
    
    Rate limit:
    - 10 requests per minute per IP
    """
    try:
        # Check if motif model loaded
        model_loader = get_model_loader()
        motif_classifier = model_loader.get_motif_classifier()
        if not model_loader.is_motif_loaded() or motif_classifier is None:
            logger.error("Motif model not loaded")
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=ResponseBuilder.error(
                    message="Model not loaded",
                    status=503,
                    errors=["Motif model loading failed or not initialized"],
                ).model_dump(),
            )
        
        # Validate file type
        if not file.filename:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ResponseBuilder.error(
                    message="Invalid request",
                    status=400,
                    errors=["Filename is required"],
                ).model_dump(),
            )
        
        # Validate content type
        if not file.content_type:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ResponseBuilder.error(
                    message="Invalid request",
                    status=400,
                    errors=["Content-Type header is missing"],
                ).model_dump(),
            )
        
        # Read file content
        file_content = await file.read()
        
        # Validate image
        is_valid, error_msg = ImageValidator.validate_full(
            file_content=file_content,
            content_type=file.content_type,
        )
        
        if not is_valid:
            logger.warning(f"Image validation failed: {error_msg}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ResponseBuilder.error(
                    message="Invalid image",
                    status=400,
                    errors=[error_msg],
                ).model_dump(),
            )
        
        # Run inference dengan timeout
        try:
            image = Image.open(io.BytesIO(file_content))
            classification_result = await asyncio.wait_for(
                asyncio.to_thread(motif_classifier.predict, image, 3),
                timeout=settings.INFERENCE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error("Classification inference timeout")
            return JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content=ResponseBuilder.error(
                    message="Inference timeout",
                    status=504,
                    errors=["Classification inference exceeded 30 seconds timeout"],
                ).model_dump(),
            )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=ResponseBuilder.success(
                data=classification_result,
                message="Classification successful",
                status=200,
            ).model_dump(),
        )
    
    except Exception as e:
        logger.error(f"Classification error: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ResponseBuilder.error(
                message="Internal server error",
                status=500,
                errors=["An unexpected error occurred"],
            ).model_dump(),
        )
