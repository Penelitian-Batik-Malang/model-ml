import logging
import asyncio
import io
import time
from PIL import Image
from fastapi import APIRouter, UploadFile, File, status, Request
from fastapi.responses import JSONResponse
from app.schemas.response import APIResponse, CBIRResult
from app.utils.response import ResponseBuilder
from app.utils.image_validator import ImageValidator
from app.services.model_loader import get_model_loader
from app.config.rate_limit import CBIR_LIMIT, limiter
from app.config.settings import settings
from app.services.s3_storage import get_s3_storage

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_path_s3_for_result(item: dict) -> str:
    """
    Map CBIR result to path_s3 from the index CSV.
    CBIR results have paths like "rotate/..." or "flip/..." (augmentation variants).
    CSV has paths like "augmentasi/rotate/..." or "augmentasi/flip/...".
    Prepend "augmentasi/" if not already present.
    """
    image_id = item.get("image_id") or ""
    
    # Prepend "augmentasi/" if not already there
    if image_id and not image_id.startswith("augmentasi/"):
        path_s3 = f"augmentasi/augmentasi/{image_id}"
    else:
        path_s3 = image_id
    
    return path_s3


@router.post(
    "/cbir",
    response_model=APIResponse,
    status_code=status.HTTP_200_OK,
    tags=["CBIR"],
    summary="Content-Based Image Retrieval",
    description=f"Search similar batik images berdasarkan gambar input. Rate limit: {CBIR_LIMIT}",
)
@limiter.limit(CBIR_LIMIT)
async def search_similar_images(request: Request, file: UploadFile = File(...)):
    """
    Content-Based Image Retrieval (CBIR) untuk search gambar batik serupa.
    
    Request:
    - file: Query image dalam format JPEG, PNG, atau WebP (max 5MB)
    
    Response:
    - results: List hasil CBIR dengan image_id, similarity, dan motif
    - search_time: Waktu pencarian dalam detik
    
    Errors:
    - 400: Invalid file (format, size, atau content type)
    - 401: Missing atau invalid API Key
    - 504: Search timeout (lebih dari 30 detik)
    - 500: Internal server error
    
    Rate limit:
    - 5 requests per minute per IP (paling berat, butuh RAM besar)
    """
    try:
        # Check if CBIR engine loaded
        model_loader = get_model_loader()
        cbir_engine = model_loader.get_cbir_engine()
        if not model_loader.is_cbir_loaded() or cbir_engine is None:
            logger.error("CBIR model not loaded")
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=ResponseBuilder.error(
                    message="Model not loaded",
                    status=503,
                    errors=["CBIR engine loading failed or not initialized"],
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
        
        # Run search dengan timeout
        try:
            image = Image.open(io.BytesIO(file_content))
            start_time = time.time()
            cluster_id, cbir_results = await asyncio.wait_for(
                asyncio.to_thread(cbir_engine.search, image, settings.CBIR_TOP_K),
                timeout=settings.INFERENCE_TIMEOUT_SECONDS,
            )
            search_time = round(time.time() - start_time, 4)
        except asyncio.TimeoutError:
            logger.error("CBIR search timeout")
            return JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content=ResponseBuilder.error(
                    message="Search timeout",
                    status=504,
                    errors=["CBIR search exceeded 30 seconds timeout"],
                ).model_dump(),
            )
        
        # Attach presigned URLs to results when S3 is configured
        try:
            storage = get_s3_storage()
            for item in cbir_results:
                # Use path_s3 from CSV index for accurate S3 key
                path_s3 = _get_path_s3_for_result(item)
                item["image_path"] = path_s3
                item["image_url"] = storage.generate_presigned_url(
                    path_s3, bucket_name=settings.S3_BUCKET_NAME_CBIR or None
                )
        except Exception:
            # don't fail the whole request if URL generation fails
            logger.debug("Presign URL generation skipped or failed", exc_info=True)

        response_data = {
            "results": cbir_results,
            "search_time": search_time,
            "cluster_id": cluster_id,
            "result_count": len(cbir_results),
        }

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=ResponseBuilder.success(
                data=response_data,
                message="CBIR search successful",
                status=200,
            ).model_dump(),
        )    
    except Exception as e:
        logger.error(f"CBIR search error: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ResponseBuilder.error(
                message="Internal server error",
                status=500,
                errors=["An unexpected error occurred"],
            ).model_dump(),
        )
