import logging
import time
from fastapi import APIRouter, Request, status
from app.schemas.response import APIResponse, HealthCheckResponse
from app.services.model_loader import get_model_loader
from app.config.settings import settings
from app.config.rate_limit import HEALTH_LIMIT, limiter

logger = logging.getLogger(__name__)
router = APIRouter()

# Start time untuk tracking uptime
START_TIME = time.time()


@router.get(
    "/health",
    response_model=APIResponse,
    status_code=status.HTTP_200_OK,
    tags=["Health"],
    summary="Health check endpoint",
    description="Check service health dan status model loading. Tidak memerlukan API Key.",
)
@limiter.limit(HEALTH_LIMIT)
async def health_check(request: Request):
    """
    Health check endpoint.
    
    Response:
    - status: 200 jika service healthy
    - model_loaded: true jika model sudah di-load
    - uptime: uptime service dalam detik
    
    Catatan:
    - Endpoint ini TIDAK memerlukan API Key (skip validation)
    - Selalu return status 200 dengan informasi detail
    """
    try:
        model_loader = get_model_loader()
        is_model_loaded = model_loader.is_model_loaded()
        model_exists = model_loader.check_model_exists(settings.MODEL_PATH)
        uptime = time.time() - START_TIME
        
        health_data = {
            "status": "healthy",
            "is_model_loaded": is_model_loaded,
            "models": {
                "motif": model_loader.is_motif_loaded(),
                "tulis": model_loader.is_tulis_loaded(),
                "cbir": model_loader.is_cbir_loaded(),
            },
            "model_exists": model_exists,
            "uptime": round(uptime, 2),
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
        }
        
        if not is_model_loaded:
            logger.warning("Model not loaded during health check")
        
        return APIResponse(
            status=200,
            message="Service is healthy",
            data=health_data,
            errors=[],
        )
    
    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        return APIResponse(
            status=503,
            message="Service unavailable",
            data=None,
            errors=[str(e)],
        )
