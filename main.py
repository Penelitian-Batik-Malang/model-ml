import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.config.rate_limit import limiter
from app.config.settings import settings
from app.middlewares.security import APIKeyMiddleware, SecurityHeadersMiddleware
from app.routes.api import api_router
from app.services.fashion_cbir_store import init_fashion_cbir_db
from app.services.model_loader import get_model_loader
from app.services.core.model_loader import ModelLoader as RecolorModelLoader
from app.utils.response import ResponseBuilder
from app.utils.session_handler import cleanup_old_sessions

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)
    logger.info("Debug mode: %s", settings.DEBUG)
    logger.info("Model path: %s", settings.MODEL_PATH)
    logger.info("Data path: %s", settings.DATA_PATH)
    logger.info("Checkpoints path: %s", settings.CHECKPOINTS_PATH)

    cleanup_old_sessions(max_age_hours=2)
    init_fashion_cbir_db()

    model_loader = get_model_loader()
    is_loaded = model_loader.load_model(
        settings.MODEL_PATH,
        settings.DATA_PATH,
        settings.CHECKPOINTS_PATH,
        settings.TPU_PATH,
    )

    if is_loaded:
        logger.info("Model loaded successfully at startup")
    else:
        logger.warning("Model failed to load at startup")

    recolor_loader = RecolorModelLoader.get_instance()
    try:
        import os
        from pathlib import Path

        fe_path = Path(settings.RECOLOR_FE_PATH)
        rd_path = Path(settings.RECOLOR_RD_PATH)

        # Auto-download recolor models from HF if not present locally
        if not fe_path.exists() or not rd_path.exists():
            logger.info("Recolor models not found locally, downloading from galeriBatikMalang/ML_models...")
            try:
                from huggingface_hub import hf_hub_download
                hf_token = os.getenv("HF_TOKEN")
                fe_path.parent.mkdir(parents=True, exist_ok=True)
                if not fe_path.exists():
                    downloaded = hf_hub_download(
                        repo_id="galeriBatikMalang/ML_models",
                        filename="FE.state_dict_paling_final.pt",
                        local_dir=str(fe_path.parent),
                        token=hf_token,
                    )
                    logger.info("Downloaded FE model to: %s", downloaded)
                if not rd_path.exists():
                    downloaded = hf_hub_download(
                        repo_id="galeriBatikMalang/ML_models",
                        filename="RD.state_dict_paling_final.pt",
                        local_dir=str(rd_path.parent),
                        token=hf_token,
                    )
                    logger.info("Downloaded RD model to: %s", downloaded)
            except Exception as dl_err:
                logger.warning("Failed to download recolor models from HF: %s", dl_err)

        recolor_loader.load(
            fe_path=str(fe_path),
            rd_path=str(rd_path),
            device=settings.RECOLOR_DEVICE,
        )
        logger.info("Recolor models loaded successfully at startup")
    except Exception as e:
        logger.warning("Recolor models failed to load: %s", e)

    yield

    logger.info("Shutting down %s", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="RESTful API untuk klasifikasi motif batik dan CBIR (Content-Based Image Retrieval)",
    lifespan=lifespan,
    docs_url="/docs",
    openapi_url="/openapi.json",
    redoc_url="/redoc",
)

from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=app.description,
        routes=app.routes,
    )
    
    # Daftarkan X-API-Key ke komponen Swagger UI
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key"
        }
    }
    # Terapkan skema ke seluruh routes secara global
    openapi_schema["security"] = [{"ApiKeyAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(APIKeyMiddleware)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    logger.warning("Rate limit exceeded for %s", request.client.host)
    return JSONResponse(
        status_code=429,
        content=ResponseBuilder.error(
            message="Rate limit exceeded",
            status=429,
            errors=["Too many requests. Please try again later."],
        ).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        field = ".".join(str(x) for x in error["loc"][1:])
        errors.append(f"{field}: {error['msg']}")

    return JSONResponse(
        status_code=422,
        content=ResponseBuilder.error(
            message="Validation error",
            status=422,
            errors=errors,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error("Unexpected error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ResponseBuilder.error(
            message="Internal server error",
            status=500,
            errors=["An unexpected error occurred"],
        ).model_dump(),
    )


app.include_router(api_router)


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Uvicorn server on %s:%s", settings.HOST, settings.PORT)
    logger.info("Workers: %s (fixed to 1 for DL inference)", settings.WORKERS)
    logger.info("Keep-alive timeout: %ss", settings.TIMEOUT_KEEP_ALIVE)

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
        timeout_keep_alive=settings.TIMEOUT_KEEP_ALIVE,
        reload=settings.DEBUG,
    )
