import asyncio

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import (
    BATIK_MOTIF_LABEL_JSON,
    BATIK_MOTIF_MODEL_H5,
    BATIK_SEARCH_FEATURES_NPY,
    BATIK_SEARCH_INDEXED_DB_CSV,
    BATIK_SEARCH_KMEANS_MODEL,
    BATIK_TYPE_MODEL_PT,
    FASHION_CBIR_FEATURES_NPZ,
    FASHION_CHECKPOINT_PATH,
    FASHION_CONFIG_FILE,
    FASHION_INFERENCE_SCRIPT,
    FASHION_LABEL_MAP_PATH,
)

from routes.fashion_segmentation_routes import router as fashion_router, init_cbir_db
from routes.search_routes import router as search_router
from routes.detection_routes import router as detection_router
from utils.session_handler import cleanup_old_sessions

app = FastAPI(title="Batik Digital Gallery API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=300)
    except asyncio.TimeoutError:
        return JSONResponse(status_code=504, content={"detail": "Request timed out"})


def _checkpoint_files_ready() -> bool:
    checkpoint_prefix = FASHION_CHECKPOINT_PATH.name
    return any(FASHION_CHECKPOINT_PATH.parent.glob(f"{checkpoint_prefix}*"))


def _health_ready() -> bool:
    required_paths = [
        BATIK_MOTIF_MODEL_H5,
        BATIK_MOTIF_LABEL_JSON,
        BATIK_TYPE_MODEL_PT,
        BATIK_SEARCH_FEATURES_NPY,
        BATIK_SEARCH_KMEANS_MODEL,
        BATIK_SEARCH_INDEXED_DB_CSV,
        FASHION_LABEL_MAP_PATH,
        FASHION_CONFIG_FILE,
        FASHION_INFERENCE_SCRIPT,
        FASHION_CBIR_FEATURES_NPZ,
    ]
    return all(path.exists() for path in required_paths) and _checkpoint_files_ready()

@app.on_event("startup")
def startup_event() -> None:
    cleanup_old_sessions(max_age_hours=2)
    init_cbir_db()

app.include_router(fashion_router, prefix="/fashion", tags=["Fashion"])
app.include_router(search_router, prefix="/search", tags=["Search"])
app.include_router(detection_router, prefix="/detection", tags=["Detection"])

@app.get("/health")
def health():
    ready = _health_ready()
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ok" if ready else "degraded", "ready": ready},
    )
