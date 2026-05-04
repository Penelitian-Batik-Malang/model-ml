from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.fashion_segmentation_routes import router as fashion_router, init_cbir_db
from routes.search_routes import router as search_router
from routes.detection_routes import router as detection_router
from utils.session_handler import cleanup_old_sessions

app = FastAPI(title="ML API (Python 3.7 / TF 1.15 / PyTorch 1.13)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event() -> None:
    cleanup_old_sessions(max_age_hours=2)
    init_cbir_db()

app.include_router(fashion_router, prefix="/fashion", tags=["Fashion"])
app.include_router(search_router, prefix="/search", tags=["Search"])
app.include_router(detection_router, prefix="/detection", tags=["Detection"])

@app.get("/health")
def health():
    return {"status": "ok", "service": "fashionpedia"}
