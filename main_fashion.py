from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.fashion_segmentation_routes import router as fashion_router, init_cbir_db
from utils.session_handler import cleanup_old_sessions

app = FastAPI(title="ML API - Fashion Service (Python 3.7 / TF 1.15.0)")

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

@app.get("/health")
def health():
    return {"status": "ok", "service": "fashion"}
