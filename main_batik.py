from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.search_routes import router as search_router
from routes.detection_routes import router as detection_router

app = FastAPI(title="ML API - Batik Service (Python 3.9+ / TF 2.x / PyTorch)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_router, prefix="/search", tags=["Search"])
app.include_router(detection_router, prefix="/detection", tags=["Detection"])

@app.get("/health")
def health():
    return {"status": "ok", "service": "batik"}
