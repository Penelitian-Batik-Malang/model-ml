from fastapi import APIRouter
from app.controllers import health, classify, cbir, tulis, color_faiss, fashion, search, detection

# Create main API router
api_router = APIRouter(prefix="/api")

# Include routes dari semua controllers
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(classify.router, tags=["Classification"])
api_router.include_router(cbir.router, tags=["CBIR"])
api_router.include_router(tulis.router, tags=["Tulis"])
api_router.include_router(color_faiss.router, tags=["Color FAISS"])
api_router.include_router(fashion.router, tags=["Fashion"])
api_router.include_router(search.router, tags=["Search"])
api_router.include_router(detection.router, tags=["Detection"])
