import asyncio
import logging
import os
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi import Request

from app.config.rate_limit import CBIR_LIMIT, limiter
from app.config.settings import settings
from app.services.retrieval import search_top_k
from app.services.multimodal_state import get_multimodal_state, is_multimodal_ready
from app.utils.response import ResponseBuilder
from pydantic import BaseModel

class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 5

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/retrieval")

@router.post(
    "/text",
    status_code=status.HTTP_200_OK,
    summary="Pencarian batik dengan teks (Multimodal)",
)
@limiter.limit(CBIR_LIMIT)
async def search_text(request: Request, body: TextSearchRequest) -> Dict[str, Any]:
    if not is_multimodal_ready():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ResponseBuilder.error(
                message="Service Unavailable",
                status=503,
                errors=["Model multimodal sedang dimuat, coba lagi sebentar."],
            ).model_dump(),
        )
    
    try:
        k_eff, results = await asyncio.wait_for(
            asyncio.to_thread(
                search_top_k, 
                body.query, 
                body.top_k, 
                get_multimodal_state(), 
                str(request.base_url).rstrip('/'),
                "/api/retrieval/raw-image" # Mengarahkan endpoint image lokal ke URL fallback raw-image
            ),
            timeout=settings.INFERENCE_TIMEOUT_SECONDS,
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=ResponseBuilder.success(
                data={"total": k_eff, "results": results},
                message="Pencarian teks berhasil",
                status=200,
            ).model_dump(),
        )
    except asyncio.TimeoutError:
        logger.error("Text search timeout")
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content=ResponseBuilder.error(
                message="Search timeout",
                status=504,
                errors=["Search exceeded timeout"],
            ).model_dump(),
        )
    except Exception as exc:
        logger.error("Text search error: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ResponseBuilder.error(
                message="Internal server error",
                status=500,
                errors=["An unexpected error occurred"],
            ).model_dump(),
        )


@router.get(
    "/raw-image/{gallery_index}",
    summary="[Fallback] Menampilkan gambar mentah .jpg dari file .npz berdasarkan urutan index array",
    description="Endpoint darurat ini HANYA digunakan oleh backend ML untuk menampilkan gambar secara lokal jika layanan cloud S3 mati. Nilai gallery_index BUKAN ID Database, melainkan urutan baris di CSV/NPZ.",
)
async def get_raw_image(gallery_index: int):
    if not is_multimodal_ready():
        raise HTTPException(status_code=503, detail="Multimodal model not ready")
        
    state = get_multimodal_state()
    paths = state.get("gallery_paths", [])
    
    if gallery_index < 0 or gallery_index >= len(paths):
        raise HTTPException(status_code=404, detail="Image index out of bounds")
        
    image_path = paths[gallery_index]
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image file not found on disk")
        
    return FileResponse(image_path)
