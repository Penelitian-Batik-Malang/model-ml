import os
 
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
 
from app.schemas.search import GalleryInfo, SearchRequest, SearchResponse
from app.services.retrieval import search_top_k
from app.state import get_state
from config import IMAGE_ENDPOINT_PREFIX
 
router = APIRouter()
 
 
@router.get("/", summary="Health check", tags=["Umum"])
def root():
    state = get_state()
    return {
        "status": "ok",
        "service": "Batik Image Retrieval API",
        "gallery": len(state.get("gallery_paths", [])),
        "device": str(state.get("device", "unknown")),
    }
 
 
@router.get("/info", response_model=GalleryInfo, summary="Statistik gallery", tags=["Umum"])
def gallery_info():
    state = get_state()
    paths = state["gallery_paths"]
    cats = state["gallery_categories"]
    emb = state["gallery_embeddings"]
 
    cat_count: dict[str, int] = {}
    for c in cats:
        cat_count[c] = cat_count.get(c, 0) + 1
 
    return GalleryInfo(
        total_images=len(paths),
        total_categories=len(cat_count),
        embedding_dim=int(emb.shape[1]),
        categories=dict(sorted(cat_count.items())),
    )
 
 
@router.post(
    "/search-text",
    response_model=SearchResponse,
    summary="Cari gambar berdasarkan teks",
    tags=["Retrieval"],
)
def search(req: SearchRequest, request: Request):
    state = get_state()
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model belum siap, coba lagi sebentar.")
 
    base_url = str(request.base_url).rstrip("/")
    k_eff, results = search_top_k(req.query, req.top_k, state, base_url, IMAGE_ENDPOINT_PREFIX)
 
    return SearchResponse(query=req.query, top_k=k_eff, results=results)
 
 
@router.get(
    IMAGE_ENDPOINT_PREFIX + "/{image_id}",
    summary="Ambil gambar berdasarkan ID",
    tags=["Gambar"],
    responses={
        200: {"content": {"image/jpeg": {}}},
        404: {"description": "Gambar tidak ditemukan"},
    },
)
def get_image(image_id: int):
    state = get_state()
    paths = state.get("gallery_paths", [])
 
    if image_id < 0 or image_id >= len(paths):
        raise HTTPException(
            status_code=404,
            detail=f"image_id {image_id} tidak valid. Range: 0–{len(paths) - 1}",
        )
 
    img_path = paths[image_id]
    if not os.path.exists(img_path):
        raise HTTPException(
            status_code=404,
            detail=f"File tidak ditemukan di disk: {img_path}",
        )
 
    ext = os.path.splitext(img_path)[1].lower()
    mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
    }
    media_type = mime_map.get(ext, "image/jpeg")
 
    return FileResponse(img_path, media_type=media_type)
