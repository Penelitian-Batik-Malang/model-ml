import io
import base64
from PIL import Image
from fastapi import APIRouter, File, UploadFile, Request, HTTPException, Form
from typing import Dict, Any

from services.batik_search_engine import search_general_batik

router = APIRouter()

async def _get_image_from_request(request: Request, file: UploadFile = None) -> Image.Image:
    if file is not None:
        try:
            image_bytes = await file.read()
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    
    try:
        data = await request.json()
        if data and "image" in data:
            b64 = data["image"].split(",")[-1]
            return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception:
        pass
    
    raise HTTPException(status_code=400, detail="No valid image provided")

@router.post("/general")
async def search_general(request: Request, file: UploadFile = File(None)) -> Dict[str, Any]:
    """
    Pencarian Umum (Query gambar untuk mencari batik serupa).
    Akan memanggil model Feature Extractor (ConvNeXt) dari Batik Service.
    """
    img = await _get_image_from_request(request, file)
    result = search_general_batik(img, top_n=10)
    
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", result.get("message", "Search failed")))
        
    return result

@router.post("/recommendation-after-like")
def search_recommendation_after_like(batik_id: str = Form(...)) -> Dict[str, Any]:
    """
    Rekomendasi setelah aksi Like.
    Mencari item terdekat di database berdasarkan ID batik yang dilike.
    """
    # TODO: Implement recommendation logic
    return {"status": "Not Implemented Yet"}

@router.post("/by-color")
def search_by_color(image: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Pencarian berdasarkan ekstraksi warna dominan (fitur mendatang).
    """
    # TODO: Implement color extraction and search
    return {"status": "Not Implemented Yet"}
