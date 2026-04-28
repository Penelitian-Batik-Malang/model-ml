import io
import base64
from PIL import Image
from fastapi import APIRouter, File, UploadFile, Request, HTTPException
from typing import Dict, Any, List

from services.motif_classification_engine import predict_motif, get_motif_labels
from services.type_classification_engine import predict_type, get_type_labels

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

@router.get("/motif/labels", response_model=List[str])
def motif_labels():
    """
    Mengembalikan daftar label motif batik yang dapat dideteksi.
    """
    return get_motif_labels()

@router.post("/motif")
async def detect_motif(request: Request, file: UploadFile = File(None)) -> Dict[str, Any]:
    """
    Deteksi label motif batik.
    """
    img = await _get_image_from_request(request, file)
    result = predict_motif(img)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
        
    if result.get("predictions"):
        top = result["predictions"][0]
        resp = {
            "label": top["class_name"],
            "confidence": top["confidence"],
            "percentage": top["percentage"]
        }
        if "prep_time_ms" in result:
            resp["meta"] = {
                "prep_ms": result.get("prep_time_ms"),
                "infer_ms": result.get("infer_time_ms"),
            }
        return resp
    raise HTTPException(status_code=400, detail="No predictions")

@router.get("/type/labels", response_model=List[str])
def type_labels():
    """
    Mengembalikan daftar label jenis batik yang dapat dideteksi.
    """
    return get_type_labels()

@router.post("/type")
async def detect_type(request: Request, file: UploadFile = File(None)) -> Dict[str, Any]:
    """
    Deteksi jenis batik (tulis/cap).
    """
    img = await _get_image_from_request(request, file)
    result = predict_type(img)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
        
    if result.get("predictions"):
        top_pred = result["predictions"][0]
        return {
            "label": top_pred["class_name"],
            "confidence": top_pred["confidence"],
            "percentage": top_pred["percentage"]
        }
    raise HTTPException(status_code=400, detail="No predictions")
