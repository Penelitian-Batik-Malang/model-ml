"""
Response schemas dengan format standard untuk seluruh API.
Format response:
{
    "status": int,
    "message": str,
    "data": [] / {} / null / T,
    "errors": [],
    "meta": {
        "size": int,
        "page": int,
        "total_page": int,
        "total_item": int
    }
}
"""

from typing import Any, Generic, List, Optional, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")


class Meta(BaseModel):
    """Metadata untuk pagination dan informasi response."""
    size: int = Field(default=0, description="Jumlah item dalam response")
    page: int = Field(default=1, description="Halaman saat ini")
    total_page: int = Field(default=0, description="Total halaman")
    total_item: int = Field(default=0, description="Total item seluruhnya")

    class Config:
        json_schema_extra = {
            "example": {
                "size": 10,
                "page": 1,
                "total_page": 5,
                "total_item": 50
            }
        }


class APIResponse(BaseModel, Generic[T]):
    """Response standard untuk semua endpoint API."""
    status: int = Field(description="HTTP status code")
    message: str = Field(description="Pesan response")
    data: Optional[Any] = Field(default=None, description="Data response (array, object, atau null)")
    errors: List[str] = Field(default_factory=list, description="List error messages")
    meta: Optional[Meta] = Field(default=None, description="Metadata pagination")

    class Config:
        json_schema_extra = {
            "example": {
                "status": 200,
                "message": "Success",
                "data": {},
                "errors": [],
                "meta": None
            }
        }


class HealthCheckResponse(BaseModel):
    """Response dari health check endpoint."""
    status: str = Field(description="Status kesehatan service")
    is_model_loaded: bool = Field(description="Apakah model sudah ter-load")
    uptime: float = Field(description="Uptime dalam detik")
    message: str = Field(default="OK", description="Pesan status")


class ClassificationRequest(BaseModel):
    """Request untuk endpoint classify."""
    pass


class ClassificationResponse(BaseModel):
    """Response dari endpoint classify."""
    motif: str = Field(description="Nama motif batik yang teridentifikasi")
    confidence: float = Field(description="Confidence score (0-1)")
    probability_distribution: dict = Field(description="Distribusi probabilitas semua kelas")


class CBIRRequest(BaseModel):
    """Request untuk endpoint CBIR."""
    pass


class CBIRResult(BaseModel):
    """Single result dari CBIR."""
    image_id: str = Field(description="ID gambar hasil search")
    similarity: float = Field(description="Similarity score (0-1)")
    motif: str = Field(description="Nama motif batik")


class CBIRResponse(BaseModel):
    """Response dari endpoint CBIR."""
    results: List[CBIRResult] = Field(description="Daftar hasil CBIR")
    search_time: float = Field(description="Waktu pencarian dalam detik")
