from pydantic import BaseModel, Field
 
from config import DEFAULT_TOP_K
 
 
class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=512,
        example="batik parang dengan motif diagonal warna coklat tua",
    )
    top_k: int = Field(
        DEFAULT_TOP_K,
        ge=1,
        le=50,
        description="Jumlah gambar yang dikembalikan (1–50)",
    )
 
 
class ImageResult(BaseModel):
    rank: int = Field(..., description="Peringkat hasil (1 = paling relevan)")
    score: float = Field(..., description="Cosine similarity score [0–1]")
    image_id: int = Field(..., description="ID gambar untuk endpoint /Data_Untuk_Clustering/{image_id}")
    category: str = Field(..., description="Kategori batik")
    filename: str = Field(..., description="Nama file gambar")
    image_url: str = Field(..., description="URL absolute untuk mengambil gambar")
 
 
class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: list[ImageResult]
 
 
class GalleryInfo(BaseModel):
    total_images: int
    total_categories: int
    embedding_dim: int
    categories: dict[str, int]
