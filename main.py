import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer

from app.models.multimodal import MultimodalRetrievalModel
from app.routers.search import router as search_router
from app.services.gallery import build_gallery_if_missing
from app.state import clear_state, set_state
from config import CORS_ALLOW_ORIGINS, MODEL_PATH, PROJ_DIM, TOKENIZER_NAME


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Startup] Device         : {device}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}\n"
            f"Pastikan file sudah tersedia sebelum menjalankan server."
        )

    print(f"[Startup] Loading model  : {MODEL_PATH}")
    model = MultimodalRetrievalModel(proj_dim=PROJ_DIM).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.eval()

    print("Model dimuat dari:", MODEL_PATH)
    if isinstance(checkpoint, dict):
        for k in ["epoch", "val_loss"]:
            if k in checkpoint:
                print(f"  {k}: {checkpoint[k]}")

    print(f"[Startup] Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    gallery_embeddings, gallery_paths, gallery_categories = build_gallery_if_missing(model, device)

    set_state(
        model=model,
        tokenizer=tokenizer,
        gallery_embeddings=gallery_embeddings,
        gallery_paths=gallery_paths,
        gallery_categories=gallery_categories,
        device=device,
    )

    yield

    clear_state()
    print("\n[Shutdown] State cleared.")


app = FastAPI(
    title="Batik Image Retrieval API",
    description=(
        "Cari gambar batik berdasarkan deskripsi teks Bahasa Indonesia.\n\n"
        "Model: ConvNeXt-Small (image) + IndoBERT (text) + Projection 768d\n\n"
        "**Cara pakai:**\n"
        "1. `POST /search-text` dengan body `{\"query\": \"batik parang warna coklat\", \"top_k\": 5}`\n"
        "2. Gunakan `image_id` dari hasil untuk mengakses gambar via `GET /Data_Untuk_Clustering/{image_id}`"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_router)
