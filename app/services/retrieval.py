import os

import torch

from app.config.settings import settings
from app.services.s3 import get_s3_presigned_url


# Direct public S3 base URL untuk bucket batik-signature-gdrive
# Format: {S3_ENDPOINT_URL}/{bucket}/{folder_category}/{filename}
_S3_ENDPOINT = settings.S3_ENDPOINT_URL.rstrip('/')
_S3_BUCKET   = settings.AWS_BUCKET_SIGNATURE_DRIVE


def _build_s3_direct_url(category: str, filename: str) -> str:
    """
    Bangun URL S3 langsung (public) tanpa presigned.
    Path di bucket: {folder_category}/{filename}
    """
    if not _S3_ENDPOINT or not _S3_BUCKET:
        return ""
    # URL-encode spasi di nama file/folder agar bisa diakses browser
    safe_cat  = category.replace(" ", "%20")
    safe_file = filename.replace(" ", "%20")
    return f"{_S3_ENDPOINT}/{_S3_BUCKET}/{safe_cat}/{safe_file}"


def encode_query(query: str, model, tokenizer, device) -> torch.Tensor:
    enc = tokenizer(
        query,
        padding="max_length",
        truncation=True,
        max_length=settings.MAX_TEXT_LEN,
        return_tensors="pt",
    )

    with torch.no_grad():
        emb = model.encode_text(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
        )
    return emb  # (1, D)


def search_top_k(query: str, top_k: int, state: dict, base_url: str, image_endpoint_prefix: str):
    model       = state["model"]
    tokenizer   = state["tokenizer"]
    device      = state["device"]
    gallery_emb = state["gallery_embeddings"]
    paths       = state["gallery_paths"]
    categories  = state["gallery_categories"]

    query_emb = encode_query(query, model, tokenizer, device)  # (1, D)

    gal_on_device = gallery_emb.to(device)
    scores = torch.matmul(query_emb, gal_on_device.T).squeeze(0)  # (N,)

    k_eff = min(top_k, len(paths))
    topk_scores, topk_indices = scores.topk(k_eff)
    topk_scores = topk_scores.cpu().tolist()
    topk_indices = topk_indices.cpu().tolist()

    results = []
    for rank, idx in enumerate(topk_indices):
        category = categories[idx]
        filename = os.path.basename(paths[idx])

        # 1. Coba bangun direct public S3 URL (category/filename)
        image_url = _build_s3_direct_url(category, filename)

        # 2. Fallback: presigned URL (jika bucket privat)
        if not image_url:
            s3_key    = f"{category}/{filename}"
            image_url = get_s3_presigned_url(s3_key)

        # 3. Last resort: endpoint raw-image lokal
        if not image_url:
            image_url = f"{base_url}{image_endpoint_prefix}/{idx}"

        results.append({
            "rank"         : rank + 1,
            "score"        : round(topk_scores[rank], 6),
            "gallery_index": idx,
            "category"     : category,
            "filename"     : filename,
            "image_url"    : image_url,
        })

    return k_eff, results