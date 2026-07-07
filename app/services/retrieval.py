import logging
import os

import torch

from app.config.settings import settings
from app.services.s3 import get_s3_presigned_url

logger = logging.getLogger(__name__)


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
        
        # S3 case-sensitive: pastikan ekstensi lowercase (.JPG -> .jpg) karena 
        # file di bucket S3 menggunakan ekstensi lowercase.
        filename_s3 = filename
        if '.' in filename_s3:
            base_name, ext = filename_s3.rsplit('.', 1)
            filename_s3 = f"{base_name}.{ext.lower()}"

        # Gunakan presigned URL — sama seperti pola color_faiss.py.
        # S3 key: {folder_category}/{filename_s3}
        # Bucket: settings.AWS_BUCKET_SIGNATURE_DRIVE (batik-signature-gdrive)
        # Presigned URL membawa signature auth sehingga aman untuk bucket privat
        # dan tidak akan menghasilkan 403 seperti direct public URL.
        s3_key    = f"{category}/{filename_s3}"
        image_url = get_s3_presigned_url(s3_key)

        # Fallback last resort: endpoint raw-image lokal
        # (hanya aktif jika S3 tidak dikonfigurasi atau credentials tidak valid)
        if not image_url:
            logger.warning(
                "Presigned URL gagal untuk %s, fallback ke raw-image endpoint", s3_key
            )
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