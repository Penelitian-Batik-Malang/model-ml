import os
 
import torch
 
from config import MAX_TEXT_LEN

def encode_query(query: str, model, tokenizer, device) -> torch.Tensor:
    enc = tokenizer(
        query,
        padding="max_length",
        truncation=True,
        max_length=MAX_TEXT_LEN,
        return_tensors="pt",
    )
 
    with torch.no_grad():
        emb = model.encode_text(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
        )
    return emb  # (1, D)
 
def search_top_k(query: str, top_k: int, state: dict, base_url: str, image_endpoint_prefix: str):
    model = state["model"]
    tokenizer = state["tokenizer"]
    device = state["device"]
    gallery_emb = state["gallery_embeddings"]
    paths = state["gallery_paths"]
    categories = state["gallery_categories"]
 
    query_emb = encode_query(query, model, tokenizer, device)  # (1, D)
 
    gal_on_device = gallery_emb.to(device)
    scores = torch.matmul(query_emb, gal_on_device.T).squeeze(0)  # (N,)
 
    k_eff = min(top_k, len(paths))
    topk_scores, topk_indices = scores.topk(k_eff)
    topk_scores = topk_scores.cpu().tolist()
    topk_indices = topk_indices.cpu().tolist()
 
    results = [
        {
            "rank": rank + 1,
            "score": round(topk_scores[rank], 6),
            "image_id": idx,
            "category": categories[idx],
            "filename": os.path.basename(paths[idx]),
            "image_url": f"{base_url}{image_endpoint_prefix}/{idx}",
        }
        for rank, idx in enumerate(topk_indices)
    ]
 
    return k_eff, results
 