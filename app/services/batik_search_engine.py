import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms

from app.config.settings import settings


features = np.array([])
kmeans = None
cluster_df = pd.DataFrame()
feature_extractor = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

query_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _resolve_file(filename: str) -> Path:
    candidates = [
        Path(settings.DATA_PATH) / filename,
        Path(settings.MODEL_PATH) / filename,
        Path(settings.CHECKPOINTS_PATH) / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_feature_extractor() -> torch.nn.Module:
    extractor = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
    extractor.classifier[-1] = torch.nn.Identity()
    return extractor.to(DEVICE).eval()


def load_cbir_models() -> bool:
    global features, kmeans, cluster_df, feature_extractor

    try:
        features_path = _resolve_file(settings.CBIR_FEATURES_FILE)
        kmeans_path = _resolve_file(settings.CBIR_KMEANS_FILE)
        indexed_path = _resolve_file(settings.CBIR_INDEX_FILE)

        if features_path.exists():
            features = np.load(features_path)
        else:
            print(f"[Batik Search] No NPY features file: {features_path}")

        if kmeans_path.exists():
            kmeans = joblib.load(kmeans_path)
        else:
            print(f"[Batik Search] No KMeans model: {kmeans_path}")

        if indexed_path.exists():
            cluster_df = pd.read_csv(indexed_path)
        else:
            print("[Batik Search] Indexed database not found")

        feature_extractor = _load_feature_extractor()

        print("[Batik Search] Models loaded successfully")
        return True
    except Exception as exc:
        print(f"[Batik Search] Error loading models: {exc}")
        return False


def extract_feature(image_pil: Image.Image) -> np.ndarray:
    if feature_extractor is None:
        load_cbir_models()

    img = image_pil.convert("RGB")
    tensor = query_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = feature_extractor(tensor).cpu().numpy()
    return feat


def retrieve_top_n(query_feature: np.ndarray, top_n: int = 10):
    if kmeans is None or len(cluster_df) == 0:
        load_cbir_models()

    if query_feature.ndim == 1:
        query_feature = query_feature.reshape(1, -1)

    cluster_id = int(kmeans.predict(query_feature)[0])
    cluster_members = cluster_df[cluster_df["cluster"] == cluster_id]

    if cluster_members.empty:
        return cluster_id, pd.DataFrame()

    member_indices = cluster_members.index
    member_features = features[member_indices]

    similarities = cosine_similarity(query_feature, member_features).flatten()

    result_df = cluster_members.copy()
    result_df["similarity"] = similarities
    result_df = result_df.sort_values(by="similarity", ascending=False).head(top_n)

    return cluster_id, result_df


def search_general_batik(image_pil: Image.Image, top_n: int = 10) -> dict:
    try:
        query_feat = extract_feature(image_pil)
        cluster_id, top_images_df = retrieve_top_n(query_feat, top_n=top_n)

        if top_images_df.empty:
            return {"success": False, "message": "No similar images found", "cluster_id": cluster_id}

        results = []
        for _, row in top_images_df.iterrows():
            s3_path = str(row.get("path_s3", "")).replace("\\", "/")
            results.append(
                {
                    "path_s3": s3_path,
                    "label": row.get("label", ""),
                    "cluster": int(row.get("cluster", cluster_id)),
                    "similarity": float(row.get("similarity", 0.0)),
                }
            )

        return {
            "success": True,
            "cluster_id": cluster_id,
            "results": results,
            "message": f"Found {len(results)} similar images in cluster {cluster_id}",
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}
