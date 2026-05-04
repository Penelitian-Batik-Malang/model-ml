import logging
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torchvision import models, transforms

logger = logging.getLogger(__name__)


class CBIREngine:
    """Content-Based Image Retrieval engine using ConvNeXt features and KMeans."""

    def __init__(
        self,
        features_path: str,
        kmeans_path: str,
        indexed_db_path: str,
        weights_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self.features_path = Path(features_path)
        self.kmeans_path = Path(kmeans_path)
        self.indexed_db_path = Path(indexed_db_path)
        self.weights_path = Path(weights_path) if weights_path else None
        self.device = torch.device(device)

        self.features = None
        self.kmeans = None
        self.indexed_df: Optional[pd.DataFrame] = None
        self.feature_extractor = None
        self.query_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.loaded = False

    def load(self) -> bool:
        """Load CBIR assets (features, kmeans, index, feature extractor)."""
        if not self.features_path.exists():
            logger.error("CBIR features file not found: %s", self.features_path)
            return False
        if not self.kmeans_path.exists():
            logger.error("CBIR kmeans file not found: %s", self.kmeans_path)
            return False
        if not self.indexed_db_path.exists():
            logger.error("CBIR index file not found: %s", self.indexed_db_path)
            return False

        try:
            self.features = np.load(self.features_path)
            self.kmeans = joblib.load(self.kmeans_path)
            self.indexed_df = pd.read_csv(self.indexed_db_path)

            if "cluster" not in self.indexed_df.columns:
                cluster_labels = self.kmeans.predict(self.features)
                self.indexed_df["cluster"] = cluster_labels

            self.feature_extractor = self._load_feature_extractor()
            self.loaded = self.feature_extractor is not None

            logger.info("CBIR assets loaded successfully")
            return self.loaded
        except Exception as exc:
            logger.error("Failed to load CBIR assets: %s", exc, exc_info=True)
            self.loaded = False
            return False

    def _load_feature_extractor(self):
        model = models.convnext_small(weights=None)
        model.classifier[-1] = torch.nn.Identity()

        if self.weights_path and self.weights_path.exists():
            try:
                state_dict = torch.load(self.weights_path, map_location=self.device)
                model.load_state_dict(state_dict, strict=False)
                logger.info("CBIR feature extractor weights loaded: %s", self.weights_path)
            except Exception as exc:
                logger.warning("Failed to load feature extractor weights: %s", exc)
        else:
            logger.warning("Feature extractor weights not found; using random weights")

        model = model.to(self.device).eval()
        return model

    def _extract_feature(self, image: Image.Image) -> np.ndarray:
        if self.feature_extractor is None:
            raise RuntimeError("Feature extractor not loaded")

        tensor = self.query_transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.feature_extractor(tensor).cpu().numpy()
        return feat

    def search(self, image: Image.Image, top_k: int = 5) -> Tuple[int, List[dict]]:
        if not self.loaded or self.indexed_df is None:
            raise RuntimeError("CBIR engine not loaded")

        query_feat = self._extract_feature(image)
        if query_feat.ndim == 1:
            query_feat = query_feat.reshape(1, -1)

        cluster_id = int(self.kmeans.predict(query_feat)[0])
        cluster_members = self.indexed_df[self.indexed_df["cluster"] == cluster_id]

        if cluster_members.empty:
            return cluster_id, []

        member_indices = cluster_members.index.to_numpy()
        member_features = self.features[member_indices]
        similarities = cosine_similarity(query_feat, member_features).flatten()

        result_df = cluster_members.copy()
        result_df["similarity"] = similarities
        result_df = result_df.sort_values(by="similarity", ascending=False).head(top_k)

        results = []
        for _, row in result_df.iterrows():
            image_id = (
                row.get("path_s3")
                or row.get("filename")
                or row.get("path_gdrive")
                or "unknown"
            )
            label = row.get("label") or row.get("label_name") or "Unknown"
            results.append(
                {
                    "image_id": str(image_id),
                    "similarity": float(row["similarity"]),
                    "motif": str(label),
                }
            )

        return cluster_id, results
