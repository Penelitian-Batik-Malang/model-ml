import logging
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ColorFaissRetriever:
    """FAISS-based retrieval for dominant color vectors (scenario s1)."""

    def __init__(self, data_dir: str, scenario: str = "s1", candidate_multiplier: int = 20) -> None:
        self.data_dir = Path(data_dir)
        self.scenario = scenario
        self.candidate_multiplier = candidate_multiplier
        self._cache: Dict[int, dict] = {}

    def _bundle_paths(self, num_clusters: int) -> Dict[str, Path]:
        base = f"{self.scenario}_c{num_clusters}"
        return {
            "index": self.data_dir / f"faiss_index_{base}.index",
            "meta": self.data_dir / f"faiss_meta_{base}.csv",
            "vectors": self.data_dir / f"raw_vectors_{base}.npy",
            "slot_means": self.data_dir / f"slot_means_{base}.npy",
        }

    def _load_bundle(self, num_clusters: int) -> dict:
        if num_clusters in self._cache:
            return self._cache[num_clusters]

        paths = self._bundle_paths(num_clusters)
        missing = [name for name, path in paths.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing FAISS artifacts: {', '.join(missing)}")

        index = faiss.read_index(str(paths["index"]))
        meta = pd.read_csv(paths["meta"])
        raw_vectors = np.load(paths["vectors"])
        slot_means = np.load(paths["slot_means"])

        meta_by_id = meta.set_index("vec_id")

        bundle = {
            "index": index,
            "meta": meta,
            "meta_by_id": meta_by_id,
            "raw_vectors": raw_vectors,
            "slot_means": slot_means,
        }
        self._cache[num_clusters] = bundle
        logger.info("Loaded FAISS bundle for c%s", num_clusters)
        return bundle

    def search(
        self,
        feature_vector: np.ndarray,
        num_clusters: int,
        selected_slots: Optional[List[int]],
        top_k: int,
    ) -> List[dict]:
        bundle = self._load_bundle(num_clusters)
        index = bundle["index"]
        raw_vectors = bundle["raw_vectors"]
        slot_means = bundle["slot_means"]
        meta_by_id = bundle["meta_by_id"]

        if feature_vector.size == 0:
            return []

        if selected_slots is None or len(selected_slots) == 0:
            selected_slots = list(range(num_clusters))

        per_color_dim = feature_vector.shape[0] // num_clusters
        full_query = feature_vector.copy()

        unselected = [i for i in range(num_clusters) if i not in selected_slots]
        for idx in unselected:
            start = idx * per_color_dim
            end = start + per_color_dim
            full_query[start:end] = slot_means[idx]

        n_candidates = min(max(top_k, 1) * self.candidate_multiplier, index.ntotal)
        distances, indices = index.search(full_query.reshape(1, -1).astype(np.float32), n_candidates)
        candidate_ids = [v for v in indices[0] if v != -1]

        rescored = []
        for vec_id in candidate_ids:
            db_vec = raw_vectors[vec_id]
            sq_dist = 0.0
            for idx in selected_slots:
                start = idx * per_color_dim
                end = start + per_color_dim
                diff = feature_vector[start:end] - db_vec[start:end]
                sq_dist += float(np.dot(diff, diff))
            rescored.append((vec_id, float(np.sqrt(sq_dist))))

        rescored.sort(key=lambda x: x[1])

        results = []
        for rank, (vec_id, dist) in enumerate(rescored[:top_k], start=1):
            if vec_id not in meta_by_id.index:
                continue
            row = meta_by_id.loc[vec_id]
            results.append(
                {
                    "rank": rank,
                    "vec_id": int(vec_id),
                    "image_id": int(row.get("image_id", vec_id)),
                    "image_path": str(row.get("image_path", "")),
                    "label": str(row.get("label", "")),
                    "distance": dist,
                }
            )

        return results


_color_faiss_retriever: Optional[ColorFaissRetriever] = None


def get_color_faiss_retriever(data_dir: str, scenario: str, candidate_multiplier: int) -> ColorFaissRetriever:
    global _color_faiss_retriever
    if _color_faiss_retriever is None:
        _color_faiss_retriever = ColorFaissRetriever(data_dir, scenario, candidate_multiplier)
    return _color_faiss_retriever
