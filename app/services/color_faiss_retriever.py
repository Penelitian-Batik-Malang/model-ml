import logging
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ColorFaissRetriever:
    """FAISS-based retrieval for dominant color vectors using s2_nodedup_max5 artifacts."""

    def __init__(self, data_dir: str, scenario: str = "s2_careful_max5") -> None:
        self.data_dir = Path(data_dir)
        self.scenario = scenario
        self._cache: Dict[int, dict] = {}

    def _bundle_paths(self, num_clusters: int) -> Dict[str, Path]:
        if self.scenario == "s2_careful_max5":
            return {
                "index": self.data_dir / "faiss_index_s2_careful_max5.index",
                "meta": self.data_dir / "faiss_meta_s2_careful_max5.csv",
                "vectors": self.data_dir / "padded_vectors_s2_careful_max5.npy",
                "slot_means": self.data_dir / "slot_means_s2_careful_max5.npy",
            }

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

        per_color_dim = 3 # For Skenario 2 (L, a, b)
        k_opt = feature_vector.shape[0] // per_color_dim

        # Limit k_opt to num_clusters just in case
        k_opt = min(k_opt, num_clusters)
        
        # If no selected_slots provided, default to range(min(top_colors_requested, k_opt))
        # Wait, the notebook says: combo = tuple(range(num_colors_to_search)), where num_colors_to_search = min(i, k_opt)
        # But in API we receive selected_slots (1-based indices converted to 0-based).
        # We should use selected_slots directly. If none, we use up to k_opt.
        if selected_slots is None or len(selected_slots) == 0:
            selected_slots = list(range(k_opt))

        # Pad to full query using slot_means
        total_dim = num_clusters * per_color_dim
        full_query = np.zeros(total_dim, dtype=np.float32)
        full_query[:k_opt * per_color_dim] = feature_vector[:k_opt * per_color_dim]

        if k_opt < num_clusters:
            for slot_idx in range(k_opt, num_clusters):
                start = slot_idx * per_color_dim
                end = start + per_color_dim
                full_query[start:end] = slot_means[slot_idx]

        # Masking unselected slots
        unselected = [i for i in range(num_clusters) if i not in selected_slots]
        for idx in unselected:
            start = idx * per_color_dim
            end = start + per_color_dim
            full_query[start:end] = slot_means[idx]

        n_candidates = min(top_k * 10, index.ntotal)
        distances, indices = index.search(full_query.reshape(1, -1).astype(np.float32), n_candidates)
        candidate_ids = [v for v in indices[0] if v != -1]

        rescored = []
        for vec_id in candidate_ids:
            db_vec = raw_vectors[vec_id]
            sq_dist = 0.0
            for idx in selected_slots:
                start = idx * per_color_dim
                end = start + per_color_dim
                diff = full_query[start:end] - db_vec[start:end]
                sq_dist += float(np.dot(diff, diff))
            rescored.append((vec_id, float(np.sqrt(sq_dist))))

        rescored.sort(key=lambda x: x[1])

        results = []
        for vec_id, dist in rescored:
            if vec_id not in meta_by_id.index:
                continue
            row = meta_by_id.loc[vec_id]
            
            color_names = []
            for col in row.index:
                if str(col).startswith("color_name_") and pd.notna(row[col]):
                    color_names.append(str(row[col]))

            results.append(
                {
                    "rank": len(results) + 1,
                    "vec_id": int(vec_id),
                    "image_id": int(row.get("image_id", vec_id)),
                    "image_path": str(row.get("image_path", "")),
                    "label": str(row.get("label", "")),
                    "color_names_label": color_names,
                    "distance": dist,
                }
            )
            
            if len(results) >= top_k:
                break

        return results


_color_faiss_retriever: Optional[ColorFaissRetriever] = None


def get_color_faiss_retriever(data_dir: str) -> ColorFaissRetriever:
    global _color_faiss_retriever
    if _color_faiss_retriever is None:
        _color_faiss_retriever = ColorFaissRetriever(data_dir)
    return _color_faiss_retriever
