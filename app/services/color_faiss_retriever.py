"""
color_faiss_retriever.py

Retriever FAISS berbasis vektor warna dominan mengikuti logika
visualize_combo_retrieval dari notebook (Skenario 2 — L,a,b).

Perubahan utama dari versi sebelumnya:
  - Slot kosong diisi dengan image-specific fill (rata-rata warna aktif citra
    itu sendiri), BUKAN slot_means global.
  - Artifacts `slot_means_*.npy` tidak lagi dibutuhkan.
  - Rescoring hanya dilakukan pada selected_slots (warna yang dipilih user).
  - diversify_topk: maks 1 gambar per kelas batik dalam top-k.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
import pandas as pd

from app.services.extract_dominant_color import (
    FEATURE_SLOT,
    LARGEST_K,
    build_full_padded_vector,
    compute_image_specific_fill,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Diversifikasi Top-K
# ═══════════════════════════════════════════════════════════════════════════════

def diversify_topk(
    rescored: List[tuple],
    vec_id_to_class: Dict[int, str],
    k: int,
) -> List[tuple]:
    """
    Mengambil maksimal 1 vec_id per kelas batik (label) hingga terkumpul k item.
    Jika kelas unik habis sebelum mencapai k, sisa diisi dari kandidat terbaik
    berikutnya (boleh duplikat kelas).

    Parameters
    ----------
    rescored : list of (vec_id, distance) — sudah terurut distance ascending
    vec_id_to_class : dict {vec_id: label_batik}
    k : jumlah item yang dikembalikan

    Returns
    -------
    list of (vec_id, distance), panjang <= k
    """
    selected = []
    used_classes = set()
    leftover = []

    for vec_id, dist in rescored:
        cls = vec_id_to_class.get(vec_id)
        cls_key = f"__missing__:{vec_id}"
        if cls is not None:
            cls_key = str(cls).strip().lower()
        if cls_key not in used_classes:
            selected.append((vec_id, dist))
            used_classes.add(cls_key)
        else:
            leftover.append((vec_id, dist))

        if len(selected) == k:
            break

    if len(selected) < k:
        needed = k - len(selected)
        selected.extend(leftover[:needed])

    logger.info(
        "Diversified retrieval: selected=%d, candidates=%d",
        len(selected),
        len(rescored),
    )
    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ColorFaissRetriever
# ═══════════════════════════════════════════════════════════════════════════════

class ColorFaissRetriever:
    """
    FAISS-based retrieval untuk vektor warna dominan (Skenario 2 — L,a,b).

    Artifacts yang dibutuhkan (tidak lagi butuh slot_means):
        faiss_index_{scenario}.index
        faiss_meta_{scenario}.csv
        padded_vectors_{scenario}.npy
    """

    def __init__(self, data_dir: str, scenario: str = "s2_careful_max14") -> None:
        self.data_dir = Path(data_dir)
        self.scenario = scenario
        self._cache: Dict[int, dict] = {}

    def _bundle_paths(self) -> Dict[str, Path]:
        """Kembalikan path artifacts berdasarkan scenario."""
        base = self.scenario
        return {
            "index"  : self.data_dir / f"faiss_index_{base}.index",
            "meta"   : self.data_dir / f"faiss_meta_{base}.csv",
            "vectors": self.data_dir / f"padded_vectors_{base}.npy",
        }

    def _load_bundle(self) -> dict:
        """Load dan cache FAISS artifacts."""
        cache_key = 0   # single bundle (tidak per-cluster lagi)
        if cache_key in self._cache:
            return self._cache[cache_key]

        paths   = self._bundle_paths()
        missing = [name for name, path in paths.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing FAISS artifacts for scenario '{self.scenario}': "
                + ", ".join(missing)
            )

        index       = faiss.read_index(str(paths["index"]))
        meta        = pd.read_csv(paths["meta"])
        raw_vectors = np.load(paths["vectors"])

        meta_by_id = meta.set_index("vec_id")

        # Bangun lookup class (label batik) untuk diversifikasi
        vec_id_to_class = dict(zip(meta["vec_id"], meta["label"]))

        bundle = {
            "index"          : index,
            "meta"           : meta,
            "meta_by_id"     : meta_by_id,
            "raw_vectors"    : raw_vectors,
            "vec_id_to_class": vec_id_to_class,
        }
        self._cache[cache_key] = bundle
        logger.info(
            "Loaded FAISS bundle: scenario=%s, ntotal=%d", self.scenario, index.ntotal
        )
        return bundle

    def search(
        self,
        result: dict,
        selected_slots: Optional[List[int]],
        top_k: int,
        n_candidates_multiplier: int = 150,
    ) -> List[dict]:
        """
        Retrieval FAISS dengan image-specific fill + rescoring + diversifikasi.

        Parameters
        ----------
        result : dict
            Output dari extract_dominant_colors_careful() — berisi:
              'k_optimal'         : int
              'full_query_vector' : np.ndarray shape (LARGEST_K * FEATURE_SLOT,)
              'feature_vector'    : np.ndarray shape (k_opt * FEATURE_SLOT,)
              'features'          : list[dict] — [{'L','a','b','P'}, ...]
        selected_slots : list[int] or None
            Indeks 0-based slot yang dipilih untuk rescoring.
            Jika None/kosong → pakai semua slot aktif (0..k_opt-1).
        top_k : int
            Jumlah hasil yang dikembalikan.
        n_candidates_multiplier : int
            FAISS mengambil top_k * multiplier kandidat sebelum rescoring.

        Returns
        -------
        list[dict] — setiap item berisi:
            rank, vec_id, image_id, image_path, label,
            color_names_label, distance
        """
        bundle = self._load_bundle()

        index           = bundle["index"]
        raw_vectors     = bundle["raw_vectors"]
        meta_by_id      = bundle["meta_by_id"]
        vec_id_to_class = bundle["vec_id_to_class"]

        k_opt             = result.get("k_optimal", 0)
        full_query_vector = result.get("full_query_vector")
        feature_vector    = result.get("feature_vector")   # shape (k_opt * FEATURE_SLOT,)

        if k_opt == 0 or full_query_vector is None or feature_vector is None:
            return []

        full_query_vector = np.array(full_query_vector, dtype=np.float32).reshape(-1)
        feature_vector    = np.array(feature_vector, dtype=np.float32).reshape(-1)

        # --- Tentukan selected_slots ---
        if selected_slots is None or len(selected_slots) == 0:
            selected_slots = list(range(k_opt))
        else:
            # Pastikan tidak melebihi k_opt
            selected_slots = [s for s in selected_slots if s < k_opt]
            if not selected_slots:
                selected_slots = list(range(k_opt))

        # --- Hitung img_filler (image-specific fill) ---
        # Ambil dari slot kosong pertama yang sudah ada di full_query_vector,
        # atau hitung rata-rata jika semua slot terisi.
        features = result.get("features", [])
        active_full_4d = np.zeros((k_opt, 4), dtype=np.float32)
        for i, f in enumerate(features[:k_opt]):
            active_full_4d[i] = [f["L"], f["a"], f["b"], f["P"]]

        img_filler = compute_image_specific_fill(
            active_full_4d,
            feature_slot=FEATURE_SLOT,
            use_area_weight=False,  # Skenario 2
        )

        # --- Masking slot yang tidak dipilih dalam full_query_vector ---
        full_query = np.asarray(full_query_vector, dtype=np.float32).reshape(-1).copy()
        unselected = [i for i in range(LARGEST_K) if i not in selected_slots]
        for idx in unselected:
            start = idx * FEATURE_SLOT
            end   = start + FEATURE_SLOT
            full_query[start:end] = img_filler
        
        # --- FAISS Search ---
        # Coerce to strict base numpy ndarray, float32, C-contiguous.
        clean_list = full_query.tolist()
        query_vec = np.array([clean_list], dtype=np.float32)

        n_candidates = min(top_k * n_candidates_multiplier, index.ntotal)
        _, I         = index.search(query_vec, n_candidates)
        candidate_ids = [v for v in I[0] if v != -1]

        # --- Rescoring (hanya pada selected_slots, terhadap feature_vector asli) ---
        rescored = []
        for vec_id in candidate_ids:
            db_vec  = raw_vectors[vec_id]
            sq_dist = 0.0
            for idx in selected_slots:
                start = idx * FEATURE_SLOT
                end   = start + FEATURE_SLOT
                # Pastikan tidak melebihi panjang feature_vector
                if end <= len(feature_vector):
                    q_slice  = np.asarray(feature_vector[start:end], dtype=np.float32)
                    db_slice = np.asarray(db_vec[start:end],         dtype=np.float32)
                    diff     = q_slice - db_slice
                    sq_dist += float(np.dot(diff, diff))
            rescored.append((vec_id, float(np.sqrt(sq_dist))))

        rescored.sort(key=lambda x: x[1])

        # --- Diversifikasi: maks 1 per kelas batik ---
        top_k_diversified = diversify_topk(rescored, vec_id_to_class, top_k)

        # --- Bangun response ---
        results = []
        for vec_id, dist in top_k_diversified:
            if vec_id not in meta_by_id.index:
                continue
            row = meta_by_id.loc[vec_id]

            color_names = [
                str(row[col])
                for col in row.index
                if str(col).startswith("color_name_") and pd.notna(row[col])
            ]

            # Fallback: coba kolom color_name{i} (tanpa underscore)
            if not color_names:
                color_names = [
                    str(row[col])
                    for col in row.index
                    if str(col).startswith("color_name") and pd.notna(row[col])
                ]

            results.append(
                {
                    "rank"             : len(results) + 1,
                    "vec_id"           : int(vec_id),
                    "image_id"         : int(row.get("image_id", vec_id)),
                    "image_path"       : str(row.get("image_path", "")),
                    "label"            : str(row.get("label", "")),
                    "color_names_label": color_names,
                    "distance"         : round(float(dist), 6),
                }
            )

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Singleton accessor
# ═══════════════════════════════════════════════════════════════════════════════

_color_faiss_retriever: Optional[ColorFaissRetriever] = None


def get_color_faiss_retriever(
    data_dir: str,
    scenario: str = "s2_careful_max14",
) -> ColorFaissRetriever:
    """
    Kembalikan singleton ColorFaissRetriever.
    Jika belum diinisialisasi, buat instance baru dengan data_dir dan scenario.
    """
    global _color_faiss_retriever
    if _color_faiss_retriever is None:
        _color_faiss_retriever = ColorFaissRetriever(data_dir, scenario=scenario)
    return _color_faiss_retriever
