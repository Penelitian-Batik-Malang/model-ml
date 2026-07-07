"""
extract_dominant_color.py

Pipeline ekstraksi warna dominan mengikuti fungsi visualize_combo_retrieval
dari notebook color_palette_multi_label_kmeans_faiss_elbow (2).py.

Alur (Skenario 2 — L, a, b only, image-specific fill):
  1. proportional_resize → cv2 LAB (0-255)
  2. cv2lab_to_stdlab     → standard LAB
  3. find_optimal_k_elbow → k_optimal berbasis CIEDE2000 + KneeLocator
  4. nieves_quantization  → grid 20.0 + threshold primary/secondary
  5. careful_seeding_quantization → urutkan densitas
  6. pad_seed_centroids_to_k → pastikan cukup seed
  7. KMeans final → features (L,a,b,P) + labels
  8. build_query_vector_from_result → vektor query full padded (image-specific fill)

Konstanta:
  LARGEST_K    = 14  (max slot warna dalam FAISS index)
  FEATURE_SLOT = 3   (L, a, b — Skenario 2)
"""

import logging
import warnings
import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.color import deltaE_ciede2000
from typing import Dict, Optional, Tuple

from app.config.color_palette_multi_label import ColorPaletteMultiLabel

try:
    from kneed import KneeLocator
    _KNEED_AVAILABLE = True
except ImportError:
    _KNEED_AVAILABLE = False

# ── Konstanta Global ─────────────────────────────────────────────────────────
LARGEST_K    = 14     # jumlah slot maksimum di FAISS index (LARGEST_K notebook)
FEATURE_SLOT = 3      # L, a, b — Skenario 2 (bukan 4 karena P tidak disimpan)

# Parameter nieves_quantization
GRID_SIZE             = 20.0
THR_PERCENT           = 0.03
THR_SECONDARY_FACTOR  = 8.0
L_THRESHOLD           = 80.0
CHROMA_PERCENTILE     = 50.0

# Parameter find_optimal_k_elbow
ELBOW_K_MIN           = 1
ELBOW_K_MAX           = 50
ELBOW_N_INIT          = 5
ELBOW_MAX_PIXELS      = 5000
_LABELER = ColorPaletteMultiLabel()

# ── Backward-compat alias ────────────────────────────────────────────────────
MIN_CLUSTERS = 2   # dipertahankan untuk kompatibilitas lama
MAX_CLUSTERS = LARGEST_K


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Helper: Konversi ruang warna
# ═══════════════════════════════════════════════════════════════════════════════

def cv2lab_to_stdlab(pixels_cv2lab: np.ndarray) -> np.ndarray:
    """
    Konversi piksel dari cv2 LAB (range 0-255) ke standard LAB (L*: 0-100,
    a*: -128..127, b*: -128..127).

    cv2 menyimpan LAB dengan encoding:
        L_cv2  = L* / 100 * 255
        a_cv2  = (a* + 128)
        b_cv2  = (b* + 128)
    """
    normalized = pixels_cv2lab.astype(np.float64) / 255.0
    L = normalized[:, 0] * 100.0
    a = normalized[:, 1] * 255.0 - 128.0
    b = normalized[:, 2] * 255.0 - 128.0
    return np.stack([L, a, b], axis=1)


def label_color(L: float, a: float, b: float) -> str:
    """Label warna via ColorPaletteMultiLabel (CIEDE2000)."""
    return _LABELER.get_label_color(L, a, b)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Elbow detection berbasis CIEDE2000
# ═══════════════════════════════════════════════════════════════════════════════

def find_optimal_k_elbow(
    pixels_std_lab: np.ndarray,
    k_min: int = ELBOW_K_MIN,
    k_max: int = ELBOW_K_MAX,
    step: int = 1,
    n_init: int = ELBOW_N_INIT,
    max_pixels_for_fit: int = ELBOW_MAX_PIXELS,
    random_state: int = 42,
) -> int:
    """
    Menentukan K optimal menggunakan rata-rata CIEDE2000 intra-cluster
    sebagai metrik SSE, lalu KneeLocator (convex, decreasing) untuk
    menemukan titik siku.

    Fallback jika kneed tidak tersedia: ambil nilai tengah dari range K.

    Parameters
    ----------
    pixels_std_lab : np.ndarray, shape (N, 3)
        Piksel dalam standard LAB (output cv2lab_to_stdlab).
    k_min, k_max, step : parameter range K yang dievaluasi
    n_init : jumlah reinisiasi KMeans
    max_pixels_for_fit : subsample jika terlalu banyak piksel
    random_state : seed reproducibility

    Returns
    -------
    int : k_optimal
    """
    pixels_for_fit = pixels_std_lab
    if len(pixels_std_lab) > max_pixels_for_fit:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(pixels_std_lab), size=max_pixels_for_fit, replace=False)
        pixels_for_fit = pixels_std_lab[idx]

    ks = list(range(k_min, k_max + 1, step))
    avg_deltaE_per_k = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in ks:
            km = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=n_init,
                random_state=random_state,
            )
            km.fit(pixels_for_fit)
            labels_km  = km.labels_
            centers    = km.cluster_centers_

            intra_cluster_de = []
            for i in range(k):
                cluster_points = pixels_for_fit[labels_km == i]
                if len(cluster_points) == 0:
                    continue
                center_rep = np.repeat(centers[i][None, :], len(cluster_points), axis=0)
                de = deltaE_ciede2000(cluster_points, center_rep)
                logging.debug(f"K={k}, cluster {i}: mean deltaE={np.mean(de):.4f}, count={len(de)}")
                intra_cluster_de.append(np.mean(de))

            avg_deltaE_per_k.append(np.mean(intra_cluster_de) if intra_cluster_de else np.nan)

    if _KNEED_AVAILABLE:
        kl = KneeLocator(ks, avg_deltaE_per_k, curve="convex", direction="decreasing")
        k_optimal = kl.elbow
        if k_optimal is None:
            k_optimal = ks[len(ks) // 2]
    else:
        # Fallback sederhana: titik penurunan terbesar (analog elbow SSE)
        k_optimal = ks[0]
        max_drop  = 0.0
        for i in range(1, len(avg_deltaE_per_k)):
            drop = avg_deltaE_per_k[i - 1] - avg_deltaE_per_k[i]
            if drop > max_drop:
                max_drop  = drop
                k_optimal = ks[i]

    return int(k_optimal)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Nieves Quantization
# ═══════════════════════════════════════════════════════════════════════════════

def nieves_quantization(
    lab_image: np.ndarray,
    grid_size: float = GRID_SIZE,
    thr_percent: float = THR_PERCENT,
    thr_secondary_factor: float = THR_SECONDARY_FACTOR,
    L_threshold: float = L_THRESHOLD,
    chroma_percentile: float = CHROMA_PERCENTILE,
) -> Dict:
    """
    Kuantisasi grid LAB dengan threshold relevance dua level (primary/secondary).

    Primary threshold  : sel dengan count >= Thr (thr_percent * N_total)
    Secondary threshold: sel dengan count >= Thr/factor DAN (L tinggi ATAU chroma tinggi)

    Returns
    -------
    dict dengan kunci:
        pixels_lab          : (N, 3) — seluruh piksel dalam cv2 LAB
        relevant_colors_lab : (n_relevant, 3) — centroid sel relevan
        counts              : (n_relevant,) — frekuensi sel relevan
        n_relevant          : int
        n_total_filled      : int — total sel grid yang terisi
        Thr, Thr_secondary  : threshold
        chroma_50_pct       : float
        all_colors_lab      : (n_filled, 3) — centroid SEMUA sel (relevan + tidak)
        all_counts          : (n_filled,) — frekuensi semua sel
        is_relevant_mask    : (n_filled,) bool
    """
    pixels_lab = lab_image.reshape(-1, 3)
    N_total    = len(pixels_lab)

    a      = pixels_lab[:, 1]
    b      = pixels_lab[:, 2]
    chroma = np.sqrt(a ** 2 + b ** 2)
    chroma_50_percentile = np.percentile(chroma, chroma_percentile)

    Thr           = thr_percent * N_total
    Thr_secondary = Thr / thr_secondary_factor

    quantized_coords = np.floor(pixels_lab / grid_size) * grid_size
    unique_cells, inverse_idx, cell_counts = np.unique(
        quantized_coords, axis=0, return_inverse=True, return_counts=True
    )
    n_filled_cells = len(unique_cells)

    sums = np.zeros((n_filled_cells, 3), dtype=np.float64)
    np.add.at(sums, inverse_idx, pixels_lab)
    cube_colors_lab = (sums / cell_counts[:, None]).astype(np.float32)

    cube_L      = cube_colors_lab[:, 0]
    cube_chroma = np.sqrt(cube_colors_lab[:, 1] ** 2 + cube_colors_lab[:, 2] ** 2)

    is_relevant = cell_counts >= Thr
    secondary_mask = (cell_counts >= Thr_secondary) & (cell_counts < Thr)
    is_relevant |= secondary_mask & (
        (cube_L > L_threshold) | (cube_chroma > chroma_50_percentile)
    )

    relevant_indices    = np.where(is_relevant)[0]
    n_relevant          = len(relevant_indices)
    relevant_colors_lab = cube_colors_lab[relevant_indices]
    relevant_counts     = cell_counts[relevant_indices]

    return {
        "pixels_lab"         : pixels_lab,
        "relevant_colors_lab": relevant_colors_lab,
        "counts"             : relevant_counts,
        "n_relevant"         : n_relevant,
        "n_total_filled"     : n_filled_cells,
        "Thr"                : Thr,
        "Thr_secondary"      : Thr_secondary,
        "chroma_50_pct"      : chroma_50_percentile,
        "all_colors_lab"     : cube_colors_lab,
        "all_counts"         : cell_counts,
        "is_relevant_mask"   : is_relevant,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Careful Seeding
# ═══════════════════════════════════════════════════════════════════════════════

def careful_seeding_quantization(
    relevant_colors_lab: np.ndarray,
    counts: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    Urutkan sel relevan berdasarkan densitas tertinggi → dipakai sebagai
    seed awal KMeans (careful seeding).

    Returns
    -------
    (seed_centroids, k_from_seeds)
        seed_centroids : (n_relevant, 3) float32 — centroid terurut
        k_from_seeds   : int — jumlah seed
    """
    order          = np.argsort(counts)[::-1]
    seed_centroids = relevant_colors_lab[order].astype(np.float32)
    return seed_centroids, len(seed_centroids)


def pad_seed_centroids_to_k(
    seed_centroids: np.ndarray,
    quant_result: Dict,
    k_target: int,
    random_state: int = 42,
    max_candidates_for_sampling: int = 2500,
) -> Tuple[np.ndarray, int]:
    """
    Lengkapi seed_centroids hingga k_target dengan cara:
      1. Ambil sel non-relevan (densitas tinggi terlebih dahulu)
      2. Fallback: farthest-point sampling dari piksel gambar

    Returns
    -------
    (seed_centroids, k_actual)
    """
    current_k      = len(seed_centroids)
    seed_centroids = seed_centroids.astype(np.float32).copy()

    if current_k >= k_target:
        return seed_centroids[:k_target], k_target

    n_missing = k_target - current_k

    all_colors = quant_result["all_colors_lab"]
    all_counts = quant_result["all_counts"]
    is_relevant = quant_result["is_relevant_mask"]

    fallback_colors = all_colors[~is_relevant]
    fallback_counts = all_counts[~is_relevant]

    if len(fallback_colors) > 0 and n_missing > 0:
        order  = np.argsort(fallback_counts)[::-1]
        take   = min(n_missing, len(fallback_colors))
        extra  = fallback_colors[order][:take].astype(np.float32)
        seed_centroids = np.vstack([seed_centroids, extra])
        n_missing -= take

    if n_missing > 0:
        pixels_lab = quant_result["pixels_lab"].astype(np.float32)
        rng = np.random.default_rng(random_state)

        if len(pixels_lab) > max_candidates_for_sampling:
            idx        = rng.choice(len(pixels_lab), size=max_candidates_for_sampling, replace=False)
            candidates = pixels_lab[idx]
        else:
            candidates = pixels_lab

        for _ in range(n_missing):
            dists = np.min(
                np.linalg.norm(
                    candidates[:, None, :] - seed_centroids[None, :, :], axis=2
                ),
                axis=1,
            )
            farthest_idx   = np.argmax(dists)
            new_centroid   = candidates[farthest_idx]
            seed_centroids = np.vstack([seed_centroids, new_centroid[None, :]])

    k_actual = len(seed_centroids)
    return seed_centroids.astype(np.float32), k_actual


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Image-specific fill (pengganti slot_means global)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_image_specific_fill(
    active_full_4d: np.ndarray,
    feature_slot: int = FEATURE_SLOT,
    use_area_weight: bool = False,
    weight_col: int = 3,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Menghitung vektor pengisi untuk slot kosong berdasarkan rata-rata
    (berbobot atau tidak) dari slot aktif MILIK CITRA ITU SENDIRI.

    Mengadaptasi Wang et al. (2023) Definisi 8 (Atangana-Baleanu mean).

    Parameters
    ----------
    active_full_4d : np.ndarray, shape (k_opt, 4)
        Nilai [L, a, b, P] dari semua slot aktif.
    feature_slot : int
        3 untuk Skenario 2 (L,a,b), 4 untuk Skenario 1 (L,a,b,P).
    use_area_weight : bool
        True → bobot = P_i (Skenario 1); False → rata-rata tak berbobot (Skenario 2).
    weight_col : int
        Indeks kolom P pada active_full_4d (default 3).
    eps : float
        Safeguard pembagian nol.

    Returns
    -------
    np.ndarray, shape (feature_slot,)
    """
    if active_full_4d.shape[0] == 0:
        return np.zeros(feature_slot, dtype=np.float32)

    if use_area_weight:
        weights = active_full_4d[:, weight_col].astype(np.float64)
        if weights.sum() <= eps:
            weights = np.ones(active_full_4d.shape[0], dtype=np.float64)
        v_full = np.average(active_full_4d, axis=0, weights=weights)
    else:
        v_full = active_full_4d.mean(axis=0)

    return v_full[:feature_slot].astype(np.float32)


def build_full_padded_vector(
    active_full_4d: np.ndarray,
    k_opt: int,
    max_clusters: int = LARGEST_K,
    feature_slot: int = FEATURE_SLOT,
    use_area_weight: bool = False,
) -> np.ndarray:
    """
    Menyusun vektor fitur penuh (flattened, panjang max_clusters*feature_slot)
    untuk satu citra:
      - Slot aktif   (0..k_opt-1)  : nilai asli [L,a,b] (atau [L,a,b,P])
      - Slot kosong  (k_opt..max_clusters-1) : image-specific fill

    Parameters
    ----------
    active_full_4d : np.ndarray, shape (k_opt, 4)
        [L, a, b, P] dari slot aktif.
    k_opt : int       — jumlah warna dominan yang ditemukan
    max_clusters : int — K_FINAL (LARGEST_K)
    feature_slot : int — 3 atau 4
    use_area_weight : bool

    Returns
    -------
    np.ndarray, shape (max_clusters * feature_slot,)
    """
    k_opt  = min(k_opt, max_clusters)
    active = active_full_4d[:k_opt]

    fill_vector = compute_image_specific_fill(
        active, feature_slot=feature_slot, use_area_weight=use_area_weight
    )

    full = np.zeros((max_clusters, feature_slot), dtype=np.float32)
    for c in range(max_clusters):
        if c < k_opt:
            full[c] = active[c, :feature_slot]
        else:
            full[c] = fill_vector

    return full.reshape(-1)


def build_query_vector_from_result(
    result: dict,
    k_opt: int,
    max_clusters: int = LARGEST_K,
    feature_slot: int = FEATURE_SLOT,
    use_area_weight: bool = False,
) -> np.ndarray:
    """
    Menyusun vektor query penuh dari dict hasil ekstraksi.

    result berisi:
        'features': list[{'L', 'a', 'b', 'P'}]  — sudah terurut desc by P
        'k_optimal': int

    Returns
    -------
    np.ndarray, shape (max_clusters * feature_slot,)
    """
    features = result.get("features", [])
    active_full_4d = np.zeros((k_opt, 4), dtype=np.float32)
    for i, f in enumerate(features[:k_opt]):
        active_full_4d[i] = [f["L"], f["a"], f["b"], f["P"]]

    return build_full_padded_vector(
        active_full_4d=active_full_4d,
        k_opt=k_opt,
        max_clusters=max_clusters,
        feature_slot=feature_slot,
        use_area_weight=use_area_weight,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Fungsi Utama Ekstraksi
# ═══════════════════════════════════════════════════════════════════════════════

def extract_dominant_colors_careful(
    image: np.ndarray,
    max_clusters: int = LARGEST_K,
    k_target: Optional[int] = None,
    find_elbow: bool = True,
) -> Optional[dict]:
    """
    Ekstraksi warna dominan mengikuti pipeline visualize_combo_retrieval dari notebook.

    CATATAN: `image` yang diterima harus dalam format BGR (uint8 atau float).
    Konversi ke LAB dilakukan di dalam fungsi ini.

    Alur:
      1. BGR → cv2 LAB
      2. cv2lab_to_stdlab → standard LAB
      3. find_optimal_k_elbow (CIEDE2000) → k_optimal
      4. nieves_quantization → sel relevan
      5. careful_seeding_quantization + pad_seed_centroids_to_k
      6. KMeans final → centroids + percentages
      7. Label CIEDE2000 per centroid
      8. build_query_vector_from_result → vektor query full padded

    Parameters
    ----------
    image : np.ndarray
        BGR image (uint8), shape (H, W, 3). Harus sudah di-resize.
    max_clusters : int
        Maksimum slot warna (default LARGEST_K=14). Dipakai sebagai k_target
        jika k_target tidak diberikan.
    k_target : int or None
        Jika diberikan, batasi k_optimal maksimal k_target.
        Jika None, gunakan max_clusters.
    find_elbow : bool
        Jika True, jalankan CIEDE2000 elbow. Jika False, langsung pakai k_target.

    Returns
    -------
    dict dengan kunci:
        'k_optimal'           : int
        'feature_vector'      : np.ndarray, shape (k_opt * FEATURE_SLOT,) — [L,a,b] per warna
        'feature_vector_4d'   : np.ndarray, shape (k_opt * 4,)            — [L,a,b,P] per warna
        'full_query_vector'   : np.ndarray, shape (LARGEST_K * FEATURE_SLOT,) — padded
        'features'            : list[dict] — [{'L','a','b','P'}, ...]
        'labels'              : list[str]
    None jika ekstraksi gagal.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel BGR image.")

    # --- 1. BGR → cv2 LAB (0-255) ---
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    lab_image_cv2 = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # --- 2. Standard LAB untuk elbow ---
    pixels_cv2lab = lab_image_cv2.reshape(-1, 3)
    pixels_std_lab = cv2lab_to_stdlab(pixels_cv2lab)

    # --- 3. Tentukan k_target ---
    _k_target = k_target if k_target is not None else max_clusters

    # --- 4. Elbow → k_optimal ---
    if find_elbow:
        k_elbow   = find_optimal_k_elbow(
            pixels_std_lab,
            k_min=ELBOW_K_MIN,
            k_max=min(ELBOW_K_MAX, _k_target),
            max_pixels_for_fit=ELBOW_MAX_PIXELS,
        )
        k_optimal = min(k_elbow, _k_target)
    else:
        if _k_target is None:
            raise ValueError("k_target tidak boleh None jika find_elbow adalah False.")
        k_optimal = _k_target

    # --- 5. Nieves Quantization ---
    q = nieves_quantization(lab_image_cv2.astype(np.float32))

    # Guard: jika tidak ada warna relevan
    if q["n_relevant"] == 0:
        return None

    # --- 6. Careful Seeding ---
    seed_centroids, _k = careful_seeding_quantization(
        q["relevant_colors_lab"], q["counts"]
    )
    seed_centroids, k_actual = pad_seed_centroids_to_k(
        seed_centroids, q, k_target=k_optimal
    )

    if k_actual < k_optimal:
        k_optimal = k_actual

    if k_optimal == 0:
        return None

    # --- 7. KMeans Final ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = KMeans(
            n_clusters=k_optimal,
            init=seed_centroids,
            n_init=1,
            random_state=42,
        )
        km.fit(q["pixels_lab"])

    centroids    = km.cluster_centers_ / 255.0   # normalisasi agar sebanding dengan fitur
    km_labels    = km.labels_
    pixel_counts = np.bincount(km_labels, minlength=k_optimal)
    percentages  = pixel_counts / len(q["pixels_lab"])

    # --- 8. Pelabelan + Urutkan Persentase Desc ---
    features_sorted = sorted(
        [
            {
                "L": float(centroids[i][0]),
                "a": float(centroids[i][1]),
                "b": float(centroids[i][2]),
                "P": float(percentages[i]),
            }
            for i in range(k_optimal)
        ],
        key=lambda x: x["P"],
        reverse=True,
    )

    final_labels = [
        label_color(f["L"], f["a"], f["b"])
        for f in features_sorted
    ]

    # Feature vector hanya L, a, b (Skenario 2)
    feature_vector = np.array(
        [v for f in features_sorted for v in [f["L"], f["a"], f["b"]]],
        dtype=np.float32,
    )

    # Feature vector 4D (L, a, b, P) — dibutuhkan oleh build_query_vector_from_result
    feature_vector_4d = np.array(
        [v for f in features_sorted for v in [f["L"], f["a"], f["b"], f["P"]]],
        dtype=np.float32,
    )

    result = {
        "k_optimal"       : k_optimal,
        "feature_vector"  : feature_vector,
        "feature_vector_4d": feature_vector_4d,
        "features"        : features_sorted,
        "labels"          : final_labels,
    }

    # --- 9. Build full padded query vector (image-specific fill) ---
    full_query_vector = build_query_vector_from_result(
        result=result,
        k_opt=k_optimal,
        max_clusters=LARGEST_K,
        feature_slot=FEATURE_SLOT,
        use_area_weight=False,  # Skenario 2: tidak berbobot P
    )
    result["full_query_vector"] = full_query_vector

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Wrapper Class (backward-compatible)
# ═══════════════════════════════════════════════════════════════════════════════

class ExtractDominantColor:
    """
    Wrapper class untuk backward-compatibility dengan kode yang menggunakan
    class API. Semua method mendelegasikan ke fungsi modul-level di atas.
    """

    MIN_CLUSTERS = MIN_CLUSTERS
    MAX_CLUSTERS = MAX_CLUSTERS   # = LARGEST_K = 14
    LARGEST_K    = LARGEST_K
    FEATURE_SLOT = FEATURE_SLOT
    LABELER      = _LABELER

    @staticmethod
    def extract_dominant_colors_careful(
        image: np.ndarray,
        max_clusters: int = LARGEST_K,
        k_target: Optional[int] = None,
        find_elbow: bool = True,
        **kwargs,
    ) -> Optional[dict]:
        """
        Wrapper backward-compatible.

        Menerima BGR image (uint8), mendelegasikan ke fungsi utama.
        """
        return extract_dominant_colors_careful(
            image,
            max_clusters=max_clusters,
            k_target=k_target,
            find_elbow=find_elbow,
        )

    # ── Expose helper functions via class ────────────────────────────────────
    @staticmethod
    def build_query_vector_from_result(
        result: dict,
        k_opt: int,
        max_clusters: int = LARGEST_K,
        feature_slot: int = FEATURE_SLOT,
        use_area_weight: bool = False,
    ) -> np.ndarray:
        return build_query_vector_from_result(
            result, k_opt, max_clusters, feature_slot, use_area_weight
        )

    @staticmethod
    def compute_image_specific_fill(
        active_full_4d: np.ndarray,
        feature_slot: int = FEATURE_SLOT,
        use_area_weight: bool = False,
    ) -> np.ndarray:
        return compute_image_specific_fill(active_full_4d, feature_slot, use_area_weight)

    @staticmethod
    def build_full_padded_vector(
        active_full_4d: np.ndarray,
        k_opt: int,
        max_clusters: int = LARGEST_K,
        feature_slot: int = FEATURE_SLOT,
        use_area_weight: bool = False,
    ) -> np.ndarray:
        return build_full_padded_vector(
            active_full_4d, k_opt, max_clusters, feature_slot, use_area_weight
        )