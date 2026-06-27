import warnings
import cv2
import numpy as np
from sklearn.cluster import KMeans
from app.config.color_palette_multi_label import ColorPaletteMultiLabel

MIN_CLUSTERS = 2
MAX_CLUSTERS = 5

_LABELER = ColorPaletteMultiLabel()


def label_color(L: float, a: float, b: float) -> str:
    return _LABELER.get_label_color(L, a, b)


def _compute_sse_curve(pixels: np.ndarray,
                       unique_colors: np.ndarray,
                       density_sorted_indices: np.ndarray,
                       min_distance: float,
                       max_k: int) -> dict:
    sse_dict = {}
    for k in range(MIN_CLUSTERS, max_k + 1):
        # Careful seeding: pilih centroid dari sel grid densitas tertinggi
        # dengan jarak minimum antar centroid = min_distance
        # (Han et al., 2025 — careful seeding strategy)
        custom_centroids = []
        for idx in density_sorted_indices:
            candidate = unique_colors[idx]
            if len(custom_centroids) == 0:
                custom_centroids.append(candidate)
            else:
                dists = np.linalg.norm(
                    np.array(custom_centroids) - candidate, axis=1)
                if np.all(dists >= min_distance):
                    custom_centroids.append(candidate)
            if len(custom_centroids) == k:
                break

        # Fallback jika sel grid unik tidak cukup memenuhi syarat jarak
        while len(custom_centroids) < k:
            custom_centroids.append(
                pixels[np.random.randint(len(pixels))])

        init_arr = np.array(custom_centroids, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            km = KMeans(n_clusters=k, init=init_arr, n_init=1,
                        random_state=42, max_iter=300)
            km.fit(pixels)
        sse_dict[k] = float(km.inertia_)

    return sse_dict


def _elbow_k(sse_dict: dict) -> int:
    ks   = sorted(sse_dict.keys())
    sses = [sse_dict[k] for k in ks]

    # Normalisasi K dan SSE ke [0, 1]
    # (Han et al., 2025 — finding the elbow position;
    #  Antunes et al., 2025 — Kneedle / distance-to-line)
    k_norm = [(k - ks[0]) / (ks[-1] - ks[0]) if ks[-1] != ks[0] else 0.0
              for k in ks]
    s_range = max(sses) - min(sses)
    s_norm  = [(s - min(sses)) / s_range if s_range > 0 else 0.0
               for s in sses]

    # Jarak setiap titik ke garis diagonal (0,1)→(1,0)
    # Persamaan garis: x + y - 1 = 0  →  jarak = |x + y - 1| / sqrt(2)
    max_dist = -1
    best_k   = ks[0]
    for i in range(len(ks)):
        dist = abs(k_norm[i] + s_norm[i] - 1) / (2 ** 0.5)
        if dist > max_dist:
            max_dist = dist
            best_k   = ks[i]

    return best_k


def extract_dominant_colors_careful(
        image: np.ndarray,
        max_clusters: int = MAX_CLUSTERS,
) -> dict | None:
    """
    Ekstraksi warna dominan berbasis CHk-means (Han et al., 2025).

    CATATAN: `image` yang diterima adalah citra yang SUDAH dalam format LAB
    (output dari proportional_resize). Tidak ada konversi warna di sini.

    Alur (Han et al., 2025 — CHk-means):
      1. Reshape piksel — seluruh piksel citra digunakan langsung.
      2. Grid density quantization: kuantisasi piksel ke sel grid 16 unit
         dalam ruang LAB, hitung frekuensi tiap sel, urutkan densitas
         tertinggi ke terendah sebagai basis careful seeding.
      3. Careful seeding + evaluasi SSE untuk K = MIN_CLUSTERS…max_k:
         centroid awal dipilih dari sel grid densitas tertinggi dengan
         syarat jarak minimum antar seed ≥ 30 unit LAB.
      4. Elbow method (normalisasi kurva — Kneedle/distance-to-line)
         → K optimal (Han et al., 2025; Antunes et al., 2025).
      5. K-Means final dengan K optimal + careful seeding yang sama.
      6. Pelabelan CIEDE2000 per centroid.

    Parameter:
        image        : np.ndarray LAB (output proportional_resize), shape (H,W,3)
        max_clusters : batas atas K, default MAX_CLUSTERS=5

    Returns:
        {
          'k_optimal'    : int,
          'feature_vector': np.ndarray,  # shape (k_optimal × 4,) — [L,a,b,P]
          'features'     : list[dict],   # [{'L','a','b','P'}, ...]
          'labels'       : list[str],
        }
        None jika ekstraksi gagal (warna unik kurang dari MIN_CLUSTERS).
    """
    np.random.seed(42)

    # --- 1. Reshape piksel ---
    pixels = image.reshape((-1, 3)).astype(np.float32)

    # --- 2. Grid Density Quantization ---
    # Kuantisasi seluruh piksel ke sel grid berukuran grid_size unit dalam LAB.
    # Frekuensi tiap sel mencerminkan densitas warna pada area tersebut.
    # Sel dengan densitas tinggi menjadi kandidat seed; sel sparse secara
    # alami tidak terpilih karena kalah urutan densitas.
    # (Han et al., 2025 — grid partitioning dalam careful seeding)
    grid_size = 16.0
    min_dist  = 30.0
    quantized = np.floor(pixels / grid_size) * grid_size
    unique_colors, counts = np.unique(quantized, axis=0, return_counts=True)

    # Guard teknis: pastikan jumlah warna unik mencukupi untuk clustering
    if len(unique_colors) < MIN_CLUSTERS:
        return None

    density_sorted = np.argsort(counts)[::-1]

    # --- 3. Evaluasi SSE dengan Careful Seeding per K ---
    effective_max_k = min(max_clusters, len(unique_colors))
    if effective_max_k < MIN_CLUSTERS:
        effective_max_k = MIN_CLUSTERS

    sse_dict = _compute_sse_curve(
        pixels, unique_colors,
        density_sorted, min_dist, max_k=effective_max_k
    )

    # --- 4. Elbow Method → K optimal ---
    k_opt = _elbow_k(sse_dict)

    # --- 5. K-Means Final dengan Careful Seeding ---
    custom_centroids = []
    for idx in density_sorted:
        candidate = unique_colors[idx]
        if len(custom_centroids) == 0:
            custom_centroids.append(candidate)
        else:
            dists = np.linalg.norm(
                np.array(custom_centroids) - candidate, axis=1)
            if np.all(dists >= min_dist):
                custom_centroids.append(candidate)
        if len(custom_centroids) == k_opt:
            break

    while len(custom_centroids) < k_opt:
        custom_centroids.append(
            pixels[np.random.randint(len(pixels))])

    init_arr = np.array(custom_centroids, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = KMeans(n_clusters=k_opt, init=init_arr, n_init=1, random_state=42)
        km.fit(pixels)

    centroids    = km.cluster_centers_ / 255.0
    km_labels    = km.labels_
    pixel_counts = np.bincount(km_labels, minlength=k_opt)
    percentages  = pixel_counts / len(pixels)

    # --- 6. Pelabelan CIEDE2000 + Urutkan Persentase Desc ---
    features_sorted = sorted(
        [{'L': float(centroids[i][0]), 'a': float(centroids[i][1]),
          'b': float(centroids[i][2]), 'P': float(percentages[i])}
         for i in range(k_opt)],
        key=lambda x: x['P'], reverse=True
    )

    final_labels = [
        label_color(f['L'], f['a'], f['b'])
        for f in features_sorted
    ]

    feature_vector = np.array(
        [v for f in features_sorted
         for v in [f['L'], f['a'], f['b'], f['P']]],
        dtype=np.float32
    )

    return {
        'k_optimal'     : k_opt,
        'feature_vector': feature_vector,
        'features'      : features_sorted,
        'labels'        : final_labels,
    }


class ExtractDominantColor:
    """Wrapper class untuk backward-compatibility dengan kode yang menggunakan class API."""

    MIN_CLUSTERS = MIN_CLUSTERS
    MAX_CLUSTERS = MAX_CLUSTERS
    LABELER      = _LABELER

    @staticmethod
    def _compute_sse_curve(pixels, unique_colors, density_sorted_indices,
                           min_distance, max_k, inlier_pixels=None):
        # inlier_pixels diabaikan — signature dipertahankan untuk kompatibilitas
        return _compute_sse_curve(pixels, unique_colors,
                                  density_sorted_indices, min_distance, max_k)

    @staticmethod
    def _elbow_k(sse_dict):
        return _elbow_k(sse_dict)

    @staticmethod
    def extract_dominant_colors_careful(
            image: np.ndarray,
            max_clusters: int = MAX_CLUSTERS,
            **kwargs,
    ) -> dict | None:
        """
        Wrapper backward-compatible. Menerima BGR image dan mengonversinya ke LAB
        sebelum meneruskan ke fungsi utama.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel BGR image.")

        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        return extract_dominant_colors_careful(lab_image, max_clusters=max_clusters)