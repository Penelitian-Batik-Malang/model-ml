import logging
import warnings
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from app.config.color_palette_multi_label import ColorPaletteMultiLabel

class ExtractDominantColor:
    MIN_CLUSTERS = 2
    MAX_CLUSTERS = 5
    LABELER = ColorPaletteMultiLabel()

    @staticmethod
    def _compute_sse_curve(pixels: np.ndarray,
                           inlier_pixels: np.ndarray,
                           unique_colors: np.ndarray,
                           density_sorted_indices: np.ndarray,
                           min_distance: float,
                           max_k: int) -> dict:
        sse_dict = {}
        for k in range(ExtractDominantColor.MIN_CLUSTERS, max_k + 1):
            # Careful seeding: pilih centroid dari grid density paling tinggi
            # dengan jarak minimum antar centroid = min_distance
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

            # Fallback: isi centroid yang kurang dengan piksel inlier acak
            while len(custom_centroids) < k:
                custom_centroids.append(
                    inlier_pixels[np.random.randint(len(inlier_pixels))])

            init_arr = np.array(custom_centroids, dtype=np.float32)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                km = KMeans(n_clusters=k, init=init_arr, n_init=1,
                            random_state=42, max_iter=300)
                km.fit(pixels)
            sse_dict[k] = float(km.inertia_)

        return sse_dict

    @staticmethod
    def _elbow_k(sse_dict: dict) -> int:
        ks   = sorted(sse_dict.keys())
        sses = [sse_dict[k] for k in ks]

        if len(ks) <= 2:
            return ks[0]

        second_diffs = []
        for i in range(1, len(ks) - 1):
            d2 = sses[i + 1] - 2 * sses[i] + sses[i - 1]
            second_diffs.append((d2, ks[i]))

        # Elbow = K dengan second-difference paling negatif
        return min(second_diffs, key=lambda x: x[0])[1]

    @staticmethod
    def extract_dominant_colors_careful(
        image: np.ndarray,
        max_clusters: int = MAX_CLUSTERS,
        subsample_size: int = 5000,
    ) -> dict | None:
        """
        Ekstraksi warna dominan Careful dengan 5 klaster default.

        Parameter:
            image        : np.ndarray BGR image, shape (H,W,3)
            max_clusters : batas atas K, default MAX_CLUSTERS=5
            subsample_size: maks piksel untuk LOF + grid density

        Returns:
            {
              'k_optimal'    : int,
              'feature_vector': np.ndarray,  # shape (k_optimal × 4,) — [L,a,b,P]
              'features'     : list[dict],
              'labels'       : list[str],
            }
            None jika ekstraksi gagal.
        """
        np.random.seed(42)

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel BGR image.")

        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        pixels = lab_image.reshape((-1, 3))

        if len(pixels) > subsample_size:
            idx = np.random.choice(len(pixels), size=subsample_size, replace=False)
            sample_pixels = pixels[idx]
        else:
            sample_pixels = pixels.copy()

        jitter = np.random.normal(0, 1e-5, sample_pixels.shape)
        if len(sample_pixels) >= 20:
            lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
            lof_labels = lof.fit_predict(sample_pixels + jitter)
            inlier_pixels = sample_pixels[lof_labels == 1]
            if len(inlier_pixels) < ExtractDominantColor.MIN_CLUSTERS:
                inlier_pixels = sample_pixels
        else:
            inlier_pixels = sample_pixels

        if len(inlier_pixels) < ExtractDominantColor.MIN_CLUSTERS:
            return None

        grid_size = 16.0
        min_dist = 30.0
        quantized = np.floor(inlier_pixels / grid_size) * grid_size
        unique_colors, counts = np.unique(quantized, axis=0, return_counts=True)
        density_sorted = np.argsort(counts)[::-1]

        if len(unique_colors) == 0:
            return None
        
        effective_max_k = min(max_clusters, len(unique_colors))
        if effective_max_k < ExtractDominantColor.MIN_CLUSTERS:
            effective_max_k = ExtractDominantColor.MIN_CLUSTERS

        sse_dict = ExtractDominantColor._compute_sse_curve(
            pixels, inlier_pixels, unique_colors,
            density_sorted, min_dist, max_k=effective_max_k
        )
        k_opt = ExtractDominantColor._elbow_k(sse_dict)


        custom_centroids = []
        for idx in density_sorted:
            candidate = unique_colors[idx]
            if len(custom_centroids) == 0:
                custom_centroids.append(candidate)
            else:
                dists = np.linalg.norm(
                    np.array(custom_centroids) - candidate,
                    axis=1,
                )
                if np.all(dists >= min_dist):
                    custom_centroids.append(candidate)
            if len(custom_centroids) == k_opt:
                break

        while len(custom_centroids) < k_opt:
            custom_centroids.append(
                inlier_pixels[np.random.randint(len(inlier_pixels))]
            )

        init_arr = np.array(custom_centroids, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            km = KMeans(
                n_clusters=k_opt,
                init=init_arr,
                n_init=1,
                random_state=42,
            )
            km.fit(pixels)

        centroids = km.cluster_centers_ / 255.0
        pixel_counts = np.bincount(km.labels_, minlength=k_opt)
        percentages = pixel_counts / len(pixels)

        features_sorted = sorted(
            [
                {
                    'L': float(centroids[i][0]),
                    'a': float(centroids[i][1]),
                    'b': float(centroids[i][2]),
                    'P': float(percentages[i]),
                }
                for i in range(k_opt)
            ],
            key=lambda x: x['P'],
            reverse=True,
        )

        raw_labels = [
            ExtractDominantColor.LABELER.get_label_color(
                f['L'], f['a'], f['b']
            )
            for f in features_sorted
        ]

        feature_vector = np.array(
            [v for f in features_sorted for v in [f['L'], f['a'], f['b'], f['P']]],
            dtype=np.float32,
        )

        return {
            'k_optimal': k_opt,
            'feature_vector': feature_vector,
            'features': features_sorted,
            'labels': raw_labels,
        }