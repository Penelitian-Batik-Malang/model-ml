from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from app.config.settings import settings


def load_batik_database(npz_path: Path) -> dict:
    data = np.load(str(npz_path), allow_pickle=True)
    filenames = data["filename"]
    labels = data["label"]
    fitur_warna = data["fitur_warna"]

    batik_root_colab = "/content/drive/MyDrive/Data Penelitian Batik 2025/Data_Untuk_Warna_Dominan"

    filenames_local = []
    for filename in filenames:
        f_str = str(filename)
        rel_path = f_str.replace(batik_root_colab, "").strip("/")
        s3_key = rel_path.replace("\\", "/")
        full_url = f"{settings.S3_BATIK_FASHION_ROOT_URL}/{s3_key}"
        filenames_local.append(full_url)

    if filenames_local:
        print(f"[CBIR] Sample S3 URL: {filenames_local[0]}")

    return {
        "filenames": filenames_local,
        "labels": labels.tolist(),
        "fitur_warna": fitur_warna,
    }


def euclidean_hungarian(query: np.ndarray, db: np.ndarray) -> float:
    count = len(query)
    cost_matrix = np.zeros((count, count))
    for i in range(count):
        for j in range(count):
            cost_matrix[i][j] = float(np.linalg.norm(query[i] - db[j]))
    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    return float(cost_matrix[row_idx, col_idx].sum() / count)


def retrieve_batik(query_centroids: np.ndarray, db: dict, top_k_list: list = None) -> dict:
    if top_k_list is None:
        top_k_list = [5, 10, 15]

    jarak_list = []
    for i in range(len(db["filenames"])):
        jarak = euclidean_hungarian(query_centroids, db["fitur_warna"][i])
        jarak_list.append(
            {
                "filename": db["filenames"][i],
                "label": db["labels"][i],
                "jarak": jarak,
            }
        )
    jarak_list.sort(key=lambda x: x["jarak"])

    max_k = max(top_k_list)
    top_results = jarak_list[:max_k]
    for item in top_results:
        item["thumbnail_b64"] = ""

    hasil = {}
    for k in top_k_list:
        hasil[f"top_{k}"] = [{"rank": i + 1, **r} for i, r in enumerate(top_results[:k])]
    return hasil


def extract_query_centroids(fashion_rgb: np.ndarray, mask_union: np.ndarray, kluster: int = 3) -> np.ndarray:
    if mask_union.dtype != np.uint8:
        mask_union = (mask_union > 0).astype(np.uint8)
    pixels = fashion_rgb[mask_union == 1]
    if len(pixels) < kluster:
        return np.zeros((kluster, 3), dtype=np.float32)
    pixels = pixels.astype(np.float32) / 255.0
    pixels_lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    kmeans = KMeans(n_clusters=kluster, random_state=42, n_init=10)
    kmeans.fit(pixels_lab)
    return kmeans.cluster_centers_.astype(np.float32)
