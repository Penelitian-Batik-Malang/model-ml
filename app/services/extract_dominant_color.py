import logging
import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage import segmentation, color, graph
import math

class ExtractDominantColor:
    @staticmethod
    def extract_dominant_colors_s1(image, num_clusters=5):
        """Return normalized feature vector for scenario 1."""
        features_sorted = ExtractDominantColor._extract_features(image, num_clusters)
        feature_vector = [val for f in features_sorted for val in [f['L'], f['a'], f['b'], f['P']]]
        return np.array(feature_vector, dtype=np.float32)

    @staticmethod
    def extract_palette_and_vector_s1(image, num_clusters=5):
        """Return palette entries and feature vector for scenario 1."""
        features_sorted = ExtractDominantColor._extract_features(image, num_clusters)
        feature_vector = [val for f in features_sorted for val in [f['L'], f['a'], f['b'], f['P']]]

        palette = []
        for idx, f in enumerate(features_sorted, start=1):
            lab_255 = [f['L'] * 255.0, f['a'] * 255.0, f['b'] * 255.0]
            palette.append(
                {
                    "no": idx,
                    "palette": ExtractDominantColor.lab_to_hex(lab_255),
                    "lab": [round(lab_255[0], 3), round(lab_255[1], 3), round(lab_255[2], 3)],
                    "percentage": round(float(f['P']), 6),
                }
            )

        return palette, np.array(feature_vector, dtype=np.float32)

    @staticmethod
    def lab_to_hex(lab_255):
        """Convert LAB (0-255 scale) to hex color."""
        lab = np.array([[lab_255]], dtype=np.uint8)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0][0]
        rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    @staticmethod
    def _extract_features(image, num_clusters=5):
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        pixels = lab_image.reshape((-1, 3))

        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        kmeans.fit(pixels)

        centroids = kmeans.cluster_centers_ / 255.0
        labels = kmeans.labels_

        counts = np.bincount(labels)
        percentages = counts / len(pixels)

        features_sorted = sorted(
            [
                {
                    'L': centroids[i][0],
                    'a': centroids[i][1],
                    'b': centroids[i][2],
                    'P': percentages[i],
                }
                for i in range(num_clusters)
            ],
            key=lambda x: x['P'],
            reverse=True,
        )
        return features_sorted