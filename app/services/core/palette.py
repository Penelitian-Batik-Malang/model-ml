"""
Palette extraction algorithms for dominant color detection
Port dari testing notebook Febrio SegmentRecolor
"""

import numpy as np
import time
from collections import Counter
import skimage.color as color_converter
from skimage import io
from skimage.transform import resize as sk_resize
from sklearn.cluster import KMeans
from app.services.core.color_utils import hex_to_rgb, rgb_to_hex


def extract_dominant_colors_kmeans(image: np.ndarray, n_candidates=30, n_final=6, resize_max=512) -> list:
    """
    Extract dominant colors using K-means clustering.
    
    Args:
        image: Numpy RGB image array
        n_candidates: Number of initial clusters
        n_final: Final number of palette colors (will be <= n_final)
        resize_max: Max dimension for resizing (for speed)
    
    Returns:
        List of hex color strings
    """
    t0 = time.time()
    
    # Ensure RGB format
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    
    # Normalize
    if image.max() > 1.0:
        img_float = image.astype(np.float64) / 255.0
    else:
        img_float = image.astype(np.float64)
    
    # Resize if needed
    h, w = img_float.shape[:2]
    if max(h, w) > resize_max:
        scale = resize_max / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_float = sk_resize(img_float, (new_h, new_w), anti_aliasing=True)
    
    # Convert to LAB color space
    img_lab = color_converter.rgb2lab(img_float)
    pixels = img_lab.reshape(-1, 3).astype(np.float32)
    
    # K-means clustering
    n_clusters_actual = min(n_candidates, len(pixels))
    kmeans = KMeans(n_clusters=n_clusters_actual, n_init=5, random_state=42)
    kmeans.fit(pixels)
    
    # Get cluster centers (already in LAB)
    lab_centers = kmeans.cluster_centers_
    
    # If more candidates than requested, select best ones
    if len(lab_centers) > n_final:
        # Use k-means again to reduce to n_final
        kmeans_final = KMeans(n_clusters=n_final, n_init=5, random_state=42)
        kmeans_final.fit(lab_centers)
        final_lab = kmeans_final.cluster_centers_
    else:
        final_lab = lab_centers[:n_final]
    
    # Convert LAB to RGB
    final_rgb = np.clip(
        color_converter.lab2rgb(final_lab.reshape(1, -1, 3)).reshape(-1, 3),
        0, 1
    )
    
    # Convert to hex
    hex_palette = [rgb_to_hex(c) for c in final_rgb]
    
    return hex_palette


def extract_palette_histogram(image: np.ndarray, n_final=6, bins_per_channel=16, resize_max=512) -> list:
    """
    Extract dominant colors using histogram binning in LAB color space.
    
    Args:
        image: Numpy RGB image array
        n_final: Number of palette colors
        bins_per_channel: Number of bins per LAB channel
        resize_max: Max dimension for resizing
    
    Returns:
        List of hex color strings
    """
    t0 = time.time()
    
    # Ensure RGB format
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    
    # Normalize
    if image.max() > 1.0:
        img_float = image.astype(np.float64) / 255.0
    else:
        img_float = image.astype(np.float64)
    
    # Resize if needed
    h, w = img_float.shape[:2]
    if max(h, w) > resize_max:
        scale = resize_max / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_float = sk_resize(img_float, (new_h, new_w), anti_aliasing=True)
    
    # Convert to LAB
    img_lab = color_converter.rgb2lab(img_float)
    
    # Normalize LAB to 0-1 range for binning
    norm_lab = img_lab.copy()
    norm_lab[:, :, 0] = norm_lab[:, :, 0] / 100.0
    norm_lab[:, :, 1] = (norm_lab[:, :, 1] + 128.0) / 255.0
    norm_lab[:, :, 2] = (norm_lab[:, :, 2] + 128.0) / 255.0
    
    # Quantize to bins
    quantized_colors = (norm_lab * bins_per_channel).astype(int)
    color_tuples = [tuple(c) for c in quantized_colors.reshape(-1, 3)]
    
    # Count occurrences
    color_counts = Counter(color_tuples)
    
    # Get top N colors
    dominant_quantized = color_counts.most_common(n_final)
    
    # Convert back to LAB
    final_lab_colors = []
    for q_color, _ in dominant_quantized:
        l = (q_color[0] + 0.5) / bins_per_channel * 100.0
        a = ((q_color[1] + 0.5) / bins_per_channel * 255.0) - 128.0
        b = ((q_color[2] + 0.5) / bins_per_channel * 255.0) - 128.0
        final_lab_colors.append([l, a, b])
    
    final_lab = np.array(final_lab_colors).reshape(1, -1, 3)
    final_rgb = np.clip(
        color_converter.lab2rgb(final_lab).reshape(-1, 3),
        0, 1
    )
    
    # Convert to hex
    hex_palette = [rgb_to_hex(c) for c in final_rgb]
    
    return hex_palette


def extract_palette_median_cut(image: np.ndarray, n_final=6, resize_max=512) -> list:
    """
    Extract dominant colors using Median Cut algorithm.
    Recursively divides color space to find dominant regions.
    
    Args:
        image: Numpy RGB image array
        n_final: Number of palette colors
        resize_max: Max dimension for resizing
    
    Returns:
        List of hex color strings
    """
    t0 = time.time()
    
    # Ensure RGB format
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    
    # Normalize
    if image.max() > 1.0:
        img_float = image.astype(np.float64) / 255.0
    else:
        img_float = image.astype(np.float64)
    
    # Resize if needed
    h, w = img_float.shape[:2]
    if max(h, w) > resize_max:
        scale = resize_max / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_float = sk_resize(img_float, (new_h, new_w), anti_aliasing=True)
    
    pixels_rgb = img_float.reshape(-1, 3)
    
    def median_cut_recursive(pixels, num_colors):
        """Recursively apply median cut"""
        if num_colors == 1:
            return [np.mean(pixels, axis=0)]
        
        # Find channel with largest range
        ranges = np.max(pixels, axis=0) - np.min(pixels, axis=0)
        channel = np.argmax(ranges)
        
        # Sort and split at median
        sorted_indices = pixels[:, channel].argsort()
        pixels_sorted = pixels[sorted_indices]
        midpoint = len(pixels_sorted) // 2
        
        # Recursive calls
        colors_left = median_cut_recursive(pixels_sorted[:midpoint], num_colors // 2)
        colors_right = median_cut_recursive(
            pixels_sorted[midpoint:],
            num_colors - (num_colors // 2)
        )
        return colors_left + colors_right
    
    # Calculate best power of 2
    num_power_of_2 = 2**int(np.ceil(np.log2(n_final))) if n_final > 1 else 1
    
    # Generate candidates
    candidate_rgb_colors = median_cut_recursive(pixels_rgb, num_power_of_2)
    candidate_rgb_colors = np.array(candidate_rgb_colors)
    
    # If more than requested, use k-means to reduce
    if len(candidate_rgb_colors) > n_final:
        candidate_lab = color_converter.rgb2lab(
            candidate_rgb_colors.reshape(1, -1, 3)
        ).reshape(-1, 3)
        
        kmeans = KMeans(n_clusters=n_final, n_init=5, random_state=42)
        kmeans.fit(candidate_lab)
        final_lab_centers = kmeans.cluster_centers_
        
        final_rgb = np.clip(
            color_converter.lab2rgb(final_lab_centers.reshape(1, -1, 3)).reshape(-1, 3),
            0, 1
        )
    else:
        final_rgb = candidate_rgb_colors
    
    # Convert to hex
    hex_palette = [rgb_to_hex(c) for c in final_rgb]
    
    return hex_palette


def extract_all_palettes(image: np.ndarray, n_colors: int = 6) -> dict:
    """
    Extract palettes using all three methods.
    
    Args:
        image: Numpy RGB image array
        n_colors: Number of colors to extract
    
    Returns:
        Dictionary with keys: 'kmeans', 'histogram', 'median_cut'
    """
    return {
        'kmeans': extract_dominant_colors_kmeans(image, n_final=n_colors),
        'histogram': extract_palette_histogram(image, n_final=n_colors),
        'median_cut': extract_palette_median_cut(image, n_final=n_colors)
    }
