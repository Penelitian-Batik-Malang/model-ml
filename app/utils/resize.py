import cv2

class Resize:
    @staticmethod
    def proportional_resize(image, max_size=768):
        """
        Resize gambar secara proporsional sehingga sisi terpanjang tidak melebihi max_size.
        """

        height, width = image.shape[:2]
        max_dim = max(height, width)
        scale = max_size / max_dim if max_dim > max_size else 1.0
        new_width, new_height = int(round(width * scale)), int(round(height * scale))
        return cv2.resize(image, (new_width, new_height))