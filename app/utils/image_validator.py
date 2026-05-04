import io
from PIL import Image
from app.config.settings import settings


class ImageValidator:
    """Validator untuk file gambar dengan kriteria ketat."""
    
    # Konfigurasi konstant
    MIN_SIZE_BYTES = getattr(settings, "MIN_IMAGE_SIZE_BYTES", 1024)  # 1 KB
    # Prefer explicit bytes setting, otherwise compute from MB setting
    MAX_SIZE_BYTES = getattr(
        settings,
        "MAX_IMAGE_SIZE_BYTES",
        getattr(settings, "MAX_IMAGE_SIZE_MB", 50) * 1024 * 1024,
    )
    ALLOWED_FORMATS = set(getattr(settings, "ALLOWED_IMAGE_FORMATS", ["JPEG", "PNG", "WEBP"]))
    ALLOWED_CONTENT_TYPES = set(getattr(
        settings, "ALLOWED_CONTENT_TYPES", ["image/jpeg", "image/png", "image/webp"]
    ))

    @staticmethod
    def validate_file_size(file_size: int) -> tuple[bool, str]:
        """
        Validasi ukuran file gambar.
        
        Args:
            file_size: Ukuran file dalam bytes
            
        Returns:
            (is_valid, error_message)
        """
        if file_size < ImageValidator.MIN_SIZE_BYTES:
            return False, f"File size must be at least {ImageValidator.MIN_SIZE_BYTES} bytes (1 KB)"
        
        if file_size > ImageValidator.MAX_SIZE_BYTES:
            max_mb = ImageValidator.MAX_SIZE_BYTES / (1024 * 1024)
            return False, f"File size must not exceed {max_mb:.1f} MB"
        
        return True, ""

    @staticmethod
    def validate_content_type(content_type: str) -> tuple[bool, str]:
        """
        Validasi content type gambar.
        
        Args:
            content_type: Content type dari file
            
        Returns:
            (is_valid, error_message)
        """
        if not content_type:
            return False, "Content-Type header is missing"
        
        if content_type not in ImageValidator.ALLOWED_CONTENT_TYPES:
            allowed = ", ".join(sorted(ImageValidator.ALLOWED_CONTENT_TYPES))
            return False, f"Invalid content type. Allowed: {allowed}"
        
        return True, ""

    @staticmethod
    def validate_image_format(file_content: bytes) -> tuple[bool, str]:
        """
        Validasi format gambar dengan membaca magic bytes dan PIL.
        Mencegah file yang hanya mengganti ekstensi.
        
        Args:
            file_content: Binary content file
            
        Returns:
            (is_valid, error_message)
        """
        try:
            # Baca image dengan PIL untuk verifikasi lebih ketat
            image = Image.open(io.BytesIO(file_content))
            
            # Verifikasi format
            image_format = image.format
            if image_format not in ImageValidator.ALLOWED_FORMATS:
                allowed = ", ".join(sorted(ImageValidator.ALLOWED_FORMATS))
                return False, f"Invalid image format: {image_format}. Allowed: {allowed}"
            
            # Verifikasi image bisa diload dengan sempurna
            image.verify()
            
            return True, ""
        
        except Image.UnidentifiedImageError:
            return False, "File is not a valid image"
        except Exception as e:
            return False, f"Image validation error: {str(e)}"

    @staticmethod
    def validate_full(
        file_content: bytes,
        content_type: str,
    ) -> tuple[bool, str]:
        """
        Validasi lengkap file gambar dengan semua kriteria.
        
        Args:
            file_content: Binary content file
            content_type: Content type dari file
            
        Returns:
            (is_valid, error_message)
        """
        # Validasi content type
        is_valid, error_msg = ImageValidator.validate_content_type(content_type)
        if not is_valid:
            return is_valid, error_msg
        
        # Validasi ukuran
        is_valid, error_msg = ImageValidator.validate_file_size(len(file_content))
        if not is_valid:
            return is_valid, error_msg
        
        # Validasi format file
        is_valid, error_msg = ImageValidator.validate_image_format(file_content)
        if not is_valid:
            return is_valid, error_msg
        
        return True, ""
