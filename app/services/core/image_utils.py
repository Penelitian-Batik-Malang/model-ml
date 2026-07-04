"""
Image utility functions for encoding/decoding and file handling
"""

import io
import base64
import os
import uuid
from datetime import datetime
import numpy as np
from PIL import Image


def numpy_to_base64(image: np.ndarray, format_type="JPEG", quality=95) -> str:
    """
    Convert numpy RGB image to base64 string.
    
    Args:
        image: Numpy array with shape (H, W, 3) in uint8 range 0-255
        format_type: PIL image format ('JPEG', 'PNG', etc.)
        quality: JPEG quality (1-100)
    
    Returns:
        Base64 encoded string
    """
    # Ensure uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Convert to PIL Image
    pil_img = Image.fromarray(image, mode='RGB')
    
    # Encode to bytes
    buffer = io.BytesIO()
    if format_type == "JPEG":
        pil_img.save(buffer, format=format_type, quality=quality)
    else:
        pil_img.save(buffer, format=format_type)
    
    # Encode to base64
    buffer.seek(0)
    img_bytes = buffer.getvalue()
    b64_string = base64.b64encode(img_bytes).decode('utf-8')
    
    return b64_string


def base64_to_numpy(b64_string: str) -> np.ndarray:
    """
    Convert base64 string to numpy RGB image.
    
    Args:
        b64_string: Base64 encoded image string
    
    Returns:
        Numpy array with shape (H, W, 3) in uint8 range 0-255
    """
    # Decode base64
    img_bytes = base64.b64decode(b64_string)
    
    # Convert to PIL Image
    buffer = io.BytesIO(img_bytes)
    pil_img = Image.open(buffer)
    
    # Convert to numpy RGB
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    
    image = np.array(pil_img)
    return image


def file_to_numpy(file_bytes: bytes, max_width: int = None, max_height: int = None) -> np.ndarray:
    """
    Convert image bytes to numpy RGB image.
    Optionally resize image while preserving aspect ratio.
    
    Args:
        file_bytes: Raw image file bytes
        max_width: Maximum width in pixels (optional, no resize if None)
        max_height: Maximum height in pixels (optional, no resize if None)
    
    Returns:
        Numpy array with shape (H, W, 3) in uint8 range 0-255
    
    Raises:
        ValueError: If image cannot be opened
    """
    try:
        # Open with PIL
        pil_img = Image.open(io.BytesIO(file_bytes))
        
        # Convert to RGB if needed
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        # Resize if max dimensions provided
        if max_width is not None and max_height is not None:
            pil_img = resize_image_preserving_ratio(pil_img, max_width, max_height)
        
        # Convert to numpy
        image = np.array(pil_img)
        return image
    except Exception as e:
        raise ValueError(f"Failed to load image from file: {e}")


def resize_image_preserving_ratio(pil_img: Image.Image, max_width: int, max_height: int) -> Image.Image:
    """
    Resize image to fit within max_width and max_height while preserving aspect ratio.
    
    Args:
        pil_img: PIL Image object
        max_width: Maximum width in pixels
        max_height: Maximum height in pixels
    
    Returns:
        Resized PIL Image object (preserving aspect ratio)
    
    Example:
        >>> img = Image.new('RGB', (2000, 1500))
        >>> resized = resize_image_preserving_ratio(img, 1280, 1280)
        >>> print(resized.size)  # Will be (1280, 960) to maintain 4:3 ratio
    """
    width, height = pil_img.size
    
    # Calculate scaling factor to fit within max dimensions
    # Use the smaller scaling factor to ensure both dimensions fit
    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h)
    
    # If scale is >= 1, no resize needed
    if scale >= 1.0:
        return pil_img
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize using high-quality resampling
    resized_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return resized_img


def allowed_file(filename: str, allowed_extensions: set) -> bool:
    """
    Check if filename has allowed extension.
    
    Args:
        filename: Filename to check
        allowed_extensions: Set of allowed extensions (without dot)
    
    Returns:
        True if file has allowed extension, False otherwise
    
    Example:
        >>> allowed_file('image.jpg', {'jpg', 'jpeg', 'png'})
        True
        >>> allowed_file('document.pdf', {'jpg', 'jpeg', 'png'})
        False
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def numpy_to_file(image: np.ndarray, upload_folder: str, format_type="JPEG", quality=95) -> str:
    """
    Convert numpy RGB image to file and save in upload folder.
    
    Args:
        image: Numpy array with shape (H, W, 3) in uint8 range 0-255
        upload_folder: Path to upload folder
        format_type: PIL image format ('JPEG', 'PNG', etc.)
        quality: JPEG quality (1-100)
    
    Returns:
        Relative file path from upload folder (e.g., 'results/image_123abc.jpg')
    
    Raises:
        ValueError: If folder doesn't exist or image cannot be saved
    """
    try:
        # Ensure upload folder exists
        os.makedirs(upload_folder, exist_ok=True)
        
        # Create results subfolder
        results_folder = os.path.join(upload_folder, 'results')
        os.makedirs(results_folder, exist_ok=True)
        
        # Ensure uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        extension = 'jpg' if format_type == 'JPEG' else format_type.lower()
        filename = f"result_{timestamp}_{unique_id}.{extension}"
        
        # Full file path
        file_path = os.path.join(results_folder, filename)
        
        # Convert to PIL Image and save
        pil_img = Image.fromarray(image, mode='RGB')
        
        if format_type == "JPEG":
            pil_img.save(file_path, format=format_type, quality=quality)
        else:
            pil_img.save(file_path, format=format_type)
        
        # Return relative path for frontend
        relative_path = os.path.join('results', filename)
        return relative_path
    
    except Exception as e:
        raise ValueError(f"Failed to save image to file: {e}")
