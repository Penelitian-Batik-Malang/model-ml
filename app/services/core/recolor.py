"""
Image recoloring logic using deep learning models
Port dari testing notebook Febrio SegmentRecolor
"""

import torch
import numpy as np
import torchvision.transforms as transforms
import skimage.color as color_converter
from app.services.core.model_loader import ModelLoader


def prepare_image(image: np.ndarray, target_size=256) -> torch.Tensor:
    """
    Preprocess numpy RGB image to LAB tensor.
    
    Args:
        image: Numpy array RGB image, uint8 range 0-255 or float 0-1
        target_size: Target size (will be padded to multiple of 16)
    
    Returns:
        Tensor of shape (1, 3, H, W) in LAB color space, normalized
    """
    # Ensure RGB format
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    
    # Normalize to 0-1 range
    if image.max() > 1.0:
        image = image.astype(np.float64) / 255.0
    else:
        image = image.astype(np.float64)
    
    # Convert to LAB color space
    img_lab = color_converter.rgb2lab(image)
    
    # Normalize LAB values
    img_normalized = (img_lab - [50, 0, 0]) / [50, 128, 128]
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float()
    
    # Resize to multiple of 16
    h = 16 * int(img_tensor.shape[1] / 16)
    w = 16 * int(img_tensor.shape[2] / 16)
    
    if h == 0 or w == 0:
        h = target_size
        w = target_size
    
    resize_transform = transforms.Resize((h, w))
    img_tensor = resize_transform(img_tensor)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor


def prepare_palette(hex_colors: list, model_size=6) -> tuple:
    """
    Prepare color palette from hex color list.
    
    Features:
    - Auto-pad if fewer than model_size colors (repeat last color)
    - Auto-truncate if more than model_size colors (keep first N)
    
    Args:
        hex_colors: List of hex color strings (e.g., ['#FF0000', '#00FF00'])
        model_size: Target palette size (default 6)
    
    Returns:
        Tuple of (palette_tensor, palette_np):
        - palette_tensor: torch.Tensor shape (1, 18) for model
        - palette_np: numpy array shape (1, 6, 3) for visualization (0-1 range)
    """
    if not isinstance(hex_colors, (list, tuple, np.ndarray)):
        raise ValueError("hex_colors must be list/tuple/array of hex strings")
    
    hex_list = list(hex_colors)
    if len(hex_list) == 0:
        raise ValueError("Palette list cannot be empty!")
    
    # PAD or TRUNCATE to match model_size
    original_length = len(hex_list)
    if len(hex_list) < model_size:
        # Pad: repeat last color
        while len(hex_list) < model_size:
            hex_list.append(hex_list[-1])
    elif len(hex_list) > model_size:
        # Truncate: keep first N colors
        hex_list = hex_list[:model_size]
    
    # Convert hex to RGB
    from app.services.core.color_utils import hex_to_rgb
    rgb_palette = [hex_to_rgb(color) for color in hex_list]
    
    # Reshape to format: (1, 6, 3)
    palette_np = np.array(rgb_palette).reshape(1, model_size, 3) / 255.0
    
    # Convert to LAB color space
    palette_lab = color_converter.rgb2lab(palette_np)
    
    # Normalize LAB
    palette_normalized = (palette_lab - [50, 0, 0]) / [50, 128, 128]
    
    # Flatten and convert to tensor
    palette_tensor = torch.from_numpy(palette_normalized.flatten()).float()
    
    # Add batch dimension
    palette_tensor = palette_tensor.unsqueeze(0)
    
    return palette_tensor, palette_np


def recolor_image(image_tensor: torch.Tensor, palette_tensor: torch.Tensor) -> np.ndarray:
    """
    Apply recoloring to image using loaded models.
    
    Args:
        image_tensor: Preprocessed image tensor (1, 3, H, W) in LAB
        palette_tensor: Prepared palette tensor (1, 18)
    
    Returns:
        Recolored RGB image as numpy array, shape (H, W, 3), uint8 range 0-255
    """
    try:
        loader = ModelLoader.get_instance()
        FE, RD, device = loader.get_models()
    except RuntimeError as e:
        raise RuntimeError(f"Models not loaded: {e}")
    
    with torch.no_grad():
        # Move to device
        image_tensor = image_tensor.to(device)
        palette_tensor = palette_tensor.to(device)
        
        # Extract illumination channel (L channel from LAB)
        illu = image_tensor[:, 0:1, :, :]
        
        # Forward pass through encoder and decoder
        c1, c2, c3, c4 = FE(image_tensor)
        out = RD(c1, c2, c3, c4, palette_tensor, illu)
        
        # Reconstruct LAB image
        final_lab = torch.cat([(illu + 1) * 50, out * 128], axis=1)
        final_lab = final_lab.permute(0, 2, 3, 1)[0].cpu().numpy()
        
        # Convert back to RGB
        final_rgb = color_converter.lab2rgb(final_lab)
    
    # Convert to uint8
    if final_rgb.max() <= 1.0:
        final_rgb = (final_rgb * 255).astype(np.uint8)
    else:
        final_rgb = final_rgb.astype(np.uint8)
    
    return final_rgb


def recolor_with_white_preserve(
    image: np.ndarray,
    palette_hex: list,
    white_threshold: float = 240,
    blend_margin: float = 10
) -> np.ndarray:
    """
    Full recoloring pipeline with white area preservation.
    
    Args:
        image: Input RGB image, numpy array uint8 0-255
        palette_hex: List of target hex colors
        white_threshold: Threshold for white pixels (0-765, default 240)
        blend_margin: Soft blend margin around white areas
    
    Returns:
        Recolored RGB image as numpy array uint8 0-255
    """
    # Load original image info
    if image.max() > 1.0:
        original_image_rgb = image.astype(np.float32) / 255.0
    else:
        original_image_rgb = image.astype(np.float32)
    
    # Prepare tensors
    img_tensor = prepare_image(image)
    palette_tensor, _ = prepare_palette(palette_hex)
    
    # Recolor
    recolored_img = recolor_image(img_tensor, palette_tensor)
    
    if recolored_img.max() > 1.0:
        recolored_img = recolored_img.astype(np.float32) / 255.0
    
    # Resize original if needed
    if original_image_rgb.shape[:2] != recolored_img.shape[:2]:
        from skimage.transform import resize
        original_resized = resize(
            original_image_rgb,
            recolored_img.shape[:2],
            anti_aliasing=True,
            preserve_range=False
        )
    else:
        original_resized = original_image_rgb
    
    # Detect white areas
    threshold_normalized = white_threshold / 255.0
    white_mask = np.all(original_resized > threshold_normalized, axis=-1)
    
    # Apply soft blending if margin > 0
    if blend_margin > 0:
        try:
            from scipy.ndimage import distance_transform_edt
            
            # Calculate distance from white area edges
            distance = distance_transform_edt(~white_mask)
            
            # Create alpha mask for smooth transition
            blend_margin_normalized = blend_margin / 255.0 * 100
            alpha = np.clip(1 - (distance / blend_margin_normalized), 0, 1)
            alpha[white_mask] = 1.0
            
            # Expand for broadcasting
            alpha_3d = alpha[:, :, np.newaxis]
            
            # Blend
            final_img = original_resized * alpha_3d + recolored_img * (1 - alpha_3d)
        except ImportError:
            # Fallback if scipy not available
            final_img = recolored_img.copy()
            final_img[white_mask] = original_resized[white_mask]
    else:
        # Hard masking
        final_img = recolored_img.copy()
        final_img[white_mask] = original_resized[white_mask]
    
    # Convert back to uint8
    if final_img.max() <= 1.0:
        final_img = (final_img * 255).astype(np.uint8)
    else:
        final_img = final_img.astype(np.uint8)
    
    return final_img
