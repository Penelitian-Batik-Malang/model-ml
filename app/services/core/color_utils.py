"""
Color utility functions for format conversion and validation
"""

import re
import numpy as np


def hex_to_rgb(hex_color: str) -> tuple:
    """
    Convert hex color to RGB tuple.
    
    Args:
        hex_color: Hex color string (e.g., '#FF0000')
    
    Returns:
        Tuple of (R, G, B) in range 0-255
    
    Example:
        >>> hex_to_rgb('#FF0000')
        (255, 0, 0)
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: np.ndarray) -> str:
    """
    Convert RGB array to hex color string.
    
    Args:
        rgb: RGB values, either as array in 0-1 range or 0-255 range
    
    Returns:
        Hex color string (e.g., '#FF0000')
    
    Example:
        >>> rgb_to_hex(np.array([1.0, 0.0, 0.0]))
        '#FF0000'
        >>> rgb_to_hex(np.array([255, 0, 0]))
        '#FF0000'
    """
    # Normalize to 0-255 range if needed
    if rgb.max() <= 1.0:
        rgb = (rgb * 255).astype(np.uint8)
    else:
        rgb = rgb.astype(np.uint8)
    
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def validate_hex_color(hex_color: str) -> bool:
    """
    Validate hex color string format.
    
    Args:
        hex_color: Hex color string to validate
    
    Returns:
        True if valid hex color format, False otherwise
    
    Example:
        >>> validate_hex_color('#FF0000')
        True
        >>> validate_hex_color('FF0000')
        False
    """
    pattern = r'^#[0-9A-Fa-f]{6}$'
    return re.match(pattern, hex_color) is not None
