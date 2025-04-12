"""
Utility functions for image processing.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple


def load_image(path: str) -> np.ndarray:
    """Load an image from the given path.
    
    Args:
        path: Path to the image file
        
    Returns:
        Image as numpy array
        
    Raises:
        FileNotFoundError: If the image file is not found
        ValueError: If the image cannot be loaded
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    
    # Use IMREAD_COLOR flag to ensure 3 channels
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img


def list_image_files(directory: str) -> List[str]:
    """List all image files in the given directory.
    
    Args:
        directory: Path to the directory
        
    Returns:
        List of paths to image files
        
    Raises:
        FileNotFoundError: If the directory is not found
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Common image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    
    image_files = []
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(directory, file))
    
    return sorted(image_files)  # Sort files for consistent ordering


def resize_image_for_display(img: np.ndarray, max_width: int = 800, max_height: int = 800) -> np.ndarray:
    """Resize an image for display while preserving aspect ratio.
    
    Args:
        img: Image as numpy array
        max_width: Maximum width for the resized image
        max_height: Maximum height for the resized image
        
    Returns:
        Resized image as numpy array
    """
    height, width = img.shape[:2]
    
    # If the image is already smaller than the maximum dimensions, don't resize
    if width <= max_width and height <= max_height:
        return img
    
    # Calculate the scaling factor
    scale = min(max_width / width, max_height / height)
    
    # Calculate the new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize the image
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized_img 