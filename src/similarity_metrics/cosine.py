"""
Cosine similarity metric for comparing images.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .base import SimilarityMetric


class CosineSimilarity(SimilarityMetric):
    """Cosine similarity metric for comparing images."""
    
    @property
    def name(self) -> str:
        return "Cosine Similarity"
    
    def calculate(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate the cosine similarity between two images.
        
        Args:
            img1: First image as numpy array
            img2: Second image as numpy array
            
        Returns:
            Float value representing the cosine similarity score
        """
        # Resize images if they have different shapes
        if img1.shape != img2.shape:
            # Resize to the smaller of the two dimensions
            min_height = min(img1.shape[0], img2.shape[0])
            min_width = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_height, :min_width]
            img2 = img2[:min_height, :min_width]
        
        # Handle grayscale images
        if len(img1.shape) == 2:
            img1 = np.expand_dims(img1, axis=2)
        if len(img2.shape) == 2:
            img2 = np.expand_dims(img2, axis=2)
            
        # Ensure all image values are float type for numerical stability
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        # Flatten the images
        img1_flat = img1.flatten().reshape(1, -1)
        img2_flat = img2.flatten().reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(img1_flat, img2_flat)[0][0]
        
        return similarity 