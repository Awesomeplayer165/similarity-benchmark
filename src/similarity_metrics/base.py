"""
Base class for all similarity metrics.
"""

from abc import ABC, abstractmethod
import numpy as np


class SimilarityMetric(ABC):
    """Base class for all similarity metrics.
    
    All similarity metrics should inherit from this class and implement
    the calculate method.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the similarity metric."""
        pass
    
    @abstractmethod
    def calculate(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate the similarity between two images.
        
        Args:
            img1: First image as numpy array
            img2: Second image as numpy array
            
        Returns:
            Float value representing the similarity score
        """
        pass 