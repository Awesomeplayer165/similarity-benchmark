"""
Similarity metrics using the sewar library.

This module implements multiple similarity metrics for comparing images:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)
- Universal Quality Image Index (UQI)
- Multi-scale Structural Similarity Index (MS-SSIM)
- Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS)
- Spatial Correlation Coefficient (SCC)
- Relative Average Spectral Error (RASE)
- Spectral Angle Mapper (SAM)
- Visual Information Fidelity (VIF)
"""

import numpy as np
from sewar.full_ref import (
    mse, rmse, psnr, ssim, uqi, msssim, ergas, scc, rase, sam, vifp
)
from .base import SimilarityMetric


class MSESimilarity(SimilarityMetric):
    """Mean Squared Error similarity metric."""
    
    @property
    def name(self) -> str:
        return "Mean Squared Error (MSE)"
    
    def calculate(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate the MSE between two images.
        
        Lower values indicate more similar images.
        
        Args:
            img1: First image as numpy array
            img2: Second image as numpy array
            
        Returns:
            Float value representing the MSE score
        """
        # Make sure images have same shape
        if img1.shape != img2.shape:
            # Resize to the smaller of the two dimensions
            min_height = min(img1.shape[0], img2.shape[0])
            min_width = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_height, :min_width]
            img2 = img2[:min_height, :min_width]
        
        # Calculate MSE
        return float(mse(img1, img2))


class RMSESimilarity(SimilarityMetric):
    """Root Mean Squared Error similarity metric."""
    
    @property
    def name(self) -> str:
        return "Root Mean Squared Error (RMSE)"
    
    def calculate(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate the RMSE between two images.
        
        Lower values indicate more similar images.
        
        Args:
            img1: First image as numpy array
            img2: Second image as numpy array
            
        Returns:
            Float value representing the RMSE score
        """
        # Make sure images have same shape
        if img1.shape != img2.shape:
            # Resize to the smaller of the two dimensions
            min_height = min(img1.shape[0], img2.shape[0])
            min_width = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_height, :min_width]
            img2 = img2[:min_height, :min_width]
        
        # Calculate RMSE
        return float(rmse(img1, img2))


class PSNRSimilarity(SimilarityMetric):
    """Peak Signal-to-Noise Ratio similarity metric."""
    
    @property
    def name(self) -> str:
        return "Peak Signal-to-Noise Ratio (PSNR)"
    
    def calculate(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate the PSNR between two images.
        
        Higher values indicate more similar images.
        
        Args:
            img1: First image as numpy array
            img2: Second image as numpy array
            
        Returns:
            Float value representing the PSNR score
        """
        # Make sure images have same shape
        if img1.shape != img2.shape:
            # Resize to the smaller of the two dimensions
            min_height = min(img1.shape[0], img2.shape[0])
            min_width = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_height, :min_width]
            img2 = img2[:min_height, :min_width]
        
        # Calculate PSNR
        return float(psnr(img1, img2))


class SSIMSimilarity(SimilarityMetric):
    """Structural Similarity Index similarity metric."""
    
    @property
    def name(self) -> str:
        return "Structural Similarity Index (SSIM)"
    
    def calculate(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate the SSIM between two images.
        
        Higher values (closer to 1) indicate more similar images.
        
        Args:
            img1: First image as numpy array
            img2: Second image as numpy array
            
        Returns:
            Float value representing the SSIM score
        """
        # Make sure images have same shape
        if img1.shape != img2.shape:
            # Resize to the smaller of the two dimensions
            min_height = min(img1.shape[0], img2.shape[0])
            min_width = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_height, :min_width]
            img2 = img2[:min_height, :min_width]
        
        # Calculate SSIM
        return float(ssim(img1, img2)[0])  # SSIM returns (score, image)


class UQISimilarity(SimilarityMetric):
    """Universal Quality Image Index similarity metric."""
    
    @property
    def name(self) -> str:
        return "Universal Quality Image Index (UQI)"
    
    def calculate(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate the UQI between two images.
        
        Higher values (closer to 1) indicate more similar images.
        
        Args:
            img1: First image as numpy array
            img2: Second image as numpy array
            
        Returns:
            Float value representing the UQI score
        """
        # Make sure images have same shape
        if img1.shape != img2.shape:
            # Resize to the smaller of the two dimensions
            min_height = min(img1.shape[0], img2.shape[0])
            min_width = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_height, :min_width]
            img2 = img2[:min_height, :min_width]
        
        # Calculate UQI
        return float(uqi(img1, img2))


class MSSSIMSimilarity(SimilarityMetric):
    """Multi-scale Structural Similarity Index similarity metric."""
    
    @property
    def name(self) -> str:
        return "Multi-scale Structural Similarity Index (MS-SSIM)"
    
    def calculate(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate the MS-SSIM between two images.
        
        Higher values (closer to 1) indicate more similar images.
        
        Args:
            img1: First image as numpy array
            img2: Second image as numpy array
            
        Returns:
            Float value representing the MS-SSIM score
        """
        # Make sure images have same shape
        if img1.shape != img2.shape:
            # Resize to the smaller of the two dimensions
            min_height = min(img1.shape[0], img2.shape[0])
            min_width = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_height, :min_width]
            img2 = img2[:min_height, :min_width]
        
        # MS-SSIM requires images to be at least 2^level larger, default is 5 levels
        # so minimum dimension should be at least 32
        min_size = 32
        if img1.shape[0] < min_size or img1.shape[1] < min_size:
            return float('nan')  # Return NaN if images are too small
        
        # Calculate MS-SSIM
        try:
            return float(msssim(img1, img2))
        except Exception:
            # Return NaN if calculation fails
            return float('nan')


class ERGASSimilarity(SimilarityMetric):
    """Erreur Relative Globale Adimensionnelle de Synthèse similarity metric."""
    
    @property
    def name(self) -> str:
        return "Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS)"
    
    def calculate(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate the ERGAS between two images.
        
        Lower values indicate more similar images.
        
        Args:
            img1: First image as numpy array
            img2: Second image as numpy array
            
        Returns:
            Float value representing the ERGAS score
        """
        # Make sure images have same shape
        if img1.shape != img2.shape:
            # Resize to the smaller of the two dimensions
            min_height = min(img1.shape[0], img2.shape[0])
            min_width = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_height, :min_width]
            img2 = img2[:min_height, :min_width]
        
        # Calculate ERGAS with a default ratio of 4
        try:
            return float(ergas(img1, img2, r=4))
        except Exception:
            # Return NaN if calculation fails
            return float('nan')


class SCCSimilarity(SimilarityMetric):
    """Spatial Correlation Coefficient similarity metric."""
    
    @property
    def name(self) -> str:
        return "Spatial Correlation Coefficient (SCC)"
    
    def calculate(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate the SCC between two images.
        
        Higher values (closer to 1) indicate more similar images.
        
        Args:
            img1: First image as numpy array
            img2: Second image as numpy array
            
        Returns:
            Float value representing the SCC score
        """
        # Make sure images have same shape
        if img1.shape != img2.shape:
            # Resize to the smaller of the two dimensions
            min_height = min(img1.shape[0], img2.shape[0])
            min_width = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_height, :min_width]
            img2 = img2[:min_height, :min_width]
        
        # Calculate SCC
        try:
            return float(scc(img1, img2))
        except Exception:
            # Return NaN if calculation fails
            return float('nan')


class RASESimilarity(SimilarityMetric):
    """Relative Average Spectral Error similarity metric."""
    
    @property
    def name(self) -> str:
        return "Relative Average Spectral Error (RASE)"
    
    def calculate(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate the RASE between two images.
        
        Lower values indicate more similar images.
        
        Args:
            img1: First image as numpy array
            img2: Second image as numpy array
            
        Returns:
            Float value representing the RASE score
        """
        # Make sure images have same shape
        if img1.shape != img2.shape:
            # Resize to the smaller of the two dimensions
            min_height = min(img1.shape[0], img2.shape[0])
            min_width = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_height, :min_width]
            img2 = img2[:min_height, :min_width]
        
        # Calculate RASE
        try:
            return float(rase(img1, img2))
        except Exception:
            # Return NaN if calculation fails
            return float('nan')


class SAMSimilarity(SimilarityMetric):
    """Spectral Angle Mapper similarity metric."""
    
    @property
    def name(self) -> str:
        return "Spectral Angle Mapper (SAM)"
    
    def calculate(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate the SAM between two images.
        
        Lower values indicate more similar images.
        
        Args:
            img1: First image as numpy array
            img2: Second image as numpy array
            
        Returns:
            Float value representing the SAM score
        """
        # Make sure images have same shape
        if img1.shape != img2.shape:
            # Resize to the smaller of the two dimensions
            min_height = min(img1.shape[0], img2.shape[0])
            min_width = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_height, :min_width]
            img2 = img2[:min_height, :min_width]
        
        # Calculate SAM
        try:
            return float(sam(img1, img2))
        except Exception:
            # Return NaN if calculation fails
            return float('nan')


class VIFSimilarity(SimilarityMetric):
    """Visual Information Fidelity similarity metric."""
    
    @property
    def name(self) -> str:
        return "Visual Information Fidelity (VIF)"
    
    def calculate(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate the VIF between two images.
        
        Higher values indicate more similar images.
        
        Args:
            img1: First image as numpy array
            img2: Second image as numpy array
            
        Returns:
            Float value representing the VIF score
        """
        # Make sure images have same shape
        if img1.shape != img2.shape:
            # Resize to the smaller of the two dimensions
            min_height = min(img1.shape[0], img2.shape[0])
            min_width = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_height, :min_width]
            img2 = img2[:min_height, :min_width]
        
        # Calculate VIF
        try:
            return float(vifp(img1, img2))
        except Exception:
            # Return NaN if calculation fails
            return float('nan') 