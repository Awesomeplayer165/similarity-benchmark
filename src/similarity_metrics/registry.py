"""
Registry for all similarity metrics.
"""

from typing import Dict, Type, List
from .base import SimilarityMetric
from .cosine import CosineSimilarity
from .sewar_metrics import (
    MSESimilarity, RMSESimilarity, PSNRSimilarity,
    SSIMSimilarity, UQISimilarity, MSSSIMSimilarity,
    ERGASSimilarity, SCCSimilarity, RASESimilarity,
    SAMSimilarity, VIFSimilarity
)


class SimilarityMetricRegistry:
    """Registry for all similarity metrics."""
    
    _metrics: Dict[str, Type[SimilarityMetric]] = {}
    
    @classmethod
    def register(cls, metric_class: Type[SimilarityMetric]) -> None:
        """Register a similarity metric.
        
        Args:
            metric_class: The similarity metric class to register
        """
        instance = metric_class()
        cls._metrics[instance.name] = metric_class
    
    @classmethod
    def get_metric(cls, name: str) -> SimilarityMetric:
        """Get a similarity metric by name.
        
        Args:
            name: Name of the similarity metric
            
        Returns:
            Instance of the similarity metric
            
        Raises:
            KeyError: If the metric is not registered
        """
        if name not in cls._metrics:
            raise KeyError(f"Similarity metric '{name}' not registered")
        
        return cls._metrics[name]()
    
    @classmethod
    def get_all_metrics(cls) -> List[SimilarityMetric]:
        """Get all registered similarity metrics.
        
        Returns:
            List of all registered similarity metrics
        """
        return [metric_class() for metric_class in cls._metrics.values()]
    
    @classmethod
    def get_metric_names(cls) -> List[str]:
        """Get the names of all registered similarity metrics.
        
        Returns:
            List of all registered similarity metric names
        """
        return list(cls._metrics.keys())


# Register all similarity metrics
SimilarityMetricRegistry.register(CosineSimilarity)
SimilarityMetricRegistry.register(MSESimilarity)
SimilarityMetricRegistry.register(RMSESimilarity)
SimilarityMetricRegistry.register(PSNRSimilarity)
SimilarityMetricRegistry.register(SSIMSimilarity)
SimilarityMetricRegistry.register(UQISimilarity)
SimilarityMetricRegistry.register(MSSSIMSimilarity)
SimilarityMetricRegistry.register(ERGASSimilarity)
SimilarityMetricRegistry.register(SCCSimilarity)
SimilarityMetricRegistry.register(RASESimilarity)
SimilarityMetricRegistry.register(SAMSimilarity)
SimilarityMetricRegistry.register(VIFSimilarity) 