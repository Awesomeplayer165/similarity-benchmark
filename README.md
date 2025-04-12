# Image Similarity Benchmark

A tool for comparing images using various similarity metrics.

## Features
- Select a directory containing images
- View images side by side (LHS and RHS)
- Calculate similarity scores between images
- Display results in a dedicated window
- Extensible architecture for adding new similarity metrics

## Current Similarity Metrics
- Cosine Similarity
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

## Setup
1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python run.py
   ```
   or
   ```
   ./run.py
   ```

## Usage
1. Launch the application
2. Choose a directory containing images (preferably with exactly 2 images)
3. The two images will be displayed in the LHS and RHS windows
4. Click "Calculate Similarity" to compute and display similarity scores
5. Results will be shown in the results window and printed to the console

## Adding New Similarity Metrics

The application is designed to be easily extendable with new similarity metrics. Follow these steps to add a new metric:

1. Create a new Python file in the `src/similarity_metrics` directory, e.g., `src/similarity_metrics/my_new_metric.py`

2. Implement a class that inherits from `SimilarityMetric` and implements the required methods:

```python
from .base import SimilarityMetric
import numpy as np

class MyNewMetric(SimilarityMetric):
    @property
    def name(self) -> str:
        return "My New Metric"
    
    def calculate(self, img1: np.ndarray, img2: np.ndarray) -> float:
        # Implement your similarity calculation logic here
        # img1 and img2 are numpy arrays representing the images
        
        # Example calculation
        result = ...  # Your calculation logic
        
        return result
```

3. Register your new metric in `src/similarity_metrics/registry.py`:

```python
from .my_new_metric import MyNewMetric

# Add this line to the end of the file
SimilarityMetricRegistry.register(MyNewMetric)
```

Your new metric will automatically appear in the results window when you calculate similarity.