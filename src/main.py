"""
Image Similarity Benchmark Application.

A tool for comparing images using various similarity metrics.
"""

import sys
import os
import threading
from typing import List, Dict, Optional
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QLabel, QFileDialog, QTextEdit, QSplitter, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot

from utils.image_utils import load_image, list_image_files, resize_image_for_display
from similarity_metrics.registry import SimilarityMetricRegistry


class SimilarityCalculator(QThread):
    """Thread for calculating similarity metrics in parallel."""
    
    # Define signals to communicate with the main thread
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int, int)  # current, total
    
    def __init__(self, lhs_image: np.ndarray, rhs_image: np.ndarray):
        super().__init__()
        self.lhs_image = lhs_image
        self.rhs_image = rhs_image
        
    def run(self):
        """Run the similarity calculations."""
        # Get all registered similarity metrics
        metrics = SimilarityMetricRegistry.get_all_metrics()
        total = len(metrics)
        
        # Calculate similarity for each metric
        results = {}
        for i, metric in enumerate(metrics):
            try:
                score = metric.calculate(self.lhs_image, self.rhs_image)
                results[metric.name] = score
            except Exception as e:
                results[metric.name] = f"Error: {str(e)}"
            
            # Emit progress signal
            self.progress.emit(i + 1, total)
        
        # Emit finished signal with results
        self.finished.emit(results)


class SimilarityBenchmark(QMainWindow):
    """Main window for the similarity benchmark application."""
    
    def __init__(self):
        super().__init__()
        
        # Images
        self.lhs_image_path: Optional[str] = None
        self.rhs_image_path: Optional[str] = None
        self.lhs_image: Optional[np.ndarray] = None
        self.rhs_image: Optional[np.ndarray] = None
        
        # Calculator thread
        self.calculator: Optional[SimilarityCalculator] = None
        
        # Initialize UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Image Similarity Benchmark")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Button to select directory
        self.select_dir_button = QPushButton("Select Image Directory")
        self.select_dir_button.clicked.connect(self.select_directory)
        main_layout.addWidget(self.select_dir_button)
        
        # Progress bar for similarity calculation
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Splitter for the three windows
        # Using Qt.Orientation.Horizontal instead of Qt.Horizontal for newer PyQt versions
        splitter = QSplitter(Qt.Orientation.Horizontal if hasattr(Qt, 'Orientation') else Qt.Horizontal)
        
        # LHS image window
        self.lhs_group = QGroupBox("Left Image (LHS)")
        lhs_layout = QVBoxLayout()
        self.lhs_label = QLabel("No image selected")
        self.lhs_label.setAlignment(Qt.AlignmentFlag.AlignCenter if hasattr(Qt, 'AlignmentFlag') else Qt.AlignCenter)
        self.lhs_label.setMinimumSize(400, 400)
        lhs_layout.addWidget(self.lhs_label)
        self.lhs_group.setLayout(lhs_layout)
        splitter.addWidget(self.lhs_group)
        
        # RHS image window
        self.rhs_group = QGroupBox("Right Image (RHS)")
        rhs_layout = QVBoxLayout()
        self.rhs_label = QLabel("No image selected")
        self.rhs_label.setAlignment(Qt.AlignmentFlag.AlignCenter if hasattr(Qt, 'AlignmentFlag') else Qt.AlignCenter)
        self.rhs_label.setMinimumSize(400, 400)
        rhs_layout.addWidget(self.rhs_label)
        self.rhs_group.setLayout(rhs_layout)
        splitter.addWidget(self.rhs_group)
        
        # Results window
        self.results_group = QGroupBox("Similarity Results")
        results_layout = QVBoxLayout()
        
        # Replace QTextEdit with QTableWidget for results
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["Metric", "Result", "Expected Range"])
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        results_layout.addWidget(self.results_table)
        
        self.results_group.setLayout(results_layout)
        splitter.addWidget(self.results_group)
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        
        # Calculate button
        self.calculate_button = QPushButton("Calculate Similarity")
        self.calculate_button.clicked.connect(self.calculate_similarity)
        self.calculate_button.setEnabled(False)
        main_layout.addWidget(self.calculate_button)
        
        # Set main layout
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
    
    def select_directory(self):
        """Open a dialog to select a directory containing images."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        
        if not dir_path:
            return
        
        try:
            # List image files in the directory
            image_files = list_image_files(dir_path)
            
            if len(image_files) < 2:
                # Clear the results table and add an error message
                self.results_table.setRowCount(1)
                error_message = f"Error: Directory contains fewer than 2 images. Found {len(image_files)} images."
                self.results_table.setItem(0, 0, QTableWidgetItem(error_message))
                self.results_table.setSpan(0, 0, 1, 3)
                return
            
            # Load the first two images
            self.lhs_image_path = image_files[0]
            self.rhs_image_path = image_files[1]
            
            self.lhs_image = load_image(self.lhs_image_path)
            self.rhs_image = load_image(self.rhs_image_path)
            
            # Display the images
            self.display_image(self.lhs_image, self.lhs_label)
            self.display_image(self.rhs_image, self.rhs_label)
            
            # Enable calculate button
            self.calculate_button.setEnabled(True)
            
            # Update window titles
            self.lhs_group.setTitle(f"Left Image (LHS): {os.path.basename(self.lhs_image_path)}")
            self.rhs_group.setTitle(f"Right Image (RHS): {os.path.basename(self.rhs_image_path)}")
            
            # Automatically calculate similarity
            self.calculate_similarity_in_parallel()
            
        except Exception as e:
            # Clear the results table and add an error message
            self.results_table.setRowCount(1)
            error_message = f"Error: {str(e)}"
            self.results_table.setItem(0, 0, QTableWidgetItem(error_message))
            self.results_table.setSpan(0, 0, 1, 3)
    
    def display_image(self, img: np.ndarray, label: QLabel):
        """Display an image in the given label while preserving aspect ratio and showing full width/height.
        
        Args:
            img: Image as numpy array
            label: Label to display the image in
        """
        # Resize image for display but ensure we don't make it too small
        img_display = resize_image_for_display(img, max_width=800, max_height=800)
        
        # Convert to QImage
        height, width, channel = img_display.shape
        bytes_per_line = 3 * width
        q_img = QImage(img_display.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Set pixmap
        pixmap = QPixmap.fromImage(q_img)
        
        # Set the pixmap and ensure it's not clipped
        label.setPixmap(pixmap)
        label.setScaledContents(True)  # This ensures the image fills the label
        
        # Set minimum size to accommodate the image
        label.setMinimumSize(pixmap.width(), pixmap.height())
    
    def calculate_similarity_in_parallel(self):
        """Calculate similarity metrics in a separate thread."""
        if self.lhs_image is None or self.rhs_image is None:
            # Clear the results table and add an error message
            self.results_table.setRowCount(1)
            error_message = "Error: Both images must be loaded before calculating similarity."
            self.results_table.setItem(0, 0, QTableWidgetItem(error_message))
            self.results_table.setSpan(0, 0, 1, 3)
            return
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Calculating: %p%")
        
        # Create and start the calculator thread
        self.calculator = SimilarityCalculator(self.lhs_image, self.rhs_image)
        self.calculator.finished.connect(self.on_calculation_finished)
        self.calculator.progress.connect(self.update_progress)
        self.calculator.start()
    
    @pyqtSlot(dict)
    def on_calculation_finished(self, results):
        """Handle the completion of similarity calculations.
        
        Args:
            results: Dictionary of similarity metric names and scores
        """
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        # Display results
        self.display_results(results)
    
    @pyqtSlot(int, int)
    def update_progress(self, current, total):
        """Update the progress bar.
        
        Args:
            current: Current progress
            total: Total steps
        """
        percentage = int((current / total) * 100)
        self.progress_bar.setValue(percentage)
    
    def calculate_similarity(self):
        """Calculate similarity between the two images (button handler)."""
        self.calculate_similarity_in_parallel()
    
    def display_results(self, results: Dict[str, float]):
        """Display similarity results in a table.
        
        Args:
            results: Dictionary of similarity metric names and scores
        """
        # Define the expected ranges for each metric
        metric_ranges = {
            "Cosine Similarity": "Range from -1 (completely dissimilar) to 1 (identical), with 0 indicating orthogonality",
            "Mean Squared Error (MSE)": "Range from 0 (identical) to larger values (more different). Lower is better",
            "Root Mean Squared Error (RMSE)": "Range from 0 (identical) to larger values (more different). Lower is better",
            "Peak Signal-to-Noise Ratio (PSNR)": "Range from 0 to ∞ dB. Higher values indicate better quality (more similarity)",
            "Structural Similarity Index (SSIM)": "Range from -1 to 1. Higher values (closer to 1) indicate more similarity",
            "Universal Quality Image Index (UQI)": "Range from -1 to 1. Higher values (closer to 1) indicate more similarity",
            "Multi-scale Structural Similarity Index (MS-SSIM)": "Range from 0 to 1. Higher values indicate more similarity",
            "Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS)": "Range from 0 to ∞. Lower values indicate more similarity",
            "Spatial Correlation Coefficient (SCC)": "Range from -1 to 1. Values closer to 1 indicate more similarity",
            "Relative Average Spectral Error (RASE)": "Range from 0 to ∞. Lower values indicate more similarity",
            "Spectral Angle Mapper (SAM)": "Range from 0 to π/2 radians. Lower values indicate more similarity",
            "Visual Information Fidelity (VIF)": "Range from 0 to 1. Higher values indicate more similarity"
        }
        
        # Clear previous results
        self.results_table.setRowCount(0)
        
        # Add results to table
        for i, (name, score) in enumerate(results.items()):
            self.results_table.insertRow(i)
            
            # Create table items
            name_item = QTableWidgetItem(name)
            
            # Format the score nicely
            if isinstance(score, float):
                score_item = QTableWidgetItem(f"{score:.6f}")
            else:
                score_item = QTableWidgetItem(str(score))
            
            # Get the range description for this metric
            range_item = QTableWidgetItem(metric_ranges.get(name, "Unknown range"))
            
            # Add items to table
            self.results_table.setItem(i, 0, name_item)
            self.results_table.setItem(i, 1, score_item)
            self.results_table.setItem(i, 2, range_item)
        
        # Resize rows to content
        self.results_table.resizeRowsToContents()
        
        # Print to console
        print("\nSimilarity Results:")
        for name, score in results.items():
            print(f"{name}: {score}")


def main():
    """Main function to run the application."""
    app = QApplication(sys.argv)
    window = SimilarityBenchmark()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 