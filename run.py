#!/usr/bin/env python3
"""
Run script for the Image Similarity Benchmark application.
"""

import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# Import and run the main function
from main import main

if __name__ == "__main__":
    main() 