#!/bin/bash

echo "Setting up Image Similarity Benchmark..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Make run.py executable
chmod +x run.py

echo "Setup complete! You can now run the application with:"
echo "./run.py"

# Ask if the user wants to run the application now
read -p "Do you want to run the application now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "./run.py"
    ./run.py
fi 