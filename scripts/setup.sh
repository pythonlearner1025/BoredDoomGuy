#!/bin/bash
set -e

echo "=== Doom IDM Environment Setup ==="
echo "Setting up Python 3.12 environment with ViZDoom"
echo

# Use Python 3.12 (PyTorch compatible)
PYVER=3.12

# Check if Python 3.12 is available
if ! command -v python$PYVER &> /dev/null; then
    echo "Error: Python $PYVER not found. Please install it first."
    echo "On Ubuntu/Debian: sudo apt-get install python$PYVER python$PYVER-venv python$PYVER-dev"
    echo "On Fedora/RHEL: sudo dnf install python$PYVER python$PYVER-devel"
    exit 1
fi

# Create virtual environment with Python 3.12
echo "Creating virtual environment with Python $PYVER..."
python$PYVER -m venv env$PYVER

# Activate environment
source env$PYVER/bin/activate

# Bootstrap pip in the venv
echo "Bootstrapping pip..."
python -m ensurepip
python -m pip install --upgrade pip

# Install system dependencies for ViZDoom (Ubuntu/Debian)
echo "Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y \
        build-essential cmake git \
        libboost-all-dev libsdl2-dev \
        libopenal-dev libsndfile1-dev \
        zlib1g-dev libjpeg-dev tar libbz2-dev \
        libgtk2.0-dev python3-dev wget
elif command -v dnf &> /dev/null; then
    sudo dnf install -y \
        cmake git boost-devel SDL2-devel \
        openal-soft-devel libsndfile-devel \
        bzip2-devel gtk2-devel python3-devel
else
    echo "Warning: Unknown package manager. Install dependencies manually."
fi

# Install Python packages
echo "Installing Python packages..."
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt

# Clone and build ViZDoom from source
echo "Building ViZDoom from source..."
if [ ! -d "ViZDoom" ]; then
    git clone https://github.com/Farama-Foundation/ViZDoom.git
fi

cd ViZDoom
python -m pip install .
cd ..

echo
echo "=== Setup Complete ==="
echo "Activate environment: source env$PYVER/bin/activate"
echo "Run training: python idm.py"
echo
