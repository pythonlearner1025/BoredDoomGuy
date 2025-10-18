#!/bin/bash
set -e

echo "=== Simplified Python 3.12 Setup ==="

PYVER=3.12

# Install Python 3.12 if not available (Ubuntu 20.04 compatible)
if ! command -v python$PYVER &> /dev/null; then
    echo "Python $PYVER not found. Installing via deadsnakes PPA..."
    sudo apt-get update
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get update
    sudo apt-get install -y python$PYVER python$PYVER-venv python$PYVER-dev
    echo "Python $PYVER installed successfully"
else
    echo "Python $PYVER already installed"
fi

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "env" ]; then
    echo "Removing existing env directory..."
    rm -rf env
fi
python$PYVER -m venv env

# Activate and upgrade pip
echo "Activating environment and upgrading pip..."
source env/bin/activate
python -m pip install --upgrade pip

# Check for GPU and install appropriate PyTorch
echo "Checking for GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected. Installing CUDA PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "No GPU detected. Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install remaining requirements
echo "Installing remaining requirements..."
pip install -r requirements.txt

# Download DOOM1.wad if not present
if [ ! -f "Doom1.WAD" ]; then
    echo "Downloading DOOM1.wad..."
    wget https://distro.ibiblio.org/slitaz/sources/packages/d/doom1.wad
    mv doom1.wad Doom1.WAD
    echo "DOOM1.wad downloaded successfully"
else
    echo "DOOM1.wad already exists, skipping download"
fi

echo
echo "=== Setup Complete ==="
echo "Activate environment: source env/bin/activate"
echo
