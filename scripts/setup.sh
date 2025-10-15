#!/bin/bash
set -e

echo "=== Doom IDM Environment Setup ==="
echo "Setting up Python 3.14t free-threading environment with ViZDoom"
echo

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "uv already installed: $(uv --version)"
fi

# Install Python 3.14t free-threading
echo "Installing Python 3.14.0+freethreaded..."
uv python install 3.14t

# Create virtual environment with Python 3.14t
echo "Creating virtual environment with Python 3.14t..."
uv venv env314t --python 3.14t

# Activate environment
source env314t/bin/activate

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
pip install --upgrade pip
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118

# Clone and build ViZDoom from source for Python 3.14t
echo "Building ViZDoom from source..."
if [ ! -d "ViZDoom" ]; then
    git clone https://github.com/Farama-Foundation/ViZDoom.git
fi

cd ViZDoom
pip install .
cd ..

echo
echo "=== Setup Complete ==="
echo "Activate environment: source env314t/bin/activate"
echo "Run training: PYTHON_GIL=0 python idm.py"
echo
