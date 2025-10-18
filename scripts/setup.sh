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

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo
echo "=== Setup Complete ==="
echo "Activate environment: source env/bin/activate"
echo
