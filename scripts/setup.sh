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
