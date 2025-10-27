#!/bin/bash

# Installation script for RandomDoomSamples-110M dataset
# This script downloads and sets up the dataset in the current directory

set -e  # Exit on any error

echo "🚀 Installing RandomDoomSamples-110M dataset..."
echo "=================================="

# Check if huggingface_hub is installed
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "❌ huggingface_hub not found. Installing..."
    pip install huggingface_hub datasets
fi

# Dataset information
DATASET_NAME="invocation02/RandomDoomSamples-110M"
LOCAL_DIR="./RandomDoomSamples-110M-data"

echo "📦 Dataset: $DATASET_NAME"
echo "📁 Target directory: $LOCAL_DIR"

# Create target directory
mkdir -p "$LOCAL_DIR"

# Download dataset metadata first
echo "📋 Fetching dataset metadata..."
python3 - << EOF
from huggingface_hub import HfApi
from pathlib import Path
import json

try:
    api = HfApi()
    repo_info = api.repo_info("$DATASET_NAME", repo_type="dataset")
    print(f"✅ Dataset accessible: {repo_info}")

    # Get file list
    files = api.list_repo_files("$DATASET_NAME", repo_type="dataset")
    print(f"📊 Found {len(files)} files in dataset")

    # Save metadata
    metadata = {
        "dataset_name": "$DATASET_NAME",
        "total_files": len(files),
        "download_date": "$(date -I)",
        "repository_info": {
            "id": repo_info.id,
            "sha": repo_info.sha,
            "last_modified": str(repo_info.lastModified)
        }
    }

    with open("dataset_info.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"💾 Saved metadata to dataset_info.json")

except Exception as e:
    print(f"❌ Error accessing dataset: {e}")
    exit(1)
EOF

# Download dataset files (sample for demonstration)
echo "📥 Downloading sample files..."

python3 - << EOF
from huggingface_hub import HfApi
from pathlib import Path
import os

try:
    api = HfApi()

    # Get repository info
    repo_info = api.repo_info("$DATASET_NAME", repo_type="dataset")

    # List files and download first few episodes as samples
    repo_files = api.list_repo_files("$DATASET_NAME", repo_type="dataset")
    files = [f for f in repo_files]

    # Filter for episode files (limit to first 3 episodes for demo)
    episode_files = [f for f in files if f.path.startswith("episodes/") and f.path.count("/") == 2]

    # Group by episode
    episodes = {}
    for file in episode_files:
        episode_num = file.path.split("/")[1]
        if episode_num not in episodes:
            episodes[episode_num] = []
        episodes[episode_num].append(file)

    # Download first 3 episodes
    downloaded_episodes = 0
    for episode_num in sorted(episodes.keys())[:3]:
        if downloaded_episodes >= 3:
            break

        episode_dir = Path("$LOCAL_DIR") / episode_num
        episode_dir.mkdir(parents=True, exist_ok=True)

        print(f"📂 Downloading episode {episode_num}...")

        for file in episodes[episode_num]:
            local_path = episode_dir / Path(file.path).name
            print(f"  📄 {file.path}")

            # Download file
            api.hf_hub_download(
                repo_id="$DATASET_NAME",
                filename=file.path,
                repo_type="dataset",
                local_dir=str(episode_dir),
                resume_download=True
            )

        downloaded_episodes += 1

    print(f"✅ Downloaded {downloaded_episodes} episodes as samples")
    print(f"📁 Sample data saved to: $LOCAL_DIR")

except Exception as e:
    print(f"❌ Download error: {e}")
    exit(1)
EOF

# Create usage script
echo "📜 Creating usage script..."

cat > load_dataset_example.py << 'EOF'
#!/usr/bin/env python3
"""
Example script to load the RandomDoomSamples-110M dataset
"""

from datasets import load_dataset
from pathlib import Path

def load_sample_data(data_dir="./RandomDoomSamples-110M-data"):
    """Load sample data from downloaded dataset."""

    print(f"🔍 Loading data from: {data_dir}")

    # Load dataset (streaming mode for memory efficiency)
    try:
        dataset = load_dataset(
            "invocation02/RandomDoomSamples-110M",
            data_dir=data_dir,
            split="train",
            streaming=True  # Important for large datasets
        )

        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Dataset info:")

        # Show first few examples
        for i, example in enumerate(dataset):
            if i >= 5:  # Limit output
                break

            print(f"\n--- Example {i+1} ---")
            if hasattr(example, 'image'):
                print(f"🖼️  Image available: {example['image']}")
            if 'frame_metadata' in example:
                print(f"📋  Metadata: {example['frame_metadata']}")

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

    return dataset

def main():
    """Main function."""
    print("🚀 RandomDoomSamples-110M Dataset Loader")
    print("=" * 40)

    # Load sample data
    dataset = load_sample_data()

    if dataset:
        print(f"\n🎯 Ready for training!")
        print(f"💡 Use streaming=True for memory efficiency with large datasets")
        print(f"🔗 Full dataset: https://huggingface.co/datasets/invocation02/RandomDoomSamples-110M")

if __name__ == "__main__":
    main()
EOF

chmod +x load_dataset_example.py

# Create requirements file
echo "📦 Creating requirements.txt..."

cat > requirements.txt << EOF
huggingface_hub>=0.20.0
datasets>=2.0.0
Pillow>=8.0.0
numpy>=1.21.0
EOF

# Final summary
echo ""
echo "🎉 Installation completed!"
echo "=================================="
echo "📁 Dataset directory: $LOCAL_DIR"
echo "🔗 Dataset URL: https://huggingface.co/datasets/invocation02/RandomDoomSamples-110M"
echo "📜 Usage example: python3 load_dataset_example.py"
echo "📦 Requirements: pip install -r requirements.txt"
echo ""
echo "💡 Next steps:"
echo "   1. Run: python3 load_dataset_example.py"
echo "   2. Use streaming=True for memory efficiency"
echo "   3. Process episodes as needed for your training"
echo ""