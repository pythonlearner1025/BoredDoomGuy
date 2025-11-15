#!/bin/bash
# Quickstart script for preprocessing and uploading Doom dataset

set -e  # Exit on error

echo "=========================================="
echo "Doom Dataset Preprocessing Quickstart"
echo "=========================================="
echo ""

# Configuration
INPUT_DIR="debug/rnd/20251023_150242"
OUTPUT_DIR="doom_dataset_processed"
HF_REPO="your-username/doom-dataset"  # Change this!
TARGET_HEIGHT=120
TARGET_WIDTH=160

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory not found: $INPUT_DIR"
    exit 1
fi

# Install dependencies
echo "Step 1: Installing dependencies..."
pip install -q datasets pyarrow pillow tqdm huggingface_hub
echo "✓ Dependencies installed"
echo ""

# Preprocess dataset
echo "Step 2: Preprocessing dataset..."
echo "  Input: $INPUT_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Resolution: ${TARGET_HEIGHT}x${TARGET_WIDTH}"
echo ""

python preprocess_for_hf.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --target_height "$TARGET_HEIGHT" \
    --target_width "$TARGET_WIDTH" \
    --chunk_size 100

echo ""
echo "✓ Preprocessing complete"
echo ""

# Benchmark (optional)
echo "Step 3: Benchmarking performance..."
echo ""
python benchmark_dataloading.py \
    --png_dir "$INPUT_DIR" \
    --arrow_dir "$OUTPUT_DIR" \
    --batch_size 4 \
    --n_hist 64 \
    --num_batches 50

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. Upload to HuggingFace Hub:"
echo "   huggingface-cli login"
echo "   python upload_to_hf.py \\"
echo "       --dataset_path $OUTPUT_DIR \\"
echo "       --repo_id $HF_REPO \\"
echo "       --private"
echo ""
echo "2. Update your training code:"
echo "   from hf_dataset_loader import HFDoomDataset"
echo "   dataset = HFDoomDataset('$HF_REPO', n_hist=64, from_hub=True)"
echo ""
echo "3. See DATASET_PREPROCESSING.md for detailed instructions"
echo ""

