#!/bin/bash

# Default directory
DEFAULT_DIR="/home/minjune/doom/debug/20251018_143410/iter_0031"
DIR="$DEFAULT_DIR"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dir)
            DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dir <directory>]"
            exit 1
            ;;
    esac
done

# Check if directory exists
if [ ! -d "$DIR" ]; then
    echo "Error: Directory '$DIR' does not exist"
    exit 1
fi

# Process all frame_*.txt files in the directory
for file in "$DIR"/frame_*.txt; do
    # Skip if no files match
    [ -e "$file" ] || continue
    
    # Extract frame number from filename (e.g., frame_0000_STRAFE_R.txt -> 0000)
    filename=$(basename "$file")
    frame_idx=$(echo "$filename" | sed -n 's/frame_\([0-9]*\)_.*/\1/p')
    
    # Extract extrinsic reward from file content
    # The file has literal \n characters, so we need to extract from the whole line
    extrinsic_reward=$(grep -o "Extrinsic Reward: [0-9.]*" "$file" | awk '{print $3}')
    
    # Check if extrinsic reward is non-zero
    if [ -n "$extrinsic_reward" ]; then
        # Use awk to check if the value is non-zero (handles floating point)
        is_nonzero=$(awk -v val="$extrinsic_reward" 'BEGIN { print (val != 0) ? 1 : 0 }')
        
        if [ "$is_nonzero" -eq 1 ]; then
            echo "Frame $frame_idx has $extrinsic_reward rwd_e"
        fi
    fi
done

