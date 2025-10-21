#!/bin/bash

# Default directory
DEFAULT_DIR="/home/minjune/doom/debug/20251018_143410"
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

# Function to process a single iter directory
process_iter() {
    local iter_dir="$1"
    local iter_name=$(basename "$iter_dir")
    
    # Use grep with -l to only list matching files, then extract frame numbers
    # This is much faster than reading each file individually
    local matches=$(grep -l "Extrinsic Reward: 10\." "$iter_dir"/frame_*.txt 2>/dev/null)
    
    if [ -n "$matches" ]; then
        echo "$iter_name:"
        echo "$matches" | while read -r file; do
            local filename=$(basename "$file")
            local frame_idx=$(echo "$filename" | sed 's/frame_\([0-9]*\)_.*/\1/')
            echo "  Frame $frame_idx has 100.0 rwd_e"
        done
    fi
}

export -f process_iter

# Get all iter directories
iter_dirs=("$DIR"/iter_*)

# Check if any directories exist
if [ ! -d "${iter_dirs[0]}" ]; then
    echo "No iter_* directories found in $DIR"
    exit 0
fi

# Process directories in parallel using all available cores
# Use xargs for parallel processing - much faster than sequential
printf '%s\n' "${iter_dirs[@]}" | xargs -P "$(nproc)" -I {} bash -c 'process_iter "$@"' _ {}