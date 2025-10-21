#!/bin/bash

# Default directory - using current default structure
DEFAULT_DIR="/root/BoredDoomGuy/debug/20251019_232841"
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

# Function to process a single iter directory and return sum
# Output format: "iter_name total_reward"
process_iter() {
    local iter_dir="$1"
    local iter_name=$(basename "$iter_dir")
    
    # Use grep to extract all extrinsic rewards efficiently
    # The -h flag suppresses filenames, -o outputs only matched parts
    local sum=$(grep -h "Extrinsic Reward: " "$iter_dir"/frame_*.txt 2>/dev/null | \
                sed 's/.*Extrinsic Reward: //' | \
                awk '{sum += $1} END {printf "%.6f", sum}')
    
    # If no files found or sum is empty, set to 0
    if [ -z "$sum" ]; then
        sum="0.000000"
    fi
    
    # Output: iter_name sum
    echo "$iter_name $sum"
}

export -f process_iter

# Get all iter directories
iter_dirs=("$DIR"/iter_*)

# Check if any directories exist
if [ ! -d "${iter_dirs[0]}" ]; then
    echo "No iter_* directories found in $DIR"
    exit 0
fi

echo "Processing ${#iter_dirs[@]} iter directories in parallel..."
echo ""

# Process directories in parallel and collect results
# Use xargs for parallel processing - much faster than sequential
results=$(printf '%s\n' "${iter_dirs[@]}" | xargs -P "$(nproc)" -I {} bash -c 'process_iter "$@"' _ {})

# Find the maximum reward and corresponding iter
max_iter=""
max_reward=0
total_iters=0

while read -r iter_name reward; do
    ((total_iters++))
    # Use awk for floating point comparison
    is_greater=$(awk -v r="$reward" -v m="$max_reward" 'BEGIN {print (r > m) ? 1 : 0}')
    if [ "$is_greater" -eq 1 ]; then
        max_reward=$reward
        max_iter=$iter_name
    fi
done <<< "$results"

# Display results
echo "Analyzed $total_iters iterations"
echo ""
echo "Maximum extrinsic reward sum: $max_reward"
echo "Found in: $max_iter"
echo ""
echo "Full path: $DIR/$max_iter"

# Optional: Show top 5 if there are many iters
if [ "$total_iters" -gt 5 ]; then
    echo ""
    echo "Top 5 iterations by total extrinsic reward:"
    echo "$results" | sort -k2 -rn | head -5 | awk '{printf "  %s: %.6f\n", $1, $2}'
fi

