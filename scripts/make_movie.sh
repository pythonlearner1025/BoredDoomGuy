#!/bin/bash

# Script to convert all iter_00xx frame directories to MP4 videos
# Usage: ./make_movie.sh           - Convert all iterations
#        ./make_movie.sh --last    - Convert only the most recent iteration

DEBUG_DIR="/home/minjune/doom/debug"

# Check if debug directory exists
if [ ! -d "$DEBUG_DIR" ]; then
    echo "Error: Debug directory not found at $DEBUG_DIR"
    exit 1
fi

# Determine which directories to process
if [ "$1" == "--last" ]; then
    # Find the most recent iter_00xx directory by modification time
    iter_dirs=($(ls -td "$DEBUG_DIR"/iter_[0-9][0-9][0-9][0-9] 2>/dev/null | head -1))
    if [ ${#iter_dirs[@]} -eq 0 ]; then
        echo "No iteration directories found"
        exit 1
    fi
    echo "Processing only the most recent iteration..."
else
    # Process all iter_00xx directories
    iter_dirs=("$DEBUG_DIR"/iter_[0-9][0-9][0-9][0-9])
fi

# Process each directory
for iter_dir in "${iter_dirs[@]}"; do
    if [ -d "$iter_dir" ]; then
        # Extract the iteration number (e.g., iter_0001)
        iter_name=$(basename "$iter_dir")

        # Check if there are any current.png frames
        frame_count=$(ls "$iter_dir"/frame_*_current.png 2>/dev/null | wc -l)

        if [ "$frame_count" -gt 0 ]; then
            echo "Processing $iter_name ($frame_count frames)..."

            output_file="$DEBUG_DIR/${iter_name}_run.mp4"

            # Create MP4 using ffmpeg
            ffmpeg -y \
                -framerate 35 \
                -pattern_type glob \
                -i "$iter_dir/frame_*_current.png" \
                -c:v libx264 \
                -pix_fmt yuv420p \
                -crf 23 \
                "$output_file" \
                -loglevel warning -stats

            if [ $? -eq 0 ]; then
                echo "✓ Created $output_file"
            else
                echo "✗ Failed to create $output_file"
            fi
        else
            echo "Skipping $iter_name (no frames found)"
        fi
    fi
done

echo ""
echo "Done! All videos saved in $DEBUG_DIR"
