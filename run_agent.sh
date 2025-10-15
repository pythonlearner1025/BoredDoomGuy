#!/bin/bash
# Helper script to run the trained Arnold agent on Doom1 E1M1

# Find the best model checkpoint
MODEL_DIR="/home/minjune/doom/Arnold/dumped/doom1_fixed/hi36wgdr8k"
MODEL_PATH="$MODEL_DIR/best-260000.pth"
PARAMS_PATH="$MODEL_DIR/params.pkl"

# Check if files exist
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Looking for available models..."
    ls -lh $MODEL_DIR/*.pth
    exit 1
fi

if [ ! -f "$PARAMS_PATH" ]; then
    echo "Error: params.pkl not found at $PARAMS_PATH"
    exit 1
fi

echo "Using model: $MODEL_PATH"
echo "Using params: $PARAMS_PATH"
echo ""

# Run the agent
python3 /home/minjune/doom/agent.py \
    --model "$MODEL_PATH" \
    --params "$PARAMS_PATH" \
    --gpu-id 0 \
    --max-frames 10000 \
    --tick-delay 0.028 \
    --num-episodes 3 \
    --save-frames \
    "$@"
