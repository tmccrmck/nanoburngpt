#!/bin/bash
set -e

# Helper script for running NanoBurnGPT on cloud GPU instances (RunPod/Lambda)

echo "--- NanoBurnGPT Cloud Helper ---"
echo "Targeting CUDA backend..."

# Check if a command was passed, otherwise default to training
if [ $# -gt 0 ]; then
    echo "Executing custom command: cargo run --release --features cuda -- $@"
    cargo run --release --features cuda -- "$@"
else
    echo "No command provided. Starting default 'gpt2-small' training on WikiText-103..."
    echo "Change this by passing arguments to 'docker run' or the Pod command."
    
    # Standard high-performance training command
    cargo run --release --features cuda -- train \
        --model gpt2-small \
        --dataset wikitext103 \
        --batch-size 32 \
        --num-epochs 10 \
        --num-workers 4
fi
