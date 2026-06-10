#!/bin/bash

# Target Docker image name
IMAGE_NAME="loop-learn"

# Check if local image exists. If not — trigger build sequence
if ! docker images --format "{{.Repository}}" | grep -q "^${IMAGE_NAME}$"; then
    echo "=== Docker image ${IMAGE_NAME} not found. Starting build sequence... ==="
    docker build -t "$IMAGE_NAME" .
    if [ $? -ne 0 ]; then
        echo "Error: Failed to build target Docker image."
        exit 1
    fi
fi

echo "=== Launching loop-learn engine inside isolated container... ==="

# Guarantee local presence of storage directory for state persistence
mkdir -p storage;

# Run container with hardware routing, active HF_TOKEN, and host-mapped cache paths
docker run -it --rm \
    --gpus all \
    -e HF_TOKEN \
    -v "$(pwd)/storage:/usr/src/loop-learn/storage" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    "$IMAGE_NAME":latest "$@"