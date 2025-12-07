#!/bin/bash
# Sophie Serverless Worker - Startup Script
set -e

echo "============================================"
echo "Sophie Serverless Worker - Starting..."
echo "============================================"

# Download model if not present
/workspace/scripts/download-model.sh

# Validate model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "[ERROR] Model not found at $MODEL_PATH"
    exit 1
fi

echo "[start] Model: $MODEL_PATH"
echo "[start] GPU Layers: $N_GPU_LAYERS"
echo "[start] Context Size: $CTX_SIZE"
echo "[start] Port: $PORT"

# Build llama-server arguments
ARGS=(
    "--model" "$MODEL_PATH"
    "--host" "$HOST"
    "--port" "$PORT"
    "--n-gpu-layers" "$N_GPU_LAYERS"
    "--ctx-size" "$CTX_SIZE"
    "--batch-size" "$BATCH_SIZE"
    "--parallel" "$N_PARALLEL"
)

# Enable embeddings if requested
if [ "$EMBEDDING" = "true" ]; then
    ARGS+=("--embedding")
fi

echo "[start] Command: llama-server ${ARGS[*]}"
echo "============================================"

# Start llama-server
exec llama-server "${ARGS[@]}"
