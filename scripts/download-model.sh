#!/bin/bash
# Download model from HuggingFace if not present
set -e

MODEL_PATH="${MODEL_PATH:-/workspace/models/buchhaltgenie-universal-v5-Q4_K_M.gguf}"
MODEL_URL="${MODEL_URL:-https://huggingface.co/kanevry/buchhaltgenie-universal-v5-gguf/resolve/main/buchhaltgenie-universal-v5-Q4_K_M.gguf}"

if [ -f "$MODEL_PATH" ]; then
    echo "[download-model] Model already exists at $MODEL_PATH"
    exit 0
fi

echo "[download-model] Downloading model from $MODEL_URL..."
mkdir -p "$(dirname "$MODEL_PATH")"

curl -L --progress-bar -o "$MODEL_PATH" "$MODEL_URL"

if [ -f "$MODEL_PATH" ]; then
    MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    echo "[download-model] Model downloaded successfully: $MODEL_PATH ($MODEL_SIZE)"
else
    echo "[download-model] ERROR: Failed to download model!"
    exit 1
fi
