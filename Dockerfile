# Sophie Serverless Worker - llama.cpp with GGUF Model
# RunPod Serverless Compatible
#
# Build: docker build -t buchhaltgenie/sophie-serverless:latest .
# Test:  docker run --gpus all -p 8080:8080 buchhaltgenie/sophie-serverless:latest

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Clone and build llama.cpp with CUDA support
WORKDIR /build
RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;89" && \
    cmake --build build --config Release -j$(nproc) && \
    cp build/bin/llama-server /usr/local/bin/

# ============================================
# Runtime Image
# ============================================
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy llama-server binary
COPY --from=builder /usr/local/bin/llama-server /usr/local/bin/llama-server

# Create model directory
WORKDIR /workspace
RUN mkdir -p /workspace/models

# Environment variables (can be overridden)
ENV MODEL_PATH="/workspace/models/buchhaltgenie-universal-v5-Q4_K_M.gguf"
ENV MODEL_URL="https://huggingface.co/kanevry/buchhaltgenie-universal-v5-gguf/resolve/main/buchhaltgenie-universal-v5-Q4_K_M.gguf"
ENV PORT=8080
ENV HOST=0.0.0.0
ENV N_GPU_LAYERS=99
ENV CTX_SIZE=4096
ENV BATCH_SIZE=512
ENV N_PARALLEL=4
ENV EMBEDDING=false

# Download model script
COPY scripts/download-model.sh /workspace/scripts/download-model.sh
RUN chmod +x /workspace/scripts/download-model.sh

# Startup script
COPY scripts/start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE ${PORT}

# Start llama-server
CMD ["/workspace/start.sh"]
