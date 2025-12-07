# RunPod Serverless Worker with Self-Healing
# Base: PyTorch 2.4.0 + CUDA 12.4.1 + Ubuntu 22.04
# Session 749: Self-healing recovery system

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Metadata
LABEL maintainer="BuchhaltGenie <office@buchhaltgenie.at>"
LABEL description="Sophie AI Serverless Worker with Self-Healing Recovery"
LABEL version="1.0.0"

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    vim \
    htop \
    procps \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for self-healing
# Note: torch is already included in base image
RUN pip install --no-cache-dir \
    psutil>=5.9.0 \
    aiohttp>=3.9.0 \
    sentry-sdk>=2.0.0

# Create log directory
RUN mkdir -p /workspace/logs

# Copy self-healing modules (files are in repo root)
COPY runpod_error_types.py /workspace/runpod-serverless/
COPY recovery_executor.py /workspace/runpod-serverless/
COPY circuit_breaker.py /workspace/runpod-serverless/
COPY health_monitor.py /workspace/runpod-serverless/
COPY monitor_daemon.py /workspace/runpod-serverless/
COPY runpod_health_service.py /workspace/runpod-serverless/

# Copy startup script
COPY start-worker.sh /workspace/runpod-serverless/

# Make scripts executable
RUN chmod +x /workspace/runpod-serverless/start-worker.sh
RUN chmod +x /workspace/runpod-serverless/monitor_daemon.py
RUN chmod +x /workspace/runpod-serverless/runpod_health_service.py

# Install llama-server (llama.cpp)
# Session 751: Fixed CUDA linking for builds WITHOUT GPU
# Problem: libcuda.so.1 (driver) not available during docker build
# Solution: Use CUDA stubs + allow-shlib-undefined (symbols resolved at runtime on RunPod)
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH

RUN cd /tmp && \
    git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    git checkout master && \
    cmake -B build \
        -DGGML_CUDA=ON \
        -DLLAMA_CURL=OFF \
        -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,--allow-shlib-undefined" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,--allow-shlib-undefined" && \
    cmake --build build --config Release --target llama-server && \
    cp build/bin/llama-server /usr/local/bin/ && \
    cd / && rm -rf /tmp/llama.cpp

# Model directory (mounted at runtime or pre-baked)
RUN mkdir -p /workspace/models

# Environment variables (can be overridden)
ENV MODEL_PATH=/workspace/models/buchhaltgenie-universal-v5-Q4_K_M.gguf
ENV N_GPU_LAYERS=999
ENV CTX_SIZE=4096
ENV BATCH_SIZE=512
ENV N_PARALLEL=4
ENV HEALTH_CHECK_INTERVAL=30
ENV RECOVERY_ENABLED=true
ENV LLAMA_SERVER_PORT=8080
ENV PORT_HEALTH=7860

# Expose ports
EXPOSE 8080
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${LLAMA_SERVER_PORT}/health || exit 1

# Entry point
CMD ["/workspace/runpod-serverless/start-worker.sh"]
