#!/bin/bash
#
# RunPod Serverless Worker Startup Script
# Starts llama-server and health monitor sidecar
#
# Usage: ./start-worker.sh
#
# Environment Variables:
#   MODEL_PATH        - Path to GGUF model (default: /workspace/models/buchhaltgenie-universal-v5-Q4_K_M.gguf)
#   N_GPU_LAYERS      - Number of GPU layers (default: 999)
#   CTX_SIZE          - Context size (default: 4096)
#   BATCH_SIZE        - Batch size (default: 512)
#   N_PARALLEL        - Parallel requests (default: 4)
#   HEALTH_CHECK_INTERVAL - Health check interval in seconds (default: 30)
#   RECOVERY_ENABLED  - Enable recovery (default: true)
#   LLAMA_SERVER_PORT - llama-server port (default: 8080)

set -e  # Exit on error

# Configuration
MODEL_PATH="${MODEL_PATH:-/workspace/models/buchhaltgenie-universal-v5-Q4_K_M.gguf}"
MODEL_URL="${MODEL_URL:-}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
CTX_SIZE="${CTX_SIZE:-4096}"
BATCH_SIZE="${BATCH_SIZE:-512}"
N_PARALLEL="${N_PARALLEL:-4}"
HEALTH_CHECK_INTERVAL="${HEALTH_CHECK_INTERVAL:-30}"
RECOVERY_ENABLED="${RECOVERY_ENABLED:-true}"
LLAMA_SERVER_PORT="${LLAMA_SERVER_PORT:-8080}"
PORT_HEALTH="${PORT_HEALTH:-7860}"

# Logging
LOG_DIR="${LOG_DIR:-/workspace/logs}"
mkdir -p "$LOG_DIR"

LLAMA_LOG="$LOG_DIR/llama-server.log"
MONITOR_LOG="$LOG_DIR/health-monitor.log"
HEALTH_SERVICE_LOG="$LOG_DIR/runpod-health-service.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting RunPod Serverless Worker"
echo "  Model: $MODEL_PATH"
echo "  Model URL: ${MODEL_URL:-none (pre-baked)}"
echo "  GPU Layers: $N_GPU_LAYERS"
echo "  Context Size: $CTX_SIZE"
echo "  llama-server Port: $LLAMA_SERVER_PORT"
echo "  Health Service Port: $PORT_HEALTH"
echo "  Recovery: $RECOVERY_ENABLED"
echo ""

# Function to check if process is running
is_running() {
    local pid=$1
    kill -0 "$pid" 2>/dev/null
}

# Function to wait for server health
wait_for_health() {
    local max_attempts=30
    local attempt=0

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for llama-server to be healthy..."

    while [ $attempt -lt $max_attempts ]; do
        if curl -sf "http://localhost:$LLAMA_SERVER_PORT/health" >/dev/null 2>&1; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] llama-server is healthy!"
            return 0
        fi

        attempt=$((attempt + 1))
        echo "  Attempt $attempt/$max_attempts..."
        sleep 2
    done

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: llama-server failed to become healthy"
    return 1
}

# Cleanup function
cleanup() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Shutting down worker..."

    # Kill RunPod health service
    if [ ! -z "$HEALTH_SERVICE_PID" ] && is_running "$HEALTH_SERVICE_PID"; then
        echo "  Stopping RunPod health service (PID=$HEALTH_SERVICE_PID)"
        kill -15 "$HEALTH_SERVICE_PID" 2>/dev/null || true
    fi

    # Kill llama-server
    if [ ! -z "$LLAMA_PID" ] && is_running "$LLAMA_PID"; then
        echo "  Stopping llama-server (PID=$LLAMA_PID)"
        kill -15 "$LLAMA_PID" 2>/dev/null || true
        sleep 2
        kill -9 "$LLAMA_PID" 2>/dev/null || true
    fi

    # Kill health monitor
    if [ ! -z "$MONITOR_PID" ] && is_running "$MONITOR_PID"; then
        echo "  Stopping health monitor (PID=$MONITOR_PID)"
        kill -15 "$MONITOR_PID" 2>/dev/null || true
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Worker shutdown complete"
}

# Register cleanup on exit
trap cleanup EXIT INT TERM

# Start llama-server
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting llama-server..."

nohup llama-server \
    --model "$MODEL_PATH" \
    --port "$LLAMA_SERVER_PORT" \
    --ctx-size "$CTX_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --n-parallel "$N_PARALLEL" \
    --host 0.0.0.0 \
    --log-format text \
    > "$LLAMA_LOG" 2>&1 &

LLAMA_PID=$!

echo "  llama-server started (PID=$LLAMA_PID)"
echo "  Logs: $LLAMA_LOG"

# Wait for llama-server to be ready
if ! wait_for_health; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FATAL: llama-server startup failed"
    echo "  Check logs: $LLAMA_LOG"
    tail -n 50 "$LLAMA_LOG"
    exit 1
fi

# Start RunPod Health Service (for load balancer /ping checks)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting RunPod Health Service..."

nohup python3 /workspace/runpod-serverless/runpod_health_service.py \
    > "$HEALTH_SERVICE_LOG" 2>&1 &

HEALTH_SERVICE_PID=$!

echo "  Health service started (PID=$HEALTH_SERVICE_PID)"
echo "  Logs: $HEALTH_SERVICE_LOG"
echo "  Endpoint: http://0.0.0.0:$PORT_HEALTH/ping"

# Wait a moment for health service to start
sleep 2

# Start health monitor sidecar (if recovery enabled)
if [ "$RECOVERY_ENABLED" = "true" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting health monitor sidecar..."

    nohup python3 /workspace/runpod-serverless/monitor_daemon.py \
        --port "$LLAMA_SERVER_PORT" \
        --interval "$HEALTH_CHECK_INTERVAL" \
        > "$MONITOR_LOG" 2>&1 &

    MONITOR_PID=$!

    echo "  Health monitor started (PID=$MONITOR_PID)"
    echo "  Logs: $MONITOR_LOG"
    echo "  Interval: ${HEALTH_CHECK_INTERVAL}s"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Health monitor DISABLED (RECOVERY_ENABLED=false)"
fi

# Worker is ready
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ====================================="
echo "RunPod Serverless Worker is READY!"
echo "====================================="
echo "  llama-server: http://localhost:$LLAMA_SERVER_PORT"
echo "  Health: http://localhost:$LLAMA_SERVER_PORT/health"
echo "  Models: http://localhost:$LLAMA_SERVER_PORT/v1/models"
echo "  Chat: POST http://localhost:$LLAMA_SERVER_PORT/v1/chat/completions"
echo ""

# Keep script running (monitor processes)
while true; do
    # Check if llama-server is still running
    if ! is_running "$LLAMA_PID"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: llama-server died (PID=$LLAMA_PID)"
        echo "  Last 100 lines of logs:"
        tail -n 100 "$LLAMA_LOG"
        exit 1
    fi

    # Check if RunPod health service is still running
    if ! is_running "$HEALTH_SERVICE_PID"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: RunPod health service died (PID=$HEALTH_SERVICE_PID)"
        echo "  Attempting restart..."

        nohup python3 /workspace/runpod-serverless/runpod_health_service.py \
            > "$HEALTH_SERVICE_LOG" 2>&1 &

        HEALTH_SERVICE_PID=$!
        echo "  Health service restarted (PID=$HEALTH_SERVICE_PID)"
    fi

    # Check if health monitor is still running (if enabled)
    if [ "$RECOVERY_ENABLED" = "true" ] && ! is_running "$MONITOR_PID"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: Health monitor died (PID=$MONITOR_PID)"
        echo "  Attempting restart..."

        nohup python3 /workspace/runpod-serverless/monitor_daemon.py \
            --port "$LLAMA_SERVER_PORT" \
            --interval "$HEALTH_CHECK_INTERVAL" \
            > "$MONITOR_LOG" 2>&1 &

        MONITOR_PID=$!
        echo "  Health monitor restarted (PID=$MONITOR_PID)"
    fi

    sleep 10
done
