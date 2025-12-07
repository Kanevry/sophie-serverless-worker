# Sophie Serverless Worker

> RunPod Serverless endpoint for BuchhaltGenie Sophie AI (Austrian Tax Assistant)

## Overview

This container provides an OpenAI-compatible API for Sophie AI using llama.cpp with the
`buchhaltgenie-universal-v5-Q4_K_M.gguf` model (Qwen3-VL based, quantized to 4-bit).

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  RunPod Serverless                       │
│  ┌─────────────────────────────────────────────────┐   │
│  │ llama-server (OpenAI-compatible API)             │   │
│  │ Model: buchhaltgenie-universal-v5-Q4_K_M.gguf   │   │
│  │ VRAM: ~5GB | Speed: 131+ tokens/sec             │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  API: /v1/chat/completions (OpenAI-compatible)         │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### Local Testing (Docker)

```bash
# Build and run
docker-compose up --build

# Test the API
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "buchhaltgenie",
    "messages": [
      {"role": "system", "content": "Du bist Sophie, die KI-Buchhaltungsassistentin."},
      {"role": "user", "content": "Was ist UStG §11?"}
    ],
    "max_tokens": 500
  }'
```

### Deploy to RunPod Serverless

1. **Build and Push Image**
   ```bash
   docker build -t ghcr.io/buchhaltgenie/sophie-serverless:latest .
   docker push ghcr.io/buchhaltgenie/sophie-serverless:latest
   ```

2. **Create Serverless Endpoint**
   - Go to [RunPod Console](https://console.runpod.io/serverless)
   - Create new endpoint
   - Image: `ghcr.io/buchhaltgenie/sophie-serverless:latest`
   - GPU: RTX 4090 / L40S / A40 (24GB+ VRAM)
   - FlashBoot: Enabled
   - Active Workers: 0 (flex) or 1 (prod)

3. **Use the Endpoint**
   ```bash
   curl https://api.runpod.ai/v2/{ENDPOINT_ID}/openai/v1/chat/completions \
     -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
     -H "Content-Type: application/json" \
     -d '{...}'
   ```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/workspace/models/...gguf` | Path to GGUF model |
| `MODEL_URL` | HuggingFace URL | Download URL for model |
| `PORT` | 8080 | API server port |
| `N_GPU_LAYERS` | 99 | Layers to offload to GPU |
| `CTX_SIZE` | 4096 | Context window size |
| `BATCH_SIZE` | 512 | Batch size for inference |
| `N_PARALLEL` | 4 | Parallel request handling |

## API Endpoints

### Chat Completions
```
POST /v1/chat/completions
```

### Health Check
```
GET /health
```

### Model Info
```
GET /v1/models
```

## Performance

| GPU | Tokens/sec | VRAM Usage | Cold Start |
|-----|------------|------------|------------|
| RTX 4090 | 131-143 | ~5GB | ~30s |
| L40S | ~120 | ~5GB | ~35s |
| A40 | ~100 | ~5GB | ~40s |

## Model Details

- **Base**: Qwen3-VL (9B parameters)
- **Quantization**: Q4_K_M (4-bit)
- **Size**: ~5GB
- **Context**: 4096 tokens (expandable)
- **Training**: Austrian tax law (UStG, BAO, DSGVO)

## License

Proprietary - BuchhaltGenie GmbH
