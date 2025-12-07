#!/usr/bin/env python3
"""
RunPod Health Service - Compliant /ping endpoint for load balancing
Runs alongside llama-server and handles RunPod health checks

Port: PORT_HEALTH (default 7860)
Endpoints:
  - GET /ping → 200 (healthy) | 204 (initializing) | 503 (unhealthy)
  - GET /health → detailed health status (for debugging)

Session: 750 S2 - Production-Ready RunPod Integration
"""

import asyncio
import os
import sys
import time
from enum import Enum
from typing import Dict, Optional

import aiohttp
from aiohttp import web

# Configuration
PORT_HEALTH = int(os.getenv("PORT_HEALTH", "7860"))
LLAMA_SERVER_PORT = int(os.getenv("LLAMA_SERVER_PORT", "8080"))
MODEL_URL = os.getenv("MODEL_URL", "")
MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/models/buchhaltgenie-universal-v5-Q4_K_M.gguf")
STARTUP_TIMEOUT_SECONDS = int(os.getenv("STARTUP_TIMEOUT_SECONDS", "300"))  # 5 min


class HealthState(Enum):
    """RunPod health states"""
    INITIALIZING = 204  # Worker starting up
    HEALTHY = 200       # Ready to serve requests
    UNHEALTHY = 503     # Error state


class RunPodHealthService:
    """RunPod-compliant health check service"""

    def __init__(self):
        self.state = HealthState.INITIALIZING
        self.startup_time = time.time()
        self.model_downloaded = False
        self.llama_server_ready = False
        self.error_message: Optional[str] = None

    async def download_model(self) -> bool:
        """Download model from MODEL_URL if specified"""
        if not MODEL_URL:
            print(f"[Health] No MODEL_URL specified, expecting pre-baked model at {MODEL_PATH}")
            self.model_downloaded = True
            return True

        print(f"[Health] Downloading model from {MODEL_URL}")
        print(f"[Health] Target path: {MODEL_PATH}")

        # Create models directory
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

        try:
            # Download with progress (streaming)
            async with aiohttp.ClientSession() as session:
                async with session.get(MODEL_URL) as response:
                    if response.status != 200:
                        self.error_message = f"Model download failed: HTTP {response.status}"
                        print(f"[Health] ERROR: {self.error_message}")
                        return False

                    total_size = response.headers.get("Content-Length")
                    downloaded = 0

                    with open(MODEL_PATH, "wb") as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                            downloaded += len(chunk)

                            # Log progress every 100MB
                            if total_size and downloaded % (100 * 1024 * 1024) < 8192:
                                progress = (downloaded / int(total_size)) * 100
                                print(f"[Health] Download progress: {progress:.1f}%")

            print(f"[Health] Model downloaded successfully ({downloaded / (1024**3):.2f} GB)")
            self.model_downloaded = True
            return True

        except Exception as e:
            self.error_message = f"Model download exception: {str(e)}"
            print(f"[Health] ERROR: {self.error_message}")
            return False

    async def check_llama_server(self) -> bool:
        """Check if llama-server is healthy"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://localhost:{LLAMA_SERVER_PORT}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        self.llama_server_ready = True
                        return True
                    else:
                        return False
        except Exception as e:
            # llama-server not yet ready (expected during startup)
            return False

    async def initialize(self):
        """Initialize worker (download model, wait for llama-server)"""
        print(f"[Health] Starting initialization...")
        print(f"[Health] Health service on port {PORT_HEALTH}")
        print(f"[Health] llama-server on port {LLAMA_SERVER_PORT}")

        # Step 1: Download model (if MODEL_URL specified)
        if not await self.download_model():
            self.state = HealthState.UNHEALTHY
            return

        # Step 2: Wait for llama-server to be ready
        print(f"[Health] Waiting for llama-server to start...")
        start_time = time.time()

        while (time.time() - start_time) < STARTUP_TIMEOUT_SECONDS:
            if await self.check_llama_server():
                print(f"[Health] llama-server is ready!")
                self.state = HealthState.HEALTHY
                startup_duration = time.time() - self.startup_time
                print(f"[Health] Initialization complete in {startup_duration:.1f}s")
                return

            await asyncio.sleep(2)

        # Timeout
        self.error_message = f"llama-server failed to start within {STARTUP_TIMEOUT_SECONDS}s"
        print(f"[Health] ERROR: {self.error_message}")
        self.state = HealthState.UNHEALTHY

    async def ping_handler(self, request: web.Request) -> web.Response:
        """
        RunPod /ping endpoint (load balancer health check)

        Returns:
          - 200: Worker is healthy and ready
          - 204: Worker is initializing
          - 503: Worker is unhealthy
        """
        return web.Response(status=self.state.value)

    async def health_handler(self, request: web.Request) -> web.Response:
        """
        Detailed health status (for debugging)

        Returns JSON with:
          - state: INITIALIZING | HEALTHY | UNHEALTHY
          - uptime: seconds since startup
          - model_downloaded: bool
          - llama_server_ready: bool
          - error_message: str | null
        """
        uptime = time.time() - self.startup_time

        health_data = {
            "state": self.state.name,
            "uptime_seconds": round(uptime, 2),
            "model_downloaded": self.model_downloaded,
            "llama_server_ready": self.llama_server_ready,
            "error_message": self.error_message,
            "config": {
                "port_health": PORT_HEALTH,
                "llama_server_port": LLAMA_SERVER_PORT,
                "model_url": MODEL_URL if MODEL_URL else "pre-baked",
                "model_path": MODEL_PATH
            }
        }

        return web.json_response(health_data)

    async def periodic_health_check(self):
        """Continuously monitor llama-server health"""
        await asyncio.sleep(10)  # Initial delay

        while True:
            if self.state == HealthState.HEALTHY:
                # Verify llama-server is still healthy
                if not await self.check_llama_server():
                    print("[Health] WARNING: llama-server became unhealthy")
                    self.state = HealthState.UNHEALTHY
                    self.error_message = "llama-server health check failed"

            await asyncio.sleep(30)  # Check every 30s

    def start(self):
        """Start health service HTTP server"""
        app = web.Application()
        app.router.add_get("/ping", self.ping_handler)
        app.router.add_get("/health", self.health_handler)

        # Run initialization in background
        asyncio.create_task(self.initialize())

        # Start periodic health checks
        asyncio.create_task(self.periodic_health_check())

        print(f"[Health] RunPod Health Service starting on port {PORT_HEALTH}")
        print(f"[Health] Endpoints:")
        print(f"[Health]   - GET http://0.0.0.0:{PORT_HEALTH}/ping (RunPod load balancer)")
        print(f"[Health]   - GET http://0.0.0.0:{PORT_HEALTH}/health (detailed status)")

        web.run_app(app, host="0.0.0.0", port=PORT_HEALTH, print=None)


if __name__ == "__main__":
    service = RunPodHealthService()
    try:
        service.start()
    except KeyboardInterrupt:
        print("\n[Health] Shutting down health service")
        sys.exit(0)
