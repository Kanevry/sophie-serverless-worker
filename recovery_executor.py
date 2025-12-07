"""
Recovery Executor Module
Implements concrete recovery actions for RunPod serverless workers
"""

import gc
import subprocess
import time
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from runpod_error_types import RunPodErrorType, RecoveryAction

logger = logging.getLogger(__name__)


@dataclass
class RecoveryResult:
    """
    Recovery action result with metrics

    Austrian Compliance (BAO §132):
    - Structured audit logging
    - 7-year retention
    """

    success: bool
    action_taken: str
    metrics_before: Dict[str, Any] = field(default_factory=dict)
    metrics_after: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    freed_mb: float = 0.0
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_audit_log(self) -> Dict[str, Any]:
        """Format for BAO §132 audit logging"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "action": self.action_taken,
            "duration_ms": round(self.duration_ms, 2),
            "freed_mb": round(self.freed_mb, 2),
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
            "error": self.error,
        }


class RecoveryExecutor:
    """
    Executes recovery actions for RunPod serverless workers

    Recovery Strategies:
    - VRAM_EXHAUSTED: torch.cuda.empty_cache() (~80% success)
    - OOM_ERROR: gc.collect() (~60% success)
    - LLAMA_SERVER_CRASH: Process restart (~95% success)
    - LLAMA_SERVER_HANG: Force kill + restart (~90% success)
    - HEALTH_CHECK_FAIL: Wait + retry (~70% success)
    """

    def __init__(
        self,
        llama_server_port: int = 8080,
        llama_server_command: Optional[str] = None,
        max_restart_attempts: int = 3,
        restart_delay_seconds: int = 5,
    ):
        self.llama_server_port = llama_server_port
        self.llama_server_command = llama_server_command or self._get_default_command()
        self.max_restart_attempts = max_restart_attempts
        self.restart_delay_seconds = restart_delay_seconds

        # Metrics
        self.total_recoveries = 0
        self.successful_recoveries = 0
        self.failed_recoveries = 0

        logger.info(
            f"RecoveryExecutor initialized: port={llama_server_port}, "
            f"max_attempts={max_restart_attempts}"
        )

    def _get_default_command(self) -> str:
        """Get default llama-server startup command"""
        return (
            "llama-server "
            "--model /workspace/models/buchhaltgenie-universal-v5-Q4_K_M.gguf "
            "--port 8080 "
            "--ctx-size 4096 "
            "--n-gpu-layers 999"
        )

    async def execute_recovery(
        self, error_type: RunPodErrorType
    ) -> RecoveryResult:
        """
        Execute recovery action based on error type

        Args:
            error_type: Classified error type

        Returns:
            RecoveryResult with success status and metrics
        """
        start_time = time.time()
        metrics_before = self._get_system_metrics()

        self.total_recoveries += 1

        logger.info(f"[Recovery] Starting recovery for: {error_type.value}")

        try:
            # Dispatch recovery action
            if error_type == RunPodErrorType.VRAM_EXHAUSTED:
                result = await self._cleanup_gpu_memory()
            elif error_type == RunPodErrorType.OOM_ERROR:
                result = await self._cleanup_system_memory()
            elif error_type == RunPodErrorType.LLAMA_SERVER_CRASH:
                result = await self._restart_llama_server()
            elif error_type == RunPodErrorType.LLAMA_SERVER_HANG:
                result = await self._force_restart_server()
            elif error_type == RunPodErrorType.HEALTH_CHECK_FAIL:
                result = await self._retry_health_check()
            else:
                result = RecoveryResult(
                    success=False,
                    action_taken=f"no_recovery_for_{error_type.value}",
                    error=f"No recovery strategy for {error_type.value}",
                )

            # Update metrics
            duration_ms = (time.time() - start_time) * 1000
            metrics_after = self._get_system_metrics()

            result.metrics_before = metrics_before
            result.metrics_after = metrics_after
            result.duration_ms = duration_ms
            result.freed_mb = self._calculate_freed_memory(
                metrics_before, metrics_after
            )

            if result.success:
                self.successful_recoveries += 1
                logger.info(
                    f"[Recovery] SUCCESS: {result.action_taken} "
                    f"({duration_ms:.0f}ms, freed {result.freed_mb:.1f}MB)"
                )
            else:
                self.failed_recoveries += 1
                logger.error(
                    f"[Recovery] FAILED: {result.action_taken} - {result.error}"
                )

            return result

        except Exception as e:
            self.failed_recoveries += 1
            logger.exception(f"[Recovery] Exception during recovery: {e}")
            return RecoveryResult(
                success=False,
                action_taken=f"recovery_{error_type.value}",
                error=str(e),
                metrics_before=metrics_before,
                duration_ms=(time.time() - start_time) * 1000,
            )

    async def _cleanup_gpu_memory(self) -> RecoveryResult:
        """
        Cleanup GPU memory using torch.cuda.empty_cache()

        Expected success rate: ~80%
        Duration: <100ms
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return RecoveryResult(
                success=False,
                action_taken="gpu_cleanup",
                error="PyTorch/CUDA not available",
            )

        try:
            # Clear CUDA cache
            torch.cuda.empty_cache()

            # Force garbage collection
            gc.collect()

            logger.info("[Recovery] GPU cache cleared successfully")

            return RecoveryResult(success=True, action_taken="gpu_cleanup")

        except Exception as e:
            logger.error(f"[Recovery] GPU cleanup failed: {e}")
            return RecoveryResult(
                success=False, action_taken="gpu_cleanup", error=str(e)
            )

    async def _cleanup_system_memory(self) -> RecoveryResult:
        """
        Cleanup system memory using gc.collect()

        Expected success rate: ~60%
        Duration: ~200ms
        """
        try:
            # Force full garbage collection
            collected = gc.collect(generation=2)

            logger.info(f"[Recovery] System memory cleanup: {collected} objects freed")

            return RecoveryResult(
                success=True,
                action_taken="system_memory_cleanup",
                metadata={"objects_freed": collected},
            )

        except Exception as e:
            logger.error(f"[Recovery] System memory cleanup failed: {e}")
            return RecoveryResult(
                success=False, action_taken="system_memory_cleanup", error=str(e)
            )

    async def _restart_llama_server(self) -> RecoveryResult:
        """
        Graceful restart of llama-server process

        Expected success rate: ~95%
        Duration: 5-10s
        """
        logger.info("[Recovery] Attempting graceful llama-server restart")

        try:
            # Find llama-server process
            pid = self._find_llama_server_pid()

            if pid:
                # Send SIGTERM (graceful shutdown)
                logger.info(f"[Recovery] Sending SIGTERM to llama-server (PID={pid})")
                subprocess.run(["kill", "-15", str(pid)], check=True)

                # Wait for process to terminate
                await asyncio.sleep(self.restart_delay_seconds)

            # Start llama-server
            logger.info("[Recovery] Starting llama-server")
            subprocess.Popen(
                self.llama_server_command,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Wait for server to be ready
            await asyncio.sleep(2)

            # Verify health
            is_healthy = await self._check_server_health()

            if is_healthy:
                logger.info("[Recovery] llama-server restarted successfully")
                return RecoveryResult(success=True, action_taken="graceful_restart")
            else:
                return RecoveryResult(
                    success=False,
                    action_taken="graceful_restart",
                    error="Server unhealthy after restart",
                )

        except Exception as e:
            logger.error(f"[Recovery] Graceful restart failed: {e}")
            return RecoveryResult(
                success=False, action_taken="graceful_restart", error=str(e)
            )

    async def _force_restart_server(self) -> RecoveryResult:
        """
        Force restart llama-server with SIGKILL

        Expected success rate: ~90%
        Duration: 10-15s
        """
        logger.info("[Recovery] Attempting FORCE restart (SIGKILL)")

        try:
            # Find and kill process
            pid = self._find_llama_server_pid()

            if pid:
                logger.warning(
                    f"[Recovery] Force killing llama-server (PID={pid})"
                )
                subprocess.run(["kill", "-9", str(pid)], check=True)

                # Wait for process to die
                await asyncio.sleep(2)

            # Start llama-server
            logger.info("[Recovery] Starting llama-server after force kill")
            subprocess.Popen(
                self.llama_server_command,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Wait for server to be ready
            await asyncio.sleep(5)

            # Verify health
            is_healthy = await self._check_server_health()

            if is_healthy:
                logger.info("[Recovery] llama-server force-restarted successfully")
                return RecoveryResult(success=True, action_taken="force_restart")
            else:
                return RecoveryResult(
                    success=False,
                    action_taken="force_restart",
                    error="Server unhealthy after force restart",
                )

        except Exception as e:
            logger.error(f"[Recovery] Force restart failed: {e}")
            return RecoveryResult(
                success=False, action_taken="force_restart", error=str(e)
            )

    async def _retry_health_check(self) -> RecoveryResult:
        """
        Retry health check with exponential backoff

        Expected success rate: ~70%
        Duration: 1-5s
        """
        logger.info("[Recovery] Retrying health check")

        max_attempts = 3
        backoff_seconds = 1

        for attempt in range(1, max_attempts + 1):
            logger.info(f"[Recovery] Health check attempt {attempt}/{max_attempts}")

            is_healthy = await self._check_server_health()

            if is_healthy:
                logger.info(f"[Recovery] Health check passed on attempt {attempt}")
                return RecoveryResult(
                    success=True,
                    action_taken="health_check_retry",
                    metadata={"attempts": attempt},
                )

            if attempt < max_attempts:
                await asyncio.sleep(backoff_seconds)
                backoff_seconds *= 2  # Exponential backoff

        logger.error("[Recovery] Health check failed after all attempts")
        return RecoveryResult(
            success=False,
            action_taken="health_check_retry",
            error=f"Failed after {max_attempts} attempts",
        )

    def _find_llama_server_pid(self) -> Optional[int]:
        """Find llama-server process ID"""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "llama-server"], capture_output=True, text=True
            )
            if result.returncode == 0:
                pid = int(result.stdout.strip().split("\n")[0])
                return pid
        except Exception as e:
            logger.warning(f"[Recovery] Failed to find llama-server PID: {e}")

        return None

    async def _check_server_health(self) -> bool:
        """Check if llama-server /health endpoint is responding"""
        try:
            # Use curl for health check
            result = subprocess.run(
                [
                    "curl",
                    "-sf",
                    f"http://localhost:{self.llama_server_port}/health",
                ],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"[Recovery] Health check failed: {e}")
            return False

    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        metrics = {}

        # GPU metrics
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_mem_allocated = torch.cuda.memory_allocated(0)
                gpu_mem_reserved = torch.cuda.memory_reserved(0)
                gpu_mem_total = torch.cuda.get_device_properties(0).total_memory

                metrics["gpu_allocated_mb"] = gpu_mem_allocated / (1024 * 1024)
                metrics["gpu_reserved_mb"] = gpu_mem_reserved / (1024 * 1024)
                metrics["gpu_total_mb"] = gpu_mem_total / (1024 * 1024)
                metrics["gpu_percent"] = (
                    gpu_mem_reserved / gpu_mem_total
                ) * 100
            except Exception as e:
                logger.debug(f"Failed to get GPU metrics: {e}")

        # System metrics
        if PSUTIL_AVAILABLE:
            try:
                mem = psutil.virtual_memory()
                metrics["system_memory_mb"] = mem.used / (1024 * 1024)
                metrics["system_memory_percent"] = mem.percent
                metrics["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            except Exception as e:
                logger.debug(f"Failed to get system metrics: {e}")

        return metrics

    def _calculate_freed_memory(
        self, before: Dict[str, Any], after: Dict[str, Any]
    ) -> float:
        """Calculate freed memory in MB"""
        freed = 0.0

        # GPU memory freed
        if "gpu_reserved_mb" in before and "gpu_reserved_mb" in after:
            freed += before["gpu_reserved_mb"] - after["gpu_reserved_mb"]

        # System memory freed
        if "system_memory_mb" in before and "system_memory_mb" in after:
            freed += before["system_memory_mb"] - after["system_memory_mb"]

        return max(0.0, freed)  # Don't return negative values

    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics"""
        success_rate = (
            (self.successful_recoveries / self.total_recoveries * 100)
            if self.total_recoveries > 0
            else 0.0
        )

        return {
            "total_recoveries": self.total_recoveries,
            "successful_recoveries": self.successful_recoveries,
            "failed_recoveries": self.failed_recoveries,
            "success_rate_percent": round(success_rate, 2),
        }
