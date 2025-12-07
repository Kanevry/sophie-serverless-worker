"""
Self-Healing Health Monitor for RunPod Serverless Worker
Monitors worker health and triggers recovery when needed
"""

import time
import asyncio
import psutil
import torch
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

from recovery_executor import RecoveryExecutor, RecoveryResult
from circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState
from runpod_error_types import RunPodErrorType

logger = logging.getLogger(__name__)


@dataclass
class HealthMetrics:
    """Health metrics for the worker"""

    timestamp: datetime = field(default_factory=datetime.now)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    error_count: int = 0
    success_count: int = 0
    avg_response_time_ms: float = 0.0
    last_error: Optional[str] = None

    @property
    def error_rate(self) -> float:
        """Calculate error rate (0.0 to 1.0)"""
        total = self.error_count + self.success_count
        if total == 0:
            return 0.0
        return self.error_count / total

    @property
    def gpu_memory_percent(self) -> float:
        """Calculate GPU memory usage percentage"""
        if self.gpu_memory_total_mb == 0:
            return 0.0
        return (self.gpu_memory_used_mb / self.gpu_memory_total_mb) * 100

    def is_healthy(self) -> bool:
        """Check if worker is healthy based on metrics"""
        checks = [
            self.memory_percent < 95.0,  # Memory not critical
            self.gpu_memory_percent < 95.0,  # GPU memory not critical
            self.error_rate < 0.3,  # Less than 30% error rate
        ]
        return all(checks)


class HealthMonitor:
    """Monitors worker health and triggers recovery actions"""

    def __init__(
        self,
        check_interval_seconds: int = 30,
        memory_threshold_percent: float = 90.0,
        gpu_memory_threshold_percent: float = 90.0,
        error_rate_threshold: float = 0.25,
        recovery_cooldown_seconds: int = 60,
        enable_recovery: bool = True,
    ):
        self.check_interval = check_interval_seconds
        self.memory_threshold = memory_threshold_percent
        self.gpu_memory_threshold = gpu_memory_threshold_percent
        self.error_rate_threshold = error_rate_threshold
        self.recovery_cooldown_seconds = recovery_cooldown_seconds
        self.enable_recovery = enable_recovery

        self.current_metrics = HealthMetrics()
        self.last_check_time = datetime.now()
        self.response_times: list[float] = []

        # Recovery components (NEW)
        self.recovery_executor = RecoveryExecutor() if enable_recovery else None
        self.circuit_breaker = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=5,
                success_threshold=2,
                timeout=30000,  # 30s
                volume_threshold=10,
                error_threshold_percentage=50,
            )
        ) if enable_recovery else None
        self.last_recovery_time: Optional[datetime] = None
        self.recovery_attempt_count = 0

        logger.info(
            f"Health Monitor initialized: "
            f"interval={check_interval_seconds}s, "
            f"memory_threshold={memory_threshold_percent}%, "
            f"gpu_threshold={gpu_memory_threshold_percent}%, "
            f"error_threshold={error_rate_threshold}, "
            f"recovery_enabled={enable_recovery}"
        )

    def record_success(self, response_time_ms: float):
        """Record a successful request"""
        self.current_metrics.success_count += 1
        self.response_times.append(response_time_ms)

        # Keep only last 100 response times
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]

        self._update_avg_response_time()

    def record_error(self, error_msg: str):
        """Record an error"""
        self.current_metrics.error_count += 1
        self.current_metrics.last_error = error_msg
        logger.warning(f"Error recorded: {error_msg}")

    def _update_avg_response_time(self):
        """Update average response time"""
        if self.response_times:
            self.current_metrics.avg_response_time_ms = sum(self.response_times) / len(self.response_times)

    def check_health(self) -> HealthMetrics:
        """Check current health status"""
        now = datetime.now()

        # Only update metrics if enough time has passed
        if (now - self.last_check_time).total_seconds() < self.check_interval:
            return self.current_metrics

        # Update system metrics
        self.current_metrics.timestamp = now
        self.current_metrics.cpu_percent = psutil.cpu_percent(interval=1)
        self.current_metrics.memory_percent = psutil.virtual_memory().percent

        # Update GPU metrics if available
        if torch.cuda.is_available():
            try:
                gpu_mem_allocated = torch.cuda.memory_allocated(0)
                gpu_mem_reserved = torch.cuda.memory_reserved(0)
                gpu_mem_total = torch.cuda.get_device_properties(0).total_memory

                self.current_metrics.gpu_memory_used_mb = gpu_mem_reserved / (1024 * 1024)
                self.current_metrics.gpu_memory_total_mb = gpu_mem_total / (1024 * 1024)
            except Exception as e:
                logger.error(f"Failed to get GPU metrics: {e}")

        self.last_check_time = now

        # Log health status
        is_healthy = self.current_metrics.is_healthy()
        log_level = logging.INFO if is_healthy else logging.WARNING
        logger.log(
            log_level,
            f"Health Check: "
            f"healthy={is_healthy}, "
            f"cpu={self.current_metrics.cpu_percent:.1f}%, "
            f"mem={self.current_metrics.memory_percent:.1f}%, "
            f"gpu_mem={self.current_metrics.gpu_memory_percent:.1f}%, "
            f"error_rate={self.current_metrics.error_rate:.2%}, "
            f"avg_response={self.current_metrics.avg_response_time_ms:.0f}ms"
        )

        return self.current_metrics

    def needs_recovery(self) -> tuple[bool, str]:
        """
        Check if worker needs recovery action

        Returns:
            (needs_recovery, reason)
        """
        metrics = self.check_health()

        # Check memory threshold
        if metrics.memory_percent > self.memory_threshold:
            return True, f"High memory usage: {metrics.memory_percent:.1f}%"

        # Check GPU memory threshold
        if metrics.gpu_memory_percent > self.gpu_memory_threshold:
            return True, f"High GPU memory usage: {metrics.gpu_memory_percent:.1f}%"

        # Check error rate
        if metrics.error_rate > self.error_rate_threshold:
            return True, f"High error rate: {metrics.error_rate:.2%}"

        return False, ""

    async def check_health_with_recovery(self) -> Tuple[bool, Optional[RecoveryResult]]:
        """
        Check health and auto-recover if needed

        Returns:
            (is_healthy, recovery_result)
        """
        if not self.enable_recovery:
            metrics = self.check_health()
            return (metrics.is_healthy(), None)

        metrics = self.check_health()
        needs_recovery, reason = self.needs_recovery()

        if not needs_recovery:
            return (True, None)

        # Check recovery cooldown
        if not self._can_attempt_recovery():
            cooldown_remaining = self._get_cooldown_remaining()
            logger.warning(
                f"[HealthMonitor] Recovery needed but in cooldown "
                f"({cooldown_remaining:.0f}s remaining)"
            )
            return (False, None)

        # Classify health issue and execute recovery
        error_type = self._classify_health_issue(metrics, reason)
        logger.info(
            f"[HealthMonitor] Executing recovery for: {error_type.value}"
        )

        recovery_result = await self.recovery_executor.execute_recovery(error_type)

        self.last_recovery_time = datetime.now()
        self.recovery_attempt_count += 1

        # Update circuit breaker
        if recovery_result.success:
            self.circuit_breaker._on_success()
            logger.info(
                f"[HealthMonitor] Recovery SUCCESS: {recovery_result.action_taken} "
                f"({recovery_result.duration_ms:.0f}ms)"
            )
        else:
            self.circuit_breaker._on_failure()
            logger.error(
                f"[HealthMonitor] Recovery FAILED: {recovery_result.action_taken} - "
                f"{recovery_result.error}"
            )

        return (recovery_result.success, recovery_result)

    async def run_monitoring_loop(self, interval_seconds: int = 30):
        """
        Continuous monitoring loop (run in background)

        Usage:
            monitor = HealthMonitor()
            asyncio.create_task(monitor.run_monitoring_loop())
        """
        logger.info(
            f"[HealthMonitor] Starting monitoring loop (interval={interval_seconds}s)"
        )

        while True:
            try:
                is_healthy, recovery_result = await self.check_health_with_recovery()

                # Export metrics
                await self._export_metrics(is_healthy, recovery_result)

                # Check circuit breaker state
                circuit_state = (
                    self.circuit_breaker.get_state()
                    if self.circuit_breaker
                    else None
                )

                if circuit_state == CircuitState.OPEN:
                    logger.critical(
                        "[HealthMonitor] Circuit breaker OPEN - escalating to RunPod Console"
                    )

            except Exception as e:
                logger.exception(f"[HealthMonitor] Error in monitoring loop: {e}")

            await asyncio.sleep(interval_seconds)

    def _can_attempt_recovery(self) -> bool:
        """Check if recovery can be attempted (cooldown check)"""
        if self.last_recovery_time is None:
            return True

        # Exponential backoff based on attempt count
        cooldown_multiplier = 2 ** min(self.recovery_attempt_count - 1, 3)
        cooldown_seconds = self.recovery_cooldown_seconds * cooldown_multiplier

        elapsed = (datetime.now() - self.last_recovery_time).total_seconds()
        return elapsed >= cooldown_seconds

    def _get_cooldown_remaining(self) -> float:
        """Get remaining cooldown time in seconds"""
        if self.last_recovery_time is None:
            return 0.0

        cooldown_multiplier = 2 ** min(self.recovery_attempt_count - 1, 3)
        cooldown_seconds = self.recovery_cooldown_seconds * cooldown_multiplier

        elapsed = (datetime.now() - self.last_recovery_time).total_seconds()
        return max(0.0, cooldown_seconds - elapsed)

    def _classify_health_issue(
        self, metrics: HealthMetrics, reason: str
    ) -> RunPodErrorType:
        """
        Classify health issue into error type

        Maps health metrics to RunPodErrorType
        """
        # GPU memory exhaustion
        if metrics.gpu_memory_percent > self.gpu_memory_threshold:
            return RunPodErrorType.VRAM_EXHAUSTED

        # System memory exhaustion
        if metrics.memory_percent > self.memory_threshold:
            return RunPodErrorType.OOM_ERROR

        # High error rate
        if metrics.error_rate > self.error_rate_threshold:
            if "timeout" in reason.lower():
                return RunPodErrorType.TIMEOUT
            elif "crash" in reason.lower():
                return RunPodErrorType.LLAMA_SERVER_CRASH
            else:
                return RunPodErrorType.HEALTH_CHECK_FAIL

        # Default
        return RunPodErrorType.UNKNOWN

    async def _export_metrics(
        self, is_healthy: bool, recovery_result: Optional[RecoveryResult]
    ):
        """
        Export metrics to logs (BAO §132 audit logging)

        Future: Can extend to Sentry, Prometheus, etc.
        """
        metrics = self.check_health()
        circuit_state = (
            self.circuit_breaker.get_state().value
            if self.circuit_breaker
            else "disabled"
        )

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "health_check",
            "is_healthy": is_healthy,
            "cpu_percent": round(metrics.cpu_percent, 2),
            "memory_percent": round(metrics.memory_percent, 2),
            "gpu_memory_percent": round(metrics.gpu_memory_percent, 2),
            "error_rate": round(metrics.error_rate, 4),
            "circuit_state": circuit_state,
        }

        if recovery_result:
            log_entry["recovery"] = recovery_result.to_audit_log()

        logger.info(f"[HealthMetrics] {log_entry}")

    def get_diagnostics(self) -> Dict:
        """Get detailed diagnostic information"""
        metrics = self.check_health()

        diagnostics = {
            "timestamp": metrics.timestamp.isoformat(),
            "status": "healthy" if metrics.is_healthy() else "unhealthy",
            "system": {
                "cpu_percent": round(metrics.cpu_percent, 2),
                "memory_percent": round(metrics.memory_percent, 2),
                "gpu_memory_used_mb": round(metrics.gpu_memory_used_mb, 2),
                "gpu_memory_total_mb": round(metrics.gpu_memory_total_mb, 2),
                "gpu_memory_percent": round(metrics.gpu_memory_percent, 2),
            },
            "performance": {
                "success_count": metrics.success_count,
                "error_count": metrics.error_count,
                "error_rate": round(metrics.error_rate, 4),
                "avg_response_time_ms": round(metrics.avg_response_time_ms, 2),
            },
            "issues": {
                "last_error": metrics.last_error,
                "needs_recovery": self.needs_recovery()[0],
                "recovery_reason": self.needs_recovery()[1],
            },
        }

        # Add recovery stats if enabled
        if self.enable_recovery and self.recovery_executor:
            diagnostics["recovery"] = self.recovery_executor.get_recovery_stats()
            diagnostics["circuit_breaker"] = (
                self.circuit_breaker.get_metrics().to_dict()
                if self.circuit_breaker
                else None
            )
            diagnostics["recovery_cooldown_remaining"] = (
                self._get_cooldown_remaining()
            )

        return diagnostics
