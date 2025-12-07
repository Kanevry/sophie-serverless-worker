"""
Circuit Breaker Pattern Implementation (Python Port)
Ported from src/lib/audit/circuit-breaker.ts

Prevents cascading failures by:
- Tracking failure rates
- Opening circuit when threshold exceeded
- Automatically testing recovery
- Providing fallback behavior

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Too many failures, requests blocked
- HALF_OPEN: Testing if service recovered
"""

import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable, TypeVar, Generic, Dict, Any
import asyncio

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states"""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half-open"


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""

    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes needed to close from half-open
    timeout: int = 30000  # Milliseconds before trying half-open
    volume_threshold: int = 10  # Minimum requests before evaluating
    error_threshold_percentage: int = 50  # Percentage of errors to trip


@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker metrics"""

    failures: int = 0
    successes: int = 0
    total_requests: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    state: CircuitState = CircuitState.CLOSED
    state_changed_at: float = field(default_factory=lambda: time.time() * 1000)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "failures": self.failures,
            "successes": self.successes,
            "total_requests": self.total_requests,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "state": self.state.value,
            "state_changed_at": self.state_changed_at,
        }


class CircuitBreaker(Generic[T]):
    """
    Circuit breaker for protecting RunPod serverless workers

    Usage:
        circuit = CircuitBreaker()
        result = await circuit.execute(some_async_function, fallback_function)
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()

        self._state: CircuitState = CircuitState.CLOSED
        self._failures = 0
        self._successes = 0
        self._total_requests = 0
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._last_failure_time = 0.0
        self._last_success_time = 0.0
        self._state_changed_at = time.time() * 1000  # milliseconds

        logger.info(
            f"Circuit Breaker initialized: "
            f"failure_threshold={self.config.failure_threshold}, "
            f"success_threshold={self.config.success_threshold}, "
            f"timeout={self.config.timeout}ms"
        )

    async def execute(
        self,
        fn: Callable[[], T],
        fallback: Optional[Callable[[], T]] = None,
    ) -> Optional[T]:
        """
        Execute a function with circuit breaker protection

        Args:
            fn: Function to execute (can be async or sync)
            fallback: Optional fallback function if circuit is open

        Returns:
            Result of fn() or fallback(), or None if circuit is open
        """
        # Check if circuit should transition to half-open
        self._check_state_transition()

        # If circuit is open, use fallback or return None
        if self._state == CircuitState.OPEN:
            logger.warning("[CircuitBreaker] Circuit is OPEN, using fallback")
            if fallback:
                try:
                    if asyncio.iscoroutinefunction(fallback):
                        return await fallback()
                    else:
                        return fallback()
                except Exception as fallback_error:
                    logger.error(
                        f"[CircuitBreaker] Fallback failed: {fallback_error}"
                    )
                    return None
            return None

        try:
            # Execute the function
            if asyncio.iscoroutinefunction(fn):
                result = await fn()
            else:
                result = fn()

            self._on_success()
            return result

        except Exception as error:
            self._on_failure()

            logger.error(f"[CircuitBreaker] Function failed: {error}")

            # Try fallback if available
            if fallback:
                try:
                    if asyncio.iscoroutinefunction(fallback):
                        return await fallback()
                    else:
                        return fallback()
                except Exception as fallback_error:
                    logger.error(
                        f"[CircuitBreaker] Fallback failed: {fallback_error}"
                    )

            # Re-raise exception if no fallback or fallback failed
            raise error

    def _check_state_transition(self):
        """Check if circuit should transition from OPEN to HALF_OPEN"""
        if self._state != CircuitState.OPEN:
            return

        current_time = time.time() * 1000  # milliseconds
        time_since_open = current_time - self._state_changed_at

        if time_since_open >= self.config.timeout:
            logger.info(
                f"[CircuitBreaker] Transitioning to HALF_OPEN after {time_since_open}ms"
            )
            self._set_state(CircuitState.HALF_OPEN)
            self._consecutive_successes = 0

    def _on_success(self):
        """Record a successful request"""
        self._successes += 1
        self._total_requests += 1
        self._last_success_time = time.time() * 1000
        self._consecutive_failures = 0
        self._consecutive_successes += 1

        # If in HALF_OPEN, check if we should close
        if self._state == CircuitState.HALF_OPEN:
            if self._consecutive_successes >= self.config.success_threshold:
                logger.info(
                    f"[CircuitBreaker] Closing circuit after {self._consecutive_successes} successes"
                )
                self._set_state(CircuitState.CLOSED)
                self._reset_metrics()

    def _on_failure(self):
        """Record a failed request"""
        self._failures += 1
        self._total_requests += 1
        self._last_failure_time = time.time() * 1000
        self._consecutive_successes = 0
        self._consecutive_failures += 1

        # Check if we should open the circuit
        if self._should_open():
            logger.warning(
                f"[CircuitBreaker] Opening circuit after {self._consecutive_failures} consecutive failures"
            )
            self._set_state(CircuitState.OPEN)

    def _should_open(self) -> bool:
        """
        Check if circuit should open based on thresholds

        Criteria:
        1. Consecutive failures >= failure_threshold
        2. Total requests >= volume_threshold
        3. Error rate >= error_threshold_percentage
        """
        # Criterion 1: Consecutive failures
        if self._consecutive_failures >= self.config.failure_threshold:
            return True

        # Criterion 2: Volume threshold not met
        if self._total_requests < self.config.volume_threshold:
            return False

        # Criterion 3: Error rate
        error_rate = (self._failures / self._total_requests) * 100
        if error_rate >= self.config.error_threshold_percentage:
            return True

        return False

    def _set_state(self, new_state: CircuitState):
        """Set circuit state and record timestamp"""
        old_state = self._state
        self._state = new_state
        self._state_changed_at = time.time() * 1000

        logger.info(
            f"[CircuitBreaker] State change: {old_state.value} → {new_state.value}"
        )

    def _reset_metrics(self):
        """Reset metrics when circuit closes"""
        self._failures = 0
        self._successes = 0
        self._total_requests = 0
        self._consecutive_failures = 0
        self._consecutive_successes = 0

    def get_state(self) -> CircuitState:
        """Get current circuit state"""
        self._check_state_transition()
        return self._state

    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get current circuit breaker metrics"""
        return CircuitBreakerMetrics(
            failures=self._failures,
            successes=self._successes,
            total_requests=self._total_requests,
            last_failure_time=self._last_failure_time,
            last_success_time=self._last_success_time,
            consecutive_failures=self._consecutive_failures,
            consecutive_successes=self._consecutive_successes,
            state=self._state,
            state_changed_at=self._state_changed_at,
        )

    def reset(self):
        """Manually reset the circuit breaker"""
        logger.info("[CircuitBreaker] Manual reset")
        self._set_state(CircuitState.CLOSED)
        self._reset_metrics()

    def __repr__(self) -> str:
        return (
            f"<CircuitBreaker state={self._state.value} "
            f"failures={self._consecutive_failures}/{self.config.failure_threshold} "
            f"successes={self._consecutive_successes}/{self.config.success_threshold}>"
        )
