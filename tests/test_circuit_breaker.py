"""
Unit Tests for CircuitBreaker
Tests circuit breaker pattern for RunPod serverless workers
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock

import sys
sys.path.insert(0, '/Users/bernhardgoetzendorfer/Projects/BuchhaltGenieV5/local-ai/runpod-serverless')

from circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitBreakerMetrics,
)


@pytest.fixture
def circuit():
    """Create CircuitBreaker with test config"""
    config = CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout=100,  # 100ms for faster tests
        volume_threshold=5,
        error_threshold_percentage=50,
    )
    return CircuitBreaker(config=config)


@pytest.mark.asyncio
async def test_execute_success_closed_state(circuit):
    """Test successful execution in CLOSED state"""
    async def successful_fn():
        return "success"

    result = await circuit.execute(successful_fn)

    assert result == "success"
    assert circuit.get_state() == CircuitState.CLOSED
    assert circuit._successes == 1
    assert circuit._failures == 0


@pytest.mark.asyncio
async def test_execute_failure_closed_state(circuit):
    """Test failed execution in CLOSED state"""
    async def failing_fn():
        raise Exception("Test failure")

    with pytest.raises(Exception, match="Test failure"):
        await circuit.execute(failing_fn)

    assert circuit.get_state() == CircuitState.CLOSED  # Still closed (only 1 failure)
    assert circuit._failures == 1
    assert circuit._consecutive_failures == 1


@pytest.mark.asyncio
async def test_circuit_opens_after_threshold_failures(circuit):
    """Test circuit opens after failure threshold"""
    async def failing_fn():
        raise Exception("Test failure")

    # Trigger 3 failures (threshold)
    for i in range(3):
        with pytest.raises(Exception):
            await circuit.execute(failing_fn)

    assert circuit.get_state() == CircuitState.OPEN
    assert circuit._consecutive_failures == 3


@pytest.mark.asyncio
async def test_execute_blocked_when_open(circuit):
    """Test execution blocked when circuit is OPEN"""
    # Force circuit to OPEN
    circuit._set_state(CircuitState.OPEN)

    async def any_fn():
        return "should not execute"

    result = await circuit.execute(any_fn)

    assert result is None  # No fallback, returns None


@pytest.mark.asyncio
async def test_execute_uses_fallback_when_open(circuit):
    """Test fallback is used when circuit is OPEN"""
    circuit._set_state(CircuitState.OPEN)

    async def blocked_fn():
        return "blocked"

    async def fallback_fn():
        return "fallback_result"

    result = await circuit.execute(blocked_fn, fallback_fn)

    assert result == "fallback_result"


@pytest.mark.asyncio
async def test_execute_fallback_on_error(circuit):
    """Test fallback is used when main function fails"""
    async def failing_fn():
        raise Exception("Main failed")

    async def fallback_fn():
        return "fallback_success"

    result = await circuit.execute(failing_fn, fallback_fn)

    assert result == "fallback_success"


@pytest.mark.asyncio
async def test_circuit_transitions_to_half_open(circuit):
    """Test circuit transitions from OPEN to HALF_OPEN after timeout"""
    # Force circuit to OPEN
    circuit._set_state(CircuitState.OPEN)
    circuit._state_changed_at = time.time() * 1000 - 200  # 200ms ago

    # Config timeout is 100ms, so should transition
    circuit._check_state_transition()

    assert circuit.get_state() == CircuitState.HALF_OPEN


@pytest.mark.asyncio
async def test_circuit_does_not_transition_before_timeout(circuit):
    """Test circuit stays OPEN before timeout"""
    circuit._set_state(CircuitState.OPEN)
    circuit._state_changed_at = time.time() * 1000 - 50  # Only 50ms ago

    circuit._check_state_transition()

    assert circuit.get_state() == CircuitState.OPEN


@pytest.mark.asyncio
async def test_circuit_closes_from_half_open_after_successes(circuit):
    """Test circuit closes from HALF_OPEN after success threshold"""
    # Set to HALF_OPEN
    circuit._set_state(CircuitState.HALF_OPEN)

    async def successful_fn():
        return "success"

    # Execute 2 successes (threshold = 2)
    await circuit.execute(successful_fn)
    assert circuit.get_state() == CircuitState.HALF_OPEN  # Still half-open

    await circuit.execute(successful_fn)
    assert circuit.get_state() == CircuitState.CLOSED  # Now closed


@pytest.mark.asyncio
async def test_circuit_reopens_from_half_open_on_failure(circuit):
    """Test circuit reopens from HALF_OPEN on failure"""
    circuit._set_state(CircuitState.HALF_OPEN)

    async def failing_fn():
        raise Exception("Test failure")

    # First success
    async def successful_fn():
        return "success"

    await circuit.execute(successful_fn)
    assert circuit.get_state() == CircuitState.HALF_OPEN

    # Then failure - should reopen
    with pytest.raises(Exception):
        await circuit.execute(failing_fn)

    assert circuit.get_state() == CircuitState.HALF_OPEN  # Stays half-open, needs more failures


@pytest.mark.asyncio
async def test_consecutive_failures_reset_on_success(circuit):
    """Test consecutive failures reset on success"""
    async def failing_fn():
        raise Exception("Failure")

    async def successful_fn():
        return "success"

    # 2 failures
    for _ in range(2):
        with pytest.raises(Exception):
            await circuit.execute(failing_fn)

    assert circuit._consecutive_failures == 2

    # 1 success - should reset
    await circuit.execute(successful_fn)

    assert circuit._consecutive_failures == 0
    assert circuit._consecutive_successes == 1


@pytest.mark.asyncio
async def test_consecutive_successes_reset_on_failure(circuit):
    """Test consecutive successes reset on failure"""
    async def successful_fn():
        return "success"

    async def failing_fn():
        raise Exception("Failure")

    # 2 successes
    for _ in range(2):
        await circuit.execute(successful_fn)

    assert circuit._consecutive_successes == 2

    # 1 failure - should reset
    with pytest.raises(Exception):
        await circuit.execute(failing_fn)

    assert circuit._consecutive_successes == 0


@pytest.mark.asyncio
async def test_error_threshold_percentage_opens_circuit(circuit):
    """Test circuit opens when error percentage threshold is exceeded"""
    # volume_threshold = 5, error_threshold_percentage = 50%
    # So 3 failures out of 5 requests = 60% error rate

    async def successful_fn():
        return "success"

    async def failing_fn():
        raise Exception("Failure")

    # 2 successes
    await circuit.execute(successful_fn)
    await circuit.execute(successful_fn)

    assert circuit.get_state() == CircuitState.CLOSED

    # 3 failures = 60% error rate over 5 requests
    for _ in range(3):
        with pytest.raises(Exception):
            await circuit.execute(failing_fn)

    # Should be open now (error rate = 60% > 50%)
    assert circuit.get_state() == CircuitState.OPEN


@pytest.mark.asyncio
async def test_volume_threshold_not_met(circuit):
    """Test circuit doesn't open before volume threshold"""
    # volume_threshold = 5

    async def failing_fn():
        raise Exception("Failure")

    # Only 2 failures (< volume_threshold)
    for _ in range(2):
        with pytest.raises(Exception):
            await circuit.execute(failing_fn)

    # Should still be closed (volume threshold not met)
    assert circuit.get_state() == CircuitState.CLOSED


def test_get_metrics(circuit):
    """Test getting circuit breaker metrics"""
    circuit._failures = 5
    circuit._successes = 10
    circuit._total_requests = 15
    circuit._consecutive_failures = 2
    circuit._consecutive_successes = 0

    metrics = circuit.get_metrics()

    assert isinstance(metrics, CircuitBreakerMetrics)
    assert metrics.failures == 5
    assert metrics.successes == 10
    assert metrics.total_requests == 15
    assert metrics.consecutive_failures == 2
    assert metrics.state == CircuitState.CLOSED


def test_metrics_to_dict(circuit):
    """Test metrics conversion to dictionary"""
    metrics = circuit.get_metrics()
    metrics_dict = metrics.to_dict()

    assert isinstance(metrics_dict, dict)
    assert metrics_dict['state'] == 'closed'
    assert 'failures' in metrics_dict
    assert 'successes' in metrics_dict


def test_reset(circuit):
    """Test manual circuit reset"""
    # Set some state
    circuit._failures = 10
    circuit._successes = 5
    circuit._consecutive_failures = 3
    circuit._set_state(CircuitState.OPEN)

    # Reset
    circuit.reset()

    assert circuit.get_state() == CircuitState.CLOSED
    assert circuit._failures == 0
    assert circuit._successes == 0
    assert circuit._consecutive_failures == 0


@pytest.mark.asyncio
async def test_sync_function_execution(circuit):
    """Test executing synchronous function"""
    def sync_fn():
        return "sync_result"

    result = await circuit.execute(sync_fn)

    assert result == "sync_result"


@pytest.mark.asyncio
async def test_sync_fallback_function(circuit):
    """Test synchronous fallback function"""
    circuit._set_state(CircuitState.OPEN)

    async def blocked_fn():
        return "blocked"

    def sync_fallback():
        return "sync_fallback"

    result = await circuit.execute(blocked_fn, sync_fallback)

    assert result == "sync_fallback"


@pytest.mark.asyncio
async def test_fallback_also_fails(circuit):
    """Test when both main and fallback fail"""
    async def failing_fn():
        raise Exception("Main failed")

    async def failing_fallback():
        raise Exception("Fallback failed")

    # Should raise the original exception
    with pytest.raises(Exception, match="Main failed"):
        await circuit.execute(failing_fn, failing_fallback)


@pytest.mark.asyncio
async def test_fallback_fails_when_open(circuit):
    """Test fallback failure when circuit is OPEN"""
    circuit._set_state(CircuitState.OPEN)

    async def any_fn():
        return "blocked"

    async def failing_fallback():
        raise Exception("Fallback failed")

    result = await circuit.execute(any_fn, failing_fallback)

    # Should return None (fallback failed)
    assert result is None


def test_repr(circuit):
    """Test string representation"""
    repr_str = repr(circuit)

    assert "CircuitBreaker" in repr_str
    assert "state=closed" in repr_str
    assert "failures=0/3" in repr_str
    assert "successes=0/2" in repr_str


def test_default_config():
    """Test default configuration values"""
    circuit = CircuitBreaker()

    assert circuit.config.failure_threshold == 5
    assert circuit.config.success_threshold == 2
    assert circuit.config.timeout == 30000  # 30s
    assert circuit.config.volume_threshold == 10
    assert circuit.config.error_threshold_percentage == 50


def test_custom_config():
    """Test custom configuration"""
    config = CircuitBreakerConfig(
        failure_threshold=10,
        success_threshold=5,
        timeout=60000,
        volume_threshold=20,
        error_threshold_percentage=75,
    )
    circuit = CircuitBreaker(config=config)

    assert circuit.config.failure_threshold == 10
    assert circuit.config.success_threshold == 5
    assert circuit.config.timeout == 60000
    assert circuit.config.volume_threshold == 20
    assert circuit.config.error_threshold_percentage == 75
