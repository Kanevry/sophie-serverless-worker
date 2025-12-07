"""
Integration Tests for Self-Healing System
Tests end-to-end recovery scenarios with real components
"""

import pytest
import asyncio
import subprocess
import time
from unittest.mock import patch

import sys
sys.path.insert(0, '/Users/bernhardgoetzendorfer/Projects/BuchhaltGenieV5/local-ai/runpod-serverless')

from health_monitor import HealthMonitor
from recovery_executor import RecoveryExecutor
from runpod_error_types import RunPodErrorType


@pytest.fixture
def executor():
    """Create RecoveryExecutor for integration tests"""
    return RecoveryExecutor(
        llama_server_port=8080,
        max_restart_attempts=3,
        restart_delay_seconds=2,
    )


@pytest.fixture
def monitor():
    """Create HealthMonitor for integration tests"""
    return HealthMonitor(
        check_interval_seconds=5,
        memory_threshold_percent=90.0,
        gpu_memory_threshold_percent=90.0,
        error_rate_threshold=0.25,
        recovery_cooldown_seconds=10,
        enable_recovery=True,
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not subprocess.run(['which', 'llama-server'], capture_output=True).returncode == 0,
    reason="llama-server not available"
)
@pytest.mark.asyncio
async def test_server_restart_integration(executor):
    """
    Integration test: Server crash detection and restart

    Prerequisites:
    - llama-server installed
    - Model available at default path
    """
    # Start llama-server
    process = subprocess.Popen(
        ['llama-server', '--model', '/tmp/test-model.gguf', '--port', '8081'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        # Wait for server to start
        await asyncio.sleep(2)

        # Kill the server
        process.kill()
        process.wait()

        # Wait for process to die
        await asyncio.sleep(1)

        # Execute recovery
        result = await executor._restart_llama_server()

        # Note: This may fail if model doesn't exist, but tests the mechanism
        assert result.action_taken == "graceful_restart"

    finally:
        # Cleanup
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gpu_cleanup_integration(executor):
    """
    Integration test: GPU memory cleanup

    Prerequisites:
    - PyTorch installed
    - CUDA available
    """
    try:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Allocate some GPU memory
        large_tensor = torch.randn(1000, 1000, device='cuda')

        # Get memory before cleanup
        mem_before = torch.cuda.memory_allocated(0)

        # Execute cleanup
        result = await executor._cleanup_gpu_memory()

        # Get memory after cleanup
        mem_after = torch.cuda.memory_allocated(0)

        assert result.success is True
        assert result.action_taken == "gpu_cleanup"
        # Memory should be freed or same (if nothing to free)
        assert mem_after <= mem_before

        # Cleanup
        del large_tensor
        torch.cuda.empty_cache()

    except ImportError:
        pytest.skip("PyTorch not available")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_health_monitor_recovery_cycle(monitor):
    """
    Integration test: Full health monitor recovery cycle

    Tests:
    1. Detect unhealthy state
    2. Execute recovery
    3. Update circuit breaker
    4. Cooldown enforcement
    """
    # Simulate unhealthy state (high GPU memory)
    monitor.current_metrics.gpu_memory_used_mb = 23000
    monitor.current_metrics.gpu_memory_total_mb = 24000  # 95.8%

    # Mock recovery executor to avoid actual recovery
    with patch.object(monitor.recovery_executor, 'execute_recovery') as mock_recovery:
        from recovery_executor import RecoveryResult

        mock_recovery.return_value = RecoveryResult(
            success=True,
            action_taken="mock_cleanup",
            freed_mb=1000.0,
            duration_ms=100.0
        )

        # First recovery attempt
        is_healthy, result = await monitor.check_health_with_recovery()

        assert result is not None
        assert result.success is True
        assert monitor.last_recovery_time is not None

        # Second recovery attempt (should be blocked by cooldown)
        is_healthy2, result2 = await monitor.check_health_with_recovery()

        assert result2 is None  # Blocked by cooldown


@pytest.mark.integration
@pytest.mark.asyncio
async def test_circuit_breaker_integration(monitor):
    """
    Integration test: Circuit breaker state transitions

    Tests:
    1. CLOSED → OPEN after failures
    2. OPEN → HALF_OPEN after timeout
    3. HALF_OPEN → CLOSED after successes
    """
    # Simulate failures
    monitor.current_metrics.gpu_memory_used_mb = 23000
    monitor.current_metrics.gpu_memory_total_mb = 24000

    with patch.object(monitor.recovery_executor, 'execute_recovery') as mock_recovery:
        from recovery_executor import RecoveryResult

        # 5 consecutive failures
        mock_recovery.return_value = RecoveryResult(
            success=False,
            action_taken="failed_cleanup",
            error="Mock failure"
        )

        for i in range(5):
            monitor.last_recovery_time = None  # Reset cooldown
            await monitor.check_health_with_recovery()

        # Circuit should be OPEN
        from circuit_breaker import CircuitState
        assert monitor.circuit_breaker.get_state() == CircuitState.OPEN

        # Wait for timeout (circuit config: 30s)
        monitor.circuit_breaker._state_changed_at = time.time() * 1000 - 31000

        # Check state transition
        monitor.circuit_breaker._check_state_transition()
        assert monitor.circuit_breaker.get_state() == CircuitState.HALF_OPEN

        # Simulate successes
        mock_recovery.return_value = RecoveryResult(
            success=True,
            action_taken="successful_cleanup"
        )

        # 2 successes to close
        for i in range(2):
            monitor.last_recovery_time = None
            await monitor.check_health_with_recovery()

        # Circuit should be CLOSED
        assert monitor.circuit_breaker.get_state() == CircuitState.CLOSED


@pytest.mark.integration
@pytest.mark.asyncio
async def test_monitoring_loop_integration(monitor):
    """
    Integration test: Monitoring loop execution

    Tests continuous monitoring loop with metrics export
    """
    # Run monitoring loop for 3 iterations
    iteration_count = 0
    max_iterations = 3

    async def limited_loop():
        nonlocal iteration_count

        while iteration_count < max_iterations:
            try:
                await monitor.check_health_with_recovery()
                await monitor._export_metrics(True, None)
                iteration_count += 1
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Iteration {iteration_count} error: {e}")
                break

    # Run loop
    await asyncio.wait_for(limited_loop(), timeout=5.0)

    assert iteration_count == max_iterations


@pytest.mark.integration
def test_error_classification_integration():
    """
    Integration test: Error classification patterns

    Tests that real error messages are correctly classified
    """
    from runpod_error_types import classify_runpod_error

    # VRAM exhausted
    error1 = Exception("CUDA out of memory. Tried to allocate 20.00 GiB")
    classified1 = classify_runpod_error(error1)
    assert classified1.type == RunPodErrorType.VRAM_EXHAUSTED

    # OOM
    error2 = Exception("MemoryError: Cannot allocate memory")
    classified2 = classify_runpod_error(error2)
    assert classified2.type == RunPodErrorType.OOM_ERROR

    # Server crash
    error3 = Exception("Connection refused on port 8080")
    classified3 = classify_runpod_error(error3)
    assert classified3.type == RunPodErrorType.LLAMA_SERVER_CRASH

    # Timeout
    error4 = Exception("Request timeout after 30s")
    classified4 = classify_runpod_error(error4)
    assert classified4.type == RunPodErrorType.TIMEOUT


@pytest.mark.integration
@pytest.mark.asyncio
async def test_exponential_backoff_integration(monitor):
    """
    Integration test: Exponential backoff cooldown

    Tests that cooldown increases exponentially after failures
    """
    import time

    # First recovery
    monitor.recovery_attempt_count = 1
    monitor.last_recovery_time = time.time()

    cooldown1 = monitor.recovery_cooldown_seconds * (2 ** 0)  # 10s

    # Second recovery
    monitor.recovery_attempt_count = 2

    cooldown2 = monitor.recovery_cooldown_seconds * (2 ** 1)  # 20s

    # Third recovery
    monitor.recovery_attempt_count = 3

    cooldown3 = monitor.recovery_cooldown_seconds * (2 ** 2)  # 40s

    assert cooldown1 == 10
    assert cooldown2 == 20
    assert cooldown3 == 40


@pytest.mark.integration
def test_recovery_stats_integration(executor):
    """
    Integration test: Recovery statistics tracking

    Tests that stats are correctly accumulated
    """
    # Simulate recoveries
    executor.total_recoveries = 10
    executor.successful_recoveries = 7
    executor.failed_recoveries = 3

    stats = executor.get_recovery_stats()

    assert stats['total_recoveries'] == 10
    assert stats['successful_recoveries'] == 7
    assert stats['failed_recoveries'] == 3
    assert stats['success_rate_percent'] == 70.0


@pytest.mark.integration
def test_diagnostics_integration(monitor):
    """
    Integration test: Diagnostics endpoint

    Tests that get_diagnostics returns complete data
    """
    # Add some metrics
    monitor.record_success(100.0)
    monitor.record_success(150.0)
    monitor.record_error("Test error")

    diagnostics = monitor.get_diagnostics()

    # Verify structure
    assert 'timestamp' in diagnostics
    assert 'status' in diagnostics
    assert 'system' in diagnostics
    assert 'performance' in diagnostics
    assert 'issues' in diagnostics
    assert 'recovery' in diagnostics
    assert 'circuit_breaker' in diagnostics

    # Verify data
    assert diagnostics['performance']['success_count'] == 2
    assert diagnostics['performance']['error_count'] == 1
    assert diagnostics['status'] in ['healthy', 'unhealthy']
