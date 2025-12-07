"""
Unit Tests for HealthMonitor
Tests health monitoring and auto-recovery integration
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

import sys
sys.path.insert(0, '/Users/bernhardgoetzendorfer/Projects/BuchhaltGenieV5/local-ai/runpod-serverless')

from health_monitor import HealthMonitor, HealthMetrics
from recovery_executor import RecoveryResult
from runpod_error_types import RunPodErrorType
from circuit_breaker import CircuitState


@pytest.fixture
def monitor():
    """Create HealthMonitor instance"""
    return HealthMonitor(
        check_interval_seconds=1,  # Fast for tests
        memory_threshold_percent=90.0,
        gpu_memory_threshold_percent=90.0,
        error_rate_threshold=0.25,
        recovery_cooldown_seconds=5,  # Short cooldown for tests
        enable_recovery=True,
    )


@pytest.fixture
def monitor_no_recovery():
    """Create HealthMonitor with recovery disabled"""
    return HealthMonitor(enable_recovery=False)


def test_initialization(monitor):
    """Test HealthMonitor initialization"""
    assert monitor.check_interval == 1
    assert monitor.memory_threshold == 90.0
    assert monitor.gpu_memory_threshold == 90.0
    assert monitor.enable_recovery is True
    assert monitor.recovery_executor is not None
    assert monitor.circuit_breaker is not None


def test_initialization_no_recovery(monitor_no_recovery):
    """Test initialization with recovery disabled"""
    assert monitor_no_recovery.enable_recovery is False
    assert monitor_no_recovery.recovery_executor is None
    assert monitor_no_recovery.circuit_breaker is None


def test_record_success(monitor):
    """Test recording successful request"""
    monitor.record_success(response_time_ms=150.0)

    assert monitor.current_metrics.success_count == 1
    assert len(monitor.response_times) == 1
    assert monitor.current_metrics.avg_response_time_ms == 150.0


def test_record_error(monitor):
    """Test recording error"""
    monitor.record_error("Test error")

    assert monitor.current_metrics.error_count == 1
    assert monitor.current_metrics.last_error == "Test error"


def test_error_rate_calculation(monitor):
    """Test error rate calculation"""
    # 2 successes, 3 errors = 60% error rate
    monitor.record_success(100.0)
    monitor.record_success(150.0)
    monitor.record_error("Error 1")
    monitor.record_error("Error 2")
    monitor.record_error("Error 3")

    assert monitor.current_metrics.error_rate == 0.6


def test_needs_recovery_high_memory(monitor):
    """Test needs_recovery detects high memory"""
    with patch('psutil.virtual_memory', return_value=Mock(percent=95.0)):
        with patch('psutil.cpu_percent', return_value=50.0):
            monitor.check_health()

            needs_recovery, reason = monitor.needs_recovery()

            assert needs_recovery is True
            assert "memory" in reason.lower()


def test_needs_recovery_high_gpu(monitor):
    """Test needs_recovery detects high GPU memory"""
    # Mock GPU at 95%
    monitor.current_metrics.gpu_memory_used_mb = 22800
    monitor.current_metrics.gpu_memory_total_mb = 24000

    needs_recovery, reason = monitor.needs_recovery()

    assert needs_recovery is True
    assert "gpu" in reason.lower()


def test_needs_recovery_high_error_rate(monitor):
    """Test needs_recovery detects high error rate"""
    # 8 errors out of 10 requests = 80% error rate
    for _ in range(2):
        monitor.record_success(100.0)
    for _ in range(8):
        monitor.record_error("Error")

    needs_recovery, reason = monitor.needs_recovery()

    assert needs_recovery is True
    assert "error rate" in reason.lower()


def test_needs_recovery_healthy(monitor):
    """Test needs_recovery returns False when healthy"""
    monitor.current_metrics.memory_percent = 50.0
    monitor.current_metrics.gpu_memory_used_mb = 10000
    monitor.current_metrics.gpu_memory_total_mb = 24000
    monitor.record_success(100.0)

    needs_recovery, reason = monitor.needs_recovery()

    assert needs_recovery is False
    assert reason == ""


@pytest.mark.asyncio
async def test_check_health_with_recovery_healthy(monitor):
    """Test check_health_with_recovery when healthy"""
    with patch.object(monitor, 'check_health', return_value=Mock(is_healthy=Mock(return_value=True))):
        is_healthy, recovery_result = await monitor.check_health_with_recovery()

        assert is_healthy is True
        assert recovery_result is None


@pytest.mark.asyncio
async def test_check_health_with_recovery_disabled(monitor_no_recovery):
    """Test check_health_with_recovery with recovery disabled"""
    with patch.object(monitor_no_recovery, 'check_health', return_value=Mock(is_healthy=Mock(return_value=False))):
        is_healthy, recovery_result = await monitor_no_recovery.check_health_with_recovery()

        assert is_healthy is False
        assert recovery_result is None


@pytest.mark.asyncio
async def test_check_health_with_recovery_executes_recovery(monitor):
    """Test check_health_with_recovery executes recovery when needed"""
    # Setup unhealthy state
    monitor.current_metrics.gpu_memory_used_mb = 23000
    monitor.current_metrics.gpu_memory_total_mb = 24000  # 95.8% GPU usage

    mock_recovery_result = RecoveryResult(
        success=True,
        action_taken="gpu_cleanup",
        freed_mb=1000.0,
        duration_ms=100.0
    )

    with patch.object(monitor.recovery_executor, 'execute_recovery', return_value=mock_recovery_result):
        is_healthy, recovery_result = await monitor.check_health_with_recovery()

        assert recovery_result is not None
        assert recovery_result.success is True
        assert recovery_result.action_taken == "gpu_cleanup"


@pytest.mark.asyncio
async def test_check_health_with_recovery_cooldown(monitor):
    """Test recovery cooldown prevents immediate retry"""
    # Set last recovery to 2 seconds ago (cooldown is 5s)
    monitor.last_recovery_time = datetime.now() - timedelta(seconds=2)
    monitor.recovery_attempt_count = 1

    # Setup unhealthy state
    monitor.current_metrics.gpu_memory_used_mb = 23000
    monitor.current_metrics.gpu_memory_total_mb = 24000

    is_healthy, recovery_result = await monitor.check_health_with_recovery()

    # Should not execute recovery (in cooldown)
    assert recovery_result is None


@pytest.mark.asyncio
async def test_check_health_with_recovery_exponential_backoff(monitor):
    """Test recovery cooldown exponential backoff"""
    monitor.recovery_attempt_count = 2  # 2nd attempt
    monitor.last_recovery_time = datetime.now() - timedelta(seconds=8)

    # Cooldown = 5s * 2^1 = 10s
    # Elapsed = 8s, so should still be in cooldown

    monitor.current_metrics.gpu_memory_used_mb = 23000
    monitor.current_metrics.gpu_memory_total_mb = 24000

    can_attempt = monitor._can_attempt_recovery()

    assert can_attempt is False


@pytest.mark.asyncio
async def test_recovery_updates_circuit_breaker_on_success(monitor):
    """Test successful recovery updates circuit breaker"""
    monitor.current_metrics.gpu_memory_used_mb = 23000
    monitor.current_metrics.gpu_memory_total_mb = 24000

    mock_recovery = RecoveryResult(success=True, action_taken="cleanup")

    with patch.object(monitor.recovery_executor, 'execute_recovery', return_value=mock_recovery):
        with patch.object(monitor.circuit_breaker, '_on_success') as mock_success:
            await monitor.check_health_with_recovery()

            mock_success.assert_called_once()


@pytest.mark.asyncio
async def test_recovery_updates_circuit_breaker_on_failure(monitor):
    """Test failed recovery updates circuit breaker"""
    monitor.current_metrics.gpu_memory_used_mb = 23000
    monitor.current_metrics.gpu_memory_total_mb = 24000

    mock_recovery = RecoveryResult(success=False, action_taken="cleanup", error="Failed")

    with patch.object(monitor.recovery_executor, 'execute_recovery', return_value=mock_recovery):
        with patch.object(monitor.circuit_breaker, '_on_failure') as mock_failure:
            await monitor.check_health_with_recovery()

            mock_failure.assert_called_once()


def test_classify_health_issue_vram_exhausted(monitor):
    """Test health issue classification for VRAM exhaustion"""
    metrics = Mock()
    metrics.gpu_memory_percent = 95.0
    metrics.memory_percent = 70.0
    metrics.error_rate = 0.1

    error_type = monitor._classify_health_issue(metrics, "High GPU memory")

    assert error_type == RunPodErrorType.VRAM_EXHAUSTED


def test_classify_health_issue_oom_error(monitor):
    """Test health issue classification for OOM"""
    metrics = Mock()
    metrics.gpu_memory_percent = 70.0
    metrics.memory_percent = 95.0
    metrics.error_rate = 0.1

    error_type = monitor._classify_health_issue(metrics, "High system memory")

    assert error_type == RunPodErrorType.OOM_ERROR


def test_classify_health_issue_timeout(monitor):
    """Test health issue classification for timeout"""
    metrics = Mock()
    metrics.gpu_memory_percent = 70.0
    metrics.memory_percent = 70.0
    metrics.error_rate = 0.3

    error_type = monitor._classify_health_issue(metrics, "Request timeout errors")

    assert error_type == RunPodErrorType.TIMEOUT


def test_classify_health_issue_server_crash(monitor):
    """Test health issue classification for server crash"""
    metrics = Mock()
    metrics.gpu_memory_percent = 70.0
    metrics.memory_percent = 70.0
    metrics.error_rate = 0.3

    error_type = monitor._classify_health_issue(metrics, "Server process crashed")

    assert error_type == RunPodErrorType.LLAMA_SERVER_CRASH


@pytest.mark.asyncio
async def test_export_metrics(monitor):
    """Test metrics export to logs"""
    mock_recovery = RecoveryResult(
        success=True,
        action_taken="cleanup",
        duration_ms=100.0,
        freed_mb=1000.0
    )

    with patch('health_monitor.logger.info') as mock_log:
        await monitor._export_metrics(is_healthy=True, recovery_result=mock_recovery)

        # Verify logging was called
        mock_log.assert_called()

        # Get the logged data
        call_args = mock_log.call_args[0][0]
        assert "HealthMetrics" in call_args


def test_get_diagnostics_with_recovery(monitor):
    """Test get_diagnostics with recovery enabled"""
    monitor.record_success(100.0)
    monitor.record_success(150.0)

    diagnostics = monitor.get_diagnostics()

    assert 'recovery' in diagnostics
    assert 'circuit_breaker' in diagnostics
    assert 'recovery_cooldown_remaining' in diagnostics
    assert diagnostics['status'] in ['healthy', 'unhealthy']


def test_get_diagnostics_no_recovery(monitor_no_recovery):
    """Test get_diagnostics with recovery disabled"""
    diagnostics = monitor_no_recovery.get_diagnostics()

    assert 'recovery' not in diagnostics
    assert 'circuit_breaker' not in diagnostics


def test_get_cooldown_remaining_no_recovery_yet(monitor):
    """Test cooldown remaining when no recovery attempted"""
    remaining = monitor._get_cooldown_remaining()

    assert remaining == 0.0


def test_get_cooldown_remaining_after_recovery(monitor):
    """Test cooldown remaining after recovery"""
    monitor.last_recovery_time = datetime.now() - timedelta(seconds=2)
    monitor.recovery_attempt_count = 1

    # Cooldown = 5s * 2^0 = 5s
    # Elapsed = 2s
    # Remaining = 3s

    remaining = monitor._get_cooldown_remaining()

    assert 2.5 < remaining < 3.5  # Allow some tolerance


def test_get_cooldown_remaining_exponential(monitor):
    """Test exponential backoff cooldown"""
    monitor.last_recovery_time = datetime.now() - timedelta(seconds=3)
    monitor.recovery_attempt_count = 2  # 2nd attempt

    # Cooldown = 5s * 2^1 = 10s
    # Elapsed = 3s
    # Remaining = 7s

    remaining = monitor._get_cooldown_remaining()

    assert 6.5 < remaining < 7.5


def test_can_attempt_recovery_no_previous(monitor):
    """Test can_attempt_recovery with no previous recovery"""
    can_attempt = monitor._can_attempt_recovery()

    assert can_attempt is True


def test_can_attempt_recovery_during_cooldown(monitor):
    """Test can_attempt_recovery during cooldown"""
    monitor.last_recovery_time = datetime.now() - timedelta(seconds=2)
    monitor.recovery_attempt_count = 1

    can_attempt = monitor._can_attempt_recovery()

    assert can_attempt is False  # Still in 5s cooldown


def test_can_attempt_recovery_after_cooldown(monitor):
    """Test can_attempt_recovery after cooldown"""
    monitor.last_recovery_time = datetime.now() - timedelta(seconds=6)
    monitor.recovery_attempt_count = 1

    can_attempt = monitor._can_attempt_recovery()

    assert can_attempt is True  # 6s > 5s cooldown


@pytest.mark.asyncio
async def test_run_monitoring_loop_stops_on_exception(monitor):
    """Test monitoring loop handles exceptions"""
    # Mock check_health_with_recovery to raise exception
    with patch.object(monitor, 'check_health_with_recovery', side_effect=Exception("Test error")):
        # Run for a short time
        task = asyncio.create_task(monitor.run_monitoring_loop(interval_seconds=0.1))

        await asyncio.sleep(0.3)  # Let it run a few iterations
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass  # Expected


def test_health_metrics_error_rate_zero_requests():
    """Test error rate with zero requests"""
    metrics = HealthMetrics()

    assert metrics.error_rate == 0.0


def test_health_metrics_gpu_memory_percent_zero_total():
    """Test GPU memory percent with zero total"""
    metrics = HealthMetrics()
    metrics.gpu_memory_total_mb = 0

    assert metrics.gpu_memory_percent == 0.0


def test_health_metrics_is_healthy_all_checks_pass():
    """Test is_healthy when all checks pass"""
    metrics = HealthMetrics()
    metrics.memory_percent = 80.0
    metrics.gpu_memory_used_mb = 18000
    metrics.gpu_memory_total_mb = 24000  # 75%
    metrics.success_count = 90
    metrics.error_count = 10  # 10% error rate

    assert metrics.is_healthy() is True


def test_health_metrics_is_healthy_memory_critical():
    """Test is_healthy when memory is critical"""
    metrics = HealthMetrics()
    metrics.memory_percent = 96.0  # Critical

    assert metrics.is_healthy() is False


def test_health_metrics_is_healthy_gpu_critical():
    """Test is_healthy when GPU memory is critical"""
    metrics = HealthMetrics()
    metrics.gpu_memory_used_mb = 23000
    metrics.gpu_memory_total_mb = 24000  # 95.8%

    assert metrics.is_healthy() is False


def test_health_metrics_is_healthy_error_rate_high():
    """Test is_healthy when error rate is high"""
    metrics = HealthMetrics()
    metrics.success_count = 30
    metrics.error_count = 70  # 70% error rate

    assert metrics.is_healthy() is False
