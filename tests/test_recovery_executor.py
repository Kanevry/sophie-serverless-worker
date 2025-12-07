"""
Unit Tests for RecoveryExecutor
Tests recovery actions for RunPod serverless workers
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import subprocess

import sys
sys.path.insert(0, '/Users/bernhardgoetzendorfer/Projects/BuchhaltGenieV5/local-ai/runpod-serverless')

from recovery_executor import RecoveryExecutor, RecoveryResult
from runpod_error_types import RunPodErrorType


@pytest.fixture
def executor():
    """Create RecoveryExecutor instance"""
    return RecoveryExecutor(
        llama_server_port=8080,
        max_restart_attempts=3,
        restart_delay_seconds=1,  # Short delay for tests
    )


@pytest.mark.asyncio
async def test_gpu_cleanup_success(executor):
    """Test GPU memory cleanup success"""
    with patch('recovery_executor.TORCH_AVAILABLE', True):
        with patch('recovery_executor.torch.cuda.is_available', return_value=True):
            with patch('recovery_executor.torch.cuda.empty_cache') as mock_empty_cache:
                result = await executor._cleanup_gpu_memory()

                assert result.success is True
                assert result.action_taken == "gpu_cleanup"
                mock_empty_cache.assert_called_once()


@pytest.mark.asyncio
async def test_gpu_cleanup_no_cuda(executor):
    """Test GPU cleanup when CUDA not available"""
    with patch('recovery_executor.TORCH_AVAILABLE', False):
        result = await executor._cleanup_gpu_memory()

        assert result.success is False
        assert "PyTorch/CUDA not available" in result.error


@pytest.mark.asyncio
async def test_system_memory_cleanup(executor):
    """Test system memory cleanup"""
    with patch('gc.collect', return_value=42) as mock_gc:
        result = await executor._cleanup_system_memory()

        assert result.success is True
        assert result.action_taken == "system_memory_cleanup"
        mock_gc.assert_called_once()


@pytest.mark.asyncio
async def test_restart_llama_server_success(executor):
    """Test graceful llama-server restart"""
    # Mock finding PID
    with patch.object(executor, '_find_llama_server_pid', return_value=12345):
        # Mock subprocess.run for kill
        with patch('subprocess.run') as mock_run:
            # Mock subprocess.Popen for restart
            with patch('subprocess.Popen') as mock_popen:
                # Mock health check
                with patch.object(executor, '_check_server_health', return_value=True):
                    result = await executor._restart_llama_server()

                    assert result.success is True
                    assert result.action_taken == "graceful_restart"

                    # Verify SIGTERM was sent
                    mock_run.assert_called_once()
                    args = mock_run.call_args[0][0]
                    assert "kill" in args
                    assert "-15" in args
                    assert "12345" in args

                    # Verify server was restarted
                    mock_popen.assert_called_once()


@pytest.mark.asyncio
async def test_restart_llama_server_no_pid(executor):
    """Test restart when no PID found"""
    with patch.object(executor, '_find_llama_server_pid', return_value=None):
        with patch('subprocess.Popen'):
            with patch.object(executor, '_check_server_health', return_value=True):
                result = await executor._restart_llama_server()

                # Should still succeed (start new server)
                assert result.success is True


@pytest.mark.asyncio
async def test_restart_llama_server_unhealthy(executor):
    """Test restart failure (server unhealthy after restart)"""
    with patch.object(executor, '_find_llama_server_pid', return_value=None):
        with patch('subprocess.Popen'):
            with patch.object(executor, '_check_server_health', return_value=False):
                result = await executor._restart_llama_server()

                assert result.success is False
                assert "unhealthy" in result.error.lower()


@pytest.mark.asyncio
async def test_force_restart_server(executor):
    """Test force restart with SIGKILL"""
    with patch.object(executor, '_find_llama_server_pid', return_value=99999):
        with patch('subprocess.run') as mock_run:
            with patch('subprocess.Popen'):
                with patch.object(executor, '_check_server_health', return_value=True):
                    result = await executor._force_restart_server()

                    assert result.success is True
                    assert result.action_taken == "force_restart"

                    # Verify SIGKILL was sent
                    args = mock_run.call_args[0][0]
                    assert "kill" in args
                    assert "-9" in args


@pytest.mark.asyncio
async def test_retry_health_check_success_first_attempt(executor):
    """Test health check retry succeeds on first attempt"""
    with patch.object(executor, '_check_server_health', return_value=True):
        result = await executor._retry_health_check()

        assert result.success is True
        assert result.action_taken == "health_check_retry"
        assert result.metadata['attempts'] == 1


@pytest.mark.asyncio
async def test_retry_health_check_success_third_attempt(executor):
    """Test health check retry succeeds on third attempt"""
    # Fail twice, succeed on third
    health_checks = [False, False, True]

    with patch.object(executor, '_check_server_health', side_effect=health_checks):
        result = await executor._retry_health_check()

        assert result.success is True
        assert result.metadata['attempts'] == 3


@pytest.mark.asyncio
async def test_retry_health_check_all_fail(executor):
    """Test health check retry fails all attempts"""
    with patch.object(executor, '_check_server_health', return_value=False):
        result = await executor._retry_health_check()

        assert result.success is False
        assert "Failed after 3 attempts" in result.error


@pytest.mark.asyncio
async def test_execute_recovery_vram_exhausted(executor):
    """Test execute_recovery for VRAM exhaustion"""
    with patch.object(executor, '_cleanup_gpu_memory', return_value=RecoveryResult(
        success=True,
        action_taken="gpu_cleanup"
    )) as mock_cleanup:
        with patch.object(executor, '_get_system_metrics', return_value={}):
            result = await executor.execute_recovery(RunPodErrorType.VRAM_EXHAUSTED)

            assert result.success is True
            mock_cleanup.assert_called_once()
            assert executor.successful_recoveries == 1


@pytest.mark.asyncio
async def test_execute_recovery_oom_error(executor):
    """Test execute_recovery for OOM error"""
    with patch.object(executor, '_cleanup_system_memory', return_value=RecoveryResult(
        success=True,
        action_taken="system_memory_cleanup"
    )) as mock_cleanup:
        with patch.object(executor, '_get_system_metrics', return_value={}):
            result = await executor.execute_recovery(RunPodErrorType.OOM_ERROR)

            assert result.success is True
            mock_cleanup.assert_called_once()


@pytest.mark.asyncio
async def test_execute_recovery_server_crash(executor):
    """Test execute_recovery for server crash"""
    with patch.object(executor, '_restart_llama_server', return_value=RecoveryResult(
        success=True,
        action_taken="graceful_restart"
    )) as mock_restart:
        with patch.object(executor, '_get_system_metrics', return_value={}):
            result = await executor.execute_recovery(RunPodErrorType.LLAMA_SERVER_CRASH)

            assert result.success is True
            mock_restart.assert_called_once()


@pytest.mark.asyncio
async def test_execute_recovery_tracks_metrics(executor):
    """Test that execute_recovery tracks metrics correctly"""
    metrics_before = {
        "gpu_reserved_mb": 20000,
        "system_memory_mb": 30000,
    }
    metrics_after = {
        "gpu_reserved_mb": 10000,
        "system_memory_mb": 28000,
    }

    with patch.object(executor, '_cleanup_gpu_memory', return_value=RecoveryResult(
        success=True,
        action_taken="gpu_cleanup"
    )):
        with patch.object(executor, '_get_system_metrics', side_effect=[metrics_before, metrics_after]):
            result = await executor.execute_recovery(RunPodErrorType.VRAM_EXHAUSTED)

            assert result.metrics_before == metrics_before
            assert result.metrics_after == metrics_after
            assert result.freed_mb == 12000  # 10000 GPU + 2000 system
            assert result.duration_ms > 0


@pytest.mark.asyncio
async def test_execute_recovery_failure_increments_counter(executor):
    """Test that failed recovery increments failure counter"""
    with patch.object(executor, '_cleanup_gpu_memory', return_value=RecoveryResult(
        success=False,
        action_taken="gpu_cleanup",
        error="Mock failure"
    )):
        with patch.object(executor, '_get_system_metrics', return_value={}):
            result = await executor.execute_recovery(RunPodErrorType.VRAM_EXHAUSTED)

            assert result.success is False
            assert executor.failed_recoveries == 1
            assert executor.successful_recoveries == 0


def test_find_llama_server_pid_found(executor):
    """Test finding llama-server PID when it exists"""
    with patch('subprocess.run', return_value=Mock(returncode=0, stdout="12345\n67890\n")):
        pid = executor._find_llama_server_pid()

        assert pid == 12345


def test_find_llama_server_pid_not_found(executor):
    """Test finding llama-server PID when not running"""
    with patch('subprocess.run', return_value=Mock(returncode=1, stdout="")):
        pid = executor._find_llama_server_pid()

        assert pid is None


@pytest.mark.asyncio
async def test_check_server_health_success(executor):
    """Test server health check success"""
    with patch('subprocess.run', return_value=Mock(returncode=0)):
        is_healthy = await executor._check_server_health()

        assert is_healthy is True


@pytest.mark.asyncio
async def test_check_server_health_failure(executor):
    """Test server health check failure"""
    with patch('subprocess.run', return_value=Mock(returncode=1)):
        is_healthy = await executor._check_server_health()

        assert is_healthy is False


def test_get_system_metrics_with_torch(executor):
    """Test getting system metrics with PyTorch available"""
    with patch('recovery_executor.TORCH_AVAILABLE', True):
        with patch('recovery_executor.torch.cuda.is_available', return_value=True):
            with patch('recovery_executor.torch.cuda.memory_allocated', return_value=5000 * 1024 * 1024):
                with patch('recovery_executor.torch.cuda.memory_reserved', return_value=10000 * 1024 * 1024):
                    with patch('recovery_executor.torch.cuda.get_device_properties') as mock_props:
                        mock_props.return_value = Mock(total_memory=24000 * 1024 * 1024)

                        with patch('recovery_executor.PSUTIL_AVAILABLE', True):
                            with patch('recovery_executor.psutil.virtual_memory') as mock_mem:
                                mock_mem.return_value = Mock(used=16000 * 1024 * 1024, percent=75.0)

                                with patch('recovery_executor.psutil.cpu_percent', return_value=45.5):
                                    metrics = executor._get_system_metrics()

                                    assert metrics['gpu_allocated_mb'] == 5000
                                    assert metrics['gpu_reserved_mb'] == 10000
                                    assert metrics['gpu_total_mb'] == 24000
                                    assert metrics['system_memory_percent'] == 75.0
                                    assert metrics['cpu_percent'] == 45.5


def test_calculate_freed_memory(executor):
    """Test freed memory calculation"""
    before = {
        "gpu_reserved_mb": 20000,
        "system_memory_mb": 30000,
    }
    after = {
        "gpu_reserved_mb": 15000,
        "system_memory_mb": 28000,
    }

    freed = executor._calculate_freed_memory(before, after)

    assert freed == 7000  # 5000 GPU + 2000 system


def test_calculate_freed_memory_negative(executor):
    """Test freed memory calculation doesn't return negative"""
    before = {
        "gpu_reserved_mb": 10000,
    }
    after = {
        "gpu_reserved_mb": 15000,  # Used MORE memory
    }

    freed = executor._calculate_freed_memory(before, after)

    assert freed == 0.0  # Never negative


def test_get_recovery_stats(executor):
    """Test getting recovery statistics"""
    executor.total_recoveries = 10
    executor.successful_recoveries = 8
    executor.failed_recoveries = 2

    stats = executor.get_recovery_stats()

    assert stats['total_recoveries'] == 10
    assert stats['successful_recoveries'] == 8
    assert stats['failed_recoveries'] == 2
    assert stats['success_rate_percent'] == 80.0


def test_get_recovery_stats_zero_recoveries(executor):
    """Test recovery stats with zero recoveries"""
    stats = executor.get_recovery_stats()

    assert stats['success_rate_percent'] == 0.0
