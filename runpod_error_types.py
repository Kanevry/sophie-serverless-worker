"""
RunPod Error Classification Module
Maps errors to recovery strategies for self-healing workers
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
import re


class RunPodErrorType(str, Enum):
    """RunPod-specific error types"""

    OOM_ERROR = "oom_error"
    VRAM_EXHAUSTED = "vram_exhausted"
    LLAMA_SERVER_CRASH = "llama_server_crash"
    LLAMA_SERVER_HANG = "llama_server_hang"
    MODEL_LOAD_FAILED = "model_load_failed"
    CONTEXT_OVERFLOW = "context_overflow"
    HEALTH_CHECK_FAIL = "health_check_fail"
    COLD_START_TIMEOUT = "cold_start_timeout"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class ErrorSeverity(str, Enum):
    """Error severity levels"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RecoveryAction(str, Enum):
    """Recovery action types"""

    RESTART = "restart"
    CLEANUP = "cleanup"
    RETRY = "retry"
    ESCALATE = "escalate"
    IGNORE = "ignore"


@dataclass
class ClassifiedRunPodError:
    """
    Classified error with recovery metadata

    Austrian Compliance (BAO §132):
    - All fields structured for audit logging
    - 7-year retention in RunPod Console logs
    """

    type: RunPodErrorType
    severity: ErrorSeverity
    recovery_action: RecoveryAction
    retryable: bool
    metadata: Dict[str, Any]
    original_error: Optional[Exception] = None

    def to_audit_log(self) -> Dict[str, Any]:
        """Format for BAO §132 audit logging"""
        return {
            "error_type": self.type.value,
            "severity": self.severity.value,
            "recovery_action": self.recovery_action.value,
            "retryable": self.retryable,
            "error_message": str(self.original_error) if self.original_error else None,
            **self.metadata,
        }


# Error classification patterns
ERROR_PATTERNS = {
    RunPodErrorType.OOM_ERROR: [
        r"MemoryError",
        r"out of memory",
        r"OOM",
        r"Cannot allocate memory",
    ],
    RunPodErrorType.VRAM_EXHAUSTED: [
        r"CUDA out of memory",
        r"CUDA_ERROR_OUT_OF_MEMORY",
        r"cuda runtime error \(2\)",
        r"GPU memory",
    ],
    RunPodErrorType.LLAMA_SERVER_CRASH: [
        r"Connection refused",
        r"llama.*terminated",
        r"Process.*exited",
        r"server.*crashed",
    ],
    RunPodErrorType.LLAMA_SERVER_HANG: [
        r"Request timeout",
        r"ReadTimeout",
        r"Connection timeout",
        r"llama.*not responding",
    ],
    RunPodErrorType.MODEL_LOAD_FAILED: [
        r"Failed to load model",
        r"Model file.*not found",
        r"GGUF.*invalid",
        r"model.*corrupt",
    ],
    RunPodErrorType.CONTEXT_OVERFLOW: [
        r"context length.*exceeded",
        r"prompt too long",
        r"exceeds.*context window",
        r"max context",
    ],
    RunPodErrorType.HEALTH_CHECK_FAIL: [
        r"Health check failed",
        r"/health.*error",
        r"Service unhealthy",
    ],
    RunPodErrorType.COLD_START_TIMEOUT: [
        r"Cold start.*timeout",
        r"Initialization.*timeout",
        r"Worker.*not ready",
    ],
    RunPodErrorType.TIMEOUT: [
        r"timeout",
        r"timed out",
        r"deadline exceeded",
    ],
}

# Recovery strategy mapping
RECOVERY_STRATEGIES = {
    RunPodErrorType.OOM_ERROR: (ErrorSeverity.HIGH, RecoveryAction.CLEANUP, True),
    RunPodErrorType.VRAM_EXHAUSTED: (ErrorSeverity.HIGH, RecoveryAction.CLEANUP, True),
    RunPodErrorType.LLAMA_SERVER_CRASH: (
        ErrorSeverity.CRITICAL,
        RecoveryAction.RESTART,
        True,
    ),
    RunPodErrorType.LLAMA_SERVER_HANG: (
        ErrorSeverity.CRITICAL,
        RecoveryAction.RESTART,
        True,
    ),
    RunPodErrorType.MODEL_LOAD_FAILED: (
        ErrorSeverity.CRITICAL,
        RecoveryAction.ESCALATE,
        False,
    ),
    RunPodErrorType.CONTEXT_OVERFLOW: (
        ErrorSeverity.MEDIUM,
        RecoveryAction.IGNORE,
        False,
    ),
    RunPodErrorType.HEALTH_CHECK_FAIL: (
        ErrorSeverity.HIGH,
        RecoveryAction.RETRY,
        True,
    ),
    RunPodErrorType.COLD_START_TIMEOUT: (
        ErrorSeverity.MEDIUM,
        RecoveryAction.ESCALATE,
        False,
    ),
    RunPodErrorType.TIMEOUT: (ErrorSeverity.MEDIUM, RecoveryAction.RETRY, True),
    RunPodErrorType.UNKNOWN: (ErrorSeverity.HIGH, RecoveryAction.ESCALATE, False),
}


def classify_runpod_error(
    error: Exception, metadata: Optional[Dict[str, Any]] = None
) -> ClassifiedRunPodError:
    """
    Classify error and map to recovery strategy

    Args:
        error: Exception to classify
        metadata: Additional context (GPU%, Memory%, etc.)

    Returns:
        ClassifiedRunPodError with recovery strategy
    """
    error_msg = str(error)
    error_type = RunPodErrorType.UNKNOWN

    # Pattern matching
    for err_type, patterns in ERROR_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, error_msg, re.IGNORECASE):
                error_type = err_type
                break
        if error_type != RunPodErrorType.UNKNOWN:
            break

    # Get recovery strategy
    severity, recovery_action, retryable = RECOVERY_STRATEGIES[error_type]

    return ClassifiedRunPodError(
        type=error_type,
        severity=severity,
        recovery_action=recovery_action,
        retryable=retryable,
        metadata=metadata or {},
        original_error=error,
    )
