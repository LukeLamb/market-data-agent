"""Error Handling Package

This package provides comprehensive error handling, logging, and recovery mechanisms
for the Market Data Agent.
"""

from .error_manager import (
    ErrorHandler,
    ErrorRecord,
    ErrorContext,
    ErrorSeverity,
    ErrorCategory,
    RetryConfig,
    with_error_handling,
    with_retry,
    get_error_handler,
    configure_error_handler,
    global_error_handler
)

__all__ = [
    "ErrorHandler",
    "ErrorRecord",
    "ErrorContext",
    "ErrorSeverity",
    "ErrorCategory",
    "RetryConfig",
    "with_error_handling",
    "with_retry",
    "get_error_handler",
    "configure_error_handler",
    "global_error_handler"
]