"""Error Handling Manager

This module provides comprehensive error handling, logging, and recovery mechanisms
for the Market Data Agent. It includes structured error reporting, retry logic,
and centralized error management.
"""

import logging
import traceback
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, Union, List
from enum import Enum
from dataclasses import dataclass, field
import functools
import json

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification"""
    DATA_SOURCE = "data_source"
    VALIDATION = "validation"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    DATABASE = "database"
    API = "api"
    SYSTEM = "system"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"


@dataclass
class ErrorContext:
    """Context information for error tracking"""
    component: str
    operation: str
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorRecord:
    """Comprehensive error record"""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    context: ErrorContext
    exception_type: str
    traceback_info: Optional[str] = None
    resolved: bool = False
    resolution_notes: Optional[str] = None
    occurrence_count: int = 1
    first_occurred: datetime = field(default_factory=datetime.now)
    last_occurred: datetime = field(default_factory=datetime.now)


class RetryConfig:
    """Configuration for retry logic"""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_backoff: bool = True,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_backoff = exponential_backoff
        self.jitter = jitter


class ErrorHandler:
    """Centralized error handling and management"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize error handler

        Args:
            config: Configuration dictionary for error handling
        """
        self.config = config or {}
        self.error_records: Dict[str, ErrorRecord] = {}
        self.error_callbacks: Dict[ErrorCategory, List[Callable]] = {}

        # Configuration
        self.max_error_records = self.config.get("max_error_records", 1000)
        self.error_retention_days = self.config.get("error_retention_days", 30)
        self.enable_error_notifications = self.config.get("enable_notifications", True)

        # Error counters for different categories
        self.error_counters: Dict[str, Dict[str, int]] = {}

        logger.info("Error handler initialized")

    def register_error_callback(self, category: ErrorCategory, callback: Callable):
        """Register a callback for specific error categories

        Args:
            category: Error category to handle
            callback: Callback function to execute on error
        """
        if category not in self.error_callbacks:
            self.error_callbacks[category] = []
        self.error_callbacks[category].append(callback)

    def handle_error(
        self,
        error: Exception,
        context: ErrorContext,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        should_raise: bool = False
    ) -> ErrorRecord:
        """Handle an error with comprehensive logging and tracking

        Args:
            error: The exception that occurred
            context: Context information about the error
            category: Category of the error
            severity: Severity level of the error
            should_raise: Whether to re-raise the exception

        Returns:
            ErrorRecord object containing error details
        """
        # Generate unique error ID
        error_id = self._generate_error_id(error, context)

        # Check if this is a recurring error
        if error_id in self.error_records:
            error_record = self.error_records[error_id]
            error_record.occurrence_count += 1
            error_record.last_occurred = datetime.now()
        else:
            # Create new error record
            error_record = ErrorRecord(
                error_id=error_id,
                category=category,
                severity=severity,
                message=str(error),
                context=context,
                exception_type=type(error).__name__,
                traceback_info=traceback.format_exc() if logger.isEnabledFor(logging.DEBUG) else None
            )
            self.error_records[error_id] = error_record

        # Log the error
        self._log_error(error_record)

        # Update error counters
        self._update_error_counters(category, severity)

        # Execute callbacks for this error category
        self._execute_error_callbacks(category, error_record)

        # Clean up old error records if needed
        self._cleanup_old_errors()

        # Re-raise if requested
        if should_raise:
            raise error

        return error_record

    def _generate_error_id(self, error: Exception, context: ErrorContext) -> str:
        """Generate a unique ID for error tracking"""
        error_signature = f"{type(error).__name__}:{str(error)}:{context.component}:{context.operation}"
        return f"err_{hash(error_signature) % 1000000:06d}"

    def _log_error(self, error_record: ErrorRecord):
        """Log error with appropriate level"""
        log_message = (
            f"[{error_record.error_id}] {error_record.category.value.upper()}: "
            f"{error_record.message} in {error_record.context.component}.{error_record.context.operation}"
        )

        if error_record.occurrence_count > 1:
            log_message += f" (occurrence #{error_record.occurrence_count})"

        # Add context data if available
        if error_record.context.additional_data:
            log_message += f" - Context: {error_record.context.additional_data}"

        # Log with appropriate level based on severity
        if error_record.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_record.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_record.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)

        # Include traceback for debug level
        if error_record.traceback_info and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Traceback for {error_record.error_id}:\n{error_record.traceback_info}")

    def _update_error_counters(self, category: ErrorCategory, severity: ErrorSeverity):
        """Update error statistics"""
        category_key = category.value
        severity_key = severity.value

        if category_key not in self.error_counters:
            self.error_counters[category_key] = {}

        if severity_key not in self.error_counters[category_key]:
            self.error_counters[category_key][severity_key] = 0

        self.error_counters[category_key][severity_key] += 1

    def _execute_error_callbacks(self, category: ErrorCategory, error_record: ErrorRecord):
        """Execute registered callbacks for error category"""
        if category in self.error_callbacks:
            for callback in self.error_callbacks[category]:
                try:
                    callback(error_record)
                except Exception as e:
                    logger.error(f"Error callback failed: {e}")

    def _cleanup_old_errors(self):
        """Clean up old error records to prevent memory bloat"""
        if len(self.error_records) <= self.max_error_records:
            return

        # First try to remove old errors based on retention period
        cutoff_date = datetime.now() - timedelta(days=self.error_retention_days)
        old_error_ids = [
            error_id for error_id, record in self.error_records.items()
            if record.last_occurred < cutoff_date
        ]

        for error_id in old_error_ids:
            del self.error_records[error_id]

        # If we still have too many errors, remove the oldest ones
        if len(self.error_records) > self.max_error_records:
            # Sort by last_occurred and remove oldest
            sorted_records = sorted(
                self.error_records.items(),
                key=lambda x: x[1].last_occurred
            )

            num_to_remove = len(self.error_records) - self.max_error_records
            for error_id, _ in sorted_records[:num_to_remove]:
                del self.error_records[error_id]

        if old_error_ids or len(self.error_records) <= self.max_error_records:
            logger.info(f"Cleaned up error records, now have {len(self.error_records)} records")

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics

        Returns:
            Dictionary containing error statistics
        """
        total_errors = sum(
            sum(severities.values()) for severities in self.error_counters.values()
        )

        return {
            "total_errors": total_errors,
            "unique_errors": len(self.error_records),
            "error_by_category": self.error_counters.copy(),
            "recent_errors": [
                {
                    "error_id": record.error_id,
                    "category": record.category.value,
                    "severity": record.severity.value,
                    "message": record.message,
                    "occurrence_count": record.occurrence_count,
                    "last_occurred": record.last_occurred.isoformat()
                }
                for record in sorted(
                    self.error_records.values(),
                    key=lambda x: x.last_occurred,
                    reverse=True
                )[:10]
            ]
        }

    def get_error_record(self, error_id: str) -> Optional[ErrorRecord]:
        """Get specific error record by ID

        Args:
            error_id: Unique error identifier

        Returns:
            ErrorRecord if found, None otherwise
        """
        return self.error_records.get(error_id)

    def mark_error_resolved(self, error_id: str, resolution_notes: str = ""):
        """Mark an error as resolved

        Args:
            error_id: Unique error identifier
            resolution_notes: Notes about the resolution
        """
        if error_id in self.error_records:
            self.error_records[error_id].resolved = True
            self.error_records[error_id].resolution_notes = resolution_notes
            logger.info(f"Error {error_id} marked as resolved: {resolution_notes}")

    def export_error_report(self, include_resolved: bool = False) -> Dict[str, Any]:
        """Export comprehensive error report

        Args:
            include_resolved: Whether to include resolved errors

        Returns:
            Dictionary containing full error report
        """
        errors_to_include = [
            record for record in self.error_records.values()
            if include_resolved or not record.resolved
        ]

        return {
            "report_generated": datetime.now().isoformat(),
            "statistics": self.get_error_statistics(),
            "errors": [
                {
                    "error_id": record.error_id,
                    "category": record.category.value,
                    "severity": record.severity.value,
                    "message": record.message,
                    "component": record.context.component,
                    "operation": record.context.operation,
                    "exception_type": record.exception_type,
                    "occurrence_count": record.occurrence_count,
                    "first_occurred": record.first_occurred.isoformat(),
                    "last_occurred": record.last_occurred.isoformat(),
                    "resolved": record.resolved,
                    "resolution_notes": record.resolution_notes
                }
                for record in sorted(errors_to_include, key=lambda x: x.last_occurred, reverse=True)
            ]
        }


def with_error_handling(
    category: ErrorCategory,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    should_raise: bool = True,
    error_handler: Optional[ErrorHandler] = None
):
    """Decorator for automatic error handling

    Args:
        category: Error category for classification
        severity: Default severity level
        should_raise: Whether to re-raise exceptions
        error_handler: Specific error handler instance to use
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                handler = error_handler or global_error_handler
                context = ErrorContext(
                    component=func.__module__,
                    operation=func.__name__,
                    additional_data={"args": str(args), "kwargs": str(kwargs)}
                )
                handler.handle_error(e, context, category, severity, should_raise)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handler = error_handler or global_error_handler
                context = ErrorContext(
                    component=func.__module__,
                    operation=func.__name__,
                    additional_data={"args": str(args), "kwargs": str(kwargs)}
                )
                handler.handle_error(e, context, category, severity, should_raise)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def with_retry(
    retry_config: Optional[RetryConfig] = None,
    category: ErrorCategory = ErrorCategory.SYSTEM,
    retryable_exceptions: tuple = (Exception,)
):
    """Decorator for automatic retry logic with exponential backoff

    Args:
        retry_config: Retry configuration
        category: Error category for failed attempts
        retryable_exceptions: Tuple of exceptions that should trigger retries
    """
    config = retry_config or RetryConfig()

    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if attempt == config.max_attempts - 1:
                        # Final attempt failed, handle error and raise
                        context = ErrorContext(
                            component=func.__module__,
                            operation=func.__name__,
                            additional_data={
                                "attempt": attempt + 1,
                                "max_attempts": config.max_attempts
                            }
                        )
                        global_error_handler.handle_error(
                            e, context, category, ErrorSeverity.HIGH, should_raise=True
                        )

                    # Calculate delay for next attempt
                    if config.exponential_backoff:
                        delay = min(config.base_delay * (2 ** attempt), config.max_delay)
                    else:
                        delay = config.base_delay

                    if config.jitter:
                        import random
                        delay *= (0.5 + random.random())

                    logger.warning(
                        f"Attempt {attempt + 1}/{config.max_attempts} failed for "
                        f"{func.__name__}: {e}. Retrying in {delay:.2f}s"
                    )

                    await asyncio.sleep(delay)

            # This shouldn't be reached, but just in case
            raise last_exception

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if attempt == config.max_attempts - 1:
                        # Final attempt failed, handle error and raise
                        context = ErrorContext(
                            component=func.__module__,
                            operation=func.__name__,
                            additional_data={
                                "attempt": attempt + 1,
                                "max_attempts": config.max_attempts
                            }
                        )
                        global_error_handler.handle_error(
                            e, context, category, ErrorSeverity.HIGH, should_raise=True
                        )

                    # Calculate delay for next attempt
                    if config.exponential_backoff:
                        delay = min(config.base_delay * (2 ** attempt), config.max_delay)
                    else:
                        delay = config.base_delay

                    if config.jitter:
                        import random
                        delay *= (0.5 + random.random())

                    logger.warning(
                        f"Attempt {attempt + 1}/{config.max_attempts} failed for "
                        f"{func.__name__}: {e}. Retrying in {delay:.2f}s"
                    )

                    import time
                    time.sleep(delay)

            # This shouldn't be reached, but just in case
            raise last_exception

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# Global error handler instance
global_error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance

    Returns:
        Global ErrorHandler instance
    """
    return global_error_handler


def configure_error_handler(config: Dict[str, Any]) -> ErrorHandler:
    """Configure the global error handler

    Args:
        config: Configuration dictionary

    Returns:
        Configured ErrorHandler instance
    """
    global global_error_handler
    global_error_handler = ErrorHandler(config)
    return global_error_handler