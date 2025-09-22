"""Tests for Error Handling System"""

import pytest
import pytest_asyncio
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.error_handling import (
    ErrorHandler,
    ErrorRecord,
    ErrorContext,
    ErrorSeverity,
    ErrorCategory,
    RetryConfig,
    with_error_handling,
    with_retry,
    get_error_handler,
    configure_error_handler
)


class TestErrorHandler:
    """Test cases for error handler"""

    def setup_method(self):
        """Set up test fixtures"""
        self.error_handler = ErrorHandler()

    def test_error_handler_initialization(self):
        """Test error handler initialization"""
        assert len(self.error_handler.error_records) == 0
        assert len(self.error_handler.error_callbacks) == 0
        assert self.error_handler.max_error_records == 1000
        assert self.error_handler.error_retention_days == 30

    def test_error_handler_with_config(self):
        """Test error handler initialization with config"""
        config = {
            "max_error_records": 500,
            "error_retention_days": 7,
            "enable_notifications": False
        }
        handler = ErrorHandler(config)
        assert handler.max_error_records == 500
        assert handler.error_retention_days == 7
        assert handler.enable_error_notifications is False

    def test_handle_basic_error(self):
        """Test basic error handling"""
        error = ValueError("Test error")
        context = ErrorContext(
            component="test_component",
            operation="test_operation"
        )

        error_record = self.error_handler.handle_error(
            error, context, ErrorCategory.SYSTEM, ErrorSeverity.MEDIUM
        )

        assert error_record.category == ErrorCategory.SYSTEM
        assert error_record.severity == ErrorSeverity.MEDIUM
        assert error_record.message == "Test error"
        assert error_record.exception_type == "ValueError"
        assert error_record.occurrence_count == 1
        assert not error_record.resolved

    def test_handle_recurring_error(self):
        """Test handling of recurring errors"""
        error = ValueError("Recurring error")
        context = ErrorContext(
            component="test_component",
            operation="test_operation"
        )

        # Handle the same error multiple times
        record1 = self.error_handler.handle_error(
            error, context, ErrorCategory.SYSTEM, ErrorSeverity.MEDIUM
        )
        record2 = self.error_handler.handle_error(
            error, context, ErrorCategory.SYSTEM, ErrorSeverity.MEDIUM
        )

        # Should be the same record with updated count
        assert record1.error_id == record2.error_id
        assert record2.occurrence_count == 2
        assert len(self.error_handler.error_records) == 1

    def test_error_callbacks(self):
        """Test error callback registration and execution"""
        callback_mock = Mock()
        self.error_handler.register_error_callback(ErrorCategory.DATA_SOURCE, callback_mock)

        error = Exception("Test error")
        context = ErrorContext(component="test", operation="test")

        self.error_handler.handle_error(
            error, context, ErrorCategory.DATA_SOURCE, ErrorSeverity.HIGH
        )

        callback_mock.assert_called_once()
        assert isinstance(callback_mock.call_args[0][0], ErrorRecord)

    def test_error_statistics(self):
        """Test error statistics collection"""
        # Generate some test errors
        errors = [
            (ValueError("Error 1"), ErrorCategory.SYSTEM, ErrorSeverity.LOW),
            (RuntimeError("Error 2"), ErrorCategory.DATA_SOURCE, ErrorSeverity.HIGH),
            (KeyError("Error 3"), ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM),
            (ValueError("Error 4"), ErrorCategory.SYSTEM, ErrorSeverity.CRITICAL),
        ]

        for error, category, severity in errors:
            context = ErrorContext(component="test", operation="test")
            self.error_handler.handle_error(error, context, category, severity)

        stats = self.error_handler.get_error_statistics()

        assert stats["total_errors"] == 4
        assert stats["unique_errors"] == 4
        assert "system" in stats["error_by_category"]
        assert "data_source" in stats["error_by_category"]
        assert len(stats["recent_errors"]) == 4

    def test_mark_error_resolved(self):
        """Test marking errors as resolved"""
        error = ValueError("Test error")
        context = ErrorContext(component="test", operation="test")

        error_record = self.error_handler.handle_error(
            error, context, ErrorCategory.SYSTEM, ErrorSeverity.MEDIUM
        )

        assert not error_record.resolved

        self.error_handler.mark_error_resolved(
            error_record.error_id, "Fixed by updating configuration"
        )

        resolved_record = self.error_handler.get_error_record(error_record.error_id)
        assert resolved_record.resolved
        assert resolved_record.resolution_notes == "Fixed by updating configuration"

    def test_export_error_report(self):
        """Test error report export"""
        # Create some test errors
        error1 = ValueError("Error 1")
        error2 = RuntimeError("Error 2")
        context = ErrorContext(component="test", operation="test")

        record1 = self.error_handler.handle_error(
            error1, context, ErrorCategory.SYSTEM, ErrorSeverity.MEDIUM
        )
        record2 = self.error_handler.handle_error(
            error2, context, ErrorCategory.DATA_SOURCE, ErrorSeverity.HIGH
        )

        # Mark one as resolved
        self.error_handler.mark_error_resolved(record1.error_id, "Test resolution")

        # Export report excluding resolved errors
        report = self.error_handler.export_error_report(include_resolved=False)
        assert len(report["errors"]) == 1  # Only unresolved error

        # Export report including resolved errors
        report_with_resolved = self.error_handler.export_error_report(include_resolved=True)
        assert len(report_with_resolved["errors"]) == 2  # Both errors

    def test_error_cleanup(self):
        """Test cleanup of old error records"""
        # Set up handler with small limits for testing
        config = {"max_error_records": 3, "error_retention_days": 1}
        handler = ErrorHandler(config)

        # Create some errors
        for i in range(5):
            error = ValueError(f"Error {i}")
            context = ErrorContext(component="test", operation="test")
            handler.handle_error(error, context, ErrorCategory.SYSTEM, ErrorSeverity.MEDIUM)

        # Should have triggered cleanup
        assert len(handler.error_records) <= 3


class TestErrorContext:
    """Test cases for error context"""

    def test_error_context_creation(self):
        """Test error context creation"""
        context = ErrorContext(
            component="data_source",
            operation="fetch_price",
            user_id="user123",
            request_id="req456",
            additional_data={"symbol": "AAPL"}
        )

        assert context.component == "data_source"
        assert context.operation == "fetch_price"
        assert context.user_id == "user123"
        assert context.request_id == "req456"
        assert context.additional_data["symbol"] == "AAPL"
        assert isinstance(context.timestamp, datetime)

    def test_error_context_defaults(self):
        """Test error context with default values"""
        context = ErrorContext(component="test", operation="test")

        assert context.user_id is None
        assert context.request_id is None
        assert len(context.additional_data) == 0


class TestRetryConfig:
    """Test cases for retry configuration"""

    def test_retry_config_defaults(self):
        """Test retry config with default values"""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_backoff is True
        assert config.jitter is True

    def test_retry_config_custom(self):
        """Test retry config with custom values"""
        config = RetryConfig(
            max_attempts=5,
            base_delay=2.0,
            max_delay=120.0,
            exponential_backoff=False,
            jitter=False
        )

        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.exponential_backoff is False
        assert config.jitter is False


class TestErrorHandlingDecorators:
    """Test cases for error handling decorators"""

    def setup_method(self):
        """Set up test fixtures"""
        self.error_handler = ErrorHandler()

    def test_with_error_handling_decorator(self):
        """Test error handling decorator"""

        @with_error_handling(
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH,
            should_raise=False,
            error_handler=self.error_handler
        )
        def failing_function():
            raise ValueError("Test error")

        # Should not raise, but should record error
        result = failing_function()
        assert result is None
        assert len(self.error_handler.error_records) == 1

    @pytest.mark.asyncio
    async def test_with_error_handling_decorator_async(self):
        """Test error handling decorator with async function"""

        @with_error_handling(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            should_raise=False,
            error_handler=self.error_handler
        )
        async def failing_async_function():
            raise ConnectionError("Network error")

        # Should not raise, but should record error
        result = await failing_async_function()
        assert result is None
        assert len(self.error_handler.error_records) == 1

    @pytest.mark.asyncio
    async def test_with_retry_decorator_success(self):
        """Test retry decorator with eventual success"""
        attempt_count = 0

        @with_retry(
            retry_config=RetryConfig(max_attempts=3, base_delay=0.1),
            retryable_exceptions=(ValueError,)
        )
        async def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = await flaky_function()
        assert result == "success"
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_with_retry_decorator_failure(self):
        """Test retry decorator with persistent failure"""

        @with_retry(
            retry_config=RetryConfig(max_attempts=2, base_delay=0.1),
            retryable_exceptions=(ValueError,)
        )
        async def always_failing_function():
            raise ValueError("Persistent failure")

        with pytest.raises(ValueError, match="Persistent failure"):
            await always_failing_function()

    def test_with_retry_decorator_sync(self):
        """Test retry decorator with synchronous function"""
        attempt_count = 0

        @with_retry(
            retry_config=RetryConfig(max_attempts=3, base_delay=0.1),
            retryable_exceptions=(ValueError,)
        )
        def flaky_sync_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = flaky_sync_function()
        assert result == "success"
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_with_retry_non_retryable_exception(self):
        """Test retry decorator with non-retryable exception"""

        @with_retry(
            retry_config=RetryConfig(max_attempts=3, base_delay=0.1),
            retryable_exceptions=(ValueError,)
        )
        async def function_with_non_retryable_error():
            raise TypeError("This should not be retried")

        with pytest.raises(TypeError, match="This should not be retried"):
            await function_with_non_retryable_error()


class TestGlobalErrorHandler:
    """Test cases for global error handler functions"""

    def test_get_error_handler(self):
        """Test getting global error handler"""
        handler = get_error_handler()
        assert isinstance(handler, ErrorHandler)

    def test_configure_error_handler(self):
        """Test configuring global error handler"""
        config = {"max_error_records": 500}
        handler = configure_error_handler(config)

        assert isinstance(handler, ErrorHandler)
        assert handler.max_error_records == 500

        # Verify it's the same as the global instance
        global_handler = get_error_handler()
        assert handler is global_handler


class TestErrorSeverityAndCategory:
    """Test cases for error severity and category enums"""

    def test_error_severity_values(self):
        """Test error severity enum values"""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"

    def test_error_category_values(self):
        """Test error category enum values"""
        expected_categories = [
            "data_source", "validation", "network", "configuration",
            "database", "api", "system", "authentication", "rate_limit"
        ]

        for category in ErrorCategory:
            assert category.value in expected_categories


class TestIntegrationScenarios:
    """Integration test scenarios"""

    def setup_method(self):
        """Set up test fixtures"""
        self.error_handler = ErrorHandler()

    def test_complex_error_scenario(self):
        """Test complex error handling scenario"""
        # Simulate a data source failure scenario
        context = ErrorContext(
            component="yfinance_source",
            operation="get_current_price",
            additional_data={"symbol": "AAPL", "retry_attempt": 1}
        )

        # First error: Rate limit
        rate_limit_error = Exception("Rate limit exceeded")
        record1 = self.error_handler.handle_error(
            rate_limit_error, context, ErrorCategory.RATE_LIMIT, ErrorSeverity.MEDIUM
        )

        # Second error: Network timeout
        network_error = Exception("Connection timeout")
        context.additional_data["retry_attempt"] = 2
        record2 = self.error_handler.handle_error(
            network_error, context, ErrorCategory.NETWORK, ErrorSeverity.HIGH
        )

        # Third error: Same rate limit (should increment count)
        record3 = self.error_handler.handle_error(
            rate_limit_error, context, ErrorCategory.RATE_LIMIT, ErrorSeverity.MEDIUM
        )

        # Verify records
        assert len(self.error_handler.error_records) == 2  # Two unique errors
        assert record1.error_id == record3.error_id  # Same error
        assert record3.occurrence_count == 2  # Incremented

        # Get statistics
        stats = self.error_handler.get_error_statistics()
        assert stats["total_errors"] == 3  # Total occurrences
        assert stats["unique_errors"] == 2  # Unique error types


if __name__ == "__main__":
    pytest.main([__file__])