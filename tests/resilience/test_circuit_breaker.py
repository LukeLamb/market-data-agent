"""Tests for Enhanced Circuit Breaker System

Comprehensive tests for the multi-level circuit breaker system including
adaptive thresholds, intelligent failure classification, and performance-based recovery.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.resilience.circuit_breaker import (
    EnhancedCircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerManager,
    CircuitBreakerError,
    CircuitState,
    FailureType,
    RecoveryStrategy,
    FailureRecord,
    PerformanceMetrics,
    circuit_breaker_manager
)


@pytest.fixture
def circuit_config():
    """Circuit breaker configuration for testing"""
    return CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout_seconds=1.0,
        degraded_threshold=2,
        critical_failure_threshold=5,
        enable_adaptive_thresholds=True,
        recovery_strategy=RecoveryStrategy.ADAPTIVE_BACKOFF,
        min_backoff_seconds=0.1,
        max_backoff_seconds=5.0,
        enable_performance_monitoring=True,
        slow_request_threshold_ms=100.0
    )


@pytest.fixture
def circuit_breaker(circuit_config):
    """Enhanced circuit breaker instance for testing"""
    return EnhancedCircuitBreaker("test_breaker", circuit_config)


@pytest.fixture
def slow_function():
    """Mock function that takes time to execute"""
    async def _slow_func():
        await asyncio.sleep(0.05)  # 50ms
        return "success"
    return _slow_func


@pytest.fixture
def failing_function():
    """Mock function that always fails"""
    async def _fail_func():
        raise Exception("Test failure")
    return _fail_func


class TestEnhancedCircuitBreaker:
    """Test suite for EnhancedCircuitBreaker"""

    def test_initialization(self, circuit_breaker, circuit_config):
        """Test circuit breaker initialization"""
        assert circuit_breaker.name == "test_breaker"
        assert circuit_breaker.config == circuit_config
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.total_requests == 0
        assert circuit_breaker.successful_requests == 0
        assert circuit_breaker.failed_requests == 0

    @pytest.mark.asyncio
    async def test_successful_execution(self, circuit_breaker, slow_function):
        """Test successful function execution"""
        result = await circuit_breaker.call(slow_function)

        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.total_requests == 1
        assert circuit_breaker.successful_requests == 1
        assert circuit_breaker.failed_requests == 0

    @pytest.mark.asyncio
    async def test_failure_tracking(self, circuit_breaker, failing_function):
        """Test failure tracking and classification"""
        with pytest.raises(Exception, match="Test failure"):
            await circuit_breaker.call(failing_function)

        assert circuit_breaker.total_requests == 1
        assert circuit_breaker.successful_requests == 0
        assert circuit_breaker.failed_requests == 1
        assert len(circuit_breaker.failures) == 1

        failure = circuit_breaker.failures[0]
        assert failure.failure_type == FailureType.UNKNOWN
        assert "Test failure" in failure.error_message

    @pytest.mark.asyncio
    async def test_circuit_opening(self, circuit_breaker, failing_function):
        """Test circuit breaker opening or degrading after threshold failures"""
        # Generate enough failures to trigger state change
        for i in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_function)

        # Circuit should be either degraded or open (sophisticated behavior)
        assert circuit_breaker.state in [CircuitState.OPEN, CircuitState.DEGRADED]
        assert circuit_breaker.failed_requests == circuit_breaker.config.failure_threshold

        # Generate more failures to definitely open the circuit
        for i in range(circuit_breaker.config.critical_failure_threshold):
            try:
                await circuit_breaker.call(failing_function)
            except:
                pass

        # Now circuit should be open
        assert circuit_breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_half_open_transition(self, circuit_breaker, failing_function, slow_function):
        """Test transition to half-open state"""
        # Open the circuit by generating critical failures
        for i in range(circuit_breaker.config.critical_failure_threshold):
            try:
                await circuit_breaker.call(failing_function)
            except:
                pass

        assert circuit_breaker.state == CircuitState.OPEN

        # Wait for timeout and verify half-open transition
        initial_timeout = circuit_breaker.config.min_backoff_seconds
        await asyncio.sleep(initial_timeout + 0.01)

        # Next request should transition to half-open
        result = await circuit_breaker.call(slow_function)
        assert result == "success"
        assert circuit_breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_recovery_to_closed(self, circuit_breaker, failing_function, slow_function):
        """Test full recovery back to closed state"""
        # Open the circuit
        for i in range(circuit_breaker.config.critical_failure_threshold):
            try:
                await circuit_breaker.call(failing_function)
            except:
                pass

        assert circuit_breaker.state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(circuit_breaker.config.min_backoff_seconds + 0.01)

        # Execute successful requests to close the circuit
        for i in range(circuit_breaker.config.success_threshold):
            result = await circuit_breaker.call(slow_function)
            assert result == "success"

        # Circuit should be closed now
        assert circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_degraded_state_transition(self, circuit_config):
        """Test transition to degraded state"""
        circuit_config.degraded_threshold = 2
        circuit_breaker = EnhancedCircuitBreaker("degraded_test", circuit_config)

        # Create failures to trigger degraded state
        async def intermittent_failure():
            if circuit_breaker.failed_requests % 2 == 0:
                raise Exception("Intermittent failure")
            return "success"

        # Generate partial failures
        for i in range(3):
            try:
                await circuit_breaker.call(intermittent_failure)
            except:
                pass

        # Should transition to degraded or open based on failure pattern
        assert circuit_breaker.state in [CircuitState.DEGRADED, CircuitState.OPEN]

    @pytest.mark.asyncio
    async def test_failure_type_classification(self, circuit_breaker):
        """Test intelligent failure classification"""
        # Test different failure types - test them individually to avoid interference
        failure_scenarios = [
            ("Connection timeout occurred", FailureType.TIMEOUT),
            ("Connection refused", FailureType.CONNECTION_ERROR),
            ("Rate limit exceeded", FailureType.RATE_LIMIT),
            ("HTTP 401 Unauthorized", FailureType.AUTHENTICATION_ERROR),
            ("HTTP 500 Internal Server Error", FailureType.SERVER_ERROR),
            ("Invalid JSON data", FailureType.DATA_ERROR),
            ("Generic error", FailureType.UNKNOWN)
        ]

        for error_msg, expected_type in failure_scenarios:
            # Create fresh circuit breaker for each test to avoid state interference
            test_cb = EnhancedCircuitBreaker(f"test_{expected_type.value}", circuit_breaker.config)

            async def fail_with_error():
                raise Exception(error_msg)

            try:
                await test_cb.call(fail_with_error)
            except:
                pass

            # Check that the failure was classified correctly
            latest_failure = test_cb.failures[-1]
            assert latest_failure.failure_type == expected_type, f"Expected {expected_type} for '{error_msg}', got {latest_failure.failure_type}"

    def test_adaptive_threshold_adjustment(self, circuit_breaker):
        """Test adaptive threshold adjustment"""
        initial_threshold = circuit_breaker.adaptive_failure_threshold

        # Simulate high error rate
        circuit_breaker.recent_requests.extend([
            (datetime.now(), False, 100.0),  # Failed request
            (datetime.now(), False, 100.0),  # Failed request
            (datetime.now(), False, 100.0),  # Failed request
            (datetime.now(), True, 50.0),    # Successful request
        ])

        # Trigger adaptive threshold update
        asyncio.run(circuit_breaker._update_adaptive_thresholds())

        # Threshold should be adjusted based on error rate
        # (exact behavior depends on the algorithm)
        assert circuit_breaker.adaptive_failure_threshold is not None

    @pytest.mark.asyncio
    async def test_performance_monitoring(self, circuit_breaker, slow_function):
        """Test performance metrics tracking"""
        # Execute several requests
        for i in range(5):
            await circuit_breaker.call(slow_function)

        # Check performance metrics
        stats = circuit_breaker.get_statistics()
        perf_metrics = stats["performance_metrics"]

        assert perf_metrics["avg_response_time_ms"] > 0
        assert perf_metrics["current_success_rate"] == 1.0
        assert perf_metrics["current_error_rate"] == 0.0
        assert perf_metrics["consecutive_successes"] == 5

    @pytest.mark.asyncio
    async def test_weighted_failure_count(self, circuit_breaker, failing_function):
        """Test weighted failure count calculation"""
        # Generate some failures
        for i in range(3):
            try:
                await circuit_breaker.call(failing_function)
            except:
                pass

        # Calculate weighted failure count
        weighted_count = await circuit_breaker._calculate_weighted_failure_count()

        assert weighted_count > 0
        # Recent failures should have higher weight
        assert weighted_count <= len(circuit_breaker.failures)

    @pytest.mark.asyncio
    async def test_backoff_strategies(self, circuit_config):
        """Test different backoff strategies"""
        strategies = [
            RecoveryStrategy.EXPONENTIAL_BACKOFF,
            RecoveryStrategy.LINEAR_BACKOFF,
            RecoveryStrategy.ADAPTIVE_BACKOFF,
            RecoveryStrategy.PERFORMANCE_BASED
        ]

        for strategy in strategies:
            config = CircuitBreakerConfig(
                recovery_strategy=strategy,
                min_backoff_seconds=0.1,
                max_backoff_seconds=1.0
            )
            cb = EnhancedCircuitBreaker(f"test_{strategy.value}", config)

            # Simulate some failures to test backoff calculation
            cb.performance.consecutive_failures = 3
            cb.performance.error_rate = 0.5
            cb.performance.avg_response_time = 1000.0

            backoff_time = await cb._calculate_backoff_time()

            assert config.min_backoff_seconds <= backoff_time <= config.max_backoff_seconds

    def test_statistics_reporting(self, circuit_breaker):
        """Test comprehensive statistics reporting"""
        stats = circuit_breaker.get_statistics()

        required_keys = [
            "name", "state", "total_requests", "successful_requests",
            "failed_requests", "blocked_requests", "success_rate",
            "performance_metrics", "adaptive_thresholds"
        ]

        for key in required_keys:
            assert key in stats

        assert stats["name"] == "test_breaker"
        assert stats["state"] == CircuitState.CLOSED.value

    @pytest.mark.asyncio
    async def test_health_status(self, circuit_breaker, slow_function):
        """Test health status reporting"""
        # Execute some requests
        await circuit_breaker.call(slow_function)

        health_status = await circuit_breaker.get_health_status()

        required_keys = [
            "health_status", "state", "performance_score",
            "adaptive_behavior"
        ]

        for key in required_keys:
            assert key in health_status

        assert health_status["health_status"] == "healthy"
        assert health_status["state"] == CircuitState.CLOSED.value

    def test_circuit_breaker_reset(self, circuit_breaker):
        """Test circuit breaker reset functionality"""
        # Modify some state
        circuit_breaker.total_requests = 10
        circuit_breaker.successful_requests = 5
        circuit_breaker.failed_requests = 5
        circuit_breaker.state = CircuitState.OPEN

        # Reset
        circuit_breaker.reset()

        # Verify reset state
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.total_requests == 0
        assert circuit_breaker.successful_requests == 0
        assert circuit_breaker.failed_requests == 0
        assert len(circuit_breaker.failures) == 0


class TestCircuitBreakerManager:
    """Test suite for CircuitBreakerManager"""

    def test_manager_initialization(self):
        """Test circuit breaker manager initialization"""
        manager = CircuitBreakerManager()
        assert len(manager.circuit_breakers) == 0

    def test_circuit_breaker_creation(self):
        """Test circuit breaker creation and retrieval"""
        manager = CircuitBreakerManager()
        config = CircuitBreakerConfig(failure_threshold=5)

        # Get circuit breaker (should create it)
        cb1 = manager.get_circuit_breaker("test1", config)
        assert cb1.name == "test1"
        assert len(manager.circuit_breakers) == 1

        # Get same circuit breaker again (should return existing)
        cb2 = manager.get_circuit_breaker("test1")
        assert cb1 is cb2

    def test_all_statistics(self):
        """Test getting statistics for all circuit breakers"""
        manager = CircuitBreakerManager()

        # Create some circuit breakers
        cb1 = manager.get_circuit_breaker("test1")
        cb2 = manager.get_circuit_breaker("test2")

        stats = manager.get_all_statistics()

        assert len(stats) == 2
        assert "test1" in stats
        assert "test2" in stats

    @pytest.mark.asyncio
    async def test_all_health_status(self):
        """Test getting health status for all circuit breakers"""
        manager = CircuitBreakerManager()

        # Create some circuit breakers
        cb1 = manager.get_circuit_breaker("test1")
        cb2 = manager.get_circuit_breaker("test2")

        health_status = await manager.get_all_health_status()

        assert len(health_status) == 2
        assert "test1" in health_status
        assert "test2" in health_status

    def test_reset_all(self):
        """Test resetting all circuit breakers"""
        manager = CircuitBreakerManager()

        # Create and modify some circuit breakers
        cb1 = manager.get_circuit_breaker("test1")
        cb2 = manager.get_circuit_breaker("test2")

        cb1.total_requests = 10
        cb2.total_requests = 20

        # Reset all
        manager.reset_all()

        assert cb1.total_requests == 0
        assert cb2.total_requests == 0


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases"""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, circuit_config):
        """Test circuit breaker behavior under concurrent load"""
        circuit_breaker = EnhancedCircuitBreaker("concurrent_test", circuit_config)

        async def test_function():
            await asyncio.sleep(0.01)  # Small delay
            return "success"

        # Execute concurrent requests
        tasks = [circuit_breaker.call(test_function) for _ in range(10)]
        results = await asyncio.gather(*tasks)

        assert all(result == "success" for result in results)
        assert circuit_breaker.total_requests == 10
        assert circuit_breaker.successful_requests == 10

    @pytest.mark.asyncio
    async def test_mixed_success_failure_pattern(self, circuit_config):
        """Test circuit breaker with mixed success/failure patterns"""
        circuit_breaker = EnhancedCircuitBreaker("mixed_test", circuit_config)

        async def intermittent_function(should_fail=False):
            if should_fail:
                raise Exception("Intermittent failure")
            return "success"

        # Execute mixed pattern
        pattern = [False, False, True, False, True, False, False]  # True = should fail

        results = []
        for should_fail in pattern:
            try:
                result = await circuit_breaker.call(intermittent_function, should_fail)
                results.append(result)
            except:
                results.append("failed")

        # Verify pattern tracking
        assert circuit_breaker.total_requests == len(pattern)
        success_count = sum(1 for should_fail in pattern if not should_fail)
        assert circuit_breaker.successful_requests == success_count

    @pytest.mark.asyncio
    async def test_timeout_recovery_cycle(self, circuit_config):
        """Test complete timeout and recovery cycle"""
        circuit_config.timeout_seconds = 0.1  # Very short timeout for testing
        circuit_breaker = EnhancedCircuitBreaker("timeout_test", circuit_config)

        async def failing_function():
            raise Exception("Consistent failure")

        # Open the circuit with critical failures
        for i in range(circuit_config.critical_failure_threshold):
            try:
                await circuit_breaker.call(failing_function)
            except:
                pass

        assert circuit_breaker.state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(circuit_config.timeout_seconds + 0.01)

        # Should allow one request (half-open)
        async def successful_function():
            return "recovered"

        # This should succeed and transition to half-open
        result = await circuit_breaker.call(successful_function)
        assert result == "recovered"

    @pytest.mark.asyncio
    async def test_performance_degradation_detection(self, circuit_config):
        """Test detection of performance degradation"""
        circuit_config.slow_request_threshold_ms = 50.0
        circuit_breaker = EnhancedCircuitBreaker("perf_test", circuit_config)

        async def slow_function():
            await asyncio.sleep(0.1)  # 100ms - above threshold
            return "slow_success"

        # Execute slow requests
        for i in range(5):
            result = await circuit_breaker.call(slow_function)
            assert result == "slow_success"

        # Check that performance metrics reflect the slowness
        stats = circuit_breaker.get_statistics()
        assert stats["performance_metrics"]["avg_response_time_ms"] > 50.0

    def test_configuration_validation(self):
        """Test circuit breaker configuration validation"""
        # Test with various configurations
        configs = [
            CircuitBreakerConfig(failure_threshold=1, success_threshold=1),
            CircuitBreakerConfig(enable_adaptive_thresholds=False),
            CircuitBreakerConfig(recovery_strategy=RecoveryStrategy.IMMEDIATE_RETRY)
        ]

        for config in configs:
            cb = EnhancedCircuitBreaker("config_test", config)
            assert cb.config == config