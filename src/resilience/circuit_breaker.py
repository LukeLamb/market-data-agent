"""Enhanced Circuit Breaker System

Multi-level circuit breaker implementation with adaptive thresholds,
intelligent failure classification, and performance-based recovery.
"""

import asyncio
import time
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import statistics
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Blocking requests
    HALF_OPEN = "half_open"    # Testing recovery
    DEGRADED = "degraded"      # Partial operation
    ADAPTIVE = "adaptive"      # Learning mode


class FailureType(Enum):
    """Types of failures for classification"""
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    HTTP_ERROR = "http_error"
    RATE_LIMIT = "rate_limit"
    DATA_ERROR = "data_error"
    AUTHENTICATION_ERROR = "authentication_error"
    SERVER_ERROR = "server_error"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Recovery strategies"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    ADAPTIVE_BACKOFF = "adaptive_backoff"
    IMMEDIATE_RETRY = "immediate_retry"
    PERFORMANCE_BASED = "performance_based"


@dataclass
class FailureRecord:
    """Record of a failure occurrence"""
    timestamp: datetime
    failure_type: FailureType
    error_message: str
    response_time_ms: Optional[float] = None
    severity: float = 1.0  # 0.1 (minor) to 1.0 (critical)


@dataclass
class PerformanceMetrics:
    """Performance metrics for adaptive behavior"""
    avg_response_time: float = 0.0
    success_rate: float = 1.0
    throughput: float = 0.0  # requests per second
    error_rate: float = 0.0
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    last_success_time: Optional[datetime] = None
    last_failure_time: Optional[datetime] = None


@dataclass
class CircuitBreakerConfig:
    """Enhanced circuit breaker configuration"""
    # Basic thresholds
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: float = 60.0

    # Multi-level thresholds
    degraded_threshold: int = 3  # Partial failures before degraded mode
    critical_failure_threshold: int = 10  # Critical failures for extended timeout

    # Adaptive behavior
    enable_adaptive_thresholds: bool = True
    adaptive_window_minutes: int = 30
    performance_degradation_threshold: float = 2.0  # Response time multiplier

    # Recovery configuration
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.ADAPTIVE_BACKOFF
    min_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 300.0
    backoff_multiplier: float = 2.0
    jitter_factor: float = 0.1

    # Performance monitoring
    enable_performance_monitoring: bool = True
    performance_window_size: int = 100
    slow_request_threshold_ms: float = 2000.0

    # Failure classification weights
    failure_type_weights: Dict[FailureType, float] = field(default_factory=lambda: {
        FailureType.TIMEOUT: 1.0,
        FailureType.CONNECTION_ERROR: 0.9,
        FailureType.HTTP_ERROR: 0.7,
        FailureType.RATE_LIMIT: 0.5,
        FailureType.DATA_ERROR: 0.6,
        FailureType.AUTHENTICATION_ERROR: 0.8,
        FailureType.SERVER_ERROR: 1.0,
        FailureType.UNKNOWN: 0.8
    })


class EnhancedCircuitBreaker:
    """Enhanced multi-level circuit breaker with adaptive behavior"""

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()

        # State management
        self.state = CircuitState.CLOSED
        self.state_changed_at = datetime.now()
        self.next_attempt_time = datetime.now()

        # Failure tracking
        self.failures: deque[FailureRecord] = deque(maxlen=1000)
        self.failure_counts: Dict[FailureType, int] = defaultdict(int)

        # Performance tracking
        self.performance = PerformanceMetrics()
        self.response_times: deque[float] = deque(maxlen=self.config.performance_window_size)
        self.recent_requests: deque[Tuple[datetime, bool, float]] = deque(maxlen=self.config.performance_window_size)

        # Adaptive thresholds
        self.adaptive_failure_threshold = self.config.failure_threshold
        self.adaptive_timeout = self.config.timeout_seconds

        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.blocked_requests = 0

        # Recovery tracking
        self.half_open_requests = 0
        self.half_open_successes = 0
        self.last_state_transition = datetime.now()

        # Locks for thread safety
        self._lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        async with self._lock:
            # Check if request should be allowed
            if not await self._should_allow_request():
                self.blocked_requests += 1
                raise CircuitBreakerError(f"Circuit breaker {self.name} is {self.state.value}")

            # Track request attempt
            self.total_requests += 1
            start_time = time.time()

            try:
                # Execute the function
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)

                # Record successful execution
                response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                await self._record_success(response_time)

                return result

            except Exception as error:
                # Record failure
                response_time = (time.time() - start_time) * 1000
                await self._record_failure(error, response_time)
                raise

    async def _should_allow_request(self) -> bool:
        """Determine if request should be allowed based on current state"""
        current_time = datetime.now()

        # Update adaptive thresholds if enabled
        if self.config.enable_adaptive_thresholds:
            await self._update_adaptive_thresholds()

        if self.state == CircuitState.CLOSED:
            return True

        elif self.state == CircuitState.OPEN:
            # Check if timeout has expired
            if current_time >= self.next_attempt_time:
                await self._transition_to_half_open()
                return True
            return False

        elif self.state == CircuitState.HALF_OPEN:
            # Allow limited requests to test recovery
            return self.half_open_requests < self.config.success_threshold

        elif self.state == CircuitState.DEGRADED:
            # Allow some requests through with throttling
            return await self._should_allow_degraded_request()

        elif self.state == CircuitState.ADAPTIVE:
            # Use performance metrics to decide
            return await self._should_allow_adaptive_request()

        return False

    async def _record_success(self, response_time_ms: float):
        """Record successful request"""
        self.successful_requests += 1
        self.performance.consecutive_successes += 1
        self.performance.consecutive_failures = 0
        self.performance.last_success_time = datetime.now()

        # Update response time tracking
        self.response_times.append(response_time_ms)
        self.recent_requests.append((datetime.now(), True, response_time_ms))

        # Update performance metrics
        await self._update_performance_metrics()

        # Handle state transitions based on success
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_successes += 1
            if self.half_open_successes >= self.config.success_threshold:
                await self._transition_to_closed()

        elif self.state == CircuitState.DEGRADED:
            # Check if we can transition back to normal
            if self.performance.consecutive_successes >= self.config.success_threshold * 2:
                await self._transition_to_closed()

        # Log performance warning for slow requests
        if response_time_ms > self.config.slow_request_threshold_ms:
            logger.warning(f"Slow request detected for {self.name}: {response_time_ms:.1f}ms")

    async def _record_failure(self, error: Exception, response_time_ms: float):
        """Record failed request with intelligent classification"""
        self.failed_requests += 1
        self.performance.consecutive_failures += 1
        self.performance.consecutive_successes = 0
        self.performance.last_failure_time = datetime.now()

        # Classify failure type
        failure_type = self._classify_failure(error)
        failure_severity = self.config.failure_type_weights.get(failure_type, 0.8)

        # Create failure record
        failure_record = FailureRecord(
            timestamp=datetime.now(),
            failure_type=failure_type,
            error_message=str(error),
            response_time_ms=response_time_ms,
            severity=failure_severity
        )

        self.failures.append(failure_record)
        self.failure_counts[failure_type] += 1

        # Update tracking
        self.recent_requests.append((datetime.now(), False, response_time_ms))
        if response_time_ms > 0:  # Only add valid response times
            self.response_times.append(response_time_ms)

        # Update performance metrics
        await self._update_performance_metrics()

        # Determine state transition based on failure pattern
        await self._evaluate_state_transition_on_failure(failure_record)

        logger.warning(f"Circuit breaker {self.name} recorded failure: {failure_type.value} - {error}")

    def _classify_failure(self, error: Exception) -> FailureType:
        """Classify failure type based on exception"""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        # Order matters - more specific patterns first
        if "timeout" in error_str or "timeout" in error_type:
            return FailureType.TIMEOUT
        elif "connection" in error_str or "connection" in error_type:
            return FailureType.CONNECTION_ERROR
        elif "rate limit" in error_str or "429" in error_str:
            return FailureType.RATE_LIMIT
        elif ("auth" in error_str or "401" in error_str or "403" in error_str or
              "unauthorized" in error_str):
            return FailureType.AUTHENTICATION_ERROR
        elif "500" in error_str or "502" in error_str or "503" in error_str:
            return FailureType.SERVER_ERROR
        elif "400" in error_str or "404" in error_str:
            return FailureType.HTTP_ERROR
        elif "data" in error_str or "parse" in error_str or "json" in error_str:
            return FailureType.DATA_ERROR
        else:
            return FailureType.UNKNOWN

    async def _evaluate_state_transition_on_failure(self, failure_record: FailureRecord):
        """Evaluate whether state transition is needed based on failure"""
        # Calculate weighted failure count (recent failures with higher weight)
        weighted_failure_count = await self._calculate_weighted_failure_count()

        # Also check simple consecutive failure count for immediate response
        consecutive_failures = self.performance.consecutive_failures

        if self.state == CircuitState.CLOSED:
            # Use either weighted count or consecutive failures for transition
            should_transition = (
                weighted_failure_count >= self.adaptive_failure_threshold or
                consecutive_failures >= self.config.failure_threshold
            )

            if should_transition:
                # Check if we should go to degraded mode first
                if (failure_record.failure_type in [FailureType.RATE_LIMIT, FailureType.HTTP_ERROR] and
                    weighted_failure_count < self.config.critical_failure_threshold and
                    consecutive_failures < self.config.critical_failure_threshold):
                    await self._transition_to_degraded()
                else:
                    await self._transition_to_open()
            elif (weighted_failure_count >= self.config.degraded_threshold or
                  consecutive_failures >= self.config.degraded_threshold):
                await self._transition_to_degraded()

        elif self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open state transitions back to open
            await self._transition_to_open()

        elif self.state == CircuitState.DEGRADED:
            if (weighted_failure_count >= self.adaptive_failure_threshold or
                consecutive_failures >= self.config.failure_threshold):
                await self._transition_to_open()

    async def _calculate_weighted_failure_count(self) -> float:
        """Calculate weighted failure count considering recency and severity"""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(minutes=self.config.adaptive_window_minutes)

        weighted_count = 0.0
        for failure in reversed(self.failures):  # Most recent first
            if failure.timestamp < cutoff_time:
                break

            # Time-based weight (more recent = higher weight)
            time_diff = (current_time - failure.timestamp).total_seconds()
            time_weight = math.exp(-time_diff / (self.config.adaptive_window_minutes * 60))

            # Severity weight
            severity_weight = failure.severity

            # Failure type weight
            type_weight = self.config.failure_type_weights.get(failure.failure_type, 0.8)

            weighted_count += time_weight * severity_weight * type_weight

        return weighted_count

    async def _update_adaptive_thresholds(self):
        """Update thresholds based on recent performance"""
        if not self.config.enable_adaptive_thresholds:
            return

        # Adjust failure threshold based on error patterns
        recent_error_rate = await self._get_recent_error_rate()
        if recent_error_rate > 0.5:  # High error rate
            self.adaptive_failure_threshold = max(2, self.config.failure_threshold - 2)
        elif recent_error_rate < 0.1:  # Low error rate
            self.adaptive_failure_threshold = min(10, self.config.failure_threshold + 2)
        else:
            self.adaptive_failure_threshold = self.config.failure_threshold

        # Adjust timeout based on response time patterns
        if self.response_times:
            avg_response_time = statistics.mean(self.response_times)
            if avg_response_time > self.config.slow_request_threshold_ms:
                self.adaptive_timeout = min(self.config.max_backoff_seconds,
                                          self.config.timeout_seconds * 1.5)
            else:
                self.adaptive_timeout = max(self.config.min_backoff_seconds,
                                          self.config.timeout_seconds * 0.8)

    async def _get_recent_error_rate(self) -> float:
        """Get error rate from recent requests"""
        if not self.recent_requests:
            return 0.0

        cutoff_time = datetime.now() - timedelta(minutes=5)  # Last 5 minutes
        recent_requests = [
            req for req in self.recent_requests
            if req[0] > cutoff_time
        ]

        if not recent_requests:
            return 0.0

        errors = sum(1 for req in recent_requests if not req[1])  # req[1] is success flag
        return errors / len(recent_requests)

    async def _update_performance_metrics(self):
        """Update performance metrics based on recent data"""
        if not self.recent_requests:
            return

        # Calculate success rate
        total_requests = len(self.recent_requests)
        successful_requests = sum(1 for req in self.recent_requests if req[1])
        self.performance.success_rate = successful_requests / total_requests
        self.performance.error_rate = 1.0 - self.performance.success_rate

        # Calculate average response time
        if self.response_times:
            self.performance.avg_response_time = statistics.mean(self.response_times)

        # Calculate throughput (requests per second over last minute)
        one_minute_ago = datetime.now() - timedelta(minutes=1)
        recent_minute_requests = [
            req for req in self.recent_requests
            if req[0] > one_minute_ago
        ]
        self.performance.throughput = len(recent_minute_requests) / 60.0

    # State transition methods
    async def _transition_to_open(self):
        """Transition to open state"""
        self.state = CircuitState.OPEN
        self.state_changed_at = datetime.now()
        self.last_state_transition = datetime.now()

        # Calculate backoff time
        backoff_time = await self._calculate_backoff_time()
        self.next_attempt_time = datetime.now() + timedelta(seconds=backoff_time)

        logger.warning(f"Circuit breaker {self.name} opened. Next attempt in {backoff_time:.1f}s")

    async def _transition_to_half_open(self):
        """Transition to half-open state"""
        self.state = CircuitState.HALF_OPEN
        self.state_changed_at = datetime.now()
        self.last_state_transition = datetime.now()
        self.half_open_requests = 0
        self.half_open_successes = 0

        logger.info(f"Circuit breaker {self.name} transitioned to half-open")

    async def _transition_to_closed(self):
        """Transition to closed state"""
        self.state = CircuitState.CLOSED
        self.state_changed_at = datetime.now()
        self.last_state_transition = datetime.now()

        # Reset counters
        self.half_open_requests = 0
        self.half_open_successes = 0

        logger.info(f"Circuit breaker {self.name} closed (recovered)")

    async def _transition_to_degraded(self):
        """Transition to degraded state"""
        self.state = CircuitState.DEGRADED
        self.state_changed_at = datetime.now()
        self.last_state_transition = datetime.now()

        logger.warning(f"Circuit breaker {self.name} transitioned to degraded mode")

    async def _should_allow_degraded_request(self) -> bool:
        """Determine if request should be allowed in degraded state"""
        # Allow 50% of requests in degraded mode
        return hash(time.time()) % 2 == 0

    async def _should_allow_adaptive_request(self) -> bool:
        """Determine if request should be allowed in adaptive state"""
        # Use performance metrics to make decision
        if self.performance.error_rate > 0.5:
            return False
        if self.performance.avg_response_time > self.config.slow_request_threshold_ms * 2:
            return False
        return True

    async def _calculate_backoff_time(self) -> float:
        """Calculate backoff time based on recovery strategy"""
        strategy = self.config.recovery_strategy

        if strategy == RecoveryStrategy.EXPONENTIAL_BACKOFF:
            # Exponential backoff based on consecutive failures
            backoff = self.config.min_backoff_seconds * (
                self.config.backoff_multiplier ** min(self.performance.consecutive_failures, 10)
            )
        elif strategy == RecoveryStrategy.LINEAR_BACKOFF:
            # Linear increase based on failures
            backoff = self.config.min_backoff_seconds + (
                self.performance.consecutive_failures * self.config.min_backoff_seconds
            )
        elif strategy == RecoveryStrategy.ADAPTIVE_BACKOFF:
            # Adaptive based on failure patterns and performance
            base_backoff = self.config.timeout_seconds
            error_multiplier = 1.0 + self.performance.error_rate
            performance_multiplier = 1.0 + (self.performance.avg_response_time / 1000.0)
            backoff = base_backoff * error_multiplier * performance_multiplier
        elif strategy == RecoveryStrategy.PERFORMANCE_BASED:
            # Based on recent performance degradation
            if self.performance.avg_response_time > self.config.slow_request_threshold_ms:
                backoff = self.config.timeout_seconds * 2
            else:
                backoff = self.config.timeout_seconds
        else:  # IMMEDIATE_RETRY
            backoff = self.config.min_backoff_seconds

        # Apply jitter to prevent thundering herd
        jitter = backoff * self.config.jitter_factor * (hash(time.time()) % 100) / 100
        backoff += jitter

        # Clamp to configured bounds
        return max(self.config.min_backoff_seconds,
                  min(self.config.max_backoff_seconds, backoff))

    # Status and monitoring methods
    def get_state(self) -> CircuitState:
        """Get current circuit breaker state"""
        return self.state

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        current_time = datetime.now()

        # Calculate time-based metrics
        time_in_current_state = (current_time - self.state_changed_at).total_seconds()
        time_since_last_transition = (current_time - self.last_state_transition).total_seconds()

        # Recent failure analysis
        recent_failures = [
            f for f in self.failures
            if f.timestamp > current_time - timedelta(minutes=10)
        ]

        failure_type_breakdown = {}
        for failure_type in FailureType:
            count = sum(1 for f in recent_failures if f.failure_type == failure_type)
            if count > 0:
                failure_type_breakdown[failure_type.value] = count

        return {
            "name": self.name,
            "state": self.state.value,
            "time_in_current_state_seconds": time_in_current_state,
            "time_since_last_transition_seconds": time_since_last_transition,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "blocked_requests": self.blocked_requests,
            "success_rate": self.successful_requests / max(1, self.total_requests),
            "performance_metrics": {
                "avg_response_time_ms": self.performance.avg_response_time,
                "current_success_rate": self.performance.success_rate,
                "current_error_rate": self.performance.error_rate,
                "throughput_per_second": self.performance.throughput,
                "consecutive_successes": self.performance.consecutive_successes,
                "consecutive_failures": self.performance.consecutive_failures
            },
            "adaptive_thresholds": {
                "failure_threshold": self.adaptive_failure_threshold,
                "timeout_seconds": self.adaptive_timeout
            },
            "recent_failures": len(recent_failures),
            "failure_type_breakdown": failure_type_breakdown,
            "next_attempt_time": self.next_attempt_time.isoformat() if self.state == CircuitState.OPEN else None,
            "half_open_progress": {
                "requests_made": self.half_open_requests,
                "successes": self.half_open_successes,
                "needed_successes": self.config.success_threshold
            } if self.state == CircuitState.HALF_OPEN else None
        }

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the circuit breaker"""
        stats = self.get_statistics()

        # Determine health status
        if self.state == CircuitState.CLOSED and stats["success_rate"] > 0.9:
            health = "healthy"
        elif self.state == CircuitState.DEGRADED:
            health = "degraded"
        elif self.state in [CircuitState.OPEN, CircuitState.HALF_OPEN]:
            health = "unhealthy"
        else:
            health = "unknown"

        return {
            "health_status": health,
            "state": self.state.value,
            "performance_score": min(100, max(0, int(stats["success_rate"] * 100))),
            "last_success_time": self.performance.last_success_time.isoformat() if self.performance.last_success_time else None,
            "last_failure_time": self.performance.last_failure_time.isoformat() if self.performance.last_failure_time else None,
            "adaptive_behavior": {
                "enabled": self.config.enable_adaptive_thresholds,
                "current_failure_threshold": self.adaptive_failure_threshold,
                "current_timeout": self.adaptive_timeout
            }
        }

    def reset(self):
        """Reset circuit breaker to initial state"""
        self.state = CircuitState.CLOSED
        self.state_changed_at = datetime.now()
        self.next_attempt_time = datetime.now()

        self.failures.clear()
        self.failure_counts.clear()

        self.performance = PerformanceMetrics()
        self.response_times.clear()
        self.recent_requests.clear()

        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.blocked_requests = 0

        self.half_open_requests = 0
        self.half_open_successes = 0

        logger.info(f"Circuit breaker {self.name} reset")


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker blocks a request"""
    pass


class CircuitBreakerManager:
    """Manager for multiple circuit breakers"""

    def __init__(self):
        self.circuit_breakers: Dict[str, EnhancedCircuitBreaker] = {}

    def get_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> EnhancedCircuitBreaker:
        """Get or create a circuit breaker"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = EnhancedCircuitBreaker(name, config)
        return self.circuit_breakers[name]

    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers"""
        return {
            name: breaker.get_statistics()
            for name, breaker in self.circuit_breakers.items()
        }

    async def get_all_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all circuit breakers"""
        results = {}
        for name, breaker in self.circuit_breakers.items():
            results[name] = await breaker.get_health_status()
        return results

    def reset_all(self):
        """Reset all circuit breakers"""
        for breaker in self.circuit_breakers.values():
            breaker.reset()
        logger.info("All circuit breakers reset")


# Global circuit breaker manager instance
circuit_breaker_manager = CircuitBreakerManager()