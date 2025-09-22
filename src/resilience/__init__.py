"""Resilience Package

Advanced resilience patterns and fault tolerance mechanisms including:
- Enhanced circuit breakers with adaptive thresholds
- Multi-level failure handling
- Performance-based recovery strategies
- Intelligent failure classification
"""

from .circuit_breaker import (
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

__all__ = [
    "EnhancedCircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerManager",
    "CircuitBreakerError",
    "CircuitState",
    "FailureType",
    "RecoveryStrategy",
    "FailureRecord",
    "PerformanceMetrics",
    "circuit_breaker_manager"
]