"""
Performance Optimization Module

This module provides intelligent caching, request batching, and performance optimization
capabilities for the Market Data Agent to achieve sub-100ms cached response times.

Key Components:
- IntelligentCache: Multi-level caching with TTL and LRU eviction
- RequestBatcher: Batch processing for efficient API utilization
- PerformanceProfiler: Real-time performance monitoring and bottleneck identification
- QueryOptimizer: Database query optimization and connection pooling

Usage:
    from performance import IntelligentCache, RequestBatcher, PerformanceProfiler

    # Initialize caching
    cache = IntelligentCache()

    # Set up request batching
    batcher = RequestBatcher()

    # Start performance profiling
    profiler = PerformanceProfiler()
"""

from .intelligent_cache import (
    IntelligentCache,
    CacheStrategy,
    CacheEntry,
    CacheStats,
    CacheKey
)

from .request_batcher import (
    RequestBatcher,
    BatchRequest,
    BatchResult,
    BatchingStrategy,
    BatchProcessor
)

from .performance_profiler import (
    PerformanceProfiler,
    ProfileResult,
    BottleneckAnalyzer,
    PerformanceMetrics
)

from .query_optimizer import (
    QueryOptimizer,
    ConnectionPool,
    QueryPlan,
    OptimizationStrategy
)

__version__ = "1.0.0"
__author__ = "Market Data Agent Team"

# Default performance instances for convenience
_default_cache = None
_default_batcher = None
_default_profiler = None
_default_optimizer = None

def get_default_cache() -> IntelligentCache:
    """Get the default cache instance."""
    global _default_cache
    if _default_cache is None:
        _default_cache = IntelligentCache()
    return _default_cache

def get_default_batcher() -> RequestBatcher:
    """Get the default request batcher instance."""
    global _default_batcher
    if _default_batcher is None:
        _default_batcher = RequestBatcher()
    return _default_batcher

def get_default_profiler() -> PerformanceProfiler:
    """Get the default performance profiler instance."""
    global _default_profiler
    if _default_profiler is None:
        _default_profiler = PerformanceProfiler()
    return _default_profiler

def get_default_optimizer() -> QueryOptimizer:
    """Get the default query optimizer instance."""
    global _default_optimizer
    if _default_optimizer is None:
        _default_optimizer = QueryOptimizer()
    return _default_optimizer

# Convenience functions for quick performance optimization
def cache_get(key: str, default=None):
    """Get value from default cache."""
    return get_default_cache().get(key, default)

def cache_set(key: str, value, ttl: int = None):
    """Set value in default cache."""
    return get_default_cache().set(key, value, ttl)

def batch_request(operation: str, data: dict):
    """Add request to default batch processor."""
    return get_default_batcher().add_request(operation, data)

def start_profiling(operation_name: str):
    """Start profiling an operation."""
    return get_default_profiler().start_operation(operation_name)

__all__ = [
    # Main classes
    'IntelligentCache',
    'RequestBatcher',
    'PerformanceProfiler',
    'QueryOptimizer',

    # Enums and strategies
    'CacheStrategy',
    'BatchingStrategy',
    'OptimizationStrategy',

    # Data classes
    'CacheEntry',
    'CacheStats',
    'CacheKey',
    'BatchRequest',
    'BatchResult',
    'ProfileResult',
    'PerformanceMetrics',
    'ConnectionPool',
    'QueryPlan',

    # Utilities
    'BottleneckAnalyzer',
    'BatchProcessor',

    # Convenience functions
    'get_default_cache',
    'get_default_batcher',
    'get_default_profiler',
    'get_default_optimizer',
    'cache_get',
    'cache_set',
    'batch_request',
    'start_profiling'
]