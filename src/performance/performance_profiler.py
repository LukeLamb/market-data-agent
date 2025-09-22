"""
Performance Profiler

Real-time performance monitoring and bottleneck identification
for optimizing system performance and achieving target response times.
"""

import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import statistics
import logging

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics"""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    QUEUE_DEPTH = "queue_depth"
    CACHE_HIT_RATE = "cache_hit_rate"


@dataclass
class PerformanceMetrics:
    """Performance metrics data"""
    operation: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    threshold: Optional[float] = None
    target: Optional[float] = None

    def is_within_threshold(self) -> bool:
        """Check if metric is within acceptable threshold"""
        if self.threshold is None:
            return True
        return self.value <= self.threshold

    def performance_score(self) -> float:
        """Calculate performance score (0-100, higher is better)"""
        if self.target is None:
            return 100.0 if self.is_within_threshold() else 0.0

        if self.metric_type in [MetricType.RESPONSE_TIME, MetricType.ERROR_RATE]:
            # Lower is better
            if self.value <= self.target:
                return 100.0
            elif self.threshold and self.value >= self.threshold:
                return 0.0
            else:
                # Linear interpolation between target and threshold
                threshold = self.threshold or (self.target * 2)
                score = 100.0 * (1 - (self.value - self.target) / (threshold - self.target))
                return max(0.0, min(100.0, score))
        else:
            # Higher is better (throughput, cache hit rate, etc.)
            if self.value >= self.target:
                return 100.0
            else:
                score = 100.0 * (self.value / self.target)
                return max(0.0, min(100.0, score))


@dataclass
class ProfileResult:
    """Result of performance profiling"""
    operation: str
    duration: float
    start_time: datetime
    end_time: datetime
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    sub_operations: List['ProfileResult'] = field(default_factory=list)

    @property
    def total_duration(self) -> float:
        """Get total duration including sub-operations"""
        return self.duration + sum(sub.total_duration for sub in self.sub_operations)


class BottleneckAnalyzer:
    """Analyzes performance data to identify bottlenecks"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.operation_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))

    def add_result(self, result: ProfileResult):
        """Add profiling result for analysis"""
        self.operation_data[result.operation].append(result)

    def identify_bottlenecks(self, threshold_percentile: float = 95) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        bottlenecks = []

        for operation, results in self.operation_data.items():
            if len(results) < 10:  # Need minimum data points
                continue

            durations = [r.duration for r in results if r.success]
            if not durations:
                continue

            # Calculate statistics
            avg_duration = statistics.mean(durations)
            p95_duration = statistics.quantiles(durations, n=100)[94] if len(durations) >= 5 else max(durations)
            p99_duration = statistics.quantiles(durations, n=100)[98] if len(durations) >= 5 else max(durations)

            # Calculate error rate
            total_requests = len(results)
            failed_requests = sum(1 for r in results if not r.success)
            error_rate = (failed_requests / total_requests) * 100

            # Identify if this is a bottleneck
            is_bottleneck = (
                p95_duration > 0.5 or  # >500ms at p95
                avg_duration > 0.2 or  # >200ms average
                error_rate > 5.0       # >5% error rate
            )

            if is_bottleneck:
                bottleneck = {
                    'operation': operation,
                    'severity': self._calculate_severity(avg_duration, p95_duration, error_rate),
                    'metrics': {
                        'average_duration': avg_duration,
                        'p95_duration': p95_duration,
                        'p99_duration': p99_duration,
                        'error_rate': error_rate,
                        'total_requests': total_requests
                    },
                    'recommendations': self._generate_recommendations(operation, avg_duration, p95_duration, error_rate)
                }
                bottlenecks.append(bottleneck)

        # Sort by severity
        bottlenecks.sort(key=lambda x: x['severity'], reverse=True)
        return bottlenecks

    def _calculate_severity(self, avg_duration: float, p95_duration: float, error_rate: float) -> float:
        """Calculate bottleneck severity score (0-100)"""
        # Duration component (higher duration = higher severity)
        duration_score = min(100, (avg_duration / 1.0) * 50 + (p95_duration / 2.0) * 50)

        # Error rate component
        error_score = min(100, error_rate * 2)

        # Combined severity
        return min(100, duration_score * 0.7 + error_score * 0.3)

    def _generate_recommendations(self, operation: str, avg_duration: float, p95_duration: float, error_rate: float) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        if avg_duration > 0.2:
            recommendations.append("Consider adding caching for this operation")
            recommendations.append("Review database queries for optimization opportunities")

        if p95_duration > 1.0:
            recommendations.append("Implement request batching to reduce overhead")
            recommendations.append("Consider asynchronous processing for non-critical operations")

        if error_rate > 5.0:
            recommendations.append("Implement circuit breaker pattern")
            recommendations.append("Add retry logic with exponential backoff")
            recommendations.append("Review error handling and logging")

        if avg_duration > 0.5:
            recommendations.append("Consider breaking down operation into smaller steps")
            recommendations.append("Implement parallel processing where possible")

        return recommendations


class PerformanceProfiler:
    """
    Real-time performance profiler for monitoring and optimizing
    system performance to achieve target response times.
    """

    def __init__(
        self,
        enable_real_time_analysis: bool = True,
        analysis_interval: int = 60,  # seconds
        max_history_size: int = 10000
    ):
        """Initialize performance profiler

        Args:
            enable_real_time_analysis: Enable real-time bottleneck analysis
            analysis_interval: Interval for analysis in seconds
            max_history_size: Maximum number of results to keep
        """
        self.enable_real_time_analysis = enable_real_time_analysis
        self.analysis_interval = analysis_interval
        self.max_history_size = max_history_size

        # Performance data storage
        self.results: deque = deque(maxlen=max_history_size)
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.active_operations: Dict[str, Dict[str, Any]] = {}

        # Analysis components
        self.bottleneck_analyzer = BottleneckAnalyzer()

        # Background tasks
        self._running = False
        self._analysis_task = None

        # Thread safety
        self._lock = threading.RLock()

        # Performance targets (can be configured per operation)
        self.targets = {
            'cached_response': 0.1,      # 100ms for cached responses
            'real_time_response': 0.5,   # 500ms for real-time responses
            'batch_processing': 2.0,     # 2s for batch processing
            'default': 1.0               # 1s default
        }

        # Thresholds (when to consider performance poor)
        self.thresholds = {
            'cached_response': 0.2,      # 200ms threshold for cached
            'real_time_response': 1.0,   # 1s threshold for real-time
            'batch_processing': 5.0,     # 5s threshold for batch
            'default': 2.0               # 2s default threshold
        }

        logger.info("PerformanceProfiler initialized")

    async def start(self):
        """Start background analysis"""
        self._running = True
        if self.enable_real_time_analysis and self._analysis_task is None:
            self._analysis_task = asyncio.create_task(self._background_analysis())
        logger.info("PerformanceProfiler started")

    async def stop(self):
        """Stop background analysis"""
        self._running = False
        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass
        logger.info("PerformanceProfiler stopped")

    def start_operation(self, operation: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start profiling an operation"""
        operation_id = f"{operation}_{int(time.time() * 1000000)}"

        with self._lock:
            self.active_operations[operation_id] = {
                'operation': operation,
                'start_time': datetime.now(),
                'start_timestamp': time.time(),
                'metadata': metadata or {}
            }

        return operation_id

    def end_operation(
        self,
        operation_id: str,
        success: bool = True,
        error: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> ProfileResult:
        """End profiling an operation"""
        end_time = time.time()
        end_datetime = datetime.now()

        with self._lock:
            if operation_id not in self.active_operations:
                logger.warning(f"Unknown operation ID: {operation_id}")
                return None

            op_data = self.active_operations.pop(operation_id)
            duration = end_time - op_data['start_timestamp']

            # Create result
            result = ProfileResult(
                operation=op_data['operation'],
                duration=duration,
                start_time=op_data['start_time'],
                end_time=end_datetime,
                success=success,
                error=error,
                metadata={**op_data['metadata'], **(additional_metadata or {})}
            )

            # Store result
            self.results.append(result)
            self.bottleneck_analyzer.add_result(result)

            # Record metrics
            self._record_metrics(result)

            return result

    def profile_operation(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for profiling operations"""
        return OperationProfiler(self, operation, metadata)

    def record_metric(
        self,
        operation: str,
        metric_type: MetricType,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """Record a performance metric"""
        target = self._get_target(operation, metric_type)
        threshold = self._get_threshold(operation, metric_type)

        metric = PerformanceMetrics(
            operation=operation,
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            threshold=threshold,
            target=target
        )

        with self._lock:
            self.metrics[f"{operation}_{metric_type.value}"].append(metric)

    def get_operation_stats(self, operation: str, hours: int = 1) -> Dict[str, Any]:
        """Get statistics for a specific operation"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        with self._lock:
            # Filter results for this operation and time window
            operation_results = [
                r for r in self.results
                if r.operation == operation and r.start_time >= cutoff_time
            ]

            if not operation_results:
                return {'operation': operation, 'no_data': True}

            # Calculate statistics
            durations = [r.duration for r in operation_results if r.success]
            success_count = sum(1 for r in operation_results if r.success)
            error_count = len(operation_results) - success_count

            stats = {
                'operation': operation,
                'total_requests': len(operation_results),
                'success_count': success_count,
                'error_count': error_count,
                'error_rate': (error_count / len(operation_results)) * 100,
                'success_rate': (success_count / len(operation_results)) * 100
            }

            if durations:
                stats.update({
                    'average_duration': statistics.mean(durations),
                    'min_duration': min(durations),
                    'max_duration': max(durations),
                    'median_duration': statistics.median(durations)
                })

                if len(durations) >= 5:
                    stats.update({
                        'p95_duration': statistics.quantiles(durations, n=100)[94],
                        'p99_duration': statistics.quantiles(durations, n=100)[98],
                        'std_deviation': statistics.stdev(durations)
                    })

            return stats

    def get_system_performance(self) -> Dict[str, Any]:
        """Get overall system performance metrics"""
        with self._lock:
            recent_results = [
                r for r in self.results
                if r.start_time >= datetime.now() - timedelta(hours=1)
            ]

            if not recent_results:
                return {'no_data': True}

            # Group by operation
            operations = defaultdict(list)
            for result in recent_results:
                operations[result.operation].append(result)

            # Calculate per-operation metrics
            operation_metrics = {}
            for operation, results in operations.items():
                durations = [r.duration for r in results if r.success]
                if durations:
                    operation_metrics[operation] = {
                        'count': len(results),
                        'average_duration': statistics.mean(durations),
                        'p95_duration': statistics.quantiles(durations, n=100)[94] if len(durations) >= 5 else max(durations),
                        'error_rate': (sum(1 for r in results if not r.success) / len(results)) * 100
                    }

            # Overall system metrics
            all_durations = [r.duration for r in recent_results if r.success]
            total_errors = sum(1 for r in recent_results if not r.success)

            system_metrics = {
                'total_operations': len(recent_results),
                'total_errors': total_errors,
                'overall_error_rate': (total_errors / len(recent_results)) * 100,
                'operations': operation_metrics
            }

            if all_durations:
                system_metrics.update({
                    'overall_average_duration': statistics.mean(all_durations),
                    'overall_p95_duration': statistics.quantiles(all_durations, n=100)[94] if len(all_durations) >= 5 else max(all_durations)
                })

            return system_metrics

    def get_bottlenecks(self) -> List[Dict[str, Any]]:
        """Get current performance bottlenecks"""
        return self.bottleneck_analyzer.identify_bottlenecks()

    def _record_metrics(self, result: ProfileResult):
        """Record metrics from profile result"""
        # Record response time metric
        self.record_metric(
            result.operation,
            MetricType.RESPONSE_TIME,
            result.duration
        )

        # Record success/error rate
        error_value = 0.0 if result.success else 100.0
        self.record_metric(
            result.operation,
            MetricType.ERROR_RATE,
            error_value
        )

    def _get_target(self, operation: str, metric_type: MetricType) -> Optional[float]:
        """Get performance target for operation and metric type"""
        if metric_type == MetricType.RESPONSE_TIME:
            # Determine operation category
            if 'cache' in operation.lower():
                return self.targets['cached_response']
            elif 'batch' in operation.lower():
                return self.targets['batch_processing']
            elif 'real_time' in operation.lower() or 'live' in operation.lower():
                return self.targets['real_time_response']
            else:
                return self.targets['default']
        return None

    def _get_threshold(self, operation: str, metric_type: MetricType) -> Optional[float]:
        """Get performance threshold for operation and metric type"""
        if metric_type == MetricType.RESPONSE_TIME:
            # Determine operation category
            if 'cache' in operation.lower():
                return self.thresholds['cached_response']
            elif 'batch' in operation.lower():
                return self.thresholds['batch_processing']
            elif 'real_time' in operation.lower() or 'live' in operation.lower():
                return self.thresholds['real_time_response']
            else:
                return self.thresholds['default']
        return None

    async def _background_analysis(self):
        """Background task for real-time analysis"""
        while self._running:
            try:
                await asyncio.sleep(self.analysis_interval)

                # Identify bottlenecks
                bottlenecks = self.get_bottlenecks()

                if bottlenecks:
                    logger.warning(f"Performance bottlenecks detected: {len(bottlenecks)} operations")
                    for bottleneck in bottlenecks[:3]:  # Log top 3
                        logger.warning(f"Bottleneck: {bottleneck['operation']} "
                                     f"(severity: {bottleneck['severity']:.1f}, "
                                     f"avg: {bottleneck['metrics']['average_duration']:.3f}s)")

                # Log overall performance
                system_perf = self.get_system_performance()
                if not system_perf.get('no_data'):
                    logger.info(f"System performance: "
                              f"{system_perf.get('total_operations', 0)} operations, "
                              f"{system_perf.get('overall_error_rate', 0):.2f}% error rate, "
                              f"{system_perf.get('overall_average_duration', 0):.3f}s avg response")

            except Exception as e:
                logger.error(f"Error in performance analysis: {e}")


class OperationProfiler:
    """Context manager for profiling operations"""

    def __init__(self, profiler: PerformanceProfiler, operation: str, metadata: Optional[Dict[str, Any]] = None):
        self.profiler = profiler
        self.operation = operation
        self.metadata = metadata
        self.operation_id = None

    def __enter__(self):
        self.operation_id = self.profiler.start_operation(self.operation, self.metadata)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        error = str(exc_val) if exc_val else None
        self.profiler.end_operation(self.operation_id, success, error)