"""Advanced Metrics Collection System

Comprehensive metrics collection with real-time monitoring, custom metrics,
histogram tracking, and intelligent aggregation for operational excellence.
"""

import time
import asyncio
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import threading
import logging

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"
    PERCENTAGE = "percentage"


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class MetricValue:
    """Individual metric value with metadata"""
    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Statistical summary of metric values"""
    count: int
    sum_value: float
    min_value: float
    max_value: float
    mean_value: float
    median_value: float
    std_dev: float
    percentile_95: float
    percentile_99: float
    last_value: float
    timestamp: datetime


@dataclass
class AlertRule:
    """Configuration for metric alerts"""
    metric_name: str
    condition: str  # "gt", "lt", "eq", "ne", "gte", "lte"
    threshold: Union[int, float]
    severity: AlertSeverity
    duration_seconds: int = 60
    cooldown_seconds: int = 300
    enabled: bool = True
    labels: Dict[str, str] = field(default_factory=dict)
    callback: Optional[Callable] = None


@dataclass
class Alert:
    """Active alert instance"""
    rule: AlertRule
    triggered_at: datetime
    metric_value: float
    labels: Dict[str, str]
    message: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class MetricsCollector:
    """Advanced metrics collection system with real-time monitoring"""

    def __init__(self, max_history_size: int = 10000, collection_interval: float = 1.0):
        """Initialize metrics collector

        Args:
            max_history_size: Maximum number of metric values to keep in history
            collection_interval: Interval in seconds for metric collection
        """
        self.max_history_size = max_history_size
        self.collection_interval = collection_interval

        # Metric storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_history_size))
        self.metric_types: Dict[str, MetricType] = {}
        self.metric_labels: Dict[str, Dict[str, str]] = {}

        # Real-time tracking
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.rates: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Alert system
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []

        # Collection tracking
        self.collection_start_time = datetime.now()

        # Collection state
        self.is_collecting = False
        self.collection_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()

        # Custom metric callbacks
        self.metric_callbacks: Dict[str, Callable] = {}

        # Statistics cache
        self.statistics_cache: Dict[str, Dict[str, MetricSummary]] = {}
        self.cache_ttl: int = 60  # seconds

        logger.info("Metrics collector initialized")

    def register_metric(self,
                       name: str,
                       metric_type: MetricType,
                       description: str = "",
                       labels: Optional[Dict[str, str]] = None) -> None:
        """Register a new metric for collection

        Args:
            name: Unique metric name
            metric_type: Type of metric
            description: Human-readable description
            labels: Default labels for this metric
        """
        self.metric_types[name] = metric_type
        self.metric_labels[name] = labels or {}

        logger.info(f"Registered metric: {name} ({metric_type.value})")

    def record_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a counter metric

        Args:
            name: Metric name
            value: Counter increment value
            labels: Additional labels
        """
        with self._lock:
            self.counters[name] += value
            self._store_metric_value(name, value, labels, MetricType.COUNTER)

    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric

        Args:
            name: Metric name
            value: Current gauge value
            labels: Additional labels
        """
        with self._lock:
            self.gauges[name] = value
            self._store_metric_value(name, value, labels, MetricType.GAUGE)

    def record_timer(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a timer metric

        Args:
            name: Metric name
            duration: Duration in seconds
            labels: Additional labels
        """
        with self._lock:
            self.timers[name].append(duration)
            # Keep only recent timer values
            if len(self.timers[name]) > 1000:
                self.timers[name] = self.timers[name][-500:]

            self._store_metric_value(name, duration, labels, MetricType.TIMER)

    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric

        Args:
            name: Metric name
            value: Value to add to histogram
            labels: Additional labels
        """
        with self._lock:
            self._store_metric_value(name, value, labels, MetricType.HISTOGRAM)

    def record_rate(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a rate metric

        Args:
            name: Metric name
            value: Rate value
            labels: Additional labels
        """
        with self._lock:
            self.rates[name].append((time.time(), value))
            self._store_metric_value(name, value, labels, MetricType.RATE)

    def timer_context(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for timing operations

        Args:
            name: Metric name
            labels: Additional labels

        Returns:
            Context manager that records execution time
        """
        return TimerContext(self, name, labels)

    def _store_metric_value(self,
                           name: str,
                           value: Union[int, float],
                           labels: Optional[Dict[str, str]],
                           metric_type: MetricType) -> None:
        """Store metric value in history"""
        combined_labels = {**self.metric_labels.get(name, {}), **(labels or {})}

        metric_value = MetricValue(
            value=value,
            timestamp=datetime.now(),
            labels=combined_labels,
            metadata={"type": metric_type.value}
        )

        self.metrics[name].append(metric_value)

        # Check for alerts
        self._check_alerts(name, value, combined_labels)

    def _check_alerts(self, metric_name: str, value: float, labels: Dict[str, str]) -> None:
        """Check if metric value triggers any alerts"""
        for rule_name, rule in self.alert_rules.items():
            if rule.metric_name != metric_name or not rule.enabled:
                continue

            # Check if labels match
            if rule.labels:
                if not all(labels.get(k) == v for k, v in rule.labels.items()):
                    continue

            # Evaluate condition
            triggered = self._evaluate_alert_condition(rule.condition, value, rule.threshold)

            if triggered:
                self._trigger_alert(rule, value, labels)
            else:
                self._resolve_alert(rule_name)

    def _evaluate_alert_condition(self, condition: str, value: float, threshold: float) -> bool:
        """Evaluate alert condition"""
        conditions = {
            "gt": value > threshold,
            "gte": value >= threshold,
            "lt": value < threshold,
            "lte": value <= threshold,
            "eq": value == threshold,
            "ne": value != threshold
        }
        return conditions.get(condition, False)

    def _trigger_alert(self, rule: AlertRule, value: float, labels: Dict[str, str]) -> None:
        """Trigger an alert"""
        rule_key = f"{rule.metric_name}_{hash(frozenset(labels.items()))}"

        # Check cooldown
        if rule_key in self.active_alerts:
            existing_alert = self.active_alerts[rule_key]
            if not existing_alert.resolved:
                # Alert already active
                return
            elif (datetime.now() - existing_alert.resolved_at).total_seconds() < rule.cooldown_seconds:
                # Still in cooldown period
                return

        # Create new alert
        alert = Alert(
            rule=rule,
            triggered_at=datetime.now(),
            metric_value=value,
            labels=labels,
            message=f"Metric {rule.metric_name} {rule.condition} {rule.threshold} (current: {value})"
        )

        self.active_alerts[rule_key] = alert
        self.alert_history.append(alert)

        # Execute callback if provided
        if rule.callback:
            try:
                rule.callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        logger.warning(f"Alert triggered: {alert.message}")

    def _resolve_alert(self, rule_key: str) -> None:
        """Resolve an active alert"""
        if rule_key in self.active_alerts:
            alert = self.active_alerts[rule_key]
            if not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                logger.info(f"Alert resolved: {alert.message}")

    def add_alert_rule(self, name: str, rule: AlertRule) -> None:
        """Add an alert rule

        Args:
            name: Unique rule name
            rule: Alert rule configuration
        """
        self.alert_rules[name] = rule
        logger.info(f"Added alert rule: {name} for metric {rule.metric_name}")

    def remove_alert_rule(self, name: str) -> bool:
        """Remove an alert rule

        Args:
            name: Rule name to remove

        Returns:
            True if rule was removed
        """
        if name in self.alert_rules:
            del self.alert_rules[name]
            logger.info(f"Removed alert rule: {name}")
            return True
        return False

    def get_metric_summary(self,
                          name: str,
                          time_range: Optional[timedelta] = None,
                          labels_filter: Optional[Dict[str, str]] = None) -> Optional[MetricSummary]:
        """Get statistical summary of a metric

        Args:
            name: Metric name
            time_range: Time range to consider (None for all data)
            labels_filter: Filter by labels

        Returns:
            Metric summary or None if metric not found
        """
        if name not in self.metrics:
            return None

        # Check cache first
        cache_key = f"{name}_{time_range}_{hash(frozenset((labels_filter or {}).items()))}"
        if cache_key in self.statistics_cache:
            cached_summary, cached_time = self.statistics_cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < self.cache_ttl:
                return cached_summary

        # Filter by time range and labels
        cutoff_time = None
        if time_range:
            cutoff_time = datetime.now() - time_range

        values = []
        for metric_value in self.metrics[name]:
            # Time filter
            if cutoff_time and metric_value.timestamp < cutoff_time:
                continue

            # Labels filter
            if labels_filter:
                if not all(metric_value.labels.get(k) == v for k, v in labels_filter.items()):
                    continue

            values.append(metric_value.value)

        if not values:
            return None

        # Calculate statistics
        values_sorted = sorted(values)
        summary = MetricSummary(
            count=len(values),
            sum_value=sum(values),
            min_value=min(values),
            max_value=max(values),
            mean_value=statistics.mean(values),
            median_value=statistics.median(values),
            std_dev=statistics.stdev(values) if len(values) > 1 else 0.0,
            percentile_95=self._percentile(values_sorted, 95),
            percentile_99=self._percentile(values_sorted, 99),
            last_value=values[-1],
            timestamp=datetime.now()
        )

        # Cache the result
        self.statistics_cache[cache_key] = (summary, datetime.now())

        return summary

    def _percentile(self, sorted_values: List[float], percentile: float) -> float:
        """Calculate percentile from sorted values"""
        if not sorted_values:
            return 0.0

        index = (percentile / 100) * (len(sorted_values) - 1)
        lower = int(index)
        upper = min(lower + 1, len(sorted_values) - 1)

        if lower == upper:
            return sorted_values[lower]

        weight = index - lower
        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

    def get_current_rates(self, name: str, window_seconds: int = 60) -> Dict[str, float]:
        """Get current rate metrics

        Args:
            name: Metric name
            window_seconds: Time window for rate calculation

        Returns:
            Dictionary with rate statistics
        """
        if name not in self.rates:
            return {}

        current_time = time.time()
        cutoff_time = current_time - window_seconds

        # Filter recent values
        recent_values = [(timestamp, value) for timestamp, value in self.rates[name]
                        if timestamp >= cutoff_time]

        if not recent_values:
            return {}

        values = [value for _, value in recent_values]
        timestamps = [timestamp for timestamp, _ in recent_values]

        if len(values) == 1:
            return {"current_rate": values[0], "count": 1}

        # Calculate rate statistics
        time_span = timestamps[-1] - timestamps[0]
        if time_span == 0:
            return {"current_rate": values[-1], "count": len(values)}

        total_rate = sum(values)
        avg_rate = total_rate / len(values)
        rate_per_second = total_rate / time_span if time_span > 0 else 0

        return {
            "current_rate": values[-1],
            "average_rate": avg_rate,
            "rate_per_second": rate_per_second,
            "count": len(values),
            "time_span": time_span
        }

    def register_custom_metric(self, name: str, callback: Callable[[], Union[int, float]]) -> None:
        """Register a custom metric with callback

        Args:
            name: Metric name
            callback: Function that returns the metric value
        """
        self.metric_callbacks[name] = callback
        logger.info(f"Registered custom metric: {name}")

    async def start_collection(self) -> None:
        """Start automatic metric collection"""
        if self.is_collecting:
            return

        self.is_collecting = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Started metric collection")

    async def stop_collection(self) -> None:
        """Stop automatic metric collection"""
        if not self.is_collecting:
            return

        self.is_collecting = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped metric collection")

    async def _collection_loop(self) -> None:
        """Main collection loop"""
        while self.is_collecting:
            try:
                # Collect custom metrics
                for name, callback in self.metric_callbacks.items():
                    try:
                        value = callback()
                        self.record_gauge(name, value, {"source": "custom"})
                    except Exception as e:
                        logger.error(f"Failed to collect custom metric {name}: {e}")

                # Clean up old cache entries
                self._cleanup_cache()

                await asyncio.sleep(self.collection_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                await asyncio.sleep(self.collection_interval)

    def _cleanup_cache(self) -> None:
        """Clean up old cache entries"""
        current_time = datetime.now()
        expired_keys = []

        for key, (_, cached_time) in self.statistics_cache.items():
            if (current_time - cached_time).total_seconds() > self.cache_ttl * 2:
                expired_keys.append(key)

        for key in expired_keys:
            del self.statistics_cache[key]

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metric values

        Returns:
            Dictionary with all metric data
        """
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "timers": {name: {
                "count": len(values),
                "mean": statistics.mean(values) if values else 0,
                "min": min(values) if values else 0,
                "max": max(values) if values else 0
            } for name, values in self.timers.items()},
            "active_alerts": len(self.active_alerts),
            "total_alerts": len(self.alert_history),
            "collection_active": self.is_collecting
        }

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current snapshot of all metrics with timestamp

        Returns:
            Dictionary with current metric values and metadata
        """
        current_time = datetime.now()

        # Get basic metrics
        metrics_data = self.get_all_metrics()

        # Add histogram data
        histograms = {}
        for name, values in self.histograms.items():
            if values:
                histograms[name] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                    "percentiles": {
                        "p50": statistics.median(values),
                        "p90": self._calculate_percentile(values, 0.9),
                        "p95": self._calculate_percentile(values, 0.95),
                        "p99": self._calculate_percentile(values, 0.99)
                    }
                }
            else:
                histograms[name] = {
                    "count": 0,
                    "mean": 0,
                    "median": 0,
                    "min": 0,
                    "max": 0,
                    "std_dev": 0,
                    "percentiles": {"p50": 0, "p90": 0, "p95": 0, "p99": 0}
                }

        # Calculate rates for counters
        rates = {}
        for name, count in self.counters.items():
            # Simple rate calculation (would be enhanced with time-based tracking)
            rates[f"{name}_per_second"] = count / max(1, (current_time - self.collection_start_time).total_seconds())

        # Enhanced response with metadata
        return {
            "timestamp": current_time.isoformat(),
            "collection_duration_seconds": (current_time - self.collection_start_time).total_seconds(),
            "counters": metrics_data["counters"],
            "gauges": metrics_data["gauges"],
            "timers": metrics_data["timers"],
            "histograms": histograms,
            "rates": rates,
            "alerts": {
                "active_count": metrics_data["active_alerts"],
                "total_count": metrics_data["total_alerts"],
                "active_alerts": [
                    {
                        "name": alert["name"],
                        "severity": alert["severity"].value,
                        "message": alert["message"],
                        "triggered_at": alert["triggered_at"].isoformat(),
                        "value": alert["value"]
                    }
                    for alert in list(self.active_alerts.values())[:10]  # Limit to 10 most recent
                ]
            },
            "system": {
                "collection_active": metrics_data["collection_active"],
                "total_metrics": len(self.counters) + len(self.gauges) + len(self.timers) + len(self.histograms),
                "memory_usage_mb": self._estimate_memory_usage()
            }
        }

    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile value from list of values

        Args:
            values: List of numeric values
            percentile: Percentile to calculate (0.0 to 1.0)

        Returns:
            Percentile value
        """
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int(percentile * (len(sorted_values) - 1))
        return sorted_values[index]

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage of metrics collection in MB

        Returns:
            Estimated memory usage in megabytes
        """
        # Simple estimation based on number of stored values
        total_values = (
            len(self.counters) +
            len(self.gauges) +
            sum(len(values) for values in self.timers.values()) +
            sum(len(values) for values in self.histograms.values()) +
            len(self.alert_history)
        )

        # Rough estimate: 100 bytes per value + overhead
        estimated_bytes = total_values * 100 + 1024 * 1024  # 1MB overhead
        return estimated_bytes / (1024 * 1024)  # Convert to MB

    def check_alerts(self) -> List[Alert]:
        """Check and return active alerts as Alert objects

        Returns:
            List of active Alert objects
        """
        return [
            alert for alert in self.active_alerts.values()
            if not alert.resolved
        ]

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts

        Returns:
            List of active alert data
        """
        return [
            {
                "rule_name": key,
                "metric_name": alert.rule.metric_name,
                "severity": alert.rule.severity.value,
                "threshold": alert.rule.threshold,
                "current_value": alert.metric_value,
                "triggered_at": alert.triggered_at.isoformat(),
                "message": alert.message,
                "labels": alert.labels
            }
            for key, alert in self.active_alerts.items()
            if not alert.resolved
        ]

    def export_metrics(self, format_type: str = "json") -> str:
        """Export metrics in various formats

        Args:
            format_type: Export format ("json", "prometheus", "csv")

        Returns:
            Formatted metrics string
        """
        if format_type == "json":
            return self._export_json()
        elif format_type == "prometheus":
            return self._export_prometheus()
        elif format_type == "csv":
            return self._export_csv()
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _export_json(self) -> str:
        """Export metrics as JSON"""
        import json
        return json.dumps(self.get_all_metrics(), indent=2)

    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []

        for name, value in self.counters.items():
            lines.append(f"# TYPE {name}_total counter")
            lines.append(f"{name}_total {value}")

        for name, value in self.gauges.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")

        return "\n".join(lines)

    def _export_csv(self) -> str:
        """Export metrics as CSV"""
        lines = ["metric_name,metric_type,value,timestamp"]

        for name, values in self.metrics.items():
            for metric_value in values:
                lines.append(f"{name},{metric_value.metadata.get('type', 'unknown')},{metric_value.value},{metric_value.timestamp.isoformat()}")

        return "\n".join(lines)

    def time_operation(self, name: str, labels: Optional[Dict[str, str]] = None) -> 'TimerContext':
        """Create a timer context for measuring operation duration

        Args:
            name: Timer metric name
            labels: Additional labels for the timer

        Returns:
            TimerContext: Context manager for timing operations

        Usage:
            with metrics_collector.time_operation('api_request', {'endpoint': '/health'}):
                # Your operation here
                pass
        """
        return TimerContext(self, name, labels)


class TimerContext:
    """Context manager for timing operations"""

    def __init__(self, collector: MetricsCollector, name: str, labels: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.name = name
        self.labels = labels
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.collector.record_timer(self.name, duration, self.labels)


# Global metrics collector instance
metrics_collector = MetricsCollector()