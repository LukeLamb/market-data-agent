"""
Test suite for the MetricsCollector component

Tests cover all aspects of metrics collection including:
- Basic metric recording (counters, gauges, timers, rates, percentages)
- Statistical analysis and calculations
- Alert rule configuration and triggering
- Timer context managers
- Historical data management
- Performance and thread safety
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from collections import deque

from src.monitoring.metrics_collector import (
    MetricsCollector,
    MetricType,
    MetricValue,
    MetricSummary,
    AlertRule,
    Alert,
    AlertSeverity
)


class TestMetricsCollector:
    """Test suite for MetricsCollector functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.collector = MetricsCollector(max_history_size=100, collection_interval=0.1)

    def test_initialization(self):
        """Test collector initialization."""
        assert self.collector.max_history_size == 100
        assert self.collector.collection_interval == 0.1
        assert len(self.collector.metrics) == 0
        assert len(self.collector.counters) == 0
        assert len(self.collector.gauges) == 0

    def test_record_counter(self):
        """Test counter metric recording."""
        # Record single increment
        self.collector.record_counter('test_counter')
        assert self.collector.counters['test_counter'] == 1

        # Record multiple increments
        self.collector.record_counter('test_counter', 5)
        assert self.collector.counters['test_counter'] == 6

        # Test with tags
        self.collector.record_counter('tagged_counter', 1, {'environment': 'test'})
        assert self.collector.counters['tagged_counter'] == 1

    def test_record_gauge(self):
        """Test gauge metric recording."""
        # Record gauge value
        self.collector.record_gauge('cpu_usage', 75.5)
        assert self.collector.gauges['cpu_usage'] == 75.5

        # Update gauge value
        self.collector.record_gauge('cpu_usage', 80.2)
        assert self.collector.gauges['cpu_usage'] == 80.2

        # Test with tags
        self.collector.record_gauge('memory_usage', 65.3, {'host': 'server1'})
        assert self.collector.gauges['memory_usage'] == 65.3

    def test_record_histogram(self):
        """Test histogram metric recording."""
        # Record multiple values
        values = [10, 20, 30, 40, 50]
        for value in values:
            self.collector.record_histogram('response_time', value)

        # Verify metrics are recorded
        assert 'response_time' in self.collector.metrics
        metric_values = list(self.collector.metrics['response_time'])
        assert len(metric_values) == 5

        # Verify values are correct
        recorded_values = [mv.value for mv in metric_values]
        assert recorded_values == values

    def test_record_timer(self):
        """Test timer metric recording."""
        duration = 0.150  # 150ms
        self.collector.record_timer('api_call', duration)

        assert 'api_call' in self.collector.metrics
        metric_values = list(self.collector.metrics['api_call'])
        assert len(metric_values) == 1
        assert metric_values[0].value == duration
        assert metric_values[0].metric_type == MetricType.TIMER

    def test_record_rate(self):
        """Test rate metric recording."""
        rate = 150.5  # requests per second
        self.collector.record_rate('request_rate', rate)

        assert 'request_rate' in self.collector.metrics
        metric_values = list(self.collector.metrics['request_rate'])
        assert len(metric_values) == 1
        assert metric_values[0].value == rate
        assert metric_values[0].metric_type == MetricType.RATE

    def test_record_percentage(self):
        """Test percentage metric recording."""
        percentage = 95.5  # 95.5%
        self.collector.record_percentage('success_rate', percentage)

        assert 'success_rate' in self.collector.metrics
        metric_values = list(self.collector.metrics['success_rate'])
        assert len(metric_values) == 1
        assert metric_values[0].value == percentage
        assert metric_values[0].metric_type == MetricType.PERCENTAGE

    def test_timer_context_manager(self):
        """Test timer context manager."""
        with self.collector.time_operation('slow_operation') as timer:
            time.sleep(0.01)  # Sleep for 10ms

        # Verify timer recorded the operation
        assert 'slow_operation' in self.collector.metrics
        metric_values = list(self.collector.metrics['slow_operation'])
        assert len(metric_values) == 1
        assert metric_values[0].metric_type == MetricType.TIMER
        assert metric_values[0].value >= 0.01  # Should be at least 10ms

    def test_timer_context_manager_with_exception(self):
        """Test timer context manager handles exceptions properly."""
        with pytest.raises(ValueError):
            with self.collector.time_operation('failing_operation'):
                raise ValueError("Test exception")

        # Verify timer still recorded despite exception
        assert 'failing_operation' in self.collector.metrics
        metric_values = list(self.collector.metrics['failing_operation'])
        assert len(metric_values) == 1
        assert metric_values[0].metric_type == MetricType.TIMER

    def test_get_summary(self):
        """Test metric summary generation."""
        # Record multiple histogram values
        values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for value in values:
            self.collector.record_histogram('test_metric', value)

        summary = self.collector.get_summary('test_metric')

        assert summary.count == 10
        assert summary.mean == 55.0
        assert summary.min_value == 10
        assert summary.max_value == 100
        assert summary.p50 == 55.0  # Median
        assert summary.p95 == 95.0
        assert summary.p99 == 99.0
        assert summary.std_dev > 0

    def test_get_summary_empty_metric(self):
        """Test summary for non-existent metric."""
        summary = self.collector.get_summary('non_existent')

        assert summary.count == 0
        assert summary.mean == 0.0
        assert summary.min_value == 0.0
        assert summary.max_value == 0.0

    def test_get_current_metrics(self):
        """Test current metrics retrieval."""
        # Record various metrics
        self.collector.record_counter('requests', 100)
        self.collector.record_gauge('cpu_usage', 75.5)
        self.collector.record_timer('response_time', 0.150)

        metrics = self.collector.get_current_metrics()

        assert 'requests' in metrics
        assert 'cpu_usage' in metrics
        assert 'response_time' in metrics

        # Verify counter and gauge values
        assert metrics['requests'] == 100
        assert metrics['cpu_usage'] == 75.5

    def test_add_alert_rule(self):
        """Test alert rule configuration."""
        rule = AlertRule(
            name='high_cpu',
            metric_name='cpu_usage',
            condition="gt",
            threshold=80.0,
            severity=AlertSeverity.HIGH
        )

        self.collector.add_alert_rule(rule)
        assert 'high_cpu' in self.collector.alert_rules

    def test_check_alerts_triggered(self):
        """Test alert triggering."""
        # Add alert rule
        rule = AlertRule(
            name='high_cpu',
            metric_name='cpu_usage',
            condition="gt",
            threshold=80.0,
            severity=AlertSeverity.HIGH
        )
        self.collector.add_alert_rule(rule)

        # Record metric that should trigger alert
        self.collector.record_gauge('cpu_usage', 85.0)

        alerts = self.collector.check_alerts()
        assert len(alerts) == 1
        assert alerts[0].name == 'high_cpu'
        assert alerts[0].severity == AlertSeverity.HIGH

    def test_check_alerts_not_triggered(self):
        """Test alert not triggering when condition not met."""
        # Add alert rule
        rule = AlertRule(
            name='high_cpu',
            metric_name='cpu_usage',
            condition="gt",
            threshold=80.0,
            severity=AlertSeverity.HIGH
        )
        self.collector.add_alert_rule(rule)

        # Record metric that should NOT trigger alert
        self.collector.record_gauge('cpu_usage', 75.0)

        alerts = self.collector.check_alerts()
        assert len(alerts) == 0

    def test_alert_conditions(self):
        """Test different alert conditions."""
        test_cases = [
            ("gt", 75.0, 80.0, False),
            ("gt", 85.0, 80.0, True),
            ("lt", 85.0, 80.0, False),
            ("lt", 75.0, 80.0, True),
            ("eq", 80.0, 80.0, True),
            ("eq", 75.0, 80.0, False),
            ("ne", 75.0, 80.0, True),
            ("ne", 80.0, 80.0, False),
        ]

        for condition, value, threshold, should_trigger in test_cases:
            # Clear previous rules
            self.collector.alert_rules.clear()

            # Add rule with specific condition
            rule = AlertRule(
                name=f'test_{condition.value}',
                metric_name='test_metric',
                condition=condition,
                threshold=threshold,
                severity=AlertSeverity.MEDIUM
            )
            self.collector.add_alert_rule(rule)

            # Record metric value
            self.collector.record_gauge('test_metric', value)

            # Check alerts
            alerts = self.collector.check_alerts()

            if should_trigger:
                assert len(alerts) == 1, f"Alert should trigger for {condition.value}: {value} vs {threshold}"
            else:
                assert len(alerts) == 0, f"Alert should NOT trigger for {condition.value}: {value} vs {threshold}"

    def test_historical_data_management(self):
        """Test historical data size management."""
        collector = MetricsCollector(max_history_size=5)

        # Record more values than max history size
        for i in range(10):
            collector.record_histogram('test_metric', i)

        # Verify only max_history_size values are kept
        metric_values = list(collector.metrics['test_metric'])
        assert len(metric_values) == 5

        # Verify it kept the most recent values (5, 6, 7, 8, 9)
        recorded_values = [mv.value for mv in metric_values]
        assert recorded_values == [5, 6, 7, 8, 9]

    def test_metric_tags(self):
        """Test metric tagging functionality."""
        tags = {'environment': 'test', 'service': 'api'}

        self.collector.record_counter('tagged_counter', 1, tags)
        self.collector.record_gauge('tagged_gauge', 50.0, tags)
        self.collector.record_timer('tagged_timer', 0.100, tags)

        # Verify tags are stored with metrics
        for metric_name in ['tagged_counter', 'tagged_gauge', 'tagged_timer']:
            if metric_name in self.collector.metrics:
                metric_values = list(self.collector.metrics[metric_name])
                assert len(metric_values) > 0
                assert metric_values[0].tags == tags

    def test_thread_safety(self):
        """Test thread safety of metrics collection."""
        def record_metrics(thread_id):
            for i in range(100):
                self.collector.record_counter(f'thread_{thread_id}_counter', 1)
                self.collector.record_gauge(f'thread_{thread_id}_gauge', i)

        # Start multiple threads
        threads = []
        for thread_id in range(5):
            thread = threading.Thread(target=record_metrics, args=(thread_id,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify metrics from all threads
        for thread_id in range(5):
            counter_name = f'thread_{thread_id}_counter'
            gauge_name = f'thread_{thread_id}_gauge'

            assert self.collector.counters[counter_name] == 100
            assert gauge_name in self.collector.gauges

    def test_clear_metrics(self):
        """Test metrics clearing functionality."""
        # Record some metrics
        self.collector.record_counter('test_counter', 5)
        self.collector.record_gauge('test_gauge', 75.0)
        self.collector.record_timer('test_timer', 0.150)

        # Verify metrics exist
        assert len(self.collector.metrics) > 0
        assert len(self.collector.counters) > 0
        assert len(self.collector.gauges) > 0

        # Clear metrics
        self.collector.clear_metrics()

        # Verify metrics are cleared
        assert len(self.collector.metrics) == 0
        assert len(self.collector.counters) == 0
        assert len(self.collector.gauges) == 0

    def test_get_metric_names(self):
        """Test retrieving metric names."""
        # Record various metrics
        self.collector.record_counter('counter1')
        self.collector.record_gauge('gauge1', 50.0)
        self.collector.record_timer('timer1', 0.100)

        metric_names = self.collector.get_metric_names()

        expected_names = {'counter1', 'gauge1', 'timer1'}
        assert expected_names.issubset(set(metric_names))

    def test_performance_metrics(self):
        """Test performance monitoring capabilities."""
        # Record a large number of metrics to test performance
        start_time = time.time()

        for i in range(1000):
            self.collector.record_counter('perf_counter')
            self.collector.record_gauge('perf_gauge', i)
            self.collector.record_histogram('perf_histogram', i)

        end_time = time.time()
        duration = end_time - start_time

        # Should be able to record 3000 metrics quickly (< 1 second)
        assert duration < 1.0

        # Verify all metrics were recorded
        assert self.collector.counters['perf_counter'] == 1000
        assert self.collector.gauges['perf_gauge'] == 999
        assert len(self.collector.metrics['perf_histogram']) == 1000


class TestTimer:
    """Test suite for Timer utility class."""

    def test_timer_basic_usage(self):
        """Test basic timer functionality."""
        timer = Timer()

        # Start timer
        timer.start()
        time.sleep(0.01)  # 10ms
        duration = timer.stop()

        assert duration >= 0.01
        assert timer.duration >= 0.01

    def test_timer_context_manager(self):
        """Test timer as context manager."""
        with Timer() as timer:
            time.sleep(0.01)

        assert timer.duration >= 0.01

    def test_timer_multiple_measurements(self):
        """Test timer reset and multiple measurements."""
        timer = Timer()

        # First measurement
        timer.start()
        time.sleep(0.01)
        duration1 = timer.stop()

        # Reset and second measurement
        timer.reset()
        timer.start()
        time.sleep(0.02)
        duration2 = timer.stop()

        assert duration1 >= 0.01
        assert duration2 >= 0.02
        assert duration2 > duration1


if __name__ == '__main__':
    pytest.main([__file__])