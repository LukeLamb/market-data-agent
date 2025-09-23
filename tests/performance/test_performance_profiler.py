"""
Tests for Performance Profiler System
"""

import pytest
import asyncio
import time
from unittest.mock import Mock

from src.performance.performance_profiler import (
    PerformanceProfiler, PerformanceMetrics, MetricType
)


class TestPerformanceProfiler:
    """Test cases for PerformanceProfiler"""

    @pytest.fixture
    def profiler(self):
        """Performance profiler instance"""
        return PerformanceProfiler(
            collection_interval=0.01,
            history_size=100
        )

    def test_initialization(self):
        """Test profiler initialization"""
        profiler = PerformanceProfiler(
            collection_interval=0.05,
            history_size=50
        )

        assert profiler.collection_interval == 0.05
        assert profiler.history_size == 50

    def test_metric_recording(self, profiler):
        """Test metric recording functionality"""
        # Record a metric
        profiler.record_metric(MetricType.RESPONSE_TIME, 0.150)
        profiler.record_metric(MetricType.THROUGHPUT, 100.0)

        # Get current metrics
        metrics = profiler.get_current_metrics()

        assert metrics is not None
        assert hasattr(metrics, 'response_times')
        assert hasattr(metrics, 'throughput')

    def test_operation_profiling(self, profiler):
        """Test operation profiling"""
        operation_name = "test_operation"

        # Start operation
        profiler.start_operation(operation_name)

        # Simulate work
        time.sleep(0.05)

        # End operation
        duration = profiler.end_operation(operation_name)

        assert duration > 0
        assert duration >= 0.05

    @pytest.mark.asyncio
    async def test_async_operation_profiling(self, profiler):
        """Test async operation profiling"""
        operation_name = "async_test_operation"

        # Profile async operation
        profiler.start_operation(operation_name)
        await asyncio.sleep(0.05)
        duration = profiler.end_operation(operation_name)

        assert duration > 0

    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics creation"""
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            response_times=[0.1, 0.15, 0.2],
            throughput=150.0,
            error_rate=0.01,
            memory_usage_mb=128.5,
            cpu_usage_percent=45.2
        )

        assert len(metrics.response_times) == 3
        assert metrics.throughput == 150.0
        assert metrics.error_rate == 0.01

    def test_metric_statistics(self, profiler):
        """Test metric statistics calculation"""
        # Record multiple metrics
        response_times = [0.1, 0.15, 0.2, 0.12, 0.18]
        for rt in response_times:
            profiler.record_metric(MetricType.RESPONSE_TIME, rt)

        stats = profiler.get_statistics()

        assert 'response_time' in stats
        assert 'avg' in stats['response_time']
        assert 'min' in stats['response_time']
        assert 'max' in stats['response_time']

    def test_profiling_history(self, profiler):
        """Test profiling history tracking"""
        # Record some operations
        for i in range(5):
            operation_name = f"operation_{i}"
            profiler.start_operation(operation_name)
            time.sleep(0.01)
            profiler.end_operation(operation_name)

        history = profiler.get_operation_history()
        assert len(history) >= 5

    def test_metric_types(self):
        """Test metric type enumeration"""
        # Verify all expected metric types exist
        expected_types = [
            'RESPONSE_TIME', 'THROUGHPUT', 'ERROR_RATE',
            'MEMORY_USAGE', 'CPU_USAGE', 'QUEUE_DEPTH', 'CACHE_HIT_RATE'
        ]

        for metric_type in expected_types:
            assert hasattr(MetricType, metric_type)

    def test_concurrent_profiling(self, profiler):
        """Test concurrent operation profiling"""
        import threading

        def worker(worker_id):
            operation_name = f"worker_{worker_id}"
            profiler.start_operation(operation_name)
            time.sleep(0.02)
            return profiler.end_operation(operation_name)

        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check that all operations were recorded
        history = profiler.get_operation_history()
        assert len(history) >= 3

    def test_profiler_reset(self, profiler):
        """Test profiler reset functionality"""
        # Record some metrics
        profiler.record_metric(MetricType.RESPONSE_TIME, 0.1)
        profiler.record_metric(MetricType.THROUGHPUT, 100.0)

        # Start an operation
        profiler.start_operation("test_op")

        # Reset profiler
        profiler.reset()

        # Verify data is cleared
        metrics = profiler.get_current_metrics()
        assert len(metrics.response_times) == 0

    def test_threshold_monitoring(self, profiler):
        """Test threshold monitoring"""
        # Set a threshold
        profiler.set_threshold(MetricType.RESPONSE_TIME, 0.1)

        # Record metrics above and below threshold
        profiler.record_metric(MetricType.RESPONSE_TIME, 0.05)  # Below
        profiler.record_metric(MetricType.RESPONSE_TIME, 0.15)  # Above

        violations = profiler.get_threshold_violations()
        assert len(violations) >= 1

    def test_profiler_context_manager(self, profiler):
        """Test profiler as context manager for operations"""
        operation_name = "context_test"

        with profiler.profile_operation(operation_name):
            time.sleep(0.05)

        history = profiler.get_operation_history()
        assert any(op['name'] == operation_name for op in history)

    @pytest.mark.asyncio
    async def test_background_collection(self, profiler):
        """Test background metric collection"""
        # Start background collection
        profiler.start_background_collection()

        # Let it collect for a short time
        await asyncio.sleep(0.1)

        # Stop collection
        profiler.stop_background_collection()

        # Should have collected some metrics
        metrics = profiler.get_current_metrics()
        assert metrics is not None

    def test_export_metrics(self, profiler):
        """Test metrics export functionality"""
        # Record some metrics
        profiler.record_metric(MetricType.RESPONSE_TIME, 0.1)
        profiler.record_metric(MetricType.THROUGHPUT, 150.0)

        # Export to dictionary
        exported = profiler.export_metrics()

        assert isinstance(exported, dict)
        assert 'timestamp' in exported
        assert 'metrics' in exported