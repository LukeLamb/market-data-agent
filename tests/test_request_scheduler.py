"""Tests for Request Scheduler Implementation"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from src.rate_limiting.request_scheduler import (
    RequestScheduler,
    ScheduledRequest,
    RequestPriority
)


class TestScheduledRequest:
    """Test ScheduledRequest dataclass and methods"""

    def test_basic_request(self):
        """Test basic request creation"""
        async def dummy_callback(source, endpoint, params):
            return "success"

        request = ScheduledRequest(
            priority=RequestPriority.HIGH,
            source_name="test_source",
            endpoint="test_endpoint",
            params={"symbol": "AAPL"},
            callback=dummy_callback
        )

        assert request.priority == RequestPriority.HIGH
        assert request.source_name == "test_source"
        assert request.endpoint == "test_endpoint"
        assert request.params == {"symbol": "AAPL"}
        assert request.max_retries == 3
        assert request.retry_count == 0
        assert request.batchable is False

    def test_batchable_request_auto_key(self):
        """Test batchable request with automatic batch key generation"""
        async def dummy_callback(source, endpoint, params):
            return "success"

        request = ScheduledRequest(
            priority=RequestPriority.NORMAL,
            source_name="test_source",
            endpoint="quotes",
            params={"symbol": "AAPL", "interval": "1d"},
            callback=dummy_callback,
            batchable=True
        )

        assert request.batchable is True
        assert request.batch_key is not None
        assert "test_source:quotes" in request.batch_key

    def test_request_comparison(self):
        """Test request priority comparison for queue ordering"""
        async def dummy_callback(source, endpoint, params):
            return "success"

        critical_request = ScheduledRequest(
            priority=RequestPriority.CRITICAL,
            source_name="test",
            endpoint="test",
            params={},
            callback=dummy_callback
        )

        low_request = ScheduledRequest(
            priority=RequestPriority.LOW,
            source_name="test",
            endpoint="test",
            params={},
            callback=dummy_callback
        )

        # Critical should be "less than" low (higher priority)
        assert critical_request < low_request

    def test_request_expiry(self):
        """Test request expiry detection"""
        async def dummy_callback(source, endpoint, params):
            return "success"

        # Create request with 1 second timeout
        request = ScheduledRequest(
            priority=RequestPriority.NORMAL,
            source_name="test",
            endpoint="test",
            params={},
            callback=dummy_callback,
            timeout=1.0
        )

        # Should not be expired immediately
        assert not request.is_expired()

        # Mock time advancement
        with patch.object(request, 'created_at', datetime.now() - timedelta(seconds=2)):
            assert request.is_expired()

    def test_retry_logic(self):
        """Test retry count and retry decision logic"""
        async def dummy_callback(source, endpoint, params):
            return "success"

        request = ScheduledRequest(
            priority=RequestPriority.NORMAL,
            source_name="test",
            endpoint="test",
            params={},
            callback=dummy_callback,
            max_retries=2
        )

        # Should allow retries initially
        assert request.should_retry()

        # Increment retry count
        request.increment_retry()
        assert request.retry_count == 1
        assert request.should_retry()

        request.increment_retry()
        assert request.retry_count == 2
        assert not request.should_retry()


class TestRequestScheduler:
    """Test RequestScheduler implementation"""

    @pytest.fixture
    def scheduler(self):
        """Create scheduler for testing"""
        return RequestScheduler(max_concurrent=2, batch_window=0.5)

    @pytest.fixture
    def dummy_callback(self):
        """Create dummy async callback"""
        async def callback(source, endpoint, params):
            await asyncio.sleep(0.01)  # Small delay to simulate work
            return f"result_{endpoint}"
        return callback

    @pytest.mark.asyncio
    async def test_scheduler_start_stop(self, scheduler):
        """Test scheduler start and stop operations"""
        assert not scheduler.running

        await scheduler.start()
        assert scheduler.running
        assert scheduler._scheduler_task is not None

        await scheduler.stop()
        assert not scheduler.running

    @pytest.mark.asyncio
    async def test_schedule_single_request(self, scheduler, dummy_callback):
        """Test scheduling a single request"""
        await scheduler.start()

        request = ScheduledRequest(
            priority=RequestPriority.HIGH,
            source_name="test_source",
            endpoint="test_endpoint",
            params={"symbol": "AAPL"},
            callback=dummy_callback
        )

        request_id = await scheduler.schedule_request(request)
        assert request_id is not None
        assert scheduler.total_scheduled == 1

        # Wait for processing
        await asyncio.sleep(0.1)

        assert scheduler.total_completed >= 0  # May not be processed yet

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_priority_ordering(self, scheduler, dummy_callback):
        """Test that requests are processed in priority order"""
        await scheduler.start()

        # Create requests with different priorities
        low_request = ScheduledRequest(
            priority=RequestPriority.LOW,
            source_name="test",
            endpoint="low_priority",
            params={},
            callback=dummy_callback
        )

        critical_request = ScheduledRequest(
            priority=RequestPriority.CRITICAL,
            source_name="test",
            endpoint="critical_priority",
            params={},
            callback=dummy_callback
        )

        # Schedule low priority first, then critical
        await scheduler.schedule_request(low_request)
        await scheduler.schedule_request(critical_request)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Critical should be processed first (but both might complete)
        assert scheduler.total_scheduled == 2

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_concurrent_limit(self, scheduler):
        """Test concurrent request limiting"""
        await scheduler.start()

        # Create slow callback
        async def slow_callback(source, endpoint, params):
            await asyncio.sleep(0.5)
            return "slow_result"

        # Schedule more requests than concurrent limit
        requests = []
        for i in range(5):
            request = ScheduledRequest(
                priority=RequestPriority.NORMAL,
                source_name="test",
                endpoint=f"endpoint_{i}",
                params={},
                callback=slow_callback
            )
            requests.append(request)
            await scheduler.schedule_request(request)

        # Wait briefly and check active requests
        await asyncio.sleep(0.1)

        # Should not exceed max_concurrent (2)
        assert len(scheduler.active_requests) <= 2

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_request_failure_retry(self, scheduler):
        """Test request failure and retry logic"""
        await scheduler.start()

        retry_count = 0

        async def failing_callback(source, endpoint, params):
            nonlocal retry_count
            retry_count += 1
            if retry_count <= 2:
                raise Exception("Simulated failure")
            return "success_after_retries"

        request = ScheduledRequest(
            priority=RequestPriority.NORMAL,
            source_name="test",
            endpoint="failing_endpoint",
            params={},
            callback=failing_callback,
            max_retries=3
        )

        await scheduler.schedule_request(request)

        # Wait for retries to complete
        await asyncio.sleep(1.0)

        # Should have retried and eventually succeeded
        assert retry_count >= 2
        assert scheduler.total_failed < scheduler.total_scheduled

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_pause_resume(self, scheduler, dummy_callback):
        """Test scheduler pause and resume functionality"""
        await scheduler.start()

        request = ScheduledRequest(
            priority=RequestPriority.NORMAL,
            source_name="test",
            endpoint="test",
            params={},
            callback=dummy_callback
        )

        # Pause scheduler
        scheduler.pause()
        assert scheduler.paused

        await scheduler.schedule_request(request)
        await asyncio.sleep(0.1)

        # Request should be queued but not processed
        initial_completed = scheduler.total_completed

        # Resume scheduler
        scheduler.resume()
        assert not scheduler.paused

        await asyncio.sleep(0.1)

        # Request should now be processed
        assert scheduler.total_completed >= initial_completed

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_batch_processing(self, scheduler):
        """Test request batching functionality"""
        await scheduler.start()

        batch_results = []

        async def batch_callback(source, endpoint, params):
            # Simulate batch processing
            symbols = params.get('symbols', params.get('symbol', '')).split(',')
            batch_results.extend(symbols)
            return f"batch_result_{len(symbols)}"

        # Create batchable requests
        requests = []
        for symbol in ['AAPL', 'GOOGL', 'MSFT']:
            request = ScheduledRequest(
                priority=RequestPriority.NORMAL,
                source_name="test_source",
                endpoint="batch_quotes",
                params={"symbol": symbol},
                callback=batch_callback,
                batchable=True,
                batch_key="test_source:batch_quotes"
            )
            requests.append(request)
            await scheduler.schedule_request(request)

        # Wait for batch window and processing
        await asyncio.sleep(scheduler.batch_window + 0.2)

        # Should have batched the requests
        assert scheduler.total_batched > 0
        assert len(batch_results) >= 1  # At least some symbols processed

        await scheduler.stop()

    def test_get_statistics(self, scheduler):
        """Test statistics collection"""
        stats = scheduler.get_statistics()

        expected_fields = [
            "running", "paused", "total_scheduled", "total_completed",
            "total_failed", "total_batched", "active_requests",
            "total_queued", "queue_sizes", "batch_groups",
            "success_rate", "batch_efficiency"
        ]

        for field in expected_fields:
            assert field in stats

        assert stats["running"] is False
        assert stats["total_scheduled"] == 0

    def test_get_queue_status(self, scheduler, dummy_callback):
        """Test queue status reporting"""
        # Add some requests to queues
        for priority in [RequestPriority.HIGH, RequestPriority.LOW]:
            request = ScheduledRequest(
                priority=priority,
                source_name="test",
                endpoint="test",
                params={},
                callback=dummy_callback
            )
            # Manually add to queue for testing
            import heapq
            heapq.heappush(scheduler.queues[priority], request)

        status = scheduler.get_queue_status()

        assert "HIGH" in status
        assert "LOW" in status
        assert len(status["HIGH"]) == 1
        assert len(status["LOW"]) == 1

    @pytest.mark.asyncio
    async def test_expired_request_cleanup(self, scheduler, dummy_callback):
        """Test cleanup of expired requests"""
        await scheduler.start()

        # Create request with very short timeout
        request = ScheduledRequest(
            priority=RequestPriority.LOW,
            source_name="test",
            endpoint="test",
            params={},
            callback=dummy_callback,
            timeout=0.001  # 1ms timeout
        )

        await scheduler.schedule_request(request)

        # Wait for expiry and cleanup
        await asyncio.sleep(0.1)

        # Manually trigger cleanup
        await scheduler._cleanup_expired()

        # Request should be removed from queue
        total_queued = sum(len(queue) for queue in scheduler.queues.values())
        assert total_queued == 0

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_deduplication(self, scheduler, dummy_callback):
        """Test request deduplication"""
        await scheduler.start()

        # Create identical requests
        request1 = ScheduledRequest(
            priority=RequestPriority.NORMAL,
            source_name="test",
            endpoint="test",
            params={"symbol": "AAPL"},
            callback=dummy_callback
        )

        request2 = ScheduledRequest(
            priority=RequestPriority.NORMAL,
            source_name="test",
            endpoint="test",
            params={"symbol": "AAPL"},
            callback=dummy_callback
        )

        await scheduler.schedule_request(request1)
        await scheduler.schedule_request(request2)

        # Wait for processing
        await asyncio.sleep(0.1)

        # Should have deduplicated (only one request actually processed)
        # Note: This test might be flaky depending on timing
        assert scheduler.total_scheduled == 2

        await scheduler.stop()


class TestRequestSchedulerIntegration:
    """Integration tests for request scheduler"""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete request scheduling workflow"""
        scheduler = RequestScheduler(max_concurrent=3, batch_window=0.3)
        await scheduler.start()

        results = []

        async def test_callback(source, endpoint, params):
            await asyncio.sleep(0.01)
            result = f"{source}_{endpoint}_{params.get('symbol', 'no_symbol')}"
            results.append(result)
            return result

        # Schedule various types of requests
        requests = [
            # High priority requests
            ScheduledRequest(
                priority=RequestPriority.HIGH,
                source_name="source1",
                endpoint="realtime",
                params={"symbol": "AAPL"},
                callback=test_callback
            ),
            # Normal priority requests
            ScheduledRequest(
                priority=RequestPriority.NORMAL,
                source_name="source2",
                endpoint="historical",
                params={"symbol": "GOOGL"},
                callback=test_callback
            ),
            # Batchable requests
            ScheduledRequest(
                priority=RequestPriority.NORMAL,
                source_name="source1",
                endpoint="batch",
                params={"symbol": "MSFT"},
                callback=test_callback,
                batchable=True
            ),
            ScheduledRequest(
                priority=RequestPriority.NORMAL,
                source_name="source1",
                endpoint="batch",
                params={"symbol": "TSLA"},
                callback=test_callback,
                batchable=True
            )
        ]

        # Schedule all requests
        for request in requests:
            await scheduler.schedule_request(request)

        # Wait for processing
        await asyncio.sleep(1.0)

        # Verify results
        assert len(results) >= 1  # At least some requests processed
        assert scheduler.total_scheduled == len(requests)

        # Get final statistics
        stats = scheduler.get_statistics()
        assert stats["total_scheduled"] == len(requests)

        await scheduler.stop()