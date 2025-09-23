"""
Tests for Request Batcher System
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from src.performance.request_batcher import (
    RequestBatcher, BatchRequest, BatchingStrategy,
    BatchProcessor, BatchResult
)


class TestRequestBatcher:
    """Test cases for RequestBatcher"""

    @pytest.fixture
    def mock_processor(self):
        """Mock batch processor"""
        processor = Mock(spec=BatchProcessor)
        processor.process_batch = AsyncMock(return_value=BatchResult(
            success=True,
            processed_count=5,
            results=[{"id": f"req_{i}", "status": "success"} for i in range(5)],
            errors=[],
            processing_time=0.1
        ))
        return processor

    @pytest.fixture
    def batcher(self, mock_processor):
        """Request batcher instance"""
        return RequestBatcher(
            processor=mock_processor,
            strategy=BatchingStrategy.TIME_BASED,
            max_batch_size=10,
            batch_timeout=1.0
        )

    def test_initialization(self, mock_processor):
        """Test batcher initialization"""
        batcher = RequestBatcher(
            processor=mock_processor,
            strategy=BatchingStrategy.SIZE_BASED,
            max_batch_size=5,
            batch_timeout=2.0
        )

        assert batcher.processor == mock_processor
        assert batcher.strategy == BatchingStrategy.SIZE_BASED
        assert batcher.max_batch_size == 5
        assert batcher.batch_timeout == 2.0
        assert not batcher.is_running

    @pytest.mark.asyncio
    async def test_single_request_processing(self, batcher):
        """Test processing a single request"""
        await batcher.start()

        request = BatchRequest(
            id="test_1",
            operation="get_price",
            data={"symbol": "AAPL"},
            timestamp=datetime.now()
        )

        result = await batcher.add_request(request)

        # Wait for processing
        await asyncio.sleep(1.5)

        assert result is not None
        batcher.processor.process_batch.assert_called_once()

        await batcher.stop()

    @pytest.mark.asyncio
    async def test_batch_size_trigger(self, mock_processor):
        """Test batching triggered by size"""
        batcher = RequestBatcher(
            processor=mock_processor,
            strategy=BatchingStrategy.SIZE_BASED,
            max_batch_size=3,
            batch_timeout=10.0
        )

        await batcher.start()

        # Add requests to trigger size-based batching
        requests = []
        for i in range(3):
            request = BatchRequest(
                id=f"test_{i}",
                operation="get_price",
                data={"symbol": f"STOCK_{i}"},
                timestamp=datetime.now()
            )
            requests.append(batcher.add_request(request))

        # Wait for batch processing
        await asyncio.gather(*requests)
        await asyncio.sleep(0.1)

        mock_processor.process_batch.assert_called_once()
        call_args = mock_processor.process_batch.call_args[0]
        assert len(call_args[0]) == 3  # Batch size

        await batcher.stop()

    @pytest.mark.asyncio
    async def test_time_based_batching(self, batcher):
        """Test time-based batching"""
        await batcher.start()

        request1 = BatchRequest(
            id="test_1",
            operation="get_price",
            data={"symbol": "AAPL"},
            timestamp=datetime.now()
        )

        request2 = BatchRequest(
            id="test_2",
            operation="get_price",
            data={"symbol": "GOOGL"},
            timestamp=datetime.now()
        )

        # Add requests with small delay
        task1 = batcher.add_request(request1)
        await asyncio.sleep(0.1)
        task2 = batcher.add_request(request2)

        # Wait for timeout-based processing
        await asyncio.gather(task1, task2)
        await asyncio.sleep(1.2)

        batcher.processor.process_batch.assert_called_once()

        await batcher.stop()

    @pytest.mark.asyncio
    async def test_priority_handling(self, batcher):
        """Test priority-based request handling"""
        await batcher.start()

        low_priority = BatchRequest(
            id="low",
            operation="get_price",
            data={"symbol": "AAPL"},
            timestamp=datetime.now(),
            priority=1
        )

        high_priority = BatchRequest(
            id="high",
            operation="get_price",
            data={"symbol": "GOOGL"},
            timestamp=datetime.now(),
            priority=10
        )

        # Add low priority first, then high priority
        await batcher.add_request(low_priority)
        await batcher.add_request(high_priority)

        await asyncio.sleep(1.2)

        # Verify batch was processed
        batcher.processor.process_batch.assert_called_once()
        call_args = batcher.processor.process_batch.call_args[0]
        batch = call_args[0]

        # High priority should be processed first
        assert batch[0].id == "high"
        assert batch[1].id == "low"

        await batcher.stop()

    @pytest.mark.asyncio
    async def test_request_deduplication(self, batcher):
        """Test duplicate request handling"""
        await batcher.start()

        request1 = BatchRequest(
            id="same_id",
            operation="get_price",
            data={"symbol": "AAPL"},
            timestamp=datetime.now()
        )

        request2 = BatchRequest(
            id="same_id",  # Same ID
            operation="get_price",
            data={"symbol": "AAPL"},
            timestamp=datetime.now()
        )

        # Add duplicate requests
        task1 = batcher.add_request(request1)
        task2 = batcher.add_request(request2)

        await asyncio.gather(task1, task2)
        await asyncio.sleep(1.2)

        batcher.processor.process_batch.assert_called_once()
        call_args = batcher.processor.process_batch.call_args[0]
        batch = call_args[0]

        # Only one request should be in batch
        assert len(batch) == 1
        assert batch[0].id == "same_id"

        await batcher.stop()

    @pytest.mark.asyncio
    async def test_request_expiration(self, mock_processor):
        """Test expired request handling"""
        batcher = RequestBatcher(
            processor=mock_processor,
            strategy=BatchingStrategy.TIME_BASED,
            max_batch_size=10,
            batch_timeout=0.1,  # Very short timeout
            request_timeout=0.05  # Very short request timeout
        )

        await batcher.start()

        request = BatchRequest(
            id="test_1",
            operation="get_price",
            data={"symbol": "AAPL"},
            timestamp=datetime.now() - timedelta(seconds=1),  # Already expired
            timeout=0.05
        )

        result = await batcher.add_request(request)
        await asyncio.sleep(0.2)

        # Expired requests should not be processed
        mock_processor.process_batch.assert_not_called()

        await batcher.stop()

    @pytest.mark.asyncio
    async def test_batch_processing_error(self, mock_processor):
        """Test error handling in batch processing"""
        mock_processor.process_batch.side_effect = Exception("Processing error")

        batcher = RequestBatcher(
            processor=mock_processor,
            strategy=BatchingStrategy.TIME_BASED,
            max_batch_size=10,
            batch_timeout=0.5
        )

        await batcher.start()

        request = BatchRequest(
            id="test_1",
            operation="get_price",
            data={"symbol": "AAPL"},
            timestamp=datetime.now()
        )

        result = await batcher.add_request(request)
        await asyncio.sleep(0.6)

        # Error should be handled gracefully
        mock_processor.process_batch.assert_called_once()

        await batcher.stop()

    @pytest.mark.asyncio
    async def test_adaptive_batching(self, mock_processor):
        """Test adaptive batching strategy"""
        batcher = RequestBatcher(
            processor=mock_processor,
            strategy=BatchingStrategy.ADAPTIVE,
            max_batch_size=10,
            batch_timeout=1.0
        )

        await batcher.start()

        # Simulate high load
        requests = []
        for i in range(5):
            request = BatchRequest(
                id=f"test_{i}",
                operation="get_price",
                data={"symbol": f"STOCK_{i}"},
                timestamp=datetime.now()
            )
            requests.append(batcher.add_request(request))

        await asyncio.gather(*requests)
        await asyncio.sleep(0.1)

        # Adaptive strategy should adjust batching behavior
        mock_processor.process_batch.assert_called()

        await batcher.stop()

    @pytest.mark.asyncio
    async def test_batcher_statistics(self, batcher):
        """Test batcher statistics collection"""
        await batcher.start()

        request = BatchRequest(
            id="test_1",
            operation="get_price",
            data={"symbol": "AAPL"},
            timestamp=datetime.now()
        )

        await batcher.add_request(request)
        await asyncio.sleep(1.2)

        stats = batcher.get_statistics()

        assert stats['total_requests'] >= 1
        assert stats['total_batches'] >= 1
        assert 'average_batch_size' in stats
        assert 'average_processing_time' in stats

        await batcher.stop()

    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, batcher):
        """Test concurrent request handling"""
        await batcher.start()

        # Create many concurrent requests
        requests = []
        for i in range(20):
            request = BatchRequest(
                id=f"concurrent_{i}",
                operation="get_price",
                data={"symbol": f"STOCK_{i}"},
                timestamp=datetime.now()
            )
            requests.append(batcher.add_request(request))

        # Wait for all requests to complete
        await asyncio.gather(*requests)
        await asyncio.sleep(1.5)

        # Verify processing occurred
        assert batcher.processor.process_batch.call_count >= 1

        stats = batcher.get_statistics()
        assert stats['total_requests'] >= 20

        await batcher.stop()

    def test_batch_request_creation(self):
        """Test BatchRequest creation and methods"""
        request = BatchRequest(
            id="test_1",
            operation="get_price",
            data={"symbol": "AAPL"},
            timestamp=datetime.now(),
            priority=5,
            timeout=30.0
        )

        assert request.id == "test_1"
        assert request.operation == "get_price"
        assert request.data["symbol"] == "AAPL"
        assert request.priority == 5
        assert not request.is_expired()

        # Test expiration
        old_request = BatchRequest(
            id="old",
            operation="get_price",
            data={"symbol": "AAPL"},
            timestamp=datetime.now() - timedelta(seconds=60),
            timeout=30.0
        )
        assert old_request.is_expired()

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, batcher):
        """Test graceful shutdown with pending requests"""
        await batcher.start()

        # Add requests but don't wait for completion
        for i in range(5):
            request = BatchRequest(
                id=f"shutdown_test_{i}",
                operation="get_price",
                data={"symbol": f"STOCK_{i}"},
                timestamp=datetime.now()
            )
            # Don't await - leave pending
            asyncio.create_task(batcher.add_request(request))

        # Small delay to ensure requests are queued
        await asyncio.sleep(0.1)

        # Shutdown should handle pending requests gracefully
        await batcher.stop()

        assert not batcher.is_running

    @pytest.mark.asyncio
    async def test_batch_result_handling(self, mock_processor):
        """Test batch result processing"""
        mock_processor.process_batch.return_value = BatchResult(
            success=True,
            processed_count=2,
            results=[
                {"id": "test_1", "status": "success", "data": {"price": 150.0}},
                {"id": "test_2", "status": "success", "data": {"price": 2800.0}}
            ],
            errors=[],
            processing_time=0.05
        )

        batcher = RequestBatcher(
            processor=mock_processor,
            strategy=BatchingStrategy.TIME_BASED,
            max_batch_size=10,
            batch_timeout=0.5
        )

        await batcher.start()

        request1 = BatchRequest(
            id="test_1",
            operation="get_price",
            data={"symbol": "AAPL"},
            timestamp=datetime.now()
        )

        request2 = BatchRequest(
            id="test_2",
            operation="get_price",
            data={"symbol": "GOOGL"},
            timestamp=datetime.now()
        )

        task1 = batcher.add_request(request1)
        task2 = batcher.add_request(request2)

        results = await asyncio.gather(task1, task2)

        assert len(results) == 2
        assert all(result is not None for result in results)

        await batcher.stop()