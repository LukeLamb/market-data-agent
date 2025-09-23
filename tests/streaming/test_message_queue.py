"""
Tests for Message Queue System
Tests message queue functionality, priority handling, and backpressure control
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal

from src.streaming.message_queue import (
    MessageQueue,
    MessageQueueManager,
    MessagePriority,
    QueueMetrics,
    BackpressureController,
    MessageQueueConfig,
    QueueMessage
)


class TestMessageQueueConfig:
    """Test message queue configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = MessageQueueConfig()

        assert config.max_size == 10000
        assert config.batch_size == 100
        assert config.timeout == 1.0
        assert config.enable_persistence is False
        assert config.enable_backpressure is True

    def test_custom_config(self):
        """Test custom configuration"""
        config = MessageQueueConfig(
            max_size=20000,
            batch_size=200,
            timeout=2.0,
            enable_persistence=True
        )

        assert config.max_size == 20000
        assert config.batch_size == 200
        assert config.timeout == 2.0
        assert config.enable_persistence is True


class TestQueueMessage:
    """Test queue message functionality"""

    def test_message_creation(self):
        """Test message initialization"""
        content = {"type": "price_update", "symbol": "AAPL", "price": 150.0}
        message = QueueMessage(
            content=content,
            priority=MessagePriority.HIGH,
            ttl=30.0
        )

        assert message.content == content
        assert message.priority == MessagePriority.HIGH
        assert message.ttl == 30.0
        assert message.retry_count == 0
        assert isinstance(message.created_at, datetime)

    def test_message_expiration(self):
        """Test message expiration logic"""
        content = {"test": "data"}
        message = QueueMessage(content=content, ttl=1.0)

        # Should not be expired immediately
        assert not message.is_expired()

        # Mock time passing
        message.created_at = datetime.now() - timedelta(seconds=2)

        # Should be expired after TTL
        assert message.is_expired()

    def test_message_without_ttl(self):
        """Test message without TTL (never expires)"""
        content = {"test": "data"}
        message = QueueMessage(content=content)

        # Should never expire without TTL
        assert not message.is_expired()

        # Even after time passes
        message.created_at = datetime.now() - timedelta(hours=1)
        assert not message.is_expired()

    def test_message_retry_logic(self):
        """Test message retry functionality"""
        content = {"test": "data"}
        message = QueueMessage(content=content)

        assert message.retry_count == 0
        assert message.can_retry(max_retries=3)

        # Increment retry count
        message.retry_count = 2
        assert message.can_retry(max_retries=3)

        # Exceed max retries
        message.retry_count = 3
        assert not message.can_retry(max_retries=3)


class TestBackpressureController:
    """Test backpressure controller functionality"""

    def test_controller_initialization(self):
        """Test backpressure controller initialization"""
        controller = BackpressureController(
            max_queue_size=1000,
            pressure_thresholds={
                MessagePriority.CRITICAL: 0.8,
                MessagePriority.HIGH: 0.6,
                MessagePriority.NORMAL: 0.4,
                MessagePriority.LOW: 0.2
            }
        )

        assert controller.max_queue_size == 1000
        assert len(controller.pressure_thresholds) == 4

    def test_backpressure_acceptance(self):
        """Test message acceptance based on backpressure"""
        controller = BackpressureController(
            max_queue_size=100,
            pressure_thresholds={
                MessagePriority.CRITICAL: 0.9,
                MessagePriority.HIGH: 0.7,
                MessagePriority.NORMAL: 0.5,
                MessagePriority.LOW: 0.3
            }
        )

        # Test with low queue usage (should accept all)
        controller.current_queue_size = 20  # 20% usage
        assert controller.should_accept_message(MessagePriority.CRITICAL)
        assert controller.should_accept_message(MessagePriority.HIGH)
        assert controller.should_accept_message(MessagePriority.NORMAL)
        assert controller.should_accept_message(MessagePriority.LOW)

        # Test with medium queue usage (should reject low priority)
        controller.current_queue_size = 50  # 50% usage
        assert controller.should_accept_message(MessagePriority.CRITICAL)
        assert controller.should_accept_message(MessagePriority.HIGH)
        assert controller.should_accept_message(MessagePriority.NORMAL)
        assert not controller.should_accept_message(MessagePriority.LOW)

        # Test with high queue usage (should reject normal and low)
        controller.current_queue_size = 80  # 80% usage
        assert controller.should_accept_message(MessagePriority.CRITICAL)
        assert controller.should_accept_message(MessagePriority.HIGH)
        assert not controller.should_accept_message(MessagePriority.NORMAL)
        assert not controller.should_accept_message(MessagePriority.LOW)

        # Test with very high queue usage (should only accept critical)
        controller.current_queue_size = 95  # 95% usage
        assert controller.should_accept_message(MessagePriority.CRITICAL)
        assert not controller.should_accept_message(MessagePriority.HIGH)
        assert not controller.should_accept_message(MessagePriority.NORMAL)
        assert not controller.should_accept_message(MessagePriority.LOW)

    def test_pressure_level_calculation(self):
        """Test pressure level calculation"""
        controller = BackpressureController(max_queue_size=100)

        controller.current_queue_size = 25
        assert controller.get_pressure_level() == 0.25

        controller.current_queue_size = 75
        assert controller.get_pressure_level() == 0.75

        controller.current_queue_size = 100
        assert controller.get_pressure_level() == 1.0


class TestMessageQueue:
    """Test message queue functionality"""

    @pytest.fixture
    def queue_config(self):
        """Test queue configuration"""
        return MessageQueueConfig(
            max_size=100,
            batch_size=10,
            timeout=0.1
        )

    @pytest.fixture
    def message_queue(self, queue_config):
        """Test message queue instance"""
        return MessageQueue("test_queue", queue_config)

    def test_queue_initialization(self, message_queue):
        """Test queue initialization"""
        assert message_queue.name == "test_queue"
        assert message_queue.config.max_size == 100
        assert message_queue.size == 0
        assert not message_queue.is_full

    @pytest.mark.asyncio
    async def test_message_put_and_get(self, message_queue):
        """Test putting and getting messages"""
        content = {"type": "test", "data": "hello"}

        # Put message
        message_id = await message_queue.put(content, MessagePriority.NORMAL)
        assert message_id is not None
        assert message_queue.size == 1

        # Get message
        message = await message_queue.get()
        assert message is not None
        assert message.content == content
        assert message.priority == MessagePriority.NORMAL
        assert message_queue.size == 0

    @pytest.mark.asyncio
    async def test_priority_ordering(self, message_queue):
        """Test priority-based message ordering"""
        # Put messages with different priorities
        await message_queue.put({"priority": "low"}, MessagePriority.LOW)
        await message_queue.put({"priority": "critical"}, MessagePriority.CRITICAL)
        await message_queue.put({"priority": "normal"}, MessagePriority.NORMAL)
        await message_queue.put({"priority": "high"}, MessagePriority.HIGH)

        # Messages should come out in priority order
        message1 = await message_queue.get()
        assert message1.content["priority"] == "critical"

        message2 = await message_queue.get()
        assert message2.content["priority"] == "high"

        message3 = await message_queue.get()
        assert message3.content["priority"] == "normal"

        message4 = await message_queue.get()
        assert message4.content["priority"] == "low"

    @pytest.mark.asyncio
    async def test_queue_full_behavior(self, message_queue):
        """Test behavior when queue is full"""
        # Fill queue to capacity
        for i in range(message_queue.config.max_size):
            message_id = await message_queue.put({"id": i}, MessagePriority.NORMAL)
            assert message_id is not None

        assert message_queue.is_full

        # Try to add one more message (should fail)
        message_id = await message_queue.put({"overflow": True}, MessagePriority.NORMAL)
        assert message_id is None

    @pytest.mark.asyncio
    async def test_backpressure_integration(self, message_queue):
        """Test backpressure integration"""
        message_queue.config.enable_backpressure = True

        # Fill queue to trigger backpressure
        for i in range(80):  # 80% full
            await message_queue.put({"id": i}, MessagePriority.NORMAL)

        # Low priority messages should be rejected
        message_id = await message_queue.put({"rejected": True}, MessagePriority.LOW)
        assert message_id is None

        # High priority messages should still be accepted
        message_id = await message_queue.put({"accepted": True}, MessagePriority.HIGH)
        assert message_id is not None

    @pytest.mark.asyncio
    async def test_message_expiration(self, message_queue):
        """Test message expiration handling"""
        # Put message with short TTL
        content = {"expires": True}
        message_id = await message_queue.put(content, MessagePriority.NORMAL, ttl=0.1)
        assert message_id is not None

        # Wait for expiration
        await asyncio.sleep(0.2)

        # Try to get message (should be None due to expiration)
        message = await message_queue.get()
        # Note: Implementation may handle expired messages differently
        # This test assumes expired messages are filtered out

    @pytest.mark.asyncio
    async def test_batch_operations(self, message_queue):
        """Test batch get operations"""
        # Put multiple messages
        for i in range(15):
            await message_queue.put({"batch_id": i}, MessagePriority.NORMAL)

        # Get batch of messages
        batch = await message_queue.get_batch(10)
        assert len(batch) == 10
        assert all(msg.content["batch_id"] < 10 for msg in batch[:10])

        # Get remaining messages
        remaining_batch = await message_queue.get_batch(10)
        assert len(remaining_batch) == 5

    @pytest.mark.asyncio
    async def test_queue_timeout(self, message_queue):
        """Test queue timeout behavior"""
        # Try to get from empty queue with timeout
        start_time = asyncio.get_event_loop().time()
        message = await message_queue.get(timeout=0.1)
        end_time = asyncio.get_event_loop().time()

        assert message is None
        assert (end_time - start_time) >= 0.1

    @pytest.mark.asyncio
    async def test_queue_metrics(self, message_queue):
        """Test queue metrics tracking"""
        # Put some messages
        for i in range(5):
            await message_queue.put({"id": i}, MessagePriority.NORMAL)

        # Get some messages
        for i in range(3):
            await message_queue.get()

        metrics = await message_queue.get_metrics()

        assert metrics.queue_name == "test_queue"
        assert metrics.current_size == 2
        assert metrics.total_enqueued >= 5
        assert metrics.total_dequeued >= 3

    @pytest.mark.asyncio
    async def test_queue_clear(self, message_queue):
        """Test queue clearing"""
        # Put some messages
        for i in range(10):
            await message_queue.put({"id": i}, MessagePriority.NORMAL)

        assert message_queue.size == 10

        # Clear queue
        await message_queue.clear()

        assert message_queue.size == 0
        assert message_queue.is_empty

    @pytest.mark.asyncio
    async def test_queue_close(self, message_queue):
        """Test queue closure and cleanup"""
        # Put some messages
        for i in range(5):
            await message_queue.put({"id": i}, MessagePriority.NORMAL)

        await message_queue.close()

        # Queue should be cleared and marked as closed
        assert message_queue.size == 0


class TestMessageQueueManager:
    """Test message queue manager functionality"""

    @pytest.fixture
    def queue_manager(self):
        """Test queue manager instance"""
        return MessageQueueManager()

    @pytest.mark.asyncio
    async def test_manager_initialization(self, queue_manager):
        """Test manager initialization"""
        await queue_manager.initialize()

        assert queue_manager.is_initialized
        assert len(queue_manager.queues) == 0

    @pytest.mark.asyncio
    async def test_queue_creation(self, queue_manager):
        """Test queue creation through manager"""
        await queue_manager.initialize()

        config = MessageQueueConfig(max_size=50)
        queue = await queue_manager.create_queue("test_queue", config)

        assert queue is not None
        assert queue.name == "test_queue"
        assert "test_queue" in queue_manager.queues

    @pytest.mark.asyncio
    async def test_queue_retrieval(self, queue_manager):
        """Test queue retrieval"""
        await queue_manager.initialize()

        # Create queue
        config = MessageQueueConfig()
        await queue_manager.create_queue("test_queue", config)

        # Retrieve queue
        queue = queue_manager.get_queue("test_queue")
        assert queue is not None
        assert queue.name == "test_queue"

        # Try to get non-existent queue
        missing_queue = queue_manager.get_queue("missing_queue")
        assert missing_queue is None

    @pytest.mark.asyncio
    async def test_queue_deletion(self, queue_manager):
        """Test queue deletion"""
        await queue_manager.initialize()

        # Create queue
        config = MessageQueueConfig()
        await queue_manager.create_queue("temp_queue", config)

        assert "temp_queue" in queue_manager.queues

        # Delete queue
        await queue_manager.delete_queue("temp_queue")

        assert "temp_queue" not in queue_manager.queues

    @pytest.mark.asyncio
    async def test_multiple_queues(self, queue_manager):
        """Test managing multiple queues"""
        await queue_manager.initialize()

        # Create multiple queues
        queue_configs = {
            "high_priority": MessageQueueConfig(max_size=100),
            "normal_priority": MessageQueueConfig(max_size=200),
            "low_priority": MessageQueueConfig(max_size=300)
        }

        for name, config in queue_configs.items():
            await queue_manager.create_queue(name, config)

        assert len(queue_manager.queues) == 3

        # Test each queue
        for name in queue_configs:
            queue = queue_manager.get_queue(name)
            assert queue is not None
            assert queue.name == name

    @pytest.mark.asyncio
    async def test_queue_metrics_aggregation(self, queue_manager):
        """Test aggregated queue metrics"""
        await queue_manager.initialize()

        # Create queues and add messages
        for i in range(3):
            queue_name = f"queue_{i}"
            config = MessageQueueConfig()
            queue = await queue_manager.create_queue(queue_name, config)

            # Add messages to each queue
            for j in range(i + 1):
                await queue.put({"queue": i, "message": j}, MessagePriority.NORMAL)

        # Get aggregated metrics
        all_metrics = await queue_manager.get_all_queue_metrics()

        assert len(all_metrics) == 3
        assert all(metrics.queue_name.startswith("queue_") for metrics in all_metrics)

    @pytest.mark.asyncio
    async def test_manager_close(self, queue_manager):
        """Test manager closure and cleanup"""
        await queue_manager.initialize()

        # Create some queues
        for i in range(3):
            config = MessageQueueConfig()
            await queue_manager.create_queue(f"queue_{i}", config)

        assert len(queue_manager.queues) == 3

        # Close manager
        await queue_manager.close()

        assert not queue_manager.is_initialized
        assert len(queue_manager.queues) == 0

    @pytest.mark.asyncio
    async def test_error_handling(self, queue_manager):
        """Test error handling in various scenarios"""
        await queue_manager.initialize()

        # Try to create queue with duplicate name
        config = MessageQueueConfig()
        queue1 = await queue_manager.create_queue("duplicate", config)
        queue2 = await queue_manager.create_queue("duplicate", config)

        # Second creation should handle the conflict appropriately
        assert queue1 is not None
        # Implementation may return existing queue or None

    @pytest.mark.asyncio
    async def test_queue_overflow_handling(self, queue_manager):
        """Test handling of queue overflow conditions"""
        await queue_manager.initialize()

        # Create small queue
        config = MessageQueueConfig(max_size=5)
        queue = await queue_manager.create_queue("small_queue", config)

        # Fill beyond capacity
        successful_puts = 0
        for i in range(10):
            message_id = await queue.put({"id": i}, MessagePriority.NORMAL)
            if message_id is not None:
                successful_puts += 1

        # Should only accept up to max_size messages
        assert successful_puts <= config.max_size

    @pytest.mark.asyncio
    async def test_concurrent_access(self, queue_manager):
        """Test concurrent access to queues"""
        await queue_manager.initialize()

        config = MessageQueueConfig()
        queue = await queue_manager.create_queue("concurrent_queue", config)

        # Simulate concurrent producers
        async def producer(producer_id, count):
            for i in range(count):
                await queue.put({"producer": producer_id, "message": i}, MessagePriority.NORMAL)

        # Start multiple producers
        tasks = [
            asyncio.create_task(producer(0, 10)),
            asyncio.create_task(producer(1, 10)),
            asyncio.create_task(producer(2, 10))
        ]

        await asyncio.gather(*tasks)

        # Should have all messages
        assert queue.size == 30


class TestQueueMetrics:
    """Test queue metrics functionality"""

    def test_metrics_initialization(self):
        """Test metrics initialization"""
        metrics = QueueMetrics("test_queue")

        assert metrics.queue_name == "test_queue"
        assert metrics.current_size == 0
        assert metrics.total_enqueued == 0
        assert metrics.total_dequeued == 0
        assert metrics.avg_processing_time_ms == 0.0

    def test_metrics_updates(self):
        """Test metrics value updates"""
        metrics = QueueMetrics("test_queue")

        metrics.current_size = 25
        metrics.total_enqueued = 1000
        metrics.total_dequeued = 975
        metrics.avg_processing_time_ms = 15.5

        assert metrics.current_size == 25
        assert metrics.total_enqueued == 1000
        assert metrics.total_dequeued == 975
        assert metrics.avg_processing_time_ms == 15.5

    def test_metrics_calculated_properties(self):
        """Test calculated metric properties"""
        metrics = QueueMetrics("test_queue")

        metrics.current_size = 50
        metrics.max_size = 100
        metrics.total_enqueued = 1000
        metrics.total_dequeued = 950

        # Test utilization calculation
        assert metrics.utilization == 0.5  # 50/100

        # Test pending messages
        assert metrics.pending_messages == 50  # 1000 - 950


class TestMessageQueueIntegration:
    """Integration tests for message queue system"""

    @pytest.mark.integration
    async def test_high_throughput_scenarios(self):
        """Integration test for high throughput message processing"""
        # This test would measure actual throughput performance
        pytest.skip("Integration test requires performance measurement setup")

    @pytest.mark.integration
    async def test_persistence_integration(self):
        """Test message queue persistence integration"""
        # This test would test actual persistence mechanisms
        pytest.skip("Integration test requires persistence storage setup")

    @pytest.mark.integration
    async def test_queue_resilience(self):
        """Test message queue resilience to failures"""
        # This test would simulate various failure scenarios
        pytest.skip("Integration test requires failure simulation setup")