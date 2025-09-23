"""
Message Queue System for Real-Time Data Distribution
Handles reliable message delivery with backpressure and fault tolerance
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import weakref
import heapq
from collections import defaultdict, deque
import uuid

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Message priority levels for queue processing"""
    CRITICAL = 1      # System alerts, errors
    HIGH = 2          # Real-time price updates
    NORMAL = 3        # Regular data updates
    LOW = 4           # Analytics, reports


class QueueStatus(Enum):
    """Message queue status"""
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class QueueMessage:
    """Message wrapper for queue processing"""
    id: str
    content: Dict[str, Any]
    priority: MessagePriority
    created_at: datetime
    retry_count: int = 0
    max_retries: int = 3
    expires_at: Optional[datetime] = None

    def __lt__(self, other):
        """Priority queue comparison (lower priority value = higher priority)"""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.created_at < other.created_at

    @property
    def is_expired(self) -> bool:
        """Check if message has expired"""
        return self.expires_at is not None and datetime.now() > self.expires_at

    @property
    def age_seconds(self) -> float:
        """Get message age in seconds"""
        return (datetime.now() - self.created_at).total_seconds()


@dataclass
class QueueConfig:
    """Message queue configuration"""
    max_size: int = 100000
    max_memory_mb: int = 100
    batch_size: int = 100
    processing_interval: float = 0.01  # 10ms
    retry_delay: float = 1.0
    max_retries: int = 3
    message_ttl: float = 300.0  # 5 minutes
    enable_persistence: bool = False
    persistence_file: Optional[str] = None


@dataclass
class QueueMetrics:
    """Queue performance metrics"""
    messages_queued: int = 0
    messages_processed: int = 0
    messages_failed: int = 0
    messages_expired: int = 0
    current_queue_size: int = 0
    avg_processing_time_ms: float = 0.0
    throughput_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class BackpressureController:
    """Backpressure control for queue management"""

    def __init__(self, max_queue_size: int, max_memory_mb: int):
        self.max_queue_size = max_queue_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_memory_usage = 0
        self.pressure_level = 0.0  # 0.0 to 1.0

    def update_pressure(self, queue_size: int, memory_usage: int) -> None:
        """Update backpressure level based on current metrics"""
        size_pressure = queue_size / self.max_queue_size
        memory_pressure = memory_usage / self.max_memory_bytes

        self.pressure_level = max(size_pressure, memory_pressure)
        self.current_memory_usage = memory_usage

    def should_accept_message(self, message_priority: MessagePriority) -> bool:
        """Determine if message should be accepted based on backpressure"""
        if self.pressure_level < 0.8:
            return True

        # Under high pressure, only accept high priority messages
        if self.pressure_level < 0.95:
            return message_priority.value <= MessagePriority.HIGH.value

        # Critical pressure, only critical messages
        return message_priority == MessagePriority.CRITICAL

    def get_processing_delay(self) -> float:
        """Get processing delay based on pressure level"""
        if self.pressure_level < 0.5:
            return 0.01  # Normal processing speed
        elif self.pressure_level < 0.8:
            return 0.02  # Slightly slower
        else:
            return 0.05  # Significantly slower under pressure


class MessageQueue:
    """High-performance priority message queue with backpressure control"""

    def __init__(self, name: str, config: QueueConfig = None):
        self.name = name
        self.config = config or QueueConfig()

        # Queue storage
        self.priority_queue: List[QueueMessage] = []
        self.message_lookup: Dict[str, QueueMessage] = {}

        # Processing
        self.processor_task: Optional[asyncio.Task] = None
        self.status = QueueStatus.STOPPED
        self.handlers: List[Callable[[QueueMessage], Any]] = []

        # Backpressure control
        self.backpressure = BackpressureController(
            self.config.max_size,
            self.config.max_memory_mb
        )

        # Metrics
        self.metrics = QueueMetrics()
        self.processing_times: deque = deque(maxlen=1000)  # Last 1000 processing times

        # Retry queue
        self.retry_queue: List[Tuple[float, QueueMessage]] = []  # (retry_time, message)

    async def start(self) -> None:
        """Start queue processing"""
        if self.status == QueueStatus.RUNNING:
            return

        self.status = QueueStatus.RUNNING
        self.processor_task = asyncio.create_task(self._processing_loop())

        logger.info(f"Message queue '{self.name}' started")

    async def stop(self) -> None:
        """Stop queue processing"""
        if self.status == QueueStatus.STOPPED:
            return

        self.status = QueueStatus.STOPPED

        if self.processor_task and not self.processor_task.done():
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass

        logger.info(f"Message queue '{self.name}' stopped")

    async def put(
        self,
        content: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        ttl: Optional[float] = None
    ) -> Optional[str]:
        """Add message to queue"""

        # Check backpressure
        if not self.backpressure.should_accept_message(priority):
            logger.warning(f"Message rejected due to backpressure in queue '{self.name}'")
            return None

        # Create message
        message_id = str(uuid.uuid4())
        expires_at = None
        if ttl is not None:
            expires_at = datetime.now() + timedelta(seconds=ttl)
        elif self.config.message_ttl > 0:
            expires_at = datetime.now() + timedelta(seconds=self.config.message_ttl)

        message = QueueMessage(
            id=message_id,
            content=content,
            priority=priority,
            created_at=datetime.now(),
            max_retries=self.config.max_retries,
            expires_at=expires_at
        )

        # Add to queue
        heapq.heappush(self.priority_queue, message)
        self.message_lookup[message_id] = message

        # Update metrics
        self.metrics.messages_queued += 1
        self._update_queue_metrics()

        return message_id

    async def put_batch(
        self,
        messages: List[Tuple[Dict[str, Any], MessagePriority]],
        ttl: Optional[float] = None
    ) -> List[Optional[str]]:
        """Add multiple messages to queue efficiently"""
        message_ids = []

        for content, priority in messages:
            message_id = await self.put(content, priority, ttl)
            message_ids.append(message_id)

        return message_ids

    def add_handler(self, handler: Callable[[QueueMessage], Any]) -> None:
        """Add message handler"""
        self.handlers.append(handler)

    def remove_handler(self, handler: Callable[[QueueMessage], Any]) -> None:
        """Remove message handler"""
        if handler in self.handlers:
            self.handlers.remove(handler)

    async def _processing_loop(self) -> None:
        """Main message processing loop"""
        while self.status == QueueStatus.RUNNING:
            try:
                # Process retry queue first
                await self._process_retry_queue()

                # Process main queue
                await self._process_messages()

                # Clean up expired messages
                await self._cleanup_expired_messages()

                # Update metrics
                self._update_queue_metrics()

                # Adaptive delay based on backpressure
                delay = self.backpressure.get_processing_delay()
                await asyncio.sleep(delay)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in processing loop for queue '{self.name}': {e}")
                await asyncio.sleep(1.0)  # Error recovery delay

    async def _process_messages(self) -> None:
        """Process messages from the priority queue"""
        processed_count = 0
        start_time = time.perf_counter()

        while (
            self.priority_queue and
            processed_count < self.config.batch_size and
            self.status == QueueStatus.RUNNING
        ):
            message = heapq.heappop(self.priority_queue)

            # Remove from lookup
            if message.id in self.message_lookup:
                del self.message_lookup[message.id]

            # Check if expired
            if message.is_expired:
                self.metrics.messages_expired += 1
                continue

            # Process message
            success = await self._process_single_message(message)

            if success:
                self.metrics.messages_processed += 1
            else:
                # Handle retry
                if message.retry_count < message.max_retries:
                    message.retry_count += 1
                    retry_time = time.time() + (self.config.retry_delay * message.retry_count)
                    heapq.heappush(self.retry_queue, (retry_time, message))
                else:
                    self.metrics.messages_failed += 1
                    logger.warning(f"Message {message.id} failed after {message.retry_count} retries")

            processed_count += 1

        # Update processing time metrics
        if processed_count > 0:
            processing_time = (time.perf_counter() - start_time) * 1000
            self.processing_times.append(processing_time)

    async def _process_single_message(self, message: QueueMessage) -> bool:
        """Process a single message with all handlers"""
        try:
            for handler in self.handlers:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            return True

        except Exception as e:
            logger.error(f"Error processing message {message.id}: {e}")
            return False

    async def _process_retry_queue(self) -> None:
        """Process messages in retry queue"""
        current_time = time.time()

        while self.retry_queue and self.retry_queue[0][0] <= current_time:
            retry_time, message = heapq.heappop(self.retry_queue)

            # Check if still valid
            if not message.is_expired:
                # Add back to main queue
                heapq.heappush(self.priority_queue, message)
                self.message_lookup[message.id] = message

    async def _cleanup_expired_messages(self) -> None:
        """Remove expired messages from queues"""
        # Clean main queue
        valid_messages = []
        while self.priority_queue:
            message = heapq.heappop(self.priority_queue)
            if not message.is_expired:
                valid_messages.append(message)
            else:
                self.metrics.messages_expired += 1
                if message.id in self.message_lookup:
                    del self.message_lookup[message.id]

        self.priority_queue = valid_messages
        heapq.heapify(self.priority_queue)

        # Clean retry queue
        current_time = time.time()
        valid_retries = []
        while self.retry_queue:
            retry_time, message = heapq.heappop(self.retry_queue)
            if not message.is_expired:
                valid_retries.append((retry_time, message))
            else:
                self.metrics.messages_expired += 1

        self.retry_queue = valid_retries
        heapq.heapify(self.retry_queue)

    def _update_queue_metrics(self) -> None:
        """Update queue performance metrics"""
        self.metrics.current_queue_size = len(self.priority_queue) + len(self.retry_queue)

        # Calculate memory usage (approximate)
        estimated_memory = 0
        for message in self.priority_queue:
            estimated_memory += len(json.dumps(message.content).encode('utf-8'))
        self.metrics.memory_usage_mb = estimated_memory / (1024 * 1024)

        # Update backpressure
        self.backpressure.update_pressure(
            self.metrics.current_queue_size,
            int(estimated_memory)
        )

        # Calculate processing metrics
        if self.processing_times:
            self.metrics.avg_processing_time_ms = sum(self.processing_times) / len(self.processing_times)

        # Calculate throughput
        if self.metrics.last_updated:
            time_diff = (datetime.now() - self.metrics.last_updated).total_seconds()
            if time_diff > 0:
                self.metrics.throughput_per_second = self.metrics.messages_processed / time_diff

        self.metrics.last_updated = datetime.now()

    def get_status(self) -> Dict[str, Any]:
        """Get queue status and metrics"""
        return {
            "name": self.name,
            "status": self.status.value,
            "metrics": {
                "messages_queued": self.metrics.messages_queued,
                "messages_processed": self.metrics.messages_processed,
                "messages_failed": self.metrics.messages_failed,
                "messages_expired": self.metrics.messages_expired,
                "current_queue_size": self.metrics.current_queue_size,
                "avg_processing_time_ms": self.metrics.avg_processing_time_ms,
                "throughput_per_second": self.metrics.throughput_per_second,
                "memory_usage_mb": self.metrics.memory_usage_mb
            },
            "backpressure": {
                "pressure_level": self.backpressure.pressure_level,
                "memory_usage_mb": self.backpressure.current_memory_usage / (1024 * 1024)
            },
            "config": {
                "max_size": self.config.max_size,
                "max_memory_mb": self.config.max_memory_mb,
                "batch_size": self.config.batch_size,
                "max_retries": self.config.max_retries
            }
        }

    async def clear(self) -> int:
        """Clear all messages from queue"""
        cleared_count = len(self.priority_queue) + len(self.retry_queue)

        self.priority_queue.clear()
        self.retry_queue.clear()
        self.message_lookup.clear()

        self._update_queue_metrics()
        logger.info(f"Cleared {cleared_count} messages from queue '{self.name}'")

        return cleared_count


class MessageQueueManager:
    """Manager for multiple message queues with load balancing"""

    def __init__(self):
        self.queues: Dict[str, MessageQueue] = {}
        self.round_robin_index = 0

    async def create_queue(self, name: str, config: QueueConfig = None) -> MessageQueue:
        """Create a new message queue"""
        if name in self.queues:
            raise ValueError(f"Queue '{name}' already exists")

        queue = MessageQueue(name, config)
        self.queues[name] = queue
        await queue.start()

        logger.info(f"Created queue '{name}'")
        return queue

    async def get_queue(self, name: str) -> Optional[MessageQueue]:
        """Get queue by name"""
        return self.queues.get(name)

    async def remove_queue(self, name: str) -> bool:
        """Remove and stop a queue"""
        if name not in self.queues:
            return False

        queue = self.queues[name]
        await queue.stop()
        del self.queues[name]

        logger.info(f"Removed queue '{name}'")
        return True

    async def broadcast(
        self,
        content: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        exclude_queues: List[str] = None
    ) -> Dict[str, Optional[str]]:
        """Broadcast message to all queues"""
        exclude_queues = exclude_queues or []
        results = {}

        for queue_name, queue in self.queues.items():
            if queue_name not in exclude_queues:
                message_id = await queue.put(content, priority)
                results[queue_name] = message_id

        return results

    def get_load_balanced_queue(self) -> Optional[MessageQueue]:
        """Get queue using round-robin load balancing"""
        if not self.queues:
            return None

        queue_names = list(self.queues.keys())
        queue_name = queue_names[self.round_robin_index % len(queue_names)]
        self.round_robin_index += 1

        return self.queues[queue_name]

    def get_least_loaded_queue(self) -> Optional[MessageQueue]:
        """Get queue with lowest current load"""
        if not self.queues:
            return None

        return min(
            self.queues.values(),
            key=lambda q: q.metrics.current_queue_size + q.backpressure.pressure_level
        )

    def get_manager_stats(self) -> Dict[str, Any]:
        """Get comprehensive manager statistics"""
        total_queued = sum(q.metrics.messages_queued for q in self.queues.values())
        total_processed = sum(q.metrics.messages_processed for q in self.queues.values())
        total_failed = sum(q.metrics.messages_failed for q in self.queues.values())
        total_queue_size = sum(q.metrics.current_queue_size for q in self.queues.values())

        return {
            "total_queues": len(self.queues),
            "total_messages_queued": total_queued,
            "total_messages_processed": total_processed,
            "total_messages_failed": total_failed,
            "total_current_queue_size": total_queue_size,
            "queue_details": {
                name: queue.get_status()
                for name, queue in self.queues.items()
            }
        }

    async def stop_all(self) -> None:
        """Stop all queues"""
        stop_tasks = []
        for queue in self.queues.values():
            stop_tasks.append(queue.stop())

        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)

        logger.info("Stopped all message queues")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop_all()


# Global message queue manager
queue_manager = MessageQueueManager()