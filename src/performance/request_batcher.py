"""
Request Batcher

Intelligent request batching system to optimize API usage by grouping
similar requests and reducing overall network overhead.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BatchingStrategy(Enum):
    """Batching strategies for different use cases"""
    TIME_BASED = "time_based"  # Batch after time window
    SIZE_BASED = "size_based"  # Batch when size threshold reached
    HYBRID = "hybrid"  # Combination of time and size
    ADAPTIVE = "adaptive"  # Adaptive based on load patterns
    IMMEDIATE = "immediate"  # Process immediately (no batching)


@dataclass
class BatchRequest:
    """Individual request in a batch"""
    id: str
    operation: str
    data: Dict[str, Any]
    timestamp: datetime
    priority: int = 1  # 1-10, higher is more urgent
    callback: Optional[Callable] = None
    timeout: Optional[float] = None
    retries: int = 0
    max_retries: int = 3

    def __hash__(self) -> int:
        return hash(self.id)

    def is_expired(self, timeout_seconds: float = 30.0) -> bool:
        """Check if request has expired"""
        if self.timeout:
            timeout_seconds = self.timeout
        return (datetime.now() - self.timestamp).total_seconds() > timeout_seconds


@dataclass
class BatchResult:
    """Result of batch processing"""
    batch_id: str
    requests: List[BatchRequest]
    results: Dict[str, Any]
    success_count: int
    error_count: int
    processing_time: float
    timestamp: datetime
    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        total = self.success_count + self.error_count
        return (self.success_count / total * 100) if total > 0 else 0.0


class BatchProcessor:
    """Processes batches of requests efficiently"""

    def __init__(self):
        self.processors: Dict[str, Callable] = {}

    def register_processor(self, operation: str, processor: Callable):
        """Register a batch processor for an operation"""
        self.processors[operation] = processor
        logger.info(f"Registered batch processor for operation: {operation}")

    async def process_batch(self, requests: List[BatchRequest]) -> BatchResult:
        """Process a batch of requests"""
        start_time = time.time()
        batch_id = f"batch_{int(time.time() * 1000)}"

        results = {}
        errors = {}
        success_count = 0
        error_count = 0

        # Group requests by operation
        grouped_requests = defaultdict(list)
        for request in requests:
            grouped_requests[request.operation].append(request)

        # Process each operation group
        for operation, op_requests in grouped_requests.items():
            try:
                if operation in self.processors:
                    # Use registered batch processor
                    processor = self.processors[operation]
                    batch_results = await self._call_processor(processor, op_requests)

                    for req, result in zip(op_requests, batch_results):
                        if isinstance(result, Exception):
                            errors[req.id] = str(result)
                            error_count += 1
                        else:
                            results[req.id] = result
                            success_count += 1

                            # Call individual callback if provided
                            if req.callback:
                                try:
                                    await self._call_callback(req.callback, result)
                                except Exception as e:
                                    logger.error(f"Error in callback for request {req.id}: {e}")
                else:
                    # No processor available, mark as error
                    for req in op_requests:
                        errors[req.id] = f"No processor registered for operation: {operation}"
                        error_count += 1

            except Exception as e:
                logger.error(f"Error processing operation {operation}: {e}")
                for req in op_requests:
                    errors[req.id] = str(e)
                    error_count += 1

        processing_time = time.time() - start_time

        return BatchResult(
            batch_id=batch_id,
            requests=requests,
            results=results,
            success_count=success_count,
            error_count=error_count,
            processing_time=processing_time,
            timestamp=datetime.now(),
            errors=errors
        )

    async def _call_processor(self, processor: Callable, requests: List[BatchRequest]) -> List[Any]:
        """Call batch processor with proper async handling"""
        if asyncio.iscoroutinefunction(processor):
            return await processor(requests)
        else:
            return processor(requests)

    async def _call_callback(self, callback: Callable, result: Any):
        """Call callback with proper async handling"""
        if asyncio.iscoroutinefunction(callback):
            await callback(result)
        else:
            callback(result)


class RequestBatcher:
    """
    Intelligent request batcher that optimizes API usage by grouping
    similar requests and processing them efficiently.
    """

    def __init__(
        self,
        strategy: BatchingStrategy = BatchingStrategy.HYBRID,
        max_batch_size: int = 50,
        max_wait_time: float = 1.0,  # seconds
        max_concurrent_batches: int = 10,
        enable_metrics: bool = True
    ):
        """Initialize request batcher

        Args:
            strategy: Batching strategy to use
            max_batch_size: Maximum requests per batch
            max_wait_time: Maximum time to wait before processing batch
            max_concurrent_batches: Maximum concurrent batch processing
            enable_metrics: Enable performance metrics
        """
        self.strategy = strategy
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.max_concurrent_batches = max_concurrent_batches
        self.enable_metrics = enable_metrics

        # Request queues organized by operation
        self._queues: Dict[str, deque] = defaultdict(deque)
        self._pending_requests: Dict[str, BatchRequest] = {}

        # Batch processing
        self.processor = BatchProcessor()
        self._active_batches: Set[asyncio.Task] = set()
        self._batch_semaphore = asyncio.Semaphore(max_concurrent_batches)

        # Background tasks
        self._running = False
        self._batch_task = None
        self._cleanup_task = None

        # Statistics
        self.stats = {
            'total_requests': 0,
            'total_batches': 0,
            'average_batch_size': 0.0,
            'average_processing_time': 0.0,
            'success_rate': 0.0,
            'queue_sizes': defaultdict(int)
        }

        # Adaptive strategy state
        self._load_history = deque(maxlen=100)
        self._performance_history = deque(maxlen=50)

        logger.info(f"Initialized RequestBatcher with strategy: {strategy.value}")

    async def start(self):
        """Start background processing"""
        self._running = True
        if self._batch_task is None:
            self._batch_task = asyncio.create_task(self._batch_processor())
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_processor())
        logger.info("RequestBatcher started")

    async def stop(self):
        """Stop background processing"""
        self._running = False

        # Cancel background tasks
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Wait for active batches to complete
        if self._active_batches:
            await asyncio.gather(*self._active_batches, return_exceptions=True)

        logger.info("RequestBatcher stopped")

    def register_processor(self, operation: str, processor: Callable):
        """Register a batch processor for an operation"""
        self.processor.register_processor(operation, processor)

    async def add_request(
        self,
        operation: str,
        data: Dict[str, Any],
        priority: int = 1,
        callback: Optional[Callable] = None,
        timeout: Optional[float] = None,
        request_id: Optional[str] = None
    ) -> str:
        """Add request to batch queue"""
        if request_id is None:
            request_id = f"{operation}_{int(time.time() * 1000000)}"

        request = BatchRequest(
            id=request_id,
            operation=operation,
            data=data,
            timestamp=datetime.now(),
            priority=priority,
            callback=callback,
            timeout=timeout
        )

        # Add to queue
        self._queues[operation].append(request)
        self._pending_requests[request_id] = request

        # Update stats
        self.stats['total_requests'] += 1
        self.stats['queue_sizes'][operation] = len(self._queues[operation])

        # Trigger immediate processing if strategy allows
        if (self.strategy == BatchingStrategy.IMMEDIATE or
            (self.strategy == BatchingStrategy.SIZE_BASED and
             len(self._queues[operation]) >= self.max_batch_size)):
            asyncio.create_task(self._process_operation_queue(operation))

        return request_id

    async def get_result(self, request_id: str, timeout: float = 30.0) -> Any:
        """Get result for a specific request"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if request_id not in self._pending_requests:
                # Request has been processed, but we need to track results
                # This is a simplified implementation - in production you'd want
                # a proper result store
                break
            await asyncio.sleep(0.1)

        # For now, return None - in a full implementation you'd store results
        return None

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        return {
            'queues': {op: len(queue) for op, queue in self._queues.items()},
            'pending_requests': len(self._pending_requests),
            'active_batches': len(self._active_batches),
            'stats': self.stats.copy()
        }

    async def _batch_processor(self):
        """Background task to process batches"""
        while self._running:
            try:
                # Determine which operations to process
                operations_to_process = []

                for operation, queue in self._queues.items():
                    if not queue:
                        continue

                    should_process = False

                    if self.strategy == BatchingStrategy.SIZE_BASED:
                        should_process = len(queue) >= self.max_batch_size
                    elif self.strategy == BatchingStrategy.TIME_BASED:
                        oldest_request = queue[0]
                        time_waiting = (datetime.now() - oldest_request.timestamp).total_seconds()
                        should_process = time_waiting >= self.max_wait_time
                    elif self.strategy == BatchingStrategy.HYBRID:
                        oldest_request = queue[0]
                        time_waiting = (datetime.now() - oldest_request.timestamp).total_seconds()
                        should_process = (len(queue) >= self.max_batch_size or
                                        time_waiting >= self.max_wait_time)
                    elif self.strategy == BatchingStrategy.ADAPTIVE:
                        should_process = self._should_process_adaptive(operation, queue)

                    if should_process:
                        operations_to_process.append(operation)

                # Process selected operations
                for operation in operations_to_process:
                    asyncio.create_task(self._process_operation_queue(operation))

                # Sleep before next iteration
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                await asyncio.sleep(1.0)

    async def _process_operation_queue(self, operation: str):
        """Process queue for a specific operation"""
        async with self._batch_semaphore:
            queue = self._queues[operation]
            if not queue:
                return

            # Extract batch
            batch_requests = []
            while queue and len(batch_requests) < self.max_batch_size:
                request = queue.popleft()

                # Check if request has expired
                if not request.is_expired():
                    batch_requests.append(request)
                else:
                    # Remove from pending
                    self._pending_requests.pop(request.id, None)

            if not batch_requests:
                return

            # Process batch
            try:
                task = asyncio.create_task(self._execute_batch(batch_requests))
                self._active_batches.add(task)

                result = await task

                # Update statistics
                self._update_stats(result)

                # Record performance for adaptive strategy
                if self.strategy == BatchingStrategy.ADAPTIVE:
                    self._record_performance(result)

            except Exception as e:
                logger.error(f"Error processing batch for operation {operation}: {e}")
            finally:
                if task in self._active_batches:
                    self._active_batches.remove(task)

    async def _execute_batch(self, requests: List[BatchRequest]) -> BatchResult:
        """Execute a batch of requests"""
        # Remove from pending
        for request in requests:
            self._pending_requests.pop(request.id, None)

        # Process batch
        result = await self.processor.process_batch(requests)

        logger.info(f"Processed batch {result.batch_id}: "
                   f"{result.success_count} successes, {result.error_count} errors, "
                   f"{result.processing_time:.3f}s")

        return result

    def _should_process_adaptive(self, operation: str, queue: deque) -> bool:
        """Determine if we should process queue using adaptive strategy"""
        if not queue:
            return False

        # Consider current load
        current_load = sum(len(q) for q in self._queues.values())
        self._load_history.append(current_load)

        # Adjust thresholds based on load
        if len(self._load_history) >= 10:
            avg_load = sum(self._load_history) / len(self._load_history)

            if current_load > avg_load * 1.5:
                # High load - process more aggressively
                return len(queue) >= max(1, self.max_batch_size // 2)
            elif current_load < avg_load * 0.5:
                # Low load - wait for larger batches
                return len(queue) >= self.max_batch_size

        # Default hybrid logic
        oldest_request = queue[0]
        time_waiting = (datetime.now() - oldest_request.timestamp).total_seconds()
        return (len(queue) >= self.max_batch_size or time_waiting >= self.max_wait_time)

    def _record_performance(self, result: BatchResult):
        """Record performance data for adaptive optimization"""
        performance_data = {
            'batch_size': len(result.requests),
            'processing_time': result.processing_time,
            'success_rate': result.success_rate,
            'timestamp': result.timestamp
        }
        self._performance_history.append(performance_data)

    def _update_stats(self, result: BatchResult):
        """Update statistics"""
        self.stats['total_batches'] += 1

        # Update averages
        total_requests = self.stats['total_requests']
        total_batches = self.stats['total_batches']

        batch_size = len(result.requests)
        self.stats['average_batch_size'] = (
            (self.stats['average_batch_size'] * (total_batches - 1) + batch_size) / total_batches
        )

        self.stats['average_processing_time'] = (
            (self.stats['average_processing_time'] * (total_batches - 1) + result.processing_time) / total_batches
        )

        # Update success rate
        total_successes = result.success_count
        total_operations = result.success_count + result.error_count
        if total_operations > 0:
            batch_success_rate = total_successes / total_operations
            self.stats['success_rate'] = (
                (self.stats['success_rate'] * (total_batches - 1) + batch_success_rate) / total_batches
            )

        # Update queue sizes
        for operation, queue in self._queues.items():
            self.stats['queue_sizes'][operation] = len(queue)

    async def _cleanup_processor(self):
        """Background task to clean up expired requests"""
        while self._running:
            try:
                await asyncio.sleep(30)  # Run every 30 seconds

                expired_count = 0
                for operation, queue in self._queues.items():
                    # Remove expired requests from front of queue
                    while queue and queue[0].is_expired():
                        expired_request = queue.popleft()
                        self._pending_requests.pop(expired_request.id, None)
                        expired_count += 1

                if expired_count > 0:
                    logger.info(f"Cleaned up {expired_count} expired requests")

            except Exception as e:
                logger.error(f"Error in cleanup processor: {e}")

    def __len__(self) -> int:
        """Get total number of pending requests"""
        return len(self._pending_requests)