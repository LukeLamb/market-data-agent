"""Request Scheduler Implementation

Intelligent request scheduling with priority queuing, batching, and optimization
for cost-effective API usage across multiple data sources.
"""

import asyncio
import heapq
import time
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    """Request priority levels"""
    CRITICAL = 1    # Real-time trading, alerts
    HIGH = 2        # User-requested data
    NORMAL = 3      # Background updates, caching
    LOW = 4         # Bulk operations, historical data
    BATCH = 5       # Batchable requests (lowest priority)


@dataclass
class ScheduledRequest:
    """Represents a scheduled API request"""
    priority: RequestPriority
    source_name: str
    endpoint: str
    params: Dict[str, Any]
    callback: Callable
    created_at: datetime = field(default_factory=datetime.now)
    timeout: Optional[float] = None
    max_retries: int = 3
    retry_count: int = 0
    cost_weight: float = 1.0  # Relative cost of this request
    batchable: bool = False   # Whether this request can be batched with others
    batch_key: Optional[str] = None  # Key for batching similar requests

    def __post_init__(self):
        """Ensure proper initialization"""
        if self.batch_key is None and self.batchable:
            # Auto-generate batch key from endpoint and common params
            batch_params = {k: v for k, v in self.params.items()
                          if k in ['symbol', 'interval', 'market']}
            self.batch_key = f"{self.source_name}:{self.endpoint}:{sorted(batch_params.items())}"

    def __lt__(self, other):
        """Priority queue comparison (lower priority value = higher priority)"""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        # Secondary sort by creation time (FIFO within same priority)
        return self.created_at < other.created_at

    def is_expired(self) -> bool:
        """Check if request has expired"""
        if self.timeout is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.timeout

    def should_retry(self) -> bool:
        """Check if request should be retried"""
        return self.retry_count < self.max_retries

    def increment_retry(self) -> None:
        """Increment retry count"""
        self.retry_count += 1


class RequestScheduler:
    """Intelligent request scheduler with priority queuing and batching

    Features:
    - Priority-based scheduling with multiple queues
    - Request batching for similar operations
    - Cost-aware scheduling to optimize API usage
    - Adaptive scheduling based on source performance
    - Request deduplication and caching awareness
    """

    def __init__(self, max_concurrent: int = 10, batch_window: float = 1.0):
        self.max_concurrent = max_concurrent
        self.batch_window = batch_window  # Time window for batching requests

        # Priority queues for different priority levels
        self.queues: Dict[RequestPriority, List[ScheduledRequest]] = {
            priority: [] for priority in RequestPriority
        }

        # Active requests tracking
        self.active_requests: Dict[str, ScheduledRequest] = {}
        self.completed_requests: List[ScheduledRequest] = []

        # Batching support
        self.batch_groups: Dict[str, List[ScheduledRequest]] = {}
        self.last_batch_check = time.time()

        # Statistics
        self.total_scheduled = 0
        self.total_completed = 0
        self.total_failed = 0
        self.total_batched = 0

        # Control flags
        self.running = False
        self.paused = False

        # Asyncio primitives
        self._scheduler_task: Optional[asyncio.Task] = None
        self._request_semaphore = asyncio.Semaphore(max_concurrent)

    async def start(self) -> None:
        """Start the request scheduler"""
        if self.running:
            return

        self.running = True
        self.paused = False
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info(f"Request scheduler started with max_concurrent={self.max_concurrent}")

    async def stop(self) -> None:
        """Stop the request scheduler"""
        if not self.running:
            return

        self.running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("Request scheduler stopped")

    def pause(self) -> None:
        """Pause request processing"""
        self.paused = True
        logger.info("Request scheduler paused")

    def resume(self) -> None:
        """Resume request processing"""
        self.paused = False
        logger.info("Request scheduler resumed")

    async def schedule_request(self, request: ScheduledRequest) -> str:
        """Schedule a request for execution

        Args:
            request: The request to schedule

        Returns:
            Request ID for tracking
        """
        request_id = f"{request.source_name}_{request.endpoint}_{id(request)}"

        # Add to appropriate priority queue
        heapq.heappush(self.queues[request.priority], request)
        self.total_scheduled += 1

        # Handle batching
        if request.batchable and request.batch_key:
            if request.batch_key not in self.batch_groups:
                self.batch_groups[request.batch_key] = []
            self.batch_groups[request.batch_key].append(request)

        logger.debug(f"Scheduled request {request_id} with priority {request.priority.name}")
        return request_id

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop"""
        while self.running:
            try:
                if not self.paused:
                    await self._process_requests()
                    await self._process_batches()
                    await self._cleanup_expired()

                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting

            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(1.0)  # Longer delay on error

    async def _process_requests(self) -> None:
        """Process requests from priority queues"""
        # Try to acquire semaphore before processing
        if self._request_semaphore.locked():
            return

        # Process requests in priority order
        for priority in RequestPriority:
            queue = self.queues[priority]

            while queue and len(self.active_requests) < self.max_concurrent:
                request = heapq.heappop(queue)

                # Skip expired requests
                if request.is_expired():
                    logger.warning(f"Skipping expired request: {request.endpoint}")
                    continue

                # Skip if already processing (deduplication)
                request_key = f"{request.source_name}_{request.endpoint}_{hash(frozenset(request.params.items()))}"
                if request_key in self.active_requests:
                    logger.debug(f"Skipping duplicate request: {request.endpoint}")
                    continue

                # Execute request
                await self._execute_request(request, request_key)

    async def _execute_request(self, request: ScheduledRequest, request_key: str) -> None:
        """Execute a single request"""
        async with self._request_semaphore:
            self.active_requests[request_key] = request

            try:
                # Execute the request callback
                start_time = time.time()
                result = await request.callback(request.source_name, request.endpoint, request.params)
                execution_time = time.time() - start_time

                # Mark as completed
                self.completed_requests.append(request)
                self.total_completed += 1

                logger.debug(f"Completed request {request.endpoint} in {execution_time:.3f}s")

            except Exception as e:
                logger.error(f"Request failed: {request.endpoint} - {e}")

                # Handle retry
                if request.should_retry():
                    request.increment_retry()
                    # Re-queue with lower priority
                    if request.priority.value < RequestPriority.LOW.value:
                        request.priority = RequestPriority(request.priority.value + 1)
                    heapq.heappush(self.queues[request.priority], request)
                    logger.info(f"Retrying request {request.endpoint} (attempt {request.retry_count})")
                else:
                    self.total_failed += 1
                    logger.error(f"Request {request.endpoint} failed after {request.max_retries} attempts")

            finally:
                # Remove from active requests
                if request_key in self.active_requests:
                    del self.active_requests[request_key]

    async def _process_batches(self) -> None:
        """Process batchable requests"""
        current_time = time.time()

        # Only check batches periodically
        if current_time - self.last_batch_check < self.batch_window:
            return

        self.last_batch_check = current_time

        # Process ready batches
        for batch_key, requests in list(self.batch_groups.items()):
            if len(requests) >= 2:  # Minimum batch size
                # Remove requests from individual queues
                for request in requests:
                    try:
                        self.queues[request.priority].remove(request)
                        heapq.heapify(self.queues[request.priority])
                    except ValueError:
                        pass  # Request may have already been processed

                # Create batch request
                await self._execute_batch(batch_key, requests)

                # Clean up batch group
                del self.batch_groups[batch_key]

    async def _execute_batch(self, batch_key: str, requests: List[ScheduledRequest]) -> None:
        """Execute a batch of similar requests"""
        if not requests:
            return

        # Combine parameters from all requests
        combined_params = {}
        symbols = []

        for request in requests:
            if 'symbol' in request.params:
                symbols.append(request.params['symbol'])
            else:
                combined_params.update(request.params)

        if symbols:
            combined_params['symbols'] = ','.join(symbols)

        # Use the highest priority request as the template
        template_request = min(requests, key=lambda r: r.priority.value)

        try:
            # Execute batch request
            start_time = time.time()
            result = await template_request.callback(
                template_request.source_name,
                template_request.endpoint,
                combined_params
            )
            execution_time = time.time() - start_time

            # Mark all requests as completed
            for request in requests:
                self.completed_requests.append(request)

            self.total_completed += len(requests)
            self.total_batched += len(requests)

            logger.info(f"Completed batch of {len(requests)} requests in {execution_time:.3f}s")

        except Exception as e:
            logger.error(f"Batch request failed: {batch_key} - {e}")

            # Re-queue individual requests for retry
            for request in requests:
                if request.should_retry():
                    request.increment_retry()
                    heapq.heappush(self.queues[request.priority], request)
                else:
                    self.total_failed += 1

    async def _cleanup_expired(self) -> None:
        """Clean up expired requests and old completed requests"""
        # Clean expired requests from queues
        for priority, queue in self.queues.items():
            expired = [req for req in queue if req.is_expired()]
            for req in expired:
                try:
                    queue.remove(req)
                    heapq.heapify(queue)
                    logger.warning(f"Removed expired request: {req.endpoint}")
                except ValueError:
                    pass

        # Clean old completed requests (keep last 1000)
        if len(self.completed_requests) > 1000:
            self.completed_requests = self.completed_requests[-1000:]

        # Clean old batch groups
        current_time = time.time()
        for batch_key, requests in list(self.batch_groups.items()):
            if requests and (current_time - requests[0].created_at.timestamp()) > self.batch_window * 2:
                # Re-queue requests individually
                for request in requests:
                    if not request.is_expired():
                        heapq.heappush(self.queues[request.priority], request)
                del self.batch_groups[batch_key]

    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        queue_sizes = {priority.name: len(queue) for priority, queue in self.queues.items()}
        batch_groups_count = len(self.batch_groups)
        total_queued = sum(queue_sizes.values())

        return {
            "running": self.running,
            "paused": self.paused,
            "total_scheduled": self.total_scheduled,
            "total_completed": self.total_completed,
            "total_failed": self.total_failed,
            "total_batched": self.total_batched,
            "active_requests": len(self.active_requests),
            "total_queued": total_queued,
            "queue_sizes": queue_sizes,
            "batch_groups": batch_groups_count,
            "success_rate": self.total_completed / max(1, self.total_scheduled),
            "batch_efficiency": self.total_batched / max(1, self.total_completed)
        }

    def get_queue_status(self) -> Dict[str, List[Dict]]:
        """Get detailed queue status"""
        status = {}

        for priority, queue in self.queues.items():
            status[priority.name] = [
                {
                    "source": req.source_name,
                    "endpoint": req.endpoint,
                    "created_at": req.created_at.isoformat(),
                    "retry_count": req.retry_count,
                    "cost_weight": req.cost_weight,
                    "batchable": req.batchable
                }
                for req in queue[:10]  # Limit to first 10 for readability
            ]

        return status