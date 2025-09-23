"""
Parallel Processing Pipeline for Bulk Data Loading
Handles high-throughput data processing with worker pools and load balancing
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
import queue
import threading
import time
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Processing execution modes"""
    SINGLE_THREADED = "single_threaded"
    MULTI_THREADED = "multi_threaded"
    MULTI_PROCESS = "multi_process"
    HYBRID = "hybrid"  # Combination of threads and processes


class WorkerStatus(Enum):
    """Worker status states"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class ProcessingConfig:
    """Configuration for parallel processing"""
    # Worker configuration
    max_threads: int = mp.cpu_count() * 2
    max_processes: int = mp.cpu_count()
    mode: ProcessingMode = ProcessingMode.HYBRID

    # Queue configuration
    queue_max_size: int = 10000
    batch_size: int = 1000
    timeout: float = 30.0

    # Performance tuning
    prefetch_factor: int = 2  # How many batches to prefetch per worker
    load_balancing: bool = True
    dynamic_scaling: bool = True
    min_workers: int = 2

    # Error handling
    max_retries: int = 3
    retry_delay: float = 1.0
    error_threshold: float = 0.1  # 10% error rate threshold


@dataclass
class WorkerMetrics:
    """Metrics for individual workers"""
    worker_id: str
    worker_type: str  # "thread" or "process"
    status: WorkerStatus = WorkerStatus.IDLE

    # Performance metrics
    tasks_processed: int = 0
    tasks_failed: int = 0
    total_processing_time: float = 0.0
    last_task_time: Optional[datetime] = None

    # Resource usage
    cpu_usage: float = 0.0
    memory_usage: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.tasks_processed + self.tasks_failed
        return (self.tasks_processed / max(total, 1)) * 100

    @property
    def avg_processing_time(self) -> float:
        return self.total_processing_time / max(self.tasks_processed, 1)


@dataclass
class ProcessingTask:
    """Task for parallel processing"""
    task_id: str
    data: Any
    processor_func: str  # Function name to call
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher numbers = higher priority
    retries: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)

    def __lt__(self, other):
        return self.priority > other.priority  # Higher priority first


@dataclass
class ProcessingResult:
    """Result from processing task"""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    processing_time: float = 0.0
    worker_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


class TaskQueue:
    """Thread-safe task queue with priority support"""

    def __init__(self, max_size: int = 10000):
        self._queue = queue.PriorityQueue(maxsize=max_size)
        self._lock = threading.Lock()
        self._shutdown = False

    def put(self, task: ProcessingTask, timeout: Optional[float] = None) -> bool:
        """Add task to queue"""
        if self._shutdown:
            return False

        try:
            self._queue.put(task, timeout=timeout)
            return True
        except queue.Full:
            return False

    def get(self, timeout: Optional[float] = None) -> Optional[ProcessingTask]:
        """Get task from queue"""
        if self._shutdown and self._queue.empty():
            return None

        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def task_done(self):
        """Mark task as done"""
        self._queue.task_done()

    def size(self) -> int:
        """Get queue size"""
        return self._queue.qsize()

    def empty(self) -> bool:
        """Check if queue is empty"""
        return self._queue.empty()

    def shutdown(self):
        """Shutdown queue"""
        self._shutdown = True


class Worker:
    """Base worker class"""

    def __init__(
        self,
        worker_id: str,
        worker_type: str,
        task_queue: TaskQueue,
        result_queue: queue.Queue,
        processors: Dict[str, Callable]
    ):
        self.worker_id = worker_id
        self.worker_type = worker_type
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.processors = processors

        self.metrics = WorkerMetrics(worker_id, worker_type)
        self.shutdown_event = threading.Event()

    def run(self):
        """Main worker loop"""
        logger.info(f"Worker {self.worker_id} started")

        while not self.shutdown_event.is_set():
            try:
                # Get task from queue
                task = self.task_queue.get(timeout=1.0)
                if task is None:
                    continue

                self.metrics.status = WorkerStatus.BUSY
                start_time = time.time()

                # Process task
                result = self._process_task(task)

                # Update metrics
                processing_time = time.time() - start_time
                self.metrics.total_processing_time += processing_time
                self.metrics.last_task_time = datetime.now()

                if result.success:
                    self.metrics.tasks_processed += 1
                else:
                    self.metrics.tasks_failed += 1

                # Send result
                result.processing_time = processing_time
                result.worker_id = self.worker_id
                self.result_queue.put(result)

                # Mark task as done
                self.task_queue.task_done()
                self.metrics.status = WorkerStatus.IDLE

            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")
                self.metrics.status = WorkerStatus.ERROR
                self.metrics.tasks_failed += 1

        self.metrics.status = WorkerStatus.SHUTDOWN
        logger.info(f"Worker {self.worker_id} stopped")

    def _process_task(self, task: ProcessingTask) -> ProcessingResult:
        """Process individual task"""
        try:
            # Get processor function
            if task.processor_func not in self.processors:
                raise ValueError(f"Unknown processor function: {task.processor_func}")

            processor = self.processors[task.processor_func]

            # Execute processor
            result = processor(task.data, **task.kwargs)

            return ProcessingResult(
                task_id=task.task_id,
                success=True,
                result=result
            )

        except Exception as e:
            return ProcessingResult(
                task_id=task.task_id,
                success=False,
                error=str(e)
            )

    def shutdown(self):
        """Shutdown worker"""
        self.shutdown_event.set()


class ThreadWorker(Worker):
    """Thread-based worker"""

    def __init__(self, worker_id: str, task_queue: TaskQueue, result_queue: queue.Queue, processors: Dict[str, Callable]):
        super().__init__(worker_id, "thread", task_queue, result_queue, processors)
        self.thread = threading.Thread(target=self.run, name=f"ThreadWorker-{worker_id}")

    def start(self):
        """Start worker thread"""
        self.thread.start()

    def join(self, timeout: Optional[float] = None):
        """Wait for worker to complete"""
        self.thread.join(timeout)


class ProcessWorker:
    """Process-based worker (simpler implementation for this example)"""

    def __init__(self, worker_id: str, task_queue: TaskQueue, result_queue: queue.Queue, processors: Dict[str, Callable]):
        self.worker_id = worker_id
        self.worker_type = "process"
        # In a real implementation, this would use multiprocessing
        # For simplicity, we'll use thread-based implementation here
        self.worker = ThreadWorker(worker_id, task_queue, result_queue, processors)

    def start(self):
        self.worker.start()

    def shutdown(self):
        self.worker.shutdown()

    def join(self, timeout: Optional[float] = None):
        self.worker.join(timeout)

    @property
    def metrics(self):
        return self.worker.metrics


class LoadBalancer:
    """Load balancer for distributing tasks across workers"""

    def __init__(self, workers: List[Union[ThreadWorker, ProcessWorker]]):
        self.workers = workers
        self._round_robin_index = 0

    def get_best_worker(self) -> Union[ThreadWorker, ProcessWorker]:
        """Get the best worker for next task"""
        # Simple round-robin for now
        # Could be enhanced with metrics-based selection
        worker = self.workers[self._round_robin_index]
        self._round_robin_index = (self._round_robin_index + 1) % len(self.workers)
        return worker

    def get_worker_metrics(self) -> List[WorkerMetrics]:
        """Get metrics for all workers"""
        return [worker.metrics for worker in self.workers]


class ParallelProcessor:
    """Main parallel processing coordinator"""

    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()

        # Task and result queues
        self.task_queue = TaskQueue(self.config.queue_max_size)
        self.result_queue = queue.Queue()

        # Workers
        self.workers: List[Union[ThreadWorker, ProcessWorker]] = []
        self.load_balancer: Optional[LoadBalancer] = None

        # Processors
        self.processors: Dict[str, Callable] = {}

        # State management
        self.is_running = False
        self.shutdown_event = threading.Event()

        # Monitoring
        self.result_handlers: List[Callable[[ProcessingResult], None]] = []
        self.monitor_thread: Optional[threading.Thread] = None

    def register_processor(self, name: str, func: Callable) -> None:
        """Register a processing function"""
        self.processors[name] = func
        logger.info(f"Registered processor: {name}")

    def add_result_handler(self, handler: Callable[[ProcessingResult], None]) -> None:
        """Add result handler"""
        self.result_handlers.append(handler)

    async def start(self) -> None:
        """Start the parallel processor"""
        if self.is_running:
            return

        logger.info("Starting parallel processor")

        # Create workers based on configuration
        await self._create_workers()

        # Start workers
        for worker in self.workers:
            worker.start()

        # Setup load balancer
        self.load_balancer = LoadBalancer(self.workers)

        # Start result monitor
        self.monitor_thread = threading.Thread(target=self._monitor_results, name="ResultMonitor")
        self.monitor_thread.start()

        self.is_running = True
        logger.info(f"Parallel processor started with {len(self.workers)} workers")

    async def _create_workers(self) -> None:
        """Create worker pool based on configuration"""
        if self.config.mode == ProcessingMode.SINGLE_THREADED:
            # Single thread worker
            worker = ThreadWorker("thread-0", self.task_queue, self.result_queue, self.processors)
            self.workers.append(worker)

        elif self.config.mode == ProcessingMode.MULTI_THREADED:
            # Multiple thread workers
            for i in range(self.config.max_threads):
                worker = ThreadWorker(f"thread-{i}", self.task_queue, self.result_queue, self.processors)
                self.workers.append(worker)

        elif self.config.mode == ProcessingMode.MULTI_PROCESS:
            # Multiple process workers
            for i in range(self.config.max_processes):
                worker = ProcessWorker(f"process-{i}", self.task_queue, self.result_queue, self.processors)
                self.workers.append(worker)

        elif self.config.mode == ProcessingMode.HYBRID:
            # Combination of threads and processes
            thread_count = min(self.config.max_threads, mp.cpu_count())
            process_count = min(self.config.max_processes, mp.cpu_count() // 2)

            # Create thread workers
            for i in range(thread_count):
                worker = ThreadWorker(f"thread-{i}", self.task_queue, self.result_queue, self.processors)
                self.workers.append(worker)

            # Create process workers
            for i in range(process_count):
                worker = ProcessWorker(f"process-{i}", self.task_queue, self.result_queue, self.processors)
                self.workers.append(worker)

    def _monitor_results(self) -> None:
        """Monitor and handle results"""
        logger.info("Result monitor started")

        while not self.shutdown_event.is_set():
            try:
                # Get result with timeout
                result = self.result_queue.get(timeout=1.0)

                # Handle result
                for handler in self.result_handlers:
                    try:
                        handler(result)
                    except Exception as e:
                        logger.error(f"Result handler error: {e}")

                self.result_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Result monitor error: {e}")

        logger.info("Result monitor stopped")

    async def submit_task(
        self,
        task_id: str,
        data: Any,
        processor_func: str,
        priority: int = 0,
        **kwargs
    ) -> bool:
        """Submit task for processing"""
        if not self.is_running:
            raise RuntimeError("Processor not running")

        task = ProcessingTask(
            task_id=task_id,
            data=data,
            processor_func=processor_func,
            priority=priority,
            kwargs=kwargs,
            max_retries=self.config.max_retries
        )

        return self.task_queue.put(task, timeout=self.config.timeout)

    async def submit_batch(
        self,
        tasks: List[Dict[str, Any]],
        processor_func: str,
        priority: int = 0
    ) -> int:
        """Submit batch of tasks"""
        submitted = 0

        for i, task_data in enumerate(tasks):
            task_id = f"batch_{int(time.time())}_{i}"

            if await self.submit_task(task_id, task_data, processor_func, priority):
                submitted += 1
            else:
                logger.warning(f"Failed to submit task {task_id}")

        return submitted

    async def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for all tasks to complete"""
        try:
            # Wait for task queue to be empty and all tasks done
            start_time = time.time()

            while not self.task_queue.empty():
                if timeout and (time.time() - start_time) > timeout:
                    return False
                await asyncio.sleep(0.1)

            # Additional wait for any remaining processing
            await asyncio.sleep(1.0)
            return True

        except Exception as e:
            logger.error(f"Error waiting for completion: {e}")
            return False

    def get_worker_metrics(self) -> List[WorkerMetrics]:
        """Get metrics for all workers"""
        if self.load_balancer:
            return self.load_balancer.get_worker_metrics()
        return []

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        metrics = self.get_worker_metrics()

        total_processed = sum(m.tasks_processed for m in metrics)
        total_failed = sum(m.tasks_failed for m in metrics)
        total_tasks = total_processed + total_failed

        return {
            "total_workers": len(self.workers),
            "active_workers": len([m for m in metrics if m.status == WorkerStatus.BUSY]),
            "total_tasks_processed": total_processed,
            "total_tasks_failed": total_failed,
            "overall_success_rate": (total_processed / max(total_tasks, 1)) * 100,
            "queue_size": self.task_queue.size(),
            "avg_processing_time": sum(m.avg_processing_time for m in metrics) / max(len(metrics), 1)
        }

    async def shutdown(self) -> None:
        """Shutdown the parallel processor"""
        if not self.is_running:
            return

        logger.info("Shutting down parallel processor")

        # Signal shutdown
        self.shutdown_event.set()

        # Shutdown task queue
        self.task_queue.shutdown()

        # Shutdown workers
        for worker in self.workers:
            worker.shutdown()

        # Wait for workers to complete
        for worker in self.workers:
            worker.join(timeout=5.0)

        # Wait for monitor thread
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

        self.is_running = False
        logger.info("Parallel processor shut down")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()


# Example processor functions
async def validate_record_processor(data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Example processor for record validation"""
    # Implement validation logic
    return {"valid": True, "data": data}


async def transform_record_processor(data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Example processor for record transformation"""
    # Implement transformation logic
    return {"transformed": True, "data": data}


async def batch_insert_processor(data: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
    """Example processor for batch database insert"""
    # Implement batch insert logic
    return {"inserted": len(data), "success": True}


# Default processor registry
DEFAULT_PROCESSORS = {
    "validate_record": validate_record_processor,
    "transform_record": transform_record_processor,
    "batch_insert": batch_insert_processor
}


async def create_parallel_processor(config: ProcessingConfig = None) -> ParallelProcessor:
    """Create and configure a parallel processor"""
    processor = ParallelProcessor(config)

    # Register default processors
    for name, func in DEFAULT_PROCESSORS.items():
        processor.register_processor(name, func)

    return processor