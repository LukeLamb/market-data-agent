"""
Tests for Parallel Processor
Tests parallel processing pipeline, worker management, and load balancing
"""

import asyncio
import pytest
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import threading
import queue

from src.bulk_loading.parallel_processor import (
    ParallelProcessor,
    ProcessingConfig,
    ProcessingMode,
    ProcessingTask,
    ProcessingResult,
    TaskQueue,
    Worker,
    ThreadWorker,
    LoadBalancer,
    WorkerMetrics,
    WorkerStatus
)


class TestProcessingConfig:
    """Test processing configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = ProcessingConfig()

        assert config.mode == ProcessingMode.HYBRID
        assert config.max_retries == 3
        assert config.load_balancing is True
        assert config.dynamic_scaling is True

    def test_custom_config(self):
        """Test custom configuration"""
        config = ProcessingConfig(
            max_threads=8,
            max_processes=4,
            mode=ProcessingMode.MULTI_THREADED,
            batch_size=500
        )

        assert config.max_threads == 8
        assert config.max_processes == 4
        assert config.mode == ProcessingMode.MULTI_THREADED
        assert config.batch_size == 500


class TestProcessingTask:
    """Test processing task functionality"""

    def test_task_creation(self):
        """Test task initialization"""
        task = ProcessingTask(
            task_id="task_1",
            data={"test": "data"},
            processor_func="test_processor",
            priority=5
        )

        assert task.task_id == "task_1"
        assert task.data == {"test": "data"}
        assert task.processor_func == "test_processor"
        assert task.priority == 5
        assert task.retries == 0

    def test_task_priority_comparison(self):
        """Test task priority comparison for sorting"""
        task1 = ProcessingTask("task_1", {}, "func", priority=1)
        task2 = ProcessingTask("task_2", {}, "func", priority=5)
        task3 = ProcessingTask("task_3", {}, "func", priority=3)

        tasks = [task1, task2, task3]
        sorted_tasks = sorted(tasks)

        # Higher priority should come first
        assert sorted_tasks[0].priority == 5
        assert sorted_tasks[1].priority == 3
        assert sorted_tasks[2].priority == 1


class TestProcessingResult:
    """Test processing result functionality"""

    def test_result_creation(self):
        """Test result initialization"""
        result = ProcessingResult(
            task_id="task_1",
            success=True,
            result={"processed": True},
            processing_time=0.5,
            worker_id="worker_1"
        )

        assert result.task_id == "task_1"
        assert result.success is True
        assert result.result == {"processed": True}
        assert result.processing_time == 0.5
        assert result.worker_id == "worker_1"

    def test_error_result(self):
        """Test error result creation"""
        result = ProcessingResult(
            task_id="task_1",
            success=False,
            error="Processing failed"
        )

        assert result.success is False
        assert result.error == "Processing failed"
        assert result.result is None


class TestWorkerMetrics:
    """Test worker metrics functionality"""

    def test_metrics_initialization(self):
        """Test metrics initialization"""
        metrics = WorkerMetrics("worker_1", "thread")

        assert metrics.worker_id == "worker_1"
        assert metrics.worker_type == "thread"
        assert metrics.status == WorkerStatus.IDLE
        assert metrics.tasks_processed == 0
        assert metrics.success_rate == 100.0

    def test_metrics_calculations(self):
        """Test metrics calculated properties"""
        metrics = WorkerMetrics("worker_1", "thread")
        metrics.tasks_processed = 100
        metrics.tasks_failed = 10
        metrics.total_processing_time = 500.0

        assert metrics.success_rate == 90.9  # 100/(100+10) * 100
        assert metrics.avg_processing_time == 5.0  # 500/100


class TestTaskQueue:
    """Test task queue functionality"""

    def test_queue_initialization(self):
        """Test queue initialization"""
        task_queue = TaskQueue(max_size=100)

        assert task_queue.size() == 0
        assert task_queue.empty()

    def test_queue_put_get(self):
        """Test putting and getting tasks"""
        task_queue = TaskQueue(max_size=10)

        task1 = ProcessingTask("task_1", {}, "func", priority=1)
        task2 = ProcessingTask("task_2", {}, "func", priority=5)

        # Put tasks
        assert task_queue.put(task1)
        assert task_queue.put(task2)
        assert task_queue.size() == 2

        # Get tasks (should come out in priority order)
        retrieved_task1 = task_queue.get(timeout=0.1)
        assert retrieved_task1.priority == 5  # Higher priority first

        retrieved_task2 = task_queue.get(timeout=0.1)
        assert retrieved_task2.priority == 1

        assert task_queue.empty()

    def test_queue_full_behavior(self):
        """Test queue behavior when full"""
        task_queue = TaskQueue(max_size=2)

        task1 = ProcessingTask("task_1", {}, "func")
        task2 = ProcessingTask("task_2", {}, "func")
        task3 = ProcessingTask("task_3", {}, "func")

        # Fill queue
        assert task_queue.put(task1)
        assert task_queue.put(task2)

        # Should reject additional tasks
        assert not task_queue.put(task3, timeout=0.1)

    def test_queue_timeout(self):
        """Test queue timeout behavior"""
        task_queue = TaskQueue()

        # Get from empty queue should timeout
        start_time = time.time()
        result = task_queue.get(timeout=0.1)
        end_time = time.time()

        assert result is None
        assert (end_time - start_time) >= 0.1

    def test_queue_shutdown(self):
        """Test queue shutdown"""
        task_queue = TaskQueue()

        task = ProcessingTask("task_1", {}, "func")
        task_queue.put(task)

        task_queue.shutdown()

        # Should not accept new tasks after shutdown
        new_task = ProcessingTask("task_2", {}, "func")
        assert not task_queue.put(new_task)


class TestWorker:
    """Test worker functionality"""

    def test_worker_initialization(self):
        """Test worker initialization"""
        task_queue = TaskQueue()
        result_queue = queue.Queue()
        processors = {"test_func": lambda x: x}

        worker = Worker("worker_1", "thread", task_queue, result_queue, processors)

        assert worker.worker_id == "worker_1"
        assert worker.worker_type == "thread"
        assert worker.metrics.worker_id == "worker_1"

    def test_worker_task_processing(self):
        """Test worker task processing"""
        task_queue = TaskQueue()
        result_queue = queue.Queue()

        def test_processor(data, **kwargs):
            return {"processed": True, "data": data}

        processors = {"test_func": test_processor}

        worker = Worker("worker_1", "thread", task_queue, result_queue, processors)

        # Create and process task
        task = ProcessingTask("task_1", {"input": "test"}, "test_func")
        result = worker._process_task(task)

        assert result.success
        assert result.result["processed"] is True
        assert result.result["data"]["input"] == "test"

    def test_worker_error_handling(self):
        """Test worker error handling"""
        task_queue = TaskQueue()
        result_queue = queue.Queue()

        def failing_processor(data, **kwargs):
            raise ValueError("Processing failed")

        processors = {"failing_func": failing_processor}

        worker = Worker("worker_1", "thread", task_queue, result_queue, processors)

        # Create task that will fail
        task = ProcessingTask("task_1", {}, "failing_func")
        result = worker._process_task(task)

        assert not result.success
        assert "Processing failed" in result.error

    def test_worker_unknown_processor(self):
        """Test worker with unknown processor function"""
        task_queue = TaskQueue()
        result_queue = queue.Queue()
        processors = {}

        worker = Worker("worker_1", "thread", task_queue, result_queue, processors)

        # Create task with unknown processor
        task = ProcessingTask("task_1", {}, "unknown_func")
        result = worker._process_task(task)

        assert not result.success
        assert "Unknown processor function" in result.error


class TestThreadWorker:
    """Test thread worker functionality"""

    def test_thread_worker_creation(self):
        """Test thread worker creation"""
        task_queue = TaskQueue()
        result_queue = queue.Queue()
        processors = {"test_func": lambda x: x}

        worker = ThreadWorker("thread_1", task_queue, result_queue, processors)

        assert worker.worker_id == "thread_1"
        assert worker.worker_type == "thread"
        assert hasattr(worker, 'thread')

    def test_thread_worker_lifecycle(self):
        """Test thread worker start and shutdown"""
        task_queue = TaskQueue()
        result_queue = queue.Queue()
        processors = {"test_func": lambda x: {"result": x}}

        worker = ThreadWorker("thread_1", task_queue, result_queue, processors)

        # Start worker
        worker.start()
        assert worker.thread.is_alive()

        # Add a task
        task = ProcessingTask("task_1", "test_data", "test_func")
        task_queue.put(task)

        # Wait a bit for processing
        time.sleep(0.1)

        # Shutdown worker
        worker.shutdown()
        worker.join(timeout=1.0)

        assert not worker.thread.is_alive()
        assert worker.metrics.status == WorkerStatus.SHUTDOWN


class TestLoadBalancer:
    """Test load balancer functionality"""

    def test_load_balancer_creation(self):
        """Test load balancer creation"""
        task_queue = TaskQueue()
        result_queue = queue.Queue()
        processors = {}

        workers = [
            ThreadWorker(f"worker_{i}", task_queue, result_queue, processors)
            for i in range(3)
        ]

        load_balancer = LoadBalancer(workers)

        assert len(load_balancer.workers) == 3

    def test_round_robin_selection(self):
        """Test round-robin worker selection"""
        task_queue = TaskQueue()
        result_queue = queue.Queue()
        processors = {}

        workers = [
            ThreadWorker(f"worker_{i}", task_queue, result_queue, processors)
            for i in range(3)
        ]

        load_balancer = LoadBalancer(workers)

        # Test round-robin selection
        selected_workers = []
        for _ in range(6):
            worker = load_balancer.get_best_worker()
            selected_workers.append(worker.worker_id)

        # Should cycle through workers
        expected = ["worker_0", "worker_1", "worker_2", "worker_0", "worker_1", "worker_2"]
        assert selected_workers == expected

    def test_worker_metrics_collection(self):
        """Test worker metrics collection"""
        task_queue = TaskQueue()
        result_queue = queue.Queue()
        processors = {}

        workers = [
            ThreadWorker(f"worker_{i}", task_queue, result_queue, processors)
            for i in range(2)
        ]

        load_balancer = LoadBalancer(workers)

        metrics = load_balancer.get_worker_metrics()

        assert len(metrics) == 2
        assert all(isinstance(m, WorkerMetrics) for m in metrics)
        assert metrics[0].worker_id == "worker_0"
        assert metrics[1].worker_id == "worker_1"


class TestParallelProcessor:
    """Test parallel processor functionality"""

    @pytest.fixture
    def processor_config(self):
        """Test processor configuration"""
        return ProcessingConfig(
            max_threads=2,
            max_processes=1,
            mode=ProcessingMode.MULTI_THREADED,
            queue_max_size=100
        )

    @pytest.fixture
    def parallel_processor(self, processor_config):
        """Create parallel processor instance"""
        return ParallelProcessor(processor_config)

    def test_processor_initialization(self, parallel_processor):
        """Test processor initialization"""
        assert not parallel_processor.is_running
        assert len(parallel_processor.processors) == 0
        assert len(parallel_processor.workers) == 0

    def test_processor_registration(self, parallel_processor):
        """Test processor function registration"""
        def test_processor(data, **kwargs):
            return {"processed": True}

        parallel_processor.register_processor("test_func", test_processor)

        assert "test_func" in parallel_processor.processors
        assert parallel_processor.processors["test_func"] == test_processor

    @pytest.mark.asyncio
    async def test_processor_start_stop(self, parallel_processor):
        """Test processor start and stop"""
        def test_processor(data, **kwargs):
            return data

        parallel_processor.register_processor("test_func", test_processor)

        # Start processor
        await parallel_processor.start()
        assert parallel_processor.is_running
        assert len(parallel_processor.workers) > 0

        # Stop processor
        await parallel_processor.shutdown()
        assert not parallel_processor.is_running

    @pytest.mark.asyncio
    async def test_task_submission(self, parallel_processor):
        """Test task submission"""
        def test_processor(data, **kwargs):
            return {"result": data, "processed": True}

        parallel_processor.register_processor("test_func", test_processor)

        await parallel_processor.start()

        # Submit task
        success = await parallel_processor.submit_task(
            task_id="task_1",
            data="test_data",
            processor_func="test_func",
            priority=1
        )

        assert success

        await parallel_processor.shutdown()

    @pytest.mark.asyncio
    async def test_batch_submission(self, parallel_processor):
        """Test batch task submission"""
        def test_processor(data, **kwargs):
            return {"processed": True}

        parallel_processor.register_processor("test_func", test_processor)

        await parallel_processor.start()

        # Submit batch
        tasks = [{"id": i, "data": f"item_{i}"} for i in range(10)]
        submitted_count = await parallel_processor.submit_batch(
            tasks=tasks,
            processor_func="test_func"
        )

        assert submitted_count == 10

        await parallel_processor.shutdown()

    @pytest.mark.asyncio
    async def test_result_handling(self, parallel_processor):
        """Test result handling"""
        results = []

        def result_handler(result):
            results.append(result)

        def test_processor(data, **kwargs):
            return {"processed": data}

        parallel_processor.register_processor("test_func", test_processor)
        parallel_processor.add_result_handler(result_handler)

        await parallel_processor.start()

        # Submit task
        await parallel_processor.submit_task(
            task_id="task_1",
            data="test_data",
            processor_func="test_func"
        )

        # Wait for processing
        await asyncio.sleep(0.2)

        await parallel_processor.shutdown()

        # Check results
        assert len(results) > 0
        assert results[0].task_id == "task_1"
        assert results[0].success

    @pytest.mark.asyncio
    async def test_processor_statistics(self, parallel_processor):
        """Test processor statistics collection"""
        def test_processor(data, **kwargs):
            return data

        parallel_processor.register_processor("test_func", test_processor)

        await parallel_processor.start()

        # Submit some tasks
        for i in range(5):
            await parallel_processor.submit_task(
                task_id=f"task_{i}",
                data=f"data_{i}",
                processor_func="test_func"
            )

        # Wait for processing
        await asyncio.sleep(0.1)

        stats = parallel_processor.get_processing_stats()

        assert "total_workers" in stats
        assert "total_tasks_processed" in stats
        assert "overall_success_rate" in stats

        await parallel_processor.shutdown()

    @pytest.mark.asyncio
    async def test_worker_mode_configuration(self):
        """Test different worker mode configurations"""
        # Test single-threaded mode
        config = ProcessingConfig(mode=ProcessingMode.SINGLE_THREADED)
        processor = ParallelProcessor(config)

        await processor.start()
        assert len(processor.workers) == 1
        assert processor.workers[0].worker_type == "thread"
        await processor.shutdown()

        # Test multi-threaded mode
        config = ProcessingConfig(
            mode=ProcessingMode.MULTI_THREADED,
            max_threads=3
        )
        processor = ParallelProcessor(config)

        await processor.start()
        assert len(processor.workers) == 3
        assert all(w.worker_type == "thread" for w in processor.workers)
        await processor.shutdown()

    @pytest.mark.asyncio
    async def test_error_handling_in_processing(self, parallel_processor):
        """Test error handling during processing"""
        results = []

        def result_handler(result):
            results.append(result)

        def failing_processor(data, **kwargs):
            if data == "fail":
                raise ValueError("Intentional failure")
            return {"processed": data}

        parallel_processor.register_processor("failing_func", failing_processor)
        parallel_processor.add_result_handler(result_handler)

        await parallel_processor.start()

        # Submit successful and failing tasks
        await parallel_processor.submit_task("task_1", "success", "failing_func")
        await parallel_processor.submit_task("task_2", "fail", "failing_func")

        # Wait for processing
        await asyncio.sleep(0.2)

        await parallel_processor.shutdown()

        # Check results
        assert len(results) == 2

        success_result = next(r for r in results if r.task_id == "task_1")
        assert success_result.success

        failure_result = next(r for r in results if r.task_id == "task_2")
        assert not failure_result.success
        assert "Intentional failure" in failure_result.error

    @pytest.mark.asyncio
    async def test_context_manager(self, processor_config):
        """Test processor as context manager"""
        def test_processor(data, **kwargs):
            return data

        async with ParallelProcessor(processor_config) as processor:
            processor.register_processor("test_func", test_processor)
            assert processor.is_running

        # Should be stopped after context exit
        assert not processor.is_running

    @pytest.mark.asyncio
    async def test_completion_waiting(self, parallel_processor):
        """Test waiting for task completion"""
        def slow_processor(data, **kwargs):
            time.sleep(0.1)  # Simulate processing time
            return data

        parallel_processor.register_processor("slow_func", slow_processor)

        await parallel_processor.start()

        # Submit tasks
        for i in range(5):
            await parallel_processor.submit_task(
                f"task_{i}", f"data_{i}", "slow_func"
            )

        # Wait for completion
        completed = await parallel_processor.wait_for_completion(timeout=2.0)
        assert completed

        await parallel_processor.shutdown()


class TestParallelProcessorIntegration:
    """Integration tests for parallel processor"""

    @pytest.mark.integration
    async def test_high_throughput_processing(self):
        """Test high throughput processing performance"""
        # This test would measure actual throughput performance
        pytest.skip("Integration test requires performance measurement setup")

    @pytest.mark.integration
    async def test_memory_usage_under_load(self):
        """Test memory usage under heavy load"""
        # This test would monitor memory usage during processing
        pytest.skip("Integration test requires memory monitoring setup")

    @pytest.mark.integration
    async def test_processor_resilience(self):
        """Test processor resilience to failures"""
        # This test would simulate various failure scenarios
        pytest.skip("Integration test requires failure simulation setup")