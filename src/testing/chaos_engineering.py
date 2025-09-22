"""
Chaos Engineering Framework

Advanced chaos engineering tools for testing system resilience,
fault tolerance, and recovery capabilities of the Market Data Agent.
"""

import asyncio
import random
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)


class ChaosEventType(Enum):
    """Types of chaos events"""
    NETWORK_PARTITION = "network_partition"
    NETWORK_DELAY = "network_delay"
    NETWORK_PACKET_LOSS = "network_packet_loss"
    CPU_STRESS = "cpu_stress"
    MEMORY_STRESS = "memory_stress"
    DISK_STRESS = "disk_stress"
    SERVICE_KILL = "service_kill"
    DEPENDENCY_FAILURE = "dependency_failure"
    TIME_SKEW = "time_skew"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


class ChaosEventSeverity(Enum):
    """Severity levels for chaos events"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ChaosEvent:
    """Chaos engineering event configuration"""
    event_type: ChaosEventType
    severity: ChaosEventSeverity
    duration_seconds: float
    target_component: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    probability: float = 1.0  # Probability of event occurring (0.0 to 1.0)
    recovery_function: Optional[Callable] = None


@dataclass
class ChaosExecution:
    """Chaos event execution record"""
    event: ChaosEvent
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    impact_metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    recovery_time: Optional[float] = None


class NetworkChaos:
    """Network-related chaos engineering utilities"""

    def __init__(self):
        self._active_delays: Set[str] = set()
        self._active_failures: Set[str] = set()

    async def inject_latency(self, min_delay: float = 0.1, max_delay: float = 2.0, target: str = "all"):
        """Inject network latency"""
        delay_key = f"latency_{target}"
        self._active_delays.add(delay_key)

        try:
            while delay_key in self._active_delays:
                delay = random.uniform(min_delay, max_delay)
                await asyncio.sleep(delay)
                logger.debug(f"Injected {delay:.2f}s network delay for {target}")
        finally:
            self._active_delays.discard(delay_key)

    async def inject_packet_loss(self, loss_rate: float = 0.1, target: str = "all"):
        """Inject packet loss"""
        failure_key = f"packet_loss_{target}"
        self._active_failures.add(failure_key)

        try:
            while failure_key in self._active_failures:
                if random.random() < loss_rate:
                    logger.debug(f"Simulating packet loss for {target}")
                    raise ConnectionError(f"Simulated packet loss for {target}")
                await asyncio.sleep(0.1)
        finally:
            self._active_failures.discard(failure_key)

    async def inject_network_partition(self, targets: List[str], duration: float):
        """Inject network partition between targets"""
        logger.info(f"Starting network partition between {targets} for {duration}s")

        partition_keys = [f"partition_{target}" for target in targets]
        for key in partition_keys:
            self._active_failures.add(key)

        try:
            await asyncio.sleep(duration)
        finally:
            for key in partition_keys:
                self._active_failures.discard(key)

        logger.info(f"Network partition ended for {targets}")

    def is_target_partitioned(self, target: str) -> bool:
        """Check if target is currently partitioned"""
        return f"partition_{target}" in self._active_failures

    def stop_all_network_chaos(self):
        """Stop all active network chaos"""
        self._active_delays.clear()
        self._active_failures.clear()


class ResourceChaos:
    """Resource-related chaos engineering utilities"""

    def __init__(self):
        self._stress_tasks: List[asyncio.Task] = []
        self._memory_hogs: List[bytearray] = []

    async def inject_cpu_stress(self, cpu_percent: float = 80.0, duration: float = 30.0):
        """Inject CPU stress"""
        logger.info(f"Starting CPU stress: {cpu_percent}% for {duration}s")

        def cpu_burn():
            """CPU burning function"""
            start_time = time.time()
            while time.time() - start_time < duration:
                # Busy loop to consume CPU
                for _ in range(10000):
                    pass
                # Brief sleep to control CPU usage
                time.sleep(0.001 * (100 - cpu_percent) / 100)

        # Start multiple threads to stress CPU
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=psutil.cpu_count()) as executor:
            futures = [executor.submit(cpu_burn) for _ in range(psutil.cpu_count())]
            await asyncio.gather(*[asyncio.wrap_future(f) for f in futures])

        logger.info("CPU stress test completed")

    async def inject_memory_stress(self, memory_mb: int = 512, duration: float = 30.0):
        """Inject memory stress"""
        logger.info(f"Starting memory stress: {memory_mb}MB for {duration}s")

        try:
            # Allocate memory in chunks to simulate gradual pressure
            chunk_size = 50 * 1024 * 1024  # 50MB chunks
            chunks_needed = memory_mb * 1024 * 1024 // chunk_size

            for i in range(chunks_needed):
                memory_chunk = bytearray(chunk_size)
                # Fill with random data to prevent optimization
                for j in range(0, len(memory_chunk), 1024):
                    memory_chunk[j] = random.randint(0, 255)
                self._memory_hogs.append(memory_chunk)

                # Brief pause between allocations
                await asyncio.sleep(0.1)

            # Hold memory for duration
            await asyncio.sleep(duration)

        finally:
            # Release memory
            self._memory_hogs.clear()
            logger.info("Memory stress test completed")

    async def inject_disk_stress(self, operations: int = 1000, file_size_mb: int = 100):
        """Inject disk I/O stress"""
        logger.info(f"Starting disk stress: {operations} operations with {file_size_mb}MB files")

        import tempfile
        import os

        temp_files = []

        try:
            # Create temporary files for I/O operations
            for i in range(min(10, operations // 100)):  # Limit number of files
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_files.append(temp_file.name)

                # Write data to file
                data = bytearray(file_size_mb * 1024 * 1024)
                temp_file.write(data)
                temp_file.flush()
                os.fsync(temp_file.fileno())
                temp_file.close()

            # Perform read/write operations
            for _ in range(operations):
                if temp_files:
                    file_path = random.choice(temp_files)

                    # Random read operation
                    if random.choice([True, False]):
                        with open(file_path, 'rb') as f:
                            f.read(1024 * 1024)  # Read 1MB
                    else:
                        # Random write operation
                        with open(file_path, 'ab') as f:
                            f.write(bytearray(1024 * 1024))

                await asyncio.sleep(0.001)  # Brief pause to prevent overwhelming

        finally:
            # Clean up temporary files
            for file_path in temp_files:
                try:
                    os.unlink(file_path)
                except:
                    pass

        logger.info("Disk stress test completed")

    def get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage metrics"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_io_read_mb': psutil.disk_io_counters().read_bytes / 1024 / 1024 if psutil.disk_io_counters() else 0,
            'disk_io_write_mb': psutil.disk_io_counters().write_bytes / 1024 / 1024 if psutil.disk_io_counters() else 0,
            'network_bytes_sent': psutil.net_io_counters().bytes_sent / 1024 / 1024,
            'network_bytes_recv': psutil.net_io_counters().bytes_recv / 1024 / 1024
        }


class ChaosOrchestrator:
    """Main chaos engineering orchestrator"""

    def __init__(self):
        self.network_chaos = NetworkChaos()
        self.resource_chaos = ResourceChaos()
        self.active_events: List[ChaosExecution] = []
        self.event_history: List[ChaosExecution] = []
        self._running = False

    async def execute_chaos_event(self, event: ChaosEvent) -> ChaosExecution:
        """Execute a single chaos event"""
        execution = ChaosExecution(
            event=event,
            start_time=datetime.now()
        )

        logger.info(f"Starting chaos event: {event.event_type.value} ({event.severity.value}) for {event.duration_seconds}s")

        try:
            # Check probability
            if random.random() > event.probability:
                execution.success = False
                execution.errors.append("Event skipped due to probability")
                return execution

            # Record baseline metrics
            baseline_metrics = self.resource_chaos.get_resource_usage()
            execution.impact_metrics['baseline'] = baseline_metrics

            # Execute chaos event based on type
            await self._execute_event_by_type(event, execution)

            # Record post-event metrics
            post_metrics = self.resource_chaos.get_resource_usage()
            execution.impact_metrics['post_event'] = post_metrics

            # Calculate impact
            execution.impact_metrics['impact'] = {
                key: post_metrics.get(key, 0) - baseline_metrics.get(key, 0)
                for key in baseline_metrics.keys()
            }

            execution.success = True

        except Exception as e:
            execution.success = False
            execution.errors.append(f"Chaos event failed: {str(e)}")
            logger.error(f"Chaos event {event.event_type.value} failed: {e}")

        finally:
            execution.end_time = datetime.now()

            # Execute recovery if provided
            if event.recovery_function:
                recovery_start = time.time()
                try:
                    await event.recovery_function()
                    execution.recovery_time = time.time() - recovery_start
                except Exception as e:
                    execution.errors.append(f"Recovery failed: {str(e)}")

        self.event_history.append(execution)
        logger.info(f"Chaos event completed: {event.event_type.value} - Success: {execution.success}")

        return execution

    async def _execute_event_by_type(self, event: ChaosEvent, execution: ChaosExecution):
        """Execute chaos event based on its type"""
        event_type = event.event_type
        duration = event.duration_seconds
        params = event.parameters

        if event_type == ChaosEventType.NETWORK_DELAY:
            min_delay = params.get('min_delay', 0.1)
            max_delay = params.get('max_delay', 2.0)
            target = params.get('target', 'all')

            task = asyncio.create_task(
                self.network_chaos.inject_latency(min_delay, max_delay, target)
            )
            await asyncio.sleep(duration)
            task.cancel()

        elif event_type == ChaosEventType.NETWORK_PACKET_LOSS:
            loss_rate = params.get('loss_rate', 0.1)
            target = params.get('target', 'all')

            task = asyncio.create_task(
                self.network_chaos.inject_packet_loss(loss_rate, target)
            )
            await asyncio.sleep(duration)
            task.cancel()

        elif event_type == ChaosEventType.NETWORK_PARTITION:
            targets = params.get('targets', ['service1', 'service2'])
            await self.network_chaos.inject_network_partition(targets, duration)

        elif event_type == ChaosEventType.CPU_STRESS:
            cpu_percent = params.get('cpu_percent', 80.0)
            await self.resource_chaos.inject_cpu_stress(cpu_percent, duration)

        elif event_type == ChaosEventType.MEMORY_STRESS:
            memory_mb = params.get('memory_mb', 512)
            await self.resource_chaos.inject_memory_stress(memory_mb, duration)

        elif event_type == ChaosEventType.DISK_STRESS:
            operations = params.get('operations', 1000)
            file_size_mb = params.get('file_size_mb', 100)
            await self.resource_chaos.inject_disk_stress(operations, file_size_mb)

        elif event_type == ChaosEventType.SERVICE_KILL:
            # Simulate service interruption
            logger.warning(f"Simulating service kill for {duration}s")
            await asyncio.sleep(duration)

        elif event_type == ChaosEventType.TIME_SKEW:
            # Simulate time-related issues
            skew_seconds = params.get('skew_seconds', 30)
            logger.warning(f"Simulating time skew of {skew_seconds}s for {duration}s")
            await asyncio.sleep(duration)

        else:
            raise ValueError(f"Unsupported chaos event type: {event_type}")

    async def run_chaos_scenario(self, events: List[ChaosEvent], concurrent: bool = False) -> List[ChaosExecution]:
        """Run multiple chaos events as a scenario"""
        self._running = True
        executions = []

        logger.info(f"Starting chaos scenario with {len(events)} events (concurrent: {concurrent})")

        try:
            if concurrent:
                # Run events concurrently
                tasks = [self.execute_chaos_event(event) for event in events]
                executions = await asyncio.gather(*tasks)
            else:
                # Run events sequentially
                for event in events:
                    if not self._running:
                        break
                    execution = await self.execute_chaos_event(event)
                    executions.append(execution)

        except Exception as e:
            logger.error(f"Chaos scenario failed: {e}")

        finally:
            self._running = False

        logger.info(f"Chaos scenario completed with {len(executions)} executions")
        return executions

    def create_random_chaos_scenario(self, duration_minutes: int = 30, event_count: int = 10) -> List[ChaosEvent]:
        """Create a random chaos scenario"""
        events = []
        scenario_duration = duration_minutes * 60

        for i in range(event_count):
            # Random event type
            event_type = random.choice(list(ChaosEventType))

            # Random severity
            severity = random.choice(list(ChaosEventSeverity))

            # Random duration (1-30 seconds)
            duration = random.uniform(1, 30)

            # Random parameters based on event type
            parameters = self._generate_random_parameters(event_type, severity)

            # Random probability (70-100%)
            probability = random.uniform(0.7, 1.0)

            event = ChaosEvent(
                event_type=event_type,
                severity=severity,
                duration_seconds=duration,
                parameters=parameters,
                probability=probability
            )

            events.append(event)

        return events

    def _generate_random_parameters(self, event_type: ChaosEventType, severity: ChaosEventSeverity) -> Dict[str, Any]:
        """Generate random parameters for chaos events"""
        severity_multipliers = {
            ChaosEventSeverity.LOW: 0.3,
            ChaosEventSeverity.MEDIUM: 0.6,
            ChaosEventSeverity.HIGH: 0.8,
            ChaosEventSeverity.CRITICAL: 1.0
        }

        multiplier = severity_multipliers[severity]
        params = {}

        if event_type == ChaosEventType.NETWORK_DELAY:
            params['min_delay'] = 0.1 * multiplier
            params['max_delay'] = 5.0 * multiplier

        elif event_type == ChaosEventType.NETWORK_PACKET_LOSS:
            params['loss_rate'] = 0.3 * multiplier

        elif event_type == ChaosEventType.CPU_STRESS:
            params['cpu_percent'] = 50 + (50 * multiplier)

        elif event_type == ChaosEventType.MEMORY_STRESS:
            params['memory_mb'] = int(256 + (1024 * multiplier))

        elif event_type == ChaosEventType.DISK_STRESS:
            params['operations'] = int(500 + (2000 * multiplier))
            params['file_size_mb'] = int(50 + (200 * multiplier))

        elif event_type == ChaosEventType.TIME_SKEW:
            params['skew_seconds'] = int(10 + (120 * multiplier))

        return params

    def stop_all_chaos(self):
        """Stop all active chaos events"""
        self._running = False
        self.network_chaos.stop_all_network_chaos()
        logger.info("All chaos events stopped")

    def get_chaos_summary(self) -> Dict[str, Any]:
        """Get summary of chaos engineering activities"""
        if not self.event_history:
            return {"error": "No chaos events executed"}

        total_events = len(self.event_history)
        successful_events = sum(1 for e in self.event_history if e.success)

        # Group by event type
        event_types = {}
        for execution in self.event_history:
            event_type = execution.event.event_type.value
            if event_type not in event_types:
                event_types[event_type] = {'count': 0, 'success': 0}
            event_types[event_type]['count'] += 1
            if execution.success:
                event_types[event_type]['success'] += 1

        # Calculate recovery times
        recovery_times = [e.recovery_time for e in self.event_history if e.recovery_time is not None]
        avg_recovery_time = sum(recovery_times) / len(recovery_times) if recovery_times else 0

        return {
            "summary": {
                "total_events": total_events,
                "successful_events": successful_events,
                "success_rate": (successful_events / total_events * 100) if total_events > 0 else 0,
                "average_recovery_time": avg_recovery_time
            },
            "event_types": event_types,
            "recent_events": [
                {
                    "type": e.event.event_type.value,
                    "severity": e.event.severity.value,
                    "start_time": e.start_time.isoformat(),
                    "duration": (e.end_time - e.start_time).total_seconds() if e.end_time else None,
                    "success": e.success,
                    "recovery_time": e.recovery_time
                }
                for e in self.event_history[-10:]  # Last 10 events
            ]
        }

    @asynccontextmanager
    async def chaos_context(self, events: List[ChaosEvent]):
        """Context manager for chaos engineering"""
        logger.info("Entering chaos context")

        # Start chaos events
        chaos_tasks = [asyncio.create_task(self.execute_chaos_event(event)) for event in events]

        try:
            yield self
        finally:
            # Clean up chaos events
            for task in chaos_tasks:
                task.cancel()

            self.stop_all_chaos()
            logger.info("Exiting chaos context")