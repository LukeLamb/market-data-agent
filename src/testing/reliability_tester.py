"""
Reliability Testing Framework

Comprehensive reliability testing including stress testing, fault injection,
recovery testing, and system resilience validation for the Market Data Agent.
"""

import asyncio
import random
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)


class TestSeverity(Enum):
    """Test severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TestResult(Enum):
    """Test result types"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ReliabilityMetrics:
    """Reliability test metrics"""
    uptime_percentage: float = 0.0
    mean_time_to_failure: float = 0.0  # seconds
    mean_time_to_recovery: float = 0.0  # seconds
    error_rate: float = 0.0
    response_time_p95: float = 0.0
    response_time_p99: float = 0.0
    throughput_requests_per_second: float = 0.0
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    failure_count: int = 0
    recovery_count: int = 0


@dataclass
class TestScenario:
    """Reliability test scenario"""
    name: str
    description: str
    severity: TestSeverity
    duration_seconds: int
    target_function: Callable
    fault_injection: Optional[Callable] = None
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    setup_function: Optional[Callable] = None
    teardown_function: Optional[Callable] = None


@dataclass
class TestExecution:
    """Test execution result"""
    scenario: TestScenario
    start_time: datetime
    end_time: Optional[datetime] = None
    result: TestResult = TestResult.ERROR
    metrics: ReliabilityMetrics = field(default_factory=ReliabilityMetrics)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_log: List[str] = field(default_factory=list)


class FaultInjector:
    """Fault injection utilities for chaos engineering"""

    @staticmethod
    async def network_delay(min_delay: float = 0.1, max_delay: float = 2.0):
        """Inject network delay"""
        delay = random.uniform(min_delay, max_delay)
        logger.debug(f"Injecting network delay: {delay:.2f}s")
        await asyncio.sleep(delay)

    @staticmethod
    async def network_failure(failure_probability: float = 0.1):
        """Inject network failure"""
        if random.random() < failure_probability:
            logger.debug("Injecting network failure")
            raise ConnectionError("Simulated network failure")

    @staticmethod
    async def memory_pressure(allocation_mb: int = 100, duration: float = 5.0):
        """Inject memory pressure"""
        logger.debug(f"Injecting memory pressure: {allocation_mb}MB for {duration}s")
        # Simulate memory allocation
        memory_hog = bytearray(allocation_mb * 1024 * 1024)
        await asyncio.sleep(duration)
        del memory_hog

    @staticmethod
    async def cpu_spike(duration: float = 5.0, intensity: float = 0.8):
        """Inject CPU spike"""
        logger.debug(f"Injecting CPU spike: {intensity*100}% for {duration}s")
        start_time = time.time()

        def cpu_burn():
            while time.time() - start_time < duration:
                # Busy wait to consume CPU
                for _ in range(10000):
                    pass
                # Brief sleep to control intensity
                time.sleep(0.001 * (1 - intensity))

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(cpu_burn) for _ in range(4)]
            await asyncio.gather(*[asyncio.wrap_future(f) for f in futures])

    @staticmethod
    async def disk_io_stress(operations: int = 1000, file_size_mb: int = 10):
        """Inject disk I/O stress"""
        logger.debug(f"Injecting disk I/O stress: {operations} operations")
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            try:
                # Write operations
                data = bytearray(file_size_mb * 1024 * 1024)
                for _ in range(operations // 2):
                    tmp_file.write(data)
                    tmp_file.flush()
                    os.fsync(tmp_file.fileno())

                # Read operations
                tmp_file.seek(0)
                for _ in range(operations // 2):
                    tmp_file.read(len(data))

            finally:
                os.unlink(tmp_file.name)


class ReliabilityTester:
    """Main reliability testing framework"""

    def __init__(self):
        self.scenarios: List[TestScenario] = []
        self.executions: List[TestExecution] = []
        self.fault_injector = FaultInjector()
        self._running = False

    def add_scenario(self, scenario: TestScenario):
        """Add test scenario"""
        self.scenarios.append(scenario)
        logger.info(f"Added test scenario: {scenario.name}")

    def add_stress_test_scenario(
        self,
        name: str,
        target_function: Callable,
        concurrent_requests: int = 100,
        duration_seconds: int = 300,
        ramp_up_seconds: int = 30
    ):
        """Add stress test scenario"""
        async def stress_test():
            """Execute stress test"""
            results = []
            start_time = time.time()

            # Ramp up period
            ramp_increment = concurrent_requests / (ramp_up_seconds * 10)
            current_requests = 0

            while time.time() - start_time < duration_seconds:
                if time.time() - start_time < ramp_up_seconds:
                    current_requests = min(concurrent_requests, current_requests + ramp_increment)
                else:
                    current_requests = concurrent_requests

                # Execute concurrent requests
                tasks = []
                for _ in range(int(current_requests)):
                    tasks.append(asyncio.create_task(self._execute_with_timing(target_function)))

                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                results.extend(batch_results)

                await asyncio.sleep(0.1)  # Brief pause between batches

            return results

        scenario = TestScenario(
            name=name,
            description=f"Stress test with {concurrent_requests} concurrent requests for {duration_seconds}s",
            severity=TestSeverity.HIGH,
            duration_seconds=duration_seconds,
            target_function=stress_test,
            success_criteria={
                'error_rate_threshold': 0.05,  # 5% max error rate
                'response_time_p95_threshold': 5.0,  # 5s max P95 response time
                'min_throughput': concurrent_requests * 0.8  # 80% of target throughput
            }
        )

        self.add_scenario(scenario)

    def add_endurance_test_scenario(
        self,
        name: str,
        target_function: Callable,
        duration_hours: int = 24,
        requests_per_minute: int = 60
    ):
        """Add endurance/soak test scenario"""
        async def endurance_test():
            """Execute endurance test"""
            results = []
            start_time = time.time()
            duration_seconds = duration_hours * 3600

            while time.time() - start_time < duration_seconds:
                # Execute requests at target rate
                batch_start = time.time()
                tasks = []

                for _ in range(requests_per_minute):
                    tasks.append(asyncio.create_task(self._execute_with_timing(target_function)))

                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                results.extend(batch_results)

                # Maintain target rate
                batch_duration = time.time() - batch_start
                sleep_time = max(0, 60 - batch_duration)  # 60 seconds per batch
                await asyncio.sleep(sleep_time)

            return results

        scenario = TestScenario(
            name=name,
            description=f"Endurance test for {duration_hours} hours at {requests_per_minute} req/min",
            severity=TestSeverity.CRITICAL,
            duration_seconds=duration_hours * 3600,
            target_function=endurance_test,
            success_criteria={
                'error_rate_threshold': 0.01,  # 1% max error rate for endurance
                'memory_leak_threshold': 50,   # 50MB max memory increase
                'uptime_threshold': 99.9       # 99.9% uptime required
            }
        )

        self.add_scenario(scenario)

    def add_chaos_engineering_scenario(
        self,
        name: str,
        target_function: Callable,
        fault_types: List[str],
        duration_seconds: int = 600
    ):
        """Add chaos engineering scenario"""
        async def chaos_test():
            """Execute chaos engineering test"""
            results = []
            fault_functions = {
                'network_delay': self.fault_injector.network_delay,
                'network_failure': lambda: self.fault_injector.network_failure(0.2),
                'memory_pressure': lambda: self.fault_injector.memory_pressure(200, 10),
                'cpu_spike': lambda: self.fault_injector.cpu_spike(10, 0.9),
                'disk_io_stress': lambda: self.fault_injector.disk_io_stress(500, 5)
            }

            start_time = time.time()
            while time.time() - start_time < duration_seconds:
                # Randomly inject faults
                if random.random() < 0.3:  # 30% chance of fault injection
                    fault_type = random.choice(fault_types)
                    if fault_type in fault_functions:
                        try:
                            await fault_functions[fault_type]()
                        except Exception as e:
                            logger.debug(f"Fault injection {fault_type} caused: {e}")

                # Execute target function under chaos
                result = await self._execute_with_timing(target_function)
                results.append(result)

                await asyncio.sleep(1)

            return results

        scenario = TestScenario(
            name=name,
            description=f"Chaos engineering with {', '.join(fault_types)} for {duration_seconds}s",
            severity=TestSeverity.HIGH,
            duration_seconds=duration_seconds,
            target_function=chaos_test,
            success_criteria={
                'error_rate_threshold': 0.15,  # 15% max error rate under chaos
                'recovery_time_threshold': 30,  # 30s max recovery time
                'availability_threshold': 95    # 95% availability under chaos
            }
        )

        self.add_scenario(scenario)

    async def _execute_with_timing(self, func: Callable) -> Tuple[float, Any, Optional[Exception]]:
        """Execute function with timing measurement"""
        start_time = time.time()
        exception = None
        result = None

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func()
            else:
                result = func()
        except Exception as e:
            exception = e

        execution_time = time.time() - start_time
        return execution_time, result, exception

    async def run_scenario(self, scenario: TestScenario) -> TestExecution:
        """Run single test scenario"""
        execution = TestExecution(
            scenario=scenario,
            start_time=datetime.now()
        )

        execution.execution_log.append(f"Starting scenario: {scenario.name}")
        logger.info(f"Running reliability test: {scenario.name}")

        try:
            # Setup
            if scenario.setup_function:
                await scenario.setup_function()
                execution.execution_log.append("Setup completed")

            # Execute test
            start_time = time.time()
            test_results = await scenario.target_function()
            end_time = time.time()

            # Analyze results
            execution.metrics = self._analyze_results(test_results, end_time - start_time)
            execution.result = self._evaluate_success_criteria(execution.metrics, scenario.success_criteria)

            execution.execution_log.append(f"Test completed in {end_time - start_time:.2f}s")

        except Exception as e:
            execution.result = TestResult.ERROR
            execution.errors.append(f"Test execution failed: {str(e)}")
            logger.error(f"Test scenario {scenario.name} failed: {e}")

        finally:
            # Teardown
            if scenario.teardown_function:
                try:
                    await scenario.teardown_function()
                    execution.execution_log.append("Teardown completed")
                except Exception as e:
                    execution.warnings.append(f"Teardown failed: {str(e)}")

            execution.end_time = datetime.now()

        self.executions.append(execution)
        logger.info(f"Completed test scenario: {scenario.name} - Result: {execution.result.value}")

        return execution

    def _analyze_results(self, test_results: List[Any], total_duration: float) -> ReliabilityMetrics:
        """Analyze test results and compute metrics"""
        metrics = ReliabilityMetrics()

        if not test_results:
            return metrics

        # Extract timing and error data
        response_times = []
        errors = 0
        successes = 0

        for result in test_results:
            if isinstance(result, tuple) and len(result) == 3:
                timing, response, exception = result
                response_times.append(timing)

                if exception:
                    errors += 1
                else:
                    successes += 1
            elif isinstance(result, Exception):
                errors += 1

        total_requests = len(test_results)

        # Calculate metrics
        if total_requests > 0:
            metrics.error_rate = errors / total_requests
            metrics.throughput_requests_per_second = total_requests / total_duration

        if response_times:
            metrics.response_time_p95 = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
            metrics.response_time_p99 = statistics.quantiles(response_times, n=100)[98]  # 99th percentile

        metrics.uptime_percentage = (successes / total_requests * 100) if total_requests > 0 else 0
        metrics.failure_count = errors

        return metrics

    def _evaluate_success_criteria(self, metrics: ReliabilityMetrics, criteria: Dict[str, Any]) -> TestResult:
        """Evaluate test success based on criteria"""
        if not criteria:
            return TestResult.PASS

        warnings = []
        failures = []

        # Check error rate
        if 'error_rate_threshold' in criteria:
            if metrics.error_rate > criteria['error_rate_threshold']:
                failures.append(f"Error rate {metrics.error_rate:.2%} exceeds threshold {criteria['error_rate_threshold']:.2%}")

        # Check response time
        if 'response_time_p95_threshold' in criteria:
            if metrics.response_time_p95 > criteria['response_time_p95_threshold']:
                failures.append(f"P95 response time {metrics.response_time_p95:.2f}s exceeds threshold {criteria['response_time_p95_threshold']:.2f}s")

        # Check throughput
        if 'min_throughput' in criteria:
            if metrics.throughput_requests_per_second < criteria['min_throughput']:
                warnings.append(f"Throughput {metrics.throughput_requests_per_second:.2f} req/s below target {criteria['min_throughput']} req/s")

        # Check uptime
        if 'uptime_threshold' in criteria:
            if metrics.uptime_percentage < criteria['uptime_threshold']:
                failures.append(f"Uptime {metrics.uptime_percentage:.1f}% below threshold {criteria['uptime_threshold']:.1f}%")

        if failures:
            return TestResult.FAIL
        elif warnings:
            return TestResult.WARNING
        else:
            return TestResult.PASS

    async def run_all_scenarios(self) -> List[TestExecution]:
        """Run all test scenarios"""
        self._running = True
        logger.info(f"Starting reliability test suite with {len(self.scenarios)} scenarios")

        results = []
        for scenario in self.scenarios:
            if not self._running:
                logger.info("Test suite interrupted")
                break

            execution = await self.run_scenario(scenario)
            results.append(execution)

        self._running = False
        logger.info("Reliability test suite completed")

        return results

    def stop_tests(self):
        """Stop running tests"""
        self._running = False
        logger.info("Stopping reliability tests...")

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        if not self.executions:
            return {"error": "No test executions found"}

        # Overall statistics
        total_tests = len(self.executions)
        passed_tests = sum(1 for e in self.executions if e.result == TestResult.PASS)
        failed_tests = sum(1 for e in self.executions if e.result == TestResult.FAIL)
        warning_tests = sum(1 for e in self.executions if e.result == TestResult.WARNING)
        error_tests = sum(1 for e in self.executions if e.result == TestResult.ERROR)

        # Aggregate metrics
        all_metrics = [e.metrics for e in self.executions if e.metrics]
        avg_error_rate = statistics.mean([m.error_rate for m in all_metrics]) if all_metrics else 0
        avg_uptime = statistics.mean([m.uptime_percentage for m in all_metrics]) if all_metrics else 0

        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "warnings": warning_tests,
                "errors": error_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                "average_error_rate": avg_error_rate,
                "average_uptime": avg_uptime
            },
            "executions": []
        }

        # Individual test results
        for execution in self.executions:
            execution_report = {
                "scenario_name": execution.scenario.name,
                "severity": execution.scenario.severity.value,
                "result": execution.result.value,
                "start_time": execution.start_time.isoformat(),
                "end_time": execution.end_time.isoformat() if execution.end_time else None,
                "duration_seconds": (execution.end_time - execution.start_time).total_seconds() if execution.end_time else None,
                "metrics": {
                    "uptime_percentage": execution.metrics.uptime_percentage,
                    "error_rate": execution.metrics.error_rate,
                    "response_time_p95": execution.metrics.response_time_p95,
                    "response_time_p99": execution.metrics.response_time_p99,
                    "throughput_rps": execution.metrics.throughput_requests_per_second,
                    "failure_count": execution.metrics.failure_count
                },
                "errors": execution.errors,
                "warnings": execution.warnings
            }
            report["executions"].append(execution_report)

        return report

    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics"""
        if not self.executions:
            return {}

        all_metrics = [e.metrics for e in self.executions]

        return {
            "total_scenarios": len(self.executions),
            "pass_rate": len([e for e in self.executions if e.result == TestResult.PASS]) / len(self.executions),
            "average_uptime": statistics.mean([m.uptime_percentage for m in all_metrics]),
            "average_error_rate": statistics.mean([m.error_rate for m in all_metrics]),
            "average_p95_response_time": statistics.mean([m.response_time_p95 for m in all_metrics if m.response_time_p95 > 0]),
            "total_failures": sum([m.failure_count for m in all_metrics])
        }