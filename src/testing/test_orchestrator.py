"""
Test Orchestrator

Unified testing framework that coordinates reliability testing, chaos engineering,
and integration testing for comprehensive system validation.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

from .reliability_tester import ReliabilityTester, TestSeverity
from .chaos_engineering import ChaosOrchestrator, ChaosEvent, ChaosEventType, ChaosEventSeverity
from .integration_tester import IntegrationTester, IntegrationTestType

logger = logging.getLogger(__name__)


class TestSuiteType(Enum):
    """Types of test suites"""
    QUICK = "quick"          # Fast tests (< 5 minutes)
    STANDARD = "standard"    # Standard tests (< 30 minutes)
    COMPREHENSIVE = "comprehensive"  # Full test suite (< 2 hours)
    EXTENDED = "extended"    # Extended testing (< 24 hours)


@dataclass
class TestSuiteConfig:
    """Test suite configuration"""
    suite_type: TestSuiteType
    include_reliability: bool = True
    include_chaos: bool = True
    include_integration: bool = True
    max_duration_minutes: int = 120
    parallel_execution: bool = True
    generate_detailed_report: bool = True
    save_results: bool = True
    results_directory: str = "test_results"


@dataclass
class TestSuiteResults:
    """Complete test suite results"""
    config: TestSuiteConfig
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration: float = 0.0
    overall_success: bool = False
    integration_results: Optional[List] = None
    reliability_results: Optional[List] = None
    chaos_results: Optional[List] = None
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    detailed_report: Optional[Dict[str, Any]] = None


class TestOrchestrator:
    """Main test orchestrator for coordinating all testing activities"""

    def __init__(self):
        self.integration_tester = IntegrationTester()
        self.reliability_tester = ReliabilityTester()
        self.chaos_orchestrator = ChaosOrchestrator()
        self.test_history: List[TestSuiteResults] = []

    async def run_quick_test_suite(self) -> TestSuiteResults:
        """Run quick test suite (< 5 minutes)"""
        config = TestSuiteConfig(
            suite_type=TestSuiteType.QUICK,
            include_reliability=True,
            include_chaos=False,  # Skip chaos for quick tests
            include_integration=True,
            max_duration_minutes=5,
            parallel_execution=True
        )

        return await self.run_test_suite(config)

    async def run_standard_test_suite(self) -> TestSuiteResults:
        """Run standard test suite (< 30 minutes)"""
        config = TestSuiteConfig(
            suite_type=TestSuiteType.STANDARD,
            include_reliability=True,
            include_chaos=True,
            include_integration=True,
            max_duration_minutes=30,
            parallel_execution=True
        )

        return await self.run_test_suite(config)

    async def run_comprehensive_test_suite(self) -> TestSuiteResults:
        """Run comprehensive test suite (< 2 hours)"""
        config = TestSuiteConfig(
            suite_type=TestSuiteType.COMPREHENSIVE,
            include_reliability=True,
            include_chaos=True,
            include_integration=True,
            max_duration_minutes=120,
            parallel_execution=True,
            generate_detailed_report=True
        )

        return await self.run_test_suite(config)

    async def run_extended_test_suite(self) -> TestSuiteResults:
        """Run extended test suite (< 24 hours)"""
        config = TestSuiteConfig(
            suite_type=TestSuiteType.EXTENDED,
            include_reliability=True,
            include_chaos=True,
            include_integration=True,
            max_duration_minutes=1440,  # 24 hours
            parallel_execution=True,
            generate_detailed_report=True
        )

        return await self.run_test_suite(config)

    async def run_test_suite(self, config: TestSuiteConfig) -> TestSuiteResults:
        """Run test suite based on configuration"""
        results = TestSuiteResults(
            config=config,
            start_time=datetime.now()
        )

        logger.info(f"Starting {config.suite_type.value} test suite")

        try:
            # Initialize test cases based on suite type
            await self._initialize_test_cases(config)

            # Run tests based on configuration
            test_tasks = []

            if config.include_integration:
                test_tasks.append(self._run_integration_tests(config))

            if config.include_reliability:
                test_tasks.append(self._run_reliability_tests(config))

            if config.include_chaos:
                test_tasks.append(self._run_chaos_tests(config))

            # Execute tests
            if config.parallel_execution and len(test_tasks) > 1:
                # Run tests in parallel
                test_results = await asyncio.gather(*test_tasks, return_exceptions=True)
            else:
                # Run tests sequentially
                test_results = []
                for task in test_tasks:
                    result = await task
                    test_results.append(result)

            # Process results
            results.integration_results = test_results[0] if config.include_integration else None
            results.reliability_results = test_results[1] if config.include_reliability else None
            results.chaos_results = test_results[2] if config.include_chaos else None

            # Generate summary and recommendations
            results.summary_stats = self._generate_summary_stats(results)
            results.recommendations = self._generate_recommendations(results)
            results.overall_success = self._evaluate_overall_success(results)

            # Generate detailed report if requested
            if config.generate_detailed_report:
                results.detailed_report = self._generate_detailed_report(results)

        except Exception as e:
            logger.error(f"Test suite execution failed: {e}")
            results.overall_success = False

        finally:
            results.end_time = datetime.now()
            results.total_duration = (results.end_time - results.start_time).total_seconds()

        # Save results if requested
        if config.save_results:
            await self._save_test_results(results)

        self.test_history.append(results)
        logger.info(f"Test suite completed: {config.suite_type.value} - Success: {results.overall_success}")

        return results

    async def _initialize_test_cases(self, config: TestSuiteConfig):
        """Initialize test cases based on suite configuration"""
        if config.include_integration:
            # Add integration test cases based on suite type
            self.integration_tester.create_api_integration_tests()
            self.integration_tester.create_data_flow_tests()

            if config.suite_type != TestSuiteType.QUICK:
                self.integration_tester.create_performance_integration_tests()
                self.integration_tester.create_configuration_integration_tests()

        if config.include_reliability:
            # Add reliability test scenarios based on suite type
            if config.suite_type == TestSuiteType.QUICK:
                # Quick stress test
                self.reliability_tester.add_stress_test_scenario(
                    "Quick API Stress Test",
                    target_function=self._mock_api_call,
                    concurrent_requests=20,
                    duration_seconds=60
                )
            elif config.suite_type == TestSuiteType.STANDARD:
                # Standard stress tests
                self.reliability_tester.add_stress_test_scenario(
                    "Standard API Stress Test",
                    target_function=self._mock_api_call,
                    concurrent_requests=50,
                    duration_seconds=300
                )
            else:
                # Comprehensive stress and endurance tests
                self.reliability_tester.add_stress_test_scenario(
                    "High Load Stress Test",
                    target_function=self._mock_api_call,
                    concurrent_requests=100,
                    duration_seconds=600
                )

                if config.suite_type == TestSuiteType.EXTENDED:
                    self.reliability_tester.add_endurance_test_scenario(
                        "24-Hour Endurance Test",
                        target_function=self._mock_api_call,
                        duration_hours=24,
                        requests_per_minute=30
                    )

    async def _run_integration_tests(self, config: TestSuiteConfig) -> List:
        """Run integration tests"""
        logger.info("Running integration tests")

        if config.suite_type == TestSuiteType.QUICK:
            # Run only API integration tests for quick suite
            return await self.integration_tester.run_tests_by_type(IntegrationTestType.API_INTEGRATION)
        else:
            # Run all integration tests
            return await self.integration_tester.run_all_tests()

    async def _run_reliability_tests(self, config: TestSuiteConfig) -> List:
        """Run reliability tests"""
        logger.info("Running reliability tests")
        return await self.reliability_tester.run_all_scenarios()

    async def _run_chaos_tests(self, config: TestSuiteConfig) -> List:
        """Run chaos engineering tests"""
        logger.info("Running chaos engineering tests")

        # Create chaos events based on suite type
        if config.suite_type == TestSuiteType.STANDARD:
            chaos_events = [
                ChaosEvent(
                    event_type=ChaosEventType.NETWORK_DELAY,
                    severity=ChaosEventSeverity.MEDIUM,
                    duration_seconds=30,
                    parameters={'min_delay': 0.1, 'max_delay': 1.0}
                ),
                ChaosEvent(
                    event_type=ChaosEventType.CPU_STRESS,
                    severity=ChaosEventSeverity.MEDIUM,
                    duration_seconds=60,
                    parameters={'cpu_percent': 70}
                )
            ]
        elif config.suite_type == TestSuiteType.COMPREHENSIVE:
            chaos_events = [
                ChaosEvent(
                    event_type=ChaosEventType.NETWORK_DELAY,
                    severity=ChaosEventSeverity.HIGH,
                    duration_seconds=60,
                    parameters={'min_delay': 0.2, 'max_delay': 2.0}
                ),
                ChaosEvent(
                    event_type=ChaosEventType.CPU_STRESS,
                    severity=ChaosEventSeverity.HIGH,
                    duration_seconds=120,
                    parameters={'cpu_percent': 85}
                ),
                ChaosEvent(
                    event_type=ChaosEventType.MEMORY_STRESS,
                    severity=ChaosEventSeverity.MEDIUM,
                    duration_seconds=90,
                    parameters={'memory_mb': 512}
                )
            ]
        else:  # EXTENDED
            # Use random chaos scenario for extended testing
            chaos_events = self.chaos_orchestrator.create_random_chaos_scenario(
                duration_minutes=config.max_duration_minutes // 4,  # 25% of total time
                event_count=20
            )

        return await self.chaos_orchestrator.run_chaos_scenario(chaos_events)

    async def _mock_api_call(self):
        """Mock API call for testing"""
        import random
        await asyncio.sleep(random.uniform(0.01, 0.1))
        if random.random() < 0.02:  # 2% error rate
            raise Exception("Mock API error")
        return {"status": "success", "data": "mock_data"}

    def _generate_summary_stats(self, results: TestSuiteResults) -> Dict[str, Any]:
        """Generate summary statistics"""
        stats = {
            "test_suite_type": results.config.suite_type.value,
            "total_duration_minutes": results.total_duration / 60,
            "test_categories_executed": 0,
            "overall_success_rate": 0.0
        }

        category_results = []

        # Integration test stats
        if results.integration_results:
            stats["test_categories_executed"] += 1
            integration_passed = sum(1 for r in results.integration_results if r.success)
            integration_total = len(results.integration_results)
            integration_success_rate = (integration_passed / integration_total * 100) if integration_total > 0 else 0

            stats["integration_tests"] = {
                "total": integration_total,
                "passed": integration_passed,
                "success_rate": integration_success_rate
            }
            category_results.append(integration_success_rate)

        # Reliability test stats
        if results.reliability_results:
            stats["test_categories_executed"] += 1
            reliability_stats = self.reliability_tester.get_summary_stats()
            reliability_success_rate = reliability_stats.get('pass_rate', 0) * 100

            stats["reliability_tests"] = reliability_stats
            category_results.append(reliability_success_rate)

        # Chaos test stats
        if results.chaos_results:
            stats["test_categories_executed"] += 1
            chaos_summary = self.chaos_orchestrator.get_chaos_summary()
            chaos_success_rate = chaos_summary.get('summary', {}).get('success_rate', 0)

            stats["chaos_tests"] = chaos_summary
            category_results.append(chaos_success_rate)

        # Overall success rate
        if category_results:
            stats["overall_success_rate"] = sum(category_results) / len(category_results)

        return stats

    def _generate_recommendations(self, results: TestSuiteResults) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []

        # Check overall success rate
        overall_rate = results.summary_stats.get("overall_success_rate", 0)
        if overall_rate < 95:
            recommendations.append(f"Overall success rate ({overall_rate:.1f}%) below 95% - investigate failures")

        # Check integration tests
        if results.integration_results:
            failed_integration = [r for r in results.integration_results if not r.success]
            if failed_integration:
                recommendations.append(f"Address {len(failed_integration)} failed integration tests")

            # Check slow tests
            slow_tests = [r for r in results.integration_results if r.execution_time > 5.0]
            if slow_tests:
                recommendations.append(f"Optimize {len(slow_tests)} slow integration tests")

        # Check reliability metrics
        if results.reliability_results:
            reliability_stats = self.reliability_tester.get_summary_stats()
            avg_error_rate = reliability_stats.get('average_error_rate', 0)
            if avg_error_rate > 0.05:
                recommendations.append(f"High error rate ({avg_error_rate:.2%}) - improve error handling")

            avg_uptime = reliability_stats.get('average_uptime', 100)
            if avg_uptime < 99:
                recommendations.append(f"Low uptime ({avg_uptime:.1f}%) - improve system stability")

        # Check chaos engineering results
        if results.chaos_results:
            chaos_summary = self.chaos_orchestrator.get_chaos_summary()
            chaos_success_rate = chaos_summary.get('summary', {}).get('success_rate', 100)
            if chaos_success_rate < 80:
                recommendations.append(f"Low chaos resilience ({chaos_success_rate:.1f}%) - improve fault tolerance")

        # Performance recommendations
        if results.total_duration > results.config.max_duration_minutes * 60:
            recommendations.append("Test suite exceeded time limit - consider optimization")

        # Positive feedback
        if not recommendations:
            recommendations.append("All tests passed successfully - system shows excellent reliability")

        return recommendations

    def _evaluate_overall_success(self, results: TestSuiteResults) -> bool:
        """Evaluate overall test suite success"""
        overall_rate = results.summary_stats.get("overall_success_rate", 0)
        return overall_rate >= 95  # 95% threshold for overall success

    def _generate_detailed_report(self, results: TestSuiteResults) -> Dict[str, Any]:
        """Generate detailed test report"""
        return {
            "test_suite_info": {
                "type": results.config.suite_type.value,
                "start_time": results.start_time.isoformat(),
                "end_time": results.end_time.isoformat() if results.end_time else None,
                "duration_minutes": results.total_duration / 60,
                "parallel_execution": results.config.parallel_execution
            },
            "summary": results.summary_stats,
            "integration_test_details": [
                {
                    "name": r.test_case.name,
                    "type": r.test_case.test_type.value,
                    "success": r.success,
                    "execution_time": r.execution_time,
                    "components": r.test_case.components,
                    "errors": r.errors,
                    "warnings": r.warnings
                }
                for r in (results.integration_results or [])
            ],
            "reliability_test_details": self.reliability_tester.generate_report() if results.reliability_results else {},
            "chaos_test_details": self.chaos_orchestrator.get_chaos_summary() if results.chaos_results else {},
            "recommendations": results.recommendations,
            "environment_info": {
                "python_version": "3.13+",
                "test_framework_version": "1.0.0",
                "execution_platform": "Market Data Agent Test Suite"
            }
        }

    async def _save_test_results(self, results: TestSuiteResults):
        """Save test results to file"""
        import os
        import json

        # Create results directory
        os.makedirs(results.config.results_directory, exist_ok=True)

        # Generate filename with timestamp
        timestamp = results.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"{results.config.suite_type.value}_test_results_{timestamp}.json"
        filepath = os.path.join(results.config.results_directory, filename)

        # Prepare data for JSON serialization
        results_data = {
            "config": {
                "suite_type": results.config.suite_type.value,
                "max_duration_minutes": results.config.max_duration_minutes,
                "parallel_execution": results.config.parallel_execution
            },
            "start_time": results.start_time.isoformat(),
            "end_time": results.end_time.isoformat() if results.end_time else None,
            "total_duration": results.total_duration,
            "overall_success": results.overall_success,
            "summary_stats": results.summary_stats,
            "recommendations": results.recommendations,
            "detailed_report": results.detailed_report
        }

        try:
            with open(filepath, 'w') as f:
                json.dump(results_data, f, indent=2)
            logger.info(f"Test results saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save test results: {e}")

    def get_test_history(self, limit: int = 10) -> List[TestSuiteResults]:
        """Get recent test history"""
        return self.test_history[-limit:]

    def get_test_trends(self) -> Dict[str, Any]:
        """Get test trend analysis"""
        if len(self.test_history) < 2:
            return {"error": "Insufficient test history for trend analysis"}

        recent_results = self.test_history[-10:]  # Last 10 test runs

        # Calculate trends
        success_rates = [r.summary_stats.get("overall_success_rate", 0) for r in recent_results]
        durations = [r.total_duration / 60 for r in recent_results]  # Convert to minutes

        return {
            "test_runs_analyzed": len(recent_results),
            "success_rate_trend": {
                "current": success_rates[-1] if success_rates else 0,
                "average": sum(success_rates) / len(success_rates) if success_rates else 0,
                "trend": "improving" if len(success_rates) > 1 and success_rates[-1] > success_rates[-2] else "stable"
            },
            "duration_trend": {
                "current_minutes": durations[-1] if durations else 0,
                "average_minutes": sum(durations) / len(durations) if durations else 0,
                "trend": "faster" if len(durations) > 1 and durations[-1] < durations[-2] else "stable"
            },
            "reliability_indicators": {
                "consistent_success": all(r.overall_success for r in recent_results[-5:]),
                "stability_score": sum(1 for r in recent_results if r.overall_success) / len(recent_results) * 100
            }
        }