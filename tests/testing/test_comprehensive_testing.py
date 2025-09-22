"""
Tests for Comprehensive Testing Framework

Tests for reliability testing, chaos engineering, integration testing,
and test orchestration capabilities.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock

from src.testing.reliability_tester import (
    ReliabilityTester,
    TestScenario,
    TestSeverity,
    TestResult,
    ReliabilityMetrics,
    FaultInjector
)
from src.testing.chaos_engineering import (
    ChaosOrchestrator,
    ChaosEvent,
    ChaosEventType,
    ChaosEventSeverity,
    NetworkChaos,
    ResourceChaos
)
from src.testing.integration_tester import (
    IntegrationTester,
    IntegrationTestType,
    IntegrationTestCase,
    ComponentMockManager
)
from src.testing.test_orchestrator import (
    TestOrchestrator,
    TestSuiteType,
    TestSuiteConfig
)


class TestReliabilityTester:
    """Test suite for ReliabilityTester"""

    @pytest.fixture
    def reliability_tester(self):
        """Create reliability tester instance"""
        return ReliabilityTester()

    @pytest.fixture
    def mock_target_function(self):
        """Mock target function for testing"""
        async def mock_function():
            await asyncio.sleep(0.01)
            return "success"
        return mock_function

    def test_reliability_tester_initialization(self, reliability_tester):
        """Test reliability tester initialization"""
        assert len(reliability_tester.scenarios) == 0
        assert len(reliability_tester.executions) == 0
        assert reliability_tester.fault_injector is not None

    def test_add_scenario(self, reliability_tester, mock_target_function):
        """Test adding test scenarios"""
        scenario = TestScenario(
            name="Test Scenario",
            description="Test description",
            severity=TestSeverity.MEDIUM,
            duration_seconds=60,
            target_function=mock_target_function
        )

        reliability_tester.add_scenario(scenario)
        assert len(reliability_tester.scenarios) == 1
        assert reliability_tester.scenarios[0] == scenario

    def test_add_stress_test_scenario(self, reliability_tester, mock_target_function):
        """Test adding stress test scenario"""
        reliability_tester.add_stress_test_scenario(
            name="Stress Test",
            target_function=mock_target_function,
            concurrent_requests=10,
            duration_seconds=5
        )

        assert len(reliability_tester.scenarios) == 1
        scenario = reliability_tester.scenarios[0]
        assert scenario.name == "Stress Test"
        assert scenario.severity == TestSeverity.HIGH

    @pytest.mark.asyncio
    async def test_run_scenario(self, reliability_tester, mock_target_function):
        """Test running a single scenario"""
        scenario = TestScenario(
            name="Test Scenario",
            description="Test description",
            severity=TestSeverity.LOW,
            duration_seconds=1,
            target_function=mock_target_function
        )

        execution = await reliability_tester.run_scenario(scenario)

        assert execution.scenario == scenario
        assert execution.start_time is not None
        assert execution.end_time is not None
        assert execution.result in [TestResult.PASS, TestResult.FAIL, TestResult.WARNING, TestResult.ERROR]

    def test_analyze_results(self, reliability_tester):
        """Test result analysis"""
        # Mock test results
        test_results = [
            (0.1, "success", None),  # Success with 0.1s response time
            (0.2, "success", None),  # Success with 0.2s response time
            (0.3, None, Exception("Error")),  # Error
        ]

        metrics = reliability_tester._analyze_results(test_results, 10.0)

        assert metrics.error_rate == 1/3  # 1 error out of 3 requests
        assert metrics.throughput_requests_per_second == 3/10  # 3 requests in 10 seconds
        assert metrics.response_time_p95 > 0
        assert metrics.failure_count == 1

    def test_evaluate_success_criteria(self, reliability_tester):
        """Test success criteria evaluation"""
        metrics = ReliabilityMetrics()
        metrics.error_rate = 0.02  # 2% error rate
        metrics.response_time_p95 = 1.5  # 1.5s P95 response time
        metrics.uptime_percentage = 98.0  # 98% uptime

        # Test passing criteria
        criteria = {
            'error_rate_threshold': 0.05,  # 5% max
            'response_time_p95_threshold': 2.0,  # 2s max
            'uptime_threshold': 95.0  # 95% min
        }

        result = reliability_tester._evaluate_success_criteria(metrics, criteria)
        assert result == TestResult.PASS

        # Test failing criteria
        criteria['error_rate_threshold'] = 0.01  # 1% max (tighter than actual 2%)
        result = reliability_tester._evaluate_success_criteria(metrics, criteria)
        assert result == TestResult.FAIL


class TestFaultInjector:
    """Test suite for FaultInjector"""

    @pytest.fixture
    def fault_injector(self):
        """Create fault injector instance"""
        return FaultInjector()

    @pytest.mark.asyncio
    async def test_network_delay(self, fault_injector):
        """Test network delay injection"""
        start_time = time.time()
        await fault_injector.network_delay(min_delay=0.1, max_delay=0.2)
        elapsed = time.time() - start_time

        assert 0.1 <= elapsed <= 0.3  # Allow some margin for execution overhead

    @pytest.mark.asyncio
    async def test_network_failure(self, fault_injector):
        """Test network failure injection"""
        # Test with 100% failure probability
        with pytest.raises(ConnectionError):
            await fault_injector.network_failure(failure_probability=1.0)

        # Test with 0% failure probability (should not raise)
        await fault_injector.network_failure(failure_probability=0.0)

    @pytest.mark.asyncio
    async def test_memory_pressure(self, fault_injector):
        """Test memory pressure injection"""
        # Test small memory allocation
        await fault_injector.memory_pressure(allocation_mb=1, duration=0.1)
        # Should complete without error

    @pytest.mark.asyncio
    async def test_cpu_spike(self, fault_injector):
        """Test CPU spike injection"""
        start_time = time.time()
        await fault_injector.cpu_spike(duration=0.1, intensity=0.5)
        elapsed = time.time() - start_time

        assert elapsed >= 0.1  # Should take at least the specified duration


class TestChaosOrchestrator:
    """Test suite for ChaosOrchestrator"""

    @pytest.fixture
    def chaos_orchestrator(self):
        """Create chaos orchestrator instance"""
        return ChaosOrchestrator()

    @pytest.fixture
    def sample_chaos_event(self):
        """Create sample chaos event"""
        return ChaosEvent(
            event_type=ChaosEventType.NETWORK_DELAY,
            severity=ChaosEventSeverity.LOW,
            duration_seconds=0.1,
            parameters={'min_delay': 0.01, 'max_delay': 0.05}
        )

    def test_chaos_orchestrator_initialization(self, chaos_orchestrator):
        """Test chaos orchestrator initialization"""
        assert chaos_orchestrator.network_chaos is not None
        assert chaos_orchestrator.resource_chaos is not None
        assert len(chaos_orchestrator.active_events) == 0
        assert len(chaos_orchestrator.event_history) == 0

    @pytest.mark.asyncio
    async def test_execute_chaos_event(self, chaos_orchestrator, sample_chaos_event):
        """Test executing a chaos event"""
        execution = await chaos_orchestrator.execute_chaos_event(sample_chaos_event)

        assert execution.event == sample_chaos_event
        assert execution.start_time is not None
        assert execution.end_time is not None
        assert isinstance(execution.success, bool)

    def test_create_random_chaos_scenario(self, chaos_orchestrator):
        """Test creating random chaos scenario"""
        events = chaos_orchestrator.create_random_chaos_scenario(
            duration_minutes=5,
            event_count=3
        )

        assert len(events) == 3
        for event in events:
            assert isinstance(event, ChaosEvent)
            assert event.event_type in ChaosEventType
            assert event.severity in ChaosEventSeverity

    def test_generate_random_parameters(self, chaos_orchestrator):
        """Test generating random parameters"""
        params = chaos_orchestrator._generate_random_parameters(
            ChaosEventType.CPU_STRESS,
            ChaosEventSeverity.HIGH
        )

        assert 'cpu_percent' in params
        assert params['cpu_percent'] > 50  # High severity should have high CPU usage

    def test_get_chaos_summary(self, chaos_orchestrator):
        """Test getting chaos summary"""
        # Initially no events
        summary = chaos_orchestrator.get_chaos_summary()
        assert "error" in summary

        # Add mock execution to history
        from src.testing.chaos_engineering import ChaosExecution
        mock_execution = ChaosExecution(
            event=ChaosEvent(
                event_type=ChaosEventType.NETWORK_DELAY,
                severity=ChaosEventSeverity.LOW,
                duration_seconds=1.0
            ),
            start_time=datetime.now(),
            end_time=datetime.now(),
            success=True
        )
        chaos_orchestrator.event_history.append(mock_execution)

        summary = chaos_orchestrator.get_chaos_summary()
        assert "summary" in summary
        assert summary["summary"]["total_events"] == 1


class TestIntegrationTester:
    """Test suite for IntegrationTester"""

    @pytest.fixture
    def integration_tester(self):
        """Create integration tester instance"""
        return IntegrationTester()

    @pytest.fixture
    def mock_test_function(self):
        """Mock test function"""
        async def mock_test():
            return {"status": "success", "data": "test_data"}
        return mock_test

    def test_integration_tester_initialization(self, integration_tester):
        """Test integration tester initialization"""
        assert len(integration_tester.test_cases) == 0
        assert len(integration_tester.test_results) == 0
        assert integration_tester.mock_manager is not None

    def test_add_test_case(self, integration_tester, mock_test_function):
        """Test adding test cases"""
        test_case = IntegrationTestCase(
            name="Test Case",
            test_type=IntegrationTestType.API_INTEGRATION,
            description="Test description",
            components=["api", "database"],
            test_function=mock_test_function
        )

        integration_tester.add_test_case(test_case)
        assert len(integration_tester.test_cases) == 1

    def test_create_api_integration_tests(self, integration_tester):
        """Test creating API integration tests"""
        integration_tester.create_api_integration_tests()

        api_tests = [tc for tc in integration_tester.test_cases
                    if tc.test_type == IntegrationTestType.API_INTEGRATION]
        assert len(api_tests) > 0

    @pytest.mark.asyncio
    async def test_run_test_case(self, integration_tester, mock_test_function):
        """Test running a single test case"""
        test_case = IntegrationTestCase(
            name="Test Case",
            test_type=IntegrationTestType.API_INTEGRATION,
            description="Test description",
            components=["api"],
            test_function=mock_test_function,
            expected_results={"status": "success"}
        )

        result = await integration_tester.run_test_case(test_case)

        assert result.test_case == test_case
        assert result.start_time is not None
        assert result.end_time is not None
        assert isinstance(result.success, bool)

    def test_validate_test_results(self, integration_tester):
        """Test test result validation"""
        actual_results = {
            "status": "success",
            "response_time": 0.5,
            "error_rate": 0.02
        }

        # Test passing validation
        expected_results = {
            "status": "success",
            "response_time_max": 1.0,
            "error_rate_max": 0.05
        }

        assert integration_tester._validate_test_results(actual_results, expected_results) == True

        # Test failing validation
        expected_results["response_time_max"] = 0.1  # Too strict
        assert integration_tester._validate_test_results(actual_results, expected_results) == False


class TestComponentMockManager:
    """Test suite for ComponentMockManager"""

    @pytest.fixture
    def mock_manager(self):
        """Create component mock manager"""
        return ComponentMockManager()

    def test_mock_component(self, mock_manager):
        """Test mocking components"""
        mock_object = MagicMock()
        mock_manager.mock_component("test_component", mock_object)

        assert "test_component" in mock_manager.mocked_components
        assert mock_manager.get_mock("test_component") == mock_object

    def test_simulate_component_failure(self, mock_manager):
        """Test simulating component failures"""
        mock_manager.simulate_component_failure("test_component", "timeout")

        behavior = mock_manager.component_behaviors["test_component"]
        with pytest.raises(TimeoutError):
            behavior()

    def test_reset_component(self, mock_manager):
        """Test resetting component behavior"""
        mock_manager.simulate_component_failure("test_component", "timeout")
        assert "test_component" in mock_manager.component_behaviors

        mock_manager.reset_component("test_component")
        assert "test_component" not in mock_manager.component_behaviors


class TestTestOrchestrator:
    """Test suite for TestOrchestrator"""

    @pytest.fixture
    def test_orchestrator(self):
        """Create test orchestrator instance"""
        return TestOrchestrator()

    def test_test_orchestrator_initialization(self, test_orchestrator):
        """Test test orchestrator initialization"""
        assert test_orchestrator.integration_tester is not None
        assert test_orchestrator.reliability_tester is not None
        assert test_orchestrator.chaos_orchestrator is not None
        assert len(test_orchestrator.test_history) == 0

    @pytest.mark.asyncio
    async def test_run_quick_test_suite(self, test_orchestrator):
        """Test running quick test suite"""
        results = await test_orchestrator.run_quick_test_suite()

        assert results.config.suite_type == TestSuiteType.QUICK
        assert results.start_time is not None
        assert results.end_time is not None
        assert isinstance(results.overall_success, bool)

    def test_generate_summary_stats(self, test_orchestrator):
        """Test generating summary statistics"""
        # Create mock results
        from src.testing.test_orchestrator import TestSuiteResults, TestSuiteConfig

        mock_results = TestSuiteResults(
            config=TestSuiteConfig(suite_type=TestSuiteType.QUICK),
            start_time=datetime.now(),
            end_time=datetime.now(),
            integration_results=[],
            reliability_results=[],
            chaos_results=[]
        )

        stats = test_orchestrator._generate_summary_stats(mock_results)

        assert "test_suite_type" in stats
        assert stats["test_suite_type"] == "quick"
        assert "total_duration_minutes" in stats

    def test_generate_recommendations(self, test_orchestrator):
        """Test generating recommendations"""
        from src.testing.test_orchestrator import TestSuiteResults, TestSuiteConfig

        mock_results = TestSuiteResults(
            config=TestSuiteConfig(suite_type=TestSuiteType.QUICK),
            start_time=datetime.now(),
            summary_stats={"overall_success_rate": 100.0}
        )

        recommendations = test_orchestrator._generate_recommendations(mock_results)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

    def test_evaluate_overall_success(self, test_orchestrator):
        """Test evaluating overall success"""
        from src.testing.test_orchestrator import TestSuiteResults, TestSuiteConfig

        # Test successful case
        mock_results = TestSuiteResults(
            config=TestSuiteConfig(suite_type=TestSuiteType.QUICK),
            start_time=datetime.now(),
            summary_stats={"overall_success_rate": 98.0}
        )

        assert test_orchestrator._evaluate_overall_success(mock_results) == True

        # Test failure case
        mock_results.summary_stats["overall_success_rate"] = 90.0
        assert test_orchestrator._evaluate_overall_success(mock_results) == False

    def test_get_test_history(self, test_orchestrator):
        """Test getting test history"""
        # Initially empty
        history = test_orchestrator.get_test_history(5)
        assert len(history) == 0

        # Add mock results to history
        from src.testing.test_orchestrator import TestSuiteResults, TestSuiteConfig

        mock_result = TestSuiteResults(
            config=TestSuiteConfig(suite_type=TestSuiteType.QUICK),
            start_time=datetime.now()
        )
        test_orchestrator.test_history.append(mock_result)

        history = test_orchestrator.get_test_history(5)
        assert len(history) == 1

    def test_get_test_trends(self, test_orchestrator):
        """Test getting test trends"""
        # Test with insufficient history
        trends = test_orchestrator.get_test_trends()
        assert "error" in trends

        # Add sufficient history
        from src.testing.test_orchestrator import TestSuiteResults, TestSuiteConfig

        for i in range(5):
            mock_result = TestSuiteResults(
                config=TestSuiteConfig(suite_type=TestSuiteType.QUICK),
                start_time=datetime.now(),
                total_duration=60.0 + i,  # Varying durations
                summary_stats={"overall_success_rate": 95.0 + i}  # Varying success rates
            )
            test_orchestrator.test_history.append(mock_result)

        trends = test_orchestrator.get_test_trends()
        assert "test_runs_analyzed" in trends
        assert "success_rate_trend" in trends
        assert "duration_trend" in trends


@pytest.mark.asyncio
async def test_comprehensive_integration():
    """Test comprehensive integration of all testing components"""
    orchestrator = TestOrchestrator()

    # Run a quick test to verify integration
    results = await orchestrator.run_quick_test_suite()

    # Verify all components worked together
    assert results.config.suite_type == TestSuiteType.QUICK
    assert results.overall_success is not None
    assert results.summary_stats is not None
    assert len(results.recommendations) > 0

    # Verify test history was updated
    assert len(orchestrator.test_history) == 1


if __name__ == '__main__':
    pytest.main([__file__])