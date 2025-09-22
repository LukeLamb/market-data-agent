"""
Comprehensive Testing Framework

Advanced testing suite for Market Data Agent including reliability testing,
chaos engineering, integration testing, and system validation.

Key Components:
- ReliabilityTester: Stress testing, endurance testing, fault tolerance
- ChaosOrchestrator: Chaos engineering with fault injection
- IntegrationTester: End-to-end integration testing across components
- TestOrchestrator: Unified testing framework and reporting

Usage:
    from testing import TestOrchestrator

    orchestrator = TestOrchestrator()
    results = await orchestrator.run_comprehensive_test_suite()
"""

from .reliability_tester import (
    ReliabilityTester,
    TestScenario,
    TestSeverity,
    TestResult,
    ReliabilityMetrics,
    TestExecution,
    FaultInjector
)

from .chaos_engineering import (
    ChaosOrchestrator,
    ChaosEvent,
    ChaosEventType,
    ChaosEventSeverity,
    ChaosExecution,
    NetworkChaos,
    ResourceChaos
)

from .integration_tester import (
    IntegrationTester,
    IntegrationTestType,
    IntegrationTestCase,
    IntegrationTestResult,
    ComponentMockManager
)

__version__ = "1.0.0"
__author__ = "Market Data Agent Team"

__all__ = [
    # Reliability Testing
    'ReliabilityTester',
    'TestScenario',
    'TestSeverity',
    'TestResult',
    'ReliabilityMetrics',
    'TestExecution',
    'FaultInjector',

    # Chaos Engineering
    'ChaosOrchestrator',
    'ChaosEvent',
    'ChaosEventType',
    'ChaosEventSeverity',
    'ChaosExecution',
    'NetworkChaos',
    'ResourceChaos',

    # Integration Testing
    'IntegrationTester',
    'IntegrationTestType',
    'IntegrationTestCase',
    'IntegrationTestResult',
    'ComponentMockManager'
]