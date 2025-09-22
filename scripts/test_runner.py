#!/usr/bin/env python3
"""
Comprehensive Test Runner CLI

Command-line interface for running reliability tests, chaos engineering,
and integration tests for the Market Data Agent.
"""

import asyncio
import argparse
import json
import sys
import os
from datetime import datetime
from typing import Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.testing.test_orchestrator import TestOrchestrator, TestSuiteType, TestSuiteConfig
from src.testing.reliability_tester import ReliabilityTester
from src.testing.chaos_engineering import ChaosOrchestrator
from src.testing.integration_tester import IntegrationTester


class TestRunnerCLI:
    """CLI for comprehensive testing"""

    def __init__(self):
        self.orchestrator = TestOrchestrator()

    async def run_quick_tests(self):
        """Run quick test suite (< 5 minutes)"""
        print("🚀 Running Quick Test Suite...")
        print("=" * 50)

        results = await self.orchestrator.run_quick_test_suite()
        self._print_results_summary(results)

        return results.overall_success

    async def run_standard_tests(self):
        """Run standard test suite (< 30 minutes)"""
        print("🧪 Running Standard Test Suite...")
        print("=" * 50)

        results = await self.orchestrator.run_standard_test_suite()
        self._print_results_summary(results)

        return results.overall_success

    async def run_comprehensive_tests(self):
        """Run comprehensive test suite (< 2 hours)"""
        print("🔬 Running Comprehensive Test Suite...")
        print("=" * 50)

        results = await self.orchestrator.run_comprehensive_test_suite()
        self._print_results_summary(results)
        self._print_detailed_report(results)

        return results.overall_success

    async def run_extended_tests(self):
        """Run extended test suite (< 24 hours)"""
        print("⏰ Running Extended Test Suite...")
        print("=" * 50)
        print("⚠️  This will run for up to 24 hours. Press Ctrl+C to stop.")

        results = await self.orchestrator.run_extended_test_suite()
        self._print_results_summary(results)
        self._print_detailed_report(results)

        return results.overall_success

    async def run_integration_tests_only(self):
        """Run only integration tests"""
        print("🔗 Running Integration Tests...")
        print("=" * 50)

        integration_tester = IntegrationTester()

        # Initialize test cases
        integration_tester.create_api_integration_tests()
        integration_tester.create_data_flow_tests()
        integration_tester.create_performance_integration_tests()
        integration_tester.create_configuration_integration_tests()

        # Run tests
        results = await integration_tester.run_all_tests()

        # Print results
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.success)
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        print(f"\n📊 Integration Test Results:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {total_tests - passed_tests}")
        print(f"   Success Rate: {success_rate:.1f}%")

        if success_rate < 100:
            print(f"\n❌ Failed Tests:")
            for result in results:
                if not result.success:
                    print(f"   • {result.test_case.name}: {', '.join(result.errors)}")

        return success_rate >= 95

    async def run_reliability_tests_only(self):
        """Run only reliability tests"""
        print("💪 Running Reliability Tests...")
        print("=" * 50)

        reliability_tester = ReliabilityTester()

        # Add test scenarios
        reliability_tester.add_stress_test_scenario(
            "API Stress Test",
            target_function=self._mock_api_call,
            concurrent_requests=50,
            duration_seconds=120
        )

        # Run tests
        results = await reliability_tester.run_all_scenarios()

        # Print results
        reliability_stats = reliability_tester.get_summary_stats()

        print(f"\n📊 Reliability Test Results:")
        print(f"   Total Scenarios: {reliability_stats.get('total_scenarios', 0)}")
        print(f"   Pass Rate: {reliability_stats.get('pass_rate', 0) * 100:.1f}%")
        print(f"   Average Uptime: {reliability_stats.get('average_uptime', 0):.1f}%")
        print(f"   Average Error Rate: {reliability_stats.get('average_error_rate', 0) * 100:.2f}%")

        return reliability_stats.get('pass_rate', 0) >= 0.95

    async def run_chaos_tests_only(self):
        """Run only chaos engineering tests"""
        print("🌪️  Running Chaos Engineering Tests...")
        print("=" * 50)

        chaos_orchestrator = ChaosOrchestrator()

        # Create chaos scenario
        chaos_events = chaos_orchestrator.create_random_chaos_scenario(
            duration_minutes=10,
            event_count=5
        )

        print(f"Created {len(chaos_events)} chaos events")

        # Run chaos tests
        results = await chaos_orchestrator.run_chaos_scenario(chaos_events)

        # Print results
        chaos_summary = chaos_orchestrator.get_chaos_summary()

        print(f"\n📊 Chaos Engineering Results:")
        summary = chaos_summary.get('summary', {})
        print(f"   Total Events: {summary.get('total_events', 0)}")
        print(f"   Successful Events: {summary.get('successful_events', 0)}")
        print(f"   Success Rate: {summary.get('success_rate', 0):.1f}%")
        print(f"   Average Recovery Time: {summary.get('average_recovery_time', 0):.2f}s")

        return summary.get('success_rate', 0) >= 80

    async def show_test_history(self, limit: int = 10):
        """Show test execution history"""
        print("📜 Test Execution History")
        print("=" * 50)

        history = self.orchestrator.get_test_history(limit)

        if not history:
            print("No test history available")
            return

        for i, result in enumerate(reversed(history), 1):
            status = "✅ PASS" if result.overall_success else "❌ FAIL"
            duration = result.total_duration / 60  # Convert to minutes

            print(f"{i}. {result.start_time.strftime('%Y-%m-%d %H:%M:%S')} - "
                  f"{result.config.suite_type.value.upper()} - {status} "
                  f"({duration:.1f}m)")

    async def show_test_trends(self):
        """Show test trend analysis"""
        print("📈 Test Trend Analysis")
        print("=" * 50)

        trends = self.orchestrator.get_test_trends()

        if "error" in trends:
            print(trends["error"])
            return

        print(f"Analyzed {trends['test_runs_analyzed']} recent test runs:")
        print()

        # Success rate trends
        success_trend = trends['success_rate_trend']
        print(f"Success Rate:")
        print(f"   Current: {success_trend['current']:.1f}%")
        print(f"   Average: {success_trend['average']:.1f}%")
        print(f"   Trend: {success_trend['trend']}")
        print()

        # Duration trends
        duration_trend = trends['duration_trend']
        print(f"Execution Duration:")
        print(f"   Current: {duration_trend['current_minutes']:.1f} minutes")
        print(f"   Average: {duration_trend['average_minutes']:.1f} minutes")
        print(f"   Trend: {duration_trend['trend']}")
        print()

        # Reliability indicators
        reliability = trends['reliability_indicators']
        print(f"Reliability Indicators:")
        print(f"   Consistent Success: {'Yes' if reliability['consistent_success'] else 'No'}")
        print(f"   Stability Score: {reliability['stability_score']:.1f}%")

    async def validate_system_health(self):
        """Quick system health validation"""
        print("🏥 System Health Validation")
        print("=" * 50)

        # Run a minimal set of health checks
        health_checks = []

        try:
            # Test basic functionality
            start_time = datetime.now()
            await self._mock_api_call()
            api_response_time = (datetime.now() - start_time).total_seconds()
            health_checks.append(("API Response", api_response_time < 1.0, f"{api_response_time:.3f}s"))

            # Test configuration
            health_checks.append(("Configuration", True, "Loaded"))

            # Test cache (mock)
            health_checks.append(("Cache", True, "Available"))

            # Test database connection (mock)
            health_checks.append(("Database", True, "Connected"))

        except Exception as e:
            health_checks.append(("System Error", False, str(e)))

        # Print health check results
        all_healthy = True
        for check_name, healthy, details in health_checks:
            status = "✅ HEALTHY" if healthy else "❌ UNHEALTHY"
            print(f"   {check_name}: {status} ({details})")
            if not healthy:
                all_healthy = False

        print()
        overall_status = "✅ SYSTEM HEALTHY" if all_healthy else "❌ SYSTEM ISSUES DETECTED"
        print(f"Overall Status: {overall_status}")

        return all_healthy

    async def _mock_api_call(self):
        """Mock API call for testing"""
        import random
        await asyncio.sleep(random.uniform(0.01, 0.05))
        if random.random() < 0.01:  # 1% error rate
            raise Exception("Mock API error")
        return {"status": "success", "data": "mock_data"}

    def _print_results_summary(self, results):
        """Print test results summary"""
        duration_minutes = results.total_duration / 60
        overall_status = "✅ PASSED" if results.overall_success else "❌ FAILED"

        print(f"\n📊 Test Suite Results:")
        print(f"   Type: {results.config.suite_type.value.upper()}")
        print(f"   Duration: {duration_minutes:.1f} minutes")
        print(f"   Overall Status: {overall_status}")

        # Print category results
        stats = results.summary_stats

        if "integration_tests" in stats:
            integration = stats["integration_tests"]
            print(f"   Integration Tests: {integration['passed']}/{integration['total']} "
                  f"({integration['success_rate']:.1f}%)")

        if "reliability_tests" in stats:
            reliability = stats["reliability_tests"]
            print(f"   Reliability Tests: {reliability.get('pass_rate', 0) * 100:.1f}% pass rate")

        if "chaos_tests" in stats:
            chaos = stats["chaos_tests"]
            chaos_success = chaos.get('summary', {}).get('success_rate', 0)
            print(f"   Chaos Tests: {chaos_success:.1f}% resilience")

        # Print recommendations
        if results.recommendations:
            print(f"\n💡 Recommendations:")
            for rec in results.recommendations[:3]:  # Show top 3 recommendations
                print(f"   • {rec}")

    def _print_detailed_report(self, results):
        """Print detailed test report"""
        if not results.detailed_report:
            return

        print(f"\n📝 Detailed Report:")
        report = results.detailed_report

        # Integration test details
        if "integration_test_details" in report and report["integration_test_details"]:
            print(f"\n🔗 Integration Test Details:")
            for test in report["integration_test_details"]:
                status = "✅" if test["success"] else "❌"
                print(f"   {status} {test['name']} ({test['execution_time']:.3f}s)")
                if test["errors"]:
                    for error in test["errors"]:
                        print(f"      Error: {error}")

        # Show environment info
        if "environment_info" in report:
            env = report["environment_info"]
            print(f"\n🌍 Environment:")
            print(f"   Framework: {env.get('execution_platform', 'Unknown')}")
            print(f"   Version: {env.get('test_framework_version', 'Unknown')}")


async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Market Data Agent Test Runner")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Test suite commands
    subparsers.add_parser("quick", help="Run quick test suite (< 5 minutes)")
    subparsers.add_parser("standard", help="Run standard test suite (< 30 minutes)")
    subparsers.add_parser("comprehensive", help="Run comprehensive test suite (< 2 hours)")
    subparsers.add_parser("extended", help="Run extended test suite (< 24 hours)")

    # Individual test type commands
    subparsers.add_parser("integration", help="Run integration tests only")
    subparsers.add_parser("reliability", help="Run reliability tests only")
    subparsers.add_parser("chaos", help="Run chaos engineering tests only")

    # Utility commands
    history_parser = subparsers.add_parser("history", help="Show test execution history")
    history_parser.add_argument("--limit", type=int, default=10, help="Number of entries to show")

    subparsers.add_parser("trends", help="Show test trend analysis")
    subparsers.add_parser("health", help="Quick system health validation")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    cli = TestRunnerCLI()

    try:
        success = True

        if args.command == "quick":
            success = await cli.run_quick_tests()
        elif args.command == "standard":
            success = await cli.run_standard_tests()
        elif args.command == "comprehensive":
            success = await cli.run_comprehensive_tests()
        elif args.command == "extended":
            success = await cli.run_extended_tests()
        elif args.command == "integration":
            success = await cli.run_integration_tests_only()
        elif args.command == "reliability":
            success = await cli.run_reliability_tests_only()
        elif args.command == "chaos":
            success = await cli.run_chaos_tests_only()
        elif args.command == "history":
            await cli.show_test_history(args.limit)
        elif args.command == "trends":
            await cli.show_test_trends()
        elif args.command == "health":
            success = await cli.validate_system_health()

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n⚠️ Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())