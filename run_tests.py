#!/usr/bin/env python3
"""
Test Runner for Market Data Agent

This script provides various options for running tests with different
configurations and reporting options.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors"""
    if description:
        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Market Data Agent tests")

    # Test selection options
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--coverage", action="store_true", help="Run tests with coverage")
    parser.add_argument("--slow", action="store_true", help="Include slow tests")
    parser.add_argument("--network", action="store_true", help="Include network tests")
    parser.add_argument("--api-key", action="store_true", help="Include tests requiring API keys")

    # Output options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet output")
    parser.add_argument("--html-report", help="Generate HTML coverage report")
    parser.add_argument("--xml-report", help="Generate XML test report")

    # Specific test options
    parser.add_argument("--file", help="Run tests from specific file")
    parser.add_argument("--test", help="Run specific test function")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")

    args = parser.parse_args()

    # Build pytest command
    cmd = ["python", "-m", "pytest"]

    # Add test paths
    if args.file:
        cmd.append(args.file)
    else:
        cmd.append("tests/")

    # Add specific test
    if args.test:
        if args.file:
            cmd[-1] += f"::{args.test}"
        else:
            cmd.append(f"-k {args.test}")

    # Add markers based on test type selection
    markers = []

    if args.unit:
        markers.append("not integration and not slow and not network and not api_key")
    elif args.integration:
        markers.append("integration")

    if not args.slow:
        if markers:
            markers[-1] += " and not slow"
        else:
            markers.append("not slow")

    if not args.network:
        if markers:
            markers[-1] += " and not network"
        else:
            markers.append("not network")

    if not args.api_key:
        if markers:
            markers[-1] += " and not api_key"
        else:
            markers.append("not api_key")

    if markers:
        cmd.extend(["-m", " and ".join(markers)])

    # Add output options
    if args.verbose:
        cmd.append("-v")
    elif args.quiet:
        cmd.append("-q")

    # Add coverage options
    if args.coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing"])
        if args.html_report:
            cmd.extend([f"--cov-report=html:{args.html_report}"])
        else:
            cmd.append("--cov-report=html")

    # Add XML report
    if args.xml_report:
        cmd.extend([f"--junit-xml={args.xml_report}"])

    # Add parallel execution
    if args.parallel:
        cmd.extend(["-n", "auto"])

    # Additional options for better output
    cmd.extend([
        "--tb=short",
        "--maxfail=10",
        "--disable-warnings"
    ])

    # Set environment variables for testing
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())

    print("Market Data Agent Test Runner")
    print("=" * 40)

    # Check if pytest is available
    try:
        subprocess.run(["python", "-m", "pytest", "--version"],
                      check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("Error: pytest is not installed. Please install it with:")
        print("pip install pytest pytest-asyncio")
        return 1

    # Run the tests
    print(f"Running command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, env=env)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTest run interrupted by user")
        return 1


def run_quick_smoke_test():
    """Run a quick smoke test to verify basic functionality"""
    print("Running quick smoke test...")

    tests_to_run = [
        "tests/test_base_data_source.py::TestDataModels::test_price_data_model",
        "tests/test_config_manager.py::TestConfigManager::test_default_config_creation",
        "tests/test_error_handling.py::TestErrorHandler::test_error_handler_initialization",
    ]

    for test in tests_to_run:
        cmd = ["python", "-m", "pytest", test, "-v", "--tb=short"]
        if not run_command(cmd, f"Smoke test: {test.split('::')[-1]}"):
            print(f"Smoke test failed: {test}")
            return False

    print("\n✓ All smoke tests passed!")
    return True


def run_full_test_suite():
    """Run the complete test suite with coverage"""
    print("Running full test suite with coverage...")

    cmd = [
        "python", "-m", "pytest",
        "tests/",
        "-v",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--tb=short",
        "--maxfail=5"
    ]

    return run_command(cmd, "Full test suite with coverage")


if __name__ == "__main__":
    # If no arguments provided, show help and run smoke test
    if len(sys.argv) == 1:
        print("No arguments provided. Running smoke test...")
        if run_quick_smoke_test():
            print("\n" + "="*60)
            print("For more testing options, run: python run_tests.py --help")
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        sys.exit(main())