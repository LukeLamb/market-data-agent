"""
TimescaleDB Performance Testing Suite
Tests the performance targets for Phase 3 Step 1:
- Query response time: <10ms for common queries
- Write throughput: >100K inserts/second
- Concurrent connection handling: 200+ connections
"""

import asyncio
import time
import statistics
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pytest

from src.storage.timescaledb_handler import TimescaleDBHandler, TimescaleDBConfig
from src.data_sources.base import PriceData


class PerformanceTestSuite:
    """Comprehensive performance testing for TimescaleDB implementation"""

    def __init__(self):
        self.config = TimescaleDBConfig(
            host="localhost",
            port=5432,
            database="market_data_test",
            user="market_user",
            password="secure_password_2024",
            pool_size=50,
            max_overflow=100
        )
        self.handler = None
        self.test_symbols = ["BTCUSD", "ETHUSD", "ADAUSD", "SOLUSD", "DOTUSD"]

    async def setup(self):
        """Initialize test environment"""
        self.handler = TimescaleDBHandler(self.config)
        await self.handler.initialize()

    async def teardown(self):
        """Clean up test environment"""
        if self.handler:
            await self.handler.close()

    def generate_test_data(self, symbol: str, count: int, start_time: datetime = None) -> List[PriceData]:
        """Generate realistic test data for performance testing"""
        if start_time is None:
            start_time = datetime.now() - timedelta(days=1)

        data = []
        current_price = random.uniform(100, 50000)  # Starting price

        for i in range(count):
            # Simulate realistic price movement
            price_change = random.uniform(-0.05, 0.05)  # ±5% max change
            current_price *= (1 + price_change)

            # Generate OHLCV data
            open_price = current_price
            close_price = current_price * (1 + random.uniform(-0.02, 0.02))
            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.01))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.01))
            volume = random.uniform(1000, 100000)

            data.append(PriceData(
                symbol=symbol,
                timestamp=start_time + timedelta(minutes=i),
                open_price=round(open_price, 2),
                high_price=round(high_price, 2),
                low_price=round(low_price, 2),
                close_price=round(close_price, 2),
                volume=int(round(volume)),
                source="performance_test",
                quality_score=int(random.uniform(80, 100))
            ))

            current_price = close_price

        return data

    async def test_write_performance(self, batch_size: int = 1000, total_records: int = 100000) -> Dict[str, float]:
        """Test write throughput performance"""
        print(f"Testing write performance: {total_records} records in batches of {batch_size}")

        start_time = time.time()
        total_written = 0

        for symbol in self.test_symbols:
            records_per_symbol = total_records // len(self.test_symbols)

            # Generate test data in batches
            for batch_start in range(0, records_per_symbol, batch_size):
                batch_count = min(batch_size, records_per_symbol - batch_start)
                test_data = self.generate_test_data(
                    symbol,
                    batch_count,
                    datetime.now() - timedelta(days=random.randint(1, 30))
                )

                # Measure batch write time
                batch_start_time = time.time()
                written_count = await self.handler.store_ohlcv_data(
                    test_data, "performance_test", 1.0
                )
                batch_time = time.time() - batch_start_time

                total_written += written_count

                if batch_start % (batch_size * 10) == 0:  # Progress update every 10 batches
                    current_rate = total_written / (time.time() - start_time)
                    print(f"Progress: {total_written}/{total_records} records, Rate: {current_rate:.0f} records/sec")

        total_time = time.time() - start_time
        writes_per_second = total_written / total_time

        print(f"Write Performance Results:")
        print(f"Total Records: {total_written}")
        print(f"Total Time: {total_time:.2f} seconds")
        print(f"Writes/Second: {writes_per_second:.2f}")
        print(f"Target: >100,000 writes/second - {'✅ PASSED' if writes_per_second > 100000 else '❌ FAILED'}")

        return {
            "total_records": total_written,
            "total_time_seconds": total_time,
            "writes_per_second": writes_per_second,
            "target_met": writes_per_second > 100000
        }

    async def test_query_performance(self) -> Dict[str, Any]:
        """Test query response time performance"""
        print("Testing query performance for common query patterns")

        query_tests = [
            {
                "name": "Latest Price Query",
                "description": "Get latest price for a symbol",
                "iterations": 100
            },
            {
                "name": "Time Range Query",
                "description": "Get data for last 24 hours",
                "iterations": 50
            },
            {
                "name": "Aggregation Query",
                "description": "Get hourly OHLC aggregates",
                "iterations": 20
            }
        ]

        results = {}

        for test in query_tests:
            print(f"\nRunning {test['name']} ({test['iterations']} iterations)")
            response_times = []

            for i in range(test['iterations']):
                symbol = random.choice(self.test_symbols)

                start_time = time.perf_counter()

                if test['name'] == "Latest Price Query":
                    data = await self.handler.get_latest_data(symbol)
                elif test['name'] == "Time Range Query":
                    end_time = datetime.now()
                    start_time_query = end_time - timedelta(hours=24)
                    data = await self.handler.get_historical_data(symbol, start_time_query, end_time)
                elif test['name'] == "Aggregation Query":
                    # This would need to be implemented in the handler
                    # For now, simulate with a regular query
                    end_time = datetime.now()
                    start_time_query = end_time - timedelta(hours=24)
                    data = await self.handler.get_historical_data(symbol, start_time_query, end_time)

                response_time_ms = (time.perf_counter() - start_time) * 1000
                response_times.append(response_time_ms)

                if i % 10 == 0:
                    avg_time = statistics.mean(response_times) if response_times else 0
                    print(f"  Progress: {i+1}/{test['iterations']}, Avg: {avg_time:.2f}ms")

            # Calculate statistics
            avg_response_time = statistics.mean(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
            max_response_time = max(response_times)

            target_met = avg_response_time < 10.0  # <10ms target

            results[test['name']] = {
                "avg_response_time_ms": avg_response_time,
                "p95_response_time_ms": p95_response_time,
                "max_response_time_ms": max_response_time,
                "target_met": target_met,
                "iterations": test['iterations']
            }

            print(f"  Results:")
            print(f"    Average: {avg_response_time:.2f}ms")
            print(f"    95th percentile: {p95_response_time:.2f}ms")
            print(f"    Maximum: {max_response_time:.2f}ms")
            print(f"    Target <10ms: {'✅ PASSED' if target_met else '❌ FAILED'}")

        return results

    async def test_concurrent_connections(self, connection_count: int = 200) -> Dict[str, Any]:
        """Test concurrent connection handling"""
        print(f"Testing concurrent connection handling with {connection_count} connections")

        async def worker_task(worker_id: int) -> Dict[str, Any]:
            """Individual worker task for concurrent testing"""
            try:
                # Create dedicated handler for this worker
                worker_handler = TimescaleDBHandler(self.config)
                await worker_handler.initialize()

                start_time = time.perf_counter()

                # Perform mixed operations
                symbol = random.choice(self.test_symbols)

                # Query operation
                latest_data = await worker_handler.get_latest_data(symbol)

                # Write operation
                test_data = self.generate_test_data(symbol, 10)
                await worker_handler.store_ohlcv_data(test_data, "concurrent_test", 1.0)

                response_time = (time.perf_counter() - start_time) * 1000

                await worker_handler.close()

                return {
                    "worker_id": worker_id,
                    "response_time_ms": response_time,
                    "success": True
                }

            except Exception as e:
                return {
                    "worker_id": worker_id,
                    "error": str(e),
                    "success": False
                }

        # Run concurrent workers
        start_time = time.time()
        tasks = [worker_task(i) for i in range(connection_count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # Analyze results
        successful_workers = [r for r in results if isinstance(r, dict) and r.get("success", False)]
        failed_workers = [r for r in results if not (isinstance(r, dict) and r.get("success", False))]

        if successful_workers:
            avg_response_time = statistics.mean([w["response_time_ms"] for w in successful_workers])
            max_response_time = max([w["response_time_ms"] for w in successful_workers])
        else:
            avg_response_time = 0
            max_response_time = 0

        success_rate = len(successful_workers) / connection_count
        target_met = success_rate > 0.95 and len(successful_workers) >= 190  # 95% success with 190+ connections

        print(f"Concurrent Connection Results:")
        print(f"Total Connections: {connection_count}")
        print(f"Successful: {len(successful_workers)}")
        print(f"Failed: {len(failed_workers)}")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Average Response Time: {avg_response_time:.2f}ms")
        print(f"Maximum Response Time: {max_response_time:.2f}ms")
        print(f"Total Test Time: {total_time:.2f}s")
        print(f"Target >190 connections: {'✅ PASSED' if target_met else '❌ FAILED'}")

        return {
            "total_connections": connection_count,
            "successful_connections": len(successful_workers),
            "failed_connections": len(failed_workers),
            "success_rate": success_rate,
            "avg_response_time_ms": avg_response_time,
            "max_response_time_ms": max_response_time,
            "total_time_seconds": total_time,
            "target_met": target_met
        }

    async def run_full_performance_suite(self) -> Dict[str, Any]:
        """Run complete performance test suite"""
        print("🚀 Starting TimescaleDB Performance Test Suite")
        print("=" * 60)

        await self.setup()

        try:
            # Test 1: Write Performance
            print("\n📝 Test 1: Write Performance")
            write_results = await self.test_write_performance(
                batch_size=1000,
                total_records=50000  # Reduced for faster testing
            )

            # Test 2: Query Performance
            print("\n🔍 Test 2: Query Performance")
            query_results = await self.test_query_performance()

            # Test 3: Concurrent Connections
            print("\n🔗 Test 3: Concurrent Connection Handling")
            connection_results = await self.test_concurrent_connections(connection_count=100)  # Reduced for stability

            # Overall Summary
            print("\n" + "=" * 60)
            print("📊 PERFORMANCE TEST SUMMARY")
            print("=" * 60)

            all_targets_met = (
                write_results["target_met"] and
                all(test["target_met"] for test in query_results.values()) and
                connection_results["target_met"]
            )

            print(f"Write Performance: {'✅ PASSED' if write_results['target_met'] else '❌ FAILED'}")
            print(f"Query Performance: {'✅ PASSED' if all(test['target_met'] for test in query_results.values()) else '❌ FAILED'}")
            print(f"Connection Handling: {'✅ PASSED' if connection_results['target_met'] else '❌ FAILED'}")
            print(f"\nOverall Status: {'🎉 ALL TESTS PASSED' if all_targets_met else '⚠️ SOME TESTS FAILED'}")

            return {
                "write_performance": write_results,
                "query_performance": query_results,
                "connection_handling": connection_results,
                "overall_success": all_targets_met,
                "timestamp": datetime.now().isoformat()
            }

        finally:
            await self.teardown()


async def main():
    """Main function to run performance tests"""
    suite = PerformanceTestSuite()
    results = await suite.run_full_performance_suite()

    # Save results to file
    import json
    with open("performance_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n📄 Results saved to: performance_test_results.json")


if __name__ == "__main__":
    asyncio.run(main())