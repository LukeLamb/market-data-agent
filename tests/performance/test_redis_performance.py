"""
Redis Cache Performance Testing Suite
Validates sub-millisecond response times and high-throughput operations
"""

import asyncio
import time
import statistics
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pytest

from src.caching.redis_cache_manager import RedisCacheManager, CacheConfig
from src.storage.hybrid_storage_service import HybridStorageService, HybridStorageConfig
from src.data_sources.base import PriceData


class CachePerformanceTestSuite:
    """Comprehensive performance testing for Redis caching layer"""

    def __init__(self):
        self.cache_config = CacheConfig(
            host="localhost",
            port=6379,
            db=14,  # Use test database
            pool_max_connections=100,
            latest_price_ttl=60,
            historical_data_ttl=300
        )
        self.cache_manager = None
        self.test_symbols = ["BTCUSD", "ETHUSD", "ADAUSD", "SOLUSD", "DOTUSD", "AVAXUSD", "MATICUSD", "LINKUSD"]

    async def setup(self):
        """Initialize test environment"""
        try:
            self.cache_manager = RedisCacheManager(self.cache_config)
            await self.cache_manager.initialize()

            # Clear test database
            await self.cache_manager.clear_cache()
            print("Cache performance test environment initialized")

        except Exception as e:
            print(f"Setup failed: {e}")
            raise

    async def teardown(self):
        """Clean up test environment"""
        if self.cache_manager:
            await self.cache_manager.clear_cache()
            await self.cache_manager.close()

    def generate_test_price_data(self, symbol: str, count: int = 1) -> List[PriceData]:
        """Generate realistic test price data"""
        data = []
        base_price = random.uniform(100, 50000)

        for i in range(count):
            timestamp = datetime.now() - timedelta(seconds=i)
            price_change = random.uniform(-0.02, 0.02)  # ±2% change
            current_price = base_price * (1 + price_change)

            data.append(PriceData(
                symbol=symbol,
                timestamp=timestamp,
                open_price=round(current_price * 0.999, 2),
                high_price=round(current_price * 1.005, 2),
                low_price=round(current_price * 0.995, 2),
                close_price=round(current_price, 2),
                volume=random.randint(10000, 1000000),
                source="perf_test",
                quality_score=random.randint(85, 100)
            ))

        return data

    async def test_cache_write_performance(self, operations: int = 10000) -> Dict[str, float]:
        """Test cache write performance"""
        print(f"Testing cache write performance: {operations} operations")

        # Generate test data
        test_data = []
        for i in range(operations):
            symbol = random.choice(self.test_symbols)
            price_data = self.generate_test_price_data(symbol)[0]
            test_data.append(price_data)

        # Measure single write operations
        start_time = time.perf_counter()

        for data in test_data:
            await self.cache_manager.cache_latest_price(data)

        total_time = time.perf_counter() - start_time
        writes_per_second = operations / total_time
        avg_write_time_ms = (total_time / operations) * 1000

        print(f"Single Write Results:")
        print(f"  Total Operations: {operations}")
        print(f"  Total Time: {total_time:.3f} seconds")
        print(f"  Writes/Second: {writes_per_second:.0f}")
        print(f"  Average Write Time: {avg_write_time_ms:.3f} ms")
        print(f"  Target <1ms: {'✅ PASSED' if avg_write_time_ms < 1.0 else '❌ FAILED'}")

        return {
            "total_operations": operations,
            "total_time_seconds": total_time,
            "writes_per_second": writes_per_second,
            "avg_write_time_ms": avg_write_time_ms,
            "target_met": avg_write_time_ms < 1.0
        }

    async def test_cache_bulk_write_performance(self, batch_size: int = 1000, batches: int = 10) -> Dict[str, float]:
        """Test bulk cache write performance"""
        print(f"Testing bulk cache write performance: {batches} batches of {batch_size}")

        total_operations = batch_size * batches
        batch_times = []

        for batch in range(batches):
            # Generate batch data
            batch_data = []
            for i in range(batch_size):
                symbol = random.choice(self.test_symbols)
                price_data = self.generate_test_price_data(symbol)[0]
                batch_data.append(price_data)

            # Measure batch write time
            start_time = time.perf_counter()
            result = await self.cache_manager.bulk_cache_prices(batch_data)
            batch_time = time.perf_counter() - start_time
            batch_times.append(batch_time)

            if batch % 2 == 0:
                print(f"  Batch {batch+1}/{batches}: {batch_time:.3f}s ({result} cached)")

        total_time = sum(batch_times)
        avg_batch_time = statistics.mean(batch_times)
        writes_per_second = total_operations / total_time
        avg_write_time_ms = (total_time / total_operations) * 1000

        print(f"Bulk Write Results:")
        print(f"  Total Operations: {total_operations}")
        print(f"  Total Time: {total_time:.3f} seconds")
        print(f"  Average Batch Time: {avg_batch_time:.3f} seconds")
        print(f"  Writes/Second: {writes_per_second:.0f}")
        print(f"  Average Write Time: {avg_write_time_ms:.3f} ms")
        print(f"  Target >50,000 writes/sec: {'✅ PASSED' if writes_per_second > 50000 else '❌ FAILED'}")

        return {
            "total_operations": total_operations,
            "total_time_seconds": total_time,
            "avg_batch_time_seconds": avg_batch_time,
            "writes_per_second": writes_per_second,
            "avg_write_time_ms": avg_write_time_ms,
            "target_met": writes_per_second > 50000
        }

    async def test_cache_read_performance(self, operations: int = 10000) -> Dict[str, float]:
        """Test cache read performance"""
        print(f"Testing cache read performance: {operations} operations")

        # Pre-populate cache with test data
        print("  Pre-populating cache...")
        for symbol in self.test_symbols:
            price_data = self.generate_test_price_data(symbol)[0]
            await self.cache_manager.cache_latest_price(price_data)

        # Measure read performance
        read_times = []
        cache_hits = 0

        start_time = time.perf_counter()

        for i in range(operations):
            symbol = random.choice(self.test_symbols)

            read_start = time.perf_counter()
            result = await self.cache_manager.get_latest_price(symbol)
            read_time = (time.perf_counter() - read_start) * 1000  # Convert to ms

            read_times.append(read_time)
            if result:
                cache_hits += 1

            if i % 1000 == 0 and i > 0:
                avg_time = statistics.mean(read_times[-1000:])
                print(f"  Progress: {i}/{operations}, Avg: {avg_time:.3f}ms")

        total_time = time.perf_counter() - start_time
        reads_per_second = operations / total_time
        avg_read_time_ms = statistics.mean(read_times)
        p95_read_time_ms = statistics.quantiles(read_times, n=20)[18]  # 95th percentile
        p99_read_time_ms = statistics.quantiles(read_times, n=100)[98]  # 99th percentile
        max_read_time_ms = max(read_times)
        hit_rate = cache_hits / operations

        print(f"Cache Read Results:")
        print(f"  Total Operations: {operations}")
        print(f"  Cache Hit Rate: {hit_rate:.2%}")
        print(f"  Reads/Second: {reads_per_second:.0f}")
        print(f"  Average Read Time: {avg_read_time_ms:.3f} ms")
        print(f"  95th Percentile: {p95_read_time_ms:.3f} ms")
        print(f"  99th Percentile: {p99_read_time_ms:.3f} ms")
        print(f"  Maximum Read Time: {max_read_time_ms:.3f} ms")
        print(f"  Target <1ms average: {'✅ PASSED' if avg_read_time_ms < 1.0 else '❌ FAILED'}")
        print(f"  Target <2ms P95: {'✅ PASSED' if p95_read_time_ms < 2.0 else '❌ FAILED'}")

        return {
            "total_operations": operations,
            "cache_hit_rate": hit_rate,
            "reads_per_second": reads_per_second,
            "avg_read_time_ms": avg_read_time_ms,
            "p95_read_time_ms": p95_read_time_ms,
            "p99_read_time_ms": p99_read_time_ms,
            "max_read_time_ms": max_read_time_ms,
            "target_avg_met": avg_read_time_ms < 1.0,
            "target_p95_met": p95_read_time_ms < 2.0
        }

    async def test_concurrent_cache_operations(self, concurrency: int = 100, operations_per_worker: int = 100) -> Dict[str, float]:
        """Test concurrent cache operations"""
        print(f"Testing concurrent cache operations: {concurrency} workers, {operations_per_worker} ops each")

        async def worker_task(worker_id: int) -> Dict[str, Any]:
            """Individual worker for concurrent testing"""
            worker_times = []
            cache_hits = 0

            try:
                for i in range(operations_per_worker):
                    symbol = random.choice(self.test_symbols)

                    # Mix of read and write operations
                    if i % 3 == 0:  # Write operation (33%)
                        price_data = self.generate_test_price_data(symbol)[0]
                        start_time = time.perf_counter()
                        await self.cache_manager.cache_latest_price(price_data)
                        op_time = (time.perf_counter() - start_time) * 1000
                    else:  # Read operation (67%)
                        start_time = time.perf_counter()
                        result = await self.cache_manager.get_latest_price(symbol)
                        op_time = (time.perf_counter() - start_time) * 1000
                        if result:
                            cache_hits += 1

                    worker_times.append(op_time)

                return {
                    "worker_id": worker_id,
                    "avg_time_ms": statistics.mean(worker_times),
                    "max_time_ms": max(worker_times),
                    "cache_hits": cache_hits,
                    "success": True
                }

            except Exception as e:
                return {
                    "worker_id": worker_id,
                    "error": str(e),
                    "success": False
                }

        # Run concurrent workers
        start_time = time.perf_counter()
        tasks = [worker_task(i) for i in range(concurrency)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.perf_counter() - start_time

        # Analyze results
        successful_workers = [r for r in results if isinstance(r, dict) and r.get("success", False)]
        failed_workers = len(results) - len(successful_workers)

        if successful_workers:
            avg_worker_time = statistics.mean([w["avg_time_ms"] for w in successful_workers])
            max_worker_time = max([w["max_time_ms"] for w in successful_workers])
            total_cache_hits = sum([w["cache_hits"] for w in successful_workers])
        else:
            avg_worker_time = 0
            max_worker_time = 0
            total_cache_hits = 0

        total_operations = len(successful_workers) * operations_per_worker
        ops_per_second = total_operations / total_time if total_time > 0 else 0
        success_rate = len(successful_workers) / concurrency

        print(f"Concurrent Operations Results:")
        print(f"  Total Workers: {concurrency}")
        print(f"  Successful Workers: {len(successful_workers)}")
        print(f"  Failed Workers: {failed_workers}")
        print(f"  Success Rate: {success_rate:.2%}")
        print(f"  Total Operations: {total_operations}")
        print(f"  Operations/Second: {ops_per_second:.0f}")
        print(f"  Average Worker Time: {avg_worker_time:.3f} ms")
        print(f"  Maximum Worker Time: {max_worker_time:.3f} ms")
        print(f"  Total Test Time: {total_time:.3f} seconds")
        print(f"  Target >95% success: {'✅ PASSED' if success_rate > 0.95 else '❌ FAILED'}")
        print(f"  Target <5ms avg time: {'✅ PASSED' if avg_worker_time < 5.0 else '❌ FAILED'}")

        return {
            "total_workers": concurrency,
            "successful_workers": len(successful_workers),
            "failed_workers": failed_workers,
            "success_rate": success_rate,
            "total_operations": total_operations,
            "ops_per_second": ops_per_second,
            "avg_worker_time_ms": avg_worker_time,
            "max_worker_time_ms": max_worker_time,
            "total_time_seconds": total_time,
            "target_success_met": success_rate > 0.95,
            "target_time_met": avg_worker_time < 5.0
        }

    async def test_cache_memory_efficiency(self) -> Dict[str, Any]:
        """Test cache memory usage and efficiency"""
        print("Testing cache memory efficiency")

        # Get initial memory usage
        initial_stats = await self.cache_manager.get_cache_statistics()
        initial_memory_mb = initial_stats.cache_size_mb

        # Load test data
        data_points = 10000
        print(f"  Loading {data_points} data points...")

        for i in range(data_points):
            symbol = f"TEST{i % 100}"  # 100 unique symbols
            price_data = self.generate_test_price_data(symbol)[0]
            await self.cache_manager.cache_latest_price(price_data)

            if i % 1000 == 0 and i > 0:
                print(f"    Progress: {i}/{data_points}")

        # Get final memory usage
        final_stats = await self.cache_manager.get_cache_statistics()
        final_memory_mb = final_stats.cache_size_mb
        memory_used_mb = final_memory_mb - initial_memory_mb

        # Calculate efficiency metrics
        avg_bytes_per_record = (memory_used_mb * 1024 * 1024) / data_points
        compression_efficiency = avg_bytes_per_record < 1000  # Less than 1KB per record

        print(f"Memory Efficiency Results:")
        print(f"  Initial Memory: {initial_memory_mb:.2f} MB")
        print(f"  Final Memory: {final_memory_mb:.2f} MB")
        print(f"  Memory Used: {memory_used_mb:.2f} MB")
        print(f"  Records Cached: {data_points}")
        print(f"  Bytes per Record: {avg_bytes_per_record:.1f}")
        print(f"  Hit Rate: {final_stats.hit_rate:.2%}")
        print(f"  Evictions: {final_stats.evictions}")
        print(f"  Target <1KB/record: {'✅ PASSED' if compression_efficiency else '❌ FAILED'}")

        return {
            "initial_memory_mb": initial_memory_mb,
            "final_memory_mb": final_memory_mb,
            "memory_used_mb": memory_used_mb,
            "records_cached": data_points,
            "bytes_per_record": avg_bytes_per_record,
            "hit_rate": final_stats.hit_rate,
            "evictions": final_stats.evictions,
            "compression_efficient": compression_efficiency
        }

    async def run_full_performance_suite(self) -> Dict[str, Any]:
        """Run complete cache performance test suite"""
        print("🚀 Starting Redis Cache Performance Test Suite")
        print("=" * 60)

        await self.setup()

        try:
            # Test 1: Write Performance
            print("\n📝 Test 1: Cache Write Performance")
            write_results = await self.test_cache_write_performance(operations=5000)

            # Test 2: Bulk Write Performance
            print("\n📦 Test 2: Bulk Cache Write Performance")
            bulk_write_results = await self.test_cache_bulk_write_performance(batch_size=500, batches=10)

            # Test 3: Read Performance
            print("\n🔍 Test 3: Cache Read Performance")
            read_results = await self.test_cache_read_performance(operations=10000)

            # Test 4: Concurrent Operations
            print("\n🔗 Test 4: Concurrent Cache Operations")
            concurrent_results = await self.test_concurrent_cache_operations(concurrency=50, operations_per_worker=100)

            # Test 5: Memory Efficiency
            print("\n💾 Test 5: Cache Memory Efficiency")
            memory_results = await self.test_cache_memory_efficiency()

            # Overall Summary
            print("\n" + "=" * 60)
            print("📊 CACHE PERFORMANCE TEST SUMMARY")
            print("=" * 60)

            all_targets_met = (
                write_results["target_met"] and
                bulk_write_results["target_met"] and
                read_results["target_avg_met"] and
                read_results["target_p95_met"] and
                concurrent_results["target_success_met"] and
                concurrent_results["target_time_met"] and
                memory_results["compression_efficient"]
            )

            print(f"Write Performance: {'✅ PASSED' if write_results['target_met'] else '❌ FAILED'}")
            print(f"Bulk Write Performance: {'✅ PASSED' if bulk_write_results['target_met'] else '❌ FAILED'}")
            print(f"Read Performance: {'✅ PASSED' if read_results['target_avg_met'] and read_results['target_p95_met'] else '❌ FAILED'}")
            print(f"Concurrent Operations: {'✅ PASSED' if concurrent_results['target_success_met'] and concurrent_results['target_time_met'] else '❌ FAILED'}")
            print(f"Memory Efficiency: {'✅ PASSED' if memory_results['compression_efficient'] else '❌ FAILED'}")
            print(f"\nOverall Status: {'🎉 ALL TESTS PASSED' if all_targets_met else '⚠️ SOME TESTS FAILED'}")

            return {
                "write_performance": write_results,
                "bulk_write_performance": bulk_write_results,
                "read_performance": read_results,
                "concurrent_operations": concurrent_results,
                "memory_efficiency": memory_results,
                "overall_success": all_targets_met,
                "timestamp": datetime.now().isoformat()
            }

        finally:
            await self.teardown()


async def main():
    """Main function to run cache performance tests"""
    suite = CachePerformanceTestSuite()

    try:
        results = await suite.run_full_performance_suite()

        # Save results to file
        import json
        with open("cache_performance_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n📄 Results saved to: cache_performance_results.json")

    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())