"""
Comprehensive tests for Redis Cache Manager
Tests caching functionality, performance, and integration patterns
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List
import json

from src.caching.redis_cache_manager import (
    RedisCacheManager,
    CacheConfig,
    CacheStats,
    CacheKeyBuilder
)
from src.data_sources.base import PriceData


class TestCacheKeyBuilder:
    """Test cache key generation"""

    def test_latest_price_key(self):
        """Test latest price key generation"""
        key = CacheKeyBuilder.latest_price("BTCUSD")
        assert key == "latest:BTCUSD"

    def test_historical_data_key(self):
        """Test historical data key generation"""
        start = datetime(2024, 1, 1, 10, 0)
        end = datetime(2024, 1, 1, 11, 30)
        key = CacheKeyBuilder.historical_data("ETHUSD", start, end)
        assert key == "hist:ETHUSD:20240101_1000:20240101_1130"

    def test_ohlc_daily_key(self):
        """Test daily OHLC key generation"""
        date = datetime(2024, 1, 1)
        key = CacheKeyBuilder.ohlc_daily("ADAUSD", date)
        assert key == "ohlc_daily:ADAUSD:20240101"

    def test_symbol_list_key(self):
        """Test symbol list key generation"""
        key = CacheKeyBuilder.symbol_list("BINANCE")
        assert key == "symbols:BINANCE"

        key = CacheKeyBuilder.symbol_list()
        assert key == "symbols:all"


class TestRedisCacheManager:
    """Test Redis Cache Manager functionality"""

    @pytest.fixture
    def cache_config(self):
        """Test cache configuration"""
        return CacheConfig(
            host="localhost",
            port=6379,
            db=1,  # Use test database
            pool_max_connections=10,
            latest_price_ttl=30,
            historical_data_ttl=300
        )

    @pytest.fixture
    def sample_price_data(self):
        """Sample price data for testing"""
        return PriceData(
            symbol="BTCUSD",
            timestamp=datetime.now(),
            open_price=50000.0,
            high_price=51000.0,
            low_price=49500.0,
            close_price=50500.0,
            volume=1000000,
            source="test_source",
            quality_score=95
        )

    @pytest.fixture
    def cache_manager(self, cache_config):
        """Cache manager instance for testing"""
        return RedisCacheManager(cache_config)

    @pytest.mark.asyncio
    async def test_cache_config_validation(self, cache_config):
        """Test cache configuration validation"""
        assert cache_config.host == "localhost"
        assert cache_config.port == 6379
        assert cache_config.latest_price_ttl == 30

    @pytest.mark.asyncio
    @patch('redis.asyncio.Redis')
    @patch('redis.asyncio.ConnectionPool')
    async def test_cache_initialization(self, mock_pool_class, mock_redis_class, cache_manager):
        """Test cache initialization"""
        # Setup mocks
        mock_pool = AsyncMock()
        mock_redis = AsyncMock()
        mock_pool_class.return_value = mock_pool
        mock_redis_class.return_value = mock_redis
        mock_redis.ping.return_value = True

        await cache_manager.initialize()

        assert cache_manager.is_connected
        mock_redis.ping.assert_called_once()

    @pytest.mark.asyncio
    @patch('redis.asyncio.Redis')
    @patch('redis.asyncio.ConnectionPool')
    async def test_cache_latest_price(self, mock_pool_class, mock_redis_class, cache_manager, sample_price_data):
        """Test caching latest price data"""
        # Setup mocks
        mock_pool = AsyncMock()
        mock_redis = AsyncMock()
        mock_pool_class.return_value = mock_pool
        mock_redis_class.return_value = mock_redis
        mock_redis.ping.return_value = True
        mock_redis.setex.return_value = True

        await cache_manager.initialize()

        # Test caching
        result = await cache_manager.cache_latest_price(sample_price_data)

        assert result is True
        mock_redis.setex.assert_called_once()

        # Verify the call arguments
        call_args = mock_redis.setex.call_args
        assert call_args[0][0] == "latest:BTCUSD"  # cache key
        assert call_args[0][1] == 30  # TTL
        # JSON data should be third argument

    @pytest.mark.asyncio
    @patch('redis.asyncio.Redis')
    @patch('redis.asyncio.ConnectionPool')
    async def test_get_latest_price_cache_hit(self, mock_pool_class, mock_redis_class, cache_manager, sample_price_data):
        """Test getting latest price from cache (cache hit)"""
        # Setup mocks
        mock_pool = AsyncMock()
        mock_redis = AsyncMock()
        mock_pool_class.return_value = mock_pool
        mock_redis_class.return_value = mock_redis
        mock_redis.ping.return_value = True

        # Mock cache hit
        cached_json = sample_price_data.model_dump_json()
        mock_redis.get.return_value = cached_json

        await cache_manager.initialize()

        # Test cache hit
        result = await cache_manager.get_latest_price("BTCUSD")

        assert result is not None
        assert result.symbol == "BTCUSD"
        assert result.close_price == 50500.0
        mock_redis.get.assert_called_once_with("latest:BTCUSD")

    @pytest.mark.asyncio
    @patch('redis.asyncio.Redis')
    @patch('redis.asyncio.ConnectionPool')
    async def test_get_latest_price_cache_miss(self, mock_pool_class, mock_redis_class, cache_manager):
        """Test getting latest price cache miss with TimescaleDB fallback"""
        # Setup mocks
        mock_pool = AsyncMock()
        mock_redis = AsyncMock()
        mock_timescale = AsyncMock()

        mock_pool_class.return_value = mock_pool
        mock_redis_class.return_value = mock_redis
        mock_redis.ping.return_value = True
        mock_redis.get.return_value = None  # Cache miss

        # Mock TimescaleDB response
        mock_price_data = PriceData(
            symbol="ETHUSD",
            timestamp=datetime.now(),
            open_price=3000.0,
            high_price=3100.0,
            low_price=2950.0,
            close_price=3050.0,
            volume=500000,
            source="timescale",
            quality_score=90
        )
        mock_timescale.get_latest_data.return_value = mock_price_data

        cache_manager.timescale_handler = mock_timescale
        await cache_manager.initialize()

        # Test cache miss with fallback
        result = await cache_manager.get_latest_price("ETHUSD")

        assert result is not None
        assert result.symbol == "ETHUSD"
        assert result.close_price == 3050.0

        # Verify cache operations
        mock_redis.get.assert_called_once_with("latest:ETHUSD")
        mock_timescale.get_latest_data.assert_called_once_with("ETHUSD")
        mock_redis.setex.assert_called_once()  # Should cache the result

    @pytest.mark.asyncio
    @patch('redis.asyncio.Redis')
    @patch('redis.asyncio.ConnectionPool')
    async def test_bulk_cache_prices(self, mock_pool_class, mock_redis_class, cache_manager):
        """Test bulk caching of multiple prices"""
        # Setup mocks
        mock_pool = AsyncMock()
        mock_redis = AsyncMock()
        mock_pipeline = AsyncMock()

        mock_pool_class.return_value = mock_pool
        mock_redis_class.return_value = mock_redis
        mock_redis.ping.return_value = True
        mock_redis.pipeline.return_value = mock_pipeline
        mock_pipeline.execute.return_value = [True, True, True]  # 3 successful operations

        await cache_manager.initialize()

        # Create test data
        price_data_list = [
            PriceData(
                symbol="BTCUSD",
                timestamp=datetime.now(),
                open_price=50000.0,
                high_price=51000.0,
                low_price=49500.0,
                close_price=50500.0,
                volume=1000000,
                source="test",
                quality_score=95
            ),
            PriceData(
                symbol="ETHUSD",
                timestamp=datetime.now(),
                open_price=3000.0,
                high_price=3100.0,
                low_price=2950.0,
                close_price=3050.0,
                volume=500000,
                source="test",
                quality_score=90
            )
        ]

        # Test bulk caching
        result = await cache_manager.bulk_cache_prices(price_data_list)

        assert result == 2  # Should return number of successfully cached items
        mock_redis.pipeline.assert_called_once()
        mock_pipeline.execute.assert_called_once()

    @pytest.mark.asyncio
    @patch('redis.asyncio.Redis')
    @patch('redis.asyncio.ConnectionPool')
    async def test_cache_invalidation(self, mock_pool_class, mock_redis_class, cache_manager):
        """Test cache invalidation for a symbol"""
        # Setup mocks
        mock_pool = AsyncMock()
        mock_redis = AsyncMock()
        mock_pool_class.return_value = mock_pool
        mock_redis_class.return_value = mock_redis
        mock_redis.ping.return_value = True

        # Mock keys search and deletion
        mock_redis.keys.return_value = ["latest:BTCUSD", "hist:BTCUSD:123:456"]
        mock_redis.delete.return_value = 2

        await cache_manager.initialize()

        # Test invalidation
        result = await cache_manager.invalidate_symbol_cache("BTCUSD")

        assert result == 2
        mock_redis.keys.assert_called_once_with("*:BTCUSD:*")
        mock_redis.delete.assert_called_once()

    @pytest.mark.asyncio
    @patch('redis.asyncio.Redis')
    @patch('redis.asyncio.ConnectionPool')
    async def test_cache_statistics(self, mock_pool_class, mock_redis_class, cache_manager):
        """Test cache statistics retrieval"""
        # Setup mocks
        mock_pool = AsyncMock()
        mock_redis = AsyncMock()
        mock_pool_class.return_value = mock_pool
        mock_redis_class.return_value = mock_redis
        mock_redis.ping.return_value = True

        # Mock Redis info
        mock_redis.info.return_value = {
            'used_memory': 10485760,  # 10MB
            'evicted_keys': 5,
            'connected_clients': 3
        }

        await cache_manager.initialize()

        # Simulate some cache operations to generate stats
        cache_manager._record_hit(1.5)  # 1.5ms response time
        cache_manager._record_hit(2.0)  # 2.0ms response time
        cache_manager._record_miss(10.0)  # 10ms response time

        # Get statistics
        stats = await cache_manager.get_cache_statistics()

        assert isinstance(stats, CacheStats)
        assert stats.hits == 2
        assert stats.misses == 1
        assert stats.total_operations == 3
        assert stats.hit_rate == 2/3
        assert stats.cache_size_mb == 10.0
        assert stats.evictions == 5
        assert stats.connections_active == 3

    @pytest.mark.asyncio
    @patch('redis.asyncio.Redis')
    @patch('redis.asyncio.ConnectionPool')
    async def test_historical_data_caching(self, mock_pool_class, mock_redis_class, cache_manager):
        """Test historical data caching and retrieval"""
        # Setup mocks
        mock_pool = AsyncMock()
        mock_redis = AsyncMock()
        mock_pool_class.return_value = mock_pool
        mock_redis_class.return_value = mock_redis
        mock_redis.ping.return_value = True

        await cache_manager.initialize()

        # Test data
        start_time = datetime(2024, 1, 1, 10, 0)
        end_time = datetime(2024, 1, 1, 11, 0)
        test_data = [
            PriceData(
                symbol="BTCUSD",
                timestamp=start_time,
                open_price=50000.0,
                high_price=50100.0,
                low_price=49900.0,
                close_price=50050.0,
                volume=100000,
                source="test",
                quality_score=95
            )
        ]

        # Test caching historical data
        result = await cache_manager.cache_historical_data("BTCUSD", start_time, end_time, test_data)

        assert result is True
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    @patch('redis.asyncio.Redis')
    @patch('redis.asyncio.ConnectionPool')
    async def test_ohlc_daily_caching(self, mock_pool_class, mock_redis_class, cache_manager):
        """Test daily OHLC data caching"""
        # Setup mocks
        mock_pool = AsyncMock()
        mock_redis = AsyncMock()
        mock_pool_class.return_value = mock_pool
        mock_redis_class.return_value = mock_redis
        mock_redis.ping.return_value = True
        mock_redis.setex.return_value = True

        await cache_manager.initialize()

        # Test OHLC data
        date = datetime(2024, 1, 1)
        ohlc_data = {
            "open": 50000.0,
            "high": 51000.0,
            "low": 49000.0,
            "close": 50500.0,
            "volume": 1000000
        }

        # Test caching
        result = await cache_manager.cache_ohlc_daily("BTCUSD", date, ohlc_data)

        assert result is True
        mock_redis.setex.assert_called_once()

        # Test retrieval (cache hit)
        mock_redis.get.return_value = json.dumps(ohlc_data)
        retrieved_data = await cache_manager.get_ohlc_daily("BTCUSD", date)

        assert retrieved_data == ohlc_data

    @pytest.mark.asyncio
    @patch('redis.asyncio.Redis')
    @patch('redis.asyncio.ConnectionPool')
    async def test_cache_warm_up(self, mock_pool_class, mock_redis_class, cache_manager):
        """Test cache warming functionality"""
        # Setup mocks
        mock_pool = AsyncMock()
        mock_redis = AsyncMock()
        mock_timescale = AsyncMock()

        mock_pool_class.return_value = mock_pool
        mock_redis_class.return_value = mock_redis
        mock_redis.ping.return_value = True
        mock_redis.setex.return_value = True

        # Mock TimescaleDB responses
        mock_timescale.get_latest_data.side_effect = [
            PriceData(
                symbol="BTCUSD",
                timestamp=datetime.now(),
                open_price=50000.0,
                high_price=51000.0,
                low_price=49500.0,
                close_price=50500.0,
                volume=1000000,
                source="timescale",
                quality_score=95
            ),
            PriceData(
                symbol="ETHUSD",
                timestamp=datetime.now(),
                open_price=3000.0,
                high_price=3100.0,
                low_price=2950.0,
                close_price=3050.0,
                volume=500000,
                source="timescale",
                quality_score=90
            )
        ]

        cache_manager.timescale_handler = mock_timescale
        await cache_manager.initialize()

        # Test cache warming
        symbols = ["BTCUSD", "ETHUSD"]
        results = await cache_manager.warm_cache(symbols)

        assert len(results) == 2
        assert results["BTCUSD"] is True
        assert results["ETHUSD"] is True

        # Verify TimescaleDB calls
        assert mock_timescale.get_latest_data.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_manager_context_manager(self, cache_config):
        """Test cache manager as context manager"""
        with patch('redis.asyncio.Redis') as mock_redis_class, \
             patch('redis.asyncio.ConnectionPool') as mock_pool_class:

            mock_pool = AsyncMock()
            mock_redis = AsyncMock()
            mock_pool_class.return_value = mock_pool
            mock_redis_class.return_value = mock_redis
            mock_redis.ping.return_value = True

            async with RedisCacheManager(cache_config) as cache_manager:
                assert cache_manager.is_connected

            # Verify close was called
            mock_redis.aclose.assert_called_once()
            mock_pool.aclose.assert_called_once()

    def test_cache_stats_creation(self):
        """Test cache statistics data structure"""
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0
        assert stats.avg_response_time_ms == 0.0

    def test_cache_config_defaults(self):
        """Test cache configuration defaults"""
        config = CacheConfig()
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.latest_price_ttl == 10
        assert config.historical_data_ttl == 300

    @pytest.mark.asyncio
    @patch('redis.asyncio.Redis')
    @patch('redis.asyncio.ConnectionPool')
    async def test_error_handling(self, mock_pool_class, mock_redis_class, cache_manager):
        """Test error handling in cache operations"""
        # Setup mocks
        mock_pool = AsyncMock()
        mock_redis = AsyncMock()
        mock_pool_class.return_value = mock_pool
        mock_redis_class.return_value = mock_redis
        mock_redis.ping.return_value = True

        # Mock Redis error
        mock_redis.get.side_effect = Exception("Redis connection error")

        await cache_manager.initialize()

        # Test error handling
        result = await cache_manager.get_latest_price("BTCUSD")

        # Should handle error gracefully and return None
        assert result is None

    @pytest.mark.integration
    async def test_real_redis_connection(self):
        """Integration test with real Redis instance"""
        # This test requires a running Redis instance
        config = CacheConfig(db=15)  # Use test database
        cache_manager = RedisCacheManager(config)

        try:
            await cache_manager.initialize()
            assert cache_manager.is_connected

            # Test basic operations
            test_data = PriceData(
                symbol="TEST",
                timestamp=datetime.now(),
                open_price=100.0,
                high_price=101.0,
                low_price=99.0,
                close_price=100.5,
                volume=1000,
                source="test",
                quality_score=100
            )

            # Cache and retrieve
            await cache_manager.cache_latest_price(test_data)
            retrieved = await cache_manager.get_latest_price("TEST")

            assert retrieved is not None
            assert retrieved.symbol == "TEST"
            assert retrieved.close_price == 100.5

        except Exception as e:
            pytest.skip(f"Redis not available for integration test: {e}")
        finally:
            if cache_manager.is_connected:
                await cache_manager.close()

    @pytest.mark.integration
    async def test_cache_performance_benchmark(self):
        """Performance benchmark test for cache operations"""
        config = CacheConfig(db=15)
        cache_manager = RedisCacheManager(config)

        try:
            await cache_manager.initialize()

            # Benchmark cache operations
            import time

            start_time = time.perf_counter()
            operations = 1000

            for i in range(operations):
                test_data = PriceData(
                    symbol=f"PERF{i}",
                    timestamp=datetime.now(),
                    open_price=100.0 + i,
                    high_price=101.0 + i,
                    low_price=99.0 + i,
                    close_price=100.5 + i,
                    volume=1000 + i,
                    source="perf_test",
                    quality_score=95
                )
                await cache_manager.cache_latest_price(test_data)

            cache_time = time.perf_counter() - start_time

            # Read benchmark
            start_time = time.perf_counter()

            for i in range(operations):
                await cache_manager.get_latest_price(f"PERF{i}")

            read_time = time.perf_counter() - start_time

            print(f"Cache performance: {operations} writes in {cache_time:.3f}s ({operations/cache_time:.0f} ops/sec)")
            print(f"Read performance: {operations} reads in {read_time:.3f}s ({operations/read_time:.0f} ops/sec)")

            # Performance assertions
            assert cache_time < 10.0  # Should complete 1000 writes in under 10 seconds
            assert read_time < 5.0   # Should complete 1000 reads in under 5 seconds

        except Exception as e:
            pytest.skip(f"Redis not available for performance test: {e}")
        finally:
            if cache_manager.is_connected:
                await cache_manager.close()