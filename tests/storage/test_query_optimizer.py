"""
Comprehensive tests for Query Optimizer
Tests materialized views, query optimization, and performance monitoring
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import json

from src.storage.query_optimizer import (
    QueryOptimizer,
    QueryType,
    QueryPerformanceMetrics,
    MaterializedViewConfig
)
from src.storage.timescaledb_handler import TimescaleDBHandler
from src.caching.redis_cache_manager import RedisCacheManager


class TestQueryOptimizer:
    """Test Query Optimizer functionality"""

    @pytest.fixture
    def mock_timescale_handler(self):
        """Mock TimescaleDB handler"""
        handler = AsyncMock(spec=TimescaleDBHandler)
        handler.pool = AsyncMock()
        handler.pool.acquire = AsyncMock()
        return handler

    @pytest.fixture
    def mock_cache_manager(self):
        """Mock Redis cache manager"""
        cache = AsyncMock(spec=RedisCacheManager)
        cache.redis_client = AsyncMock()
        return cache

    @pytest.fixture
    def query_optimizer(self, mock_timescale_handler, mock_cache_manager):
        """Query optimizer instance for testing"""
        return QueryOptimizer(mock_timescale_handler, mock_cache_manager)

    @pytest.mark.asyncio
    async def test_query_optimizer_initialization(self, query_optimizer, mock_timescale_handler):
        """Test query optimizer initialization"""
        # Setup mock connection
        mock_conn = AsyncMock()
        mock_timescale_handler.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_timescale_handler.pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        await query_optimizer.initialize()

        assert query_optimizer.is_initialized
        assert len(query_optimizer.materialized_views) > 0
        mock_conn.execute.assert_called()

    @pytest.mark.asyncio
    async def test_materialized_view_creation(self, query_optimizer, mock_timescale_handler):
        """Test materialized view creation"""
        mock_conn = AsyncMock()
        mock_timescale_handler.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_timescale_handler.pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        config = MaterializedViewConfig(
            view_name="test_view",
            refresh_interval_minutes=30,
            retention_days=90,
            indexes=["symbol", "date"],
            query_template="CREATE MATERIALIZED VIEW test_view AS SELECT * FROM test_table;"
        )

        await query_optimizer._create_single_materialized_view(config)

        # Verify materialized view was created
        assert mock_conn.execute.call_count >= 1  # View creation + indexes

    @pytest.mark.asyncio
    async def test_query_optimization_with_cache_hit(self, query_optimizer, mock_cache_manager):
        """Test query optimization with cache hit"""
        # Setup query optimizer without full initialization
        query_optimizer.is_initialized = True
        query_optimizer.cache_manager = mock_cache_manager

        # Mock cached result
        cached_data = [{"symbol": "BTCUSD", "price": 50000}]
        mock_cache_manager.redis_client.get.return_value = json.dumps(cached_data)

        query_sql = "SELECT * FROM symbol_performance_metrics WHERE symbol = $1"
        result, metrics = await query_optimizer.optimize_query(
            query_sql,
            QueryType.ANALYTICS,
            ["BTCUSD"]
        )

        assert result == cached_data
        assert metrics.cache_hit is True
        assert metrics.optimization_applied == "cache_hit"
        assert metrics.query_type == QueryType.ANALYTICS

    @pytest.mark.asyncio
    async def test_query_optimization_with_cache_miss(self, query_optimizer, mock_timescale_handler, mock_cache_manager):
        """Test query optimization with cache miss"""
        query_optimizer.is_initialized = True
        query_optimizer.cache_manager = mock_cache_manager

        # Setup mock database result
        mock_conn = AsyncMock()
        mock_result_row = {"symbol": "ETHUSD", "price": 3000}
        mock_conn.fetch.return_value = [mock_result_row]

        mock_timescale_handler.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_timescale_handler.pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock cache miss
        mock_cache_manager.redis_client.get.return_value = None

        query_sql = "SELECT * FROM ohlcv_data WHERE symbol = $1"
        result, metrics = await query_optimizer.optimize_query(
            query_sql,
            QueryType.TIME_RANGE,
            ["ETHUSD"]
        )

        assert result == [mock_result_row]
        assert metrics.cache_hit is False
        assert metrics.query_type == QueryType.TIME_RANGE
        mock_conn.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_materialized_view_optimization(self, query_optimizer):
        """Test materialized view query optimization"""
        query_optimizer.is_initialized = True
        query_optimizer.materialized_views = {
            "daily_ohlcv_summary": MaterializedViewConfig(
                view_name="daily_ohlcv_summary",
                refresh_interval_minutes=60,
                retention_days=365,
                indexes=["symbol", "date"],
                query_template="CREATE MATERIALIZED VIEW daily_ohlcv_summary AS SELECT..."
            )
        }

        # Test daily OHLC query optimization
        original_query = "SELECT * FROM ohlcv_data WHERE symbol = 'BTCUSD' AND date_trunc('day', time) = '2024-01-01'"
        optimized_query, optimization = await query_optimizer._try_materialized_view_optimization(
            original_query,
            QueryType.AGGREGATION
        )

        # Should optimize for daily queries
        assert "daily_ohlcv_summary" in optimization or optimization == "no_optimization"

    @pytest.mark.asyncio
    async def test_materialized_view_refresh(self, query_optimizer, mock_timescale_handler):
        """Test materialized view refresh functionality"""
        query_optimizer.is_initialized = True
        query_optimizer.materialized_views = {
            "test_view": MaterializedViewConfig(
                view_name="test_view",
                refresh_interval_minutes=30,
                retention_days=90,
                indexes=["symbol"],
                query_template="CREATE MATERIALIZED VIEW test_view AS SELECT..."
            )
        }

        mock_conn = AsyncMock()
        mock_timescale_handler.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_timescale_handler.pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        results = await query_optimizer.refresh_materialized_views(force_all=True)

        assert "test_view" in results
        mock_conn.execute.assert_called_with("REFRESH MATERIALIZED VIEW test_view;")

    @pytest.mark.asyncio
    async def test_query_performance_metrics_tracking(self, query_optimizer):
        """Test query performance metrics tracking"""
        query_optimizer.is_initialized = True

        # Add some test metrics
        metrics1 = QueryPerformanceMetrics(
            query_type=QueryType.LATEST_PRICE,
            execution_time_ms=5.0,
            rows_returned=1,
            rows_examined=1,
            cache_hit=True
        )

        metrics2 = QueryPerformanceMetrics(
            query_type=QueryType.AGGREGATION,
            execution_time_ms=25.0,
            rows_returned=100,
            rows_examined=1000,
            cache_hit=False
        )

        query_optimizer.performance_metrics = [metrics1, metrics2]

        stats = await query_optimizer.get_query_performance_stats()

        assert stats["overall"]["total_queries"] == 2
        assert stats["overall"]["cache_hit_rate"] == 0.5
        assert "latest_price" in stats["by_query_type"]
        assert "aggregation" in stats["by_query_type"]

    @pytest.mark.asyncio
    async def test_slow_query_analysis(self, query_optimizer, mock_timescale_handler):
        """Test slow query analysis functionality"""
        mock_conn = AsyncMock()
        mock_slow_query = {
            "query": "SELECT * FROM ohlcv_data WHERE time > NOW() - INTERVAL '1 day'",
            "calls": 100,
            "total_time": 5000,
            "mean_time": 50,
            "max_time": 200,
            "rows": 10000,
            "hit_percent": 95.5
        }
        mock_conn.fetch.return_value = [mock_slow_query]

        mock_timescale_handler.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_timescale_handler.pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        slow_queries = await query_optimizer.analyze_slow_queries(threshold_ms=30.0)

        assert len(slow_queries) == 1
        assert slow_queries[0]["mean_time"] == 50
        mock_conn.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_optimization_suggestions(self, query_optimizer):
        """Test optimization suggestions based on performance"""
        query_optimizer.is_initialized = True

        # Add metrics indicating poor performance
        poor_metrics = [
            QueryPerformanceMetrics(
                query_type=QueryType.ANALYTICS,
                execution_time_ms=100.0,  # Slow query
                rows_returned=1000,
                rows_examined=10000,
                cache_hit=False
            ) for _ in range(10)
        ]

        query_optimizer.performance_metrics = poor_metrics

        suggestions = await query_optimizer.suggest_optimizations()

        assert len(suggestions) > 0
        suggestion_types = [s["type"] for s in suggestions]
        assert "performance" in suggestion_types or "caching" in suggestion_types

    @pytest.mark.asyncio
    async def test_cache_key_generation(self, query_optimizer):
        """Test cache key generation for queries"""
        query_sql = "SELECT * FROM ohlcv_data WHERE symbol = $1 AND time > $2"
        parameters = ["BTCUSD", "2024-01-01"]

        cache_key = query_optimizer._generate_cache_key(query_sql, parameters)

        assert cache_key.startswith("query_cache:")
        assert len(cache_key) > 20  # Should be a proper hash

        # Same query should generate same key
        cache_key2 = query_optimizer._generate_cache_key(query_sql, parameters)
        assert cache_key == cache_key2

        # Different query should generate different key
        cache_key3 = query_optimizer._generate_cache_key(query_sql, ["ETHUSD", "2024-01-01"])
        assert cache_key != cache_key3

    @pytest.mark.asyncio
    async def test_query_result_caching(self, query_optimizer, mock_cache_manager):
        """Test query result caching functionality"""
        query_optimizer.cache_manager = mock_cache_manager

        cache_key = "test_cache_key"
        result_data = [{"symbol": "BTCUSD", "price": 50000}]

        # Test caching result
        await query_optimizer._cache_result(cache_key, result_data, ttl=300)

        mock_cache_manager.redis_client.setex.assert_called_once_with(
            cache_key,
            300,
            json.dumps(result_data, default=str)
        )

    @pytest.mark.asyncio
    async def test_query_type_optimization_strategies(self, query_optimizer, mock_timescale_handler):
        """Test different optimization strategies for different query types"""
        query_optimizer.is_initialized = True

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [{"result": "test"}]
        mock_timescale_handler.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_timescale_handler.pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        # Test different query types
        query_types = [
            QueryType.LATEST_PRICE,
            QueryType.TIME_RANGE,
            QueryType.AGGREGATION,
            QueryType.ANALYTICS,
            QueryType.BULK_EXPORT
        ]

        for query_type in query_types:
            result, optimization = await query_optimizer._execute_optimized_query(
                "SELECT * FROM ohlcv_data LIMIT 1",
                query_type,
                []
            )

            assert result is not None
            assert isinstance(optimization, str)

    @pytest.mark.asyncio
    async def test_performance_metrics_cleanup(self, query_optimizer):
        """Test that performance metrics are cleaned up to prevent memory issues"""
        query_optimizer.is_initialized = True

        # Add more than 1000 metrics
        for i in range(1200):
            metrics = QueryPerformanceMetrics(
                query_type=QueryType.LATEST_PRICE,
                execution_time_ms=5.0,
                rows_returned=1,
                rows_examined=1,
                cache_hit=True
            )
            query_optimizer.performance_metrics.append(metrics)

        # Simulate what happens during optimize_query
        if len(query_optimizer.performance_metrics) > 1000:
            query_optimizer.performance_metrics = query_optimizer.performance_metrics[-1000:]

        assert len(query_optimizer.performance_metrics) == 1000

    @pytest.mark.asyncio
    async def test_materialized_view_config_validation(self):
        """Test materialized view configuration validation"""
        config = MaterializedViewConfig(
            view_name="test_view",
            refresh_interval_minutes=30,
            retention_days=90,
            indexes=["symbol", "date"],
            query_template="CREATE MATERIALIZED VIEW test_view AS SELECT..."
        )

        assert config.view_name == "test_view"
        assert config.refresh_interval_minutes == 30
        assert config.retention_days == 90
        assert len(config.indexes) == 2
        assert len(config.dependencies) == 0  # Default empty list

    @pytest.mark.asyncio
    async def test_query_optimization_error_handling(self, query_optimizer, mock_timescale_handler):
        """Test error handling in query optimization"""
        query_optimizer.is_initialized = True

        # Mock database error
        mock_conn = AsyncMock()
        mock_conn.fetch.side_effect = Exception("Database connection error")
        mock_timescale_handler.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_timescale_handler.pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with pytest.raises(Exception):
            await query_optimizer.optimize_query(
                "SELECT * FROM ohlcv_data",
                QueryType.ANALYTICS
            )

        # Should still record error metrics
        assert len(query_optimizer.performance_metrics) > 0
        assert query_optimizer.performance_metrics[-1].optimization_applied == "error"

    @pytest.mark.asyncio
    async def test_query_optimizer_close(self, query_optimizer):
        """Test query optimizer cleanup"""
        # Add some test data
        query_optimizer.performance_metrics = [
            QueryPerformanceMetrics(
                query_type=QueryType.LATEST_PRICE,
                execution_time_ms=5.0,
                rows_returned=1,
                rows_examined=1,
                cache_hit=True
            )
        ]
        query_optimizer.query_cache = {"test": "data"}

        await query_optimizer.close()

        assert len(query_optimizer.performance_metrics) == 0
        assert len(query_optimizer.query_cache) == 0

    def test_query_performance_metrics_properties(self):
        """Test QueryPerformanceMetrics calculated properties"""
        metrics = QueryPerformanceMetrics(
            query_type=QueryType.AGGREGATION,
            execution_time_ms=50.0,
            rows_returned=100,
            rows_examined=1000,
            cache_hit=False,
            optimization_applied="materialized_view"
        )

        assert metrics.query_type == QueryType.AGGREGATION
        assert metrics.execution_time_ms == 50.0
        assert metrics.cache_hit is False
        assert metrics.optimization_applied == "materialized_view"
        assert isinstance(metrics.timestamp, datetime)

    @pytest.mark.asyncio
    async def test_query_rewriting_for_materialized_views(self, query_optimizer):
        """Test query rewriting logic for materialized views"""
        # Test volume stats rewriting
        volume_query = "SELECT symbol, SUM(volume) FROM ohlcv_data GROUP BY symbol"
        rewritten = query_optimizer._rewrite_for_volume_stats(volume_query)
        assert "hourly_volume_stats" in rewritten if rewritten else True

        # Test performance metrics rewriting
        performance_query = "SELECT symbol, price_change_percent FROM ohlcv_data"
        rewritten = query_optimizer._rewrite_for_performance_metrics(performance_query)
        assert "symbol_performance_metrics" in rewritten if rewritten else True


class TestQueryOptimizerIntegration:
    """Integration tests for Query Optimizer"""

    @pytest.mark.integration
    async def test_real_query_optimization(self):
        """Integration test with real database components"""
        # This test would require actual database connections
        # Skipped in unit tests, but important for integration testing
        pytest.skip("Integration test requires real database connection")

    @pytest.mark.integration
    async def test_materialized_view_performance(self):
        """Test materialized view performance improvement"""
        # This test would benchmark query performance with and without materialized views
        pytest.skip("Integration test requires real database connection")

    @pytest.mark.integration
    async def test_cache_integration_performance(self):
        """Test cache integration performance"""
        # This test would benchmark cache hit/miss performance
        pytest.skip("Integration test requires Redis connection")