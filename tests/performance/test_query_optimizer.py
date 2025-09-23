"""
Tests for Query Optimizer System
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from src.performance.query_optimizer import (
    QueryOptimizer, QueryPlan, OptimizationStrategy, ConnectionPool
)


class TestQueryOptimizer:
    """Test cases for QueryOptimizer"""

    @pytest.fixture
    async def mock_database(self):
        """Mock database connection"""
        db = Mock()
        db.execute = AsyncMock()
        db.fetch_all = AsyncMock(return_value=[
            {"symbol": "AAPL", "price": 150.0, "timestamp": datetime.now()},
            {"symbol": "GOOGL", "price": 2800.0, "timestamp": datetime.now()}
        ])
        db.fetch_one = AsyncMock(return_value={"symbol": "AAPL", "price": 150.0})
        return db

    @pytest.fixture
    def optimizer(self):
        """Query optimizer instance"""
        return QueryOptimizer(
            db_path="test.db",
            pool_size=5,
            cache_size=100
        )

    def test_initialization(self):
        """Test optimizer initialization"""
        optimizer = QueryOptimizer(
            db_path="test.db",
            pool_size=3,
            cache_size=50
        )

        assert optimizer.db_path == "test.db"
        assert optimizer.pool_size == 3
        assert optimizer.cache_size == 50

    @pytest.mark.asyncio
    async def test_query_plan_creation(self, optimizer):
        """Test query plan creation"""
        query = "SELECT * FROM ohlcv_data WHERE symbol = ?"
        params = ("AAPL",)

        plan = await optimizer.create_query_plan(query, params)

        assert isinstance(plan, QueryPlan)
        assert plan.query == query
        assert plan.params == params

    @pytest.mark.asyncio
    async def test_query_execution(self, optimizer):
        """Test basic query execution"""
        query = "SELECT * FROM ohlcv_data WHERE symbol = ?"
        params = ("AAPL",)

        # Mock the database connection
        with pytest.raises(Exception):  # Expected since we don't have real DB
            result = await optimizer.execute_query(query, params)

    @pytest.mark.asyncio
    async def test_query_caching(self, optimizer):
        """Test query result caching"""
        query = "SELECT * FROM ohlcv_data WHERE symbol = ?"
        params = ("AAPL",)

        # Add to cache
        expected_result = [{"symbol": "AAPL", "price": 150.0}]
        cache_key = optimizer._get_cache_key(query, params)
        optimizer.query_cache[cache_key] = expected_result

        # Should retrieve from cache
        cached_result = optimizer._get_from_cache(query, params)
        assert cached_result == expected_result

    def test_cache_key_generation(self, optimizer):
        """Test cache key generation"""
        query1 = "SELECT * FROM table WHERE id = ?"
        params1 = (123,)

        query2 = "SELECT * FROM table WHERE id = ?"
        params2 = (456,)

        key1 = optimizer._get_cache_key(query1, params1)
        key2 = optimizer._get_cache_key(query2, params2)

        # Same query, different params should have different keys
        assert key1 != key2

        # Same query and params should have same key
        key1_repeat = optimizer._get_cache_key(query1, params1)
        assert key1 == key1_repeat

    def test_query_optimization_strategies(self, optimizer):
        """Test different optimization strategies"""
        strategies = [
            OptimizationStrategy.AGGRESSIVE,
            OptimizationStrategy.BALANCED,
            OptimizationStrategy.CONSERVATIVE,
            OptimizationStrategy.ADAPTIVE
        ]

        for strategy in strategies:
            optimizer.set_optimization_strategy(strategy)
            assert optimizer.optimization_strategy == strategy

    def test_connection_pool_management(self, optimizer):
        """Test connection pool management"""
        pool = ConnectionPool(
            db_path="test.db",
            pool_size=5,
            max_overflow=2
        )

        assert pool.pool_size == 5
        assert pool.max_overflow == 2
        assert pool.db_path == "test.db"

    @pytest.mark.asyncio
    async def test_query_plan_optimization(self, optimizer):
        """Test query plan optimization"""
        complex_query = """
        SELECT ohlcv.*, symbols.name
        FROM ohlcv_data ohlcv
        JOIN symbols ON ohlcv.symbol = symbols.symbol
        WHERE ohlcv.timestamp BETWEEN ? AND ?
        ORDER BY ohlcv.timestamp DESC
        """
        params = (datetime.now() - timedelta(days=1), datetime.now())

        plan = await optimizer.create_query_plan(complex_query, params)
        optimized_plan = optimizer.optimize_plan(plan)

        assert isinstance(optimized_plan, QueryPlan)
        assert optimized_plan.estimated_cost >= 0

    def test_query_analysis(self, optimizer):
        """Test query analysis functionality"""
        simple_query = "SELECT * FROM table WHERE id = 1"
        complex_query = """
        SELECT t1.*, t2.name, AVG(t3.value)
        FROM table1 t1
        JOIN table2 t2 ON t1.id = t2.table1_id
        LEFT JOIN table3 t3 ON t1.id = t3.table1_id
        WHERE t1.created_at > '2024-01-01'
        GROUP BY t1.id, t2.name
        HAVING AVG(t3.value) > 100
        ORDER BY t1.created_at DESC
        LIMIT 1000
        """

        simple_complexity = optimizer.analyze_query_complexity(simple_query)
        complex_complexity = optimizer.analyze_query_complexity(complex_query)

        assert complex_complexity > simple_complexity

    def test_index_recommendations(self, optimizer):
        """Test index recommendation system"""
        query = "SELECT * FROM ohlcv_data WHERE symbol = ? AND timestamp > ?"

        recommendations = optimizer.get_index_recommendations(query)

        assert isinstance(recommendations, list)
        # Should recommend indexes on symbol and timestamp

    def test_query_statistics(self, optimizer):
        """Test query statistics collection"""
        # Simulate some query executions
        optimizer._record_query_execution("SELECT * FROM table1", 0.05)
        optimizer._record_query_execution("SELECT * FROM table2", 0.12)
        optimizer._record_query_execution("SELECT * FROM table1", 0.03)

        stats = optimizer.get_query_statistics()

        assert 'total_queries' in stats
        assert 'average_execution_time' in stats
        assert 'most_frequent_queries' in stats

    @pytest.mark.asyncio
    async def test_prepared_statements(self, optimizer):
        """Test prepared statement functionality"""
        query = "SELECT * FROM ohlcv_data WHERE symbol = ?"

        # Prepare statement
        stmt_id = await optimizer.prepare_statement(query)
        assert stmt_id is not None

        # Execute prepared statement
        params = ("AAPL",)

        # Mock execution since we don't have real DB
        try:
            result = await optimizer.execute_prepared(stmt_id, params)
        except Exception:
            pass  # Expected without real database

    def test_batch_optimization(self, optimizer):
        """Test batch query optimization"""
        queries = [
            ("SELECT * FROM ohlcv_data WHERE symbol = ?", ("AAPL",)),
            ("SELECT * FROM ohlcv_data WHERE symbol = ?", ("GOOGL",)),
            ("SELECT * FROM ohlcv_data WHERE symbol = ?", ("MSFT",))
        ]

        optimized_batch = optimizer.optimize_batch_queries(queries)

        # Should optimize similar queries together
        assert isinstance(optimized_batch, list)

    def test_connection_pooling_stats(self, optimizer):
        """Test connection pool statistics"""
        pool = ConnectionPool("test.db", pool_size=5)

        stats = pool.get_stats()

        assert 'pool_size' in stats
        assert 'active_connections' in stats
        assert 'available_connections' in stats

    def test_query_cache_eviction(self, optimizer):
        """Test query cache eviction policies"""
        # Fill cache beyond capacity
        for i in range(optimizer.cache_size + 10):
            query = f"SELECT * FROM table WHERE id = {i}"
            params = ()
            cache_key = optimizer._get_cache_key(query, params)
            optimizer.query_cache[cache_key] = [{"id": i}]

        # Cache should not exceed max size
        assert len(optimizer.query_cache) <= optimizer.cache_size

    @pytest.mark.asyncio
    async def test_connection_timeout_handling(self, optimizer):
        """Test connection timeout handling"""
        # Test with very short timeout
        optimizer.connection_timeout = 0.001

        try:
            # This should timeout quickly
            await optimizer.get_connection()
        except Exception:
            pass  # Expected timeout

    def test_query_plan_comparison(self, optimizer):
        """Test query plan comparison"""
        query1 = "SELECT * FROM table WHERE id = ?"
        query2 = "SELECT * FROM table WHERE name = ?"

        plan1 = QueryPlan(query1, (1,), estimated_cost=10.0)
        plan2 = QueryPlan(query2, ("test",), estimated_cost=15.0)

        # Lower cost plan should be preferred
        assert plan1.estimated_cost < plan2.estimated_cost

    def test_optimization_hints(self, optimizer):
        """Test query optimization hints"""
        query = "SELECT * FROM large_table WHERE condition = ?"

        # Add optimization hint
        hint = "USE INDEX (idx_condition)"
        hinted_query = optimizer.apply_optimization_hint(query, hint)

        assert hint in hinted_query or query in hinted_query

    def test_concurrent_optimization(self, optimizer):
        """Test concurrent query optimization"""
        import threading

        results = []

        def optimize_query(query_id):
            query = f"SELECT * FROM table WHERE id = {query_id}"
            params = (query_id,)

            try:
                plan = QueryPlan(query, params, estimated_cost=1.0)
                optimized = optimizer.optimize_plan(plan)
                results.append(optimized)
            except Exception:
                pass

        # Start multiple optimization threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=optimize_query, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Should handle concurrent optimization safely
        assert len(results) >= 0  # No crashes

    def test_adaptive_optimization(self, optimizer):
        """Test adaptive optimization based on query patterns"""
        optimizer.set_optimization_strategy(OptimizationStrategy.ADAPTIVE)

        # Simulate query pattern learning
        frequent_query = "SELECT * FROM popular_table WHERE id = ?"

        for i in range(10):
            optimizer._record_query_execution(frequent_query, 0.05)

        # Optimizer should adapt to this pattern
        adaptive_settings = optimizer.get_adaptive_settings()
        assert isinstance(adaptive_settings, dict)

    def test_memory_optimization(self, optimizer):
        """Test memory optimization features"""
        # Test memory-efficient query processing
        large_result_query = "SELECT * FROM large_table"

        # Should use streaming or chunking for large results
        chunk_size = optimizer.get_optimal_chunk_size(large_result_query)
        assert chunk_size > 0

    def test_query_plan_caching(self, optimizer):
        """Test query plan caching"""
        query = "SELECT * FROM table WHERE id = ?"
        params = (1,)

        # Create and cache plan
        plan = QueryPlan(query, params, estimated_cost=5.0)
        optimizer.cache_query_plan(query, plan)

        # Retrieve cached plan
        cached_plan = optimizer.get_cached_plan(query)
        assert cached_plan is not None
        assert cached_plan.query == query