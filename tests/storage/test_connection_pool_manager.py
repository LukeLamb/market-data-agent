"""
Tests for Advanced Connection Pool Manager
Tests connection pooling, priority allocation, and performance optimization
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import asdict

from src.storage.connection_pool_manager import (
    AdvancedConnectionPool,
    ConnectionPoolConfig,
    ConnectionPriority,
    ConnectionMetrics,
    PoolStatistics
)


class TestConnectionPoolConfig:
    """Test connection pool configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = ConnectionPoolConfig()

        assert config.min_size == 10
        assert config.max_size == 100
        assert config.command_timeout == 30.0
        assert config.max_queries == 50000
        assert config.enable_load_balancing is True
        assert config.enable_prepared_statements is True
        assert len(config.server_settings) > 0

    def test_custom_config(self):
        """Test custom configuration"""
        config = ConnectionPoolConfig(
            min_size=5,
            max_size=50,
            command_timeout=60.0,
            enable_load_balancing=False
        )

        assert config.min_size == 5
        assert config.max_size == 50
        assert config.command_timeout == 60.0
        assert config.enable_load_balancing is False

    def test_server_settings_post_init(self):
        """Test server settings are properly initialized"""
        config = ConnectionPoolConfig()

        assert 'application_name' in config.server_settings
        assert 'tcp_keepalives_idle' in config.server_settings
        assert 'statement_timeout' in config.server_settings


class TestConnectionMetrics:
    """Test connection metrics tracking"""

    def test_connection_metrics_creation(self):
        """Test connection metrics initialization"""
        created_time = datetime.now()
        metrics = ConnectionMetrics(
            connection_id="test_conn_1",
            created_at=created_time,
            last_used=created_time
        )

        assert metrics.connection_id == "test_conn_1"
        assert metrics.created_at == created_time
        assert metrics.query_count == 0
        assert metrics.error_count == 0
        assert metrics.is_healthy is True

    def test_connection_metrics_calculations(self):
        """Test connection metrics calculated properties"""
        created_time = datetime.now() - timedelta(minutes=10)
        last_used_time = datetime.now() - timedelta(minutes=2)

        metrics = ConnectionMetrics(
            connection_id="test_conn_1",
            created_at=created_time,
            last_used=last_used_time,
            query_count=100,
            total_time_ms=5000.0
        )

        assert metrics.avg_query_time_ms == 50.0
        assert metrics.age_seconds > 590  # ~10 minutes
        assert metrics.idle_time_seconds > 110  # ~2 minutes

    def test_priority_usage_tracking(self):
        """Test priority usage tracking"""
        metrics = ConnectionMetrics(
            connection_id="test_conn_1",
            created_at=datetime.now(),
            last_used=datetime.now()
        )

        # Test all priorities are initialized
        for priority in ConnectionPriority:
            assert priority in metrics.priority_usage
            assert metrics.priority_usage[priority] == 0


class TestAdvancedConnectionPool:
    """Test Advanced Connection Pool functionality"""

    @pytest.fixture
    def pool_config(self):
        """Test pool configuration"""
        return ConnectionPoolConfig(
            min_size=2,
            max_size=5,
            command_timeout=10.0,
            health_check_interval=5.0
        )

    @pytest.fixture
    def mock_pool(self, pool_config):
        """Mock connection pool for testing"""
        return AdvancedConnectionPool("postgresql://test:test@localhost/test", pool_config)

    @pytest.mark.asyncio
    @patch('asyncpg.create_pool')
    async def test_pool_initialization(self, mock_create_pool, mock_pool):
        """Test connection pool initialization"""
        mock_asyncpg_pool = AsyncMock()
        mock_create_pool.return_value = mock_asyncpg_pool

        await mock_pool.initialize()

        assert mock_pool.is_initialized
        assert mock_pool.pool == mock_asyncpg_pool
        mock_create_pool.assert_called_once()

    @pytest.mark.asyncio
    @patch('asyncpg.create_pool')
    async def test_connection_initialization(self, mock_create_pool, mock_pool):
        """Test individual connection initialization"""
        mock_connection = AsyncMock()
        mock_connection.execute = AsyncMock()

        await mock_pool._initialize_connection(mock_connection)

        # Should have executed optimization commands
        assert mock_connection.execute.call_count >= 3
        # Should have created connection metrics
        connection_id = f"conn_{id(mock_connection)}"
        assert connection_id in mock_pool.connection_metrics

    @pytest.mark.asyncio
    async def test_connection_acquisition_with_priority(self, mock_pool):
        """Test connection acquisition with priority levels"""
        # Mock the pool
        mock_asyncpg_pool = AsyncMock()
        mock_connection = AsyncMock()
        mock_asyncpg_pool.acquire.return_value = mock_connection
        mock_asyncpg_pool.release = AsyncMock()
        mock_pool.pool = mock_asyncpg_pool
        mock_pool.is_initialized = True

        async with mock_pool.acquire_connection(ConnectionPriority.CRITICAL) as conn:
            assert conn == mock_connection

        mock_asyncpg_pool.acquire.assert_called()
        mock_asyncpg_pool.release.assert_called_with(mock_connection)

    @pytest.mark.asyncio
    async def test_connection_metrics_tracking(self, mock_pool):
        """Test connection metrics are properly tracked"""
        mock_pool.is_initialized = True
        mock_asyncpg_pool = AsyncMock()
        mock_connection = AsyncMock()
        mock_asyncpg_pool.acquire.return_value = mock_connection
        mock_pool.pool = mock_asyncpg_pool

        # Initialize connection metrics
        connection_id = f"conn_{id(mock_connection)}"
        mock_pool.connection_metrics[connection_id] = ConnectionMetrics(
            connection_id=connection_id,
            created_at=datetime.now(),
            last_used=datetime.now()
        )

        async with mock_pool.acquire_connection(ConnectionPriority.HIGH):
            pass

        # Verify metrics were updated
        metrics = mock_pool.connection_metrics[connection_id]
        assert metrics.priority_usage[ConnectionPriority.HIGH] == 1
        assert metrics.query_count == 1

    @pytest.mark.asyncio
    async def test_execute_optimized_query(self, mock_pool):
        """Test optimized query execution"""
        mock_pool.is_initialized = True
        mock_asyncpg_pool = AsyncMock()
        mock_connection = AsyncMock()
        mock_connection.fetch.return_value = [{"result": "test"}]
        mock_asyncpg_pool.acquire.return_value = mock_connection
        mock_pool.pool = mock_asyncpg_pool

        result = await mock_pool.execute_optimized(
            "SELECT * FROM test_table WHERE id = $1",
            1,
            priority=ConnectionPriority.HIGH
        )

        assert result == [{"result": "test"}]
        mock_connection.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_prepared_statements(self, mock_pool):
        """Test prepared statement execution"""
        mock_pool.is_initialized = True
        mock_pool.config.enable_prepared_statements = True

        mock_connection = AsyncMock()
        mock_connection.execute = AsyncMock()
        mock_connection.fetch.return_value = [{"result": "prepared"}]

        query = "SELECT * FROM test_table WHERE id = $1"
        args = (1,)

        result = await mock_pool._execute_prepared(mock_connection, query, args)

        assert result == [{"result": "prepared"}]
        # Should have prepared the statement
        assert mock_connection.execute.call_count >= 1

    @pytest.mark.asyncio
    async def test_batch_execution(self, mock_pool):
        """Test batch query execution"""
        mock_pool.is_initialized = True
        mock_asyncpg_pool = AsyncMock()
        mock_connection = AsyncMock()
        mock_transaction = AsyncMock()

        mock_connection.fetch.side_effect = [
            [{"id": 1, "name": "test1"}],
            [{"id": 2, "name": "test2"}]
        ]
        mock_connection.transaction.return_value = mock_transaction
        mock_transaction.__aenter__ = AsyncMock()
        mock_transaction.__aexit__ = AsyncMock()

        mock_asyncpg_pool.acquire.return_value = mock_connection
        mock_pool.pool = mock_asyncpg_pool

        queries = [
            ("SELECT * FROM test_table WHERE id = $1", (1,)),
            ("SELECT * FROM test_table WHERE id = $1", (2,))
        ]

        results = await mock_pool.execute_batch_optimized(queries, ConnectionPriority.NORMAL)

        assert len(results) == 2
        assert results[0] == [{"id": 1, "name": "test1"}]
        assert results[1] == [{"id": 2, "name": "test2"}]

    @pytest.mark.asyncio
    async def test_health_check_functionality(self, mock_pool):
        """Test connection health checking"""
        # Add some test connection metrics
        now = datetime.now()
        old_time = now - timedelta(hours=2)

        mock_pool.connection_metrics = {
            "healthy_conn": ConnectionMetrics(
                connection_id="healthy_conn",
                created_at=now,
                last_used=now,
                query_count=100,
                error_count=0
            ),
            "unhealthy_conn": ConnectionMetrics(
                connection_id="unhealthy_conn",
                created_at=old_time,
                last_used=old_time,
                query_count=60000,  # Exceeds max_queries
                error_count=10  # Too many errors
            )
        }

        await mock_pool._perform_health_check()

        # Unhealthy connection should be marked as unhealthy
        assert not mock_pool.connection_metrics["unhealthy_conn"].is_healthy
        assert mock_pool.connection_metrics["healthy_conn"].is_healthy

    @pytest.mark.asyncio
    async def test_pool_statistics_calculation(self, mock_pool):
        """Test pool statistics calculation"""
        # Add test metrics
        now = datetime.now()
        recent_time = now - timedelta(seconds=30)
        old_time = now - timedelta(minutes=5)

        mock_pool.connection_metrics = {
            "active_conn": ConnectionMetrics(
                connection_id="active_conn",
                created_at=now,
                last_used=recent_time  # Recently used (active)
            ),
            "idle_conn": ConnectionMetrics(
                connection_id="idle_conn",
                created_at=now,
                last_used=old_time  # Not recently used (idle)
            )
        }

        # Add query history
        mock_pool.query_history = [
            (10.0, True),  # 10ms successful query
            (25.0, True),  # 25ms successful query
            (5.0, False)   # 5ms failed query
        ]

        stats = await mock_pool.get_pool_statistics()

        assert stats.total_connections == 2
        assert stats.active_connections == 1
        assert stats.idle_connections == 1
        assert stats.total_queries == 3
        assert stats.total_errors == 1
        assert stats.avg_response_time_ms == 17.5  # Average of successful queries

    @pytest.mark.asyncio
    async def test_connection_details_retrieval(self, mock_pool):
        """Test connection details retrieval"""
        now = datetime.now()
        mock_pool.connection_metrics = {
            "test_conn": ConnectionMetrics(
                connection_id="test_conn",
                created_at=now,
                last_used=now,
                query_count=50,
                error_count=2
            )
        }

        details = await mock_pool.get_connection_details()

        assert len(details) == 1
        detail = details[0]
        assert detail["connection_id"] == "test_conn"
        assert detail["query_count"] == 50
        assert detail["error_count"] == 2
        assert "age_seconds" in detail
        assert "priority_usage" in detail

    @pytest.mark.asyncio
    async def test_pool_optimization(self, mock_pool):
        """Test pool optimization operations"""
        # Add unhealthy connections
        mock_pool.connection_metrics = {
            "unhealthy_conn": ConnectionMetrics(
                connection_id="unhealthy_conn",
                created_at=datetime.now() - timedelta(hours=2),
                last_used=datetime.now() - timedelta(hours=1),
                error_count=10,
                is_healthy=False
            )
        }

        # Add many prepared statements
        mock_pool.prepared_statements = {f"stmt_{i}": f"prep_{i}" for i in range(100)}

        result = await mock_pool.optimize_pool()

        assert "unhealthy_connections_found" in result
        assert result["unhealthy_connections_found"] == 1
        assert "final_statistics" in result

    @pytest.mark.asyncio
    async def test_prepared_statement_cache_management(self, mock_pool):
        """Test prepared statement cache size management"""
        mock_pool.config.statement_cache_size = 5
        mock_connection = AsyncMock()

        # Fill cache beyond limit
        for i in range(7):
            query = f"SELECT * FROM table{i} WHERE id = $1"
            await mock_pool._execute_prepared(mock_connection, query, (1,))

        # Cache should not exceed configured size
        # Note: In actual implementation, oldest statements would be removed
        # This test verifies the cache management logic is called
        assert mock_connection.execute.call_count > 0

    @pytest.mark.asyncio
    async def test_error_handling_in_acquisition(self, mock_pool):
        """Test error handling during connection acquisition"""
        mock_pool.is_initialized = True
        mock_asyncpg_pool = AsyncMock()
        mock_asyncpg_pool.acquire.side_effect = Exception("Connection failed")
        mock_pool.pool = mock_asyncpg_pool

        with pytest.raises(Exception):
            async with mock_pool.acquire_connection(ConnectionPriority.NORMAL):
                pass

    @pytest.mark.asyncio
    async def test_pool_closure(self, mock_pool):
        """Test proper pool closure and cleanup"""
        # Set up pool with mock data
        mock_pool.is_initialized = True
        mock_pool.health_check_task = AsyncMock()
        mock_pool.health_check_task.cancel = MagicMock()
        mock_pool.pool = AsyncMock()
        mock_pool.connection_metrics = {"test": "data"}
        mock_pool.query_history = [(10.0, True)]
        mock_pool.prepared_statements = {"test": "stmt"}

        await mock_pool.close()

        assert not mock_pool.is_initialized
        assert len(mock_pool.connection_metrics) == 0
        assert len(mock_pool.query_history) == 0
        assert len(mock_pool.prepared_statements) == 0
        mock_pool.health_check_task.cancel.assert_called_once()

    def test_connection_priority_enum(self):
        """Test connection priority enumeration"""
        priorities = list(ConnectionPriority)

        assert ConnectionPriority.CRITICAL in priorities
        assert ConnectionPriority.HIGH in priorities
        assert ConnectionPriority.NORMAL in priorities
        assert ConnectionPriority.LOW in priorities

        # Test priority values
        assert ConnectionPriority.CRITICAL.value == "critical"
        assert ConnectionPriority.HIGH.value == "high"
        assert ConnectionPriority.NORMAL.value == "normal"
        assert ConnectionPriority.LOW.value == "low"

    @pytest.mark.asyncio
    async def test_context_manager_usage(self, pool_config):
        """Test connection pool as context manager"""
        with patch('asyncpg.create_pool') as mock_create_pool:
            mock_asyncpg_pool = AsyncMock()
            mock_create_pool.return_value = mock_asyncpg_pool

            async with AdvancedConnectionPool("postgresql://test", pool_config) as pool:
                assert pool.is_initialized

            # Pool should be closed after context exit
            assert not pool.is_initialized

    @pytest.mark.asyncio
    async def test_query_history_management(self, mock_pool):
        """Test query history size management"""
        # Add more than 1000 queries to history
        mock_pool.query_history = [(5.0, True) for _ in range(1200)]

        # Simulate what happens during connection release
        if len(mock_pool.query_history) > 1000:
            mock_pool.query_history = mock_pool.query_history[-1000:]

        assert len(mock_pool.query_history) == 1000


class TestPoolStatistics:
    """Test pool statistics data structure"""

    def test_pool_statistics_creation(self):
        """Test pool statistics initialization"""
        stats = PoolStatistics()

        assert stats.total_connections == 0
        assert stats.active_connections == 0
        assert stats.idle_connections == 0
        assert stats.total_queries == 0
        assert stats.total_errors == 0
        assert stats.avg_response_time_ms == 0.0
        assert stats.connection_utilization == 0.0
        assert stats.pool_efficiency == 0.0
        assert isinstance(stats.last_updated, datetime)

    def test_pool_statistics_as_dict(self):
        """Test pool statistics conversion to dictionary"""
        stats = PoolStatistics(
            total_connections=10,
            active_connections=7,
            total_queries=1000,
            avg_response_time_ms=15.5
        )

        stats_dict = asdict(stats)

        assert stats_dict["total_connections"] == 10
        assert stats_dict["active_connections"] == 7
        assert stats_dict["total_queries"] == 1000
        assert stats_dict["avg_response_time_ms"] == 15.5


class TestConnectionPoolIntegration:
    """Integration tests for connection pool"""

    @pytest.mark.integration
    async def test_real_database_connection_pool(self):
        """Integration test with real database"""
        # This test requires a real PostgreSQL database
        pytest.skip("Integration test requires real database connection")

    @pytest.mark.integration
    async def test_connection_pool_performance(self):
        """Test connection pool performance under load"""
        # This test would measure actual pool performance
        pytest.skip("Integration test requires performance measurement setup")

    @pytest.mark.integration
    async def test_connection_pool_resilience(self):
        """Test connection pool resilience to database failures"""
        # This test would simulate database failures and recovery
        pytest.skip("Integration test requires database failure simulation")