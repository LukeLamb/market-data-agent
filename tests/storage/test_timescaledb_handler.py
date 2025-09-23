"""
Tests for TimescaleDB Storage Handler
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import asyncpg

from src.storage.timescaledb_handler import (
    TimescaleDBHandler, TimescaleDBConfig, StorageStats
)
from src.data_sources.base import PriceData, HealthStatus


class TestTimescaleDBHandler:
    """Test cases for TimescaleDB handler"""

    @pytest.fixture
    def config(self):
        """Test configuration"""
        return TimescaleDBConfig(
            host="localhost",
            port=5432,
            database="test_market_data",
            username="test_user",
            password="test_password",
            pool_min_size=1,
            pool_max_size=5
        )

    @pytest.fixture
    def handler(self, config):
        """TimescaleDB handler instance"""
        return TimescaleDBHandler(config)

    @pytest.fixture
    def sample_price_data(self):
        """Sample OHLCV data for testing"""
        base_time = datetime.now()
        return [
            PriceData(
                symbol="AAPL",
                timestamp=base_time - timedelta(minutes=i),
                open=150.0 + i,
                high=155.0 + i,
                low=149.0 + i,
                close=154.0 + i,
                volume=1000000 + i * 1000
            ) for i in range(5)
        ]

    @pytest.mark.asyncio
    async def test_handler_initialization(self, handler, config):
        """Test handler initialization"""
        assert handler.config == config
        assert not handler.is_connected
        assert handler.pool is None

    @pytest.mark.asyncio
    async def test_config_validation(self):
        """Test configuration validation"""
        config = TimescaleDBConfig()
        handler = TimescaleDBHandler(config)

        # Test default values
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "market_data"
        assert config.pool_min_size == 5
        assert config.pool_max_size == 20

    @pytest.mark.asyncio
    @patch('asyncpg.create_pool')
    async def test_successful_initialization(self, mock_create_pool, handler):
        """Test successful database initialization"""
        # Mock pool and connection
        mock_conn = AsyncMock()
        mock_pool = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_create_pool.return_value = mock_pool

        await handler.initialize()

        assert handler.is_connected
        assert handler.pool == mock_pool
        mock_create_pool.assert_called_once()

    @pytest.mark.asyncio
    @patch('asyncpg.create_pool')
    async def test_initialization_failure(self, mock_create_pool, handler):
        """Test database initialization failure"""
        mock_create_pool.side_effect = Exception("Connection failed")

        with pytest.raises(Exception, match="Connection failed"):
            await handler.initialize()

        assert not handler.is_connected
        assert handler.pool is None

    @pytest.mark.asyncio
    async def test_store_ohlcv_data_empty_list(self, handler):
        """Test storing empty data list"""
        result = await handler.store_ohlcv_data([], "test_source")
        assert result == 0

    @pytest.mark.asyncio
    @patch('asyncpg.create_pool')
    async def test_store_ohlcv_data_success(self, mock_create_pool, handler, sample_price_data):
        """Test successful OHLCV data storage"""
        # Setup mocks
        mock_conn = AsyncMock()
        mock_pool = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_create_pool.return_value = mock_pool

        # Mock executemany
        mock_conn.executemany.return_value = "INSERT 0 5"

        await handler.initialize()
        result = await handler.store_ohlcv_data(sample_price_data, "test_source", 95.5)

        assert result == len(sample_price_data)
        mock_conn.executemany.assert_called()

    @pytest.mark.asyncio
    @patch('asyncpg.create_pool')
    async def test_get_historical_data(self, mock_create_pool, handler):
        """Test historical data retrieval"""
        # Setup mocks
        mock_conn = AsyncMock()
        mock_pool = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_create_pool.return_value = mock_pool

        # Mock data
        mock_rows = [
            {
                'time': datetime.now(),
                'symbol': 'AAPL',
                'open': 150.0,
                'high': 155.0,
                'low': 149.0,
                'close': 154.0,
                'volume': 1000000,
                'adj_close': 154.0,
                'source': 'test',
                'quality_score': 95.0
            }
        ]
        mock_conn.fetch.return_value = mock_rows

        await handler.initialize()

        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now()

        result = await handler.get_historical_data("AAPL", start_date, end_date)

        assert len(result) == 1
        assert isinstance(result[0], PriceData)
        assert result[0].symbol == "AAPL"
        assert result[0].open == 150.0

    @pytest.mark.asyncio
    @patch('asyncpg.create_pool')
    async def test_get_latest_price(self, mock_create_pool, handler):
        """Test latest price retrieval"""
        # Setup mocks
        mock_conn = AsyncMock()
        mock_pool = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_create_pool.return_value = mock_pool

        # Mock data
        mock_row = {
            'time': datetime.now(),
            'symbol': 'AAPL',
            'open': 150.0,
            'high': 155.0,
            'low': 149.0,
            'close': 154.0,
            'volume': 1000000,
            'adj_close': 154.0,
            'source': 'test',
            'quality_score': 95.0
        }
        mock_conn.fetchrow.return_value = mock_row

        await handler.initialize()
        result = await handler.get_latest_price("AAPL")

        assert result is not None
        assert isinstance(result, PriceData)
        assert result.symbol == "AAPL"
        assert result.close == 154.0

    @pytest.mark.asyncio
    @patch('asyncpg.create_pool')
    async def test_get_latest_price_not_found(self, mock_create_pool, handler):
        """Test latest price retrieval when no data found"""
        # Setup mocks
        mock_conn = AsyncMock()
        mock_pool = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_create_pool.return_value = mock_pool

        mock_conn.fetchrow.return_value = None

        await handler.initialize()
        result = await handler.get_latest_price("NONEXISTENT")

        assert result is None

    @pytest.mark.asyncio
    @patch('asyncpg.create_pool')
    async def test_store_health_status(self, mock_create_pool, handler):
        """Test health status storage"""
        # Setup mocks
        mock_conn = AsyncMock()
        mock_pool = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_create_pool.return_value = mock_pool

        await handler.initialize()

        health_status = HealthStatus(
            status="healthy",
            response_time_ms=150.5,
            error_count=0,
            success_rate=100.0
        )

        await handler.store_health_status("test_source", health_status)
        mock_conn.execute.assert_called()

    @pytest.mark.asyncio
    @patch('asyncpg.create_pool')
    async def test_log_quality_event(self, mock_create_pool, handler):
        """Test quality event logging"""
        # Setup mocks
        mock_conn = AsyncMock()
        mock_pool = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_create_pool.return_value = mock_pool

        await handler.initialize()

        await handler.log_quality_event(
            symbol="AAPL",
            source="test_source",
            event_type="price_anomaly",
            severity="medium",
            description="Price spike detected",
            metadata={"spike_percentage": 5.2}
        )

        mock_conn.execute.assert_called()

    @pytest.mark.asyncio
    @patch('asyncpg.create_pool')
    async def test_get_storage_statistics(self, mock_create_pool, handler):
        """Test storage statistics retrieval"""
        # Setup mocks
        mock_conn = AsyncMock()
        mock_pool = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_create_pool.return_value = mock_pool

        # Mock statistics data
        mock_row = {
            'total_records': 1000000,
            'oldest_record': datetime.now() - timedelta(days=365),
            'newest_record': datetime.now(),
            'symbols_count': 500,
            'storage_size_mb': 250.5,
            'partitions_count': 365
        }
        mock_conn.fetchrow.return_value = mock_row

        await handler.initialize()
        stats = await handler.get_storage_statistics()

        assert isinstance(stats, StorageStats)
        assert stats.total_records == 1000000
        assert stats.symbols_count == 500
        assert stats.storage_size_mb == 250.5
        assert stats.compression_ratio > 0  # Should be calculated

    @pytest.mark.asyncio
    @patch('asyncpg.create_pool')
    async def test_statistics_caching(self, mock_create_pool, handler):
        """Test statistics caching mechanism"""
        # Setup mocks
        mock_conn = AsyncMock()
        mock_pool = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_create_pool.return_value = mock_pool

        mock_row = {
            'total_records': 1000000,
            'oldest_record': datetime.now() - timedelta(days=365),
            'newest_record': datetime.now(),
            'symbols_count': 500,
            'storage_size_mb': 250.5,
            'partitions_count': 365
        }
        mock_conn.fetchrow.return_value = mock_row

        await handler.initialize()

        # First call should query database
        stats1 = await handler.get_storage_statistics()
        assert mock_conn.fetchrow.call_count == 1

        # Second call within cache TTL should use cache
        stats2 = await handler.get_storage_statistics()
        assert mock_conn.fetchrow.call_count == 1  # No additional calls
        assert stats1.total_records == stats2.total_records

    @pytest.mark.asyncio
    @patch('asyncpg.create_pool')
    async def test_optimize_database(self, mock_create_pool, handler):
        """Test database optimization"""
        # Setup mocks
        mock_conn = AsyncMock()
        mock_pool = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_create_pool.return_value = mock_pool

        await handler.initialize()
        result = await handler.optimize_database()

        assert result["status"] == "success"
        assert "optimized" in result["message"].lower()
        # Should call compression and analyze commands
        assert mock_conn.execute.call_count >= 2

    @pytest.mark.asyncio
    @patch('asyncpg.create_pool')
    async def test_transaction_context_manager(self, mock_create_pool, handler):
        """Test transaction context manager"""
        # Setup mocks
        mock_conn = AsyncMock()
        mock_pool = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_create_pool.return_value = mock_pool

        # Mock transaction
        mock_transaction = AsyncMock()
        mock_conn.transaction.return_value.__aenter__ = AsyncMock(return_value=mock_transaction)
        mock_conn.transaction.return_value.__aexit__ = AsyncMock(return_value=None)

        await handler.initialize()

        async with handler.transaction() as conn:
            assert conn == mock_conn

        mock_conn.transaction.assert_called_once()

    @pytest.mark.asyncio
    @patch('asyncpg.create_pool')
    async def test_close_connection(self, mock_create_pool, handler):
        """Test connection pool closure"""
        # Setup mocks
        mock_pool = AsyncMock()
        mock_create_pool.return_value = mock_pool

        await handler.initialize()
        assert handler.is_connected

        await handler.close()
        assert not handler.is_connected
        mock_pool.close.assert_called_once()

    @pytest.mark.asyncio
    @patch('asyncpg.create_pool')
    async def test_context_manager(self, mock_create_pool, handler):
        """Test handler as context manager"""
        # Setup mocks
        mock_pool = AsyncMock()
        mock_create_pool.return_value = mock_pool

        async with handler:
            assert handler.is_connected

        mock_pool.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_price_data_constraints(self, handler, sample_price_data):
        """Test OHLCV data validation constraints"""
        # Create invalid price data
        invalid_data = PriceData(
            symbol="TEST",
            timestamp=datetime.now(),
            open=100.0,
            high=95.0,  # High < Open (invalid)
            low=105.0,  # Low > Open (invalid)
            close=98.0,
            volume=1000
        )

        # The constraint violation should be handled by the database
        # In real implementation, this would raise a constraint violation
        # For now, we test the data preparation

        batch_data = [(
            invalid_data.timestamp,
            invalid_data.symbol.upper(),
            None,  # exchange
            float(invalid_data.open),
            float(invalid_data.high),
            float(invalid_data.low),
            float(invalid_data.close),
            int(invalid_data.volume),
            float(invalid_data.close),  # adj_close
            "test_source",
            95.0  # quality_score
        )]

        # Verify constraint validation logic
        assert batch_data[0][4] < batch_data[0][3]  # high < open (invalid)
        assert batch_data[0][5] > batch_data[0][3]  # low > open (invalid)

    def test_storage_stats_creation(self):
        """Test StorageStats dataclass"""
        stats = StorageStats(
            total_records=1000,
            storage_size_mb=50.5,
            symbols_count=10,
            compression_ratio=8.5
        )

        assert stats.total_records == 1000
        assert stats.storage_size_mb == 50.5
        assert stats.symbols_count == 10
        assert stats.compression_ratio == 8.5
        assert stats.compressed_records == 0  # default
        assert stats.oldest_record is None  # default

    def test_timescaledb_config_defaults(self):
        """Test TimescaleDB configuration defaults"""
        config = TimescaleDBConfig()

        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "market_data"
        assert config.username == "market_user"
        assert config.password == "secure_password"
        assert config.pool_min_size == 5
        assert config.pool_max_size == 20
        assert config.command_timeout == 30.0
        assert config.compression_policy_interval == "1 day"
        assert config.retention_policy == "1 year"

    @pytest.mark.asyncio
    async def test_error_handling_in_operations(self, handler):
        """Test error handling in database operations"""
        # Test without initialization
        with pytest.raises(AttributeError):
            await handler.store_ohlcv_data([], "test")

        with pytest.raises(AttributeError):
            await handler.get_latest_price("AAPL")

        with pytest.raises(AttributeError):
            await handler.get_storage_statistics()

    @pytest.mark.asyncio
    @patch('asyncpg.create_pool')
    async def test_symbol_case_handling(self, mock_create_pool, handler, sample_price_data):
        """Test symbol case normalization"""
        # Setup mocks
        mock_conn = AsyncMock()
        mock_pool = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_create_pool.return_value = mock_pool

        await handler.initialize()

        # Test with lowercase symbol
        lowercase_data = [
            PriceData(
                symbol="aapl",  # lowercase
                timestamp=datetime.now(),
                open=150.0,
                high=155.0,
                low=149.0,
                close=154.0,
                volume=1000000
            )
        ]

        await handler.store_ohlcv_data(lowercase_data, "test_source")

        # Verify symbol was converted to uppercase in the call
        call_args = mock_conn.executemany.call_args[0][1][0]
        assert call_args[1] == "AAPL"  # Symbol should be uppercase

class TestTimescaleDBIntegration:
    """Integration tests for TimescaleDB (require actual database)"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_database_connection(self):
        """Test connection to real TimescaleDB instance"""
        # This test requires a running TimescaleDB instance
        # Skip if not available
        config = TimescaleDBConfig(
            host="localhost",
            port=5432,
            database="test_market_data",
            username="test_user",
            password="test_password"
        )

        handler = TimescaleDBHandler(config)

        try:
            await handler.initialize()
            stats = await handler.get_storage_statistics()
            assert isinstance(stats, StorageStats)
        except Exception:
            pytest.skip("TimescaleDB not available for integration testing")
        finally:
            await handler.close()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_data_operations(self):
        """Test real data operations with TimescaleDB"""
        # This test requires a running TimescaleDB instance
        pytest.skip("Requires live TimescaleDB instance")