"""Tests for SQLite Storage Handler"""

import pytest
import tempfile
import os
from datetime import datetime, date, timedelta
from pathlib import Path
import pytest_asyncio

from src.storage.sqlite_handler import (
    SQLiteHandler,
    StorageError,
    DatabaseConnectionError,
    DataIntegrityError
)
from src.data_sources.base import PriceData, HealthStatus, DataSourceStatus


class TestSQLiteHandler:
    """Test cases for SQLite storage handler"""

    @pytest_asyncio.fixture
    async def temp_db(self):
        """Create temporary database for testing"""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_market_data.db")

        handler = SQLiteHandler(db_path)
        await handler.connect()

        yield handler

        await handler.disconnect()
        # Clean up
        if os.path.exists(db_path):
            os.remove(db_path)
        os.rmdir(temp_dir)

    @pytest.mark.asyncio
    async def test_database_initialization(self, temp_db):
        """Test database creation and table setup"""
        handler = temp_db

        # Verify database file exists
        assert handler.database_path.exists()

        # Verify connection is established
        assert handler._connection is not None

        # Test that we can execute queries on all tables
        tables = ["ohlcv_data", "data_sources", "symbols", "quality_events"]
        for table in tables:
            cursor = await handler._connection.execute(f"SELECT COUNT(*) FROM {table}")
            count = await cursor.fetchone()
            assert count[0] == 0  # Should be empty initially

    @pytest.mark.asyncio
    async def test_store_price_data_success(self, temp_db):
        """Test successful price data storage"""
        handler = temp_db

        # Create test price data
        price_data = [
            PriceData(
                symbol="AAPL",
                timestamp=datetime(2023, 1, 1, 9, 30, 0),
                open_price=150.0,
                high_price=155.0,
                low_price=149.0,
                close_price=154.0,
                volume=1000000,
                adjusted_close=154.0,
                source="test_source",
                quality_score=95
            ),
            PriceData(
                symbol="AAPL",
                timestamp=datetime(2023, 1, 2, 9, 30, 0),
                open_price=154.0,
                high_price=158.0,
                low_price=153.0,
                close_price=157.0,
                volume=1200000,
                adjusted_close=157.0,
                source="test_source",
                quality_score=95
            )
        ]

        # Store data
        inserted_count = await handler.store_price_data(price_data)
        assert inserted_count == 2

        # Verify data was stored
        retrieved_data = await handler.get_price_data("AAPL")
        assert len(retrieved_data) == 2
        assert retrieved_data[0].symbol == "AAPL"
        assert retrieved_data[0].open_price == 150.0

    @pytest.mark.asyncio
    async def test_store_price_data_duplicate_handling(self, temp_db):
        """Test duplicate data handling"""
        handler = temp_db

        price_data = PriceData(
            symbol="AAPL",
            timestamp=datetime(2023, 1, 1, 9, 30, 0),
            open_price=150.0,
            high_price=155.0,
            low_price=149.0,
            close_price=154.0,
            volume=1000000,
            source="test_source",
            quality_score=95
        )

        # Store data twice
        inserted_count1 = await handler.store_price_data([price_data])
        inserted_count2 = await handler.store_price_data([price_data])

        assert inserted_count1 == 1
        assert inserted_count2 == 0  # Should be ignored as duplicate

        # Verify only one record exists
        retrieved_data = await handler.get_price_data("AAPL")
        assert len(retrieved_data) == 1

    @pytest.mark.asyncio
    async def test_store_price_data_validation(self, temp_db):
        """Test price data validation"""
        handler = temp_db

        # Test invalid open price
        invalid_data = PriceData(
            symbol="AAPL",
            timestamp=datetime(2023, 1, 1, 9, 30, 0),
            open_price=-10.0,  # Invalid negative price
            high_price=155.0,
            low_price=149.0,
            close_price=154.0,
            volume=1000000,
            source="test_source",
            quality_score=95
        )

        with pytest.raises(StorageError):
            await handler.store_price_data([invalid_data])

    @pytest.mark.asyncio
    async def test_get_price_data_with_filters(self, temp_db):
        """Test price data retrieval with various filters"""
        handler = temp_db

        # Store test data across multiple days and sources
        price_data = [
            PriceData(
                symbol="AAPL",
                timestamp=datetime(2023, 1, 1, 9, 30, 0),
                open_price=150.0,
                high_price=155.0,
                low_price=149.0,
                close_price=154.0,
                volume=1000000,
                source="source1",
                quality_score=95
            ),
            PriceData(
                symbol="AAPL",
                timestamp=datetime(2023, 1, 2, 9, 30, 0),
                open_price=154.0,
                high_price=158.0,
                low_price=153.0,
                close_price=157.0,
                volume=1200000,
                source="source1",
                quality_score=95
            ),
            PriceData(
                symbol="AAPL",
                timestamp=datetime(2023, 1, 2, 9, 30, 0),
                open_price=154.0,
                high_price=158.0,
                low_price=153.0,
                close_price=157.0,
                volume=1200000,
                source="source2",
                quality_score=90
            )
        ]

        await handler.store_price_data(price_data)

        # Test date range filter
        data = await handler.get_price_data(
            "AAPL",
            start_date=date(2023, 1, 2),
            end_date=date(2023, 1, 2)
        )
        assert len(data) == 2  # Both sources for Jan 2

        # Test source filter
        data = await handler.get_price_data("AAPL", source="source1")
        assert len(data) == 2
        assert all(d.source == "source1" for d in data)

        # Test limit
        data = await handler.get_price_data("AAPL", limit=1)
        assert len(data) == 1

    @pytest.mark.asyncio
    async def test_get_latest_price(self, temp_db):
        """Test latest price retrieval"""
        handler = temp_db

        # Store data with different timestamps
        price_data = [
            PriceData(
                symbol="AAPL",
                timestamp=datetime(2023, 1, 1, 9, 30, 0),
                open_price=150.0,
                high_price=155.0,
                low_price=149.0,
                close_price=154.0,
                volume=1000000,
                source="test_source",
                quality_score=95
            ),
            PriceData(
                symbol="AAPL",
                timestamp=datetime(2023, 1, 2, 9, 30, 0),  # Later timestamp
                open_price=154.0,
                high_price=158.0,
                low_price=153.0,
                close_price=157.0,
                volume=1200000,
                source="test_source",
                quality_score=95
            )
        ]

        await handler.store_price_data(price_data)

        # Get latest price
        latest = await handler.get_latest_price("AAPL")
        assert latest is not None
        assert latest.close_price == 157.0  # Should be the latest entry
        assert latest.timestamp == datetime(2023, 1, 2, 9, 30, 0)

    @pytest.mark.asyncio
    async def test_data_source_status_management(self, temp_db):
        """Test data source status tracking"""
        handler = temp_db

        # Create test health status
        health_status = HealthStatus(
            status=DataSourceStatus.HEALTHY,
            last_successful_call=datetime.now(),
            error_count=0,
            response_time_ms=150.0,
            rate_limit_remaining=100,
            message="All systems operational"
        )

        # Update status
        await handler.update_data_source_status("test_source", health_status)

        # Retrieve status
        status = await handler.get_data_source_status("test_source")
        assert status is not None
        assert status["name"] == "test_source"
        assert status["status"] == "healthy"
        assert status["error_count"] == 0
        assert status["response_time_ms"] == 150.0

    @pytest.mark.asyncio
    async def test_quality_event_logging(self, temp_db):
        """Test quality event logging"""
        handler = temp_db

        # Log a quality event
        await handler.log_quality_event(
            source="test_source",
            event_type="outlier",
            severity="medium",
            description="Price jumped 50% in one minute",
            symbol="AAPL"
        )

        # Verify event was logged
        cursor = await handler._connection.execute(
            "SELECT * FROM quality_events WHERE source = ?",
            ("test_source",)
        )
        events = await cursor.fetchall()
        assert len(events) == 1
        assert events[0][1] == "test_source"  # source
        assert events[0][2] == "AAPL"  # symbol
        assert events[0][4] == "outlier"  # event_type
        assert events[0][5] == "medium"  # severity

    @pytest.mark.asyncio
    async def test_get_symbols(self, temp_db):
        """Test symbols retrieval"""
        handler = temp_db

        # Store data for multiple symbols
        price_data = [
            PriceData(
                symbol="AAPL",
                timestamp=datetime(2023, 1, 1, 9, 30, 0),
                open_price=150.0,
                high_price=155.0,
                low_price=149.0,
                close_price=154.0,
                volume=1000000,
                source="test_source",
                quality_score=95
            ),
            PriceData(
                symbol="GOOGL",
                timestamp=datetime(2023, 1, 1, 9, 30, 0),
                open_price=2800.0,
                high_price=2850.0,
                low_price=2790.0,
                close_price=2830.0,
                volume=500000,
                source="test_source",
                quality_score=95
            )
        ]

        await handler.store_price_data(price_data)

        # Get symbols
        symbols = await handler.get_symbols()
        assert "AAPL" in symbols
        assert "GOOGL" in symbols
        assert len(symbols) == 2

    @pytest.mark.asyncio
    async def test_database_stats(self, temp_db):
        """Test database statistics"""
        handler = temp_db

        # Store some test data
        price_data = PriceData(
            symbol="AAPL",
            timestamp=datetime(2023, 1, 1, 9, 30, 0),
            open_price=150.0,
            high_price=155.0,
            low_price=149.0,
            close_price=154.0,
            volume=1000000,
            source="test_source",
            quality_score=95
        )

        await handler.store_price_data([price_data])

        # Log a quality event
        await handler.log_quality_event(
            source="test_source",
            event_type="test",
            severity="low",
            description="Test event"
        )

        # Get stats
        stats = await handler.get_database_stats()

        assert stats["ohlcv_data_count"] == 1
        assert stats["symbols_count"] == 1
        assert stats["quality_events_count"] == 1
        assert stats["unique_symbols"] == 1
        assert "database_size_bytes" in stats

    @pytest.mark.asyncio
    async def test_connection_error_handling(self):
        """Test database connection error handling"""
        # Try to use handler without connecting
        handler = SQLiteHandler("/invalid/path/test.db")

        with pytest.raises(StorageError):
            await handler.get_symbols()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager functionality"""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_context.db")

        async with SQLiteHandler(db_path) as handler:
            # Should be connected
            assert handler._connection is not None

            # Test basic operation
            symbols = await handler.get_symbols()
            assert isinstance(symbols, list)

        # Should be disconnected after context exit
        assert handler._connection is None

        # Clean up
        if os.path.exists(db_path):
            os.remove(db_path)
        os.rmdir(temp_dir)

    @pytest.mark.asyncio
    async def test_data_integrity_validation(self, temp_db):
        """Test comprehensive data integrity validation"""
        handler = temp_db

        # Test high < low error
        invalid_data = PriceData(
            symbol="AAPL",
            timestamp=datetime(2023, 1, 1, 9, 30, 0),
            open_price=150.0,
            high_price=149.0,  # High less than low
            low_price=151.0,
            close_price=150.0,
            volume=1000000,
            source="test_source",
            quality_score=95
        )

        with pytest.raises(StorageError):
            await handler.store_price_data([invalid_data])

    @pytest.mark.asyncio
    async def test_empty_data_handling(self, temp_db):
        """Test handling of empty data"""
        handler = temp_db

        # Store empty list
        inserted_count = await handler.store_price_data([])
        assert inserted_count == 0

        # Get data for non-existent symbol
        data = await handler.get_price_data("NONEXISTENT")
        assert len(data) == 0

        # Get latest price for non-existent symbol
        latest = await handler.get_latest_price("NONEXISTENT")
        assert latest is None