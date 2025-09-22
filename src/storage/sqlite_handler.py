"""SQLite Storage Handler

This module provides SQLite-based storage for market data with async operations.
Handles OHLCV data, data source metadata, and quality tracking.
"""

import asyncio
import sqlite3
from datetime import datetime, date
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import logging

import aiosqlite
import pandas as pd

from ..data_sources.base import PriceData, CurrentPrice, HealthStatus, DataSourceStatus

logger = logging.getLogger(__name__)


class StorageError(Exception):
    """Base exception for storage operations"""
    pass


class DatabaseConnectionError(StorageError):
    """Raised when database connection fails"""
    pass


class DataIntegrityError(StorageError):
    """Raised when data integrity constraints are violated"""
    pass


class SQLiteHandler:
    """SQLite database handler for market data storage

    Features:
    - Async operations using aiosqlite
    - OHLCV data storage with deduplication
    - Data source health tracking
    - Symbol metadata management
    - Quality event logging
    - Efficient querying with proper indexing
    """

    def __init__(self, database_path: str):
        """Initialize SQLite handler

        Args:
            database_path: Path to SQLite database file
        """
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = None

        logger.info(f"Initialized SQLite handler: {database_path}")

    async def connect(self) -> None:
        """Establish database connection and create tables"""
        try:
            self._connection = await aiosqlite.connect(str(self.database_path))
            await self._connection.execute("PRAGMA foreign_keys = ON")
            await self._create_tables()
            await self._create_indexes()
            logger.info(f"Connected to database: {self.database_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise DatabaseConnectionError(f"Failed to connect to database: {e}")

    async def disconnect(self) -> None:
        """Close database connection"""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("Disconnected from database")

    async def _create_tables(self) -> None:
        """Create database tables if they don't exist"""

        # OHLCV data table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open_price REAL NOT NULL,
                high_price REAL NOT NULL,
                low_price REAL NOT NULL,
                close_price REAL NOT NULL,
                volume INTEGER NOT NULL,
                adjusted_close REAL,
                source TEXT NOT NULL,
                quality_score INTEGER CHECK (quality_score >= 0 AND quality_score <= 100),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp, source)
            )
        """)

        # Data sources tracking table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS data_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                status TEXT NOT NULL,
                last_successful_call DATETIME,
                error_count INTEGER DEFAULT 0,
                response_time_ms REAL,
                rate_limit_remaining INTEGER,
                message TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Symbols metadata table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE NOT NULL,
                name TEXT,
                exchange TEXT,
                sector TEXT,
                is_active BOOLEAN DEFAULT TRUE,
                first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Data quality events table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS quality_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                symbol TEXT,
                timestamp DATETIME NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await self._connection.commit()

    async def _create_indexes(self) -> None:
        """Create database indexes for performance"""

        indexes = [
            # OHLCV data indexes
            "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol ON ohlcv_data(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_ohlcv_timestamp ON ohlcv_data(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timestamp ON ohlcv_data(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_ohlcv_source ON ohlcv_data(source)",

            # Symbols indexes
            "CREATE INDEX IF NOT EXISTS idx_symbols_symbol ON symbols(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_symbols_exchange ON symbols(exchange)",

            # Quality events indexes
            "CREATE INDEX IF NOT EXISTS idx_quality_source ON quality_events(source)",
            "CREATE INDEX IF NOT EXISTS idx_quality_symbol ON quality_events(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_quality_timestamp ON quality_events(timestamp)",
        ]

        for index_sql in indexes:
            await self._connection.execute(index_sql)

        await self._connection.commit()

    async def store_price_data(self, price_data: List[PriceData]) -> int:
        """Store OHLCV price data

        Args:
            price_data: List of PriceData objects to store

        Returns:
            Number of records inserted (excluding duplicates)

        Raises:
            DataIntegrityError: If data validation fails
            StorageError: For other storage errors
        """
        if not self._connection:
            raise StorageError("Database not connected")

        if not price_data:
            return 0

        inserted_count = 0

        try:
            for data in price_data:
                # Validate data
                self._validate_price_data(data)

                # Register symbol if not exists
                await self._register_symbol(data.symbol)

                # Insert data with conflict resolution
                cursor = await self._connection.execute("""
                    INSERT OR IGNORE INTO ohlcv_data
                    (symbol, timestamp, open_price, high_price, low_price, close_price,
                     volume, adjusted_close, source, quality_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data.symbol,
                    data.timestamp,
                    data.open_price,
                    data.high_price,
                    data.low_price,
                    data.close_price,
                    data.volume,
                    data.adjusted_close,
                    data.source,
                    data.quality_score
                ))

                if cursor.rowcount > 0:
                    inserted_count += 1

            await self._connection.commit()
            logger.info(f"Stored {inserted_count} price records out of {len(price_data)} total")
            return inserted_count

        except Exception as e:
            await self._connection.rollback()
            logger.error(f"Failed to store price data: {e}")
            raise StorageError(f"Failed to store price data: {e}")

    async def get_price_data(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        source: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[PriceData]:
        """Retrieve OHLCV price data

        Args:
            symbol: Stock symbol to retrieve
            start_date: Start date filter (optional)
            end_date: End date filter (optional)
            source: Data source filter (optional)
            limit: Maximum number of records (optional)

        Returns:
            List of PriceData objects sorted by timestamp

        Raises:
            StorageError: For database errors
        """
        if not self._connection:
            raise StorageError("Database not connected")

        try:
            # Build query dynamically
            query = "SELECT * FROM ohlcv_data WHERE symbol = ?"
            params = [symbol]

            if start_date:
                query += " AND date(timestamp) >= ?"
                params.append(start_date.isoformat())

            if end_date:
                query += " AND date(timestamp) <= ?"
                params.append(end_date.isoformat())

            if source:
                query += " AND source = ?"
                params.append(source)

            query += " ORDER BY timestamp ASC"

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor = await self._connection.execute(query, params)
            rows = await cursor.fetchall()

            # Convert to PriceData objects
            price_data = []
            for row in rows:
                price_data.append(PriceData(
                    symbol=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    open_price=row[3],
                    high_price=row[4],
                    low_price=row[5],
                    close_price=row[6],
                    volume=row[7],
                    adjusted_close=row[8],
                    source=row[9],
                    quality_score=row[10]
                ))

            logger.debug(f"Retrieved {len(price_data)} price records for {symbol}")
            return price_data

        except Exception as e:
            logger.error(f"Failed to retrieve price data for {symbol}: {e}")
            raise StorageError(f"Failed to retrieve price data: {e}")

    async def get_latest_price(self, symbol: str, source: Optional[str] = None) -> Optional[PriceData]:
        """Get the most recent price data for a symbol

        Args:
            symbol: Stock symbol
            source: Data source filter (optional)

        Returns:
            Latest PriceData object or None if not found
        """
        try:
            query = "SELECT * FROM ohlcv_data WHERE symbol = ?"
            params = [symbol]

            if source:
                query += " AND source = ?"
                params.append(source)

            query += " ORDER BY timestamp DESC LIMIT 1"

            cursor = await self._connection.execute(query, params)
            row = await cursor.fetchone()

            if row:
                return PriceData(
                    symbol=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    open_price=row[3],
                    high_price=row[4],
                    low_price=row[5],
                    close_price=row[6],
                    volume=row[7],
                    adjusted_close=row[8],
                    source=row[9],
                    quality_score=row[10]
                )

            return None

        except Exception as e:
            logger.error(f"Failed to get latest price for {symbol}: {e}")
            raise StorageError(f"Failed to get latest price: {e}")

    async def update_data_source_status(self, source_name: str, status: HealthStatus) -> None:
        """Update data source health status

        Args:
            source_name: Name of the data source
            status: HealthStatus object with current status
        """
        if not self._connection:
            raise StorageError("Database not connected")

        try:
            await self._connection.execute("""
                INSERT OR REPLACE INTO data_sources
                (name, status, last_successful_call, error_count, response_time_ms,
                 rate_limit_remaining, message, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                source_name,
                status.status.value,
                status.last_successful_call,
                status.error_count,
                status.response_time_ms,
                status.rate_limit_remaining,
                status.message
            ))

            await self._connection.commit()
            logger.debug(f"Updated status for data source: {source_name}")

        except Exception as e:
            logger.error(f"Failed to update data source status: {e}")
            raise StorageError(f"Failed to update data source status: {e}")

    async def get_data_source_status(self, source_name: str) -> Optional[Dict[str, Any]]:
        """Get data source health status

        Args:
            source_name: Name of the data source

        Returns:
            Dictionary with status information or None if not found
        """
        if not self._connection:
            raise StorageError("Database not connected")

        try:
            cursor = await self._connection.execute(
                "SELECT * FROM data_sources WHERE name = ?",
                (source_name,)
            )
            row = await cursor.fetchone()

            if row:
                return {
                    "name": row[1],
                    "status": row[2],
                    "last_successful_call": row[3],
                    "error_count": row[4],
                    "response_time_ms": row[5],
                    "rate_limit_remaining": row[6],
                    "message": row[7],
                    "updated_at": row[8]
                }

            return None

        except Exception as e:
            logger.error(f"Failed to get data source status: {e}")
            raise StorageError(f"Failed to get data source status: {e}")

    async def log_quality_event(
        self,
        source: str,
        event_type: str,
        severity: str,
        description: str,
        symbol: Optional[str] = None
    ) -> None:
        """Log a data quality event

        Args:
            source: Data source name
            event_type: Type of event (e.g., 'missing_data', 'outlier')
            severity: Severity level ('low', 'medium', 'high', 'critical')
            description: Event description
            symbol: Related symbol (optional)
        """
        if not self._connection:
            raise StorageError("Database not connected")

        try:
            await self._connection.execute("""
                INSERT INTO quality_events
                (source, symbol, timestamp, event_type, severity, description)
                VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?, ?)
            """, (source, symbol, event_type, severity, description))

            await self._connection.commit()
            logger.info(f"Logged quality event: {event_type} ({severity}) for {source}")

        except Exception as e:
            logger.error(f"Failed to log quality event: {e}")
            raise StorageError(f"Failed to log quality event: {e}")

    async def get_symbols(self, active_only: bool = True) -> List[str]:
        """Get list of symbols in the database

        Args:
            active_only: Return only active symbols

        Returns:
            List of symbol strings
        """
        if not self._connection:
            raise StorageError("Database not connected")

        try:
            query = "SELECT DISTINCT symbol FROM symbols"
            if active_only:
                query += " WHERE is_active = TRUE"
            query += " ORDER BY symbol"

            cursor = await self._connection.execute(query)
            rows = await cursor.fetchall()

            return [row[0] for row in rows]

        except Exception as e:
            logger.error(f"Failed to get symbols: {e}")
            raise StorageError(f"Failed to get symbols: {e}")

    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics

        Returns:
            Dictionary with database statistics
        """
        if not self._connection:
            raise StorageError("Database not connected")

        try:
            stats = {}

            # Count records in each table
            tables = ["ohlcv_data", "data_sources", "symbols", "quality_events"]
            for table in tables:
                cursor = await self._connection.execute(f"SELECT COUNT(*) FROM {table}")
                count = await cursor.fetchone()
                stats[f"{table}_count"] = count[0]

            # Get date range of price data
            cursor = await self._connection.execute("""
                SELECT MIN(date(timestamp)), MAX(date(timestamp)) FROM ohlcv_data
            """)
            date_range = await cursor.fetchone()
            stats["price_data_date_range"] = {
                "start": date_range[0],
                "end": date_range[1]
            }

            # Get unique symbols count
            cursor = await self._connection.execute("SELECT COUNT(DISTINCT symbol) FROM ohlcv_data")
            unique_symbols = await cursor.fetchone()
            stats["unique_symbols"] = unique_symbols[0]

            # Get database file size
            stats["database_size_bytes"] = self.database_path.stat().st_size

            return stats

        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            raise StorageError(f"Failed to get database stats: {e}")

    async def _register_symbol(self, symbol: str) -> None:
        """Register a symbol in the symbols table

        Args:
            symbol: Symbol to register
        """
        await self._connection.execute("""
            INSERT OR IGNORE INTO symbols (symbol)
            VALUES (?)
        """, (symbol,))

    def _validate_price_data(self, data: PriceData) -> None:
        """Validate price data before storage

        Args:
            data: PriceData object to validate

        Raises:
            DataIntegrityError: If data is invalid
        """
        if data.open_price <= 0:
            raise DataIntegrityError(f"Invalid open price: {data.open_price}")

        if data.high_price <= 0:
            raise DataIntegrityError(f"Invalid high price: {data.high_price}")

        if data.low_price <= 0:
            raise DataIntegrityError(f"Invalid low price: {data.low_price}")

        if data.close_price <= 0:
            raise DataIntegrityError(f"Invalid close price: {data.close_price}")

        if data.volume < 0:
            raise DataIntegrityError(f"Invalid volume: {data.volume}")

        if data.high_price < data.low_price:
            raise DataIntegrityError(f"High price ({data.high_price}) less than low price ({data.low_price})")

        if not (data.low_price <= data.open_price <= data.high_price):
            raise DataIntegrityError(f"Open price ({data.open_price}) outside high-low range")

        if not (data.low_price <= data.close_price <= data.high_price):
            raise DataIntegrityError(f"Close price ({data.close_price}) outside high-low range")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()