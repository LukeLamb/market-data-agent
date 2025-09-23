"""
TimescaleDB Storage Handler

Enterprise-grade time-series database handler for high-performance
OHLCV data storage with automatic partitioning and compression.
"""

import asyncio
import asyncpg
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import json

from ..data_sources.base import PriceData, HealthStatus

logger = logging.getLogger(__name__)


@dataclass
class TimescaleDBConfig:
    """TimescaleDB configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "market_data"
    username: str = "market_user"
    password: str = "secure_password"
    pool_min_size: int = 5
    pool_max_size: int = 20
    command_timeout: float = 30.0
    compression_policy_interval: str = "1 day"
    retention_policy: str = "1 year"


@dataclass
class StorageStats:
    """Database storage statistics"""
    total_records: int = 0
    compressed_records: int = 0
    storage_size_mb: float = 0.0
    compression_ratio: float = 0.0
    oldest_record: Optional[datetime] = None
    newest_record: Optional[datetime] = None
    symbols_count: int = 0
    partitions_count: int = 0


class TimescaleDBHandler:
    """High-performance TimescaleDB storage handler"""

    def __init__(self, config: TimescaleDBConfig = None):
        self.config = config or TimescaleDBConfig()
        self.pool = None
        self.is_connected = False
        self._stats_cache = None
        self._stats_cache_time = None
        self._cache_ttl = 300  # 5 minutes

    async def initialize(self) -> None:
        """Initialize database connection pool and schema"""
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                min_size=self.config.pool_min_size,
                max_size=self.config.pool_max_size,
                command_timeout=self.config.command_timeout
            )

            # Initialize schema
            await self._initialize_schema()

            # Set up compression policies
            await self._setup_compression_policies()

            # Set up retention policies
            await self._setup_retention_policies()

            self.is_connected = True
            logger.info("TimescaleDB connection pool initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize TimescaleDB: {e}")
            raise

    async def _initialize_schema(self) -> None:
        """Create database schema and hypertables"""
        schema_sql = """
        -- Create TimescaleDB extension if not exists
        CREATE EXTENSION IF NOT EXISTS timescaledb;

        -- OHLCV data hypertable
        CREATE TABLE IF NOT EXISTS ohlcv_data (
            time TIMESTAMPTZ NOT NULL,
            symbol TEXT NOT NULL,
            exchange TEXT,
            open NUMERIC(12,4),
            high NUMERIC(12,4),
            low NUMERIC(12,4),
            close NUMERIC(12,4),
            volume BIGINT,
            adj_close NUMERIC(12,4),
            source TEXT NOT NULL,
            quality_score NUMERIC(5,2) DEFAULT 0.0,
            created_at TIMESTAMPTZ DEFAULT NOW(),

            -- Constraints
            CONSTRAINT ohlcv_data_pkey PRIMARY KEY (time, symbol, source),
            CONSTRAINT ohlcv_data_prices_check CHECK (
                open > 0 AND high > 0 AND low > 0 AND close > 0 AND
                high >= low AND high >= open AND high >= close AND
                low <= open AND low <= close
            ),
            CONSTRAINT ohlcv_data_volume_check CHECK (volume >= 0)
        );

        -- Create hypertable if not already done
        SELECT create_hypertable(
            'ohlcv_data', 'time',
            chunk_time_interval => INTERVAL '1 day',
            if_not_exists => TRUE
        );

        -- Symbols metadata table
        CREATE TABLE IF NOT EXISTS symbols (
            symbol TEXT PRIMARY KEY,
            name TEXT,
            exchange TEXT,
            sector TEXT,
            industry TEXT,
            market_cap BIGINT,
            is_active BOOLEAN DEFAULT TRUE,
            first_seen TIMESTAMPTZ DEFAULT NOW(),
            last_updated TIMESTAMPTZ DEFAULT NOW()
        );

        -- Data sources health table
        CREATE TABLE IF NOT EXISTS data_sources_health (
            time TIMESTAMPTZ NOT NULL,
            source_name TEXT NOT NULL,
            status TEXT NOT NULL,
            response_time_ms NUMERIC(8,2),
            error_count INTEGER DEFAULT 0,
            success_rate NUMERIC(5,2),
            created_at TIMESTAMPTZ DEFAULT NOW(),

            CONSTRAINT sources_health_pkey PRIMARY KEY (time, source_name)
        );

        SELECT create_hypertable(
            'data_sources_health', 'time',
            chunk_time_interval => INTERVAL '1 hour',
            if_not_exists => TRUE
        );

        -- Quality events table
        CREATE TABLE IF NOT EXISTS quality_events (
            time TIMESTAMPTZ NOT NULL,
            symbol TEXT NOT NULL,
            source TEXT NOT NULL,
            event_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            description TEXT,
            metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),

            CONSTRAINT quality_events_pkey PRIMARY KEY (time, symbol, source, event_type)
        );

        SELECT create_hypertable(
            'quality_events', 'time',
            chunk_time_interval => INTERVAL '1 hour',
            if_not_exists => TRUE
        );

        -- Create optimized indexes
        CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time
        ON ohlcv_data (symbol, time DESC);

        CREATE INDEX IF NOT EXISTS idx_ohlcv_exchange_time
        ON ohlcv_data (exchange, time DESC);

        CREATE INDEX IF NOT EXISTS idx_ohlcv_source_time
        ON ohlcv_data (source, time DESC);

        CREATE INDEX IF NOT EXISTS idx_ohlcv_quality
        ON ohlcv_data (quality_score DESC, time DESC);

        CREATE INDEX IF NOT EXISTS idx_symbols_exchange
        ON symbols (exchange, is_active);

        CREATE INDEX IF NOT EXISTS idx_quality_events_severity
        ON quality_events (severity, time DESC);
        """

        async with self.pool.acquire() as conn:
            await conn.execute(schema_sql)
            logger.info("TimescaleDB schema initialized successfully")

    async def _setup_compression_policies(self) -> None:
        """Set up automatic compression policies"""
        compression_sql = f"""
        -- Add compression policy for OHLCV data (compress after 1 day)
        SELECT add_compression_policy('ohlcv_data', INTERVAL '{self.config.compression_policy_interval}', if_not_exists => TRUE);

        -- Add compression policy for health data (compress after 1 hour)
        SELECT add_compression_policy('data_sources_health', INTERVAL '1 hour', if_not_exists => TRUE);

        -- Add compression policy for quality events (compress after 2 hours)
        SELECT add_compression_policy('quality_events', INTERVAL '2 hours', if_not_exists => TRUE);
        """

        async with self.pool.acquire() as conn:
            await conn.execute(compression_sql)
            logger.info("Compression policies configured successfully")

    async def _setup_retention_policies(self) -> None:
        """Set up automatic data retention policies"""
        retention_sql = f"""
        -- Add retention policy for OHLCV data
        SELECT add_retention_policy('ohlcv_data', INTERVAL '{self.config.retention_policy}', if_not_exists => TRUE);

        -- Add retention policy for health data (keep 3 months)
        SELECT add_retention_policy('data_sources_health', INTERVAL '3 months', if_not_exists => TRUE);

        -- Add retention policy for quality events (keep 6 months)
        SELECT add_retention_policy('quality_events', INTERVAL '6 months', if_not_exists => TRUE);
        """

        async with self.pool.acquire() as conn:
            await conn.execute(retention_sql)
            logger.info("Retention policies configured successfully")

    async def store_ohlcv_data(
        self,
        data: List[PriceData],
        source: str,
        quality_score: float = 0.0
    ) -> int:
        """Store OHLCV data with high performance bulk insert"""
        if not data:
            return 0

        insert_sql = """
        INSERT INTO ohlcv_data (
            time, symbol, exchange, open, high, low, close,
            volume, adj_close, source, quality_score
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        ON CONFLICT (time, symbol, source)
        DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            adj_close = EXCLUDED.adj_close,
            quality_score = EXCLUDED.quality_score,
            created_at = NOW()
        """

        # Prepare batch data
        batch_data = []
        for price_data in data:
            batch_data.append((
                price_data.timestamp,
                price_data.symbol.upper(),
                getattr(price_data, 'exchange', None),
                float(price_data.open),
                float(price_data.high),
                float(price_data.low),
                float(price_data.close),
                int(price_data.volume),
                float(getattr(price_data, 'adj_close', price_data.close)),
                source,
                quality_score
            ))

        try:
            async with self.pool.acquire() as conn:
                result = await conn.executemany(insert_sql, batch_data)

            # Update symbol metadata
            await self._update_symbol_metadata([d.symbol for d in data])

            # Clear stats cache
            self._stats_cache = None

            logger.info(f"Stored {len(batch_data)} OHLCV records from {source}")
            return len(batch_data)

        except Exception as e:
            logger.error(f"Failed to store OHLCV data: {e}")
            raise

    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        source: Optional[str] = None
    ) -> List[PriceData]:
        """Retrieve historical OHLCV data with optimal query performance"""
        base_query = """
        SELECT time, symbol, open, high, low, close, volume, adj_close, source, quality_score
        FROM ohlcv_data
        WHERE symbol = $1 AND time >= $2 AND time <= $3
        """

        params = [symbol.upper(), start_date, end_date]

        if source:
            base_query += " AND source = $4"
            params.append(source)

        base_query += " ORDER BY time ASC"

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(base_query, *params)

            return [
                PriceData(
                    symbol=row['symbol'],
                    timestamp=row['time'],
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=int(row['volume'])
                ) for row in rows
            ]

        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            raise

    async def get_latest_price(self, symbol: str, source: Optional[str] = None) -> Optional[PriceData]:
        """Get the latest price data for a symbol"""
        base_query = """
        SELECT time, symbol, open, high, low, close, volume, adj_close, source, quality_score
        FROM ohlcv_data
        WHERE symbol = $1
        """

        params = [symbol.upper()]

        if source:
            base_query += " AND source = $2"
            params.append(source)

        base_query += " ORDER BY time DESC LIMIT 1"

        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(base_query, *params)

            if row:
                return PriceData(
                    symbol=row['symbol'],
                    timestamp=row['time'],
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=int(row['volume'])
                )

            return None

        except Exception as e:
            logger.error(f"Failed to get latest price for {symbol}: {e}")
            raise

    async def _update_symbol_metadata(self, symbols: List[str]) -> None:
        """Update symbol metadata table"""
        insert_sql = """
        INSERT INTO symbols (symbol, first_seen, last_updated)
        VALUES ($1, NOW(), NOW())
        ON CONFLICT (symbol)
        DO UPDATE SET last_updated = NOW()
        """

        try:
            async with self.pool.acquire() as conn:
                await conn.executemany(insert_sql, [(s.upper(),) for s in set(symbols)])
        except Exception as e:
            logger.error(f"Failed to update symbol metadata: {e}")

    async def store_health_status(self, source_name: str, status: HealthStatus) -> None:
        """Store data source health information"""
        insert_sql = """
        INSERT INTO data_sources_health (
            time, source_name, status, response_time_ms, error_count, success_rate
        ) VALUES (NOW(), $1, $2, $3, $4, $5)
        """

        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    insert_sql,
                    source_name,
                    status.status,
                    status.response_time_ms,
                    status.error_count,
                    status.success_rate
                )
        except Exception as e:
            logger.error(f"Failed to store health status for {source_name}: {e}")

    async def log_quality_event(
        self,
        symbol: str,
        source: str,
        event_type: str,
        severity: str,
        description: str,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Log quality events for monitoring and analysis"""
        insert_sql = """
        INSERT INTO quality_events (
            time, symbol, source, event_type, severity, description, metadata
        ) VALUES (NOW(), $1, $2, $3, $4, $5, $6)
        """

        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    insert_sql,
                    symbol.upper(),
                    source,
                    event_type,
                    severity,
                    description,
                    json.dumps(metadata) if metadata else None
                )
        except Exception as e:
            logger.error(f"Failed to log quality event: {e}")

    async def get_storage_statistics(self) -> StorageStats:
        """Get comprehensive storage statistics with caching"""
        now = datetime.now()

        # Check cache
        if (self._stats_cache and self._stats_cache_time and
            (now - self._stats_cache_time).seconds < self._cache_ttl):
            return self._stats_cache

        stats_query = """
        WITH ohlcv_stats AS (
            SELECT
                COUNT(*) as total_records,
                MIN(time) as oldest_record,
                MAX(time) as newest_record,
                COUNT(DISTINCT symbol) as symbols_count
            FROM ohlcv_data
        ),
        size_stats AS (
            SELECT
                pg_size_pretty(pg_total_relation_size('ohlcv_data')) as table_size,
                pg_total_relation_size('ohlcv_data') as size_bytes
            FROM pg_tables WHERE tablename = 'ohlcv_data'
        ),
        chunk_stats AS (
            SELECT COUNT(*) as partitions_count
            FROM timescaledb_information.chunks
            WHERE hypertable_name = 'ohlcv_data'
        )
        SELECT
            o.total_records,
            o.oldest_record,
            o.newest_record,
            o.symbols_count,
            s.size_bytes::FLOAT / (1024*1024) as storage_size_mb,
            c.partitions_count
        FROM ohlcv_stats o, size_stats s, chunk_stats c
        """

        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(stats_query)

            if row:
                stats = StorageStats(
                    total_records=row['total_records'],
                    storage_size_mb=float(row['storage_size_mb']),
                    oldest_record=row['oldest_record'],
                    newest_record=row['newest_record'],
                    symbols_count=row['symbols_count'],
                    partitions_count=row['partitions_count']
                )

                # Calculate compression ratio estimate
                if stats.total_records > 0:
                    # Estimate uncompressed size (assuming ~80 bytes per record)
                    estimated_uncompressed_mb = (stats.total_records * 80) / (1024 * 1024)
                    if stats.storage_size_mb > 0:
                        stats.compression_ratio = estimated_uncompressed_mb / stats.storage_size_mb

                # Cache results
                self._stats_cache = stats
                self._stats_cache_time = now

                return stats

        except Exception as e:
            logger.error(f"Failed to get storage statistics: {e}")
            return StorageStats()

    async def optimize_database(self) -> Dict[str, Any]:
        """Perform comprehensive database optimization operations"""
        try:
            async with self.pool.acquire() as conn:
                # Manual compression trigger for recent data
                await conn.execute("CALL run_job((SELECT job_id FROM timescaledb_information.jobs WHERE proc_name = 'policy_compression'));")

                # Update table statistics for query planner
                await conn.execute("ANALYZE ohlcv_data;")
                await conn.execute("ANALYZE data_sources_health;")
                await conn.execute("ANALYZE quality_events;")

                # Optimize hypertable chunks
                await conn.execute("SELECT reorder_chunk(c, 'idx_ohlcv_symbol_time') FROM show_chunks('ohlcv_data') c;")

                # Vacuum analyze for maintenance
                await conn.execute("VACUUM ANALYZE ohlcv_data;")

            logger.info("Database optimization completed")
            return {"status": "success", "message": "Database optimized successfully"}

        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            return {"status": "error", "message": str(e)}

    async def create_advanced_indexes(self) -> Dict[str, Any]:
        """Create advanced performance indexes for query optimization"""
        advanced_indexes = [
            # Partial index for recent high-volume data (hot data path)
            """CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ohlcv_recent_volume
               ON ohlcv_data (symbol, time DESC, volume DESC)
               WHERE time > NOW() - INTERVAL '7 days' AND volume > 1000""",

            # Expression index for price volatility analysis
            """CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ohlcv_volatility
               ON ohlcv_data (symbol, ((high - low) / NULLIF(low, 0) * 100))
               WHERE high > low AND low > 0""",

            # Composite index for OHLC pattern matching
            """CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ohlcv_patterns
               ON ohlcv_data (symbol, time DESC, open, high, low, close)
               WHERE ABS(close - open) / NULLIF(open, 0) > 0.02""",

            # Index for aggregation queries (daily/hourly summaries)
            """CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ohlcv_time_bucket
               ON ohlcv_data (symbol, date_trunc('hour', time), time)""",

            # Covering index for dashboard queries
            """CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ohlcv_dashboard
               ON ohlcv_data (symbol, time DESC)
               INCLUDE (open, high, low, close, volume, quality_score)"""
        ]

        results = []
        try:
            async with self.pool.acquire() as conn:
                for idx_sql in advanced_indexes:
                    try:
                        await conn.execute(idx_sql)
                        results.append({"index": idx_sql.split("idx_")[1].split()[0], "status": "created"})
                    except Exception as e:
                        results.append({"index": idx_sql.split("idx_")[1].split()[0], "status": f"error: {e}"})

            logger.info(f"Advanced indexes creation completed: {len(results)} processed")
            return {"status": "success", "indexes": results}

        except Exception as e:
            logger.error(f"Advanced indexes creation failed: {e}")
            return {"status": "error", "message": str(e)}

    async def analyze_query_performance(self, query: str, params: list = None) -> Dict[str, Any]:
        """Analyze query performance and provide optimization suggestions"""
        try:
            async with self.pool.acquire() as conn:
                # Get query execution plan
                explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"
                if params:
                    result = await conn.fetchval(explain_query, *params)
                else:
                    result = await conn.fetchval(explain_query)

                plan = result[0] if result else {}

                # Extract key performance metrics
                execution_time = plan.get("Execution Time", 0)
                planning_time = plan.get("Planning Time", 0)
                total_cost = plan.get("Plan", {}).get("Total Cost", 0)

                # Provide optimization suggestions
                suggestions = []
                if execution_time > 100:  # > 100ms
                    suggestions.append("Consider adding indexes for frequently filtered columns")
                if planning_time > 10:  # > 10ms
                    suggestions.append("Query planning is slow - consider prepared statements")

                return {
                    "status": "success",
                    "execution_time_ms": execution_time,
                    "planning_time_ms": planning_time,
                    "total_cost": total_cost,
                    "plan": plan,
                    "suggestions": suggestions
                }

        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return {"status": "error", "message": str(e)}

    @asynccontextmanager
    async def transaction(self):
        """Database transaction context manager"""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                yield conn

    async def close(self) -> None:
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            self.is_connected = False
            logger.info("TimescaleDB connection pool closed")

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Global instance for easy access
timescale_handler = None

async def get_timescale_handler() -> TimescaleDBHandler:
    """Get or create global TimescaleDB handler instance"""
    global timescale_handler
    if timescale_handler is None:
        timescale_handler = TimescaleDBHandler()
        await timescale_handler.initialize()
    return timescale_handler