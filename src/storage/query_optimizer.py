"""
Advanced Query Optimizer for Market Data Agent
Implements materialized views, query analysis, and performance optimization
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import time
from enum import Enum

from .timescaledb_handler import TimescaleDBHandler
from ..caching.redis_cache_manager import RedisCacheManager

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries for optimization strategies"""
    LATEST_PRICE = "latest_price"
    TIME_RANGE = "time_range"
    AGGREGATION = "aggregation"
    ANALYTICS = "analytics"
    BULK_EXPORT = "bulk_export"


@dataclass
class QueryPerformanceMetrics:
    """Query performance tracking metrics"""
    query_type: QueryType
    execution_time_ms: float
    rows_returned: int
    rows_examined: int
    cache_hit: bool
    optimization_applied: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class MaterializedViewConfig:
    """Configuration for materialized views"""
    view_name: str
    refresh_interval_minutes: int
    retention_days: int
    indexes: List[str]
    query_template: str
    dependencies: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class QueryOptimizer:
    """Advanced query optimization system with materialized views and intelligent caching"""

    def __init__(
        self,
        timescale_handler: TimescaleDBHandler,
        cache_manager: Optional[RedisCacheManager] = None
    ):
        self.timescale_handler = timescale_handler
        self.cache_manager = cache_manager
        self.performance_metrics: List[QueryPerformanceMetrics] = []
        self.materialized_views: Dict[str, MaterializedViewConfig] = {}
        self.query_cache: Dict[str, Any] = {}
        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize query optimizer with materialized views"""
        try:
            await self._create_materialized_views()
            await self._create_optimization_indexes()
            await self._setup_query_monitoring()

            self.is_initialized = True
            logger.info("Query optimizer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize query optimizer: {e}")
            raise

    async def _create_materialized_views(self) -> None:
        """Create materialized views for common aggregations"""

        materialized_views = [
            MaterializedViewConfig(
                view_name="daily_ohlcv_summary",
                refresh_interval_minutes=60,  # Refresh hourly
                retention_days=365,
                indexes=["symbol", "date"],
                query_template="""
                CREATE MATERIALIZED VIEW IF NOT EXISTS daily_ohlcv_summary AS
                SELECT
                    symbol,
                    date_trunc('day', time) as date,
                    first(open_price, time) as open_price,
                    max(high_price) as high_price,
                    min(low_price) as low_price,
                    last(close_price, time) as close_price,
                    sum(volume) as total_volume,
                    count(*) as data_points,
                    avg(quality_score) as avg_quality_score,
                    min(time) as first_timestamp,
                    max(time) as last_timestamp
                FROM ohlcv_data
                WHERE time >= NOW() - INTERVAL '1 year'
                GROUP BY symbol, date_trunc('day', time)
                ORDER BY symbol, date DESC;
                """,
                dependencies=["ohlcv_data"]
            ),

            MaterializedViewConfig(
                view_name="hourly_volume_stats",
                refresh_interval_minutes=15,  # Refresh every 15 minutes
                retention_days=90,
                indexes=["symbol", "hour"],
                query_template="""
                CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_volume_stats AS
                SELECT
                    symbol,
                    date_trunc('hour', time) as hour,
                    sum(volume) as total_volume,
                    avg(volume) as avg_volume,
                    max(volume) as max_volume,
                    count(*) as trade_count,
                    avg(high_price - low_price) as avg_spread,
                    stddev(close_price) as price_volatility
                FROM ohlcv_data
                WHERE time >= NOW() - INTERVAL '90 days'
                GROUP BY symbol, date_trunc('hour', time)
                ORDER BY symbol, hour DESC;
                """,
                dependencies=["ohlcv_data"]
            ),

            MaterializedViewConfig(
                view_name="symbol_performance_metrics",
                refresh_interval_minutes=30,  # Refresh every 30 minutes
                retention_days=30,
                indexes=["symbol", "calculation_time"],
                query_template="""
                CREATE MATERIALIZED VIEW IF NOT EXISTS symbol_performance_metrics AS
                SELECT
                    symbol,
                    NOW() as calculation_time,

                    -- Price metrics
                    last(close_price, time) as current_price,
                    first(close_price, time) as period_start_price,
                    (last(close_price, time) - first(close_price, time)) / first(close_price, time) * 100 as price_change_percent,

                    -- Volume metrics
                    sum(volume) as total_volume,
                    avg(volume) as avg_volume,

                    -- Volatility metrics
                    stddev(close_price) as price_volatility,
                    max(high_price) - min(low_price) as price_range,

                    -- Quality metrics
                    avg(quality_score) as avg_quality_score,
                    count(*) as data_points,

                    -- Time metrics
                    min(time) as period_start,
                    max(time) as period_end

                FROM ohlcv_data
                WHERE time >= NOW() - INTERVAL '24 hours'
                GROUP BY symbol;
                """,
                dependencies=["ohlcv_data"]
            ),

            MaterializedViewConfig(
                view_name="top_volume_symbols",
                refresh_interval_minutes=5,  # Refresh every 5 minutes for real-time ranking
                retention_days=7,
                indexes=["ranking_time", "volume_rank"],
                query_template="""
                CREATE MATERIALIZED VIEW IF NOT EXISTS top_volume_symbols AS
                SELECT
                    symbol,
                    NOW() as ranking_time,
                    sum(volume) as total_volume_24h,
                    avg(close_price) as avg_price_24h,
                    count(*) as updates_24h,
                    row_number() OVER (ORDER BY sum(volume) DESC) as volume_rank
                FROM ohlcv_data
                WHERE time >= NOW() - INTERVAL '24 hours'
                GROUP BY symbol
                ORDER BY total_volume_24h DESC
                LIMIT 100;
                """,
                dependencies=["ohlcv_data"]
            )
        ]

        for view_config in materialized_views:
            try:
                await self._create_single_materialized_view(view_config)
                self.materialized_views[view_config.view_name] = view_config

            except Exception as e:
                logger.error(f"Failed to create materialized view {view_config.view_name}: {e}")

        logger.info(f"Created {len(self.materialized_views)} materialized views")

    async def _create_single_materialized_view(self, config: MaterializedViewConfig) -> None:
        """Create a single materialized view with indexes"""
        async with self.timescale_handler.pool.acquire() as conn:
            # Create the materialized view
            await conn.execute(config.query_template)

            # Create indexes on the materialized view
            for index_column in config.indexes:
                index_name = f"idx_{config.view_name}_{index_column}"
                index_sql = f"""
                CREATE INDEX IF NOT EXISTS {index_name}
                ON {config.view_name} ({index_column});
                """
                await conn.execute(index_sql)

            logger.info(f"Created materialized view: {config.view_name}")

    async def _create_optimization_indexes(self) -> None:
        """Create additional performance optimization indexes"""
        optimization_indexes = [
            # Performance indexes for complex queries
            """CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ohlcv_symbol_time_volume
               ON ohlcv_data (symbol, time DESC, volume DESC)
               WHERE volume > 1000""",

            # Index for price change analysis
            """CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ohlcv_price_movement
               ON ohlcv_data (symbol, time, ((close_price - open_price) / open_price))
               WHERE ABS((close_price - open_price) / open_price) > 0.01""",

            # Index for quality filtering
            """CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ohlcv_high_quality
               ON ohlcv_data (symbol, time DESC, quality_score)
               WHERE quality_score >= 90""",

            # Composite index for analytics queries
            """CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ohlcv_analytics
               ON ohlcv_data (symbol, date_trunc('hour', time), volume, high_price, low_price)""",

            # Index for recent data with covering columns
            """CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ohlcv_recent_covering
               ON ohlcv_data (symbol, time DESC)
               INCLUDE (open_price, high_price, low_price, close_price, volume)
               WHERE time > NOW() - INTERVAL '7 days'"""
        ]

        async with self.timescale_handler.pool.acquire() as conn:
            for index_sql in optimization_indexes:
                try:
                    await conn.execute(index_sql)
                    logger.debug(f"Created optimization index")
                except Exception as e:
                    logger.warning(f"Index creation warning: {e}")

        logger.info("Optimization indexes created")

    async def _setup_query_monitoring(self) -> None:
        """Set up query performance monitoring"""
        # Enable query statistics collection
        monitoring_sql = [
            "CREATE EXTENSION IF NOT EXISTS pg_stat_statements;",
            "SELECT pg_stat_statements_reset();",  # Reset stats for clean monitoring
        ]

        async with self.timescale_handler.pool.acquire() as conn:
            for sql in monitoring_sql:
                try:
                    await conn.execute(sql)
                except Exception as e:
                    logger.warning(f"Monitoring setup warning: {e}")

        logger.info("Query monitoring enabled")

    async def refresh_materialized_views(self, force_all: bool = False) -> Dict[str, bool]:
        """Refresh materialized views based on their schedules"""
        results = {}
        current_time = datetime.now()

        for view_name, config in self.materialized_views.items():
            try:
                # Check if refresh is needed (or forced)
                should_refresh = force_all or await self._should_refresh_view(view_name, config)

                if should_refresh:
                    start_time = time.perf_counter()

                    async with self.timescale_handler.pool.acquire() as conn:
                        await conn.execute(f"REFRESH MATERIALIZED VIEW {view_name};")

                    refresh_time = (time.perf_counter() - start_time) * 1000
                    logger.info(f"Refreshed materialized view {view_name} in {refresh_time:.2f}ms")
                    results[view_name] = True
                else:
                    results[view_name] = False

            except Exception as e:
                logger.error(f"Failed to refresh materialized view {view_name}: {e}")
                results[view_name] = False

        return results

    async def _should_refresh_view(self, view_name: str, config: MaterializedViewConfig) -> bool:
        """Determine if a materialized view should be refreshed"""
        try:
            async with self.timescale_handler.pool.acquire() as conn:
                # Check last refresh time (simplified - in production would track this properly)
                result = await conn.fetchval(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = $1",
                    view_name
                )
                # For now, refresh based on interval (simplified logic)
                return True  # In production, implement proper refresh tracking

        except Exception as e:
            logger.error(f"Error checking refresh status for {view_name}: {e}")
            return False

    async def optimize_query(
        self,
        query_sql: str,
        query_type: QueryType,
        parameters: List[Any] = None
    ) -> Tuple[Any, QueryPerformanceMetrics]:
        """Execute query with optimization strategies"""
        start_time = time.perf_counter()
        cache_hit = False
        optimization_applied = None
        result = None

        try:
            # Generate cache key for cacheable queries
            cache_key = None
            if query_type in [QueryType.AGGREGATION, QueryType.ANALYTICS]:
                cache_key = self._generate_cache_key(query_sql, parameters)

                # Check cache first
                if cache_key and self.cache_manager:
                    cached_result = await self._get_cached_result(cache_key)
                    if cached_result:
                        result = cached_result
                        cache_hit = True
                        optimization_applied = "cache_hit"

            # If not cached, execute query with optimizations
            if result is None:
                result, optimization_applied = await self._execute_optimized_query(
                    query_sql, query_type, parameters
                )

                # Cache the result if appropriate
                if cache_key and self.cache_manager and query_type in [QueryType.AGGREGATION, QueryType.ANALYTICS]:
                    await self._cache_result(cache_key, result)

            # Calculate performance metrics
            execution_time = (time.perf_counter() - start_time) * 1000

            metrics = QueryPerformanceMetrics(
                query_type=query_type,
                execution_time_ms=execution_time,
                rows_returned=len(result) if isinstance(result, list) else 1,
                rows_examined=0,  # Would need EXPLAIN ANALYZE for accurate count
                cache_hit=cache_hit,
                optimization_applied=optimization_applied
            )

            # Track performance metrics
            self.performance_metrics.append(metrics)

            # Keep only recent metrics (last 1000)
            if len(self.performance_metrics) > 1000:
                self.performance_metrics = self.performance_metrics[-1000:]

            return result, metrics

        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Query optimization failed: {e}")

            # Still record metrics for failed queries
            metrics = QueryPerformanceMetrics(
                query_type=query_type,
                execution_time_ms=execution_time,
                rows_returned=0,
                rows_examined=0,
                cache_hit=False,
                optimization_applied="error"
            )
            self.performance_metrics.append(metrics)

            raise

    async def _execute_optimized_query(
        self,
        query_sql: str,
        query_type: QueryType,
        parameters: List[Any] = None
    ) -> Tuple[Any, str]:
        """Execute query with appropriate optimization strategy"""

        # Try materialized view optimization first
        optimized_query, optimization = await self._try_materialized_view_optimization(
            query_sql, query_type
        )

        if optimized_query:
            async with self.timescale_handler.pool.acquire() as conn:
                if parameters:
                    result = await conn.fetch(optimized_query, *parameters)
                else:
                    result = await conn.fetch(optimized_query)

                return [dict(row) for row in result], optimization

        # Fallback to original query with connection optimization
        async with self.timescale_handler.pool.acquire() as conn:
            if parameters:
                result = await conn.fetch(query_sql, *parameters)
            else:
                result = await conn.fetch(query_sql)

            return [dict(row) for row in result], "standard_execution"

    async def _try_materialized_view_optimization(
        self,
        query_sql: str,
        query_type: QueryType
    ) -> Tuple[Optional[str], str]:
        """Attempt to optimize query using materialized views"""

        query_lower = query_sql.lower()

        # Daily OHLC optimization
        if ("daily" in query_lower or "day" in query_lower) and "ohlcv" in query_lower:
            if "daily_ohlcv_summary" in self.materialized_views:
                # Replace complex aggregation with materialized view query
                optimized_query = query_sql.replace("ohlcv_data", "daily_ohlcv_summary")
                return optimized_query, "materialized_view_daily_ohlcv"

        # Volume statistics optimization
        if "volume" in query_lower and ("hour" in query_lower or "sum" in query_lower):
            if "hourly_volume_stats" in self.materialized_views:
                optimized_query = self._rewrite_for_volume_stats(query_sql)
                if optimized_query:
                    return optimized_query, "materialized_view_volume_stats"

        # Symbol performance optimization
        if "performance" in query_lower or ("percent" in query_lower and "change" in query_lower):
            if "symbol_performance_metrics" in self.materialized_views:
                optimized_query = self._rewrite_for_performance_metrics(query_sql)
                if optimized_query:
                    return optimized_query, "materialized_view_performance"

        return None, "no_optimization"

    def _rewrite_for_volume_stats(self, query_sql: str) -> Optional[str]:
        """Rewrite query to use hourly volume stats materialized view"""
        # Simplified rewrite logic - in production, would use a proper SQL parser
        if "sum(volume)" in query_sql.lower() and "group by" in query_sql.lower():
            return query_sql.replace("ohlcv_data", "hourly_volume_stats")
        return None

    def _rewrite_for_performance_metrics(self, query_sql: str) -> Optional[str]:
        """Rewrite query to use performance metrics materialized view"""
        # Simplified rewrite logic
        if "price_change" in query_sql.lower() or "performance" in query_sql.lower():
            return query_sql.replace("ohlcv_data", "symbol_performance_metrics")
        return None

    def _generate_cache_key(self, query_sql: str, parameters: List[Any] = None) -> str:
        """Generate cache key for query result caching"""
        import hashlib

        cache_content = query_sql
        if parameters:
            cache_content += str(parameters)

        return f"query_cache:{hashlib.md5(cache_content.encode()).hexdigest()}"

    async def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached query result"""
        if not self.cache_manager:
            return None

        try:
            cached_data = await self.cache_manager.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")

        return None

    async def _cache_result(self, cache_key: str, result: Any, ttl: int = 300) -> None:
        """Cache query result"""
        if not self.cache_manager:
            return

        try:
            await self.cache_manager.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(result, default=str)
            )
        except Exception as e:
            logger.warning(f"Cache storage error: {e}")

    async def get_query_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive query performance statistics"""
        if not self.performance_metrics:
            return {"status": "no_data"}

        # Calculate statistics by query type
        stats_by_type = {}
        for query_type in QueryType:
            type_metrics = [m for m in self.performance_metrics if m.query_type == query_type]

            if type_metrics:
                execution_times = [m.execution_time_ms for m in type_metrics]
                cache_hits = sum(1 for m in type_metrics if m.cache_hit)

                stats_by_type[query_type.value] = {
                    "total_queries": len(type_metrics),
                    "avg_execution_time_ms": sum(execution_times) / len(execution_times),
                    "min_execution_time_ms": min(execution_times),
                    "max_execution_time_ms": max(execution_times),
                    "cache_hit_rate": cache_hits / len(type_metrics) if type_metrics else 0,
                    "total_rows_returned": sum(m.rows_returned for m in type_metrics)
                }

        # Overall statistics
        all_times = [m.execution_time_ms for m in self.performance_metrics]
        all_cache_hits = sum(1 for m in self.performance_metrics if m.cache_hit)

        return {
            "overall": {
                "total_queries": len(self.performance_metrics),
                "avg_execution_time_ms": sum(all_times) / len(all_times),
                "cache_hit_rate": all_cache_hits / len(self.performance_metrics),
                "materialized_views": len(self.materialized_views)
            },
            "by_query_type": stats_by_type,
            "materialized_views": list(self.materialized_views.keys()),
            "last_updated": datetime.now().isoformat()
        }

    async def analyze_slow_queries(self, threshold_ms: float = 100) -> List[Dict[str, Any]]:
        """Analyze slow queries for optimization opportunities"""
        try:
            async with self.timescale_handler.pool.acquire() as conn:
                slow_queries = await conn.fetch("""
                    SELECT
                        query,
                        calls,
                        total_time,
                        mean_time,
                        max_time,
                        rows,
                        100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
                    FROM pg_stat_statements
                    WHERE mean_time > $1
                    ORDER BY mean_time DESC
                    LIMIT 20;
                """, threshold_ms)

                return [dict(row) for row in slow_queries]

        except Exception as e:
            logger.error(f"Error analyzing slow queries: {e}")
            return []

    async def suggest_optimizations(self) -> List[Dict[str, str]]:
        """Suggest query optimizations based on performance analysis"""
        suggestions = []

        # Analyze recent performance metrics
        recent_metrics = self.performance_metrics[-100:] if self.performance_metrics else []

        if recent_metrics:
            avg_time = sum(m.execution_time_ms for m in recent_metrics) / len(recent_metrics)
            cache_hit_rate = sum(1 for m in recent_metrics if m.cache_hit) / len(recent_metrics)

            if avg_time > 50:  # >50ms average
                suggestions.append({
                    "type": "performance",
                    "suggestion": "Consider adding more specific indexes for frequently used query patterns",
                    "priority": "high"
                })

            if cache_hit_rate < 0.7:  # <70% cache hit rate
                suggestions.append({
                    "type": "caching",
                    "suggestion": "Increase cache TTL for stable data or implement more aggressive cache warming",
                    "priority": "medium"
                })

        # Check materialized view freshness
        suggestions.append({
            "type": "maintenance",
            "suggestion": "Schedule regular materialized view refreshes during off-peak hours",
            "priority": "low"
        })

        return suggestions

    async def close(self) -> None:
        """Clean up resources"""
        self.performance_metrics.clear()
        self.query_cache.clear()
        logger.info("Query optimizer closed")


# Global query optimizer instance
query_optimizer = None

async def get_query_optimizer(
    timescale_handler: TimescaleDBHandler,
    cache_manager: Optional[RedisCacheManager] = None
) -> QueryOptimizer:
    """Get or create global query optimizer instance"""
    global query_optimizer
    if query_optimizer is None:
        query_optimizer = QueryOptimizer(timescale_handler, cache_manager)
        await query_optimizer.initialize()
    return query_optimizer