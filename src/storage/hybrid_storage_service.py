"""
Hybrid Storage Service for Market Data Agent
Combines TimescaleDB and Redis for optimal performance
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import time

from .timescaledb_handler import TimescaleDBHandler, TimescaleDBConfig
from ..caching.redis_cache_manager import RedisCacheManager, CacheConfig
from ..data_sources.base import PriceData, HealthStatus

logger = logging.getLogger(__name__)


@dataclass
class HybridStorageConfig:
    """Configuration for hybrid storage service"""
    timescale_config: Optional[TimescaleDBConfig] = None
    cache_config: Optional[CacheConfig] = None
    enable_cache: bool = True
    cache_warmup_symbols: List[str] = None
    write_through_cache: bool = True
    read_through_cache: bool = True
    batch_write_size: int = 1000
    cache_warmup_on_startup: bool = True


@dataclass
class StoragePerformanceMetrics:
    """Performance metrics for hybrid storage"""
    cache_hit_rate: float = 0.0
    avg_read_time_ms: float = 0.0
    avg_write_time_ms: float = 0.0
    cache_response_time_ms: float = 0.0
    db_response_time_ms: float = 0.0
    total_operations: int = 0
    cache_operations: int = 0
    db_operations: int = 0
    write_throughput_per_sec: float = 0.0


class HybridStorageService:
    """
    High-performance hybrid storage service combining TimescaleDB and Redis

    Features:
    - Sub-millisecond read performance with Redis caching
    - High-throughput writes to TimescaleDB
    - Intelligent cache-through patterns
    - Automatic cache warming and invalidation
    - Performance monitoring and optimization
    """

    def __init__(self, config: HybridStorageConfig = None):
        self.config = config or HybridStorageConfig()

        # Initialize handlers
        self.timescale_handler = TimescaleDBHandler(
            self.config.timescale_config or TimescaleDBConfig()
        )

        self.cache_manager = None
        if self.config.enable_cache:
            self.cache_manager = RedisCacheManager(
                self.config.cache_config or CacheConfig(),
                self.timescale_handler
            )

        self.is_initialized = False
        self._metrics = StoragePerformanceMetrics()
        self._metrics_start_time = time.time()

    async def initialize(self) -> None:
        """Initialize both TimescaleDB and Redis connections"""
        try:
            # Initialize TimescaleDB
            await self.timescale_handler.initialize()
            logger.info("TimescaleDB handler initialized")

            # Initialize Redis cache if enabled
            if self.cache_manager:
                await self.cache_manager.initialize()
                logger.info("Redis cache manager initialized")

                # Warm cache on startup if configured
                if self.config.cache_warmup_on_startup and self.config.cache_warmup_symbols:
                    await self._warm_startup_cache()

            self.is_initialized = True
            logger.info("Hybrid storage service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize hybrid storage service: {e}")
            raise

    async def store_ohlcv_data(
        self,
        price_data: List[PriceData],
        source: str,
        quality_threshold: float = 0.8
    ) -> int:
        """Store OHLCV data with write-through caching"""
        if not self.is_initialized:
            raise RuntimeError("Storage service not initialized")

        start_time = time.perf_counter()
        stored_count = 0

        try:
            # Primary write to TimescaleDB
            stored_count = await self.timescale_handler.store_ohlcv_data(
                price_data, source, quality_threshold
            )

            # Write-through to cache if enabled
            if (self.cache_manager and
                self.config.write_through_cache and
                stored_count > 0):

                # Cache latest prices for each symbol
                await self._cache_latest_prices(price_data)

            # Update performance metrics
            write_time = (time.perf_counter() - start_time) * 1000
            self._update_write_metrics(stored_count, write_time)

            return stored_count

        except Exception as e:
            logger.error(f"Error storing OHLCV data: {e}")
            raise

    async def get_latest_data(self, symbol: str) -> Optional[PriceData]:
        """Get latest price data with cache-first strategy"""
        if not self.is_initialized:
            raise RuntimeError("Storage service not initialized")

        start_time = time.perf_counter()

        try:
            # Try cache first if enabled
            if self.cache_manager and self.config.read_through_cache:
                data = await self.cache_manager.get_latest_price(symbol)
                if data:
                    response_time = (time.perf_counter() - start_time) * 1000
                    self._update_read_metrics(response_time, from_cache=True)
                    return data

            # Fallback to TimescaleDB
            data = await self.timescale_handler.get_latest_data(symbol)

            # Cache the result if found
            if data and self.cache_manager:
                await self.cache_manager.cache_latest_price(data)

            response_time = (time.perf_counter() - start_time) * 1000
            self._update_read_metrics(response_time, from_cache=False)

            return data

        except Exception as e:
            logger.error(f"Error getting latest data for {symbol}: {e}")
            raise

    async def get_historical_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[PriceData]:
        """Get historical data with intelligent caching"""
        if not self.is_initialized:
            raise RuntimeError("Storage service not initialized")

        perf_start = time.perf_counter()

        try:
            # Try cache first for recent data
            if (self.cache_manager and
                self.config.read_through_cache and
                self._is_recent_data(end_time)):

                data = await self.cache_manager.get_historical_data(symbol, start_time, end_time)
                if data:
                    response_time = (time.perf_counter() - perf_start) * 1000
                    self._update_read_metrics(response_time, from_cache=True)
                    return data

            # Fetch from TimescaleDB
            data = await self.timescale_handler.get_historical_data(symbol, start_time, end_time)

            # Cache recent data
            if (data and
                self.cache_manager and
                self._is_recent_data(end_time) and
                len(data) <= 1000):  # Don't cache very large datasets

                await self.cache_manager.cache_historical_data(symbol, start_time, end_time, data)

            response_time = (time.perf_counter() - perf_start) * 1000
            self._update_read_metrics(response_time, from_cache=False)

            return data

        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            raise

    async def get_ohlc_aggregated(
        self,
        symbol: str,
        date: datetime,
        force_refresh: bool = False
    ) -> Optional[Dict[str, float]]:
        """Get daily OHLC aggregated data with caching"""
        if not self.is_initialized:
            raise RuntimeError("Storage service not initialized")

        try:
            # Check cache first (unless force refresh)
            if (self.cache_manager and
                not force_refresh):

                cached_data = await self.cache_manager.get_ohlc_daily(symbol, date)
                if cached_data:
                    return cached_data

            # Calculate from TimescaleDB
            start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = start_of_day + timedelta(days=1)

            historical_data = await self.timescale_handler.get_historical_data(
                symbol, start_of_day, end_of_day
            )

            if not historical_data:
                return None

            # Calculate OHLC aggregation
            ohlc_data = {
                "open": historical_data[0].open_price,
                "high": max(d.high_price for d in historical_data),
                "low": min(d.low_price for d in historical_data),
                "close": historical_data[-1].close_price,
                "volume": sum(d.volume for d in historical_data),
                "count": len(historical_data)
            }

            # Cache the result
            if self.cache_manager:
                await self.cache_manager.cache_ohlc_daily(symbol, date, ohlc_data)

            return ohlc_data

        except Exception as e:
            logger.error(f"Error getting OHLC aggregated data for {symbol}: {e}")
            raise

    async def bulk_cache_latest_prices(self, symbols: List[str]) -> Dict[str, bool]:
        """Bulk cache latest prices for multiple symbols"""
        if not self.cache_manager:
            return {symbol: False for symbol in symbols}

        results = {}

        try:
            # Fetch latest data for all symbols
            price_data_list = []
            for symbol in symbols:
                data = await self.timescale_handler.get_latest_data(symbol)
                if data:
                    price_data_list.append(data)

            # Bulk cache the data
            if price_data_list:
                cached_count = await self.cache_manager.bulk_cache_prices(price_data_list)

                # Update results
                for data in price_data_list:
                    results[data.symbol] = True

                # Mark missing symbols
                for symbol in symbols:
                    if symbol not in results:
                        results[symbol] = False

                logger.info(f"Bulk cached {cached_count} latest prices")

            return results

        except Exception as e:
            logger.error(f"Error bulk caching latest prices: {e}")
            return {symbol: False for symbol in symbols}

    async def invalidate_symbol_cache(self, symbol: str) -> bool:
        """Invalidate all cached data for a symbol"""
        if not self.cache_manager:
            return True

        try:
            deleted_count = await self.cache_manager.invalidate_symbol_cache(symbol)
            logger.info(f"Invalidated {deleted_count} cache entries for {symbol}")
            return True

        except Exception as e:
            logger.error(f"Error invalidating cache for {symbol}: {e}")
            return False

    async def optimize_storage(self) -> Dict[str, Any]:
        """Optimize both TimescaleDB and Redis storage"""
        results = {"timescale": {}, "redis": {}}

        try:
            # Optimize TimescaleDB
            timescale_result = await self.timescale_handler.optimize_database()
            results["timescale"] = timescale_result

            # Create advanced indexes if not exists
            index_result = await self.timescale_handler.create_advanced_indexes()
            results["timescale"]["advanced_indexes"] = index_result

            # Redis optimization (memory usage, eviction policies)
            if self.cache_manager:
                cache_stats = await self.cache_manager.get_cache_statistics()
                results["redis"] = {
                    "cache_size_mb": cache_stats.cache_size_mb,
                    "hit_rate": cache_stats.hit_rate,
                    "evictions": cache_stats.evictions
                }

            logger.info("Storage optimization completed")
            return results

        except Exception as e:
            logger.error(f"Error optimizing storage: {e}")
            return {"error": str(e)}

    async def get_performance_metrics(self) -> StoragePerformanceMetrics:
        """Get comprehensive performance metrics"""
        try:
            # Update cache metrics if available
            if self.cache_manager:
                cache_stats = await self.cache_manager.get_cache_statistics()
                self._metrics.cache_hit_rate = cache_stats.hit_rate
                self._metrics.cache_response_time_ms = cache_stats.avg_response_time_ms

            # Calculate write throughput
            elapsed_time = time.time() - self._metrics_start_time
            if elapsed_time > 0:
                self._metrics.write_throughput_per_sec = self._metrics.db_operations / elapsed_time

            return self._metrics

        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return self._metrics

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for hybrid storage"""
        health_status = {
            "status": "healthy",
            "components": {},
            "metrics": {}
        }

        try:
            # TimescaleDB health check
            try:
                timescale_stats = await self.timescale_handler.get_storage_statistics()
                health_status["components"]["timescale"] = {
                    "status": "healthy",
                    "total_records": timescale_stats.total_records,
                    "storage_size_mb": timescale_stats.storage_size_mb
                }
            except Exception as e:
                health_status["components"]["timescale"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["status"] = "degraded"

            # Redis health check
            if self.cache_manager:
                try:
                    cache_stats = await self.cache_manager.get_cache_statistics()
                    health_status["components"]["redis"] = {
                        "status": "healthy",
                        "hit_rate": cache_stats.hit_rate,
                        "cache_size_mb": cache_stats.cache_size_mb
                    }
                except Exception as e:
                    health_status["components"]["redis"] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
                    health_status["status"] = "degraded"

            # Overall performance metrics
            metrics = await self.get_performance_metrics()
            health_status["metrics"] = {
                "avg_read_time_ms": metrics.avg_read_time_ms,
                "cache_hit_rate": metrics.cache_hit_rate,
                "total_operations": metrics.total_operations
            }

            return health_status

        except Exception as e:
            logger.error(f"Error in health check: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def _cache_latest_prices(self, price_data_list: List[PriceData]) -> int:
        """Cache latest prices from a list of price data"""
        if not self.cache_manager:
            return 0

        try:
            # Group by symbol and get the latest for each
            symbol_latest = {}
            for data in price_data_list:
                if (data.symbol not in symbol_latest or
                    data.timestamp > symbol_latest[data.symbol].timestamp):
                    symbol_latest[data.symbol] = data

            # Bulk cache the latest prices
            latest_prices = list(symbol_latest.values())
            return await self.cache_manager.bulk_cache_prices(latest_prices)

        except Exception as e:
            logger.error(f"Error caching latest prices: {e}")
            return 0

    async def _warm_startup_cache(self) -> None:
        """Warm cache with frequently accessed symbols on startup"""
        if not self.config.cache_warmup_symbols:
            return

        try:
            logger.info(f"Warming cache for {len(self.config.cache_warmup_symbols)} symbols")

            results = await self.cache_manager.warm_cache(self.config.cache_warmup_symbols)
            successful = sum(1 for success in results.values() if success)

            logger.info(f"Cache warming completed: {successful}/{len(results)} symbols cached")

        except Exception as e:
            logger.error(f"Error warming startup cache: {e}")

    def _is_recent_data(self, end_time: datetime) -> bool:
        """Check if data is recent enough to cache"""
        now = datetime.now()
        return (now - end_time) < timedelta(hours=24)

    def _update_write_metrics(self, record_count: int, write_time_ms: float):
        """Update write performance metrics"""
        self._metrics.total_operations += 1
        self._metrics.db_operations += 1

        # Update average write time with exponential moving average
        alpha = 0.1
        if self._metrics.avg_write_time_ms == 0:
            self._metrics.avg_write_time_ms = write_time_ms
        else:
            self._metrics.avg_write_time_ms = (
                alpha * write_time_ms +
                (1 - alpha) * self._metrics.avg_write_time_ms
            )

    def _update_read_metrics(self, response_time_ms: float, from_cache: bool):
        """Update read performance metrics"""
        self._metrics.total_operations += 1

        if from_cache:
            self._metrics.cache_operations += 1
            self._metrics.cache_response_time_ms = response_time_ms
        else:
            self._metrics.db_operations += 1
            self._metrics.db_response_time_ms = response_time_ms

        # Update average read time
        alpha = 0.1
        if self._metrics.avg_read_time_ms == 0:
            self._metrics.avg_read_time_ms = response_time_ms
        else:
            self._metrics.avg_read_time_ms = (
                alpha * response_time_ms +
                (1 - alpha) * self._metrics.avg_read_time_ms
            )

    async def close(self) -> None:
        """Close all storage connections"""
        try:
            if self.cache_manager:
                await self.cache_manager.close()

            await self.timescale_handler.close()

            self.is_initialized = False
            logger.info("Hybrid storage service closed")

        except Exception as e:
            logger.error(f"Error closing hybrid storage service: {e}")

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Global hybrid storage service instance
hybrid_storage = None

async def get_hybrid_storage() -> HybridStorageService:
    """Get or create global hybrid storage service instance"""
    global hybrid_storage
    if hybrid_storage is None:
        hybrid_storage = HybridStorageService()
        await hybrid_storage.initialize()
    return hybrid_storage