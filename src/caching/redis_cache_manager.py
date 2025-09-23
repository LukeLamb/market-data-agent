"""
Redis Cache Manager for Market Data Agent
Provides sub-millisecond response times for frequently accessed data
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import time
from contextlib import asynccontextmanager

import redis.asyncio as redis
from redis.asyncio import ConnectionPool

from ..data_sources.base import PriceData, CurrentPrice
from ..storage.timescaledb_handler import TimescaleDBHandler

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Redis cache configuration"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    pool_max_connections: int = 50
    socket_timeout: float = 1.0
    socket_connect_timeout: float = 1.0
    retry_on_timeout: bool = True
    decode_responses: bool = True

    # Cache TTL settings (seconds)
    latest_price_ttl: int = 10  # 10 seconds for latest prices
    historical_data_ttl: int = 300  # 5 minutes for historical data
    aggregated_data_ttl: int = 600  # 10 minutes for aggregated data
    symbol_list_ttl: int = 3600  # 1 hour for symbol lists
    health_status_ttl: int = 30  # 30 seconds for health status


@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    avg_response_time_ms: float = 0.0
    total_operations: int = 0
    cache_size_mb: float = 0.0
    evictions: int = 0
    connections_active: int = 0


class CacheKeyBuilder:
    """Standardized cache key building"""

    @staticmethod
    def latest_price(symbol: str) -> str:
        """Key for latest price data"""
        return f"latest:{symbol.upper()}"

    @staticmethod
    def historical_data(symbol: str, start_time: datetime, end_time: datetime) -> str:
        """Key for historical data range"""
        start_str = start_time.strftime("%Y%m%d_%H%M")
        end_str = end_time.strftime("%Y%m%d_%H%M")
        return f"hist:{symbol.upper()}:{start_str}:{end_str}"

    @staticmethod
    def ohlc_daily(symbol: str, date: datetime) -> str:
        """Key for daily OHLC data"""
        date_str = date.strftime("%Y%m%d")
        return f"ohlc_daily:{symbol.upper()}:{date_str}"

    @staticmethod
    def volume_data(symbol: str, timeframe: str) -> str:
        """Key for volume aggregation data"""
        return f"volume:{symbol.upper()}:{timeframe}"

    @staticmethod
    def symbol_list(exchange: Optional[str] = None) -> str:
        """Key for symbol lists"""
        if exchange:
            return f"symbols:{exchange.upper()}"
        return "symbols:all"

    @staticmethod
    def health_status(source: str) -> str:
        """Key for data source health status"""
        return f"health:{source}"

    @staticmethod
    def aggregated_stats(symbol: str, period: str) -> str:
        """Key for aggregated statistics"""
        return f"stats:{symbol.upper()}:{period}"


class RedisCacheManager:
    """High-performance Redis cache manager for market data"""

    def __init__(self, config: CacheConfig = None, timescale_handler: TimescaleDBHandler = None):
        self.config = config or CacheConfig()
        self.timescale_handler = timescale_handler
        self.redis_pool = None
        self.redis_client = None
        self.is_connected = False
        self._stats = CacheStats()
        self._stats_last_update = datetime.now()

    async def initialize(self) -> None:
        """Initialize Redis connection pool"""
        try:
            # Create connection pool
            self.redis_pool = ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                max_connections=self.config.pool_max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                decode_responses=self.config.decode_responses
            )

            # Create Redis client
            self.redis_client = redis.Redis(connection_pool=self.redis_pool)

            # Test connection
            await self.redis_client.ping()
            self.is_connected = True

            logger.info("Redis cache manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Redis cache manager: {e}")
            self.is_connected = False
            raise

    async def get_latest_price(self, symbol: str) -> Optional[PriceData]:
        """Get latest price with cache-through pattern"""
        cache_key = CacheKeyBuilder.latest_price(symbol)
        start_time = time.perf_counter()

        try:
            # Try cache first
            cached_data = await self.redis_client.get(cache_key)
            response_time = (time.perf_counter() - start_time) * 1000

            if cached_data:
                self._record_hit(response_time)
                return PriceData.model_validate_json(cached_data)
            else:
                self._record_miss(response_time)

            # Cache miss - fetch from TimescaleDB
            if self.timescale_handler:
                data = await self.timescale_handler.get_latest_data(symbol)
                if data:
                    # Cache the result
                    await self.redis_client.setex(
                        cache_key,
                        self.config.latest_price_ttl,
                        data.model_dump_json()
                    )
                    return data

            return None

        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            # Fallback to database on cache error
            if self.timescale_handler:
                return await self.timescale_handler.get_latest_data(symbol)
            return None

    async def get_historical_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[PriceData]:
        """Get historical data with intelligent caching"""
        cache_key = CacheKeyBuilder.historical_data(symbol, start_time, end_time)
        start_perf = time.perf_counter()

        try:
            # Check cache
            cached_data = await self.redis_client.get(cache_key)
            response_time = (time.perf_counter() - start_perf) * 1000

            if cached_data:
                self._record_hit(response_time)
                data_list = json.loads(cached_data)
                return [PriceData.model_validate(item) for item in data_list]
            else:
                self._record_miss(response_time)

            # Cache miss - fetch from TimescaleDB
            if self.timescale_handler:
                data = await self.timescale_handler.get_historical_data(symbol, start_time, end_time)
                if data:
                    # Cache with appropriate TTL
                    ttl = self._calculate_historical_ttl(start_time, end_time)
                    data_json = json.dumps([item.model_dump() for item in data])
                    await self.redis_client.setex(cache_key, ttl, data_json)
                    return data

            return []

        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            # Fallback to database
            if self.timescale_handler:
                return await self.timescale_handler.get_historical_data(symbol, start_time, end_time)
            return []

    async def cache_latest_price(self, price_data: PriceData) -> bool:
        """Cache latest price data"""
        try:
            cache_key = CacheKeyBuilder.latest_price(price_data.symbol)
            await self.redis_client.setex(
                cache_key,
                self.config.latest_price_ttl,
                price_data.model_dump_json()
            )
            return True
        except Exception as e:
            logger.error(f"Error caching latest price: {e}")
            return False

    async def cache_ohlc_daily(self, symbol: str, date: datetime, ohlc_data: Dict[str, float]) -> bool:
        """Cache daily OHLC aggregation"""
        try:
            cache_key = CacheKeyBuilder.ohlc_daily(symbol, date)
            await self.redis_client.setex(
                cache_key,
                self.config.aggregated_data_ttl,
                json.dumps(ohlc_data)
            )
            return True
        except Exception as e:
            logger.error(f"Error caching daily OHLC: {e}")
            return False

    async def get_ohlc_daily(self, symbol: str, date: datetime) -> Optional[Dict[str, float]]:
        """Get cached daily OHLC data"""
        try:
            cache_key = CacheKeyBuilder.ohlc_daily(symbol, date)
            start_time = time.perf_counter()

            cached_data = await self.redis_client.get(cache_key)
            response_time = (time.perf_counter() - start_time) * 1000

            if cached_data:
                self._record_hit(response_time)
                return json.loads(cached_data)
            else:
                self._record_miss(response_time)
                return None

        except Exception as e:
            logger.error(f"Error getting cached OHLC data: {e}")
            return None

    async def invalidate_symbol_cache(self, symbol: str) -> int:
        """Invalidate all cache entries for a symbol"""
        try:
            pattern = f"*:{symbol.upper()}:*"
            keys = await self.redis_client.keys(pattern)
            if keys:
                return await self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Error invalidating cache for {symbol}: {e}")
            return 0

    async def bulk_cache_prices(self, price_data_list: List[PriceData]) -> int:
        """Bulk cache multiple price data points"""
        try:
            pipeline = self.redis_client.pipeline()

            for price_data in price_data_list:
                cache_key = CacheKeyBuilder.latest_price(price_data.symbol)
                pipeline.setex(
                    cache_key,
                    self.config.latest_price_ttl,
                    price_data.model_dump_json()
                )

            results = await pipeline.execute()
            return sum(1 for result in results if result)

        except Exception as e:
            logger.error(f"Error bulk caching prices: {e}")
            return 0

    async def cache_symbol_list(self, symbols: List[str], exchange: Optional[str] = None) -> bool:
        """Cache symbol list for fast retrieval"""
        try:
            cache_key = CacheKeyBuilder.symbol_list(exchange)
            await self.redis_client.setex(
                cache_key,
                self.config.symbol_list_ttl,
                json.dumps(symbols)
            )
            return True
        except Exception as e:
            logger.error(f"Error caching symbol list: {e}")
            return False

    async def get_symbol_list(self, exchange: Optional[str] = None) -> Optional[List[str]]:
        """Get cached symbol list"""
        try:
            cache_key = CacheKeyBuilder.symbol_list(exchange)
            start_time = time.perf_counter()

            cached_data = await self.redis_client.get(cache_key)
            response_time = (time.perf_counter() - start_time) * 1000

            if cached_data:
                self._record_hit(response_time)
                return json.loads(cached_data)
            else:
                self._record_miss(response_time)
                return None

        except Exception as e:
            logger.error(f"Error getting cached symbol list: {e}")
            return None

    async def cache_historical_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        data: List[PriceData]
    ) -> bool:
        """Cache historical data for a time range"""
        try:
            cache_key = CacheKeyBuilder.historical_data(symbol, start_time, end_time)
            ttl = self._calculate_historical_ttl(start_time, end_time)
            data_json = json.dumps([item.model_dump() for item in data])

            await self.redis_client.setex(cache_key, ttl, data_json)
            return True

        except Exception as e:
            logger.error(f"Error caching historical data: {e}")
            return False

    async def warm_cache(self, symbols: List[str]) -> Dict[str, bool]:
        """Pre-warm cache with latest data for specified symbols"""
        results = {}

        if not self.timescale_handler:
            logger.warning("No TimescaleDB handler available for cache warming")
            return results

        for symbol in symbols:
            try:
                # Fetch latest data from database
                latest_data = await self.timescale_handler.get_latest_data(symbol)
                if latest_data:
                    success = await self.cache_latest_price(latest_data)
                    results[symbol] = success
                else:
                    results[symbol] = False

            except Exception as e:
                logger.error(f"Error warming cache for {symbol}: {e}")
                results[symbol] = False

        logger.info(f"Cache warming completed for {len(symbols)} symbols")
        return results

    def _calculate_historical_ttl(self, start_time: datetime, end_time: datetime) -> int:
        """Calculate appropriate TTL for historical data based on age"""
        now = datetime.now()
        data_age = now - end_time

        if data_age < timedelta(hours=1):
            return 60  # 1 minute for very recent data
        elif data_age < timedelta(days=1):
            return 300  # 5 minutes for today's data
        elif data_age < timedelta(days=7):
            return 1800  # 30 minutes for this week's data
        else:
            return 3600  # 1 hour for older data

    def _record_hit(self, response_time_ms: float):
        """Record cache hit statistics"""
        self._stats.hits += 1
        self._stats.total_operations += 1
        self._update_avg_response_time(response_time_ms)
        self._update_hit_rate()

    def _record_miss(self, response_time_ms: float):
        """Record cache miss statistics"""
        self._stats.misses += 1
        self._stats.total_operations += 1
        self._update_avg_response_time(response_time_ms)
        self._update_hit_rate()

    def _update_hit_rate(self):
        """Update hit rate calculation"""
        if self._stats.total_operations > 0:
            self._stats.hit_rate = self._stats.hits / self._stats.total_operations

    def _update_avg_response_time(self, response_time_ms: float):
        """Update average response time with exponential moving average"""
        alpha = 0.1  # Smoothing factor
        if self._stats.avg_response_time_ms == 0:
            self._stats.avg_response_time_ms = response_time_ms
        else:
            self._stats.avg_response_time_ms = (
                alpha * response_time_ms +
                (1 - alpha) * self._stats.avg_response_time_ms
            )

    async def get_cache_statistics(self) -> CacheStats:
        """Get comprehensive cache statistics"""
        try:
            # Get Redis info
            info = await self.redis_client.info()

            # Update statistics
            self._stats.cache_size_mb = info.get('used_memory', 0) / (1024 * 1024)
            self._stats.evictions = info.get('evicted_keys', 0)
            self._stats.connections_active = info.get('connected_clients', 0)

            return self._stats

        except Exception as e:
            logger.error(f"Error getting cache statistics: {e}")
            return self._stats

    async def clear_cache(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries matching pattern"""
        try:
            if pattern:
                keys = await self.redis_client.keys(pattern)
                if keys:
                    return await self.redis_client.delete(*keys)
                return 0
            else:
                await self.redis_client.flushdb()
                return 1

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0

    @asynccontextmanager
    async def pipeline(self):
        """Redis pipeline context manager for batch operations"""
        pipe = self.redis_client.pipeline()
        try:
            yield pipe
            await pipe.execute()
        except Exception as e:
            await pipe.reset()
            logger.error(f"Pipeline error: {e}")
            raise

    async def close(self) -> None:
        """Close Redis connections"""
        if self.redis_client:
            await self.redis_client.aclose()
        if self.redis_pool:
            await self.redis_pool.aclose()
        self.is_connected = False
        logger.info("Redis cache manager closed")

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Global cache manager instance
cache_manager = None

async def get_cache_manager() -> RedisCacheManager:
    """Get or create global cache manager instance"""
    global cache_manager
    if cache_manager is None:
        cache_manager = RedisCacheManager()
        await cache_manager.initialize()
    return cache_manager