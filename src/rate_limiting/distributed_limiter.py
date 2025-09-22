"""Distributed Rate Limiter Implementation

Redis-based distributed rate limiting for multi-instance deployments.
Supports sliding window and token bucket algorithms across multiple processes.
"""

import asyncio
import time
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

try:
    import aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("aioredis not available - distributed rate limiting will use local fallback")


@dataclass
class DistributedLimitConfig:
    """Configuration for distributed rate limiting"""
    key: str                    # Redis key for this limit
    limit: int                  # Number of requests allowed
    window_seconds: int         # Time window in seconds
    burst_allowance: float = 1.2  # Allow 20% burst by default
    strict_mode: bool = False   # Strict enforcement without burst


class DistributedRateLimiter:
    """Redis-based distributed rate limiter

    Features:
    - Sliding window rate limiting
    - Token bucket algorithm support
    - Cross-instance coordination
    - Automatic cleanup of old data
    - Fallback to local limiting when Redis unavailable
    """

    def __init__(self, redis_url: str = "redis://localhost:6379",
                 key_prefix: str = "rate_limit"):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.redis_client: Optional[aioredis.Redis] = None
        self.connected = False

        # Local fallback
        self.local_windows: Dict[str, List[float]] = {}
        self.local_tokens: Dict[str, Dict[str, Any]] = {}

        # Lua scripts for atomic operations
        self.sliding_window_script = """
            local key = KEYS[1]
            local window = tonumber(ARGV[1])
            local limit = tonumber(ARGV[2])
            local now = tonumber(ARGV[3])
            local burst_limit = tonumber(ARGV[4])

            -- Remove old entries
            redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)

            -- Count current entries
            local current = redis.call('ZCARD', key)

            -- Check if under limit
            if current < limit then
                -- Add this request
                redis.call('ZADD', key, now, now)
                redis.call('EXPIRE', key, window)
                return {current + 1, limit - current - 1}
            elseif current < burst_limit then
                -- Allow burst but warn
                redis.call('ZADD', key, now, now)
                redis.call('EXPIRE', key, window)
                return {current + 1, 0, 'burst'}
            else
                -- Rate limited
                return {current, 0, 'limited'}
            end
        """

        self.token_bucket_script = """
            local key = KEYS[1]
            local capacity = tonumber(ARGV[1])
            local refill_rate = tonumber(ARGV[2])
            local tokens_requested = tonumber(ARGV[3])
            local now = tonumber(ARGV[4])

            -- Get current bucket state
            local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
            local tokens = tonumber(bucket[1]) or capacity
            local last_refill = tonumber(bucket[2]) or now

            -- Calculate tokens to add
            local elapsed = now - last_refill
            local tokens_to_add = elapsed * refill_rate
            tokens = math.min(capacity, tokens + tokens_to_add)

            -- Try to consume tokens
            if tokens >= tokens_requested then
                tokens = tokens - tokens_requested
                redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
                redis.call('EXPIRE', key, 3600)  -- 1 hour expiry
                return {tokens, 'success'}
            else
                -- Update state without consuming
                redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
                redis.call('EXPIRE', key, 3600)
                local wait_time = (tokens_requested - tokens) / refill_rate
                return {tokens, 'limited', wait_time}
            end
        """

    async def connect(self) -> bool:
        """Connect to Redis"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, using local fallback")
            return False

        try:
            self.redis_client = aioredis.from_url(self.redis_url)
            # Test connection
            await self.redis_client.ping()
            self.connected = True
            logger.info(f"Connected to Redis at {self.redis_url}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from Redis"""
        if self.redis_client:
            await self.redis_client.close()
            self.connected = False
            logger.info("Disconnected from Redis")

    def _get_key(self, identifier: str) -> str:
        """Generate Redis key for rate limit"""
        return f"{self.key_prefix}:{identifier}"

    async def check_sliding_window(self, config: DistributedLimitConfig) -> Dict[str, Any]:
        """Check rate limit using sliding window algorithm

        Args:
            config: Rate limit configuration

        Returns:
            Dictionary with status information
        """
        if self.connected and self.redis_client:
            return await self._check_sliding_window_redis(config)
        else:
            return await self._check_sliding_window_local(config)

    async def _check_sliding_window_redis(self, config: DistributedLimitConfig) -> Dict[str, Any]:
        """Redis-based sliding window check"""
        key = self._get_key(config.key)
        now = time.time()
        burst_limit = int(config.limit * config.burst_allowance)

        try:
            result = await self.redis_client.eval(
                self.sliding_window_script,
                1,
                key,
                config.window_seconds,
                config.limit,
                now,
                burst_limit
            )

            current_count = result[0]
            remaining = result[1]
            status = result[2] if len(result) > 2 else 'allowed'

            return {
                "allowed": status != 'limited',
                "current_count": current_count,
                "limit": config.limit,
                "remaining": remaining,
                "window_seconds": config.window_seconds,
                "status": status,
                "reset_time": now + config.window_seconds
            }

        except Exception as e:
            logger.error(f"Redis sliding window check failed: {e}")
            # Fallback to local
            return await self._check_sliding_window_local(config)

    async def _check_sliding_window_local(self, config: DistributedLimitConfig) -> Dict[str, Any]:
        """Local fallback sliding window check"""
        now = time.time()
        key = config.key

        # Initialize if needed
        if key not in self.local_windows:
            self.local_windows[key] = []

        # Clean old entries
        cutoff = now - config.window_seconds
        self.local_windows[key] = [t for t in self.local_windows[key] if t > cutoff]

        current_count = len(self.local_windows[key])
        burst_limit = int(config.limit * config.burst_allowance)

        if current_count < config.limit:
            self.local_windows[key].append(now)
            return {
                "allowed": True,
                "current_count": current_count + 1,
                "limit": config.limit,
                "remaining": config.limit - current_count - 1,
                "window_seconds": config.window_seconds,
                "status": "allowed",
                "reset_time": now + config.window_seconds
            }
        elif not config.strict_mode and current_count < burst_limit:
            self.local_windows[key].append(now)
            return {
                "allowed": True,
                "current_count": current_count + 1,
                "limit": config.limit,
                "remaining": 0,
                "window_seconds": config.window_seconds,
                "status": "burst",
                "reset_time": now + config.window_seconds
            }
        else:
            return {
                "allowed": False,
                "current_count": current_count,
                "limit": config.limit,
                "remaining": 0,
                "window_seconds": config.window_seconds,
                "status": "limited",
                "reset_time": now + config.window_seconds
            }

    async def check_token_bucket(self, bucket_id: str, capacity: int,
                               refill_rate: float, tokens_requested: int = 1) -> Dict[str, Any]:
        """Check rate limit using token bucket algorithm

        Args:
            bucket_id: Unique identifier for the bucket
            capacity: Maximum tokens in bucket
            refill_rate: Tokens added per second
            tokens_requested: Number of tokens to consume

        Returns:
            Dictionary with status information
        """
        if self.connected and self.redis_client:
            return await self._check_token_bucket_redis(bucket_id, capacity, refill_rate, tokens_requested)
        else:
            return await self._check_token_bucket_local(bucket_id, capacity, refill_rate, tokens_requested)

    async def _check_token_bucket_redis(self, bucket_id: str, capacity: int,
                                      refill_rate: float, tokens_requested: int) -> Dict[str, Any]:
        """Redis-based token bucket check"""
        key = self._get_key(f"bucket:{bucket_id}")
        now = time.time()

        try:
            result = await self.redis_client.eval(
                self.token_bucket_script,
                1,
                key,
                capacity,
                refill_rate,
                tokens_requested,
                now
            )

            remaining_tokens = result[0]
            status = result[1]
            wait_time = result[2] if len(result) > 2 else 0

            return {
                "allowed": status == 'success',
                "remaining_tokens": remaining_tokens,
                "capacity": capacity,
                "refill_rate": refill_rate,
                "wait_time": wait_time,
                "status": status
            }

        except Exception as e:
            logger.error(f"Redis token bucket check failed: {e}")
            # Fallback to local
            return await self._check_token_bucket_local(bucket_id, capacity, refill_rate, tokens_requested)

    async def _check_token_bucket_local(self, bucket_id: str, capacity: int,
                                      refill_rate: float, tokens_requested: int) -> Dict[str, Any]:
        """Local fallback token bucket check"""
        now = time.time()

        # Initialize bucket if needed
        if bucket_id not in self.local_tokens:
            self.local_tokens[bucket_id] = {
                "tokens": capacity,
                "last_refill": now
            }

        bucket = self.local_tokens[bucket_id]

        # Refill tokens
        elapsed = now - bucket["last_refill"]
        tokens_to_add = elapsed * refill_rate
        bucket["tokens"] = min(capacity, bucket["tokens"] + tokens_to_add)
        bucket["last_refill"] = now

        # Try to consume tokens
        if bucket["tokens"] >= tokens_requested:
            bucket["tokens"] -= tokens_requested
            return {
                "allowed": True,
                "remaining_tokens": bucket["tokens"],
                "capacity": capacity,
                "refill_rate": refill_rate,
                "wait_time": 0,
                "status": "success"
            }
        else:
            wait_time = (tokens_requested - bucket["tokens"]) / refill_rate
            return {
                "allowed": False,
                "remaining_tokens": bucket["tokens"],
                "capacity": capacity,
                "refill_rate": refill_rate,
                "wait_time": wait_time,
                "status": "limited"
            }

    async def get_global_stats(self, pattern: str = "*") -> Dict[str, Any]:
        """Get global rate limiting statistics

        Args:
            pattern: Redis key pattern to match

        Returns:
            Statistics dictionary
        """
        if not self.connected or not self.redis_client:
            return {
                "connected": False,
                "local_windows": len(self.local_windows),
                "local_buckets": len(self.local_tokens)
            }

        try:
            search_pattern = f"{self.key_prefix}:{pattern}"
            keys = await self.redis_client.keys(search_pattern)

            stats = {
                "connected": True,
                "total_keys": len(keys),
                "windows": 0,
                "buckets": 0,
                "active_limits": []
            }

            for key in keys[:20]:  # Limit to first 20 for performance
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key

                if "bucket:" in key_str:
                    bucket_data = await self.redis_client.hgetall(key_str)
                    if bucket_data:
                        stats["buckets"] += 1
                        stats["active_limits"].append({
                            "key": key_str,
                            "type": "bucket",
                            "tokens": float(bucket_data.get(b"tokens", 0)),
                            "last_refill": float(bucket_data.get(b"last_refill", 0))
                        })
                else:
                    count = await self.redis_client.zcard(key_str)
                    if count > 0:
                        stats["windows"] += 1
                        stats["active_limits"].append({
                            "key": key_str,
                            "type": "window",
                            "current_count": count
                        })

            return stats

        except Exception as e:
            logger.error(f"Failed to get global stats: {e}")
            return {"connected": False, "error": str(e)}

    async def cleanup_expired(self, max_age_hours: int = 24) -> int:
        """Clean up expired rate limit data

        Args:
            max_age_hours: Maximum age of data to keep

        Returns:
            Number of keys cleaned up
        """
        if not self.connected or not self.redis_client:
            # Clean local data
            now = time.time()
            cutoff = now - (max_age_hours * 3600)

            cleaned = 0
            for key in list(self.local_windows.keys()):
                self.local_windows[key] = [t for t in self.local_windows[key] if t > cutoff]
                if not self.local_windows[key]:
                    del self.local_windows[key]
                    cleaned += 1

            return cleaned

        try:
            search_pattern = f"{self.key_prefix}:*"
            keys = await self.redis_client.keys(search_pattern)
            cleaned = 0

            for key in keys:
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key

                # Check TTL
                ttl = await self.redis_client.ttl(key_str)
                if ttl == -1:  # No expiry set
                    await self.redis_client.expire(key_str, 3600)  # Set 1 hour expiry

                # For windows, clean old entries
                if "bucket:" not in key_str:
                    cutoff = time.time() - (max_age_hours * 3600)
                    removed = await self.redis_client.zremrangebyscore(key_str, '-inf', cutoff)
                    if removed > 0:
                        cleaned += removed

            logger.info(f"Cleaned up {cleaned} expired rate limit entries")
            return cleaned

        except Exception as e:
            logger.error(f"Failed to cleanup expired data: {e}")
            return 0

    async def reset_limit(self, identifier: str) -> bool:
        """Reset a specific rate limit

        Args:
            identifier: Rate limit identifier to reset

        Returns:
            True if reset successful
        """
        key = self._get_key(identifier)

        if self.connected and self.redis_client:
            try:
                await self.redis_client.delete(key)
                logger.info(f"Reset distributed rate limit: {identifier}")
                return True
            except Exception as e:
                logger.error(f"Failed to reset distributed limit {identifier}: {e}")
                return False
        else:
            # Reset local data
            if identifier in self.local_windows:
                del self.local_windows[identifier]
            if identifier in self.local_tokens:
                del self.local_tokens[identifier]
            logger.info(f"Reset local rate limit: {identifier}")
            return True