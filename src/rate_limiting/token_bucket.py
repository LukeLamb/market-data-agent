"""Token Bucket Rate Limiting Implementation

High-performance token bucket algorithm with burst handling, multiple time windows,
and integration with existing data source rate limiting.
"""

import asyncio
import time
import threading
from typing import Dict, Optional, Union, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class TokenBucketConfig:
    """Configuration for token bucket rate limiter"""
    capacity: int  # Maximum tokens in bucket
    refill_rate: float  # Tokens added per second
    refill_period: float = 1.0  # How often to refill (seconds)
    initial_tokens: Optional[int] = None  # Initial token count (defaults to capacity)
    allow_burst: bool = True  # Allow burst consumption up to capacity
    strict_mode: bool = False  # Strict mode prevents any bursting


class TokenBucket:
    """Thread-safe token bucket implementation for rate limiting

    The token bucket algorithm allows for controlled rate limiting with burst handling.
    Tokens are refilled at a steady rate, and requests consume tokens. When the bucket
    is empty, requests are denied or delayed.
    """

    def __init__(self, config: TokenBucketConfig):
        self.config = config
        self.capacity = config.capacity
        self.refill_rate = config.refill_rate
        self.refill_period = config.refill_period
        self.allow_burst = config.allow_burst
        self.strict_mode = config.strict_mode

        # Initialize token count
        self.tokens = config.initial_tokens if config.initial_tokens is not None else config.capacity
        self.last_refill = time.time()

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self.total_requests = 0
        self.denied_requests = 0
        self.last_denial = None

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill

        if elapsed >= self.refill_period:
            # Calculate tokens to add
            tokens_to_add = (elapsed / self.refill_period) * self.refill_rate

            # Add tokens up to capacity
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now

    def try_consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from the bucket

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were successfully consumed, False otherwise
        """
        with self._lock:
            self._refill_tokens()
            self.total_requests += 1

            # Check if we have enough tokens
            if self.tokens >= tokens:
                # In strict mode, don't allow consuming more than refill rate
                if self.strict_mode and tokens > self.refill_rate:
                    self.denied_requests += 1
                    self.last_denial = datetime.now()
                    return False

                self.tokens -= tokens
                return True
            else:
                self.denied_requests += 1
                self.last_denial = datetime.now()
                return False

    def wait_for_tokens(self, tokens: int = 1) -> float:
        """Calculate how long to wait for tokens to be available

        Args:
            tokens: Number of tokens needed

        Returns:
            Wait time in seconds (0 if tokens are available)
        """
        with self._lock:
            self._refill_tokens()

            if self.tokens >= tokens:
                return 0.0

            # Calculate how long until we have enough tokens
            tokens_needed = tokens - self.tokens
            wait_time = tokens_needed / self.refill_rate

            return wait_time

    async def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """Async acquire tokens, waiting if necessary

        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait (None for unlimited)

        Returns:
            True if tokens were acquired, False if timeout
        """
        start_time = time.time()

        while True:
            # Try to consume tokens immediately
            if self.try_consume(tokens):
                return True

            # Check timeout
            if timeout is not None and (time.time() - start_time) >= timeout:
                return False

            # Calculate wait time
            wait_time = self.wait_for_tokens(tokens)

            # Wait for a short period or until tokens are available
            sleep_time = min(0.1, wait_time)  # Don't sleep too long at once
            await asyncio.sleep(sleep_time)

    def get_available_tokens(self) -> float:
        """Get current number of available tokens"""
        with self._lock:
            self._refill_tokens()
            return self.tokens

    def get_statistics(self) -> Dict:
        """Get bucket statistics"""
        with self._lock:
            self._refill_tokens()
            success_rate = (self.total_requests - self.denied_requests) / max(1, self.total_requests)

            return {
                "capacity": self.capacity,
                "current_tokens": self.tokens,
                "refill_rate": self.refill_rate,
                "total_requests": self.total_requests,
                "denied_requests": self.denied_requests,
                "success_rate": success_rate,
                "last_denial": self.last_denial.isoformat() if self.last_denial else None,
                "utilization": 1.0 - (self.tokens / self.capacity)
            }

    def reset(self) -> None:
        """Reset the bucket to initial state"""
        with self._lock:
            self.tokens = self.config.initial_tokens if self.config.initial_tokens is not None else self.capacity
            self.last_refill = time.time()
            self.total_requests = 0
            self.denied_requests = 0
            self.last_denial = None


class TokenBucketRateLimiter:
    """Multi-bucket rate limiter for different time windows and limits

    Manages multiple token buckets for different rate limiting scenarios:
    - Per-second, per-minute, per-hour, per-day limits
    - Different buckets for different API endpoints or operations
    - Hierarchical rate limiting (global -> source -> endpoint)
    """

    def __init__(self):
        self.buckets: Dict[str, TokenBucket] = {}
        self._lock = threading.RLock()

    def add_bucket(self, name: str, config: TokenBucketConfig) -> None:
        """Add a token bucket with the given configuration"""
        with self._lock:
            self.buckets[name] = TokenBucket(config)
            logger.info(f"Added token bucket '{name}' with capacity {config.capacity} and refill rate {config.refill_rate}/s")

    def remove_bucket(self, name: str) -> None:
        """Remove a token bucket"""
        with self._lock:
            if name in self.buckets:
                del self.buckets[name]
                logger.info(f"Removed token bucket '{name}'")

    def try_consume(self, bucket_names: Union[str, List[str]], tokens: int = 1) -> bool:
        """Try to consume tokens from specified buckets

        Args:
            bucket_names: Name(s) of buckets to check
            tokens: Number of tokens to consume

        Returns:
            True if all buckets have sufficient tokens and consumption succeeded
        """
        if isinstance(bucket_names, str):
            bucket_names = [bucket_names]

        with self._lock:
            # Check all buckets first (don't consume if any would fail)
            for name in bucket_names:
                if name not in self.buckets:
                    logger.warning(f"Token bucket '{name}' not found")
                    return False

                if not self.buckets[name].try_consume(0):  # Check without consuming
                    if self.buckets[name].tokens < tokens:
                        return False

            # If all checks pass, consume from all buckets
            success = True
            consumed_buckets = []

            for name in bucket_names:
                if self.buckets[name].try_consume(tokens):
                    consumed_buckets.append(name)
                else:
                    success = False
                    break

            # If any consumption failed, rollback
            if not success:
                for bucket_name in consumed_buckets:
                    # Add tokens back (best effort rollback)
                    with self.buckets[bucket_name]._lock:
                        self.buckets[bucket_name].tokens = min(
                            self.buckets[bucket_name].capacity,
                            self.buckets[bucket_name].tokens + tokens
                        )
                return False

            return True

    async def acquire(self, bucket_names: Union[str, List[str]], tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """Async acquire tokens from specified buckets

        Args:
            bucket_names: Name(s) of buckets to acquire from
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait

        Returns:
            True if tokens were acquired from all buckets
        """
        if isinstance(bucket_names, str):
            bucket_names = [bucket_names]

        start_time = time.time()

        while True:
            if self.try_consume(bucket_names, tokens):
                return True

            if timeout is not None and (time.time() - start_time) >= timeout:
                return False

            # Calculate wait time based on most restrictive bucket
            max_wait = 0.0
            for name in bucket_names:
                if name in self.buckets:
                    wait_time = self.buckets[name].wait_for_tokens(tokens)
                    max_wait = max(max_wait, wait_time)

            # Wait for a short period
            sleep_time = min(0.1, max_wait)
            await asyncio.sleep(sleep_time)

    def get_bucket_status(self, name: str) -> Optional[Dict]:
        """Get status of a specific bucket"""
        with self._lock:
            if name in self.buckets:
                return self.buckets[name].get_statistics()
            return None

    def get_all_status(self) -> Dict[str, Dict]:
        """Get status of all buckets"""
        with self._lock:
            return {name: bucket.get_statistics() for name, bucket in self.buckets.items()}

    def reset_bucket(self, name: str) -> None:
        """Reset a specific bucket"""
        with self._lock:
            if name in self.buckets:
                self.buckets[name].reset()
                logger.info(f"Reset token bucket '{name}'")

    def reset_all_buckets(self) -> None:
        """Reset all buckets"""
        with self._lock:
            for bucket in self.buckets.values():
                bucket.reset()
            logger.info("Reset all token buckets")

    def create_hierarchical_buckets(self,
                                   source_name: str,
                                   per_second: Optional[int] = None,
                                   per_minute: Optional[int] = None,
                                   per_hour: Optional[int] = None,
                                   per_day: Optional[int] = None) -> List[str]:
        """Create hierarchical buckets for a data source

        Args:
            source_name: Name of the data source
            per_second: Requests per second limit
            per_minute: Requests per minute limit
            per_hour: Requests per hour limit
            per_day: Requests per day limit

        Returns:
            List of bucket names created
        """
        bucket_names = []

        if per_second:
            name = f"{source_name}_per_second"
            config = TokenBucketConfig(
                capacity=per_second * 2,  # Allow burst
                refill_rate=per_second,
                refill_period=1.0
            )
            self.add_bucket(name, config)
            bucket_names.append(name)

        if per_minute:
            name = f"{source_name}_per_minute"
            config = TokenBucketConfig(
                capacity=per_minute,
                refill_rate=per_minute / 60.0,
                refill_period=1.0
            )
            self.add_bucket(name, config)
            bucket_names.append(name)

        if per_hour:
            name = f"{source_name}_per_hour"
            config = TokenBucketConfig(
                capacity=per_hour,
                refill_rate=per_hour / 3600.0,
                refill_period=60.0  # Refill every minute
            )
            self.add_bucket(name, config)
            bucket_names.append(name)

        if per_day:
            name = f"{source_name}_per_day"
            config = TokenBucketConfig(
                capacity=per_day,
                refill_rate=per_day / 86400.0,
                refill_period=3600.0  # Refill every hour
            )
            self.add_bucket(name, config)
            bucket_names.append(name)

        logger.info(f"Created {len(bucket_names)} hierarchical buckets for {source_name}: {bucket_names}")
        return bucket_names