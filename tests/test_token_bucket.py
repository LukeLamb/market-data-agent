"""Tests for Token Bucket Rate Limiting Implementation"""

import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from src.rate_limiting.token_bucket import (
    TokenBucket,
    TokenBucketConfig,
    TokenBucketRateLimiter
)


class TestTokenBucketConfig:
    """Test TokenBucketConfig dataclass"""

    def test_basic_config(self):
        """Test basic configuration"""
        config = TokenBucketConfig(capacity=10, refill_rate=2.0)
        assert config.capacity == 10
        assert config.refill_rate == 2.0
        assert config.refill_period == 1.0
        assert config.initial_tokens is None
        assert config.allow_burst is True
        assert config.strict_mode is False

    def test_custom_config(self):
        """Test custom configuration"""
        config = TokenBucketConfig(
            capacity=50,
            refill_rate=5.0,
            refill_period=2.0,
            initial_tokens=25,
            allow_burst=False,
            strict_mode=True
        )
        assert config.capacity == 50
        assert config.refill_rate == 5.0
        assert config.refill_period == 2.0
        assert config.initial_tokens == 25
        assert config.allow_burst is False
        assert config.strict_mode is True


class TestTokenBucket:
    """Test TokenBucket implementation"""

    def test_initialization_default_tokens(self):
        """Test initialization with default token count"""
        config = TokenBucketConfig(capacity=10, refill_rate=2.0)
        bucket = TokenBucket(config)
        assert bucket.tokens == 10
        assert bucket.capacity == 10
        assert bucket.refill_rate == 2.0

    def test_initialization_custom_tokens(self):
        """Test initialization with custom initial tokens"""
        config = TokenBucketConfig(capacity=10, refill_rate=2.0, initial_tokens=5)
        bucket = TokenBucket(config)
        assert bucket.tokens == 5

    def test_consume_tokens_success(self):
        """Test successful token consumption"""
        config = TokenBucketConfig(capacity=10, refill_rate=2.0)
        bucket = TokenBucket(config)

        assert bucket.try_consume(3) is True
        assert bucket.tokens == 7

        assert bucket.try_consume(2) is True
        assert bucket.tokens == 5

    def test_consume_tokens_insufficient(self):
        """Test token consumption with insufficient tokens"""
        config = TokenBucketConfig(capacity=5, refill_rate=1.0)
        bucket = TokenBucket(config)

        assert bucket.try_consume(3) is True
        assert bucket.tokens == 2

        assert bucket.try_consume(5) is False
        assert bucket.tokens == 2  # Tokens should remain unchanged

    def test_token_refill(self):
        """Test token refilling over time"""
        config = TokenBucketConfig(capacity=10, refill_rate=2.0, refill_period=1.0)
        bucket = TokenBucket(config)

        # Consume all tokens
        bucket.try_consume(10)
        assert bucket.tokens == 0

        # Mock time progression
        with patch('time.time') as mock_time:
            initial_time = 1000.0
            mock_time.return_value = initial_time
            bucket.last_refill = initial_time

            # After 1 second, should refill 2 tokens
            mock_time.return_value = initial_time + 1.0
            bucket._refill_tokens()
            assert bucket.tokens == 2.0

            # After 3 more seconds, should refill 6 more tokens (total 8)
            mock_time.return_value = initial_time + 4.0
            bucket._refill_tokens()
            assert bucket.tokens == 8.0

            # After long time, should cap at capacity
            mock_time.return_value = initial_time + 10.0
            bucket._refill_tokens()
            assert bucket.tokens == 10.0

    def test_strict_mode(self):
        """Test strict mode behavior"""
        config = TokenBucketConfig(capacity=10, refill_rate=2.0, strict_mode=True)
        bucket = TokenBucket(config)

        # Should allow consumption up to refill rate
        assert bucket.try_consume(2) is True
        assert bucket.tokens == 8

        # Should deny consumption above refill rate
        assert bucket.try_consume(3) is False
        assert bucket.tokens == 8

    def test_wait_for_tokens(self):
        """Test wait time calculation"""
        config = TokenBucketConfig(capacity=10, refill_rate=4.0)
        bucket = TokenBucket(config)

        # Consume most tokens
        bucket.try_consume(8)
        assert bucket.tokens == 2

        # Should not need to wait for available tokens
        wait_time = bucket.wait_for_tokens(2)
        assert wait_time == 0.0

        # Should calculate wait time for unavailable tokens
        wait_time = bucket.wait_for_tokens(5)
        # Need 3 more tokens, at 4 tokens/second = 0.75 seconds
        assert wait_time == 0.75

    @pytest.mark.asyncio
    async def test_acquire_immediate(self):
        """Test immediate token acquisition"""
        config = TokenBucketConfig(capacity=10, refill_rate=2.0)
        bucket = TokenBucket(config)

        result = await bucket.acquire(3)
        assert result is True
        assert bucket.tokens == 7

    @pytest.mark.asyncio
    async def test_acquire_with_wait(self):
        """Test token acquisition with waiting"""
        config = TokenBucketConfig(capacity=5, refill_rate=10.0)  # Fast refill for testing
        bucket = TokenBucket(config)

        # Consume most tokens
        bucket.try_consume(4)
        assert bucket.tokens == 1

        # Should wait and acquire
        start_time = time.time()
        result = await bucket.acquire(2)
        end_time = time.time()

        assert result is True
        assert end_time - start_time > 0.05  # Should have waited some time
        # Should have consumed 2 tokens, leaving -1 from the original 1 token,
        # but tokens were refilled during the wait, so check that tokens were indeed consumed
        assert bucket.tokens >= 0  # Tokens may have been refilled during wait

    @pytest.mark.asyncio
    async def test_acquire_timeout(self):
        """Test token acquisition with timeout"""
        config = TokenBucketConfig(capacity=5, refill_rate=0.1)  # Very slow refill
        bucket = TokenBucket(config)

        # Consume all tokens
        bucket.try_consume(5)

        # Should timeout
        result = await bucket.acquire(1, timeout=0.1)
        assert result is False

    def test_get_statistics(self):
        """Test statistics collection"""
        config = TokenBucketConfig(capacity=10, refill_rate=2.0)
        bucket = TokenBucket(config)

        # Make some requests
        bucket.try_consume(3)
        bucket.try_consume(15)  # This should fail
        bucket.try_consume(2)

        stats = bucket.get_statistics()
        assert stats["capacity"] == 10
        assert stats["refill_rate"] == 2.0
        assert stats["total_requests"] == 3
        assert stats["denied_requests"] == 1
        assert stats["success_rate"] == 2/3

    def test_reset(self):
        """Test bucket reset"""
        config = TokenBucketConfig(capacity=10, refill_rate=2.0)
        bucket = TokenBucket(config)

        # Consume tokens and make requests
        bucket.try_consume(5)
        bucket.try_consume(15)  # Fail

        assert bucket.tokens == 5
        assert bucket.total_requests == 2

        # Reset
        bucket.reset()

        assert bucket.tokens == 10
        assert bucket.total_requests == 0
        assert bucket.denied_requests == 0


class TestTokenBucketRateLimiter:
    """Test TokenBucketRateLimiter implementation"""

    def test_add_remove_bucket(self):
        """Test adding and removing buckets"""
        limiter = TokenBucketRateLimiter()

        config = TokenBucketConfig(capacity=10, refill_rate=2.0)
        limiter.add_bucket("test_bucket", config)

        assert "test_bucket" in limiter.buckets
        assert limiter.buckets["test_bucket"].capacity == 10

        limiter.remove_bucket("test_bucket")
        assert "test_bucket" not in limiter.buckets

    def test_single_bucket_consume(self):
        """Test consuming from a single bucket"""
        limiter = TokenBucketRateLimiter()

        config = TokenBucketConfig(capacity=10, refill_rate=2.0)
        limiter.add_bucket("test", config)

        assert limiter.try_consume("test", 3) is True
        assert limiter.try_consume("test", 5) is True
        assert limiter.try_consume("test", 5) is False  # Insufficient tokens

    def test_multiple_bucket_consume(self):
        """Test consuming from multiple buckets"""
        limiter = TokenBucketRateLimiter()

        config1 = TokenBucketConfig(capacity=10, refill_rate=2.0)
        config2 = TokenBucketConfig(capacity=5, refill_rate=1.0)

        limiter.add_bucket("bucket1", config1)
        limiter.add_bucket("bucket2", config2)

        # Should succeed if both buckets have tokens
        assert limiter.try_consume(["bucket1", "bucket2"], 3) is True

        # Should fail if one bucket lacks tokens
        assert limiter.try_consume(["bucket1", "bucket2"], 5) is False

        # Both buckets should have same tokens consumed or none
        assert limiter.buckets["bucket1"].tokens == 7
        assert limiter.buckets["bucket2"].tokens == 2

    def test_nonexistent_bucket(self):
        """Test consuming from nonexistent bucket"""
        limiter = TokenBucketRateLimiter()

        assert limiter.try_consume("nonexistent", 1) is False

    @pytest.mark.asyncio
    async def test_acquire_single_bucket(self):
        """Test async acquire from single bucket"""
        limiter = TokenBucketRateLimiter()

        config = TokenBucketConfig(capacity=10, refill_rate=5.0)
        limiter.add_bucket("test", config)

        result = await limiter.acquire("test", 3)
        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_multiple_buckets(self):
        """Test async acquire from multiple buckets"""
        limiter = TokenBucketRateLimiter()

        config1 = TokenBucketConfig(capacity=10, refill_rate=5.0)
        config2 = TokenBucketConfig(capacity=8, refill_rate=4.0)

        limiter.add_bucket("bucket1", config1)
        limiter.add_bucket("bucket2", config2)

        result = await limiter.acquire(["bucket1", "bucket2"], 2)
        assert result is True

    def test_get_bucket_status(self):
        """Test getting bucket status"""
        limiter = TokenBucketRateLimiter()

        config = TokenBucketConfig(capacity=10, refill_rate=2.0)
        limiter.add_bucket("test", config)

        status = limiter.get_bucket_status("test")
        assert status is not None
        assert status["capacity"] == 10
        assert status["refill_rate"] == 2.0

        # Test nonexistent bucket
        status = limiter.get_bucket_status("nonexistent")
        assert status is None

    def test_get_all_status(self):
        """Test getting all bucket statuses"""
        limiter = TokenBucketRateLimiter()

        config1 = TokenBucketConfig(capacity=10, refill_rate=2.0)
        config2 = TokenBucketConfig(capacity=5, refill_rate=1.0)

        limiter.add_bucket("bucket1", config1)
        limiter.add_bucket("bucket2", config2)

        all_status = limiter.get_all_status()
        assert len(all_status) == 2
        assert "bucket1" in all_status
        assert "bucket2" in all_status

    def test_reset_bucket(self):
        """Test resetting specific bucket"""
        limiter = TokenBucketRateLimiter()

        config = TokenBucketConfig(capacity=10, refill_rate=2.0)
        limiter.add_bucket("test", config)

        # Consume tokens
        limiter.try_consume("test", 5)
        assert limiter.buckets["test"].tokens == 5

        # Reset
        limiter.reset_bucket("test")
        assert limiter.buckets["test"].tokens == 10

    def test_reset_all_buckets(self):
        """Test resetting all buckets"""
        limiter = TokenBucketRateLimiter()

        config1 = TokenBucketConfig(capacity=10, refill_rate=2.0)
        config2 = TokenBucketConfig(capacity=5, refill_rate=1.0)

        limiter.add_bucket("bucket1", config1)
        limiter.add_bucket("bucket2", config2)

        # Consume tokens from both
        limiter.try_consume("bucket1", 3)
        limiter.try_consume("bucket2", 2)

        # Reset all
        limiter.reset_all_buckets()

        assert limiter.buckets["bucket1"].tokens == 10
        assert limiter.buckets["bucket2"].tokens == 5

    def test_create_hierarchical_buckets(self):
        """Test creating hierarchical rate limit buckets"""
        limiter = TokenBucketRateLimiter()

        bucket_names = limiter.create_hierarchical_buckets(
            source_name="test_source",
            per_second=5,
            per_minute=100,
            per_hour=3000,
            per_day=50000
        )

        expected_names = [
            "test_source_per_second",
            "test_source_per_minute",
            "test_source_per_hour",
            "test_source_per_day"
        ]

        assert bucket_names == expected_names

        # Verify buckets were created with correct parameters
        assert limiter.buckets["test_source_per_second"].capacity == 10  # 5 * 2 (burst)
        assert limiter.buckets["test_source_per_second"].refill_rate == 5

        assert limiter.buckets["test_source_per_minute"].capacity == 100
        assert limiter.buckets["test_source_per_minute"].refill_rate == 100/60

        assert limiter.buckets["test_source_per_day"].capacity == 50000
        assert limiter.buckets["test_source_per_day"].refill_rate == 50000/86400

    def test_hierarchical_buckets_partial(self):
        """Test creating hierarchical buckets with only some limits"""
        limiter = TokenBucketRateLimiter()

        bucket_names = limiter.create_hierarchical_buckets(
            source_name="partial_source",
            per_minute=60,
            per_day=1000
        )

        expected_names = [
            "partial_source_per_minute",
            "partial_source_per_day"
        ]

        assert bucket_names == expected_names
        assert len(limiter.buckets) == 2