"""
Test suite for IntelligentCache

Tests cover all aspects of intelligent caching including:
- Multi-strategy caching (LRU, LFU, TTL, Adaptive)
- TTL management and expiration
- Memory management and eviction
- Thread safety and concurrent access
- Cache warming and invalidation
- Performance optimization
"""

import pytest
import asyncio
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.performance.intelligent_cache import (
    IntelligentCache,
    CacheStrategy,
    CacheKey,
    CacheEntry,
    CacheStats
)


class TestIntelligentCache:
    """Test suite for IntelligentCache functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cache = IntelligentCache(
            max_size=100,
            max_memory_mb=10,
            default_ttl=3600,
            strategy=CacheStrategy.LRU,
            enable_metrics=True
        )

    def teardown_method(self):
        """Clean up after tests."""
        asyncio.run(self.cache.stop())

    def test_initialization(self):
        """Test cache initialization."""
        assert self.cache.max_size == 100
        assert self.cache.default_ttl == 3600
        assert self.cache.strategy == CacheStrategy.LRU
        assert self.cache.enable_metrics is True
        assert len(self.cache) == 0

    def test_basic_get_set(self):
        """Test basic cache get and set operations."""
        # Test set and get
        result = self.cache.set("test_key", "test_value", ttl=60)
        assert result is True

        value = self.cache.get("test_key")
        assert value == "test_value"

        # Test cache stats
        stats = self.cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 0
        assert stats.hit_rate == 100.0

    def test_ttl_expiration(self):
        """Test TTL-based cache expiration."""
        # Set with short TTL
        self.cache.set("expire_key", "expire_value", ttl=1)

        # Should be available immediately
        assert self.cache.get("expire_key") == "expire_value"

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired now
        assert self.cache.get("expire_key") is None

    def test_cache_key_object(self):
        """Test using CacheKey objects."""
        cache_key = CacheKey(
            key="test_key",
            namespace="test_ns",
            tags={"type", "test"},
            priority=5
        )

        self.cache.set(cache_key, "test_value")
        assert self.cache.get(cache_key) == "test_value"
        assert self.cache.exists(cache_key) is True

    def test_cache_miss(self):
        """Test cache miss scenarios."""
        # Non-existent key
        assert self.cache.get("non_existent") is None
        assert self.cache.get("non_existent", "default") == "default"

        # Check stats
        stats = self.cache.get_stats()
        assert stats.misses == 2

    def test_cache_delete(self):
        """Test cache deletion."""
        self.cache.set("delete_key", "delete_value")
        assert self.cache.get("delete_key") == "delete_value"

        # Delete key
        result = self.cache.delete("delete_key")
        assert result is True

        # Should be gone
        assert self.cache.get("delete_key") is None

        # Delete non-existent key
        result = self.cache.delete("non_existent")
        assert result is False

    def test_cache_clear(self):
        """Test cache clearing."""
        # Add some entries
        for i in range(10):
            self.cache.set(f"key_{i}", f"value_{i}")

        assert len(self.cache) == 10

        # Clear all
        self.cache.clear()
        assert len(self.cache) == 0

    def test_clear_by_namespace(self):
        """Test clearing by namespace."""
        # Add entries in different namespaces
        self.cache.set(CacheKey("key1", "ns1"), "value1")
        self.cache.set(CacheKey("key2", "ns1"), "value2")
        self.cache.set(CacheKey("key3", "ns2"), "value3")

        assert len(self.cache) == 3

        # Clear namespace ns1
        self.cache.clear(namespace="ns1")
        assert len(self.cache) == 1
        assert self.cache.get(CacheKey("key3", "ns2")) == "value3"

    def test_clear_by_tags(self):
        """Test clearing by tags."""
        # Add entries with different tags
        self.cache.set("key1", "value1", tags={"tag1", "tag2"})
        self.cache.set("key2", "value2", tags={"tag2", "tag3"})
        self.cache.set("key3", "value3", tags={"tag3", "tag4"})

        assert len(self.cache) == 3

        # Clear by tag2
        self.cache.clear(tags={"tag2"})
        assert len(self.cache) == 1
        assert self.cache.get("key3") == "value3"

    def test_lru_eviction(self):
        """Test LRU eviction strategy."""
        cache = IntelligentCache(max_size=3, strategy=CacheStrategy.LRU)

        # Fill cache
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        assert len(cache) == 3

        # Access key1 to make it recently used
        cache.get("key1")

        # Add new key, should evict key2 (least recently used)
        cache.set("key4", "value4")

        assert len(cache) == 3
        assert cache.get("key1") == "value1"  # Still there
        assert cache.get("key2") is None      # Evicted
        assert cache.get("key3") == "value3"  # Still there
        assert cache.get("key4") == "value4"  # New entry

    def test_lfu_eviction(self):
        """Test LFU eviction strategy."""
        cache = IntelligentCache(max_size=3, strategy=CacheStrategy.LFU)

        # Fill cache
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 multiple times
        for _ in range(5):
            cache.get("key1")

        # Access key2 once
        cache.get("key2")

        # key3 has no accesses, key2 has 1, key1 has 5
        # Adding new key should evict key3 (least frequently used)
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"  # Most frequent
        assert cache.get("key2") == "value2"  # Second most frequent
        assert cache.get("key3") is None      # Evicted (least frequent)
        assert cache.get("key4") == "value4"  # New entry

    def test_priority_eviction(self):
        """Test priority-based eviction."""
        cache = IntelligentCache(max_size=3, strategy=CacheStrategy.TTL_PRIORITY)

        # Add entries with different priorities
        cache.set("low_priority", "value1", priority=1)
        cache.set("med_priority", "value2", priority=5)
        cache.set("high_priority", "value3", priority=10)

        assert len(cache) == 3

        # Add new entry, should evict lowest priority
        cache.set("new_entry", "value4", priority=7)

        assert cache.get("low_priority") is None    # Evicted (lowest priority)
        assert cache.get("med_priority") == "value2"
        assert cache.get("high_priority") == "value3"
        assert cache.get("new_entry") == "value4"

    def test_adaptive_strategy(self):
        """Test adaptive caching strategy."""
        cache = IntelligentCache(max_size=5, strategy=CacheStrategy.ADAPTIVE)

        # Add entries with different access patterns
        cache.set("frequent", "value1", priority=5)
        cache.set("recent", "value2", priority=3)
        cache.set("old", "value3", priority=1)

        # Access frequent entry multiple times
        for _ in range(10):
            cache.get("frequent")

        # Access recent entry once
        cache.get("recent")

        # Don't access old entry

        # Fill cache to trigger eviction
        cache.set("new1", "newvalue1")
        cache.set("new2", "newvalue2")
        cache.set("new3", "newvalue3")  # Should trigger eviction

        # Frequent entry should still be there due to high access count and priority
        assert cache.get("frequent") == "value1"

    def test_memory_limit_eviction(self):
        """Test memory-based eviction."""
        cache = IntelligentCache(max_memory_mb=1, max_size=1000)  # 1MB limit

        # Add large entries
        large_value = "x" * 100000  # 100KB string

        for i in range(15):  # Should exceed 1MB
            cache.set(f"large_key_{i}", large_value)

        # Should have evicted some entries due to memory limit
        assert len(cache) < 15

    def test_cache_warming(self):
        """Test cache warming functionality."""
        def data_loader():
            return "warmed_value"

        # Warm cache
        self.cache.warm_cache("warm_key", data_loader)

        # Should be immediately available
        assert self.cache.get("warm_key") == "warmed_value"

    def test_cache_warming_with_error(self):
        """Test cache warming with loader error."""
        def failing_loader():
            raise Exception("Loader failed")

        # Should not crash
        self.cache.warm_cache("fail_key", failing_loader)

        # Key should not exist
        assert self.cache.get("fail_key") is None

    def test_invalidation_by_tags(self):
        """Test cache invalidation by tags."""
        # Add entries with tags
        self.cache.set("key1", "value1", tags={"user:123", "type:profile"})
        self.cache.set("key2", "value2", tags={"user:123", "type:settings"})
        self.cache.set("key3", "value3", tags={"user:456", "type:profile"})

        assert len(self.cache) == 3

        # Invalidate by user tag
        self.cache.invalidate_by_tags({"user:123"})

        # Should remove key1 and key2, keep key3
        assert len(self.cache) == 1
        assert self.cache.get("key3") == "value3"

    def test_get_keys(self):
        """Test getting cache keys."""
        # Add keys in different namespaces
        self.cache.set(CacheKey("key1", "ns1"), "value1")
        self.cache.set(CacheKey("key2", "ns1"), "value2")
        self.cache.set(CacheKey("key3", "ns2"), "value3")

        # Get all keys
        all_keys = self.cache.get_keys()
        assert len(all_keys) == 3

        # Get keys by namespace
        ns1_keys = self.cache.get_keys(namespace="ns1")
        assert len(ns1_keys) == 2

    def test_exists(self):
        """Test key existence checking."""
        assert self.cache.exists("non_existent") is False

        self.cache.set("existing", "value", ttl=60)
        assert self.cache.exists("existing") is True

        # Test with expired key
        self.cache.set("expired", "value", ttl=1)
        time.sleep(1.1)
        assert self.cache.exists("expired") is False

    def test_thread_safety(self):
        """Test thread safety of cache operations."""
        def worker(thread_id):
            for i in range(100):
                key = f"thread_{thread_id}_key_{i}"
                value = f"thread_{thread_id}_value_{i}"

                self.cache.set(key, value)
                retrieved = self.cache.get(key)
                assert retrieved == value

        # Start multiple threads
        threads = []
        for thread_id in range(5):
            thread = threading.Thread(target=worker, args=(thread_id,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify cache integrity
        stats = self.cache.get_stats()
        assert stats.hits > 0

    @pytest.mark.asyncio
    async def test_background_cleanup(self):
        """Test background cleanup of expired entries."""
        # Create cache with short cleanup interval
        cache = IntelligentCache(
            max_size=100,
            max_memory_mb=10,
            default_ttl=3600,
            strategy=CacheStrategy.LRU,
            enable_metrics=True,
            cleanup_interval=1  # 1 second cleanup interval
        )

        # Start cache
        await cache.start()

        # Add entry with short TTL
        cache.set("cleanup_test", "value", ttl=1)
        assert cache.get("cleanup_test") == "value"

        # Wait for expiration and cleanup
        await asyncio.sleep(2.5)  # Wait longer to ensure cleanup runs

        # Entry should be cleaned up
        assert len(cache) == 0

        await cache.stop()

    def test_cache_stats(self):
        """Test cache statistics collection."""
        # Perform various operations
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")

        self.cache.get("key1")  # Hit
        self.cache.get("key1")  # Hit
        self.cache.get("non_existent")  # Miss

        stats = self.cache.get_stats()
        assert stats.hits == 2
        assert stats.misses == 1
        assert stats.hit_rate == 66.67  # 2 hits out of 3 total = 66.67%
        assert stats.total_entries == 2

    def test_cache_contains(self):
        """Test __contains__ method."""
        self.cache.set("test_key", "test_value")

        assert "test_key" in self.cache
        assert "non_existent" not in self.cache

        # Test with CacheKey
        cache_key = CacheKey("cache_key_test", "ns")
        self.cache.set(cache_key, "value")
        assert cache_key in self.cache

    def test_cache_len(self):
        """Test __len__ method."""
        assert len(self.cache) == 0

        for i in range(10):
            self.cache.set(f"key_{i}", f"value_{i}")

        assert len(self.cache) == 10

    def test_large_value_handling(self):
        """Test handling of large values."""
        # Create large value
        large_value = "x" * 1000000  # 1MB string

        result = self.cache.set("large_key", large_value)
        assert result is True

        retrieved = self.cache.get("large_key")
        assert retrieved == large_value

    def test_complex_data_types(self):
        """Test caching complex data types."""
        test_data = {
            "list": [1, 2, 3, {"nested": "dict"}],
            "dict": {"key": "value", "number": 42},
            "tuple": (1, "two", 3.0),
            "nested": {
                "level1": {
                    "level2": {
                        "data": "deep_value"
                    }
                }
            }
        }

        self.cache.set("complex_key", test_data)
        retrieved = self.cache.get("complex_key")

        assert retrieved == test_data
        assert retrieved["list"][3]["nested"] == "dict"
        assert retrieved["nested"]["level1"]["level2"]["data"] == "deep_value"

    def test_performance_metrics(self):
        """Test performance metrics collection."""
        # Create cache with larger size to avoid evictions
        cache = IntelligentCache(max_size=2000, strategy=CacheStrategy.LRU)

        # Perform operations and measure
        start_time = time.time()

        for i in range(1000):
            cache.set(f"perf_key_{i}", f"perf_value_{i}")

        for i in range(1000):
            cache.get(f"perf_key_{i}")

        end_time = time.time()
        total_time = end_time - start_time

        # Should be fast (< 1 second for 2000 operations)
        assert total_time < 1.0

        stats = cache.get_stats()
        assert stats.hits == 1000
        assert stats.average_access_time_ms > 0


if __name__ == '__main__':
    pytest.main([__file__])