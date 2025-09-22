"""
Intelligent Cache System

Advanced multi-level caching with TTL, LRU eviction, cache warming,
and intelligent invalidation for sub-100ms response times.
"""

import time
import threading
import hashlib
import pickle
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Set
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategies for different use cases"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL_PRIORITY = "ttl_priority"  # TTL-based eviction
    ADAPTIVE = "adaptive"  # Adaptive strategy based on usage patterns


@dataclass
class CacheKey:
    """Cache key with metadata"""
    key: str
    namespace: str = "default"
    tags: Set[str] = field(default_factory=set)
    priority: int = 1  # 1-10, higher is more important

    def __str__(self) -> str:
        return f"{self.namespace}:{self.key}"

    def __hash__(self) -> int:
        return hash((self.namespace, self.key))


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: CacheKey
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    accessed_at: datetime
    access_count: int = 0
    size_bytes: int = 0
    tags: Set[str] = field(default_factory=set)
    priority: int = 1

    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def access(self):
        """Mark entry as accessed"""
        self.accessed_at = datetime.now()
        self.access_count += 1

    def time_to_live(self) -> Optional[float]:
        """Get remaining TTL in seconds"""
        if self.expires_at is None:
            return None
        remaining = (self.expires_at - datetime.now()).total_seconds()
        return max(0, remaining)


@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    total_entries: int = 0
    average_access_time_ms: float = 0.0
    hit_rate: float = 0.0
    memory_usage_mb: float = 0.0

    def update_hit_rate(self):
        """Update hit rate calculation"""
        total_requests = self.hits + self.misses
        self.hit_rate = round((self.hits / total_requests * 100), 2) if total_requests > 0 else 0.0


class IntelligentCache:
    """
    Advanced multi-level cache with intelligent eviction, TTL management,
    and performance optimization features.
    """

    def __init__(
        self,
        max_size: int = 10000,
        max_memory_mb: int = 512,
        default_ttl: int = 3600,  # 1 hour
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        enable_metrics: bool = True,
        cleanup_interval: int = 300  # seconds
    ):
        """Initialize intelligent cache

        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl: Default TTL in seconds
            strategy: Cache eviction strategy
            enable_metrics: Enable performance metrics
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.strategy = strategy
        self.enable_metrics = enable_metrics

        # Cache storage
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order = OrderedDict()  # For LRU
        self._frequency_counter = defaultdict(int)  # For LFU

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self.stats = CacheStats()

        # Cache warming and invalidation
        self._warming_callbacks: Dict[str, Callable] = {}
        self._invalidation_rules: List[Callable[[CacheEntry], bool]] = []

        # Background tasks
        self._cleanup_interval = cleanup_interval
        self._cleanup_task = None
        self._running = False

        # Performance tracking
        self._access_times: List[float] = []

        logger.info(f"Initialized IntelligentCache with strategy: {strategy.value}")

    async def start(self):
        """Start background tasks"""
        self._running = True
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._background_cleanup())
        logger.info("IntelligentCache background tasks started")

    async def stop(self):
        """Stop background tasks"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("IntelligentCache background tasks stopped")

    def get(self, key: Union[str, CacheKey], default=None) -> Any:
        """Get value from cache"""
        start_time = time.time()

        with self._lock:
            cache_key = self._normalize_key(key)
            key_str = str(cache_key)

            entry = self._cache.get(key_str)

            if entry is None:
                self.stats.misses += 1
                if self.enable_metrics:
                    self._record_access_time(start_time)
                return default

            if entry.is_expired():
                self._remove_entry(key_str)
                self.stats.misses += 1
                if self.enable_metrics:
                    self._record_access_time(start_time)
                return default

            # Update access patterns
            entry.access()
            self._update_access_patterns(key_str)

            self.stats.hits += 1
            self.stats.update_hit_rate()

            if self.enable_metrics:
                self._record_access_time(start_time)

            return entry.value

    def set(
        self,
        key: Union[str, CacheKey],
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[Set[str]] = None,
        priority: int = 1
    ) -> bool:
        """Set value in cache"""
        with self._lock:
            cache_key = self._normalize_key(key)
            key_str = str(cache_key)

            # Calculate TTL
            if ttl is None:
                ttl = self.default_ttl
            expires_at = datetime.now() + timedelta(seconds=ttl) if ttl > 0 else None

            # Calculate size
            size_bytes = self._calculate_size(value)

            # Check memory constraints
            if not self._can_fit(size_bytes, key_str):
                self._evict_entries(size_bytes)

            # Create cache entry
            entry = CacheEntry(
                key=cache_key,
                value=value,
                created_at=datetime.now(),
                expires_at=expires_at,
                accessed_at=datetime.now(),
                access_count=1,
                size_bytes=size_bytes,
                tags=tags or set(),
                priority=priority
            )

            # Remove existing entry if present
            if key_str in self._cache:
                self._remove_entry(key_str)

            # Add new entry
            self._cache[key_str] = entry
            self._update_access_patterns(key_str)

            # Update stats
            self.stats.total_entries = len(self._cache)
            self.stats.total_size_bytes += size_bytes
            self.stats.memory_usage_mb = self.stats.total_size_bytes / (1024 * 1024)

            return True

    def delete(self, key: Union[str, CacheKey]) -> bool:
        """Delete entry from cache"""
        with self._lock:
            cache_key = self._normalize_key(key)
            key_str = str(cache_key)

            if key_str in self._cache:
                self._remove_entry(key_str)
                return True
            return False

    def clear(self, namespace: Optional[str] = None, tags: Optional[Set[str]] = None):
        """Clear cache entries"""
        with self._lock:
            if namespace is None and tags is None:
                # Clear all
                self._cache.clear()
                self._access_order.clear()
                self._frequency_counter.clear()
                self.stats = CacheStats()
            else:
                # Clear by namespace or tags
                keys_to_remove = []
                for key_str, entry in self._cache.items():
                    should_remove = False

                    if namespace and entry.key.namespace == namespace:
                        should_remove = True

                    if tags and tags.intersection(entry.tags):
                        should_remove = True

                    if should_remove:
                        keys_to_remove.append(key_str)

                for key_str in keys_to_remove:
                    self._remove_entry(key_str)

    def invalidate_by_tags(self, tags: Set[str]):
        """Invalidate entries by tags"""
        self.clear(tags=tags)

    def warm_cache(self, key: Union[str, CacheKey], loader: Callable[[], Any]):
        """Warm cache with data loader"""
        cache_key = self._normalize_key(key)
        key_str = str(cache_key)

        # Register warming callback
        self._warming_callbacks[key_str] = loader

        # Load data immediately
        try:
            value = loader()
            self.set(key, value)
            logger.info(f"Cache warmed for key: {key_str}")
        except Exception as e:
            logger.error(f"Failed to warm cache for key {key_str}: {e}")

    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        with self._lock:
            self.stats.total_entries = len(self._cache)
            self.stats.total_size_bytes = sum(entry.size_bytes for entry in self._cache.values())
            self.stats.memory_usage_mb = self.stats.total_size_bytes / (1024 * 1024)

            if self._access_times:
                self.stats.average_access_time_ms = sum(self._access_times) / len(self._access_times) * 1000

            # Update hit rate
            self.stats.update_hit_rate()

            return self.stats

    def get_keys(self, namespace: Optional[str] = None) -> List[str]:
        """Get all cache keys, optionally filtered by namespace"""
        with self._lock:
            if namespace is None:
                return list(self._cache.keys())
            else:
                return [
                    key_str for key_str, entry in self._cache.items()
                    if entry.key.namespace == namespace
                ]

    def exists(self, key: Union[str, CacheKey]) -> bool:
        """Check if key exists and is not expired"""
        with self._lock:
            cache_key = self._normalize_key(key)
            key_str = str(cache_key)

            entry = self._cache.get(key_str)
            if entry is None:
                return False

            if entry.is_expired():
                self._remove_entry(key_str)
                return False

            return True

    def _normalize_key(self, key: Union[str, CacheKey]) -> CacheKey:
        """Normalize key to CacheKey object"""
        if isinstance(key, str):
            return CacheKey(key=key)
        return key

    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes"""
        try:
            return len(pickle.dumps(value))
        except Exception:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(
                    self._calculate_size(k) + self._calculate_size(v)
                    for k, v in value.items()
                )
            else:
                return 64  # Default estimate

    def _can_fit(self, size_bytes: int, exclude_key: str = None) -> bool:
        """Check if new entry can fit in cache"""
        current_size = sum(
            entry.size_bytes for key, entry in self._cache.items()
            if key != exclude_key
        )

        # Check both size and memory constraints
        would_exceed_size = len(self._cache) >= self.max_size
        would_exceed_memory = (current_size + size_bytes) > self.max_memory_bytes

        return not (would_exceed_size or would_exceed_memory)

    def _evict_entries(self, needed_bytes: int):
        """Evict entries based on strategy"""
        if self.strategy == CacheStrategy.LRU:
            self._evict_lru(needed_bytes)
        elif self.strategy == CacheStrategy.LFU:
            self._evict_lfu(needed_bytes)
        elif self.strategy == CacheStrategy.FIFO:
            self._evict_fifo(needed_bytes)
        elif self.strategy == CacheStrategy.TTL_PRIORITY:
            self._evict_ttl_priority(needed_bytes)
        elif self.strategy == CacheStrategy.ADAPTIVE:
            self._evict_adaptive(needed_bytes)

    def _evict_lru(self, needed_bytes: int):
        """Evict least recently used entries"""
        freed_bytes = 0
        while freed_bytes < needed_bytes and self._access_order:
            key_str = next(iter(self._access_order))
            freed_bytes += self._cache[key_str].size_bytes
            self._remove_entry(key_str)

    def _evict_lfu(self, needed_bytes: int):
        """Evict least frequently used entries"""
        freed_bytes = 0
        sorted_keys = sorted(
            self._cache.keys(),
            key=lambda k: self._frequency_counter[k]
        )

        for key_str in sorted_keys:
            if freed_bytes >= needed_bytes:
                break
            freed_bytes += self._cache[key_str].size_bytes
            self._remove_entry(key_str)

    def _evict_fifo(self, needed_bytes: int):
        """Evict first in, first out"""
        freed_bytes = 0
        sorted_keys = sorted(
            self._cache.keys(),
            key=lambda k: self._cache[k].created_at
        )

        for key_str in sorted_keys:
            if freed_bytes >= needed_bytes:
                break
            freed_bytes += self._cache[key_str].size_bytes
            self._remove_entry(key_str)

    def _evict_ttl_priority(self, needed_bytes: int):
        """Evict based on TTL and priority"""
        freed_bytes = 0

        # Sort by TTL (ascending) and priority (ascending)
        sorted_keys = sorted(
            self._cache.keys(),
            key=lambda k: (
                self._cache[k].time_to_live() or float('inf'),
                self._cache[k].priority
            )
        )

        for key_str in sorted_keys:
            if freed_bytes >= needed_bytes:
                break
            freed_bytes += self._cache[key_str].size_bytes
            self._remove_entry(key_str)

    def _evict_adaptive(self, needed_bytes: int):
        """Adaptive eviction based on access patterns"""
        freed_bytes = 0

        # Score entries based on multiple factors
        scored_keys = []
        for key_str, entry in self._cache.items():
            # Lower score = more likely to evict
            score = (
                entry.access_count * 0.3 +  # Frequency factor
                (time.time() - entry.accessed_at.timestamp()) * -0.2 +  # Recency factor (negative for recent)
                entry.priority * 0.3 +  # Priority factor
                (entry.time_to_live() or 3600) * 0.2  # TTL factor
            )
            scored_keys.append((score, key_str))

        # Sort by score (ascending - lowest first)
        scored_keys.sort()

        for score, key_str in scored_keys:
            if freed_bytes >= needed_bytes:
                break
            freed_bytes += self._cache[key_str].size_bytes
            self._remove_entry(key_str)

    def _remove_entry(self, key_str: str):
        """Remove entry and update data structures"""
        if key_str in self._cache:
            entry = self._cache[key_str]

            # Update stats
            self.stats.total_size_bytes -= entry.size_bytes
            self.stats.evictions += 1

            # Remove from data structures
            del self._cache[key_str]
            self._access_order.pop(key_str, None)
            self._frequency_counter.pop(key_str, None)

    def _update_access_patterns(self, key_str: str):
        """Update access patterns for strategies"""
        # Update LRU order
        self._access_order[key_str] = time.time()
        self._access_order.move_to_end(key_str)

        # Update LFU counter
        self._frequency_counter[key_str] += 1

    def _record_access_time(self, start_time: float):
        """Record access time for performance metrics"""
        access_time = time.time() - start_time
        self._access_times.append(access_time)

        # Keep only recent access times (last 1000)
        if len(self._access_times) > 1000:
            self._access_times = self._access_times[-1000:]

    async def _background_cleanup(self):
        """Background task to clean up expired entries"""
        while self._running:
            try:
                await asyncio.sleep(self._cleanup_interval)

                with self._lock:
                    expired_keys = []
                    for key_str, entry in self._cache.items():
                        if entry.is_expired():
                            expired_keys.append(key_str)

                    for key_str in expired_keys:
                        self._remove_entry(key_str)

                    if expired_keys:
                        logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")

    def __len__(self) -> int:
        """Get number of entries in cache"""
        return len(self._cache)

    def __contains__(self, key: Union[str, CacheKey]) -> bool:
        """Check if key exists in cache"""
        return self.exists(key)