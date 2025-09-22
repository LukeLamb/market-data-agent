"""
Query Optimizer

Database query optimization and connection pooling for improved
database performance and efficient resource utilization.
"""

import asyncio
import aiosqlite
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import hashlib
import logging

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Query optimization strategies"""
    AGGRESSIVE = "aggressive"  # Maximum optimization
    BALANCED = "balanced"     # Balance between performance and resource usage
    CONSERVATIVE = "conservative"  # Minimal optimization
    ADAPTIVE = "adaptive"     # Adapt based on query patterns


@dataclass
class QueryPlan:
    """Query execution plan"""
    query_hash: str
    original_query: str
    optimized_query: str
    indexes_used: List[str]
    estimated_cost: float
    execution_time_ms: float = 0.0
    rows_examined: int = 0
    rows_returned: int = 0
    cache_hit: bool = False
    optimization_applied: List[str] = field(default_factory=list)

    def efficiency_score(self) -> float:
        """Calculate query efficiency score (0-100)"""
        if self.rows_examined == 0:
            return 100.0

        # Base score on selectivity (rows returned / rows examined)
        selectivity = self.rows_returned / self.rows_examined
        selectivity_score = min(100.0, selectivity * 100)

        # Adjust for execution time
        if self.execution_time_ms < 10:
            time_score = 100.0
        elif self.execution_time_ms < 100:
            time_score = 90.0
        elif self.execution_time_ms < 500:
            time_score = 70.0
        elif self.execution_time_ms < 1000:
            time_score = 50.0
        else:
            time_score = 20.0

        # Combine scores
        return (selectivity_score * 0.6) + (time_score * 0.4)


@dataclass
class ConnectionStats:
    """Connection pool statistics"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    connections_created: int = 0
    connections_closed: int = 0
    total_queries: int = 0
    average_query_time: float = 0.0
    connection_wait_time: float = 0.0


class ConnectionPool:
    """Async connection pool for SQLite databases"""

    def __init__(
        self,
        database_path: str,
        max_connections: int = 10,
        min_connections: int = 2,
        connection_timeout: float = 30.0,
        idle_timeout: float = 300.0  # 5 minutes
    ):
        """Initialize connection pool

        Args:
            database_path: Path to SQLite database
            max_connections: Maximum number of connections
            min_connections: Minimum number of connections to maintain
            connection_timeout: Timeout for acquiring connection
            idle_timeout: Time before idle connections are closed
        """
        self.database_path = database_path
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.connection_timeout = connection_timeout
        self.idle_timeout = idle_timeout

        # Connection management
        self._pool: deque = deque()
        self._in_use: set = set()
        self._created_count = 0
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)

        # Statistics
        self.stats = ConnectionStats()

        # Background cleanup
        self._running = False
        self._cleanup_task = None

        logger.info(f"Initialized ConnectionPool for {database_path}")

    async def start(self):
        """Start the connection pool"""
        self._running = True

        # Create minimum connections
        async with self._lock:
            for _ in range(self.min_connections):
                conn = await self._create_connection()
                self._pool.append({
                    'connection': conn,
                    'created_at': datetime.now(),
                    'last_used': datetime.now()
                })

        # Start cleanup task
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_idle_connections())

        logger.info("ConnectionPool started")

    async def stop(self):
        """Stop the connection pool"""
        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        async with self._lock:
            while self._pool:
                conn_data = self._pool.popleft()
                await conn_data['connection'].close()

            for conn_data in self._in_use:
                await conn_data['connection'].close()
            self._in_use.clear()

        logger.info("ConnectionPool stopped")

    async def acquire(self) -> aiosqlite.Connection:
        """Acquire a connection from the pool"""
        start_time = time.time()

        async with self._condition:
            # Wait for available connection
            while not self._pool and len(self._in_use) >= self.max_connections:
                try:
                    await asyncio.wait_for(
                        self._condition.wait(),
                        timeout=self.connection_timeout
                    )
                except asyncio.TimeoutError:
                    raise Exception("Connection timeout: no connections available")

            # Get connection from pool or create new one
            if self._pool:
                conn_data = self._pool.popleft()
                conn_data['last_used'] = datetime.now()
            else:
                if len(self._in_use) < self.max_connections:
                    conn = await self._create_connection()
                    conn_data = {
                        'connection': conn,
                        'created_at': datetime.now(),
                        'last_used': datetime.now()
                    }
                else:
                    raise Exception("Maximum connections reached")

            self._in_use.add(conn_data)

        # Update statistics
        wait_time = time.time() - start_time
        self.stats.connection_wait_time = (
            (self.stats.connection_wait_time * self.stats.total_queries + wait_time) /
            (self.stats.total_queries + 1)
        ) if self.stats.total_queries > 0 else wait_time

        return conn_data['connection']

    async def release(self, connection: aiosqlite.Connection):
        """Release a connection back to the pool"""
        async with self._condition:
            # Find connection data
            conn_data = None
            for data in self._in_use:
                if data['connection'] is connection:
                    conn_data = data
                    break

            if conn_data:
                self._in_use.remove(conn_data)
                conn_data['last_used'] = datetime.now()

                # Return to pool if under max size
                if len(self._pool) < self.max_connections:
                    self._pool.append(conn_data)
                else:
                    await conn_data['connection'].close()
                    self.stats.connections_closed += 1

                # Notify waiting tasks
                self._condition.notify()

    async def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute query with connection pooling"""
        connection = await self.acquire()
        try:
            start_time = time.time()

            async with connection.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                # Convert to list of dictionaries
                if rows and cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    result = [dict(zip(columns, row)) for row in rows]
                else:
                    result = []

            # Update statistics
            execution_time = time.time() - start_time
            self.stats.total_queries += 1
            self.stats.average_query_time = (
                (self.stats.average_query_time * (self.stats.total_queries - 1) + execution_time) /
                self.stats.total_queries
            )

            return result

        finally:
            await self.release(connection)

    def get_stats(self) -> ConnectionStats:
        """Get connection pool statistics"""
        self.stats.total_connections = len(self._pool) + len(self._in_use)
        self.stats.active_connections = len(self._in_use)
        self.stats.idle_connections = len(self._pool)
        return self.stats

    async def _create_connection(self) -> aiosqlite.Connection:
        """Create a new database connection"""
        conn = await aiosqlite.connect(self.database_path)

        # Enable WAL mode for better concurrency
        await conn.execute("PRAGMA journal_mode=WAL")

        # Optimize for performance
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute("PRAGMA cache_size=10000")
        await conn.execute("PRAGMA temp_store=MEMORY")

        self._created_count += 1
        self.stats.connections_created += 1

        return conn

    async def _cleanup_idle_connections(self):
        """Background task to clean up idle connections"""
        while self._running:
            try:
                await asyncio.sleep(60)  # Run every minute

                cutoff_time = datetime.now() - timedelta(seconds=self.idle_timeout)

                async with self._lock:
                    # Remove idle connections (keep minimum)
                    while (len(self._pool) > self.min_connections and
                           self._pool and
                           self._pool[0]['last_used'] < cutoff_time):
                        conn_data = self._pool.popleft()
                        await conn_data['connection'].close()
                        self.stats.connections_closed += 1

            except Exception as e:
                logger.error(f"Error in connection cleanup: {e}")


class QueryOptimizer:
    """
    Database query optimizer with connection pooling and intelligent
    query caching for improved database performance.
    """

    def __init__(
        self,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        enable_query_cache: bool = True,
        cache_size: int = 1000,
        enable_index_analysis: bool = True
    ):
        """Initialize query optimizer

        Args:
            strategy: Optimization strategy to use
            enable_query_cache: Enable query result caching
            cache_size: Maximum number of cached queries
            enable_index_analysis: Enable index usage analysis
        """
        self.strategy = strategy
        self.enable_query_cache = enable_query_cache
        self.cache_size = cache_size
        self.enable_index_analysis = enable_index_analysis

        # Connection pools (one per database)
        self.connection_pools: Dict[str, ConnectionPool] = {}

        # Query optimization
        self.query_cache: Dict[str, Any] = {}
        self.query_plans: Dict[str, QueryPlan] = {}
        self.query_stats: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Index analysis
        self.index_usage: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.missing_indexes: List[Dict[str, Any]] = []

        # Thread safety
        self._lock = threading.RLock()

        # Background optimization
        self._running = False
        self._optimization_task = None

        logger.info(f"QueryOptimizer initialized with strategy: {strategy.value}")

    async def start(self):
        """Start the query optimizer"""
        self._running = True

        # Start all connection pools
        for pool in self.connection_pools.values():
            await pool.start()

        # Start background optimization
        if self._optimization_task is None:
            self._optimization_task = asyncio.create_task(self._background_optimization())

        logger.info("QueryOptimizer started")

    async def stop(self):
        """Stop the query optimizer"""
        self._running = False

        # Stop background optimization
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass

        # Stop all connection pools
        for pool in self.connection_pools.values():
            await pool.stop()

        logger.info("QueryOptimizer stopped")

    def add_database(self, name: str, database_path: str, **pool_kwargs):
        """Add a database with connection pooling"""
        self.connection_pools[name] = ConnectionPool(database_path, **pool_kwargs)
        logger.info(f"Added database: {name} -> {database_path}")

    async def execute_query(
        self,
        database: str,
        query: str,
        params: tuple = (),
        use_cache: bool = True,
        optimize: bool = True
    ) -> List[Dict[str, Any]]:
        """Execute optimized query"""
        if database not in self.connection_pools:
            raise ValueError(f"Database '{database}' not configured")

        query_hash = self._hash_query(query, params)

        # Check cache first
        if use_cache and self.enable_query_cache and query_hash in self.query_cache:
            cache_entry = self.query_cache[query_hash]
            if not self._is_cache_expired(cache_entry):
                logger.debug(f"Cache hit for query: {query_hash[:8]}...")
                return cache_entry['result']

        # Optimize query if requested
        if optimize:
            optimized_query = self._optimize_query(query)
        else:
            optimized_query = query

        # Execute query
        start_time = time.time()
        pool = self.connection_pools[database]
        result = await pool.execute_query(optimized_query, params)
        execution_time = (time.time() - start_time) * 1000  # Convert to ms

        # Create query plan
        plan = QueryPlan(
            query_hash=query_hash,
            original_query=query,
            optimized_query=optimized_query,
            indexes_used=self._analyze_indexes_used(optimized_query),
            estimated_cost=self._estimate_query_cost(optimized_query),
            execution_time_ms=execution_time,
            rows_examined=len(result),  # Simplified
            rows_returned=len(result)
        )

        # Store query plan and statistics
        with self._lock:
            self.query_plans[query_hash] = plan
            self.query_stats[query_hash].append({
                'execution_time': execution_time,
                'rows_returned': len(result),
                'timestamp': datetime.now()
            })

            # Update index usage statistics
            if self.enable_index_analysis:
                for index in plan.indexes_used:
                    self.index_usage[database][index] += 1

        # Cache result if enabled
        if use_cache and self.enable_query_cache:
            cache_entry = {
                'result': result,
                'cached_at': datetime.now(),
                'ttl': self._get_cache_ttl(query),
                'query_hash': query_hash
            }

            # Manage cache size
            if len(self.query_cache) >= self.cache_size:
                self._evict_cache_entries()

            self.query_cache[query_hash] = cache_entry

        return result

    def get_query_stats(self, query_hash: Optional[str] = None) -> Dict[str, Any]:
        """Get query performance statistics"""
        with self._lock:
            if query_hash:
                if query_hash in self.query_stats:
                    stats = list(self.query_stats[query_hash])
                    return {
                        'query_hash': query_hash,
                        'executions': len(stats),
                        'average_time': sum(s['execution_time'] for s in stats) / len(stats),
                        'min_time': min(s['execution_time'] for s in stats),
                        'max_time': max(s['execution_time'] for s in stats),
                        'recent_executions': stats[-10:]  # Last 10 executions
                    }
                else:
                    return {'query_hash': query_hash, 'no_data': True}
            else:
                # Overall statistics
                all_stats = []
                for hash_key, stats in self.query_stats.items():
                    all_stats.extend(stats)

                if all_stats:
                    return {
                        'total_queries': len(all_stats),
                        'unique_queries': len(self.query_stats),
                        'average_execution_time': sum(s['execution_time'] for s in all_stats) / len(all_stats),
                        'cache_size': len(self.query_cache),
                        'cache_hit_rate': self._calculate_cache_hit_rate()
                    }
                else:
                    return {'no_data': True}

    def get_optimization_recommendations(self, database: str) -> List[Dict[str, Any]]:
        """Get query optimization recommendations"""
        recommendations = []

        with self._lock:
            # Analyze slow queries
            for query_hash, plan in self.query_plans.items():
                if plan.execution_time_ms > 500:  # Slow query threshold
                    recommendations.append({
                        'type': 'slow_query',
                        'query_hash': query_hash,
                        'execution_time': plan.execution_time_ms,
                        'recommendation': 'Consider adding indexes or optimizing query structure',
                        'efficiency_score': plan.efficiency_score()
                    })

            # Analyze index usage
            if database in self.index_usage:
                unused_indexes = []
                for index, usage_count in self.index_usage[database].items():
                    if usage_count == 0:
                        unused_indexes.append(index)

                if unused_indexes:
                    recommendations.append({
                        'type': 'unused_indexes',
                        'indexes': unused_indexes,
                        'recommendation': 'Consider dropping unused indexes to improve write performance'
                    })

            # Suggest missing indexes
            for suggestion in self.missing_indexes:
                if suggestion.get('database') == database:
                    recommendations.append({
                        'type': 'missing_index',
                        'table': suggestion['table'],
                        'columns': suggestion['columns'],
                        'recommendation': f"Consider adding index on {suggestion['table']}({', '.join(suggestion['columns'])})"
                    })

        return recommendations

    def _optimize_query(self, query: str) -> str:
        """Optimize SQL query based on strategy"""
        optimized = query.strip()

        if self.strategy == OptimizationStrategy.CONSERVATIVE:
            # Minimal optimization - just clean up whitespace
            optimized = ' '.join(optimized.split())

        elif self.strategy in [OptimizationStrategy.BALANCED, OptimizationStrategy.AGGRESSIVE]:
            # Apply various optimizations
            optimizations = []

            # Add LIMIT if not present and looks like it might return many rows
            if 'LIMIT' not in optimized.upper() and 'SELECT' in optimized.upper():
                if not any(word in optimized.upper() for word in ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN(']):
                    # This is a heuristic - in practice you'd want more sophisticated analysis
                    pass  # Could add LIMIT based on context

            # Suggest using indexes for WHERE clauses
            if 'WHERE' in optimized.upper():
                # Analyze WHERE conditions for potential index usage
                # This is simplified - real implementation would parse SQL properly
                pass

            # Clean up whitespace
            optimized = ' '.join(optimized.split())

        elif self.strategy == OptimizationStrategy.ADAPTIVE:
            # Adaptive optimization based on historical performance
            query_hash = self._hash_query(query, ())
            if query_hash in self.query_stats:
                stats = list(self.query_stats[query_hash])
                avg_time = sum(s['execution_time'] for s in stats) / len(stats)

                if avg_time > 1000:  # If average > 1 second, apply aggressive optimization
                    optimized = self._apply_aggressive_optimizations(optimized)
                else:
                    optimized = ' '.join(optimized.split())

        return optimized

    def _apply_aggressive_optimizations(self, query: str) -> str:
        """Apply aggressive query optimizations"""
        optimized = query.strip()

        # Force index usage hints (SQLite specific)
        if 'SELECT' in optimized.upper() and 'WHERE' in optimized.upper():
            # This is a simplified example - real implementation would be more sophisticated
            pass

        return ' '.join(optimized.split())

    def _hash_query(self, query: str, params: tuple) -> str:
        """Generate hash for query and parameters"""
        query_str = f"{query.strip()}{str(params)}"
        return hashlib.md5(query_str.encode()).hexdigest()

    def _analyze_indexes_used(self, query: str) -> List[str]:
        """Analyze which indexes are used by query (simplified)"""
        # In a real implementation, you'd use EXPLAIN QUERY PLAN
        indexes = []

        # Simple heuristics for common patterns
        if 'WHERE' in query.upper():
            # Look for common index patterns
            if 'id =' in query.lower():
                indexes.append('primary_key_index')
            if 'created_at' in query.lower():
                indexes.append('created_at_index')
            if 'symbol' in query.lower():
                indexes.append('symbol_index')

        return indexes

    def _estimate_query_cost(self, query: str) -> float:
        """Estimate query execution cost (simplified)"""
        cost = 1.0

        # Simple cost estimation based on query complexity
        if 'JOIN' in query.upper():
            cost += query.upper().count('JOIN') * 2.0

        if 'WHERE' in query.upper():
            cost += 0.5

        if 'ORDER BY' in query.upper():
            cost += 1.0

        if 'GROUP BY' in query.upper():
            cost += 1.5

        return cost

    def _get_cache_ttl(self, query: str) -> int:
        """Get cache TTL for query based on its characteristics"""
        # Simple heuristics for cache TTL
        if any(table in query.lower() for table in ['ohlcv_data', 'prices']):
            return 300  # 5 minutes for price data

        if any(table in query.lower() for table in ['symbols', 'data_sources']):
            return 3600  # 1 hour for reference data

        return 1800  # 30 minutes default

    def _is_cache_expired(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry has expired"""
        cached_at = cache_entry['cached_at']
        ttl = cache_entry['ttl']
        return (datetime.now() - cached_at).total_seconds() > ttl

    def _evict_cache_entries(self):
        """Evict old cache entries to maintain size limit"""
        # Simple LRU eviction - remove oldest 10% of entries
        entries_to_remove = max(1, len(self.query_cache) // 10)

        sorted_entries = sorted(
            self.query_cache.items(),
            key=lambda x: x[1]['cached_at']
        )

        for i in range(entries_to_remove):
            del self.query_cache[sorted_entries[i][0]]

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate (simplified)"""
        # This would need more sophisticated tracking in a real implementation
        return 0.0  # Placeholder

    async def _background_optimization(self):
        """Background task for continuous optimization"""
        while self._running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes

                # Analyze query patterns and suggest optimizations
                with self._lock:
                    # Look for frequently executed slow queries
                    slow_queries = []
                    for query_hash, plan in self.query_plans.items():
                        if plan.execution_time_ms > 200:  # >200ms
                            stats = list(self.query_stats[query_hash])
                            if len(stats) > 10:  # Frequently executed
                                slow_queries.append((query_hash, plan))

                    if slow_queries:
                        logger.info(f"Found {len(slow_queries)} frequently executed slow queries")

                    # Clean up old statistics
                    cutoff_time = datetime.now() - timedelta(hours=24)
                    for query_hash, stats in self.query_stats.items():
                        while stats and stats[0]['timestamp'] < cutoff_time:
                            stats.popleft()

            except Exception as e:
                logger.error(f"Error in query optimization background task: {e}")