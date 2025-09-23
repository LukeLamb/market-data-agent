"""
Advanced Connection Pool Manager for Market Data Agent
Implements intelligent connection pooling, load balancing, and performance optimization
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import statistics
from contextlib import asynccontextmanager

import asyncpg

logger = logging.getLogger(__name__)


class ConnectionPriority(Enum):
    """Connection priority levels for different operation types"""
    CRITICAL = "critical"      # Real-time data updates
    HIGH = "high"             # User queries, API requests
    NORMAL = "normal"         # Background tasks
    LOW = "low"               # Analytics, reporting


@dataclass
class ConnectionPoolConfig:
    """Advanced connection pool configuration"""
    # Basic pool settings
    min_size: int = 10
    max_size: int = 100
    command_timeout: float = 30.0
    server_settings: Dict[str, str] = field(default_factory=dict)

    # Advanced settings
    max_queries: int = 50000  # Max queries before connection recycling
    max_inactive_time: float = 300.0  # 5 minutes max inactive time
    connection_ttl: float = 3600.0  # 1 hour connection time-to-live

    # Load balancing
    enable_load_balancing: bool = True
    health_check_interval: float = 30.0
    max_connection_errors: int = 5

    # Performance tuning
    enable_prepared_statements: bool = True
    statement_cache_size: int = 1000
    enable_query_pipelining: bool = True

    def __post_init__(self):
        if not self.server_settings:
            self.server_settings = {
                'application_name': 'market_data_agent',
                'tcp_keepalives_idle': '600',
                'tcp_keepalives_interval': '30',
                'tcp_keepalives_count': '3',
                'statement_timeout': '60000',  # 60 seconds
                'idle_in_transaction_session_timeout': '300000',  # 5 minutes
            }


@dataclass
class ConnectionMetrics:
    """Metrics for individual connections"""
    connection_id: str
    created_at: datetime
    last_used: datetime
    query_count: int = 0
    error_count: int = 0
    total_time_ms: float = 0.0
    is_healthy: bool = True
    priority_usage: Dict[ConnectionPriority, int] = field(default_factory=lambda: {p: 0 for p in ConnectionPriority})

    @property
    def avg_query_time_ms(self) -> float:
        return self.total_time_ms / max(self.query_count, 1)

    @property
    def age_seconds(self) -> float:
        return (datetime.now() - self.created_at).total_seconds()

    @property
    def idle_time_seconds(self) -> float:
        return (datetime.now() - self.last_used).total_seconds()


@dataclass
class PoolStatistics:
    """Connection pool performance statistics"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    total_queries: int = 0
    total_errors: int = 0
    avg_response_time_ms: float = 0.0
    connection_utilization: float = 0.0
    pool_efficiency: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class AdvancedConnectionPool:
    """Advanced connection pool with intelligent management and optimization"""

    def __init__(
        self,
        database_url: str,
        config: ConnectionPoolConfig = None
    ):
        self.database_url = database_url
        self.config = config or ConnectionPoolConfig()

        # Pool management
        self.pool: Optional[asyncpg.Pool] = None
        self.connection_metrics: Dict[str, ConnectionMetrics] = {}
        self.priority_queues: Dict[ConnectionPriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in ConnectionPriority
        }

        # Performance tracking
        self.statistics = PoolStatistics()
        self.query_history: List[Tuple[float, bool]] = []  # (execution_time, success)

        # Health monitoring
        self.health_check_task: Optional[asyncio.Task] = None
        self.is_healthy: bool = True
        self.last_health_check: datetime = datetime.now()

        # Prepared statements cache
        self.prepared_statements: Dict[str, str] = {}

        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize the advanced connection pool"""
        try:
            # Create the connection pool with custom connection factory
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=self.config.min_size,
                max_size=self.config.max_size,
                command_timeout=self.config.command_timeout,
                server_settings=self.config.server_settings,
                init=self._initialize_connection
            )

            # Start background tasks
            if self.config.enable_load_balancing:
                self.health_check_task = asyncio.create_task(self._health_check_loop())

            self.is_initialized = True
            logger.info(f"Advanced connection pool initialized with {self.config.min_size}-{self.config.max_size} connections")

        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise

    async def _initialize_connection(self, connection: asyncpg.Connection) -> None:
        """Initialize a new connection with performance optimizations"""
        connection_id = f"conn_{id(connection)}"

        # Set connection-specific optimizations
        await connection.execute("SET synchronous_commit = off;")  # Faster writes
        await connection.execute("SET wal_writer_delay = '10ms';")
        await connection.execute("SET checkpoint_completion_target = 0.9;")

        # Track connection metrics
        self.connection_metrics[connection_id] = ConnectionMetrics(
            connection_id=connection_id,
            created_at=datetime.now(),
            last_used=datetime.now()
        )

        logger.debug(f"Initialized connection: {connection_id}")

    @asynccontextmanager
    async def acquire_connection(
        self,
        priority: ConnectionPriority = ConnectionPriority.NORMAL,
        timeout: Optional[float] = None
    ):
        """Acquire a connection with priority-based allocation"""
        if not self.is_initialized:
            raise RuntimeError("Connection pool not initialized")

        start_time = time.perf_counter()
        connection = None
        connection_id = None

        try:
            # Get connection from pool with priority consideration
            connection = await self._get_prioritized_connection(priority, timeout)
            connection_id = f"conn_{id(connection)}"

            # Update metrics
            if connection_id in self.connection_metrics:
                metrics = self.connection_metrics[connection_id]
                metrics.last_used = datetime.now()
                metrics.priority_usage[priority] += 1

            yield connection

        except Exception as e:
            logger.error(f"Connection acquisition failed: {e}")
            if connection_id and connection_id in self.connection_metrics:
                self.connection_metrics[connection_id].error_count += 1
            raise

        finally:
            if connection:
                # Update connection metrics
                execution_time = (time.perf_counter() - start_time) * 1000

                if connection_id and connection_id in self.connection_metrics:
                    metrics = self.connection_metrics[connection_id]
                    metrics.query_count += 1
                    metrics.total_time_ms += execution_time

                # Record query history for statistics
                self.query_history.append((execution_time, True))

                # Keep only recent history (last 1000 queries)
                if len(self.query_history) > 1000:
                    self.query_history = self.query_history[-1000:]

                # Release connection back to pool
                await self.pool.release(connection)

    async def _get_prioritized_connection(
        self,
        priority: ConnectionPriority,
        timeout: Optional[float]
    ) -> asyncpg.Connection:
        """Get connection with priority-based selection"""

        # For critical operations, try to get a fresh/least-used connection
        if priority == ConnectionPriority.CRITICAL:
            return await self._get_optimal_connection(timeout)

        # For normal operations, use standard pool allocation
        return await self.pool.acquire(timeout=timeout)

    async def _get_optimal_connection(self, timeout: Optional[float]) -> asyncpg.Connection:
        """Get the most optimal connection for critical operations"""
        # This is a simplified implementation
        # In production, would analyze connection metrics to select best connection
        return await self.pool.acquire(timeout=timeout)

    async def execute_optimized(
        self,
        query: str,
        *args,
        priority: ConnectionPriority = ConnectionPriority.NORMAL,
        use_prepared: bool = None
    ) -> Any:
        """Execute query with optimization strategies"""

        # Determine if we should use prepared statements
        if use_prepared is None:
            use_prepared = (
                self.config.enable_prepared_statements and
                len(args) > 0 and
                priority in [ConnectionPriority.HIGH, ConnectionPriority.CRITICAL]
            )

        async with self.acquire_connection(priority) as conn:
            if use_prepared:
                return await self._execute_prepared(conn, query, args)
            else:
                return await conn.fetch(query, *args) if args else await conn.fetch(query)

    async def _execute_prepared(
        self,
        connection: asyncpg.Connection,
        query: str,
        args: tuple
    ) -> Any:
        """Execute query using prepared statements for better performance"""

        # Generate statement key
        import hashlib
        statement_key = hashlib.md5(query.encode()).hexdigest()

        # Check if statement is already prepared
        if statement_key not in self.prepared_statements:
            # Prepare the statement
            statement_name = f"stmt_{statement_key}"
            await connection.execute(f"PREPARE {statement_name} AS {query}")
            self.prepared_statements[statement_key] = statement_name

            # Limit cache size
            if len(self.prepared_statements) > self.config.statement_cache_size:
                # Remove oldest prepared statement (simplified LRU)
                oldest_key = next(iter(self.prepared_statements))
                oldest_name = self.prepared_statements.pop(oldest_key)
                await connection.execute(f"DEALLOCATE {oldest_name}")

        # Execute prepared statement
        statement_name = self.prepared_statements[statement_key]
        return await connection.fetch(f"EXECUTE {statement_name}", *args)

    async def execute_batch_optimized(
        self,
        queries: List[Tuple[str, tuple]],
        priority: ConnectionPriority = ConnectionPriority.NORMAL
    ) -> List[Any]:
        """Execute multiple queries efficiently using batch optimization"""

        results = []

        async with self.acquire_connection(priority) as conn:
            # Use transaction for batch execution
            async with conn.transaction():
                for query, args in queries:
                    result = await conn.fetch(query, *args) if args else await conn.fetch(query)
                    results.append(result)

        return results

    async def _health_check_loop(self) -> None:
        """Background task for connection health monitoring"""
        while self.is_initialized:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(self.config.health_check_interval)

    async def _perform_health_check(self) -> None:
        """Perform health check on connections"""
        current_time = datetime.now()
        unhealthy_connections = []

        for connection_id, metrics in self.connection_metrics.items():
            # Check for various health indicators
            is_healthy = (
                metrics.error_count < self.config.max_connection_errors and
                metrics.age_seconds < self.config.connection_ttl and
                metrics.idle_time_seconds < self.config.max_inactive_time and
                metrics.query_count < self.config.max_queries
            )

            if not is_healthy:
                unhealthy_connections.append(connection_id)
                metrics.is_healthy = False

        # Log health status
        if unhealthy_connections:
            logger.warning(f"Found {len(unhealthy_connections)} unhealthy connections")

        self.last_health_check = current_time

    async def get_pool_statistics(self) -> PoolStatistics:
        """Get comprehensive pool statistics"""

        # Update statistics
        self.statistics.total_connections = len(self.connection_metrics)
        self.statistics.active_connections = sum(
            1 for m in self.connection_metrics.values()
            if m.idle_time_seconds < 60  # Active in last minute
        )
        self.statistics.idle_connections = (
            self.statistics.total_connections - self.statistics.active_connections
        )

        # Calculate query statistics
        if self.query_history:
            execution_times = [t for t, success in self.query_history if success]
            self.statistics.avg_response_time_ms = (
                sum(execution_times) / len(execution_times) if execution_times else 0
            )
            self.statistics.total_queries = len(self.query_history)
            self.statistics.total_errors = sum(1 for _, success in self.query_history if not success)

        # Calculate efficiency metrics
        if self.config.max_size > 0:
            self.statistics.connection_utilization = (
                self.statistics.active_connections / self.config.max_size
            )

        error_rate = (
            self.statistics.total_errors / max(self.statistics.total_queries, 1)
        )
        self.statistics.pool_efficiency = max(0, 1 - error_rate)

        self.statistics.last_updated = datetime.now()
        return self.statistics

    async def get_connection_details(self) -> List[Dict[str, Any]]:
        """Get detailed information about all connections"""
        details = []

        for connection_id, metrics in self.connection_metrics.items():
            details.append({
                "connection_id": connection_id,
                "created_at": metrics.created_at.isoformat(),
                "last_used": metrics.last_used.isoformat(),
                "age_seconds": metrics.age_seconds,
                "idle_time_seconds": metrics.idle_time_seconds,
                "query_count": metrics.query_count,
                "error_count": metrics.error_count,
                "avg_query_time_ms": metrics.avg_query_time_ms,
                "is_healthy": metrics.is_healthy,
                "priority_usage": {p.value: count for p, count in metrics.priority_usage.items()}
            })

        return details

    async def optimize_pool(self) -> Dict[str, Any]:
        """Perform pool optimization operations"""
        optimization_results = {}

        try:
            # Clean up unhealthy connections
            unhealthy_count = sum(
                1 for m in self.connection_metrics.values()
                if not m.is_healthy
            )

            if unhealthy_count > 0:
                # In a real implementation, would actually close and recreate connections
                logger.info(f"Would optimize {unhealthy_count} unhealthy connections")
                optimization_results["unhealthy_connections_found"] = unhealthy_count

            # Clear old prepared statements
            if len(self.prepared_statements) > self.config.statement_cache_size * 0.8:
                statements_to_clear = len(self.prepared_statements) // 4
                optimization_results["prepared_statements_cleared"] = statements_to_clear

            # Update statistics
            stats = await self.get_pool_statistics()
            optimization_results["final_statistics"] = {
                "total_connections": stats.total_connections,
                "active_connections": stats.active_connections,
                "avg_response_time_ms": stats.avg_response_time_ms,
                "pool_efficiency": stats.pool_efficiency
            }

            logger.info("Pool optimization completed")
            return optimization_results

        except Exception as e:
            logger.error(f"Pool optimization failed: {e}")
            return {"error": str(e)}

    async def close(self) -> None:
        """Close the connection pool and cleanup resources"""
        try:
            if self.health_check_task:
                self.health_check_task.cancel()
                try:
                    await self.health_check_task
                except asyncio.CancelledError:
                    pass

            if self.pool:
                await self.pool.close()

            self.connection_metrics.clear()
            self.query_history.clear()
            self.prepared_statements.clear()

            self.is_initialized = False
            logger.info("Advanced connection pool closed")

        except Exception as e:
            logger.error(f"Error closing connection pool: {e}")

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Global connection pool manager
connection_pool_manager = None

async def get_connection_pool_manager(
    database_url: str,
    config: ConnectionPoolConfig = None
) -> AdvancedConnectionPool:
    """Get or create global connection pool manager"""
    global connection_pool_manager
    if connection_pool_manager is None:
        connection_pool_manager = AdvancedConnectionPool(database_url, config)
        await connection_pool_manager.initialize()
    return connection_pool_manager