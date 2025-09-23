"""
Custom Market Data Metrics Exporter
Phase 4 Step 2: Enterprise Monitoring & Observability
"""

import time
import asyncio
from typing import Dict, Any, Optional, List
from prometheus_client import (
    Counter, Histogram, Gauge, Enum, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
import logging
from dataclasses import dataclass
from enum import Enum as PyEnum
import json


class MetricType(PyEnum):
    """Metric types for market data"""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    ENUM = "enum"
    INFO = "info"


@dataclass
class MetricDefinition:
    """Definition for a custom metric"""
    name: str
    description: str
    metric_type: MetricType
    labels: List[str] = None
    buckets: List[float] = None
    states: List[str] = None


class MarketDataMetricsExporter:
    """Custom metrics exporter for market data agent"""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize the metrics exporter"""
        self.registry = registry or CollectorRegistry()
        self.logger = logging.getLogger(__name__)
        self.metrics: Dict[str, Any] = {}
        self._initialize_metrics()

    def _initialize_metrics(self):
        """Initialize all market data metrics"""

        # Data source metrics
        self.metrics['data_source_requests_total'] = Counter(
            'market_data_source_requests_total',
            'Total number of requests to data sources',
            ['source', 'symbol', 'endpoint', 'status'],
            registry=self.registry
        )

        self.metrics['data_source_requests_failed_total'] = Counter(
            'market_data_source_requests_failed_total',
            'Total number of failed requests to data sources',
            ['source', 'symbol', 'endpoint', 'error_type'],
            registry=self.registry
        )

        self.metrics['data_source_response_time'] = Histogram(
            'market_data_source_response_time_seconds',
            'Response time for data source requests',
            ['source', 'endpoint'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )

        self.metrics['data_source_health'] = Enum(
            'market_data_source_health',
            'Health status of data sources',
            ['source'],
            states=['healthy', 'degraded', 'unhealthy', 'unknown'],
            registry=self.registry
        )

        self.metrics['data_source_rate_limit_usage'] = Gauge(
            'market_data_rate_limit_usage_ratio',
            'Rate limit usage ratio for data sources',
            ['source'],
            registry=self.registry
        )

        # Data quality metrics
        self.metrics['data_quality_score'] = Gauge(
            'market_data_quality_score',
            'Data quality score (0-1)',
            ['source', 'symbol', 'dimension'],
            registry=self.registry
        )

        self.metrics['data_quality_violations'] = Counter(
            'market_data_quality_violations_total',
            'Total number of data quality violations',
            ['source', 'symbol', 'violation_type', 'severity'],
            registry=self.registry
        )

        self.metrics['data_freshness'] = Gauge(
            'market_data_last_update_timestamp',
            'Timestamp of last data update',
            ['source', 'symbol'],
            registry=self.registry
        )

        # Price and market data metrics
        self.metrics['price_updates'] = Counter(
            'market_data_price_updates_total',
            'Total number of price updates',
            ['source', 'symbol', 'data_type'],
            registry=self.registry
        )

        self.metrics['price_anomaly_score'] = Gauge(
            'market_data_price_anomaly_score',
            'Price anomaly detection score (0-1)',
            ['symbol', 'anomaly_type'],
            registry=self.registry
        )

        self.metrics['symbol_coverage'] = Gauge(
            'market_data_symbol_coverage_total',
            'Total number of symbols being monitored',
            ['source', 'asset_type'],
            registry=self.registry
        )

        # Cache performance metrics
        self.metrics['cache_hits'] = Counter(
            'market_data_cache_hits_total',
            'Total number of cache hits',
            ['cache_type', 'key_pattern'],
            registry=self.registry
        )

        self.metrics['cache_misses'] = Counter(
            'market_data_cache_misses_total',
            'Total number of cache misses',
            ['cache_type', 'key_pattern'],
            registry=self.registry
        )

        self.metrics['cache_response_time'] = Histogram(
            'market_data_cache_response_time_seconds',
            'Cache response time',
            ['cache_type', 'operation'],
            buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
            registry=self.registry
        )

        self.metrics['cache_memory_usage'] = Gauge(
            'market_data_cache_memory_usage_bytes',
            'Cache memory usage in bytes',
            ['cache_type'],
            registry=self.registry
        )

        self.metrics['cache_memory_limit'] = Gauge(
            'market_data_cache_memory_limit_bytes',
            'Cache memory limit in bytes',
            ['cache_type'],
            registry=self.registry
        )

        # Database metrics
        self.metrics['db_connections_active'] = Gauge(
            'market_data_db_connections_active',
            'Number of active database connections',
            ['database', 'pool'],
            registry=self.registry
        )

        self.metrics['db_connections_max'] = Gauge(
            'market_data_db_connections_max',
            'Maximum number of database connections',
            ['database', 'pool'],
            registry=self.registry
        )

        self.metrics['db_query_duration'] = Histogram(
            'market_data_db_query_duration_seconds',
            'Database query execution time',
            ['database', 'operation', 'table'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            registry=self.registry
        )

        self.metrics['db_queries_total'] = Counter(
            'market_data_db_queries_total',
            'Total number of database queries',
            ['database', 'operation', 'table', 'status'],
            registry=self.registry
        )

        # API metrics
        self.metrics['api_requests_total'] = Counter(
            'market_data_api_requests_total',
            'Total number of API requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )

        self.metrics['api_request_duration'] = Histogram(
            'market_data_api_request_duration_seconds',
            'API request duration',
            ['method', 'endpoint'],
            buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )

        self.metrics['api_concurrent_requests'] = Gauge(
            'market_data_api_concurrent_requests',
            'Number of concurrent API requests',
            ['endpoint'],
            registry=self.registry
        )

        # WebSocket metrics
        self.metrics['websocket_connections'] = Gauge(
            'market_data_websocket_connections_total',
            'Total number of WebSocket connections',
            ['endpoint', 'status'],
            registry=self.registry
        )

        self.metrics['websocket_messages_sent'] = Counter(
            'market_data_websocket_messages_sent_total',
            'Total number of WebSocket messages sent',
            ['endpoint', 'message_type'],
            registry=self.registry
        )

        self.metrics['websocket_messages_received'] = Counter(
            'market_data_websocket_messages_received_total',
            'Total number of WebSocket messages received',
            ['endpoint', 'message_type'],
            registry=self.registry
        )

        # System resource metrics
        self.metrics['memory_usage'] = Gauge(
            'market_data_memory_usage_bytes',
            'Memory usage in bytes',
            ['component'],
            registry=self.registry
        )

        self.metrics['cpu_usage'] = Gauge(
            'market_data_cpu_usage_ratio',
            'CPU usage ratio',
            ['component'],
            registry=self.registry
        )

        # Business metrics
        self.metrics['symbols_processed'] = Gauge(
            'market_data_symbols_processed_total',
            'Total number of unique symbols processed',
            ['source', 'time_window'],
            registry=self.registry
        )

        self.metrics['data_volume_bytes'] = Counter(
            'market_data_volume_bytes_total',
            'Total volume of data processed in bytes',
            ['source', 'data_type'],
            registry=self.registry
        )

        self.metrics['trading_sessions'] = Gauge(
            'market_data_trading_sessions_active',
            'Number of active trading sessions',
            ['market', 'session_type'],
            registry=self.registry
        )

        # Application info
        self.metrics['app_info'] = Info(
            'market_data_app_info',
            'Application information',
            registry=self.registry
        )

        self.logger.info("Initialized %d custom metrics", len(self.metrics))

    def record_data_source_request(self, source: str, symbol: str,
                                 endpoint: str, status: str,
                                 response_time: float = None):
        """Record a data source request"""
        self.metrics['data_source_requests_total'].labels(
            source=source, symbol=symbol, endpoint=endpoint, status=status
        ).inc()

        if status == 'failed':
            self.metrics['data_source_requests_failed_total'].labels(
                source=source, symbol=symbol, endpoint=endpoint, error_type='unknown'
            ).inc()

        if response_time is not None:
            self.metrics['data_source_response_time'].labels(
                source=source, endpoint=endpoint
            ).observe(response_time)

    def update_data_source_health(self, source: str, status: str):
        """Update data source health status"""
        self.metrics['data_source_health'].labels(source=source).state(status)

    def update_rate_limit_usage(self, source: str, usage_ratio: float):
        """Update rate limit usage ratio"""
        self.metrics['data_source_rate_limit_usage'].labels(source=source).set(usage_ratio)

    def record_quality_score(self, source: str, symbol: str,
                           dimension: str, score: float):
        """Record data quality score"""
        self.metrics['data_quality_score'].labels(
            source=source, symbol=symbol, dimension=dimension
        ).set(score)

    def record_quality_violation(self, source: str, symbol: str,
                               violation_type: str, severity: str):
        """Record data quality violation"""
        self.metrics['data_quality_violations'].labels(
            source=source, symbol=symbol,
            violation_type=violation_type, severity=severity
        ).inc()

    def update_data_freshness(self, source: str, symbol: str, timestamp: float):
        """Update data freshness timestamp"""
        self.metrics['data_freshness'].labels(source=source, symbol=symbol).set(timestamp)

    def record_price_update(self, source: str, symbol: str, data_type: str):
        """Record a price update"""
        self.metrics['price_updates'].labels(
            source=source, symbol=symbol, data_type=data_type
        ).inc()

    def update_anomaly_score(self, symbol: str, anomaly_type: str, score: float):
        """Update price anomaly score"""
        self.metrics['price_anomaly_score'].labels(
            symbol=symbol, anomaly_type=anomaly_type
        ).set(score)

    def record_cache_hit(self, cache_type: str, key_pattern: str, response_time: float):
        """Record cache hit"""
        self.metrics['cache_hits'].labels(
            cache_type=cache_type, key_pattern=key_pattern
        ).inc()
        self.metrics['cache_response_time'].labels(
            cache_type=cache_type, operation='get'
        ).observe(response_time)

    def record_cache_miss(self, cache_type: str, key_pattern: str):
        """Record cache miss"""
        self.metrics['cache_misses'].labels(
            cache_type=cache_type, key_pattern=key_pattern
        ).inc()

    def update_cache_memory_usage(self, cache_type: str, usage_bytes: int, limit_bytes: int):
        """Update cache memory usage"""
        self.metrics['cache_memory_usage'].labels(cache_type=cache_type).set(usage_bytes)
        self.metrics['cache_memory_limit'].labels(cache_type=cache_type).set(limit_bytes)

    def record_api_request(self, method: str, endpoint: str,
                          status_code: int, duration: float):
        """Record API request"""
        self.metrics['api_requests_total'].labels(
            method=method, endpoint=endpoint, status_code=str(status_code)
        ).inc()
        self.metrics['api_request_duration'].labels(
            method=method, endpoint=endpoint
        ).observe(duration)

    def update_websocket_connections(self, endpoint: str, status: str, count: int):
        """Update WebSocket connection count"""
        self.metrics['websocket_connections'].labels(
            endpoint=endpoint, status=status
        ).set(count)

    def set_app_info(self, version: str, build_date: str, git_commit: str):
        """Set application information"""
        self.metrics['app_info'].info({
            'version': version,
            'build_date': build_date,
            'git_commit': git_commit,
            'component': 'market-data-agent'
        })

    def get_metrics(self) -> str:
        """Get all metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')

    def get_metrics_response(self) -> Response:
        """Get metrics as FastAPI response"""
        metrics_data = generate_latest(self.registry)
        return Response(
            content=metrics_data,
            media_type=CONTENT_TYPE_LATEST
        )


class MetricsMiddleware:
    """FastAPI middleware for automatic metrics collection"""

    def __init__(self, exporter: MarketDataMetricsExporter):
        self.exporter = exporter

    async def __call__(self, request, call_next):
        """Process request and collect metrics"""
        start_time = time.time()

        # Extract endpoint info
        method = request.method
        path = request.url.path

        # Increment concurrent requests
        self.exporter.metrics['api_concurrent_requests'].labels(endpoint=path).inc()

        try:
            response = await call_next(request)
            status_code = response.status_code

            # Record successful request
            duration = time.time() - start_time
            self.exporter.record_api_request(method, path, status_code, duration)

            return response

        except Exception as e:
            # Record failed request
            duration = time.time() - start_time
            self.exporter.record_api_request(method, path, 500, duration)
            raise

        finally:
            # Decrement concurrent requests
            self.exporter.metrics['api_concurrent_requests'].labels(endpoint=path).dec()


def create_metrics_app(exporter: MarketDataMetricsExporter) -> FastAPI:
    """Create FastAPI app for metrics endpoint"""
    app = FastAPI(
        title="Market Data Agent Metrics",
        description="Prometheus metrics for Market Data Agent",
        version="1.0.0"
    )

    @app.get("/metrics", response_class=PlainTextResponse)
    async def get_metrics():
        """Prometheus metrics endpoint"""
        return exporter.get_metrics()

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "metrics_count": len(exporter.metrics)}

    return app


# Global metrics exporter instance
metrics_exporter = MarketDataMetricsExporter()
metrics_middleware = MetricsMiddleware(metrics_exporter)
metrics_app = create_metrics_app(metrics_exporter)