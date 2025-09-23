"""
Test suite for Market Data Metrics Exporter
Phase 4 Step 2: Enterprise Monitoring & Observability
"""

import pytest
import time
from unittest.mock import Mock, patch
from prometheus_client import CollectorRegistry, REGISTRY
from src.monitoring.metrics_exporter import (
    MarketDataMetricsExporter, MetricsMiddleware, create_metrics_app
)


class TestMarketDataMetricsExporter:
    """Test the custom metrics exporter"""

    def setup_method(self):
        """Set up test fixtures"""
        self.registry = CollectorRegistry()
        self.exporter = MarketDataMetricsExporter(registry=self.registry)

    def test_initialization(self):
        """Test that all metrics are properly initialized"""
        assert len(self.exporter.metrics) > 0
        assert 'data_source_requests_total' in self.exporter.metrics
        assert 'cache_hits' in self.exporter.metrics
        assert 'api_requests_total' in self.exporter.metrics

    def test_record_data_source_request(self):
        """Test recording data source requests"""
        # Record successful request
        self.exporter.record_data_source_request(
            source="yfinance",
            symbol="AAPL",
            endpoint="/quote",
            status="success",
            response_time=0.123
        )

        # Verify metrics were updated
        metrics = self.exporter.get_metrics()
        assert 'market_data_source_requests_total' in metrics
        assert 'source="yfinance"' in metrics
        assert 'symbol="AAPL"' in metrics

    def test_record_data_source_failure(self):
        """Test recording failed data source requests"""
        self.exporter.record_data_source_request(
            source="alpha_vantage",
            symbol="TSLA",
            endpoint="/time_series",
            status="failed",
            response_time=5.0
        )

        metrics = self.exporter.get_metrics()
        assert 'market_data_source_requests_failed_total' in metrics
        assert 'source="alpha_vantage"' in metrics

    def test_update_data_source_health(self):
        """Test updating data source health status"""
        self.exporter.update_data_source_health("yfinance", "healthy")
        self.exporter.update_data_source_health("alpha_vantage", "degraded")

        metrics = self.exporter.get_metrics()
        assert 'market_data_source_health' in metrics

    def test_record_quality_metrics(self):
        """Test recording data quality metrics"""
        # Record quality score
        self.exporter.record_quality_score(
            source="yfinance",
            symbol="AAPL",
            dimension="completeness",
            score=0.95
        )

        # Record quality violation
        self.exporter.record_quality_violation(
            source="yfinance",
            symbol="AAPL",
            violation_type="missing_data",
            severity="low"
        )

        metrics = self.exporter.get_metrics()
        assert 'market_data_quality_score' in metrics
        assert 'market_data_quality_violations_total' in metrics

    def test_cache_metrics(self):
        """Test cache performance metrics"""
        # Record cache hit
        self.exporter.record_cache_hit(
            cache_type="redis",
            key_pattern="quote:*",
            response_time=0.001
        )

        # Record cache miss
        self.exporter.record_cache_miss(
            cache_type="redis",
            key_pattern="quote:*"
        )

        # Update memory usage
        self.exporter.update_cache_memory_usage(
            cache_type="redis",
            usage_bytes=1024*1024*100,  # 100MB
            limit_bytes=1024*1024*500   # 500MB
        )

        metrics = self.exporter.get_metrics()
        assert 'market_data_cache_hits_total' in metrics
        assert 'market_data_cache_misses_total' in metrics
        assert 'market_data_cache_memory_usage_bytes' in metrics

    def test_api_metrics(self):
        """Test API request metrics"""
        self.exporter.record_api_request(
            method="GET",
            endpoint="/api/quotes",
            status_code=200,
            duration=0.050
        )

        self.exporter.record_api_request(
            method="POST",
            endpoint="/api/quotes",
            status_code=500,
            duration=1.0
        )

        metrics = self.exporter.get_metrics()
        assert 'market_data_api_requests_total' in metrics
        assert 'market_data_api_request_duration_seconds' in metrics

    def test_websocket_metrics(self):
        """Test WebSocket connection metrics"""
        self.exporter.update_websocket_connections(
            endpoint="/ws/quotes",
            status="active",
            count=25
        )

        metrics = self.exporter.get_metrics()
        assert 'market_data_websocket_connections_total' in metrics

    def test_anomaly_score_update(self):
        """Test anomaly score updates"""
        self.exporter.update_anomaly_score(
            symbol="AAPL",
            anomaly_type="price_spike",
            score=0.85
        )

        metrics = self.exporter.get_metrics()
        assert 'market_data_price_anomaly_score' in metrics

    def test_app_info_setting(self):
        """Test setting application information"""
        self.exporter.set_app_info(
            version="1.0.0",
            build_date="2023-01-01",
            git_commit="abc123"
        )

        metrics = self.exporter.get_metrics()
        assert 'market_data_app_info' in metrics

    def test_metrics_response_format(self):
        """Test that metrics are returned in Prometheus format"""
        # Add some metrics
        self.exporter.record_api_request("GET", "/test", 200, 0.1)

        response = self.exporter.get_metrics_response()
        assert response.media_type == 'text/plain; version=0.0.4; charset=utf-8'
        assert isinstance(response.body, bytes)


class TestMetricsMiddleware:
    """Test the FastAPI metrics middleware"""

    def setup_method(self):
        """Set up test fixtures"""
        self.registry = CollectorRegistry()
        self.exporter = MarketDataMetricsExporter(registry=self.registry)
        self.middleware = MetricsMiddleware(self.exporter)

    @pytest.mark.asyncio
    async def test_successful_request(self):
        """Test middleware with successful request"""
        # Mock request and response
        request = Mock()
        request.method = "GET"
        request.url.path = "/api/quotes"

        response = Mock()
        response.status_code = 200

        async def mock_call_next(req):
            await asyncio.sleep(0.01)  # Simulate processing time
            return response

        # Process request through middleware
        result = await self.middleware(request, mock_call_next)

        assert result == response

        # Check that metrics were recorded
        metrics = self.exporter.get_metrics()
        assert 'market_data_api_requests_total' in metrics

    @pytest.mark.asyncio
    async def test_failed_request(self):
        """Test middleware with failed request"""
        request = Mock()
        request.method = "POST"
        request.url.path = "/api/orders"

        async def mock_call_next(req):
            raise Exception("Internal error")

        # Should propagate the exception but still record metrics
        with pytest.raises(Exception):
            await self.middleware(request, mock_call_next)

        metrics = self.exporter.get_metrics()
        assert 'market_data_api_requests_total' in metrics


class TestMetricsApp:
    """Test the metrics FastAPI app"""

    def setup_method(self):
        """Set up test fixtures"""
        self.registry = CollectorRegistry()
        self.exporter = MarketDataMetricsExporter(registry=self.registry)
        self.app = create_metrics_app(self.exporter)

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self):
        """Test the /metrics endpoint"""
        from fastapi.testclient import TestClient

        client = TestClient(self.app)

        # Add some metrics
        self.exporter.record_api_request("GET", "/test", 200, 0.1)

        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert "market_data_api_requests_total" in response.text

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test the /health endpoint"""
        from fastapi.testclient import TestClient

        client = TestClient(self.app)

        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "metrics_count" in data


@pytest.mark.integration
class TestMetricsIntegration:
    """Integration tests for the complete metrics system"""

    def setup_method(self):
        """Set up integration test environment"""
        self.registry = CollectorRegistry()
        self.exporter = MarketDataMetricsExporter(registry=self.registry)

    def test_complete_workflow(self):
        """Test a complete monitoring workflow"""
        # Simulate a complete request cycle
        start_time = time.time()

        # 1. Record data source request
        self.exporter.record_data_source_request(
            source="yfinance",
            symbol="AAPL",
            endpoint="/quote",
            status="success",
            response_time=0.123
        )

        # 2. Update data freshness
        self.exporter.update_data_freshness(
            source="yfinance",
            symbol="AAPL",
            timestamp=start_time
        )

        # 3. Record price update
        self.exporter.record_price_update(
            source="yfinance",
            symbol="AAPL",
            data_type="real_time"
        )

        # 4. Record cache operations
        self.exporter.record_cache_hit(
            cache_type="redis",
            key_pattern="quote:AAPL",
            response_time=0.001
        )

        # 5. Record API request
        self.exporter.record_api_request(
            method="GET",
            endpoint="/api/quote/AAPL",
            status_code=200,
            duration=0.150
        )

        # 6. Update quality score
        self.exporter.record_quality_score(
            source="yfinance",
            symbol="AAPL",
            dimension="timeliness",
            score=0.98
        )

        # Verify all metrics are present
        metrics = self.exporter.get_metrics()

        expected_metrics = [
            'market_data_source_requests_total',
            'market_data_last_update_timestamp',
            'market_data_price_updates_total',
            'market_data_cache_hits_total',
            'market_data_api_requests_total',
            'market_data_quality_score'
        ]

        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"

    def test_high_volume_metrics(self):
        """Test metrics under high volume"""
        import concurrent.futures
        import threading

        def generate_metrics(thread_id):
            """Generate metrics from multiple threads"""
            for i in range(100):
                self.exporter.record_api_request(
                    method="GET",
                    endpoint=f"/api/test/{thread_id}",
                    status_code=200,
                    duration=0.01
                )

        # Generate metrics from multiple threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(generate_metrics, i) for i in range(5)]
            concurrent.futures.wait(futures)

        # Verify metrics were recorded
        metrics = self.exporter.get_metrics()
        assert 'market_data_api_requests_total' in metrics

        # Count should reflect all requests
        lines = metrics.split('\n')
        api_request_lines = [line for line in lines if 'market_data_api_requests_total' in line and not line.startswith('#')]
        assert len(api_request_lines) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])