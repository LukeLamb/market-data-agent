"""
Integration Test Suite for Complete Monitoring System
Phase 4 Step 2: Enterprise Monitoring & Observability
"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from src.monitoring.metrics_exporter import (
    MarketDataMetricsExporter, create_metrics_app
)
from src.monitoring.anomaly_detector import (
    AnomalyDetectionEngine, AnomalyType
)


@pytest.mark.integration
class TestMonitoringSystemIntegration:
    """Integration tests for the complete monitoring system"""

    def setup_method(self):
        """Set up integration test environment"""
        self.registry = CollectorRegistry()
        self.metrics_exporter = MarketDataMetricsExporter(registry=self.registry)
        self.anomaly_engine = AnomalyDetectionEngine()
        self.metrics_app = create_metrics_app(self.metrics_exporter)

    @pytest.mark.asyncio
    async def test_complete_monitoring_workflow(self):
        """Test the complete monitoring workflow from data ingestion to alerting"""

        # 1. Simulate market data ingestion with metrics collection
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        sources = ['yfinance', 'alpha_vantage']

        for symbol in symbols:
            for source in sources:
                # Simulate data source request
                response_time = np.random.uniform(0.05, 0.3)
                status = "success" if np.random.random() > 0.05 else "failed"

                self.metrics_exporter.record_data_source_request(
                    source=source,
                    symbol=symbol,
                    endpoint="/quote",
                    status=status,
                    response_time=response_time
                )

                # Update data freshness
                self.metrics_exporter.update_data_freshness(
                    source=source,
                    symbol=symbol,
                    timestamp=time.time()
                )

                # Record price update
                self.metrics_exporter.record_price_update(
                    source=source,
                    symbol=symbol,
                    data_type="real_time"
                )

                # Simulate cache operations
                cache_hit = np.random.random() > 0.2  # 80% hit rate
                if cache_hit:
                    self.metrics_exporter.record_cache_hit(
                        cache_type="redis",
                        key_pattern=f"quote:{symbol}",
                        response_time=0.001
                    )
                else:
                    self.metrics_exporter.record_cache_miss(
                        cache_type="redis",
                        key_pattern=f"quote:{symbol}"
                    )

        # 2. Simulate API requests
        endpoints = ['/api/quotes', '/api/historical', '/api/realtime']
        for _ in range(50):
            endpoint = np.random.choice(endpoints)
            method = "GET"
            status_code = np.random.choice([200, 200, 200, 400, 500], p=[0.85, 0.1, 0.03, 0.01, 0.01])
            duration = np.random.uniform(0.02, 0.5)

            self.metrics_exporter.record_api_request(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
                duration=duration
            )

        # 3. Update system health metrics
        self.metrics_exporter.update_data_source_health("yfinance", "healthy")
        self.metrics_exporter.update_data_source_health("alpha_vantage", "degraded")

        # 4. Verify metrics collection
        metrics_output = self.metrics_exporter.get_metrics()

        # Check that key metrics are present
        expected_metrics = [
            'market_data_source_requests_total',
            'market_data_price_updates_total',
            'market_data_cache_hits_total',
            'market_data_api_requests_total',
            'market_data_source_health'
        ]

        for metric in expected_metrics:
            assert metric in metrics_output, f"Missing metric: {metric}"

        # 5. Test metrics API endpoint
        client = TestClient(self.metrics_app)
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "market_data_source_requests_total" in response.text

        # 6. Test health endpoint
        health_response = client.get("/health")
        assert health_response.status_code == 200
        health_data = health_response.json()
        assert health_data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_anomaly_detection_integration(self):
        """Test integration between metrics and anomaly detection"""

        # 1. Train anomaly detection models with synthetic data
        training_data = self._generate_training_data()
        await self.anomaly_engine.train_models(training_data)

        # 2. Generate test data with known anomalies
        anomalous_data = self._generate_anomalous_data()

        # 3. Detect anomalies
        results = await self.anomaly_engine.detect_anomalies(anomalous_data)

        # 4. Verify anomaly detection results update metrics
        for result in results:
            if result.is_anomaly:
                self.metrics_exporter.update_anomaly_score(
                    result.symbol,
                    result.anomaly_type.value,
                    result.score
                )

        # 5. Check that anomaly metrics are recorded
        metrics_output = self.metrics_exporter.get_metrics()
        if results:
            assert 'market_data_price_anomaly_score' in metrics_output

    def test_high_volume_monitoring(self):
        """Test monitoring system under high volume load"""
        import concurrent.futures
        import threading

        def generate_high_volume_metrics(thread_id: int):
            """Generate metrics from multiple threads"""
            for i in range(1000):
                # Simulate high-frequency trading data
                symbol = f"STOCK{i % 100}"
                source = "high_freq_feed"

                self.metrics_exporter.record_data_source_request(
                    source=source,
                    symbol=symbol,
                    endpoint="/tick",
                    status="success",
                    response_time=0.001
                )

                self.metrics_exporter.record_price_update(
                    source=source,
                    symbol=symbol,
                    data_type="tick"
                )

                # Simulate API requests
                self.metrics_exporter.record_api_request(
                    method="GET",
                    endpoint=f"/api/tick/{symbol}",
                    status_code=200,
                    duration=0.005
                )

        # Generate metrics from multiple threads
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(generate_high_volume_metrics, i) for i in range(10)]
            concurrent.futures.wait(futures)

        elapsed_time = time.time() - start_time

        # Should complete within reasonable time (< 30 seconds)
        assert elapsed_time < 30

        # Verify metrics were recorded
        metrics_output = self.metrics_exporter.get_metrics()
        assert 'market_data_source_requests_total' in metrics_output
        assert 'market_data_api_requests_total' in metrics_output

    def test_metrics_persistence_and_recovery(self):
        """Test metrics persistence and system recovery"""

        # 1. Generate initial metrics
        self.metrics_exporter.record_api_request("GET", "/api/test", 200, 0.1)
        initial_metrics = self.metrics_exporter.get_metrics()

        # 2. Simulate system restart by creating new exporter
        new_registry = CollectorRegistry()
        new_exporter = MarketDataMetricsExporter(registry=new_registry)

        # 3. Generate new metrics
        new_exporter.record_api_request("POST", "/api/test", 201, 0.2)
        new_metrics = new_exporter.get_metrics()

        # 4. Verify both instances work independently
        assert 'market_data_api_requests_total' in initial_metrics
        assert 'market_data_api_requests_total' in new_metrics

    @pytest.mark.asyncio
    async def test_monitoring_error_handling(self):
        """Test monitoring system error handling and resilience"""

        # 1. Test metrics exporter with invalid data
        with pytest.raises(Exception):
            # This should not crash the system
            self.metrics_exporter.record_api_request(
                method=None,  # Invalid method
                endpoint="/test",
                status_code=200,
                duration=0.1
            )

        # 2. Test anomaly detection with invalid data
        invalid_data = {
            'price': pd.DataFrame({
                'invalid_column': [1, 2, 3]
            })
        }

        # Should not crash, just return empty results
        results = await self.anomaly_engine.detect_anomalies(invalid_data)
        assert isinstance(results, list)

    def test_monitoring_performance_benchmarks(self):
        """Test monitoring system performance benchmarks"""

        # Benchmark metrics collection
        start_time = time.time()
        for i in range(10000):
            self.metrics_exporter.record_api_request(
                method="GET",
                endpoint=f"/api/test/{i % 100}",
                status_code=200,
                duration=0.01
            )
        metrics_collection_time = time.time() - start_time

        # Should be able to handle 10k metrics in reasonable time
        assert metrics_collection_time < 5  # 5 seconds

        # Benchmark metrics export
        start_time = time.time()
        metrics_output = self.metrics_exporter.get_metrics()
        export_time = time.time() - start_time

        # Export should be fast
        assert export_time < 1  # 1 second
        assert len(metrics_output) > 0

    def _generate_training_data(self) -> dict:
        """Generate synthetic training data for anomaly detection"""
        size = 1000

        # Price data
        price_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=size, freq='H'),
            'symbol': np.random.choice(['AAPL', 'GOOGL', 'MSFT'], size),
            'price': np.random.uniform(100, 200, size),
            'volume': np.random.uniform(10000, 100000, size)
        })

        # System data
        system_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=size, freq='H'),
            'cpu_usage': np.random.uniform(20, 70, size),
            'memory_usage': np.random.uniform(30, 80, size),
            'disk_usage': np.random.uniform(40, 90, size),
            'response_time': np.random.uniform(0.05, 0.3, size),
            'error_rate': np.random.uniform(0.001, 0.02, size),
            'request_rate': np.random.uniform(100, 1000, size)
        })

        return {
            'price': price_data,
            'system': system_data
        }

    def _generate_anomalous_data(self) -> dict:
        """Generate data with known anomalies"""
        now = datetime.now()

        # Price data with anomalies
        price_data = pd.DataFrame({
            'timestamp': [now],
            'symbol': ['AAPL'],
            'price': [500.0],  # Obvious anomaly
            'volume': [1000000]  # High volume
        })

        # System data with anomalies
        system_data = pd.DataFrame({
            'timestamp': [now],
            'cpu_usage': [95.0],  # High CPU
            'memory_usage': [95.0],  # High memory
            'disk_usage': [95.0],  # High disk
            'response_time': [5.0],  # High latency
            'error_rate': [0.5],  # High error rate
            'request_rate': [10000]  # High request rate
        })

        return {
            'price': price_data,
            'system': system_data
        }


@pytest.mark.integration
class TestMonitoringAlertingIntegration:
    """Test integration with alerting systems"""

    def setup_method(self):
        """Set up alerting test environment"""
        self.registry = CollectorRegistry()
        self.metrics_exporter = MarketDataMetricsExporter(registry=self.registry)

    def test_prometheus_metrics_format(self):
        """Test that metrics are correctly formatted for Prometheus"""
        # Generate some metrics
        self.metrics_exporter.record_api_request("GET", "/api/test", 200, 0.1)
        self.metrics_exporter.update_data_source_health("yfinance", "healthy")

        metrics_output = self.metrics_exporter.get_metrics()

        # Check Prometheus format
        lines = metrics_output.strip().split('\n')

        # Should have HELP and TYPE comments
        help_lines = [line for line in lines if line.startswith('# HELP')]
        type_lines = [line for line in lines if line.startswith('# TYPE')]

        assert len(help_lines) > 0
        assert len(type_lines) > 0

        # Should have actual metric values
        metric_lines = [line for line in lines if not line.startswith('#') and line.strip()]
        assert len(metric_lines) > 0

    def test_grafana_dashboard_compatibility(self):
        """Test metrics compatibility with Grafana dashboards"""
        # Generate comprehensive metrics that would be used in dashboards
        metrics_scenarios = [
            # API metrics
            ("GET", "/api/quotes", 200, 0.05),
            ("POST", "/api/orders", 201, 0.1),
            ("GET", "/api/quotes", 500, 1.0),

            # Data source metrics
            ("yfinance", "AAPL", "/quote", "success", 0.123),
            ("alpha_vantage", "GOOGL", "/time_series", "failed", 2.0),
        ]

        # Record API metrics
        for method, endpoint, status, duration in metrics_scenarios[:3]:
            self.metrics_exporter.record_api_request(method, endpoint, status, duration)

        # Record data source metrics
        for source, symbol, endpoint, status, response_time in metrics_scenarios[3:]:
            self.metrics_exporter.record_data_source_request(
                source, symbol, endpoint, status, response_time
            )

        # Update various gauge metrics
        self.metrics_exporter.update_cache_memory_usage("redis", 1024*1024*100, 1024*1024*500)
        self.metrics_exporter.update_websocket_connections("/ws/quotes", "active", 25)
        self.metrics_exporter.update_rate_limit_usage("yfinance", 0.75)

        metrics_output = self.metrics_exporter.get_metrics()

        # Verify metrics that would be used in Grafana dashboards
        expected_dashboard_metrics = [
            'market_data_api_requests_total',
            'market_data_api_request_duration_seconds',
            'market_data_source_requests_total',
            'market_data_cache_memory_usage_bytes',
            'market_data_websocket_connections_total',
            'market_data_rate_limit_usage_ratio'
        ]

        for metric in expected_dashboard_metrics:
            assert metric in metrics_output

    @patch('src.monitoring.anomaly_detector.redis.Redis')
    def test_anomaly_alerting_integration(self, mock_redis):
        """Test integration between anomaly detection and alerting"""
        # Mock Redis for anomaly storage
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance

        # Simulate high anomaly score that should trigger alert
        self.metrics_exporter.update_anomaly_score("AAPL", "price_spike", 0.95)

        metrics_output = self.metrics_exporter.get_metrics()

        # Check that anomaly metric is recorded
        assert 'market_data_price_anomaly_score' in metrics_output

        # Verify the metric value is high enough to trigger alerts
        lines = metrics_output.split('\n')
        anomaly_lines = [line for line in lines if 'market_data_price_anomaly_score' in line and '0.95' in line]
        assert len(anomaly_lines) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])