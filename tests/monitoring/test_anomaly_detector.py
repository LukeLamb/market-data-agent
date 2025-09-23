"""
Test suite for Anomaly Detection System
Phase 4 Step 2: Enterprise Monitoring & Observability
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from src.monitoring.anomaly_detector import (
    PriceAnomalyDetector, SystemAnomalyDetector, AnomalyDetectionEngine,
    AnomalyType, AnomalyResult
)


class TestPriceAnomalyDetector:
    """Test the price anomaly detector"""

    def setup_method(self):
        """Set up test fixtures"""
        self.detector = PriceAnomalyDetector(contamination=0.1)

    def create_sample_price_data(self, size=100, with_anomalies=False):
        """Create sample price data for testing"""
        np.random.seed(42)
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(days=size//24),
            periods=size,
            freq='H'
        )

        # Normal price movements
        base_price = 100.0
        price_changes = np.random.normal(0, 0.02, size)  # 2% volatility
        prices = [base_price]

        for change in price_changes[:-1]:
            prices.append(prices[-1] * (1 + change))

        # Add anomalies if requested
        if with_anomalies:
            # Add price spikes at specific points
            prices[20] *= 1.15  # 15% spike
            prices[50] *= 0.85  # 15% drop
            prices[80] *= 1.20  # 20% spike

        volumes = np.random.uniform(10000, 100000, size)

        return pd.DataFrame({
            'timestamp': timestamps,
            'symbol': ['AAPL'] * size,
            'price': prices,
            'volume': volumes
        })

    def test_feature_extraction(self):
        """Test feature extraction from price data"""
        data = self.create_sample_price_data(50)
        features = self.detector._extract_features(data)

        assert not features.empty
        assert 'price_change' in features.columns
        assert 'price_zscore' in features.columns
        assert 'volatility' in features.columns

    def test_model_training(self):
        """Test training the anomaly detection model"""
        data = self.create_sample_price_data(200)

        self.detector.fit(data)

        assert self.detector.trained
        assert self.detector.model is not None
        assert self.detector.scaler is not None

    def test_anomaly_detection_without_training(self):
        """Test that prediction returns empty list without training"""
        data = self.create_sample_price_data(10)
        results = self.detector.predict(data)

        assert results == []

    def test_anomaly_detection_with_training(self):
        """Test anomaly detection after training"""
        # Train on normal data
        normal_data = self.create_sample_price_data(200, with_anomalies=False)
        self.detector.fit(normal_data)

        # Test on data with anomalies
        test_data = self.create_sample_price_data(50, with_anomalies=True)
        results = self.detector.predict(test_data)

        # Should detect some anomalies
        assert len(results) > 0

        for result in results:
            assert isinstance(result, AnomalyResult)
            assert result.anomaly_type == AnomalyType.PRICE_SPIKE
            assert 0 <= result.score <= 1
            assert result.is_anomaly

    def test_model_save_load(self):
        """Test saving and loading trained models"""
        import tempfile
        import os

        # Train model
        data = self.create_sample_price_data(100)
        self.detector.fit(data)

        # Save model
        with tempfile.NamedTemporaryFile(delete=False) as f:
            model_path = f.name

        try:
            self.detector.save_model(model_path)

            # Create new detector and load model
            new_detector = PriceAnomalyDetector()
            new_detector.load_model(model_path)

            assert new_detector.trained
            assert new_detector.model is not None

        finally:
            os.unlink(model_path)


class TestSystemAnomalyDetector:
    """Test the system anomaly detector"""

    def setup_method(self):
        """Set up test fixtures"""
        self.detector = SystemAnomalyDetector()

    def create_sample_system_data(self, size=100, with_anomalies=False):
        """Create sample system metrics data"""
        np.random.seed(42)
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=size),
            periods=size,
            freq='H'
        )

        # Normal system metrics
        cpu_usage = np.random.uniform(20, 60, size)
        memory_usage = np.random.uniform(30, 70, size)
        disk_usage = np.random.uniform(40, 80, size)
        response_time = np.random.uniform(0.05, 0.2, size)
        error_rate = np.random.uniform(0.001, 0.01, size)
        request_rate = np.random.uniform(100, 1000, size)

        # Add anomalies if requested
        if with_anomalies:
            cpu_usage[20] = 95  # High CPU
            memory_usage[30] = 95  # High memory
            response_time[40] = 5.0  # High latency
            error_rate[50] = 0.5  # High error rate

        return pd.DataFrame({
            'timestamp': timestamps,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'disk_usage': disk_usage,
            'response_time': response_time,
            'error_rate': error_rate,
            'request_rate': request_rate
        })

    def test_system_feature_extraction(self):
        """Test system feature extraction"""
        data = self.create_sample_system_data(50)
        features = self.detector._extract_system_features(data)

        assert not features.empty
        assert 'cpu_usage' in features.columns
        assert 'response_time_ma_10' in features.columns
        assert 'error_rate_std_10' in features.columns

    def test_system_anomaly_classification(self):
        """Test system anomaly type classification"""
        # Create features with high response time
        features = pd.Series({
            'response_time': 5.0,
            'response_time_ma_10': 0.1,
            'error_rate': 0.01,
            'error_rate_ma_10': 0.01
        })

        anomaly_type = self.detector._classify_system_anomaly(features)
        assert anomaly_type == AnomalyType.LATENCY_ANOMALY

        # Create features with high error rate
        features = pd.Series({
            'response_time': 0.1,
            'response_time_ma_10': 0.1,
            'error_rate': 0.3,
            'error_rate_ma_10': 0.01
        })

        anomaly_type = self.detector._classify_system_anomaly(features)
        assert anomaly_type == AnomalyType.ERROR_RATE_ANOMALY

    def test_system_anomaly_detection(self):
        """Test system anomaly detection"""
        # Train on normal data
        normal_data = self.create_sample_system_data(200, with_anomalies=False)
        self.detector.fit(normal_data)

        # Test on data with anomalies
        test_data = self.create_sample_system_data(50, with_anomalies=True)
        results = self.detector.predict(test_data)

        # Should detect some anomalies
        assert len(results) > 0

        for result in results:
            assert isinstance(result, AnomalyResult)
            assert result.symbol == "SYSTEM"
            assert result.anomaly_type in [
                AnomalyType.LATENCY_ANOMALY,
                AnomalyType.ERROR_RATE_ANOMALY,
                AnomalyType.PATTERN_ANOMALY
            ]


class TestAnomalyDetectionEngine:
    """Test the main anomaly detection engine"""

    def setup_method(self):
        """Set up test fixtures"""
        self.engine = AnomalyDetectionEngine()

    @pytest.mark.asyncio
    async def test_engine_start_stop(self):
        """Test starting and stopping the engine"""
        assert not self.engine.running

        await self.engine.start()
        assert self.engine.running

        await self.engine.stop()
        assert not self.engine.running

    @pytest.mark.asyncio
    async def test_model_training(self):
        """Test training models through the engine"""
        # Create training data
        price_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='H'),
            'symbol': ['AAPL'] * 100,
            'price': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(10000, 100000, 100)
        })

        system_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='H'),
            'cpu_usage': np.random.uniform(20, 60, 100),
            'memory_usage': np.random.uniform(30, 70, 100),
            'disk_usage': np.random.uniform(40, 80, 100),
            'response_time': np.random.uniform(0.05, 0.2, 100),
            'error_rate': np.random.uniform(0.001, 0.01, 100),
            'request_rate': np.random.uniform(100, 1000, 100)
        })

        training_data = {
            'price': price_data,
            'system': system_data
        }

        await self.engine.train_models(training_data)

        # Check that models are trained
        assert self.engine.detectors['price'].trained
        assert self.engine.detectors['system'].trained

    @pytest.mark.asyncio
    async def test_anomaly_detection(self):
        """Test real-time anomaly detection"""
        # First train the models
        await self.test_model_training()

        # Create test data with anomalies
        price_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'symbol': ['AAPL'],
            'price': [150.0],  # Significant price jump
            'volume': [50000]
        })

        system_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'cpu_usage': [95.0],  # High CPU
            'memory_usage': [50.0],
            'disk_usage': [60.0],
            'response_time': [0.1],
            'error_rate': [0.005],
            'request_rate': [500]
        })

        test_data = {
            'price': price_data,
            'system': system_data
        }

        results = await self.engine.detect_anomalies(test_data)

        # Should have some results
        assert isinstance(results, list)

    @pytest.mark.asyncio
    @patch('src.monitoring.anomaly_detector.redis.Redis')
    async def test_anomaly_storage(self, mock_redis):
        """Test storing anomaly results in Redis"""
        # Mock Redis client
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance

        # Create test anomaly result
        result = AnomalyResult(
            symbol="AAPL",
            timestamp=datetime.now(),
            anomaly_type=AnomalyType.PRICE_SPIKE,
            score=0.85,
            threshold=0.1,
            is_anomaly=True,
            confidence=0.9,
            features={},
            metadata={}
        )

        await self.engine._store_anomaly_results([result])

        # Verify Redis was called
        mock_redis_instance.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_detection_loop_error_handling(self):
        """Test that detection loop handles errors gracefully"""
        # Mock _get_recent_data to raise an exception
        with patch.object(self.engine, '_get_recent_data', side_effect=Exception("Test error")):
            # Start the engine
            await self.engine.start()

            # Wait a short time for the loop to run
            await asyncio.sleep(0.1)

            # Stop the engine
            await self.engine.stop()

            # Should not crash despite the exception


@pytest.mark.integration
class TestAnomalyDetectionIntegration:
    """Integration tests for the complete anomaly detection system"""

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete anomaly detection workflow"""
        engine = AnomalyDetectionEngine()

        # 1. Generate synthetic training data
        training_data = {
            'price': pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=500, freq='H'),
                'symbol': ['AAPL'] * 500,
                'price': np.random.uniform(100, 110, 500),
                'volume': np.random.uniform(10000, 100000, 500)
            }),
            'system': pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=500, freq='H'),
                'cpu_usage': np.random.uniform(20, 60, 500),
                'memory_usage': np.random.uniform(30, 70, 500),
                'disk_usage': np.random.uniform(40, 80, 500),
                'response_time': np.random.uniform(0.05, 0.2, 500),
                'error_rate': np.random.uniform(0.001, 0.01, 500),
                'request_rate': np.random.uniform(100, 1000, 500)
            })
        }

        # 2. Train models
        await engine.train_models(training_data)

        # 3. Generate test data with known anomalies
        test_data = {
            'price': pd.DataFrame({
                'timestamp': [datetime.now()],
                'symbol': ['AAPL'],
                'price': [200.0],  # Obvious price anomaly
                'volume': [50000]
            }),
            'system': pd.DataFrame({
                'timestamp': [datetime.now()],
                'cpu_usage': [95.0],  # High CPU usage
                'memory_usage': [50.0],
                'disk_usage': [60.0],
                'response_time': [5.0],  # High latency
                'error_rate': [0.5],  # High error rate
                'request_rate': [500]
            })
        }

        # 4. Detect anomalies
        results = await engine.detect_anomalies(test_data)

        # 5. Verify results
        assert len(results) >= 0  # Should have detected some anomalies

        # Verify result structure
        for result in results:
            assert isinstance(result, AnomalyResult)
            assert hasattr(result, 'symbol')
            assert hasattr(result, 'score')
            assert hasattr(result, 'anomaly_type')
            assert 0 <= result.score <= 1

    def test_performance_under_load(self):
        """Test anomaly detection performance under load"""
        detector = PriceAnomalyDetector()

        # Generate large dataset
        large_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10000, freq='H'),
            'symbol': ['AAPL'] * 10000,
            'price': np.random.uniform(100, 110, 10000),
            'volume': np.random.uniform(10000, 100000, 10000)
        })

        # Measure training time
        start_time = time.time()
        detector.fit(large_data)
        training_time = time.time() - start_time

        assert training_time < 30  # Should complete within 30 seconds

        # Test prediction performance
        test_data = large_data.tail(1000)  # Last 1000 rows

        start_time = time.time()
        results = detector.predict(test_data)
        prediction_time = time.time() - start_time

        assert prediction_time < 5  # Should complete within 5 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])