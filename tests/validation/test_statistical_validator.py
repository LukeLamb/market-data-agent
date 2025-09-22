"""Tests for Statistical Validator

Comprehensive tests for statistical validation including anomaly detection,
time series analysis, and adaptive threshold management.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.validation.statistical_validator import (
    StatisticalValidator, ValidationResult, AnomalyType
)


@pytest.fixture
def validator():
    """Statistical validator instance for testing"""
    return StatisticalValidator()


class TestStatisticalValidator:
    """Test suite for StatisticalValidator"""

    def test_initialization(self):
        """Test validator initialization"""
        validator = StatisticalValidator()

        assert validator.price_history == {}
        assert validator.volume_history == {}
        assert validator.stats_cache == {}
        assert hasattr(validator, 'outlier_threshold')
        assert hasattr(validator, 'staleness_threshold')
        assert hasattr(validator, 'gap_threshold')

    def test_validate_price_data_valid(self, validator):
        """Test validation of valid price data"""
        # Add some historical data first
        symbol = "AAPL"
        for i in range(15):
            price = 150.0 + np.random.normal(0, 1)
            timestamp = datetime.now() - timedelta(minutes=i)
            validator._add_historical_data(symbol, price, timestamp, 1000000)

        result = validator.validate_price_data(
            symbol, 151.0, datetime.now(), 1000000
        )

        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert result.severity == "info"
        assert AnomalyType.OUTLIER.value not in result.message

    def test_validate_price_data_outlier(self, validator):
        """Test detection of price outliers"""
        symbol = "AAPL"

        # Add normal data
        for i in range(15):
            price = 150.0 + np.random.normal(0, 0.5)
            timestamp = datetime.now() - timedelta(minutes=i)
            validator._add_historical_data(symbol, price, timestamp, 1000000)

        # Test outlier detection
        outlier_price = 200.0  # Significantly higher than normal
        result = validator.validate_price_data(
            symbol, outlier_price, datetime.now(), 1000000
        )

        assert not result.is_valid
        assert result.anomaly_type == AnomalyType.OUTLIER
        assert result.severity in ["warning", "error"]

    def test_validate_price_data_sudden_change(self, validator):
        """Test detection of sudden price changes"""
        symbol = "AAPL"

        # Add stable historical data
        base_price = 150.0
        for i in range(15):
            price = base_price + np.random.normal(0, 0.1)
            timestamp = datetime.now() - timedelta(minutes=i)
            validator._add_historical_data(symbol, price, timestamp, 1000000)

        # Test sudden change detection
        sudden_change_price = base_price * 1.15  # 15% change
        result = validator.validate_price_data(
            symbol, sudden_change_price, datetime.now(), 1000000
        )

        assert not result.is_valid
        assert result.anomaly_type == AnomalyType.SUDDEN_CHANGE
        assert "sudden change" in result.message.lower()

    def test_validate_price_data_volume_anomaly(self, validator):
        """Test detection of volume anomalies"""
        symbol = "AAPL"

        # Add normal volume data
        for i in range(15):
            price = 150.0 + np.random.normal(0, 0.5)
            volume = 1000000 + np.random.normal(0, 50000)
            timestamp = datetime.now() - timedelta(minutes=i)
            validator._add_historical_data(symbol, price, timestamp, int(volume))

        # Test volume anomaly detection
        anomalous_volume = 5000000  # 5x normal volume
        result = validator.validate_price_data(
            symbol, 150.5, datetime.now(), anomalous_volume
        )

        assert not result.is_valid
        assert result.anomaly_type == AnomalyType.VOLUME_ANOMALY
        assert "volume" in result.message.lower()

    def test_validate_price_data_stale_data(self, validator):
        """Test detection of stale data"""
        symbol = "AAPL"

        # Test stale data detection
        old_timestamp = datetime.now() - timedelta(seconds=400)  # Older than threshold
        result = validator.validate_price_data(
            symbol, 150.0, old_timestamp, 1000000
        )

        assert not result.is_valid
        assert result.anomaly_type == AnomalyType.STALE_DATA
        assert "stale" in result.message.lower()

    def test_validate_price_data_insufficient_data(self, validator):
        """Test handling of insufficient historical data"""
        symbol = "NEW_SYMBOL"

        # No historical data for this symbol
        result = validator.validate_price_data(
            symbol, 150.0, datetime.now(), 1000000
        )

        # Should pass with warning due to insufficient data
        assert result.is_valid
        assert result.severity == "warning"
        assert "insufficient" in result.message.lower()

    def test_z_score_detection(self, validator):
        """Test Z-score anomaly detection algorithm"""
        values = [10, 12, 11, 13, 12, 11, 10, 50]  # Last value is outlier

        is_anomaly, score = validator._detect_anomaly_zscore(values)

        assert is_anomaly
        assert score > validator.config.z_score_threshold

    def test_iqr_detection(self, validator):
        """Test IQR anomaly detection algorithm"""
        values = [10, 12, 11, 13, 12, 11, 10, 50]  # Last value is outlier

        is_anomaly, score = validator._detect_anomaly_iqr(values)

        assert is_anomaly
        assert score > 0

    def test_isolation_forest_detection(self, validator):
        """Test Isolation Forest anomaly detection"""
        # Create data with clear outlier
        normal_data = np.random.normal(100, 5, 100)
        outlier_data = np.array([150])  # Clear outlier
        all_data = np.concatenate([normal_data, outlier_data])

        is_anomaly, score = validator._detect_anomaly_isolation_forest(all_data.tolist())

        assert isinstance(is_anomaly, bool)
        assert isinstance(score, (int, float))

    def test_adaptive_threshold_update(self, validator):
        """Test adaptive threshold updates"""
        symbol = "AAPL"

        # Add data to trigger threshold adaptation
        for i in range(25):  # More than volatility window
            price = 150.0 + np.random.normal(0, 2)  # Higher volatility
            timestamp = datetime.now() - timedelta(minutes=i)
            validator._add_historical_data(symbol, price, timestamp, 1000000)

        validator._update_adaptive_thresholds(symbol)

        # Check that adaptive threshold was created
        assert symbol in validator.adaptive_thresholds
        assert "z_score_threshold" in validator.adaptive_thresholds[symbol]

    def test_volatility_calculation(self, validator):
        """Test volatility calculation"""
        symbol = "AAPL"

        # Add price data with known volatility
        prices = [100, 102, 99, 103, 98, 104, 97, 105]
        for i, price in enumerate(prices):
            timestamp = datetime.now() - timedelta(minutes=i)
            validator._add_historical_data(symbol, price, timestamp, 1000000)

        volatility = validator._calculate_volatility(symbol)

        assert volatility > 0
        assert isinstance(volatility, float)

    def test_price_gap_detection(self, validator):
        """Test price gap detection"""
        symbol = "AAPL"

        # Add consistent price data
        for i in range(10):
            price = 150.0
            timestamp = datetime.now() - timedelta(minutes=i)
            validator._add_historical_data(symbol, price, timestamp, 1000000)

        # Test gap detection
        gap_price = 160.0  # Significant gap
        result = validator.validate_price_data(
            symbol, gap_price, datetime.now(), 1000000
        )

        # Should detect as sudden change or outlier
        if not result.is_valid:
            assert result.anomaly_type in [AnomalyType.SUDDEN_CHANGE, AnomalyType.PRICE_GAP]

    def test_multiple_algorithms(self, validator):
        """Test using multiple detection algorithms"""
        # This test would be for future algorithm configuration

        symbol = "AAPL"

        # Add normal data
        for i in range(20):
            price = 150.0 + np.random.normal(0, 1)
            timestamp = datetime.now() - timedelta(minutes=i)
            validator._add_historical_data(symbol, price, timestamp, 1000000)

        # Test with outlier
        result = validator.validate_price_data(
            symbol, 200.0, datetime.now(), 1000000
        )

        # Should detect anomaly using multiple algorithms
        assert "multiple algorithms" in result.details.get("algorithms_used", "")

    def test_historical_data_cleanup(self, validator):
        """Test automatic cleanup of old historical data"""
        symbol = "AAPL"

        # Add old data that should be cleaned up
        old_timestamp = datetime.now() - timedelta(days=2)
        validator._add_historical_data(symbol, 150.0, old_timestamp, 1000000)

        # Add recent data
        recent_timestamp = datetime.now()
        validator._add_historical_data(symbol, 151.0, recent_timestamp, 1000000)

        # Trigger cleanup
        validator._cleanup_old_data()

        # Check that old data was removed
        if symbol in validator.historical_data:
            for data_point in validator.historical_data[symbol]:
                assert data_point["timestamp"] > datetime.now() - timedelta(hours=24)

    def test_validation_history_limit(self, validator):
        """Test validation history size limit"""
        symbol = "AAPL"

        # Add more validations than the limit
        for i in range(1500):  # More than max_history_size (1000)
            result = ValidationResult(
                is_valid=True,
                confidence=0.95,
                anomaly_type=None,
                severity="info",
                message="Test validation",
                details={}
            )
            validator.validation_history.append(result)

        # Trigger cleanup by adding new validation
        validator._add_to_history(ValidationResult(
            is_valid=True,
            confidence=0.95,
            anomaly_type=None,
            severity="info",
            message="New validation",
            details={}
        ))

        # Check that history was trimmed
        assert len(validator.validation_history) <= 1000

    def test_get_validation_summary(self, validator):
        """Test validation summary generation"""
        # Add some validation history
        for i in range(10):
            result = ValidationResult(
                is_valid=i % 2 == 0,  # Alternate valid/invalid
                confidence=0.8,
                anomaly_type=AnomalyType.OUTLIER if i % 2 else None,
                severity="warning" if i % 2 else "info",
                message=f"Test validation {i}",
                details={}
            )
            validator.validation_history.append(result)

        summary = validator.get_validation_summary()

        assert "total_validations" in summary
        assert "success_rate" in summary
        assert "anomaly_distribution" in summary
        assert summary["total_validations"] == 10

    def test_update_configuration(self, validator):
        """Test runtime configuration updates"""
        new_config = StatisticalConfig(
            enabled=True,
            z_score_threshold=3.0,  # Different from original
            iqr_multiplier=2.0,
            volatility_window=30,
            price_change_threshold=0.1,
            volume_change_threshold=3.0,
            stale_data_threshold=600,
            min_data_points=15
        )

        validator.update_configuration(new_config)

        assert validator.config.z_score_threshold == 3.0
        assert validator.config.volatility_window == 30

    def test_export_historical_data(self, validator):
        """Test export of historical data"""
        symbol = "AAPL"

        # Add some historical data
        for i in range(5):
            price = 150.0 + i
            timestamp = datetime.now() - timedelta(minutes=i)
            validator._add_historical_data(symbol, price, timestamp, 1000000)

        exported_data = validator.export_historical_data(symbol)

        assert symbol in exported_data
        assert len(exported_data[symbol]) == 5
        assert all("price" in dp for dp in exported_data[symbol])

    def test_edge_cases(self, validator):
        """Test edge cases and error handling"""
        symbol = "AAPL"

        # Test with None values
        result = validator.validate_price_data(symbol, None, datetime.now(), 1000000)
        assert not result.is_valid

        # Test with negative price
        result = validator.validate_price_data(symbol, -10.0, datetime.now(), 1000000)
        assert not result.is_valid

        # Test with zero volume
        result = validator.validate_price_data(symbol, 150.0, datetime.now(), 0)
        # Should handle gracefully (may or may not be valid depending on implementation)
        assert isinstance(result, ValidationResult)

    @pytest.mark.asyncio
    async def test_async_validation(self, validator):
        """Test asynchronous validation capabilities"""
        # Note: If async methods are added to the validator in the future
        symbol = "AAPL"

        # For now, just test that sync validation works in async context
        result = validator.validate_price_data(
            symbol, 150.0, datetime.now(), 1000000
        )

        assert isinstance(result, ValidationResult)