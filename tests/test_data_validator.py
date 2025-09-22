"""Tests for Data Validation Framework"""

import pytest
from datetime import datetime, timedelta

from src.validation.data_validator import (
    DataValidator,
    ValidationResult,
    ValidationSeverity
)
from src.data_sources.base import PriceData, CurrentPrice


class TestDataValidator:
    """Test cases for data validation framework"""

    def setup_method(self):
        """Set up test fixtures"""
        self.validator = DataValidator()

    def test_validator_initialization(self):
        """Test validator initialization with default config"""
        assert self.validator.max_price_change_percent == 50.0
        assert self.validator.min_volume == 0
        assert self.validator.min_price == 0.01
        assert self.validator.max_price == 100000.0

    def test_validator_initialization_with_config(self):
        """Test validator initialization with custom config"""
        config = {
            "max_price_change_percent": 25.0,
            "min_volume": 1000,
            "min_price": 1.0
        }
        validator = DataValidator(config)
        assert validator.max_price_change_percent == 25.0
        assert validator.min_volume == 1000
        assert validator.min_price == 1.0

    def test_validate_valid_price_data(self):
        """Test validation of valid price data"""
        data = PriceData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open_price=150.0,
            high_price=155.0,
            low_price=149.0,
            close_price=154.0,
            volume=1000000,
            source="test_source",
            quality_score=95
        )

        result = self.validator.validate_price_data(data)

        assert result.is_valid is True
        assert result.quality_score > 90
        assert len(result.issues) == 0

    def test_validate_negative_price(self):
        """Test validation with negative price"""
        data = PriceData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open_price=-150.0,  # Invalid negative price
            high_price=155.0,
            low_price=149.0,
            close_price=154.0,
            volume=1000000,
            source="test_source",
            quality_score=95
        )

        result = self.validator.validate_price_data(data)

        assert result.is_valid is False
        assert any(issue["type"] == "negative_price" for issue in result.issues)
        assert any(issue["severity"] == "critical" for issue in result.issues)

    def test_validate_ohlc_inconsistency(self):
        """Test validation with OHLC inconsistencies"""
        data = PriceData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open_price=150.0,
            high_price=148.0,  # High less than open - invalid
            low_price=149.0,
            close_price=154.0,
            volume=1000000,
            source="test_source",
            quality_score=95
        )

        result = self.validator.validate_price_data(data)

        assert result.is_valid is False
        assert any(issue["type"] == "ohlc_inconsistency" for issue in result.issues)

    def test_validate_high_less_than_low(self):
        """Test validation when high price is less than low price"""
        data = PriceData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open_price=150.0,
            high_price=148.0,  # High less than low - critical error
            low_price=149.0,
            close_price=148.5,
            volume=1000000,
            source="test_source",
            quality_score=95
        )

        result = self.validator.validate_price_data(data)

        assert result.is_valid is False
        critical_issues = [i for i in result.issues if i["severity"] == "critical"]
        assert len(critical_issues) > 0

    def test_validate_future_timestamp(self):
        """Test validation with future timestamp"""
        future_time = datetime.now() + timedelta(hours=1)
        data = PriceData(
            symbol="AAPL",
            timestamp=future_time,
            open_price=150.0,
            high_price=155.0,
            low_price=149.0,
            close_price=154.0,
            volume=1000000,
            source="test_source",
            quality_score=95
        )

        result = self.validator.validate_price_data(data)

        assert any(issue["type"] == "future_timestamp" for issue in result.issues)

    def test_validate_current_price_valid(self):
        """Test validation of valid current price"""
        price = CurrentPrice(
            symbol="AAPL",
            price=150.0,
            timestamp=datetime.now(),
            volume=1000000,
            bid=149.95,
            ask=150.05,
            source="test_source",
            quality_score=95
        )

        result = self.validator.validate_current_price(price)

        assert result.is_valid is True
        assert result.quality_score > 90

    def test_validate_current_price_invalid(self):
        """Test validation of invalid current price"""
        price = CurrentPrice(
            symbol="AAPL",
            price=-150.0,  # Invalid negative price
            timestamp=datetime.now(),
            source="test_source",
            quality_score=95
        )

        result = self.validator.validate_current_price(price)

        assert result.is_valid is False
        assert any(issue["type"] == "invalid_price" for issue in result.issues)

    def test_validate_bid_ask_spread(self):
        """Test bid-ask spread validation"""
        price = CurrentPrice(
            symbol="AAPL",
            price=150.0,
            timestamp=datetime.now(),
            bid=150.05,  # Bid higher than ask - invalid
            ask=149.95,
            source="test_source",
            quality_score=95
        )

        result = self.validator.validate_current_price(price)

        assert any(issue["type"] == "invalid_bid_ask" for issue in result.issues)

    def test_validate_wide_bid_ask_spread(self):
        """Test wide bid-ask spread detection"""
        price = CurrentPrice(
            symbol="AAPL",
            price=100.0,
            timestamp=datetime.now(),
            bid=90.0,  # 20% spread - very wide
            ask=110.0,
            source="test_source",
            quality_score=95
        )

        result = self.validator.validate_current_price(price)

        assert any(issue["type"] == "wide_bid_ask_spread" for issue in result.issues)

    def test_validate_price_history_valid(self):
        """Test validation of valid price history"""
        data_list = [
            PriceData(
                symbol="AAPL",
                timestamp=datetime(2023, 1, 1, 9, 30),
                open_price=150.0,
                high_price=155.0,
                low_price=149.0,
                close_price=154.0,
                volume=1000000,
                source="test_source",
                quality_score=95
            ),
            PriceData(
                symbol="AAPL",
                timestamp=datetime(2023, 1, 2, 9, 30),
                open_price=154.0,
                high_price=158.0,
                low_price=153.0,
                close_price=157.0,
                volume=1100000,
                source="test_source",
                quality_score=95
            )
        ]

        result = self.validator.validate_price_history(data_list)

        assert result.is_valid is True
        assert result.quality_score > 80

    def test_validate_price_history_empty(self):
        """Test validation of empty price history"""
        result = self.validator.validate_price_history([])

        assert result.is_valid is False
        assert any(issue["type"] == "empty_data" for issue in result.issues)

    def test_validate_large_price_change(self):
        """Test detection of large price changes"""
        data_list = [
            PriceData(
                symbol="AAPL",
                timestamp=datetime(2023, 1, 1, 9, 30),
                open_price=150.0,
                high_price=155.0,
                low_price=149.0,
                close_price=154.0,
                volume=1000000,
                source="test_source",
                quality_score=95
            ),
            PriceData(
                symbol="AAPL",
                timestamp=datetime(2023, 1, 2, 9, 30),
                open_price=154.0,
                high_price=300.0,  # 100% jump - suspicious
                low_price=153.0,
                close_price=280.0,
                volume=1100000,
                source="test_source",
                quality_score=95
            )
        ]

        result = self.validator.validate_price_history(data_list)

        assert any(issue["type"] == "sequence_large_price_change" for issue in result.issues)

    def test_compare_sources_matching(self):
        """Test comparison of matching data from different sources"""
        data1 = PriceData(
            symbol="AAPL",
            timestamp=datetime(2023, 1, 1, 9, 30),
            open_price=150.0,
            high_price=155.0,
            low_price=149.0,
            close_price=154.0,
            volume=1000000,
            source="source1",
            quality_score=95
        )

        data2 = PriceData(
            symbol="AAPL",
            timestamp=datetime(2023, 1, 1, 9, 30),
            open_price=150.5,  # 0.33% difference - within tolerance
            high_price=155.5,
            low_price=149.2,
            close_price=154.3,
            volume=1050000,
            source="source2",
            quality_score=95
        )

        result = self.validator.compare_sources(data1, data2, tolerance_percent=1.0)

        assert result.is_valid is True
        assert result.quality_score > 80

    def test_compare_sources_divergent(self):
        """Test comparison of divergent data from different sources"""
        data1 = PriceData(
            symbol="AAPL",
            timestamp=datetime(2023, 1, 1, 9, 30),
            open_price=150.0,
            high_price=155.0,
            low_price=149.0,
            close_price=154.0,
            volume=1000000,
            source="source1",
            quality_score=95
        )

        data2 = PriceData(
            symbol="AAPL",
            timestamp=datetime(2023, 1, 1, 9, 30),
            open_price=160.0,  # 6.67% difference - outside tolerance
            high_price=165.0,
            low_price=159.0,
            close_price=164.0,
            volume=1000000,
            source="source2",
            quality_score=95
        )

        result = self.validator.compare_sources(data1, data2, tolerance_percent=5.0)

        assert any(issue["type"] == "price_divergence" for issue in result.issues)

    def test_compare_sources_symbol_mismatch(self):
        """Test comparison with mismatched symbols"""
        data1 = PriceData(
            symbol="AAPL",
            timestamp=datetime(2023, 1, 1, 9, 30),
            open_price=150.0,
            high_price=155.0,
            low_price=149.0,
            close_price=154.0,
            volume=1000000,
            source="source1",
            quality_score=95
        )

        data2 = PriceData(
            symbol="GOOGL",  # Different symbol
            timestamp=datetime(2023, 1, 1, 9, 30),
            open_price=2800.0,
            high_price=2850.0,
            low_price=2790.0,
            close_price=2830.0,
            volume=500000,
            source="source2",
            quality_score=95
        )

        result = self.validator.compare_sources(data1, data2)

        assert any(issue["type"] == "symbol_mismatch" for issue in result.issues)
        assert any(issue["severity"] == "critical" for issue in result.issues)

    def test_validation_result_operations(self):
        """Test ValidationResult operations"""
        result = ValidationResult(is_valid=True, quality_score=100)

        # Test adding issues
        result.add_issue(
            "test_issue",
            ValidationSeverity.MEDIUM,
            "Test issue description",
            "test_field"
        )

        assert len(result.issues) == 1
        assert result.issues[0]["type"] == "test_issue"
        assert result.issues[0]["severity"] == "medium"

        # Test adding warnings
        result.add_warning("Test warning")
        assert len(result.warnings) == 1
        assert result.warnings[0] == "Test warning"

        # Test severity counts
        severity_counts = result.get_severity_counts()
        assert severity_counts["medium"] == 1
        assert severity_counts["high"] == 0

    def test_price_out_of_range(self):
        """Test price range validation"""
        config = {"min_price": 10.0, "max_price": 1000.0}
        validator = DataValidator(config)

        data = PriceData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open_price=5.0,  # Below minimum
            high_price=155.0,
            low_price=149.0,
            close_price=154.0,
            volume=1000000,
            source="test_source",
            quality_score=95
        )

        result = validator.validate_price_data(data)

        assert any(issue["type"] == "price_out_of_range" for issue in result.issues)

    def test_volume_validation(self):
        """Test volume validation"""
        config = {"min_volume": 1000}
        validator = DataValidator(config)

        data = PriceData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open_price=150.0,
            high_price=155.0,
            low_price=149.0,
            close_price=154.0,
            volume=500,  # Below minimum
            source="test_source",
            quality_score=95
        )

        result = validator.validate_price_data(data)

        assert any(issue["type"] == "invalid_volume" for issue in result.issues)

    def test_anomaly_detection(self):
        """Test statistical anomaly detection"""
        # Create data with one outlier
        base_price = 100.0
        data_list = []

        # Normal data points
        for i in range(10):
            price_change = 0.01 * (i - 5)  # Small changes
            data_list.append(PriceData(
                symbol="AAPL",
                timestamp=datetime(2023, 1, 1 + i),
                open_price=base_price + price_change,
                high_price=base_price + price_change + 1,
                low_price=base_price + price_change - 1,
                close_price=base_price + price_change,
                volume=1000000,
                source="test_source",
                quality_score=95
            ))

        # Add outlier with a more dramatic price change to trigger the validation
        data_list.append(PriceData(
            symbol="AAPL",
            timestamp=datetime(2023, 1, 11),
            open_price=base_price + 45,
            high_price=base_price + 200,  # Very large jump
            low_price=base_price + 40,
            close_price=base_price + 180,  # 80% increase from previous close
            volume=1000000,
            source="test_source",
            quality_score=95
        ))

        result = self.validator.validate_price_history(data_list)

        # Should detect the large price change
        assert any(issue["type"] == "sequence_large_price_change" for issue in result.issues)

    def test_quality_score_calculation(self):
        """Test quality score calculation with various issues"""
        data = PriceData(
            symbol="AAPL",
            timestamp=datetime.now() - timedelta(hours=2),  # Slightly old
            open_price=150.0,
            high_price=155.0,
            low_price=149.0,
            close_price=154.0,
            volume=1000000,
            source="test_source",
            quality_score=95
        )

        result = self.validator.validate_price_data(data)

        # Should have high quality score for valid data
        assert result.quality_score >= 85

        # Now test with issues
        bad_data = PriceData(
            symbol="AAPL",
            timestamp=datetime.now() + timedelta(hours=1),  # Future timestamp
            open_price=150.0,
            high_price=148.0,  # Inconsistent OHLC
            low_price=149.0,
            close_price=154.0,
            volume=-1000,  # Negative volume
            source="test_source",
            quality_score=95
        )

        bad_result = self.validator.validate_price_data(bad_data)

        # Should have much lower quality score
        assert bad_result.quality_score < result.quality_score