"""
Tests for Data Quality Controller
Tests data validation, quality metrics, and data profiling functionality
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from src.bulk_loading.data_quality_controller import (
    DataQualityController,
    MarketDataValidator,
    QualityMetrics,
    QualityLevel,
    ValidationIssue,
    ValidationSeverity,
    ValidationRule,
    DataProfile
)


class TestQualityMetrics:
    """Test quality metrics functionality"""

    def test_metrics_initialization(self):
        """Test quality metrics initialization"""
        metrics = QualityMetrics()

        assert metrics.completeness_score == 0.0
        assert metrics.accuracy_score == 0.0
        assert metrics.overall_score == 0.0
        assert metrics.quality_level == QualityLevel.UNACCEPTABLE

    def test_metrics_calculations(self):
        """Test quality metrics calculations"""
        metrics = QualityMetrics()
        metrics.completeness_score = 90.0
        metrics.accuracy_score = 85.0
        metrics.consistency_score = 88.0
        metrics.validity_score = 92.0
        metrics.timeliness_score = 87.0
        metrics.uniqueness_score = 95.0

        # Overall score should be average of all scores
        expected_overall = (90.0 + 85.0 + 88.0 + 92.0 + 87.0 + 95.0) / 6
        assert abs(metrics.overall_score - expected_overall) < 0.01

    def test_quality_level_determination(self):
        """Test quality level determination based on scores"""
        # Test excellent quality
        metrics = QualityMetrics()
        metrics.completeness_score = 95.0
        metrics.accuracy_score = 92.0
        metrics.consistency_score = 94.0
        metrics.validity_score = 96.0
        metrics.timeliness_score = 93.0
        metrics.uniqueness_score = 97.0

        assert metrics.quality_level == QualityLevel.EXCELLENT

        # Test good quality
        metrics = QualityMetrics()
        for field in ['completeness_score', 'accuracy_score', 'consistency_score',
                     'validity_score', 'timeliness_score', 'uniqueness_score']:
            setattr(metrics, field, 75.0)

        assert metrics.quality_level == QualityLevel.GOOD

        # Test poor quality
        metrics = QualityMetrics()
        for field in ['completeness_score', 'accuracy_score', 'consistency_score',
                     'validity_score', 'timeliness_score', 'uniqueness_score']:
            setattr(metrics, field, 35.0)

        assert metrics.quality_level == QualityLevel.POOR


class TestValidationIssue:
    """Test validation issue functionality"""

    def test_issue_creation(self):
        """Test validation issue creation"""
        issue = ValidationIssue(
            rule_name="test_rule",
            severity=ValidationSeverity.CRITICAL,
            message="Test validation failed",
            field="test_field",
            value="invalid_value",
            suggestion="Use valid value"
        )

        assert issue.rule_name == "test_rule"
        assert issue.severity == ValidationSeverity.CRITICAL
        assert issue.message == "Test validation failed"
        assert issue.field == "test_field"
        assert issue.value == "invalid_value"
        assert issue.suggestion == "Use valid value"
        assert isinstance(issue.timestamp, datetime)


class TestValidationRule:
    """Test validation rule functionality"""

    def test_rule_creation(self):
        """Test validation rule creation"""
        rule = ValidationRule(
            name="price_positive",
            description="Prices must be positive",
            severity=ValidationSeverity.CRITICAL,
            enabled=True,
            params={"min_value": 0}
        )

        assert rule.name == "price_positive"
        assert rule.description == "Prices must be positive"
        assert rule.severity == ValidationSeverity.CRITICAL
        assert rule.enabled is True
        assert rule.params["min_value"] == 0


class TestMarketDataValidator:
    """Test market data validator functionality"""

    @pytest.fixture
    def validator(self):
        """Create validator instance"""
        return MarketDataValidator()

    @pytest.mark.asyncio
    async def test_valid_symbol_validation(self, validator):
        """Test validation of valid symbols"""
        valid_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "A"]

        for symbol in valid_symbols:
            issues = await validator.validate_symbol(symbol)
            assert len(issues) == 0

    @pytest.mark.asyncio
    async def test_invalid_symbol_validation(self, validator):
        """Test validation of invalid symbols"""
        invalid_symbols = [
            "",                    # Empty
            "aapl",               # Lowercase
            "AAPL123",            # Contains numbers
            "VERY_LONG_SYMBOL",   # Too long
            "AA-PL",              # Contains hyphen
            None                  # None value
        ]

        for symbol in invalid_symbols:
            issues = await validator.validate_symbol(symbol or "")
            assert len(issues) > 0
            assert any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)

    @pytest.mark.asyncio
    async def test_valid_price_data_validation(self, validator):
        """Test validation of valid price data"""
        valid_record = {
            'symbol': 'AAPL',
            'timestamp': '2024-01-01T10:00:00',
            'open': 150.0,
            'high': 155.0,
            'low': 149.0,
            'close': 154.0,
            'volume': 1000000
        }

        issues = await validator.validate_price_data(valid_record)
        assert len(issues) == 0

    @pytest.mark.asyncio
    async def test_invalid_price_relationships(self, validator):
        """Test validation of invalid price relationships"""
        # High < Low
        invalid_record1 = {
            'open': 150.0,
            'high': 148.0,  # High less than low
            'low': 149.0,
            'close': 154.0,
            'volume': 1000000
        }

        issues = await validator.validate_price_data(invalid_record1)
        assert len(issues) > 0
        assert any('high' in issue.message.lower() and 'low' in issue.message.lower()
                  for issue in issues)

        # Open outside high-low range
        invalid_record2 = {
            'open': 160.0,  # Open > High
            'high': 155.0,
            'low': 149.0,
            'close': 154.0,
            'volume': 1000000
        }

        issues = await validator.validate_price_data(invalid_record2)
        assert len(issues) > 0
        assert any('open' in issue.message.lower() for issue in issues)

    @pytest.mark.asyncio
    async def test_negative_price_validation(self, validator):
        """Test validation of negative prices"""
        negative_price_record = {
            'open': -150.0,
            'high': 155.0,
            'low': 149.0,
            'close': 154.0,
            'volume': 1000000
        }

        issues = await validator.validate_price_data(negative_price_record)
        assert len(issues) > 0
        assert any('positive' in issue.message.lower() for issue in issues)

    @pytest.mark.asyncio
    async def test_negative_volume_validation(self, validator):
        """Test validation of negative volume"""
        negative_volume_record = {
            'open': 150.0,
            'high': 155.0,
            'low': 149.0,
            'close': 154.0,
            'volume': -1000000
        }

        issues = await validator.validate_price_data(negative_volume_record)
        assert len(issues) > 0
        assert any('volume' in issue.message.lower() and 'negative' in issue.message.lower()
                  for issue in issues)

    @pytest.mark.asyncio
    async def test_excessive_volatility_detection(self, validator):
        """Test detection of excessive volatility"""
        high_volatility_record = {
            'open': 100.0,
            'high': 200.0,  # 100% increase
            'low': 50.0,    # 50% decrease
            'close': 150.0,
            'volume': 1000000
        }

        issues = await validator.validate_price_data(high_volatility_record)
        assert len(issues) > 0
        assert any('volatility' in issue.message.lower() for issue in issues)

    @pytest.mark.asyncio
    async def test_valid_timestamp_validation(self, validator):
        """Test validation of valid timestamps"""
        valid_timestamps = [
            '2024-01-01T10:00:00',
            '2024-01-01T10:00:00Z',
            '2024-01-01T10:00:00+00:00',
            datetime.now(),
            1704105600  # Unix timestamp
        ]

        for timestamp in valid_timestamps:
            issues = await validator.validate_timestamp(timestamp)
            # Should have no critical issues (may have info/warning for weekend)
            critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
            assert len(critical_issues) == 0

    @pytest.mark.asyncio
    async def test_invalid_timestamp_validation(self, validator):
        """Test validation of invalid timestamps"""
        invalid_timestamps = [
            "",
            "invalid_date",
            "2024-13-01T10:00:00",  # Invalid month
            None
        ]

        for timestamp in invalid_timestamps:
            issues = await validator.validate_timestamp(timestamp)
            assert len(issues) > 0
            assert any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)

    @pytest.mark.asyncio
    async def test_future_timestamp_validation(self, validator):
        """Test validation of future timestamps"""
        future_timestamp = datetime.now() + timedelta(days=2)

        issues = await validator.validate_timestamp(future_timestamp)
        assert len(issues) > 0
        assert any('future' in issue.message.lower() for issue in issues)

    @pytest.mark.asyncio
    async def test_weekend_timestamp_detection(self, validator):
        """Test detection of weekend trading data"""
        # Create a Saturday timestamp
        saturday = datetime(2024, 1, 6, 10, 0, 0)  # Saturday

        issues = await validator.validate_timestamp(saturday)
        weekend_issues = [i for i in issues if 'weekend' in i.message.lower()]
        assert len(weekend_issues) > 0
        assert weekend_issues[0].severity == ValidationSeverity.INFO


class TestDataQualityController:
    """Test data quality controller functionality"""

    @pytest.fixture
    def quality_controller(self):
        """Create quality controller instance"""
        return DataQualityController()

    @pytest.mark.asyncio
    async def test_controller_initialization(self, quality_controller):
        """Test controller initialization"""
        assert 'market_data' in quality_controller.validators
        assert len(quality_controller.validation_rules) > 0
        assert len(quality_controller.custom_validators) == 0

    @pytest.mark.asyncio
    async def test_valid_record_validation(self, quality_controller):
        """Test validation of valid market data record"""
        valid_record = {
            'symbol': 'AAPL',
            'timestamp': '2024-01-01T10:00:00',
            'open': 150.0,
            'high': 155.0,
            'low': 149.0,
            'close': 154.0,
            'volume': 1000000
        }

        issues, metrics = await quality_controller.validate_record(valid_record)

        assert len(issues) == 0 or all(issue.severity != ValidationSeverity.CRITICAL for issue in issues)
        assert metrics.overall_score > 50  # Should have decent quality score

    @pytest.mark.asyncio
    async def test_invalid_record_validation(self, quality_controller):
        """Test validation of invalid market data record"""
        invalid_record = {
            'symbol': 'invalid_symbol_123',
            'timestamp': 'invalid_date',
            'open': -150.0,
            'high': 155.0,
            'low': 149.0,
            'close': 154.0,
            'volume': -1000000
        }

        issues, metrics = await quality_controller.validate_record(invalid_record)

        assert len(issues) > 0
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        assert len(critical_issues) > 0
        assert metrics.overall_score < 50  # Should have low quality score

    @pytest.mark.asyncio
    async def test_batch_validation(self, quality_controller):
        """Test batch validation functionality"""
        records = [
            {
                'symbol': 'AAPL',
                'timestamp': '2024-01-01T10:00:00',
                'open': 150.0,
                'high': 155.0,
                'low': 149.0,
                'close': 154.0,
                'volume': 1000000
            },
            {
                'symbol': 'GOOGL',
                'timestamp': '2024-01-01T10:00:00',
                'open': 2800.0,
                'high': 2850.0,
                'low': 2790.0,
                'close': 2820.0,
                'volume': 500000
            },
            {
                'symbol': 'invalid',
                'timestamp': 'invalid',
                'open': -100.0,
                'high': 105.0,
                'low': 99.0,
                'close': 102.0,
                'volume': -50000
            }
        ]

        all_issues, all_metrics, batch_metrics = await quality_controller.validate_batch(records)

        assert len(all_issues) == 3
        assert len(all_metrics) == 3

        # First two records should be valid
        assert len(all_issues[0]) == 0 or all(i.severity != ValidationSeverity.CRITICAL for i in all_issues[0])
        assert len(all_issues[1]) == 0 or all(i.severity != ValidationSeverity.CRITICAL for i in all_issues[1])

        # Third record should have issues
        assert len(all_issues[2]) > 0
        assert any(i.severity == ValidationSeverity.CRITICAL for i in all_issues[2])

        # Batch metrics should reflect overall quality
        assert isinstance(batch_metrics, QualityMetrics)

    @pytest.mark.asyncio
    async def test_data_profiling(self, quality_controller):
        """Test data profiling functionality"""
        records = [
            {
                'symbol': 'AAPL',
                'timestamp': '2024-01-01T10:00:00',
                'open': 150.0,
                'high': 155.0,
                'low': 149.0,
                'close': 154.0,
                'volume': 1000000,
                'source': 'nasdaq'
            },
            {
                'symbol': 'GOOGL',
                'timestamp': '2024-01-02T10:00:00',
                'open': 2800.0,
                'high': 2850.0,
                'low': 2790.0,
                'close': 2820.0,
                'volume': 500000,
                'source': 'nasdaq'
            },
            {
                'symbol': 'AAPL',
                'timestamp': '2024-01-03T10:00:00',
                'open': 152.0,
                'high': 157.0,
                'low': 151.0,
                'close': 156.0,
                'volume': 1200000,
                'source': 'nyse'
            }
        ]

        profile = await quality_controller.profile_data(records)

        assert isinstance(profile, DataProfile)
        assert profile.total_records == 3
        assert profile.total_fields > 0

        # Check field completeness
        assert 'symbol' in profile.field_completeness
        assert profile.field_completeness['symbol'] == 100.0  # All records have symbol

        # Check numeric statistics
        assert 'open' in profile.numeric_stats
        assert 'min' in profile.numeric_stats['open']
        assert 'max' in profile.numeric_stats['open']
        assert 'mean' in profile.numeric_stats['open']

        # Check string statistics
        assert 'symbol' in profile.string_stats
        assert 'unique_values' in profile.string_stats['symbol']

        # Check value distributions
        assert 'symbol' in profile.value_distributions

    @pytest.mark.asyncio
    async def test_anomaly_detection(self, quality_controller):
        """Test anomaly detection functionality"""
        # Create records with one outlier
        records = []
        for i in range(10):
            records.append({
                'symbol': 'AAPL',
                'volume': 1000000 + i * 10000  # Normal volumes
            })

        # Add outlier
        records.append({
            'symbol': 'AAPL',
            'volume': 10000000  # Outlier volume
        })

        anomalies = await quality_controller.detect_anomalies(records, 'volume')

        assert len(anomalies) > 0
        assert anomalies[0]['type'] == 'statistical_outlier'
        assert anomalies[0]['field'] == 'volume'
        assert anomalies[0]['z_score'] > 3

    @pytest.mark.asyncio
    async def test_data_correction_suggestions(self, quality_controller):
        """Test data correction suggestions"""
        record = {
            'symbol': 'aapl',  # Should be uppercase
            'timestamp': '2024-01-01T10:00:00',
            'open': -150.0,    # Should be positive
            'high': 155.0,
            'low': 149.0,
            'close': 154.0,
            'volume': 1000000
        }

        issues, _ = await quality_controller.validate_record(record)

        suggestions = await quality_controller.suggest_data_corrections(record, issues)

        assert 'symbol' in suggestions
        assert suggestions['symbol'] == 'AAPL'

        # Should suggest absolute value for negative price
        if 'open' in suggestions:
            assert suggestions['open'] == 150.0

    @pytest.mark.asyncio
    async def test_custom_validator_integration(self, quality_controller):
        """Test custom validator integration"""
        async def volume_threshold_validator(record):
            issues = []
            volume = record.get('volume', 0)
            if volume < 100000:
                issues.append(ValidationIssue(
                    rule_name="volume_threshold",
                    severity=ValidationSeverity.WARNING,
                    message="Volume below recommended threshold",
                    field="volume",
                    value=volume
                ))
            return issues

        quality_controller.add_custom_validator(volume_threshold_validator)

        # Test record with low volume
        record = {
            'symbol': 'AAPL',
            'timestamp': '2024-01-01T10:00:00',
            'open': 150.0,
            'high': 155.0,
            'low': 149.0,
            'close': 154.0,
            'volume': 50000  # Low volume
        }

        issues, _ = await quality_controller.validate_record(record)

        # Should have custom validation issue
        custom_issues = [i for i in issues if i.rule_name == "volume_threshold"]
        assert len(custom_issues) > 0

    @pytest.mark.asyncio
    async def test_quality_summary_generation(self, quality_controller):
        """Test quality summary generation"""
        # Create metrics with different quality levels
        metrics_list = []

        # Excellent quality
        excellent_metrics = QualityMetrics()
        for field in ['completeness_score', 'accuracy_score', 'consistency_score',
                     'validity_score', 'timeliness_score', 'uniqueness_score']:
            setattr(excellent_metrics, field, 95.0)
        metrics_list.append(excellent_metrics)

        # Good quality
        good_metrics = QualityMetrics()
        for field in ['completeness_score', 'accuracy_score', 'consistency_score',
                     'validity_score', 'timeliness_score', 'uniqueness_score']:
            setattr(good_metrics, field, 75.0)
        metrics_list.append(good_metrics)

        # Poor quality
        poor_metrics = QualityMetrics()
        for field in ['completeness_score', 'accuracy_score', 'consistency_score',
                     'validity_score', 'timeliness_score', 'uniqueness_score']:
            setattr(poor_metrics, field, 35.0)
        metrics_list.append(poor_metrics)

        summary = quality_controller.get_quality_summary(metrics_list)

        assert summary['total_records'] == 3
        assert 'average_quality_score' in summary
        assert 'quality_level_distribution' in summary
        assert 'records_above_threshold' in summary

        # Check distribution
        distribution = summary['quality_level_distribution']
        assert distribution['excellent'] == 1
        assert distribution['good'] == 1
        assert distribution['poor'] == 1

    def test_quality_threshold_configuration(self, quality_controller):
        """Test quality threshold configuration"""
        # Test default thresholds
        assert quality_controller.quality_thresholds[QualityLevel.EXCELLENT] == 90.0
        assert quality_controller.quality_thresholds[QualityLevel.GOOD] == 70.0

        # Test threshold modification
        quality_controller.quality_thresholds[QualityLevel.EXCELLENT] = 95.0
        assert quality_controller.quality_thresholds[QualityLevel.EXCELLENT] == 95.0


class TestDataQualityIntegration:
    """Integration tests for data quality system"""

    @pytest.mark.integration
    async def test_large_dataset_validation(self):
        """Test validation of large datasets"""
        # This test would validate performance with large datasets
        pytest.skip("Integration test requires large dataset processing")

    @pytest.mark.integration
    async def test_real_market_data_validation(self):
        """Test validation with real market data"""
        # This test would use real market data for validation
        pytest.skip("Integration test requires real market data")

    @pytest.mark.integration
    async def test_quality_monitoring_over_time(self):
        """Test quality monitoring over time"""
        # This test would track quality metrics over time
        pytest.skip("Integration test requires time-series quality tracking")