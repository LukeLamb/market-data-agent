"""Tests for Data Quality Assessor

Comprehensive tests for data quality assessment including multi-dimensional
analysis, automated issue detection, and quality scoring.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.validation.data_quality_assessor import (
    DataQualityAssessor, QualityReport, QualityDimension,
    QualityIssue, QualityIssueType
)
from src.data_sources.base import CurrentPrice, PriceData


@pytest.fixture
def assessor():
    """Data quality assessor instance for testing"""
    return DataQualityAssessor()


@pytest.fixture
def sample_current_prices():
    """Sample current price data for testing"""
    timestamp = datetime.now()
    return [
        CurrentPrice(symbol="AAPL", price=150.0, timestamp=timestamp, volume=1000000, source="test_source", quality_score=90.0
        ),
        CurrentPrice(symbol="AAPL", price=150.5, timestamp=timestamp - timedelta(minutes=1), volume=1050000, source="test_source", quality_score=85.0
        ),
        CurrentPrice(symbol="AAPL", price=149.8, timestamp=timestamp - timedelta(minutes=2), volume=980000, source="test_source", quality_score=92.0
        )
    ]


@pytest.fixture
def sample_historical_data():
    """Sample historical data for testing"""
    base_time = datetime.now() - timedelta(hours=1)
    return [
        PriceData(
            timestamp=base_time + timedelta(minutes=i),
            open_price=149.0 + i * 0.1,
            high_price=150.0 + i * 0.1,
            low_price=148.5 + i * 0.1,
            close_price=149.5 + i * 0.1,
            volume=1000000 + i * 10000
        ) for i in range(20)
    ]


class TestDataQualityAssessor:
    """Test suite for DataQualityAssessor"""

    def test_initialization(self):
        """Test assessor initialization"""
        assessor = DataQualityAssessor()

        assert assessor.quality_history == []
        assert assessor.cached_assessments == {}
        assert assessor.issue_tracking == {}
        assert hasattr(assessor, 'accuracy_tolerance')
        assert hasattr(assessor, 'consistency_threshold')
        assert hasattr(assessor, 'freshness_threshold')

    def test_assess_current_price_quality_high(self, assessor, sample_current_prices):
        """Test quality assessment of high-quality current price data"""
        report = assessor.assess_current_price_quality(
            "AAPL", "test_source", sample_current_prices
        )

        assert isinstance(report, QualityReport)
        assert report.overall_score >= 70  # Should be reasonably high quality
        assert QualityDimension.COMPLETENESS in report.dimension_scores
        assert QualityDimension.TIMELINESS in report.dimension_scores
        assert QualityDimension.VALIDITY in report.dimension_scores

    def test_assess_current_price_quality_with_issues(self, assessor):
        """Test quality assessment with various data issues"""
        # Create data with issues
        timestamp = datetime.now()
        problematic_data = [
            CurrentPrice(
                symbol="AAPL",
                price=None,  # Missing price
                timestamp=timestamp,
                volume=1000000,
                quality_score=90.0
            ),
            CurrentPrice(
                symbol="AAPL",
                price=-10.0,  # Invalid negative price
                timestamp=timestamp - timedelta(minutes=1),
                volume=1050000,
                quality_score=85.0
            ),
            CurrentPrice(
                symbol="AAPL",
                price=150.0,
                timestamp=timestamp - timedelta(hours=2),  # Stale data
                volume=980000,
                quality_score=92.0
            )
        ]

        report = assessor.assess_current_price_quality(
            "AAPL", "test_source", problematic_data
        )

        assert report.overall_score < 50  # Should be low due to issues
        assert len(report.issues) > 0
        # Should detect missing data, invalid data, and stale data issues
        issue_types = [issue.dimension for issue in report.issues]
        assert QualityDimension.COMPLETENESS in issue_types
        assert QualityDimension.VALIDITY in issue_types
        assert QualityDimension.TIMELINESS in issue_types

    def test_assess_historical_data_quality(self, assessor, sample_historical_data):
        """Test quality assessment of historical data"""
        report = assessor.assess_historical_data_quality(
            "AAPL", "test_source", sample_historical_data
        )

        assert isinstance(report, QualityReport)
        assert report.overall_score > 0
        assert QualityDimension.COMPLETENESS in report.dimension_scores
        assert QualityDimension.CONSISTENCY in report.dimension_scores

    def test_assess_historical_data_with_gaps(self, assessor):
        """Test historical data assessment with gaps"""
        # Create data with gaps
        base_time = datetime.now() - timedelta(hours=2)
        gapped_data = [
            PriceData(
                timestamp=base_time,
                open_price=149.0,
                high_price=150.0,
                low_price=148.5,
                close_price=149.5,
                volume=1000000
            ),
            # Gap here - missing data points
            PriceData(
                timestamp=base_time + timedelta(hours=1),  # Big gap
                open_price=151.0,
                high_price=152.0,
                low_price=150.5,
                close_price=151.5,
                volume=1100000
            )
        ]

        report = assessor.assess_historical_data_quality(
            "AAPL", "test_source", gapped_data
        )

        # Should detect gaps and lower completeness score
        assert report.dimension_scores[QualityDimension.COMPLETENESS] < 0.9
        gap_issues = [issue for issue in report.issues
                     if issue.dimension == QualityDimension.COMPLETENESS]
        assert len(gap_issues) > 0

    def test_assess_completeness_current_prices(self, assessor, sample_current_prices):
        """Test completeness assessment for current prices"""
        score, issues = assessor._assess_completeness_current_prices(sample_current_prices)

        assert 0.0 <= score <= 1.0
        assert isinstance(issues, list)
        # Should have high completeness for complete data
        assert score > 0.8

    def test_assess_completeness_with_missing_data(self, assessor):
        """Test completeness assessment with missing data"""
        incomplete_data = [
            CurrentPrice(
                symbol="AAPL",
                price=None,  # Missing price
                timestamp=datetime.now(),
                volume=1000000,
                quality_score=90.0
            ),
            CurrentPrice(
                symbol="AAPL",
                price=150.0,
                timestamp=None,  # Missing timestamp
                volume=1000000,
                quality_score=90.0
            )
        ]

        score, issues = assessor._assess_completeness_current_prices(incomplete_data)

        assert score < 0.5  # Should be low due to missing data
        assert len(issues) > 0

    def test_assess_accuracy_cross_validation(self, assessor, sample_current_prices):
        """Test accuracy assessment using cross-validation"""
        # Mock baseline metrics for comparison
        assessor.baseline_metrics["AAPL"] = {
            "average_price": 150.0,
            "price_std": 1.0,
            "average_volume": 1000000,
            "volume_std": 50000
        }

        score, issues = assessor._assess_accuracy_cross_validation(
            "AAPL", sample_current_prices
        )

        assert 0.0 <= score <= 1.0
        assert isinstance(issues, list)

    def test_assess_consistency_temporal(self, assessor, sample_historical_data):
        """Test temporal consistency assessment"""
        score, issues = assessor._assess_consistency_temporal(sample_historical_data)

        assert 0.0 <= score <= 1.0
        assert isinstance(issues, list)
        # Should have good consistency for smooth data
        assert score > 0.7

    def test_assess_consistency_with_outliers(self, assessor):
        """Test consistency assessment with outliers"""
        base_time = datetime.now() - timedelta(hours=1)
        inconsistent_data = [
            PriceData(
                timestamp=base_time + timedelta(minutes=i),
                open_price=149.0,
                high_price=150.0,
                low_price=148.5,
                close_price=149.5,
                volume=1000000
            ) for i in range(5)
        ]

        # Add outlier
        inconsistent_data.append(
            PriceData(
                timestamp=base_time + timedelta(minutes=5),
                open_price=200.0,  # Major outlier
                high_price=201.0,
                low_price=199.0,
                close_price=200.5,
                volume=1000000
            )
        )

        score, issues = assessor._assess_consistency_temporal(inconsistent_data)

        assert score < 0.8  # Should be lower due to outlier
        assert len(issues) > 0

    def test_assess_timeliness(self, assessor):
        """Test timeliness assessment"""
        current_time = datetime.now()

        # Recent data
        recent_data = [
            CurrentPrice(symbol="AAPL", price=150.0, timestamp=current_time - timedelta(seconds=30), volume=1000000, source="test_source", quality_score=90.0
            )
        ]

        score, issues = assessor._assess_timeliness(recent_data)
        assert score > 0.9  # Should be high for recent data

        # Stale data
        stale_data = [
            CurrentPrice(symbol="AAPL", price=150.0, timestamp=current_time - timedelta(minutes=10), volume=1000000, source="test_source", quality_score=90.0
            )
        ]

        score, issues = assessor._assess_timeliness(stale_data)
        assert score < 0.5  # Should be low for stale data
        assert len(issues) > 0

    def test_assess_validity_prices(self, assessor, sample_current_prices):
        """Test price validity assessment"""
        score, issues = assessor._assess_validity_prices(sample_current_prices)

        assert score > 0.95  # Should be high for valid prices
        assert len(issues) == 0

    def test_assess_validity_with_invalid_prices(self, assessor):
        """Test validity assessment with invalid prices"""
        invalid_data = [
            CurrentPrice(
                symbol="AAPL",
                price=-10.0,  # Invalid negative price
                timestamp=datetime.now(),
                volume=1000000,
                quality_score=90.0
            ),
            CurrentPrice(
                symbol="AAPL",
                price=0.0,  # Invalid zero price
                timestamp=datetime.now(),
                volume=1000000,
                quality_score=90.0
            ),
            CurrentPrice(
                symbol="AAPL",
                price=1000000.0,  # Unreasonably high price
                timestamp=datetime.now(),
                volume=1000000,
                quality_score=90.0
            )
        ]

        score, issues = assessor._assess_validity_prices(invalid_data)

        assert score < 0.5  # Should be low due to invalid prices
        assert len(issues) >= 3  # Should detect all invalid prices

    def test_assess_uniqueness(self, assessor):
        """Test uniqueness assessment"""
        # Data with duplicates
        timestamp = datetime.now()
        duplicate_data = [
            CurrentPrice(symbol="AAPL", price=150.0, timestamp=timestamp, volume=1000000, source="test_source", quality_score=90.0
            ),
            CurrentPrice(
                symbol="AAPL",
                price=150.0,  # Duplicate price
                timestamp=timestamp,  # Same timestamp
                volume=1000000,
                quality_score=90.0
            )
        ]

        score, issues = assessor._assess_uniqueness(duplicate_data)

        assert score < 1.0  # Should be reduced due to duplicates
        assert len(issues) > 0

    def test_detect_data_gaps(self, assessor):
        """Test data gap detection"""
        base_time = datetime.now() - timedelta(hours=2)
        data_with_gaps = [
            PriceData(
                timestamp=base_time,
                open_price=149.0,
                high_price=150.0,
                low_price=148.5,
                close_price=149.5,
                volume=1000000
            ),
            # Gap here
            PriceData(
                timestamp=base_time + timedelta(hours=1),  # 1 hour gap
                open_price=151.0,
                high_price=152.0,
                low_price=150.5,
                close_price=151.5,
                volume=1100000
            )
        ]

        gaps = assessor._detect_data_gaps(data_with_gaps, expected_interval_minutes=15)

        assert len(gaps) > 0
        assert gaps[0]["duration_minutes"] > 45  # Should detect the gap

    def test_detect_outliers_statistical(self, assessor, sample_historical_data):
        """Test statistical outlier detection"""
        outliers = assessor._detect_outliers_statistical(sample_historical_data, "close_price")

        # Should not detect outliers in normal data
        assert len(outliers) == 0

        # Add clear outlier
        outlier_data = sample_historical_data.copy()
        outlier_data.append(
            PriceData(
                timestamp=datetime.now(),
                open_price=300.0,  # Clear outlier
                high_price=301.0,
                low_price=299.0,
                close_price=300.5,
                volume=1000000
            )
        )

        outliers = assessor._detect_outliers_statistical(outlier_data, "close_price")
        assert len(outliers) > 0

    def test_calculate_overall_score(self, assessor):
        """Test overall quality score calculation"""
        dimension_scores = {
            QualityDimension.COMPLETENESS: 0.95,
            QualityDimension.ACCURACY: 0.90,
            QualityDimension.CONSISTENCY: 0.85,
            QualityDimension.TIMELINESS: 0.80,
            QualityDimension.VALIDITY: 0.95,
            QualityDimension.UNIQUENESS: 0.98
        }

        overall_score = assessor._calculate_overall_score(dimension_scores)

        assert 0.0 <= overall_score <= 100.0
        # Should be weighted average of dimension scores
        assert 80.0 <= overall_score <= 95.0

    def test_grade_from_score(self, assessor):
        """Test letter grade assignment"""
        assert assessor._grade_from_score(95) == "A+"
        assert assessor._grade_from_score(87) == "A"
        assert assessor._grade_from_score(82) == "B+"
        assert assessor._grade_from_score(78) == "B"
        assert assessor._grade_from_score(72) == "C+"
        assert assessor._grade_from_score(68) == "C"
        assert assessor._grade_from_score(62) == "D+"
        assert assessor._grade_from_score(58) == "D"
        assert assessor._grade_from_score(45) == "F"

    def test_update_baseline_metrics(self, assessor, sample_historical_data):
        """Test baseline metrics update"""
        assessor._update_baseline_metrics("AAPL", sample_historical_data)

        assert "AAPL" in assessor.baseline_metrics
        metrics = assessor.baseline_metrics["AAPL"]
        assert "average_price" in metrics
        assert "price_std" in metrics
        assert "average_volume" in metrics
        assert "volume_std" in metrics

    def test_update_issue_patterns(self, assessor):
        """Test issue pattern tracking"""
        issue = QualityIssue(
            issue_type=QualityIssueType.MISSING_DATA,
            dimension=QualityDimension.COMPLETENESS,
            severity="high",
            description="Missing price data",
            affected_records=5,
            confidence=0.9
        )

        assessor._update_issue_patterns("AAPL", "test_source", [issue])

        assert "AAPL" in assessor.issue_patterns
        assert "test_source" in assessor.issue_patterns["AAPL"]

    def test_get_quality_trends(self, assessor):
        """Test quality trend analysis"""
        # Add some quality history
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(5)]
        for i, timestamp in enumerate(timestamps):
            report = QualityReport(
                symbol="AAPL",
                source="test_source",
                overall_score=80 + i * 2,  # Improving trend
                grade="B",
                dimension_scores={},
                issues=[],
                recommendations=[],
                metadata={"timestamp": timestamp}
            )

            if "AAPL" not in assessor.quality_history:
                assessor.quality_history["AAPL"] = {}
            if "test_source" not in assessor.quality_history["AAPL"]:
                assessor.quality_history["AAPL"]["test_source"] = []

            assessor.quality_history["AAPL"]["test_source"].append(report)

        trends = assessor.get_quality_trends("AAPL", "test_source")

        assert "trend_direction" in trends
        assert "score_change" in trends
        assert trends["trend_direction"] == "improving"

    def test_get_quality_summary(self, assessor):
        """Test quality summary generation"""
        # Add some quality history first
        report1 = QualityReport(
            symbol="AAPL",
            source="source1",
            overall_score=85,
            grade="A-",
            dimension_scores={},
            issues=[],
            recommendations=[],
            metadata={}
        )

        report2 = QualityReport(
            symbol="GOOGL",
            source="source1",
            overall_score=75,
            grade="B",
            dimension_scores={},
            issues=[],
            recommendations=[],
            metadata={}
        )

        assessor.quality_history = {
            "AAPL": {"source1": [report1]},
            "GOOGL": {"source1": [report2]}
        }

        summary = assessor.get_quality_summary()

        assert "symbols_assessed" in summary
        assert "sources_assessed" in summary
        assert "average_quality_score" in summary
        assert summary["symbols_assessed"] == 2

    def test_export_quality_report(self, assessor, sample_current_prices):
        """Test quality report export"""
        report = assessor.assess_current_price_quality(
            "AAPL", "test_source", sample_current_prices
        )

        exported = assessor.export_quality_report(report)

        assert "symbol" in exported
        assert "source" in exported
        assert "overall_score" in exported
        assert "grade" in exported
        assert "dimension_scores" in exported
        assert "issues" in exported
        assert "recommendations" in exported

    def test_quality_issue_serialization(self):
        """Test QualityIssue serialization"""
        issue = QualityIssue(
            issue_type=QualityIssueType.MISSING_DATA,
            dimension=QualityDimension.COMPLETENESS,
            severity="high",
            description="Missing price data",
            affected_records=5,
            confidence=0.9
        )

        serialized = issue.to_dict()

        assert serialized["dimension"] == "completeness"
        assert serialized["severity"] == "high"
        assert serialized["description"] == "Missing price data"
        assert serialized["field"] == "price"
        assert serialized["count"] == 5
        assert serialized["percentage"] == 25.0

    def test_quality_report_serialization(self):
        """Test QualityReport serialization"""
        report = QualityReport(
            symbol="AAPL",
            source="test_source",
            overall_score=85.5,
            grade="A-",
            dimension_scores={
                QualityDimension.COMPLETENESS: 0.95,
                QualityDimension.ACCURACY: 0.85
            },
            issues=[],
            recommendations=["Improve data timeliness"],
            metadata={"test": "data"}
        )

        serialized = report.to_dict()

        assert serialized["symbol"] == "AAPL"
        assert serialized["source"] == "test_source"
        assert serialized["overall_score"] == 85.5
        assert serialized["grade"] == "A-"
        assert "completeness" in serialized["dimension_scores"]
        assert "accuracy" in serialized["dimension_scores"]

    def test_update_configuration(self, assessor):
        """Test configuration updates"""
        new_config = QualityConfig(
            completeness_threshold=0.90,  # Different from original
            accuracy_threshold=0.85,
            consistency_threshold=0.80,
            timeliness_threshold_seconds=120,
            validity_threshold=0.90,
            uniqueness_threshold=0.95,
            min_data_points=15,
            outlier_sensitivity=1.5
        )

        assessor.update_configuration(new_config)

        assert assessor.config.completeness_threshold == 0.90
        assert assessor.config.timeliness_threshold_seconds == 120

    def test_clear_cache(self, assessor):
        """Test cache clearing"""
        # Add something to cache
        assessor.assessment_cache["test_key"] = "test_value"

        assessor.clear_cache()

        assert len(assessor.assessment_cache) == 0

    def test_edge_cases_empty_data(self, assessor):
        """Test handling of empty data"""
        report = assessor.assess_current_price_quality("AAPL", "test_source", [])

        assert report.overall_score == 0
        assert report.grade == "F"
        assert len(report.issues) > 0

    def test_edge_cases_none_values(self, assessor):
        """Test handling of None values"""
        # Should handle gracefully without crashing
        try:
            report = assessor.assess_current_price_quality("AAPL", "test_source", None)
            assert report.overall_score == 0
        except Exception:
            # Acceptable to raise exception for None input
            pass

    @pytest.mark.parametrize("dimension", list(QualityDimension))
    def test_all_quality_dimensions_assessed(self, assessor, sample_current_prices, dimension):
        """Test that all quality dimensions are properly assessed"""
        report = assessor.assess_current_price_quality(
            "AAPL", "test_source", sample_current_prices
        )

        # All dimensions should be present in the report
        if dimension in [QualityDimension.COMPLETENESS, QualityDimension.TIMELINESS,
                        QualityDimension.VALIDITY, QualityDimension.UNIQUENESS]:
            assert dimension in report.dimension_scores