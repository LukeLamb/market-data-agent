"""Tests for Quality Scoring Engine

Comprehensive tests for the A-F quality scoring system including
dimension scoring, grade determination, and recommendation generation.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.quality.quality_scoring_engine import (
    QualityScoringEngine, QualityScoreCard, QualityGrade,
    SeverityLevel, DimensionWeight, ScoringConfig, DetailedScore
)
from src.validation.data_quality_assessor import QualityReport, QualityDimension, QualityIssue, QualityIssueType
from src.data_sources.base import CurrentPrice


@pytest.fixture
def scoring_config():
    """Standard scoring configuration for testing"""
    return ScoringConfig()


@pytest.fixture
def scoring_engine(scoring_config):
    """Quality scoring engine instance for testing"""
    return QualityScoringEngine(scoring_config)


@pytest.fixture
def sample_current_prices():
    """Sample current price data for testing"""
    timestamp = datetime.now()
    return [
        CurrentPrice(
            symbol="AAPL",
            price=150.0,
            timestamp=timestamp,
            volume=1000000,
            source="test_source",
            quality_score=90
        ),
        CurrentPrice(
            symbol="AAPL",
            price=150.1,
            timestamp=timestamp - timedelta(minutes=1),
            volume=1050000,
            source="test_source",
            quality_score=85
        ),
        CurrentPrice(
            symbol="AAPL",
            price=149.9,
            timestamp=timestamp - timedelta(minutes=2),
            volume=980000,
            source="test_source",
            quality_score=88
        )
    ]


@pytest.fixture
def mock_quality_report():
    """Mock quality report for testing"""
    return QualityReport(
        overall_score=85.0,
        dimension_scores={
            QualityDimension.COMPLETENESS: 0.95,
            QualityDimension.ACCURACY: 0.90,
            QualityDimension.CONSISTENCY: 0.85,
            QualityDimension.TIMELINESS: 0.80,
            QualityDimension.VALIDITY: 0.95,
            QualityDimension.UNIQUENESS: 0.98
        },
        issues=[],
        recommendations=[],
        assessment_period=(datetime.now() - timedelta(hours=1), datetime.now()),
        total_records_analyzed=100
    )


class TestQualityScoringEngine:
    """Test suite for QualityScoringEngine"""

    def test_initialization_default_config(self):
        """Test engine initialization with default configuration"""
        engine = QualityScoringEngine()

        assert engine.config is not None
        assert isinstance(engine.config, ScoringConfig)
        assert engine.quality_assessor is not None
        assert engine.score_history == {}
        assert engine.trend_cache == {}

    def test_initialization_custom_config(self, scoring_config):
        """Test engine initialization with custom configuration"""
        engine = QualityScoringEngine(scoring_config)

        assert engine.config == scoring_config

    def test_dimension_weight_validation(self):
        """Test dimension weight validation"""
        # Valid weights (sum to 1.0)
        valid_weights = DimensionWeight(
            completeness=0.25,
            accuracy=0.20,
            consistency=0.15,
            timeliness=0.20,
            validity=0.15,
            uniqueness=0.05
        )
        assert valid_weights is not None

        # Invalid weights (don't sum to 1.0)
        with pytest.raises(ValueError):
            DimensionWeight(
                completeness=0.5,
                accuracy=0.5,
                consistency=0.5,
                timeliness=0.5,
                validity=0.5,
                uniqueness=0.5
            )

    def test_generate_score_card_basic(self, scoring_engine, sample_current_prices):
        """Test basic score card generation"""
        with patch.object(scoring_engine.quality_assessor, 'assess_current_prices') as mock_assess:
            mock_assess.return_value = QualityReport(
                overall_score=85.0,
                dimension_scores={QualityDimension.COMPLETENESS: 0.90},
                issues=[],
                recommendations=[],
                assessment_period=(datetime.now() - timedelta(hours=1), datetime.now()),
                total_records_analyzed=3
            )

            score_card = scoring_engine.generate_score_card("AAPL", "test_source", sample_current_prices)

            assert isinstance(score_card, QualityScoreCard)
            assert score_card.symbol == "AAPL"
            assert score_card.source == "test_source"
            assert isinstance(score_card.overall_grade, QualityGrade)
            assert 0 <= score_card.overall_score <= 100
            assert score_card.data_points_analyzed == 3

    def test_grade_determination(self, scoring_engine):
        """Test grade determination from scores"""
        # Test all grade thresholds
        assert scoring_engine._determine_grade(98.0) == QualityGrade.A_PLUS
        assert scoring_engine._determine_grade(94.0) == QualityGrade.A
        assert scoring_engine._determine_grade(91.0) == QualityGrade.A_MINUS
        assert scoring_engine._determine_grade(88.0) == QualityGrade.B_PLUS
        assert scoring_engine._determine_grade(84.0) == QualityGrade.B
        assert scoring_engine._determine_grade(81.0) == QualityGrade.B_MINUS
        assert scoring_engine._determine_grade(78.0) == QualityGrade.C_PLUS
        assert scoring_engine._determine_grade(74.0) == QualityGrade.C
        assert scoring_engine._determine_grade(71.0) == QualityGrade.C_MINUS
        assert scoring_engine._determine_grade(68.0) == QualityGrade.D_PLUS
        assert scoring_engine._determine_grade(62.0) == QualityGrade.D
        assert scoring_engine._determine_grade(30.0) == QualityGrade.F

    def test_severity_mapping(self, scoring_engine):
        """Test severity string to enum mapping"""
        assert scoring_engine._map_severity_to_enum("critical") == SeverityLevel.CRITICAL
        assert scoring_engine._map_severity_to_enum("high") == SeverityLevel.HIGH
        assert scoring_engine._map_severity_to_enum("medium") == SeverityLevel.MEDIUM
        assert scoring_engine._map_severity_to_enum("low") == SeverityLevel.LOW
        assert scoring_engine._map_severity_to_enum("info") == SeverityLevel.INFO
        assert scoring_engine._map_severity_to_enum("unknown") == SeverityLevel.MEDIUM  # Default

    def test_dimension_penalties(self, scoring_engine):
        """Test dimension penalty calculation"""
        issues = [
            QualityIssue(
                issue_type=QualityIssueType.MISSING_DATA,
                dimension=QualityDimension.COMPLETENESS,
                severity="high",
                description="Missing price data",
                affected_records=10,
                confidence=0.9
            ),
            QualityIssue(
                issue_type=QualityIssueType.STALE_DATA,
                dimension=QualityDimension.TIMELINESS,
                severity="medium",
                description="Data is stale",
                affected_records=5,
                confidence=0.8
            )
        ]

        # Test completeness penalties
        completeness_penalties = scoring_engine._calculate_dimension_penalties(
            QualityDimension.COMPLETENESS, issues
        )
        assert len(completeness_penalties) == 1
        assert completeness_penalties[0][0] == "Missing price data"
        assert completeness_penalties[0][1] > 0

        # Test timeliness penalties
        timeliness_penalties = scoring_engine._calculate_dimension_penalties(
            QualityDimension.TIMELINESS, issues
        )
        assert len(timeliness_penalties) == 1
        assert timeliness_penalties[0][0] == "Data is stale"

    def test_dimension_bonuses(self, scoring_engine, sample_current_prices):
        """Test dimension bonus calculation"""
        # Test perfect completeness bonus
        completeness_bonuses = scoring_engine._calculate_dimension_bonuses(
            QualityDimension.COMPLETENESS, 99.8, sample_current_prices
        )
        assert any("perfect_completeness" in bonus[0] for bonus in completeness_bonuses)

        # Test exceptional timeliness bonus (very fresh data)
        fresh_prices = [
            CurrentPrice(
                symbol="AAPL",
                price=150.0,
                timestamp=datetime.now() - timedelta(seconds=10),
                volume=1000000,
                source="test_source",
                quality_score=90
            )
        ]
        timeliness_bonuses = scoring_engine._calculate_dimension_bonuses(
            QualityDimension.TIMELINESS, 85.0, fresh_prices
        )
        assert any("exceptional_timeliness" in bonus[0] for bonus in timeliness_bonuses)

    def test_time_decay_penalty(self, scoring_engine):
        """Test time-based decay penalty calculation"""
        # Fresh data (no penalty)
        fresh_prices = [
            CurrentPrice(
                symbol="AAPL",
                price=150.0,
                timestamp=datetime.now() - timedelta(hours=1),
                volume=1000000,
                source="test_source",
                quality_score=90
            )
        ]
        penalty = scoring_engine._calculate_time_decay_penalty(fresh_prices)
        assert penalty == 0.0

        # Old data (penalty expected)
        old_prices = [
            CurrentPrice(
                symbol="AAPL",
                price=150.0,
                timestamp=datetime.now() - timedelta(hours=30),
                volume=1000000,
                source="test_source",
                quality_score=90
            )
        ]
        penalty = scoring_engine._calculate_time_decay_penalty(old_prices)
        assert penalty > 0.0

    def test_issue_penalty_calculation(self, scoring_engine):
        """Test individual issue penalty calculation"""
        issue = QualityIssue(
            issue_type=QualityIssueType.MISSING_DATA,
            dimension=QualityDimension.COMPLETENESS,
            severity="high",
            description="Missing data",
            affected_records=15,
            confidence=0.95
        )

        penalty = scoring_engine._calculate_issue_penalty(issue, 10.0, 20.0)

        assert 10.0 <= penalty <= 20.0
        assert penalty > 10.0  # Should be scaled up due to high confidence and affected records

    def test_overall_score_calculation(self, scoring_engine):
        """Test weighted overall score calculation"""
        dimension_scores = {
            QualityDimension.COMPLETENESS: DetailedScore(
                base_score=90.0,
                penalties=[],
                bonuses=[],
                final_score=90.0,
                weight=0.25,
                weighted_contribution=22.5
            ),
            QualityDimension.ACCURACY: DetailedScore(
                base_score=85.0,
                penalties=[],
                bonuses=[],
                final_score=85.0,
                weight=0.20,
                weighted_contribution=17.0
            )
        }

        overall_score = scoring_engine._calculate_overall_score(dimension_scores)
        assert overall_score == 39.5  # 22.5 + 17.0

    def test_confidence_level_calculation(self, scoring_engine, sample_current_prices):
        """Test confidence level calculation"""
        mock_quality_report = QualityReport(
            overall_score=85.0,
            dimension_scores={},
            issues=[],
            recommendations=[],
            assessment_period=(datetime.now() - timedelta(hours=1), datetime.now()),
            total_records_analyzed=3
        )

        dimension_scores = {
            QualityDimension.COMPLETENESS: DetailedScore(
                base_score=90.0, penalties=[], bonuses=[], final_score=90.0,
                weight=0.25, weighted_contribution=22.5
            )
        }

        confidence = scoring_engine._calculate_confidence_level(
            sample_current_prices, mock_quality_report, dimension_scores
        )

        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should have decent confidence with good data

    def test_recommendation_generation(self, scoring_engine):
        """Test recommendation generation"""
        dimension_scores = {
            QualityDimension.COMPLETENESS: DetailedScore(
                base_score=60.0, penalties=[("missing_data", 10.0)], bonuses=[],
                final_score=50.0, weight=0.25, weighted_contribution=12.5
            ),
            QualityDimension.ACCURACY: DetailedScore(
                base_score=90.0, penalties=[], bonuses=[],
                final_score=90.0, weight=0.20, weighted_contribution=18.0
            )
        }

        issues = [
            QualityIssue(
                issue_type=QualityIssueType.MISSING_DATA,
                dimension=QualityDimension.COMPLETENESS,
                severity="critical",
                description="Critical data missing",
                affected_records=50,
                confidence=0.95
            )
        ]

        priority_recs, improvement_suggestions = scoring_engine._generate_recommendations(
            dimension_scores, issues
        )

        assert len(priority_recs) > 0
        assert "URGENT" in priority_recs[0]  # Critical issue should be urgent
        assert len(improvement_suggestions) > 0
        assert any("completeness" in rec.lower() for rec in improvement_suggestions)

    def test_pattern_based_recommendations(self, scoring_engine):
        """Test pattern-based recommendation generation"""
        # Low scores across all dimensions
        low_dimension_scores = {
            dim: DetailedScore(
                base_score=50.0, penalties=[], bonuses=[], final_score=50.0,
                weight=0.1, weighted_contribution=5.0
            )
            for dim in QualityDimension
        }

        suggestions = []
        scoring_engine._add_pattern_based_recommendations(low_dimension_scores, suggestions)

        assert any("comprehensive" in suggestion.lower() for suggestion in suggestions)

    def test_score_history_management(self, scoring_engine, sample_current_prices):
        """Test score history tracking and management"""
        with patch.object(scoring_engine.quality_assessor, 'assess_current_prices') as mock_assess:
            mock_assess.return_value = QualityReport(
                overall_score=85.0,
                dimension_scores={QualityDimension.COMPLETENESS: 0.90},
                issues=[],
                recommendations=[],
                assessment_period=(datetime.now() - timedelta(hours=1), datetime.now()),
                total_records_analyzed=3
            )

            # Generate multiple score cards
            for i in range(5):
                score_card = scoring_engine.generate_score_card("AAPL", "test_source", sample_current_prices)

            # Check history was stored
            assert "AAPL" in scoring_engine.score_history
            assert "test_source" in scoring_engine.score_history["AAPL"]
            assert len(scoring_engine.score_history["AAPL"]["test_source"]) == 5

    def test_quality_trends_no_data(self, scoring_engine):
        """Test quality trends with no historical data"""
        trends = scoring_engine.get_quality_trends("NONEXISTENT", "test_source")

        assert "error" in trends
        assert "No historical data" in trends["error"]

    def test_quality_trends_with_data(self, scoring_engine, sample_current_prices):
        """Test quality trends calculation with historical data"""
        with patch.object(scoring_engine.quality_assessor, 'assess_current_prices') as mock_assess:
            mock_assess.return_value = QualityReport(
                overall_score=85.0,
                dimension_scores={QualityDimension.COMPLETENESS: 0.90},
                issues=[],
                recommendations=[],
                assessment_period=(datetime.now() - timedelta(hours=1), datetime.now()),
                total_records_analyzed=3
            )

            # Generate score cards with different scores
            for score in [80.0, 85.0, 90.0]:
                mock_assess.return_value.overall_score = score
                scoring_engine.generate_score_card("AAPL", "test_source", sample_current_prices)

            trends = scoring_engine.get_quality_trends("AAPL", "test_source")

            assert "symbol" in trends
            assert trends["symbol"] == "AAPL"
            assert trends["source"] == "test_source"
            assert "trend_direction" in trends
            assert trends["trend_direction"] in ["improving", "declining", "stable"]

    def test_scoring_summary(self, scoring_engine):
        """Test scoring summary generation"""
        summary = scoring_engine.get_scoring_summary()

        assert "total_assessments" in summary
        assert "average_processing_time_ms" in summary
        assert "grade_distribution" in summary
        assert "symbols_tracked" in summary
        assert "configuration" in summary

    def test_configuration_update(self, scoring_engine):
        """Test configuration updates"""
        new_config = ScoringConfig(
            enable_time_decay=False,
            max_data_age_hours=48.0
        )

        scoring_engine.update_configuration(new_config)

        assert scoring_engine.config.enable_time_decay is False
        assert scoring_engine.config.max_data_age_hours == 48.0
        assert len(scoring_engine.trend_cache) == 0  # Should be cleared

    def test_history_reset(self, scoring_engine, sample_current_prices):
        """Test history reset functionality"""
        with patch.object(scoring_engine.quality_assessor, 'assess_current_prices') as mock_assess:
            mock_assess.return_value = QualityReport(
                overall_score=85.0,
                dimension_scores={QualityDimension.COMPLETENESS: 0.90},
                issues=[],
                recommendations=[],
                assessment_period=(datetime.now() - timedelta(hours=1), datetime.now()),
                total_records_analyzed=3
            )

            # Add some history
            scoring_engine.generate_score_card("AAPL", "test_source", sample_current_prices)
            scoring_engine.generate_score_card("GOOGL", "test_source", sample_current_prices)

            # Reset specific symbol/source
            scoring_engine.reset_history("AAPL", "test_source")
            assert len(scoring_engine.score_history["AAPL"]["test_source"]) == 0

            # Reset all
            scoring_engine.reset_history()
            assert len(scoring_engine.score_history) == 0

    def test_error_score_card_creation(self, scoring_engine):
        """Test error score card creation"""
        error_card = scoring_engine._create_error_score_card("AAPL", "test_source", "Test error")

        assert error_card.symbol == "AAPL"
        assert error_card.source == "test_source"
        assert error_card.overall_grade == QualityGrade.F
        assert error_card.overall_score == 0.0
        assert error_card.critical_issues == 1
        assert "Test error" in error_card.priority_recommendations[0]

    def test_detailed_score_serialization(self):
        """Test DetailedScore serialization"""
        score = DetailedScore(
            base_score=85.0,
            penalties=[("test_penalty", 5.0)],
            bonuses=[("test_bonus", 2.0)],
            final_score=82.0,
            weight=0.25,
            weighted_contribution=20.5
        )

        serialized = score.to_dict()

        assert serialized["base_score"] == 85.0
        assert serialized["penalties"] == [("test_penalty", 5.0)]
        assert serialized["bonuses"] == [("test_bonus", 2.0)]
        assert serialized["final_score"] == 82.0
        assert serialized["weight"] == 0.25
        assert serialized["weighted_contribution"] == 20.5

    def test_quality_score_card_serialization(self, scoring_engine):
        """Test QualityScoreCard serialization"""
        score_card = QualityScoreCard(
            symbol="AAPL",
            source="test_source",
            overall_grade=QualityGrade.A,
            overall_score=90.0,
            dimension_scores={},
            total_issues=2,
            critical_issues=0,
            high_issues=1,
            medium_issues=1,
            low_issues=0,
            priority_recommendations=["Fix issue A"],
            improvement_suggestions=["Improve area B"],
            assessment_period=(datetime.now() - timedelta(hours=1), datetime.now()),
            data_points_analyzed=10,
            confidence_level=0.85
        )

        serialized = score_card.to_dict()

        assert serialized["symbol"] == "AAPL"
        assert serialized["source"] == "test_source"
        assert serialized["overall_grade"] == "A"
        assert serialized["overall_score"] == 90.0
        assert serialized["total_issues"] == 2
        assert serialized["confidence_level"] == 0.85

    def test_edge_cases_empty_data(self, scoring_engine):
        """Test handling of empty or invalid data"""
        # Empty price list
        with patch.object(scoring_engine.quality_assessor, 'assess_current_prices') as mock_assess:
            mock_assess.return_value = QualityReport(
                overall_score=0.0,
                dimension_scores={},
                issues=[],
                recommendations=[],
                assessment_period=(datetime.now(), datetime.now()),
                total_records_analyzed=0
            )

            score_card = scoring_engine.generate_score_card("AAPL", "test_source", [])

            assert score_card.overall_grade == QualityGrade.F
            assert score_card.data_points_analyzed == 0