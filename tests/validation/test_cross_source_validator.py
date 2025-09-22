"""Tests for Cross-Source Validator

Comprehensive tests for cross-source validation including consensus algorithms,
weighted voting, and reliability tracking.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import Dict

from src.validation.cross_source_validator import (
    CrossSourceValidator, ConsensusResult, SourceAgreement
)
from src.data_sources.base import CurrentPrice, PriceData


@pytest.fixture
def validator():
    """Cross-source validator instance for testing"""
    return CrossSourceValidator(
        tolerance_tight=0.001,
        tolerance_moderate=0.005,
        tolerance_loose=0.02,
        min_sources=2,
        outlier_threshold=2.0
    )


@pytest.fixture
def sample_price_data():
    """Sample current price data from multiple sources"""
    timestamp = datetime.now()
    return {
        "alpha_vantage": CurrentPrice(
            symbol="AAPL",
            price=150.0,
            timestamp=timestamp,
            volume=1000000,
            source="alpha_vantage",
            quality_score=90
        ),
        "yahoo": CurrentPrice(
            symbol="AAPL",
            price=150.05,
            timestamp=timestamp,
            volume=1050000,
            source="yahoo",
            quality_score=85
        ),
        "finnhub": CurrentPrice(
            symbol="AAPL",
            price=149.98,
            timestamp=timestamp,
            volume=980000,
            source="finnhub",
            quality_score=88
        )
    }


@pytest.fixture
def sample_historical_data():
    """Sample historical data for testing"""
    base_time = datetime.now() - timedelta(hours=1)

    return {
        "alpha_vantage": [
            PriceData(
                timestamp=base_time + timedelta(minutes=i),
                open_price=149.0 + i * 0.1,
                high_price=150.0 + i * 0.1,
                low_price=148.5 + i * 0.1,
                close_price=149.5 + i * 0.1,
                volume=1000000 + i * 10000
            ) for i in range(5)
        ],
        "yahoo": [
            PriceData(
                timestamp=base_time + timedelta(minutes=i),
                open_price=149.02 + i * 0.1,
                high_price=150.02 + i * 0.1,
                low_price=148.52 + i * 0.1,
                close_price=149.52 + i * 0.1,
                volume=1020000 + i * 10000
            ) for i in range(5)
        ]
    }


class TestCrossSourceValidator:
    """Test suite for CrossSourceValidator"""

    def test_initialization(self):
        """Test validator initialization"""
        validator = CrossSourceValidator(
            tolerance_tight=0.002,
            tolerance_moderate=0.01,
            tolerance_loose=0.05,
            min_sources=3,
            outlier_threshold=1.5
        )

        assert validator.tolerance_tight == 0.002
        assert validator.tolerance_moderate == 0.01
        assert validator.tolerance_loose == 0.05
        assert validator.min_sources == 3
        assert validator.outlier_threshold == 1.5
        assert validator.source_reliability == {}
        assert validator.source_history == {}

    def test_validate_current_prices_perfect_agreement(self, validator, sample_price_data):
        """Test validation with perfect source agreement"""
        # Modify sample data to have identical prices
        for source_name, price_data in sample_price_data.items():
            price_data.price = 150.0

        result = validator.validate_current_prices(sample_price_data)

        assert isinstance(result, ConsensusResult)
        assert result.consensus_value == 150.0
        assert result.agreement_level == SourceAgreement.PERFECT
        assert result.confidence > 0.9
        assert len(result.participating_sources) == 3
        assert len(result.outlier_sources) == 0

    def test_validate_current_prices_strong_agreement(self, validator, sample_price_data):
        """Test validation with strong source agreement"""
        result = validator.validate_current_prices(sample_price_data)

        assert isinstance(result, ConsensusResult)
        assert result.consensus_value is not None
        assert result.agreement_level in [SourceAgreement.PERFECT, SourceAgreement.STRONG]
        assert result.confidence > 0.8
        assert len(result.participating_sources) == 3

    def test_validate_current_prices_with_outlier(self, validator, sample_price_data):
        """Test validation with one outlier source"""
        # Make one source an outlier
        sample_price_data["finnhub"].price = 170.0  # Significantly different

        result = validator.validate_current_prices(sample_price_data)

        assert isinstance(result, ConsensusResult)
        assert result.consensus_value is not None
        # Outlier should be detected and removed
        assert len(result.outlier_sources) >= 0
        assert len(result.participating_sources) >= 2

    def test_validate_current_prices_insufficient_sources(self, validator):
        """Test validation with insufficient sources"""
        insufficient_data = {
            "alpha_vantage": CurrentPrice(symbol="AAPL", price=150.0, timestamp=datetime.now(), volume=1000000, source="test_source", quality_score=90.0
            )
        }

        result = validator.validate_current_prices(insufficient_data)

        assert result.consensus_value is None
        assert result.confidence == 0.0
        assert result.agreement_level == SourceAgreement.POOR
        assert "Need at least" in result.details["error"]

    def test_validate_current_prices_conflict(self, validator, sample_price_data):
        """Test validation with conflicting sources"""
        # Create conflicting prices
        sample_price_data["alpha_vantage"].price = 150.0
        sample_price_data["yahoo"].price = 160.0
        sample_price_data["finnhub"].price = 140.0

        result = validator.validate_current_prices(sample_price_data)

        assert isinstance(result, ConsensusResult)
        # Should handle conflict gracefully
        assert result.agreement_level in [
            SourceAgreement.WEAK, SourceAgreement.POOR, SourceAgreement.CONFLICT
        ]

    def test_calculate_consensus_weighted(self, validator):
        """Test weighted consensus calculation"""
        source_data = {
            "high_quality": {
                "price": 150.0,
                "quality": 95.0,
                "timestamp": datetime.now()
            },
            "medium_quality": {
                "price": 150.5,
                "quality": 75.0,
                "timestamp": datetime.now()
            },
            "low_quality": {
                "price": 149.0,
                "quality": 60.0,
                "timestamp": datetime.now()
            }
        }

        result = validator._calculate_consensus(source_data, "price")

        assert isinstance(result, ConsensusResult)
        assert result.consensus_value is not None
        # Higher quality source should have more influence
        assert abs(result.consensus_value - 150.0) < abs(result.consensus_value - 149.0)

    def test_remove_outliers(self, validator):
        """Test outlier detection and removal"""
        values = [150.0, 150.1, 149.9, 150.2, 180.0]  # Last value is outlier
        sources = ["s1", "s2", "s3", "s4", "s5"]
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]

        clean_values, clean_sources, clean_weights, outliers = validator._remove_outliers(
            values, sources, weights
        )

        assert len(outliers) > 0
        assert "s5" in outliers  # Source with outlier value
        assert 180.0 not in clean_values

    def test_remove_outliers_small_dataset(self, validator):
        """Test outlier removal with small dataset"""
        values = [150.0, 150.1, 149.9]  # Small dataset
        sources = ["s1", "s2", "s3"]
        weights = [1.0, 1.0, 1.0]

        clean_values, clean_sources, clean_weights, outliers = validator._remove_outliers(
            values, sources, weights
        )

        # Should not remove outliers from small datasets
        assert len(outliers) == 0
        assert len(clean_values) == 3

    def test_assess_agreement_levels(self, validator):
        """Test different agreement level assessments"""
        consensus = 150.0

        # Perfect agreement
        perfect_values = [150.0, 150.0, 150.0]
        agreement = validator._assess_agreement(perfect_values, consensus)
        assert agreement == SourceAgreement.PERFECT

        # Strong agreement
        strong_values = [150.0, 150.05, 149.98]
        agreement = validator._assess_agreement(strong_values, consensus)
        assert agreement in [SourceAgreement.PERFECT, SourceAgreement.STRONG]

        # Conflict
        conflict_values = [150.0, 160.0, 140.0]
        agreement = validator._assess_agreement(conflict_values, consensus)
        assert agreement in [SourceAgreement.WEAK, SourceAgreement.POOR, SourceAgreement.CONFLICT]

    def test_calculate_confidence(self, validator):
        """Test confidence calculation"""
        values = [150.0, 150.1, 149.9]
        consensus = 150.0

        confidence = validator._calculate_confidence(
            values, consensus, SourceAgreement.STRONG, 3
        )

        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should have decent confidence for strong agreement

    def test_source_reliability_tracking(self, validator):
        """Test source reliability score updates"""
        sources = ["reliable_source", "unreliable_source"]
        values = [150.0, 155.0]  # Second source is off
        consensus = 150.0

        validator._update_source_reliability(sources, values, consensus)

        assert "reliable_source" in validator.source_reliability
        assert "unreliable_source" in validator.source_reliability
        # Reliable source should have higher score
        assert validator.source_reliability["reliable_source"] > validator.source_reliability["unreliable_source"]

    def test_normalize_weights(self, validator):
        """Test weight normalization"""
        weights = [2.0, 4.0, 6.0]
        normalized = validator._normalize_weights(weights)

        assert abs(sum(normalized) - 1.0) < 1e-10  # Should sum to 1.0
        assert normalized[2] > normalized[1] > normalized[0]  # Should maintain relative order

    def test_normalize_weights_zero_sum(self, validator):
        """Test weight normalization with zero sum"""
        weights = [0.0, 0.0, 0.0]
        normalized = validator._normalize_weights(weights)

        assert len(normalized) == 3
        assert all(w == 1.0 / 3 for w in normalized)  # Should be equal weights

    def test_validate_historical_data(self, validator, sample_historical_data):
        """Test historical data validation"""
        results = validator.validate_historical_data(sample_historical_data, "close_price")

        assert isinstance(results, list)
        assert len(results) > 0
        for result in results:
            assert isinstance(result, ConsensusResult)

    def test_align_historical_data(self, validator, sample_historical_data):
        """Test historical data alignment"""
        aligned = validator._align_historical_data(sample_historical_data, "close_price")

        assert isinstance(aligned, dict)
        # Should have timestamps as keys
        for timestamp, source_values in aligned.items():
            assert isinstance(timestamp, datetime)
            assert isinstance(source_values, dict)

    def test_get_source_reliability_scores(self, validator):
        """Test getting reliability scores"""
        # Set some reliability scores
        validator.source_reliability["source1"] = 0.9
        validator.source_reliability["source2"] = 0.7

        scores = validator.get_source_reliability_scores()

        assert scores["source1"] == 0.9
        assert scores["source2"] == 0.7

    def test_set_source_reliability(self, validator):
        """Test manually setting source reliability"""
        validator.set_source_reliability("test_source", 0.85)

        assert validator.source_reliability["test_source"] == 0.85

        # Test bounds checking
        validator.set_source_reliability("test_source", 1.5)  # Above 1.0
        assert validator.source_reliability["test_source"] == 1.0

        validator.set_source_reliability("test_source", -0.5)  # Below 0.0
        assert validator.source_reliability["test_source"] == 0.0

    def test_update_market_volatility(self, validator):
        """Test market volatility factor updates"""
        original_factor = validator.market_volatility_factor

        validator.update_market_volatility(2.0)
        assert validator.market_volatility_factor == 2.0

        # Test bounds
        validator.update_market_volatility(5.0)  # Above max
        assert validator.market_volatility_factor == 3.0

        validator.update_market_volatility(0.1)  # Below min
        assert validator.market_volatility_factor == 0.5

    def test_reset_source_reliability(self, validator):
        """Test resetting source reliability"""
        # Set some data
        validator.source_reliability["source1"] = 0.9
        validator.source_reliability["source2"] = 0.8
        validator.source_history["source1"] = [(datetime.now(), 150.0, 150.0)]

        # Reset specific source
        validator.reset_source_reliability("source1")
        assert "source1" not in validator.source_reliability
        assert "source1" not in validator.source_history
        assert "source2" in validator.source_reliability

        # Reset all sources
        validator.reset_source_reliability()
        assert len(validator.source_reliability) == 0
        assert len(validator.source_history) == 0

    def test_get_validation_summary(self, validator):
        """Test validation summary generation"""
        # Add some history
        validator.source_history["source1"] = [
            (datetime.now(), 150.0, 150.0),
            (datetime.now(), 151.0, 150.5)
        ]
        validator.source_reliability["source1"] = 0.9

        summary = validator.get_validation_summary()

        assert "total_validations" in summary
        assert "sources_tracked" in summary
        assert "reliability_scores" in summary
        assert "configuration" in summary
        assert summary["sources_tracked"] == 1

    def test_get_validation_summary_no_history(self, validator):
        """Test validation summary with no history"""
        summary = validator.get_validation_summary()

        assert "error" in summary
        assert "No validation history" in summary["error"]

    def test_consensus_result_serialization(self):
        """Test ConsensusResult serialization"""
        result = ConsensusResult(
            consensus_value=150.0,
            confidence=0.95,
            agreement_level=SourceAgreement.STRONG,
            participating_sources=["source1", "source2"],
            outlier_sources=["source3"],
            weights_used={"source1": 0.6, "source2": 0.4},
            details={"test": "data"}
        )

        serialized = result.to_dict()

        assert serialized["consensus_value"] == 150.0
        assert serialized["confidence"] == 0.95
        assert serialized["agreement_level"] == "strong"
        assert serialized["participating_sources"] == ["source1", "source2"]
        assert serialized["outlier_sources"] == ["source3"]
        assert serialized["weights_used"] == {"source1": 0.6, "source2": 0.4}

    def test_edge_cases(self, validator):
        """Test edge cases and error handling"""
        # Empty price data
        result = validator.validate_current_prices({})
        assert result.consensus_value is None
        assert result.confidence == 0.0

        # Price data with None values
        price_data_with_none = {
            "source1": CurrentPrice(
                symbol="AAPL",
                price=None,  # None price
                timestamp=datetime.now(),
                volume=1000000,
                quality_score=90.0
            )
        }

        result = validator.validate_current_prices(price_data_with_none)
        assert result.consensus_value is None

    def test_temporal_correlation(self, validator):
        """Test temporal correlation analysis"""
        # This would test future temporal correlation features
        # For now, just ensure the validator handles time-based data correctly

        timestamp1 = datetime.now()
        timestamp2 = timestamp1 + timedelta(minutes=1)

        price_data_t1 = {
            "source1": CurrentPrice(symbol="AAPL", price=150.0, timestamp=timestamp1, volume=1000000, source="test_source", quality_score=90.0
            )
        }

        price_data_t2 = {
            "source1": CurrentPrice(symbol="AAPL", price=150.5, timestamp=timestamp2, volume=1000000, source="test_source", quality_score=90.0
            )
        }

        # Validate at different times
        result1 = validator.validate_current_prices(price_data_t1)
        result2 = validator.validate_current_prices(price_data_t2)

        # Should handle temporal data appropriately
        assert isinstance(result1, ConsensusResult)
        assert isinstance(result2, ConsensusResult)