"""Tests for Validation Engine

Comprehensive tests for the central validation orchestration engine including
parallel validation, configuration management, and performance monitoring.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict

from src.validation.validation_engine import (
    ValidationEngine, ValidationSummary, ValidationConfig, ValidationMode
)
from src.validation.statistical_validator import ValidationResult, AnomalyType
from src.validation.cross_source_validator import ConsensusResult, SourceAgreement
from src.validation.real_time_validator import RealTimeValidationResult, ValidationStatus
from src.validation.data_quality_assessor import QualityReport, QualityDimension
from src.data_sources.base import CurrentPrice, PriceData


@pytest.fixture
def config():
    """Standard configuration for testing"""
    return ValidationConfig(
        mode=ValidationMode.STRICT,
        enable_statistical=True,
        enable_cross_source=True,
        enable_real_time=True,
        enable_quality_assessment=True,
        parallel_validation=True,
        max_validation_time_ms=100.0,
        confidence_threshold=0.7
    )


@pytest.fixture
def engine(config):
    """Validation engine instance for testing"""
    return ValidationEngine(config)


@pytest.fixture
def sample_price_data():
    """Sample current price data for testing"""
    return CurrentPrice(symbol="AAPL", price=150.0, timestamp=datetime.now(), volume=1000000, source="test_source", quality_score=90.0
    )


@pytest.fixture
def sample_additional_sources():
    """Sample additional source data for cross-validation"""
    timestamp = datetime.now()
    return {
        "source1": CurrentPrice(symbol="AAPL", price=150.0, timestamp=timestamp, volume=1000000, source="test_source", quality_score=90.0
        ),
        "source2": CurrentPrice(symbol="AAPL", price=150.05, timestamp=timestamp, volume=1050000, source="test_source", quality_score=85.0
        )
    }


@pytest.fixture
def mock_validators(engine):
    """Mock all validation components"""
    # Mock statistical validator
    engine.statistical_validator = Mock()
    engine.statistical_validator.validate_price_data.return_value = ValidationResult(
        is_valid=True,
        confidence=0.95,
        anomaly_type=None,
        severity="info",
        message="Valid price data",
        details={}
    )

    # Mock cross-source validator
    engine.cross_source_validator = Mock()
    engine.cross_source_validator.validate_current_prices.return_value = ConsensusResult(
        consensus_value=150.0,
        confidence=0.9,
        agreement_level=SourceAgreement.STRONG,
        participating_sources=["source1", "source2"],
        outlier_sources=[],
        weights_used={"source1": 0.5, "source2": 0.5}
    )

    # Mock real-time validator
    engine.real_time_validator = AsyncMock()
    engine.real_time_validator.validate_real_time_price.return_value = RealTimeValidationResult(
        status=ValidationStatus.PASS,
        data_accepted=True,
        confidence=0.95,
        processing_time_ms=5.0,
        validations_performed=["basic_constraints"],
        warnings=[],
        errors=[],
        metadata={}
    )

    # Mock quality assessor
    engine.quality_assessor = Mock()
    engine.quality_assessor.assess_current_price_quality.return_value = QualityReport(
        symbol="AAPL",
        source="test_source",
        overall_score=85.0,
        grade="A-",
        dimension_scores={QualityDimension.COMPLETENESS: 0.95},
        issues=[],
        recommendations=[],
        metadata={}
    )

    return engine


class TestValidationEngine:
    """Test suite for ValidationEngine"""

    def test_initialization(self, config):
        """Test engine initialization"""
        engine = ValidationEngine(config)

        assert engine.config == config
        assert engine.statistical_validator is not None
        assert engine.cross_source_validator is not None
        assert engine.real_time_validator is not None
        assert engine.quality_assessor is not None
        assert engine.validation_count == 0
        assert len(engine.validation_history) == 0

    def test_initialization_disabled_components(self):
        """Test engine initialization with disabled components"""
        minimal_config = ValidationConfig(
            mode=ValidationMode.PERMISSIVE,
            enable_statistical=False,
            enable_cross_source=False,
            enable_real_time=False,
            enable_quality_assessment=False,
            parallel_validation=False
        )

        engine = ValidationEngine(minimal_config)

        assert engine.statistical_validator is None
        assert engine.cross_source_validator is None
        assert engine.real_time_validator is None
        assert engine.quality_assessor is None

    @pytest.mark.asyncio
    async def test_validate_current_price_all_components(self, mock_validators, sample_price_data, sample_additional_sources):
        """Test current price validation with all components enabled"""
        result = await mock_validators.validate_current_price(
            "AAPL", "test_source", sample_price_data, sample_additional_sources
        )

        assert isinstance(result, ValidationSummary)
        assert result.is_valid
        assert result.overall_confidence > 0
        assert "statistical" in result.validation_results
        assert "cross_source" in result.validation_results
        assert "real_time" in result.validation_results
        assert "quality" in result.validation_results

    @pytest.mark.asyncio
    async def test_validate_current_price_minimal(self, engine, sample_price_data):
        """Test current price validation with minimal configuration"""
        engine.config.statistical_validation = False
        engine.config.cross_source_validation = False
        engine.config.quality_assessment = False

        result = await engine.validate_current_price("AAPL", "test_source", sample_price_data)

        assert isinstance(result, ValidationSummary)
        # Should still have real-time validation
        assert "real_time" in result.validation_results

    @pytest.mark.asyncio
    async def test_validate_current_price_parallel_execution(self, mock_validators, sample_price_data, sample_additional_sources):
        """Test parallel validation execution"""
        mock_validators.config.parallel_validation = True

        start_time = asyncio.get_event_loop().time()
        result = await mock_validators.validate_current_price(
            "AAPL", "test_source", sample_price_data, sample_additional_sources
        )
        elapsed_time = asyncio.get_event_loop().time() - start_time

        assert isinstance(result, ValidationSummary)
        # Parallel execution should be reasonably fast
        assert elapsed_time < 1.0
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_validate_current_price_sequential_execution(self, mock_validators, sample_price_data, sample_additional_sources):
        """Test sequential validation execution"""
        mock_validators.config.parallel_validation = False

        result = await mock_validators.validate_current_price(
            "AAPL", "test_source", sample_price_data, sample_additional_sources
        )

        assert isinstance(result, ValidationSummary)
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_validate_current_price_caching(self, mock_validators, sample_price_data):
        """Test validation result caching"""
        mock_validators.config.enable_caching = True

        # First validation
        result1 = await mock_validators.validate_current_price("AAPL", "test_source", sample_price_data)

        # Second validation (should use cache)
        result2 = await mock_validators.validate_current_price("AAPL", "test_source", sample_price_data)

        assert result1.is_valid == result2.is_valid
        # Cache hit should be faster
        assert result2.processing_time_ms <= result1.processing_time_ms

    @pytest.mark.asyncio
    async def test_validate_current_price_validation_failure(self, mock_validators, sample_price_data):
        """Test handling of validation failures"""
        # Make statistical validator return failure
        mock_validators.statistical_validator.validate_price_data.return_value = ValidationResult(
            is_valid=False,
            confidence=0.2,
            anomaly_type=AnomalyType.OUTLIER,
            severity="error",
            message="Price outlier detected",
            details={}
        )

        result = await mock_validators.validate_current_price("AAPL", "test_source", sample_price_data)

        assert isinstance(result, ValidationSummary)
        # Overall result depends on validation mode and component results
        assert result.overall_confidence < 1.0

    @pytest.mark.asyncio
    async def test_validate_historical_data(self, mock_validators):
        """Test historical data validation"""
        historical_data = [
            PriceData(
                timestamp=datetime.now() - timedelta(minutes=i),
                open_price=149.0 + i * 0.1,
                high_price=150.0 + i * 0.1,
                low_price=148.5 + i * 0.1,
                close_price=149.5 + i * 0.1,
                volume=1000000 + i * 10000
            ) for i in range(10)
        ]

        result = await mock_validators.validate_historical_data("AAPL", "test_source", historical_data)

        assert isinstance(result, ValidationSummary)
        assert result.data_points_validated == 10

    def test_determine_overall_confidence_strict_mode(self, engine):
        """Test overall confidence calculation in strict mode"""
        engine.config.validation_mode = ValidationMode.STRICT

        validation_results = {
            "statistical": ValidationResult(is_valid=True, confidence=0.9, anomaly_type=None, severity="info", message="", details={}),
            "cross_source": ConsensusResult(consensus_value=150.0, confidence=0.8, agreement_level=SourceAgreement.STRONG, participating_sources=[], outlier_sources=[], weights_used={}),
            "real_time": RealTimeValidationResult(status=ValidationStatus.PASS, data_accepted=True, confidence=0.95, processing_time_ms=5.0, validations_performed=[], warnings=[], errors=[], metadata={}),
        }

        confidence = engine._determine_overall_confidence(validation_results)

        # In strict mode, should use conservative approach
        assert 0.0 <= confidence <= 1.0
        assert confidence <= min(0.9, 0.8, 0.95)

    def test_determine_overall_confidence_lenient_mode(self, engine):
        """Test overall confidence calculation in lenient mode"""
        engine.config.validation_mode = ValidationMode.PERMISSIVE

        validation_results = {
            "statistical": ValidationResult(is_valid=True, confidence=0.9, anomaly_type=None, severity="info", message="", details={}),
            "real_time": RealTimeValidationResult(status=ValidationStatus.PASS, data_accepted=True, confidence=0.95, processing_time_ms=5.0, validations_performed=[], warnings=[], errors=[], metadata={}),
        }

        confidence = engine._determine_overall_confidence(validation_results)

        # In lenient mode, should be more optimistic
        assert 0.0 <= confidence <= 1.0
        assert confidence >= 0.9

    def test_is_validation_valid_strict_mode(self, engine):
        """Test validation validity check in strict mode"""
        engine.config.validation_mode = ValidationMode.STRICT

        # All components pass
        all_pass_results = {
            "statistical": ValidationResult(is_valid=True, confidence=0.9, anomaly_type=None, severity="info", message="", details={}),
            "real_time": RealTimeValidationResult(status=ValidationStatus.PASS, data_accepted=True, confidence=0.95, processing_time_ms=5.0, validations_performed=[], warnings=[], errors=[], metadata={}),
        }

        assert engine._is_validation_valid(all_pass_results, 0.9)

        # One component fails
        one_fail_results = {
            "statistical": ValidationResult(is_valid=False, confidence=0.3, anomaly_type=AnomalyType.OUTLIER, severity="error", message="", details={}),
            "real_time": RealTimeValidationResult(status=ValidationStatus.PASS, data_accepted=True, confidence=0.95, processing_time_ms=5.0, validations_performed=[], warnings=[], errors=[], metadata={}),
        }

        assert not engine._is_validation_valid(one_fail_results, 0.6)

    def test_is_validation_valid_lenient_mode(self, engine):
        """Test validation validity check in lenient mode"""
        engine.config.validation_mode = ValidationMode.PERMISSIVE

        # Some components fail but overall confidence is acceptable
        mixed_results = {
            "statistical": ValidationResult(is_valid=False, confidence=0.4, anomaly_type=AnomalyType.OUTLIER, severity="warning", message="", details={}),
            "real_time": RealTimeValidationResult(status=ValidationStatus.PASS, data_accepted=True, confidence=0.95, processing_time_ms=5.0, validations_performed=[], warnings=[], errors=[], metadata={}),
        }

        # Should pass in lenient mode with decent overall confidence
        assert engine._is_validation_valid(mixed_results, 0.75)

    def test_create_cache_key(self, engine, sample_price_data):
        """Test cache key generation"""
        key = engine._create_cache_key("AAPL", "test_source", sample_price_data)

        assert isinstance(key, str)
        assert "AAPL" in key
        assert "test_source" in key
        # Should include price and timestamp for uniqueness
        assert str(sample_price_data.price) in key

    def test_get_from_cache(self, engine):
        """Test cache retrieval"""
        # Add something to cache
        cache_key = "test_key"
        test_result = ValidationSummary(
            symbol="AAPL",
            source="test_source",
            is_valid=True,
            overall_confidence=0.9,
            validation_results={},
            processing_time_ms=5.0,
            timestamp=datetime.now()
        )

        engine.validation_cache[cache_key] = {
            "result": test_result,
            "timestamp": datetime.now()
        }

        cached_result = engine._get_from_cache(cache_key)
        assert cached_result == test_result

        # Test cache miss
        assert engine._get_from_cache("nonexistent_key") is None

    def test_cache_expiration(self, engine):
        """Test cache entry expiration"""
        cache_key = "test_key"
        old_result = ValidationSummary(
            symbol="AAPL",
            source="test_source",
            is_valid=True,
            overall_confidence=0.9,
            validation_results={},
            processing_time_ms=5.0,
            timestamp=datetime.now()
        )

        # Add expired entry
        engine.validation_cache[cache_key] = {
            "result": old_result,
            "timestamp": datetime.now() - timedelta(seconds=400)  # Older than TTL
        }

        # Should return None for expired entry
        assert engine._get_from_cache(cache_key) is None

    def test_add_to_cache(self, engine):
        """Test adding to cache"""
        cache_key = "test_key"
        test_result = ValidationSummary(
            symbol="AAPL",
            source="test_source",
            is_valid=True,
            overall_confidence=0.9,
            validation_results={},
            processing_time_ms=5.0,
            timestamp=datetime.now()
        )

        engine._add_to_cache(cache_key, test_result)

        assert cache_key in engine.validation_cache
        assert engine.validation_cache[cache_key]["result"] == test_result

    def test_cleanup_cache(self, engine):
        """Test cache cleanup"""
        # Add fresh and expired entries
        fresh_result = ValidationSummary(
            symbol="AAPL", source="test_source", is_valid=True,
            overall_confidence=0.9, validation_results={}, processing_time_ms=5.0,
            timestamp=datetime.now()
        )

        expired_result = ValidationSummary(
            symbol="GOOGL", source="test_source", is_valid=True,
            overall_confidence=0.9, validation_results={}, processing_time_ms=5.0,
            timestamp=datetime.now()
        )

        engine.validation_cache["fresh"] = {
            "result": fresh_result,
            "timestamp": datetime.now()
        }

        engine.validation_cache["expired"] = {
            "result": expired_result,
            "timestamp": datetime.now() - timedelta(seconds=400)
        }

        engine._cleanup_cache()

        assert "fresh" in engine.validation_cache
        assert "expired" not in engine.validation_cache

    def test_update_performance_metrics(self, engine):
        """Test performance metrics tracking"""
        initial_count = len(engine.performance_history)

        test_result = ValidationSummary(
            symbol="AAPL",
            source="test_source",
            is_valid=True,
            overall_confidence=0.9,
            validation_results={},
            processing_time_ms=15.0,
            timestamp=datetime.now()
        )

        engine._update_performance_metrics(test_result)

        assert len(engine.performance_history) == initial_count + 1
        assert engine.performance_history[-1] == test_result

    def test_performance_history_limit(self, engine):
        """Test performance history size limiting"""
        # Add many results to exceed limit
        for i in range(1200):  # More than max_history_size
            result = ValidationSummary(
                symbol="AAPL",
                source="test_source",
                is_valid=True,
                overall_confidence=0.9,
                validation_results={},
                processing_time_ms=5.0,
                timestamp=datetime.now()
            )
            engine.performance_history.append(result)

        engine._update_performance_metrics(result)

        assert len(engine.performance_history) <= 1000

    def test_get_performance_metrics(self, engine):
        """Test performance metrics retrieval"""
        # Add some performance history
        for i in range(10):
            result = ValidationSummary(
                symbol="AAPL",
                source="test_source",
                is_valid=i % 2 == 0,
                overall_confidence=0.8 + i * 0.01,
                validation_results={},
                processing_time_ms=5.0 + i,
                timestamp=datetime.now()
            )
            engine.performance_history.append(result)

        metrics = engine.get_performance_metrics()

        assert "total_validations" in metrics
        assert "success_rate" in metrics
        assert "average_processing_time_ms" in metrics
        assert "average_confidence" in metrics
        assert metrics["total_validations"] == 10

    def test_get_performance_metrics_no_data(self, engine):
        """Test performance metrics with no data"""
        metrics = engine.get_performance_metrics()

        assert metrics["total_validations"] == 0
        assert metrics["success_rate"] == 0.0

    def test_get_validation_summary(self, engine):
        """Test validation summary retrieval"""
        # Add some performance history
        engine.performance_history = [
            ValidationSummary(
                symbol="AAPL", source="source1", is_valid=True,
                overall_confidence=0.9, validation_results={}, processing_time_ms=5.0,
                timestamp=datetime.now()
            ),
            ValidationSummary(
                symbol="GOOGL", source="source1", is_valid=False,
                overall_confidence=0.4, validation_results={}, processing_time_ms=8.0,
                timestamp=datetime.now()
            )
        ]

        summary = engine.get_validation_summary()

        assert "total_validations" in summary
        assert "symbols_validated" in summary
        assert "sources_used" in summary
        assert "configuration" in summary

    def test_update_configuration(self, engine):
        """Test configuration updates"""
        new_config = ValidationConfig(
            statistical_validation=False,  # Different from original
            cross_source_validation=True,
            real_time_validation=True,
            quality_assessment=False,
            parallel_validation=False,
            validation_mode=ValidationMode.PERMISSIVE,
            max_processing_time_ms=200.0,
            confidence_threshold=0.6
        )

        engine.update_configuration(new_config)

        assert engine.config.statistical_validation is False
        assert engine.config.max_processing_time_ms == 200.0
        assert engine.config.validation_mode == ValidationMode.PERMISSIVE

    def test_reset_cache(self, engine):
        """Test cache reset"""
        # Add some cache entries
        engine.validation_cache["test1"] = {"result": Mock(), "timestamp": datetime.now()}
        engine.validation_cache["test2"] = {"result": Mock(), "timestamp": datetime.now()}

        engine.reset_cache()

        assert len(engine.validation_cache) == 0

    def test_reset_performance_history(self, engine):
        """Test performance history reset"""
        # Add some history
        engine.performance_history = [Mock(), Mock(), Mock()]

        engine.reset_performance_history()

        assert len(engine.performance_history) == 0

    def test_validation_summary_serialization(self):
        """Test ValidationSummary serialization"""
        summary = ValidationSummary(
            symbol="AAPL",
            source="test_source",
            is_valid=True,
            overall_confidence=0.95,
            validation_results={
                "statistical": ValidationResult(
                    is_valid=True, confidence=0.9, anomaly_type=None,
                    severity="info", message="Valid", details={}
                )
            },
            processing_time_ms=15.5,
            timestamp=datetime.now(),
            data_points_validated=1,
            warnings=["Minor warning"],
            errors=[],
            metadata={"test": "data"}
        )

        serialized = summary.to_dict()

        assert serialized["symbol"] == "AAPL"
        assert serialized["source"] == "test_source"
        assert serialized["is_valid"] is True
        assert serialized["overall_confidence"] == 0.95
        assert serialized["processing_time_ms"] == 15.5
        assert "statistical" in serialized["validation_results"]

    @pytest.mark.asyncio
    async def test_error_handling_component_failure(self, engine, sample_price_data):
        """Test handling of component failures"""
        # Make real-time validator raise exception
        engine.real_time_validator = AsyncMock()
        engine.real_time_validator.validate_real_time_price.side_effect = Exception("Validator error")

        result = await engine.validate_current_price("AAPL", "test_source", sample_price_data)

        # Should handle error gracefully and continue with other validations
        assert isinstance(result, ValidationSummary)
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_timeout_handling(self, engine, sample_price_data):
        """Test handling of validation timeouts"""
        engine.config.max_processing_time_ms = 1.0  # Very short timeout

        # Make validator slow
        async def slow_validator(*args, **kwargs):
            await asyncio.sleep(0.1)  # Longer than timeout
            return RealTimeValidationResult(
                status=ValidationStatus.PASS, data_accepted=True, confidence=0.95,
                processing_time_ms=100.0, validations_performed=[], warnings=[], errors=[], metadata={}
            )

        engine.real_time_validator.validate_real_time_price = slow_validator

        result = await engine.validate_current_price("AAPL", "test_source", sample_price_data)

        # Should complete despite timeout (may have warnings)
        assert isinstance(result, ValidationSummary)