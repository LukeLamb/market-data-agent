"""Tests for Real-Time Validator

Comprehensive tests for real-time validation including performance constraints,
emergency modes, and circuit breaker functionality.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.validation.real_time_validator import (
    RealTimeValidator, RealTimeValidationResult, ValidationStatus
)
from src.validation.statistical_validator import ValidationResult, AnomalyType
from src.validation.cross_source_validator import ConsensusResult, SourceAgreement
from src.data_sources.base import CurrentPrice


@pytest.fixture
def validator():
    """Real-time validator instance for testing"""
    return RealTimeValidator(
        max_processing_time_ms=5.0,
        enable_statistical_validation=True,
        enable_cross_source_validation=True,
        quarantine_threshold=0.3,
        emergency_mode=False
    )


@pytest.fixture
def sample_price_data():
    """Sample current price data for testing"""
    return CurrentPrice(symbol="AAPL", price=150.0, timestamp=datetime.now(), volume=1000000, source="test_source", quality_score=90.0
    )


@pytest.fixture
def mock_statistical_validator():
    """Mock statistical validator"""
    mock = Mock()
    mock.validate_price_data.return_value = ValidationResult(
        is_valid=True,
        confidence=0.95,
        anomaly_type=None,
        severity="info",
        message="Valid price data",
        details={}
    )
    return mock


@pytest.fixture
def mock_cross_source_validator():
    """Mock cross-source validator"""
    mock = Mock()
    mock.validate_current_prices.return_value = ConsensusResult(
        consensus_value=150.0,
        confidence=0.9,
        agreement_level=SourceAgreement.STRONG,
        participating_sources=["source1", "source2"],
        outlier_sources=[],
        weights_used={"source1": 0.5, "source2": 0.5}
    )
    return mock


class TestRealTimeValidator:
    """Test suite for RealTimeValidator"""

    def test_initialization(self):
        """Test validator initialization"""
        validator = RealTimeValidator(
            max_processing_time_ms=10.0,
            enable_statistical_validation=False,
            enable_cross_source_validation=False,
            quarantine_threshold=0.5,
            emergency_mode=True
        )

        assert validator.max_processing_time_ms == 10.0
        assert not validator.enable_statistical_validation
        assert not validator.enable_cross_source_validation
        assert validator.quarantine_threshold == 0.5
        assert validator.emergency_mode
        assert validator.statistical_validator is None
        assert validator.cross_source_validator is None

    @pytest.mark.asyncio
    async def test_validate_real_time_price_normal(self, validator, sample_price_data):
        """Test normal real-time price validation"""
        result = await validator.validate_real_time_price("AAPL", "test_source", sample_price_data)

        assert isinstance(result, RealTimeValidationResult)
        assert result.status in [ValidationStatus.PASS, ValidationStatus.PASS_WITH_WARNINGS]
        assert result.data_accepted
        assert result.processing_time_ms >= 0
        assert "basic_constraints" in result.validations_performed

    @pytest.mark.asyncio
    async def test_validate_real_time_price_emergency_mode(self, validator, sample_price_data):
        """Test validation in emergency mode"""
        validator.set_emergency_mode(True)

        result = await validator.validate_real_time_price("AAPL", "test_source", sample_price_data)

        assert result.status == ValidationStatus.BYPASSED
        assert result.data_accepted
        assert result.confidence == 1.0
        assert "emergency_bypass" in result.validations_performed
        assert result.metadata["emergency_mode"]

    @pytest.mark.asyncio
    async def test_validate_real_time_price_invalid_basic_constraints(self, validator):
        """Test validation with invalid basic constraints"""
        invalid_price_data = CurrentPrice(
            symbol="AAPL",
            price=-10.0,  # Invalid negative price
            timestamp=datetime.now(),
            volume=1000000,
            quality_score=90.0
        )

        result = await validator.validate_real_time_price("AAPL", "test_source", invalid_price_data)

        assert result.status == ValidationStatus.FAIL
        assert not result.data_accepted
        assert result.confidence == 0.0
        assert len(result.errors) > 0
        assert "Invalid price" in result.errors[0]

    @pytest.mark.asyncio
    async def test_validate_real_time_price_stale_data(self, validator):
        """Test validation with stale data"""
        stale_price_data = CurrentPrice(
            symbol="AAPL",
            price=150.0,
            timestamp=datetime.now() - timedelta(hours=2),  # Very old
            volume=1000000,
            quality_score=90.0
        )

        result = await validator.validate_real_time_price("AAPL", "test_source", stale_price_data)

        assert result.status == ValidationStatus.FAIL
        assert not result.data_accepted
        assert "too old" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_validate_real_time_price_unreasonable_price(self, validator):
        """Test validation with unreasonably high price"""
        unreasonable_price_data = CurrentPrice(
            symbol="AAPL",
            price=150000.0,  # Unreasonably high
            timestamp=datetime.now(),
            volume=1000000,
            quality_score=90.0
        )

        result = await validator.validate_real_time_price("AAPL", "test_source", unreasonable_price_data)

        assert result.status == ValidationStatus.FAIL
        assert not result.data_accepted
        assert "unreasonably high" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_statistical_validation_integration(self, validator, sample_price_data):
        """Test integration with statistical validator"""
        # Mock statistical validator to return invalid result
        mock_validator = Mock()
        mock_validator.validate_price_data.return_value = ValidationResult(
            is_valid=False,
            confidence=0.3,
            anomaly_type=AnomalyType.OUTLIER,
            severity="error",
            message="Statistical anomaly detected",
            details={}
        )
        validator.statistical_validator = mock_validator

        result = await validator.validate_real_time_price("AAPL", "test_source", sample_price_data)

        assert "statistical_analysis" in result.validations_performed
        assert len(result.errors) > 0
        assert result.confidence < 1.0

    @pytest.mark.asyncio
    async def test_cross_source_validation_integration(self, validator, sample_price_data):
        """Test integration with cross-source validator"""
        # Mock cross-source validator
        mock_validator = Mock()
        mock_validator.validate_current_prices.return_value = ConsensusResult(
            consensus_value=149.0,  # Different from input price
            confidence=0.6,
            agreement_level=SourceAgreement.WEAK,
            participating_sources=["source1", "source2"],
            outlier_sources=[],
            weights_used={"source1": 0.5, "source2": 0.5}
        )
        validator.cross_source_validator = mock_validator

        # Add some buffered data to trigger cross-source validation
        validator._buffer_price_data("AAPL", "source1", sample_price_data)
        validator._buffer_price_data("AAPL", "source2", sample_price_data)

        result = await validator.validate_real_time_price("AAPL", "test_source", sample_price_data)

        assert "cross_source_consensus" in result.validations_performed
        assert result.metadata.get("cross_source_consensus") == 149.0

    def test_validate_basic_constraints_valid(self, validator, sample_price_data):
        """Test basic constraints validation with valid data"""
        result = validator._validate_basic_constraints(sample_price_data)

        assert result["valid"]

    def test_validate_basic_constraints_negative_price(self, validator):
        """Test basic constraints with negative price"""
        invalid_data = CurrentPrice(symbol="AAPL", price=-10.0, timestamp=datetime.now(), volume=1000000, source="test_source", quality_score=90.0
        )

        result = validator._validate_basic_constraints(invalid_data)

        assert not result["valid"]
        assert "Invalid price" in result["error"]

    def test_validate_basic_constraints_negative_volume(self, validator):
        """Test basic constraints with negative volume"""
        invalid_data = CurrentPrice(symbol="AAPL", price=150.0, timestamp=datetime.now(), volume=-1000, source="test_source", quality_score=90.0
        )

        result = validator._validate_basic_constraints(invalid_data)

        assert not result["valid"]
        assert "Invalid volume" in result["error"]

    def test_buffer_price_data(self, validator, sample_price_data):
        """Test price data buffering"""
        validator._buffer_price_data("AAPL", "test_source", sample_price_data)

        assert "AAPL" in validator.current_price_buffer
        assert "test_source" in validator.current_price_buffer["AAPL"]
        assert validator.current_price_buffer["AAPL"]["test_source"] == sample_price_data

    def test_buffer_data_cleanup(self, validator):
        """Test automatic cleanup of old buffered data"""
        old_timestamp = datetime.now() - timedelta(seconds=2)
        old_price_data = CurrentPrice(symbol="AAPL", price=150.0, timestamp=old_timestamp, volume=1000000, source="test_source", quality_score=90.0
        )

        validator._buffer_price_data("AAPL", "old_source", old_price_data)

        # Trigger cleanup by buffering new data
        new_price_data = CurrentPrice(symbol="AAPL", price=151.0, timestamp=datetime.now(), volume=1000000, source="test_source", quality_score=90.0
        )
        validator._buffer_price_data("AAPL", "new_source", new_price_data)

        # Old data should be cleaned up after timeout
        time.sleep(validator.buffer_timeout + 0.1)
        validator._buffer_price_data("AAPL", "trigger_cleanup", new_price_data)

        if "AAPL" in validator.current_price_buffer:
            assert "old_source" not in validator.current_price_buffer["AAPL"]

    def test_has_multiple_sources(self, validator, sample_price_data):
        """Test checking for multiple sources"""
        # Initially no sources
        assert not validator._has_multiple_sources("AAPL")

        # Add one source
        validator._buffer_price_data("AAPL", "source1", sample_price_data)
        assert not validator._has_multiple_sources("AAPL")

        # Add second source
        validator._buffer_price_data("AAPL", "source2", sample_price_data)
        assert validator._has_multiple_sources("AAPL")

    def test_determine_final_status(self, validator):
        """Test final status determination logic"""
        # Test with no errors or warnings
        status = validator._determine_final_status(0.9, [], [], 3.0)
        assert status == ValidationStatus.PASS

        # Test with warnings
        status = validator._determine_final_status(0.9, ["warning"], [], 3.0)
        assert status == ValidationStatus.PASS_WITH_WARNINGS

        # Test with errors
        status = validator._determine_final_status(0.9, [], ["error"], 3.0)
        assert status == ValidationStatus.FAIL

        # Test with low confidence (quarantine)
        status = validator._determine_final_status(0.2, [], [], 3.0)
        assert status == ValidationStatus.QUARANTINE

        # Test with processing time exceeded
        status = validator._determine_final_status(0.9, [], [], 10.0)  # Over 5ms limit
        assert status == ValidationStatus.BYPASSED

    @pytest.mark.asyncio
    async def test_circuit_breaker_trigger(self, validator, sample_price_data):
        """Test circuit breaker triggering"""
        # Manually trigger circuit breaker
        validator._trigger_circuit_breaker("Test trigger")

        assert validator.circuit_breaker_triggered

        result = await validator.validate_real_time_price("AAPL", "test_source", sample_price_data)

        assert result.status == ValidationStatus.BYPASSED
        assert "circuit_breaker_bypass" in result.validations_performed
        assert "circuit breaker" in result.warnings[0].lower()

    @pytest.mark.asyncio
    async def test_circuit_breaker_reset(self, validator, sample_price_data):
        """Test circuit breaker automatic reset"""
        # Trigger circuit breaker
        validator._trigger_circuit_breaker("Test trigger")

        # Set last check to more than 60 seconds ago to trigger reset
        validator.last_circuit_breaker_check = datetime.now() - timedelta(seconds=65)

        result = await validator.validate_real_time_price("AAPL", "test_source", sample_price_data)

        # Circuit breaker should be reset
        assert not validator.circuit_breaker_triggered

    def test_performance_metrics_update(self, validator):
        """Test performance metrics tracking"""
        initial_count = validator.validation_count

        result = RealTimeValidationResult(
            status=ValidationStatus.PASS,
            data_accepted=True,
            confidence=0.95,
            processing_time_ms=3.5,
            validations_performed=["basic_constraints"],
            warnings=[],
            errors=[],
            metadata={}
        )

        validator._update_performance_metrics(result)

        assert validator.validation_count == initial_count + 1
        assert len(validator.validation_history) > 0

    def test_performance_metrics_circuit_breaker(self, validator):
        """Test circuit breaker triggering due to performance"""
        # Create slow result
        slow_result = RealTimeValidationResult(
            status=ValidationStatus.PASS,
            data_accepted=True,
            confidence=0.95,
            processing_time_ms=15.0,  # Slower than 2x max time
            validations_performed=["basic_constraints"],
            warnings=[],
            errors=[],
            metadata={}
        )

        validator._update_performance_metrics(slow_result)

        assert validator.circuit_breaker_triggered

    def test_validation_history_limit(self, validator):
        """Test validation history size limiting"""
        # Add many results to exceed limit
        for i in range(1500):
            result = RealTimeValidationResult(
                status=ValidationStatus.PASS,
                data_accepted=True,
                confidence=0.95,
                processing_time_ms=2.0,
                validations_performed=["basic_constraints"],
                warnings=[],
                errors=[],
                metadata={}
            )
            validator.validation_history.append(result)

        # Add one more to trigger cleanup
        validator._update_performance_metrics(result)

        assert len(validator.validation_history) <= 1000

    @pytest.mark.asyncio
    async def test_callback_execution(self, validator, sample_price_data):
        """Test callback execution"""
        callback_called = False

        async def test_callback(result, symbol, source, price_data):
            nonlocal callback_called
            callback_called = True

        validator.on_validation_complete = test_callback

        await validator.validate_real_time_price("AAPL", "test_source", sample_price_data)

        assert callback_called

    @pytest.mark.asyncio
    async def test_safe_callback_error_handling(self, validator):
        """Test safe callback error handling"""
        def failing_callback(result, symbol, source, price_data):
            raise Exception("Callback error")

        validator.on_validation_complete = failing_callback

        # Should not raise exception despite callback failure
        result = RealTimeValidationResult(
            status=ValidationStatus.PASS,
            data_accepted=True,
            confidence=0.95,
            processing_time_ms=2.0,
            validations_performed=["basic_constraints"],
            warnings=[],
            errors=[],
            metadata={}
        )

        await validator._trigger_callbacks(result, "AAPL", "test_source", CurrentPrice(symbol="AAPL", price=150.0, timestamp=datetime.now(), volume=1000000, source="test_source", quality_score=90.0
        ))

    def test_get_performance_metrics(self, validator):
        """Test performance metrics retrieval"""
        # Add some validation history
        for i in range(10):
            result = RealTimeValidationResult(
                status=ValidationStatus.PASS if i % 2 == 0 else ValidationStatus.FAIL,
                data_accepted=i % 2 == 0,
                confidence=0.95,
                processing_time_ms=2.0 + i * 0.1,
                validations_performed=["basic_constraints"],
                warnings=[],
                errors=[],
                metadata={}
            )
            validator._update_performance_metrics(result)

        metrics = validator.get_performance_metrics()

        assert "total_validations" in metrics
        assert "average_processing_time_ms" in metrics
        assert "recent_status_distribution" in metrics
        assert metrics["total_validations"] == 10

    def test_get_performance_metrics_no_data(self, validator):
        """Test performance metrics with no validation data"""
        metrics = validator.get_performance_metrics()

        assert "error" in metrics
        assert "No validations performed" in metrics["error"]

    def test_reset_performance_metrics(self, validator):
        """Test performance metrics reset"""
        # Add some data
        validator.validation_count = 10
        validator.total_processing_time = 50.0
        validator.validation_history = [Mock()] * 5
        validator.circuit_breaker_triggered = True

        validator.reset_performance_metrics()

        assert validator.validation_count == 0
        assert validator.total_processing_time == 0.0
        assert len(validator.validation_history) == 0
        assert not validator.circuit_breaker_triggered

    def test_get_buffered_data_summary(self, validator, sample_price_data):
        """Test buffered data summary"""
        # Add some buffered data
        validator._buffer_price_data("AAPL", "source1", sample_price_data)

        price_data_2 = CurrentPrice(symbol="AAPL", price=151.0, timestamp=datetime.now(), volume=1100000, source="test_source", quality_score=85.0
        )
        validator._buffer_price_data("AAPL", "source2", price_data_2)

        summary = validator.get_buffered_data_summary()

        assert "AAPL" in summary
        assert summary["AAPL"]["source_count"] == 2
        assert "source1" in summary["AAPL"]["sources"]
        assert "source2" in summary["AAPL"]["sources"]
        assert "price_range" in summary["AAPL"]

    def test_set_emergency_mode(self, validator):
        """Test emergency mode toggle"""
        assert not validator.emergency_mode

        validator.set_emergency_mode(True)
        assert validator.emergency_mode

        validator.set_emergency_mode(False)
        assert not validator.emergency_mode

    @pytest.mark.asyncio
    async def test_validation_error_handling(self, validator):
        """Test error handling during validation"""
        # Create price data that will cause an exception in validation
        with patch.object(validator, '_perform_validations', side_effect=Exception("Test error")):
            result = await validator.validate_real_time_price(
                "AAPL", "test_source", CurrentPrice(symbol="AAPL", price=150.0, timestamp=datetime.now(), volume=1000000, source="test_source", quality_score=90.0
                )
            )

            assert result.status == ValidationStatus.FAIL
            assert not result.data_accepted
            assert len(result.errors) > 0
            assert "Validation error" in result.errors[0]

    def test_real_time_validation_result_serialization(self):
        """Test RealTimeValidationResult serialization"""
        result = RealTimeValidationResult(
            status=ValidationStatus.PASS,
            data_accepted=True,
            confidence=0.95,
            processing_time_ms=3.5,
            validations_performed=["basic_constraints", "statistical_analysis"],
            warnings=["minor warning"],
            errors=[],
            metadata={"test": "data"}
        )

        serialized = result.to_dict()

        assert serialized["status"] == "pass"
        assert serialized["data_accepted"] is True
        assert serialized["confidence"] == 0.95
        assert serialized["processing_time_ms"] == 3.5
        assert serialized["validations_performed"] == ["basic_constraints", "statistical_analysis"]
        assert serialized["warnings"] == ["minor warning"]
        assert serialized["metadata"] == {"test": "data"}

    @pytest.mark.asyncio
    async def test_performance_constraint_adherence(self, validator, sample_price_data):
        """Test that validation adheres to performance constraints"""
        start_time = time.perf_counter()

        result = await validator.validate_real_time_price("AAPL", "test_source", sample_price_data)

        elapsed_time_ms = (time.perf_counter() - start_time) * 1000

        # Result should report accurate processing time
        assert result.processing_time_ms >= 0
        # Should generally complete within reasonable time (allowing for test overhead)
        assert elapsed_time_ms < 50  # 50ms should be plenty for test environment