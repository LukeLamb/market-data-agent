"""Real-Time Validation Implementation

High-performance real-time data validation for streaming financial data with
minimal latency and intelligent filtering to ensure data quality during ingestion.
"""

import asyncio
import time
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

from .statistical_validator import StatisticalValidator, ValidationResult, AnomalyType
from .cross_source_validator import CrossSourceValidator, ConsensusResult
from ..data_sources.base import CurrentPrice, PriceData

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Real-time validation status"""
    PASS = "pass"               # Data passed all validations
    PASS_WITH_WARNINGS = "pass_with_warnings"  # Passed but with concerns
    FAIL = "fail"               # Data failed validation
    QUARANTINE = "quarantine"   # Data quarantined for manual review
    BYPASSED = "bypassed"       # Validation bypassed (emergency mode)


@dataclass
class RealTimeValidationResult:
    """Result of real-time validation"""
    status: ValidationStatus
    data_accepted: bool
    confidence: float = 0.0
    processing_time_ms: float = 0.0
    validations_performed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "status": self.status.value,
            "data_accepted": self.data_accepted,
            "confidence": self.confidence,
            "processing_time_ms": self.processing_time_ms,
            "validations_performed": self.validations_performed,
            "warnings": self.warnings,
            "errors": self.errors,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class RealTimeValidator:
    """High-performance real-time data validator

    Features:
    - Sub-millisecond validation for high-frequency data
    - Configurable validation pipelines
    - Emergency bypass modes for critical situations
    - Intelligent filtering based on data importance
    - Real-time anomaly detection with immediate alerts
    """

    def __init__(self,
                 max_processing_time_ms: float = 5.0,
                 enable_statistical_validation: bool = True,
                 enable_cross_source_validation: bool = True,
                 quarantine_threshold: float = 0.3,
                 emergency_mode: bool = False):
        """Initialize real-time validator

        Args:
            max_processing_time_ms: Maximum allowed processing time
            enable_statistical_validation: Enable statistical analysis
            enable_cross_source_validation: Enable cross-source consensus
            quarantine_threshold: Confidence threshold for quarantine
            emergency_mode: Bypass validation in emergency
        """
        self.max_processing_time_ms = max_processing_time_ms
        self.enable_statistical_validation = enable_statistical_validation
        self.enable_cross_source_validation = enable_cross_source_validation
        self.quarantine_threshold = quarantine_threshold
        self.emergency_mode = emergency_mode

        # Validation components
        self.statistical_validator = StatisticalValidator() if enable_statistical_validation else None
        self.cross_source_validator = CrossSourceValidator() if enable_cross_source_validation else None

        # Performance tracking
        self.validation_count = 0
        self.total_processing_time = 0.0
        self.validation_history: List[RealTimeValidationResult] = []

        # Data buffers for cross-source validation
        self.current_price_buffer: Dict[str, Dict[str, CurrentPrice]] = {}  # symbol -> source -> price
        self.buffer_timeout = 1.0  # seconds

        # Emergency and circuit breaker state
        self.circuit_breaker_triggered = False
        self.last_circuit_breaker_check = datetime.now()

        # Validation callbacks
        self.on_validation_complete: Optional[Callable] = None
        self.on_anomaly_detected: Optional[Callable] = None
        self.on_quarantine: Optional[Callable] = None

    async def validate_real_time_price(self, symbol: str, source: str,
                                     price_data: CurrentPrice) -> RealTimeValidationResult:
        """Validate real-time price data with performance constraints

        Args:
            symbol: Stock symbol
            source: Data source name
            price_data: Current price data

        Returns:
            RealTimeValidationResult with validation outcome
        """
        start_time = time.perf_counter()

        try:
            # Emergency mode bypass
            if self.emergency_mode:
                return RealTimeValidationResult(
                    status=ValidationStatus.BYPASSED,
                    data_accepted=True,
                    confidence=1.0,
                    processing_time_ms=0.1,
                    validations_performed=["emergency_bypass"],
                    metadata={"emergency_mode": True}
                )

            # Circuit breaker check
            if self.circuit_breaker_triggered:
                return await self._handle_circuit_breaker(symbol, source, price_data)

            # Perform validations
            result = await self._perform_validations(symbol, source, price_data, start_time)

            # Update performance metrics
            self._update_performance_metrics(result)

            # Trigger callbacks
            await self._trigger_callbacks(result, symbol, source, price_data)

            return result

        except Exception as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Real-time validation error for {symbol}/{source}: {e}")

            return RealTimeValidationResult(
                status=ValidationStatus.FAIL,
                data_accepted=False,
                confidence=0.0,
                processing_time_ms=processing_time,
                validations_performed=["error_handling"],
                errors=[f"Validation error: {str(e)}"],
                metadata={"exception_type": type(e).__name__}
            )

    async def _perform_validations(self, symbol: str, source: str,
                                 price_data: CurrentPrice, start_time: float) -> RealTimeValidationResult:
        """Perform the actual validation checks"""
        validations_performed = []
        warnings = []
        errors = []
        confidence = 1.0

        # Basic sanity checks (always performed)
        basic_valid = self._validate_basic_constraints(price_data)
        validations_performed.append("basic_constraints")

        if not basic_valid["valid"]:
            return RealTimeValidationResult(
                status=ValidationStatus.FAIL,
                data_accepted=False,
                confidence=0.0,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
                validations_performed=validations_performed,
                errors=[basic_valid["error"]],
                metadata={"failed_validation": "basic_constraints"}
            )

        # Statistical validation (if enabled and time permits)
        if self.statistical_validator and self.enable_statistical_validation:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if elapsed_ms < self.max_processing_time_ms * 0.5:  # Use half time budget
                try:
                    stat_result = self.statistical_validator.validate_price_data(
                        symbol, price_data.price, price_data.timestamp, price_data.volume
                    )
                    validations_performed.append("statistical_analysis")

                    if not stat_result.is_valid:
                        if stat_result.severity in ["error", "critical"]:
                            errors.append(stat_result.message)
                            confidence *= 0.5
                        else:
                            warnings.append(stat_result.message)
                            confidence *= 0.8

                except Exception as e:
                    warnings.append(f"Statistical validation failed: {e}")
                    confidence *= 0.9

        # Buffer data for cross-source validation
        self._buffer_price_data(symbol, source, price_data)

        # Cross-source validation (if enabled and multiple sources available)
        cross_source_result = None
        if (self.cross_source_validator and self.enable_cross_source_validation and
            self._has_multiple_sources(symbol)):

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if elapsed_ms < self.max_processing_time_ms * 0.8:  # Use remaining time budget
                try:
                    cross_source_result = await self._perform_cross_source_validation(symbol)
                    validations_performed.append("cross_source_consensus")

                    if cross_source_result and cross_source_result.confidence < 0.7:
                        warnings.append(f"Low cross-source confidence: {cross_source_result.confidence:.2f}")
                        confidence *= cross_source_result.confidence

                except Exception as e:
                    warnings.append(f"Cross-source validation failed: {e}")
                    confidence *= 0.9

        # Determine final status
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        status = self._determine_final_status(confidence, warnings, errors, processing_time_ms)

        # Create metadata
        metadata = {
            "symbol": symbol,
            "source": source,
            "price": price_data.price,
            "quality_score": price_data.quality_score
        }

        if cross_source_result:
            metadata["cross_source_consensus"] = cross_source_result.consensus_value
            metadata["agreement_level"] = cross_source_result.agreement_level.value

        return RealTimeValidationResult(
            status=status,
            data_accepted=status in [ValidationStatus.PASS, ValidationStatus.PASS_WITH_WARNINGS],
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            validations_performed=validations_performed,
            warnings=warnings,
            errors=errors,
            metadata=metadata
        )

    def _validate_basic_constraints(self, price_data: CurrentPrice) -> Dict[str, Any]:
        """Perform basic validation constraints"""
        # Price must be positive
        if price_data.price <= 0:
            return {"valid": False, "error": f"Invalid price: {price_data.price}"}

        # Price must be reasonable
        if price_data.price > 100000:  # $100k per share seems unreasonable for most stocks
            return {"valid": False, "error": f"Price seems unreasonably high: ${price_data.price:,.2f}"}

        # Timestamp must be recent
        if price_data.timestamp:
            age = (datetime.now() - price_data.timestamp).total_seconds()
            if age > 3600:  # 1 hour old
                return {"valid": False, "error": f"Data is too old: {age:.0f} seconds"}

        # Volume validation (if present)
        if price_data.volume is not None and price_data.volume < 0:
            return {"valid": False, "error": f"Invalid volume: {price_data.volume}"}

        return {"valid": True}

    def _buffer_price_data(self, symbol: str, source: str, price_data: CurrentPrice) -> None:
        """Buffer price data for cross-source validation"""
        if symbol not in self.current_price_buffer:
            self.current_price_buffer[symbol] = {}

        self.current_price_buffer[symbol][source] = price_data

        # Clean old data
        cutoff_time = datetime.now() - timedelta(seconds=self.buffer_timeout)
        for sym in list(self.current_price_buffer.keys()):
            for src in list(self.current_price_buffer[sym].keys()):
                if self.current_price_buffer[sym][src].timestamp < cutoff_time:
                    del self.current_price_buffer[sym][src]
            if not self.current_price_buffer[sym]:
                del self.current_price_buffer[sym]

    def _has_multiple_sources(self, symbol: str) -> bool:
        """Check if multiple sources have recent data for symbol"""
        return (symbol in self.current_price_buffer and
                len(self.current_price_buffer[symbol]) >= 2)

    async def _perform_cross_source_validation(self, symbol: str) -> Optional[ConsensusResult]:
        """Perform cross-source validation for buffered data"""
        if not self._has_multiple_sources(symbol):
            return None

        try:
            source_data = self.current_price_buffer[symbol]
            return self.cross_source_validator.validate_current_prices(source_data)
        except Exception as e:
            logger.warning(f"Cross-source validation failed for {symbol}: {e}")
            return None

    def _determine_final_status(self, confidence: float, warnings: List[str],
                              errors: List[str], processing_time_ms: float) -> ValidationStatus:
        """Determine final validation status"""
        # Check processing time constraint
        if processing_time_ms > self.max_processing_time_ms:
            return ValidationStatus.BYPASSED

        # Check for errors
        if errors:
            return ValidationStatus.FAIL

        # Check quarantine threshold
        if confidence < self.quarantine_threshold:
            return ValidationStatus.QUARANTINE

        # Check for warnings
        if warnings:
            return ValidationStatus.PASS_WITH_WARNINGS

        return ValidationStatus.PASS

    async def _handle_circuit_breaker(self, symbol: str, source: str,
                                    price_data: CurrentPrice) -> RealTimeValidationResult:
        """Handle validation when circuit breaker is triggered"""
        # Check if circuit breaker should be reset
        time_since_trigger = (datetime.now() - self.last_circuit_breaker_check).total_seconds()
        if time_since_trigger > 60:  # Reset after 1 minute
            self.circuit_breaker_triggered = False
            self.last_circuit_breaker_check = datetime.now()
            logger.info("Circuit breaker reset - resuming normal validation")

        return RealTimeValidationResult(
            status=ValidationStatus.BYPASSED,
            data_accepted=True,
            confidence=0.5,
            processing_time_ms=0.1,
            validations_performed=["circuit_breaker_bypass"],
            warnings=["Validation bypassed due to circuit breaker"],
            metadata={
                "circuit_breaker": True,
                "time_since_trigger": time_since_trigger
            }
        )

    def _update_performance_metrics(self, result: RealTimeValidationResult) -> None:
        """Update performance tracking metrics"""
        self.validation_count += 1
        self.total_processing_time += result.processing_time_ms

        # Keep recent history
        self.validation_history.append(result)
        if len(self.validation_history) > 1000:
            self.validation_history = self.validation_history[-1000:]

        # Trigger circuit breaker if performance degrades
        if (result.processing_time_ms > self.max_processing_time_ms * 2 and
            not self.circuit_breaker_triggered):
            self._trigger_circuit_breaker("Performance degradation")

    def _trigger_circuit_breaker(self, reason: str) -> None:
        """Trigger circuit breaker to bypass validation"""
        self.circuit_breaker_triggered = True
        self.last_circuit_breaker_check = datetime.now()
        logger.warning(f"Circuit breaker triggered: {reason}")

    async def _trigger_callbacks(self, result: RealTimeValidationResult,
                               symbol: str, source: str, price_data: CurrentPrice) -> None:
        """Trigger registered callbacks based on validation result"""
        try:
            # Validation complete callback
            if self.on_validation_complete:
                await self._safe_callback(self.on_validation_complete, result, symbol, source, price_data)

            # Anomaly detection callback
            if result.status == ValidationStatus.FAIL and self.on_anomaly_detected:
                await self._safe_callback(self.on_anomaly_detected, result, symbol, source, price_data)

            # Quarantine callback
            if result.status == ValidationStatus.QUARANTINE and self.on_quarantine:
                await self._safe_callback(self.on_quarantine, result, symbol, source, price_data)

        except Exception as e:
            logger.error(f"Callback execution failed: {e}")

    async def _safe_callback(self, callback: Callable, *args) -> None:
        """Safely execute callback without blocking validation"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            logger.error(f"Callback failed: {e}")

    def set_emergency_mode(self, enabled: bool) -> None:
        """Enable or disable emergency mode"""
        self.emergency_mode = enabled
        logger.warning(f"Emergency mode {'enabled' if enabled else 'disabled'}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get real-time validation performance metrics"""
        if self.validation_count == 0:
            return {"error": "No validations performed yet"}

        avg_processing_time = self.total_processing_time / self.validation_count

        # Calculate recent performance (last 100 validations)
        recent_results = self.validation_history[-100:]
        if recent_results:
            recent_avg_time = sum(r.processing_time_ms for r in recent_results) / len(recent_results)
            status_counts = {}
            for result in recent_results:
                status_counts[result.status.value] = status_counts.get(result.status.value, 0) + 1
        else:
            recent_avg_time = 0
            status_counts = {}

        return {
            "total_validations": self.validation_count,
            "average_processing_time_ms": avg_processing_time,
            "recent_average_processing_time_ms": recent_avg_time,
            "max_processing_time_ms": self.max_processing_time_ms,
            "recent_status_distribution": status_counts,
            "circuit_breaker_triggered": self.circuit_breaker_triggered,
            "emergency_mode": self.emergency_mode,
            "performance_efficiency": min(1.0, self.max_processing_time_ms / max(1.0, avg_processing_time)),
            "buffered_symbols": len(self.current_price_buffer)
        }

    def reset_performance_metrics(self) -> None:
        """Reset performance tracking metrics"""
        self.validation_count = 0
        self.total_processing_time = 0.0
        self.validation_history.clear()
        self.circuit_breaker_triggered = False

    def get_buffered_data_summary(self) -> Dict[str, Any]:
        """Get summary of currently buffered data"""
        summary = {}
        for symbol, sources in self.current_price_buffer.items():
            summary[symbol] = {
                "source_count": len(sources),
                "sources": list(sources.keys()),
                "latest_timestamp": max(data.timestamp for data in sources.values()).isoformat(),
                "price_range": {
                    "min": min(data.price for data in sources.values()),
                    "max": max(data.price for data in sources.values())
                }
            }
        return summary