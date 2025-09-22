"""Validation Engine Implementation

Central orchestration engine that coordinates all validation components for
comprehensive data quality assurance across statistical, cross-source, and real-time validation.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

from .statistical_validator import StatisticalValidator, ValidationResult
from .cross_source_validator import CrossSourceValidator, ConsensusResult
from .real_time_validator import RealTimeValidator, RealTimeValidationResult
from .data_quality_assessor import DataQualityAssessor, QualityReport
from ..data_sources.base import CurrentPrice, PriceData

logger = logging.getLogger(__name__)


class ValidationMode(Enum):
    """Validation operation modes"""
    STRICT = "strict"           # All validations must pass
    BALANCED = "balanced"       # Most validations must pass
    PERMISSIVE = "permissive"   # Allow data with warnings
    EMERGENCY = "emergency"     # Minimal validation for critical situations


@dataclass
class ValidationConfig:
    """Configuration for the validation engine"""
    mode: ValidationMode = ValidationMode.BALANCED
    enable_statistical: bool = True
    enable_cross_source: bool = True
    enable_real_time: bool = True
    enable_quality_assessment: bool = True

    # Performance settings
    max_validation_time_ms: float = 10.0
    parallel_validation: bool = True

    # Thresholds
    confidence_threshold: float = 0.7
    quality_score_threshold: float = 70.0

    # Error handling
    continue_on_error: bool = True
    fallback_to_best_source: bool = True

    # Alerting
    alert_on_critical_issues: bool = True
    alert_on_consensus_failure: bool = True


@dataclass
class ValidationSummary:
    """Summary of comprehensive validation results"""
    data_accepted: bool
    overall_confidence: float
    quality_score: float
    validation_mode: ValidationMode

    # Individual validation results
    statistical_result: Optional[ValidationResult] = None
    cross_source_result: Optional[ConsensusResult] = None
    real_time_result: Optional[RealTimeValidationResult] = None
    quality_report: Optional[QualityReport] = None

    # Summary information
    validations_performed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Performance metrics
    total_processing_time_ms: float = 0.0
    validation_breakdown: Dict[str, float] = field(default_factory=dict)

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {
            "data_accepted": self.data_accepted,
            "overall_confidence": self.overall_confidence,
            "quality_score": self.quality_score,
            "validation_mode": self.validation_mode.value,
            "validations_performed": self.validations_performed,
            "warnings": self.warnings,
            "errors": self.errors,
            "recommendations": self.recommendations,
            "total_processing_time_ms": self.total_processing_time_ms,
            "validation_breakdown": self.validation_breakdown,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

        # Add individual results if available
        if self.statistical_result:
            result["statistical_result"] = self.statistical_result.to_dict()
        if self.cross_source_result:
            result["cross_source_result"] = self.cross_source_result.to_dict()
        if self.real_time_result:
            result["real_time_result"] = self.real_time_result.to_dict()
        if self.quality_report:
            result["quality_report"] = self.quality_report.to_dict()

        return result


class ValidationEngine:
    """Central validation engine that orchestrates all validation components

    Features:
    - Coordinated validation across multiple validators
    - Configurable validation modes and thresholds
    - Performance monitoring and optimization
    - Intelligent error handling and fallback mechanisms
    - Real-time alerting and notification system
    """

    def __init__(self, config: ValidationConfig = None):
        """Initialize validation engine

        Args:
            config: Validation configuration (uses defaults if None)
        """
        self.config = config or ValidationConfig()

        # Initialize validation components
        self.statistical_validator = StatisticalValidator() if self.config.enable_statistical else None
        self.cross_source_validator = CrossSourceValidator() if self.config.enable_cross_source else None
        self.real_time_validator = RealTimeValidator() if self.config.enable_real_time else None
        self.quality_assessor = DataQualityAssessor() if self.config.enable_quality_assessment else None

        # Performance tracking
        self.validation_count = 0
        self.total_processing_time = 0.0
        self.validation_history: List[ValidationSummary] = []

        # Alert callbacks
        self.alert_callbacks: List[Callable] = []

        # State management
        self.emergency_mode = False
        self.circuit_breaker_active = False
        self.last_performance_check = datetime.now()

    async def validate_current_price(self, symbol: str, source: str,
                                   price_data: CurrentPrice,
                                   additional_sources: Dict[str, CurrentPrice] = None) -> ValidationSummary:
        """Validate current price data comprehensively

        Args:
            symbol: Stock symbol
            source: Primary data source
            price_data: Current price data to validate
            additional_sources: Additional sources for cross-validation

        Returns:
            ValidationSummary with comprehensive validation results
        """
        start_time = asyncio.get_event_loop().time()

        try:
            # Emergency mode check
            if self.emergency_mode or self.config.mode == ValidationMode.EMERGENCY:
                return await self._emergency_validation(symbol, source, price_data)

            # Circuit breaker check
            if self.circuit_breaker_active:
                return await self._circuit_breaker_validation(symbol, source, price_data)

            # Perform validation suite
            summary = await self._perform_comprehensive_validation(
                symbol, source, price_data, additional_sources, start_time
            )

            # Update performance metrics
            self._update_performance_metrics(summary)

            # Trigger alerts if necessary
            await self._check_and_trigger_alerts(summary, symbol, source)

            return summary

        except Exception as e:
            logger.error(f"Validation engine error for {symbol}/{source}: {e}")

            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            return ValidationSummary(
                data_accepted=self.config.continue_on_error,
                overall_confidence=0.1,
                quality_score=0.0,
                validation_mode=self.config.mode,
                errors=[f"Validation engine error: {str(e)}"],
                total_processing_time_ms=processing_time,
                metadata={"exception": type(e).__name__}
            )

    async def validate_historical_data(self, historical_data: Dict[str, List[PriceData]]) -> ValidationSummary:
        """Validate historical data across sources

        Args:
            historical_data: Dictionary mapping source names to historical data

        Returns:
            ValidationSummary for historical data validation
        """
        start_time = asyncio.get_event_loop().time()

        try:
            validations_performed = []
            warnings = []
            errors = []
            validation_breakdown = {}

            # Quality assessment (primary validation for historical data)
            quality_report = None
            if self.quality_assessor and self.config.enable_quality_assessment:
                assess_start = asyncio.get_event_loop().time()
                quality_report = self.quality_assessor.assess_historical_data(historical_data)
                assess_time = (asyncio.get_event_loop().time() - assess_start) * 1000

                validations_performed.append("quality_assessment")
                validation_breakdown["quality_assessment"] = assess_time

                if quality_report.overall_score < self.config.quality_score_threshold:
                    warnings.append(f"Low quality score: {quality_report.overall_score:.1f}")

            # Cross-source validation for overlapping time periods
            cross_source_results = []
            if self.cross_source_validator and self.config.enable_cross_source and len(historical_data) > 1:
                cross_start = asyncio.get_event_loop().time()
                cross_source_results = self.cross_source_validator.validate_historical_data(historical_data)
                cross_time = (asyncio.get_event_loop().time() - cross_start) * 1000

                validations_performed.append("cross_source_validation")
                validation_breakdown["cross_source_validation"] = cross_time

                # Analyze consensus results
                if cross_source_results:
                    low_confidence_count = sum(1 for r in cross_source_results if r.confidence < 0.5)
                    if low_confidence_count > len(cross_source_results) * 0.2:
                        warnings.append(f"Low cross-source confidence in {low_confidence_count} time periods")

            # Calculate overall metrics
            total_processing_time = (asyncio.get_event_loop().time() - start_time) * 1000

            overall_confidence = 0.8  # Default for historical data
            if quality_report:
                overall_confidence = quality_report.overall_score / 100.0

            quality_score = quality_report.overall_score if quality_report else 75.0

            # Determine acceptance
            data_accepted = self._determine_data_acceptance(
                overall_confidence, quality_score, warnings, errors
            )

            return ValidationSummary(
                data_accepted=data_accepted,
                overall_confidence=overall_confidence,
                quality_score=quality_score,
                validation_mode=self.config.mode,
                quality_report=quality_report,
                validations_performed=validations_performed,
                warnings=warnings,
                errors=errors,
                recommendations=quality_report.recommendations if quality_report else [],
                total_processing_time_ms=total_processing_time,
                validation_breakdown=validation_breakdown,
                metadata={
                    "data_type": "historical",
                    "sources_count": len(historical_data),
                    "total_records": sum(len(data) for data in historical_data.values())
                }
            )

        except Exception as e:
            logger.error(f"Historical data validation error: {e}")

            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            return ValidationSummary(
                data_accepted=False,
                overall_confidence=0.0,
                quality_score=0.0,
                validation_mode=self.config.mode,
                errors=[f"Historical validation error: {str(e)}"],
                total_processing_time_ms=processing_time
            )

    async def _perform_comprehensive_validation(self, symbol: str, source: str,
                                              price_data: CurrentPrice,
                                              additional_sources: Dict[str, CurrentPrice],
                                              start_time: float) -> ValidationSummary:
        """Perform comprehensive validation using all available validators"""

        validations_performed = []
        warnings = []
        errors = []
        validation_breakdown = {}

        # Prepare validation tasks
        validation_tasks = []

        # Statistical validation
        if self.statistical_validator and self.config.enable_statistical:
            task = self._run_statistical_validation(symbol, price_data)
            validation_tasks.append(("statistical", task))

        # Real-time validation
        if self.real_time_validator and self.config.enable_real_time:
            task = self._run_real_time_validation(symbol, source, price_data)
            validation_tasks.append(("real_time", task))

        # Cross-source validation (if additional sources provided)
        cross_source_result = None
        if (self.cross_source_validator and self.config.enable_cross_source and
            additional_sources and len(additional_sources) > 0):

            all_sources = {source: price_data}
            all_sources.update(additional_sources)
            task = self._run_cross_source_validation(all_sources)
            validation_tasks.append(("cross_source", task))

        # Execute validations
        if self.config.parallel_validation and len(validation_tasks) > 1:
            # Run validations in parallel
            results = await self._execute_parallel_validations(validation_tasks, validation_breakdown)
        else:
            # Run validations sequentially
            results = await self._execute_sequential_validations(validation_tasks, validation_breakdown)

        # Extract results
        statistical_result = results.get("statistical")
        real_time_result = results.get("real_time")
        cross_source_result = results.get("cross_source")

        # Quality assessment for current price
        quality_report = None
        if self.quality_assessor and self.config.enable_quality_assessment:
            assess_start = asyncio.get_event_loop().time()

            # Create assessment data structure
            assessment_data = {symbol: {source: price_data}}
            if additional_sources:
                assessment_data[symbol].update(additional_sources)

            quality_report = self.quality_assessor.assess_current_prices(assessment_data)
            assess_time = (asyncio.get_event_loop().time() - assess_start) * 1000

            validations_performed.append("quality_assessment")
            validation_breakdown["quality_assessment"] = assess_time

        # Analyze results and calculate metrics
        overall_confidence, quality_score = self._calculate_overall_metrics(
            statistical_result, cross_source_result, real_time_result, quality_report
        )

        # Collect warnings and errors
        self._collect_warnings_and_errors(
            statistical_result, cross_source_result, real_time_result, quality_report,
            warnings, errors
        )

        # Update validations performed list
        validations_performed.extend(results.keys())

        # Calculate total processing time
        total_processing_time = (asyncio.get_event_loop().time() - start_time) * 1000

        # Check performance constraints
        if total_processing_time > self.config.max_validation_time_ms:
            warnings.append(f"Validation exceeded time limit: {total_processing_time:.1f}ms")

        # Determine final acceptance
        data_accepted = self._determine_data_acceptance(overall_confidence, quality_score, warnings, errors)

        return ValidationSummary(
            data_accepted=data_accepted,
            overall_confidence=overall_confidence,
            quality_score=quality_score,
            validation_mode=self.config.mode,
            statistical_result=statistical_result,
            cross_source_result=cross_source_result,
            real_time_result=real_time_result,
            quality_report=quality_report,
            validations_performed=validations_performed,
            warnings=warnings,
            errors=errors,
            recommendations=quality_report.recommendations if quality_report else [],
            total_processing_time_ms=total_processing_time,
            validation_breakdown=validation_breakdown,
            metadata={
                "symbol": symbol,
                "primary_source": source,
                "additional_sources_count": len(additional_sources) if additional_sources else 0,
                "price": price_data.price
            }
        )

    async def _run_statistical_validation(self, symbol: str, price_data: CurrentPrice) -> ValidationResult:
        """Run statistical validation"""
        return self.statistical_validator.validate_price_data(
            symbol, price_data.price, price_data.timestamp, price_data.volume
        )

    async def _run_real_time_validation(self, symbol: str, source: str,
                                      price_data: CurrentPrice) -> RealTimeValidationResult:
        """Run real-time validation"""
        return await self.real_time_validator.validate_real_time_price(symbol, source, price_data)

    async def _run_cross_source_validation(self, all_sources: Dict[str, CurrentPrice]) -> ConsensusResult:
        """Run cross-source validation"""
        return self.cross_source_validator.validate_current_prices(all_sources)

    async def _execute_parallel_validations(self, validation_tasks: List[tuple],
                                          validation_breakdown: Dict[str, float]) -> Dict[str, Any]:
        """Execute validations in parallel"""
        results = {}

        # Create tasks with timing
        timed_tasks = []
        for name, task in validation_tasks:
            timed_task = self._time_validation(name, task)
            timed_tasks.append((name, timed_task))

        # Execute all tasks
        completed_tasks = await asyncio.gather(
            *[task for _, task in timed_tasks],
            return_exceptions=True
        )

        # Collect results
        for (name, _), result in zip(timed_tasks, completed_tasks):
            if isinstance(result, Exception):
                logger.error(f"Validation {name} failed: {result}")
                continue

            validation_result, timing = result
            results[name] = validation_result
            validation_breakdown[name] = timing

        return results

    async def _execute_sequential_validations(self, validation_tasks: List[tuple],
                                            validation_breakdown: Dict[str, float]) -> Dict[str, Any]:
        """Execute validations sequentially"""
        results = {}

        for name, task in validation_tasks:
            try:
                start_time = asyncio.get_event_loop().time()
                result = await task
                end_time = asyncio.get_event_loop().time()

                results[name] = result
                validation_breakdown[name] = (end_time - start_time) * 1000

            except Exception as e:
                logger.error(f"Validation {name} failed: {e}")
                continue

        return results

    async def _time_validation(self, name: str, task) -> tuple:
        """Time a validation task"""
        start_time = asyncio.get_event_loop().time()
        result = await task
        end_time = asyncio.get_event_loop().time()
        timing = (end_time - start_time) * 1000
        return result, timing

    def _calculate_overall_metrics(self, statistical_result: Optional[ValidationResult],
                                 cross_source_result: Optional[ConsensusResult],
                                 real_time_result: Optional[RealTimeValidationResult],
                                 quality_report: Optional[QualityReport]) -> tuple[float, float]:
        """Calculate overall confidence and quality score"""

        confidences = []
        quality_scores = []

        # Statistical validation confidence
        if statistical_result:
            if statistical_result.is_valid:
                confidences.append(0.9)
            else:
                severity_weights = {"info": 0.8, "warning": 0.6, "error": 0.3, "critical": 0.1}
                confidences.append(severity_weights.get(statistical_result.severity, 0.5))

        # Cross-source validation confidence
        if cross_source_result:
            confidences.append(cross_source_result.confidence)

        # Real-time validation confidence
        if real_time_result:
            confidences.append(real_time_result.confidence)

        # Quality assessment
        if quality_report:
            quality_scores.append(quality_report.overall_score)
            confidences.append(quality_report.overall_score / 100.0)

        # Calculate averages
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 75.0

        return overall_confidence, quality_score

    def _collect_warnings_and_errors(self, statistical_result: Optional[ValidationResult],
                                   cross_source_result: Optional[ConsensusResult],
                                   real_time_result: Optional[RealTimeValidationResult],
                                   quality_report: Optional[QualityReport],
                                   warnings: List[str], errors: List[str]) -> None:
        """Collect warnings and errors from all validation results"""

        # Statistical validation
        if statistical_result and not statistical_result.is_valid:
            if statistical_result.severity in ["error", "critical"]:
                errors.append(f"Statistical: {statistical_result.message}")
            else:
                warnings.append(f"Statistical: {statistical_result.message}")

        # Cross-source validation
        if cross_source_result:
            if cross_source_result.confidence < 0.5:
                warnings.append(f"Low cross-source confidence: {cross_source_result.confidence:.2f}")
            if cross_source_result.outlier_sources:
                warnings.append(f"Outlier sources detected: {', '.join(cross_source_result.outlier_sources)}")

        # Real-time validation
        if real_time_result:
            warnings.extend(real_time_result.warnings)
            errors.extend(real_time_result.errors)

        # Quality assessment
        if quality_report:
            critical_issues = [issue for issue in quality_report.issues if issue.severity == "critical"]
            high_issues = [issue for issue in quality_report.issues if issue.severity == "high"]

            if critical_issues:
                errors.extend([f"Quality: {issue.description}" for issue in critical_issues[:3]])
            if high_issues:
                warnings.extend([f"Quality: {issue.description}" for issue in high_issues[:3]])

    def _determine_data_acceptance(self, confidence: float, quality_score: float,
                                 warnings: List[str], errors: List[str]) -> bool:
        """Determine whether to accept the data based on validation results"""

        # Emergency mode - always accept
        if self.config.mode == ValidationMode.EMERGENCY:
            return True

        # Critical errors - never accept in strict mode
        if errors and self.config.mode == ValidationMode.STRICT:
            return False

        # Check confidence and quality thresholds
        confidence_ok = confidence >= self.config.confidence_threshold
        quality_ok = quality_score >= self.config.quality_score_threshold

        if self.config.mode == ValidationMode.STRICT:
            return confidence_ok and quality_ok and not errors
        elif self.config.mode == ValidationMode.BALANCED:
            return (confidence_ok or quality_ok) and not errors
        elif self.config.mode == ValidationMode.PERMISSIVE:
            return confidence > 0.3 or quality_score > 50.0

        return False

    async def _emergency_validation(self, symbol: str, source: str,
                                  price_data: CurrentPrice) -> ValidationSummary:
        """Perform minimal validation in emergency mode"""

        # Basic sanity checks only
        data_accepted = price_data.price > 0

        return ValidationSummary(
            data_accepted=data_accepted,
            overall_confidence=0.5,
            quality_score=50.0,
            validation_mode=ValidationMode.EMERGENCY,
            validations_performed=["emergency_basic_check"],
            warnings=["Emergency mode - minimal validation performed"],
            total_processing_time_ms=0.1,
            metadata={"emergency_mode": True}
        )

    async def _circuit_breaker_validation(self, symbol: str, source: str,
                                        price_data: CurrentPrice) -> ValidationSummary:
        """Handle validation when circuit breaker is active"""

        return ValidationSummary(
            data_accepted=True,
            overall_confidence=0.3,
            quality_score=30.0,
            validation_mode=self.config.mode,
            validations_performed=["circuit_breaker_bypass"],
            warnings=["Circuit breaker active - validation bypassed"],
            total_processing_time_ms=0.1,
            metadata={"circuit_breaker": True}
        )

    async def _check_and_trigger_alerts(self, summary: ValidationSummary,
                                      symbol: str, source: str) -> None:
        """Check for alert conditions and trigger notifications"""

        # Critical issues alert
        if (self.config.alert_on_critical_issues and
            (summary.errors or summary.overall_confidence < 0.3)):
            await self._trigger_alert("critical_issue", {
                "symbol": symbol,
                "source": source,
                "confidence": summary.overall_confidence,
                "errors": summary.errors
            })

        # Consensus failure alert
        if (self.config.alert_on_consensus_failure and summary.cross_source_result and
            summary.cross_source_result.confidence < 0.5):
            await self._trigger_alert("consensus_failure", {
                "symbol": symbol,
                "confidence": summary.cross_source_result.confidence,
                "agreement_level": summary.cross_source_result.agreement_level.value
            })

    async def _trigger_alert(self, alert_type: str, data: Dict[str, Any]) -> None:
        """Trigger alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert_type, data)
                else:
                    callback(alert_type, data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def _update_performance_metrics(self, summary: ValidationSummary) -> None:
        """Update performance tracking metrics"""
        self.validation_count += 1
        self.total_processing_time += summary.total_processing_time_ms

        # Keep recent history
        self.validation_history.append(summary)
        if len(self.validation_history) > 1000:
            self.validation_history = self.validation_history[-1000:]

        # Check for performance issues
        if summary.total_processing_time_ms > self.config.max_validation_time_ms * 3:
            self._consider_circuit_breaker()

    def _consider_circuit_breaker(self) -> None:
        """Consider activating circuit breaker due to performance issues"""
        recent_validations = self.validation_history[-10:]
        if len(recent_validations) >= 5:
            avg_time = sum(v.total_processing_time_ms for v in recent_validations) / len(recent_validations)
            if avg_time > self.config.max_validation_time_ms * 2:
                self.circuit_breaker_active = True
                logger.warning("Circuit breaker activated due to performance degradation")

    def add_alert_callback(self, callback: Callable) -> None:
        """Add alert callback function"""
        self.alert_callbacks.append(callback)

    def set_emergency_mode(self, enabled: bool) -> None:
        """Enable or disable emergency mode"""
        self.emergency_mode = enabled
        logger.warning(f"Emergency mode {'enabled' if enabled else 'disabled'}")

    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker"""
        self.circuit_breaker_active = False
        logger.info("Circuit breaker reset")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if self.validation_count == 0:
            return {"error": "No validations performed"}

        avg_time = self.total_processing_time / self.validation_count
        recent_validations = self.validation_history[-100:]

        if recent_validations:
            recent_avg_time = sum(v.total_processing_time_ms for v in recent_validations) / len(recent_validations)
            acceptance_rate = sum(1 for v in recent_validations if v.data_accepted) / len(recent_validations)
            avg_confidence = sum(v.overall_confidence for v in recent_validations) / len(recent_validations)
        else:
            recent_avg_time = avg_time
            acceptance_rate = 0.0
            avg_confidence = 0.0

        return {
            "total_validations": self.validation_count,
            "average_processing_time_ms": avg_time,
            "recent_average_processing_time_ms": recent_avg_time,
            "recent_acceptance_rate": acceptance_rate,
            "recent_average_confidence": avg_confidence,
            "circuit_breaker_active": self.circuit_breaker_active,
            "emergency_mode": self.emergency_mode,
            "configuration": {
                "mode": self.config.mode.value,
                "max_time_ms": self.config.max_validation_time_ms,
                "confidence_threshold": self.config.confidence_threshold,
                "quality_threshold": self.config.quality_score_threshold
            }
        }