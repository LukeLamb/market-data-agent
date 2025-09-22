"""Data Validation Framework

This module provides comprehensive data validation for market data including
price validation, quality scoring, and anomaly detection.
"""

import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import logging

from ..data_sources.base import PriceData, CurrentPrice

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationResult:
    """Result of data validation"""

    def __init__(
        self,
        is_valid: bool,
        quality_score: int,
        issues: List[Dict[str, Any]] = None,
        warnings: List[str] = None
    ):
        self.is_valid = is_valid
        self.quality_score = quality_score  # 0-100 score
        self.issues = issues or []
        self.warnings = warnings or []

    def add_issue(
        self,
        issue_type: str,
        severity: ValidationSeverity,
        description: str,
        field: Optional[str] = None
    ) -> None:
        """Add a validation issue"""
        self.issues.append({
            "type": issue_type,
            "severity": severity.value,
            "description": description,
            "field": field,
            "timestamp": datetime.now()
        })

    def add_warning(self, message: str) -> None:
        """Add a validation warning"""
        self.warnings.append(message)

    def get_severity_counts(self) -> Dict[str, int]:
        """Get count of issues by severity"""
        counts = {severity.value: 0 for severity in ValidationSeverity}
        for issue in self.issues:
            counts[issue["severity"]] += 1
        return counts


class DataValidator:
    """Comprehensive data validation framework

    Features:
    - Price data validation with business rules
    - Quality scoring based on multiple factors
    - Anomaly detection using statistical methods
    - Cross-source data comparison
    - Configurable validation rules
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize data validator

        Args:
            config: Configuration dictionary with validation parameters
        """
        self.config = config or {}

        # Default validation parameters
        self.max_price_change_percent = self.config.get("max_price_change_percent", 50.0)
        self.min_volume = self.config.get("min_volume", 0)
        self.max_bid_ask_spread_percent = self.config.get("max_bid_ask_spread_percent", 10.0)
        self.outlier_threshold_std = self.config.get("outlier_threshold_std", 3.0)

        # Price validation ranges
        self.min_price = self.config.get("min_price", 0.01)
        self.max_price = self.config.get("max_price", 100000.0)

        # Quality scoring weights
        self.quality_weights = self.config.get("quality_weights", {
            "price_validity": 0.3,
            "volume_validity": 0.2,
            "ohlc_consistency": 0.2,
            "freshness": 0.15,
            "completeness": 0.15
        })

        logger.info("Initialized data validator with configuration")

    def validate_price_data(self, data: PriceData) -> ValidationResult:
        """Validate a single PriceData object

        Args:
            data: PriceData object to validate

        Returns:
            ValidationResult with validation outcome and quality score
        """
        result = ValidationResult(is_valid=True, quality_score=100)

        # Basic price validation
        self._validate_prices(data, result)

        # Volume validation
        self._validate_volume(data, result)

        # OHLC consistency validation
        self._validate_ohlc_consistency(data, result)

        # Timestamp validation
        self._validate_timestamp(data, result)

        # Data completeness validation
        self._validate_completeness(data, result)

        # Calculate final quality score
        result.quality_score = self._calculate_quality_score(data, result)

        # Determine if data is valid based on critical issues
        critical_issues = [i for i in result.issues if i["severity"] == "critical"]
        result.is_valid = len(critical_issues) == 0

        return result

    def validate_current_price(self, price: CurrentPrice) -> ValidationResult:
        """Validate a CurrentPrice object

        Args:
            price: CurrentPrice object to validate

        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult(is_valid=True, quality_score=100)

        # Basic price validation
        if price.price <= 0:
            result.add_issue(
                "invalid_price",
                ValidationSeverity.CRITICAL,
                f"Price must be positive, got {price.price}",
                "price"
            )

        if not (self.min_price <= price.price <= self.max_price):
            result.add_issue(
                "price_out_of_range",
                ValidationSeverity.HIGH,
                f"Price {price.price} outside valid range [{self.min_price}, {self.max_price}]",
                "price"
            )

        # Bid-ask spread validation
        if price.bid and price.ask:
            if price.bid >= price.ask:
                result.add_issue(
                    "invalid_bid_ask",
                    ValidationSeverity.HIGH,
                    f"Bid ({price.bid}) should be less than ask ({price.ask})",
                    "bid_ask"
                )
            else:
                spread_percent = ((price.ask - price.bid) / price.price) * 100
                if spread_percent > self.max_bid_ask_spread_percent:
                    result.add_issue(
                        "wide_bid_ask_spread",
                        ValidationSeverity.MEDIUM,
                        f"Bid-ask spread {spread_percent:.2f}% exceeds threshold {self.max_bid_ask_spread_percent}%",
                        "bid_ask"
                    )

        # Volume validation
        if price.volume is not None and price.volume < 0:
            result.add_issue(
                "negative_volume",
                ValidationSeverity.CRITICAL,
                f"Volume cannot be negative, got {price.volume}",
                "volume"
            )

        # Timestamp freshness validation
        if price.timestamp:
            age_minutes = (datetime.now() - price.timestamp).total_seconds() / 60
            if age_minutes > 1440:  # 24 hours
                result.add_warning(f"Price data is {age_minutes:.0f} minutes old")

        # Calculate quality score
        result.quality_score = self._calculate_current_price_quality_score(price, result)

        # Determine validity
        critical_issues = [i for i in result.issues if i["severity"] == "critical"]
        result.is_valid = len(critical_issues) == 0

        return result

    def validate_price_history(self, data_list: List[PriceData]) -> ValidationResult:
        """Validate a list of historical price data

        Args:
            data_list: List of PriceData objects to validate

        Returns:
            ValidationResult with overall validation outcome
        """
        if not data_list:
            result = ValidationResult(is_valid=False, quality_score=0)
            result.add_issue(
                "empty_data",
                ValidationSeverity.CRITICAL,
                "No price data provided",
                "data_list"
            )
            return result

        result = ValidationResult(is_valid=True, quality_score=100)

        # Validate each individual record
        individual_scores = []
        for i, data in enumerate(data_list):
            individual_result = self.validate_price_data(data)
            individual_scores.append(individual_result.quality_score)

            # Aggregate issues with record index
            for issue in individual_result.issues:
                issue["record_index"] = i
                result.issues.append(issue)

            result.warnings.extend(individual_result.warnings)

        # Validate data sequence
        self._validate_data_sequence(data_list, result)

        # Detect anomalies across the dataset
        self._detect_anomalies(data_list, result)

        # Calculate overall quality score
        if individual_scores:
            result.quality_score = int(statistics.mean(individual_scores))

        # Apply penalties for sequence issues
        sequence_penalty = len([i for i in result.issues if i["type"].startswith("sequence_")]) * 5
        result.quality_score = max(0, result.quality_score - sequence_penalty)

        # Determine validity
        critical_issues = [i for i in result.issues if i["severity"] == "critical"]
        result.is_valid = len(critical_issues) == 0

        return result

    def compare_sources(
        self,
        data1: PriceData,
        data2: PriceData,
        tolerance_percent: float = 5.0
    ) -> ValidationResult:
        """Compare price data from two different sources

        Args:
            data1: First price data point
            data2: Second price data point
            tolerance_percent: Acceptable difference percentage

        Returns:
            ValidationResult with comparison outcome
        """
        result = ValidationResult(is_valid=True, quality_score=100)

        if data1.symbol != data2.symbol:
            result.add_issue(
                "symbol_mismatch",
                ValidationSeverity.CRITICAL,
                f"Symbol mismatch: {data1.symbol} vs {data2.symbol}",
                "symbol"
            )
            return result

        # Compare timestamps (should be within reasonable range)
        time_diff = abs((data1.timestamp - data2.timestamp).total_seconds())
        if time_diff > 3600:  # 1 hour
            result.add_warning(f"Large time difference between sources: {time_diff:.0f} seconds")

        # Compare prices
        price_fields = ["open_price", "high_price", "low_price", "close_price"]
        for field in price_fields:
            price1 = getattr(data1, field)
            price2 = getattr(data2, field)

            diff_percent = abs(price1 - price2) / price1 * 100
            if diff_percent > tolerance_percent:
                severity = ValidationSeverity.HIGH if diff_percent > tolerance_percent * 2 else ValidationSeverity.MEDIUM
                result.add_issue(
                    "price_divergence",
                    severity,
                    f"{field} differs by {diff_percent:.2f}% between sources: {price1} vs {price2}",
                    field
                )

        # Compare volumes (higher tolerance as volumes can vary significantly)
        if data1.volume > 0 and data2.volume > 0:
            volume_diff_percent = abs(data1.volume - data2.volume) / data1.volume * 100
            if volume_diff_percent > tolerance_percent * 4:  # Higher tolerance for volume
                result.add_issue(
                    "volume_divergence",
                    ValidationSeverity.MEDIUM,
                    f"Volume differs by {volume_diff_percent:.2f}% between sources: {data1.volume} vs {data2.volume}",
                    "volume"
                )

        # Calculate quality score based on divergence
        divergence_issues = [i for i in result.issues if "divergence" in i["type"]]
        penalty = len(divergence_issues) * 15
        result.quality_score = max(0, 100 - penalty)

        return result

    def _validate_prices(self, data: PriceData, result: ValidationResult) -> None:
        """Validate price fields"""
        prices = {
            "open_price": data.open_price,
            "high_price": data.high_price,
            "low_price": data.low_price,
            "close_price": data.close_price
        }

        for field, price in prices.items():
            if price <= 0:
                result.add_issue(
                    "negative_price",
                    ValidationSeverity.CRITICAL,
                    f"{field} must be positive, got {price}",
                    field
                )
            elif not (self.min_price <= price <= self.max_price):
                result.add_issue(
                    "price_out_of_range",
                    ValidationSeverity.HIGH,
                    f"{field} {price} outside valid range [{self.min_price}, {self.max_price}]",
                    field
                )

    def _validate_volume(self, data: PriceData, result: ValidationResult) -> None:
        """Validate volume field"""
        if data.volume < self.min_volume:
            result.add_issue(
                "invalid_volume",
                ValidationSeverity.CRITICAL if data.volume < 0 else ValidationSeverity.MEDIUM,
                f"Volume {data.volume} below minimum {self.min_volume}",
                "volume"
            )

    def _validate_ohlc_consistency(self, data: PriceData, result: ValidationResult) -> None:
        """Validate OHLC price relationships"""
        # High should be >= all other prices
        if data.high_price < data.open_price:
            result.add_issue(
                "ohlc_inconsistency",
                ValidationSeverity.HIGH,
                f"High ({data.high_price}) < Open ({data.open_price})",
                "high_price"
            )

        if data.high_price < data.low_price:
            result.add_issue(
                "ohlc_inconsistency",
                ValidationSeverity.CRITICAL,
                f"High ({data.high_price}) < Low ({data.low_price})",
                "high_price"
            )

        if data.high_price < data.close_price:
            result.add_issue(
                "ohlc_inconsistency",
                ValidationSeverity.HIGH,
                f"High ({data.high_price}) < Close ({data.close_price})",
                "high_price"
            )

        # Low should be <= all other prices
        if data.low_price > data.open_price:
            result.add_issue(
                "ohlc_inconsistency",
                ValidationSeverity.HIGH,
                f"Low ({data.low_price}) > Open ({data.open_price})",
                "low_price"
            )

        if data.low_price > data.close_price:
            result.add_issue(
                "ohlc_inconsistency",
                ValidationSeverity.HIGH,
                f"Low ({data.low_price}) > Close ({data.close_price})",
                "low_price"
            )

    def _validate_timestamp(self, data: PriceData, result: ValidationResult) -> None:
        """Validate timestamp field"""
        now = datetime.now()

        # Check for future timestamps
        if data.timestamp > now:
            result.add_issue(
                "future_timestamp",
                ValidationSeverity.HIGH,
                f"Timestamp {data.timestamp} is in the future",
                "timestamp"
            )

        # Check for very old timestamps (more than 10 years)
        ten_years_ago = now - timedelta(days=10*365)
        if data.timestamp < ten_years_ago:
            result.add_warning(f"Very old timestamp: {data.timestamp}")

    def _validate_completeness(self, data: PriceData, result: ValidationResult) -> None:
        """Validate data completeness"""
        required_fields = ["symbol", "timestamp", "open_price", "high_price", "low_price", "close_price", "volume"]

        for field in required_fields:
            value = getattr(data, field)
            if value is None:
                result.add_issue(
                    "missing_data",
                    ValidationSeverity.HIGH,
                    f"Required field {field} is missing",
                    field
                )

    def _validate_data_sequence(self, data_list: List[PriceData], result: ValidationResult) -> None:
        """Validate sequence of price data"""
        if len(data_list) < 2:
            return

        # Sort by timestamp for validation
        sorted_data = sorted(data_list, key=lambda x: x.timestamp)

        for i in range(1, len(sorted_data)):
            current = sorted_data[i]
            previous = sorted_data[i-1]

            # Check for duplicate timestamps
            if current.timestamp == previous.timestamp:
                result.add_issue(
                    "sequence_duplicate_timestamp",
                    ValidationSeverity.MEDIUM,
                    f"Duplicate timestamp: {current.timestamp}",
                    "timestamp"
                )

            # Check for large price gaps
            price_change = abs(current.close_price - previous.close_price) / previous.close_price * 100
            if price_change > self.max_price_change_percent:
                result.add_issue(
                    "sequence_large_price_change",
                    ValidationSeverity.HIGH,
                    f"Large price change: {price_change:.2f}% from {previous.close_price} to {current.close_price}",
                    "close_price"
                )

    def _detect_anomalies(self, data_list: List[PriceData], result: ValidationResult) -> None:
        """Detect statistical anomalies in the dataset"""
        if len(data_list) < 5:  # Need minimum data for statistical analysis
            return

        # Analyze price changes
        price_changes = []
        for i in range(1, len(data_list)):
            change = (data_list[i].close_price - data_list[i-1].close_price) / data_list[i-1].close_price
            price_changes.append(change)

        if price_changes:
            mean_change = statistics.mean(price_changes)
            std_change = statistics.stdev(price_changes) if len(price_changes) > 1 else 0

            # Find outliers
            for i, change in enumerate(price_changes):
                if std_change > 0 and abs(change - mean_change) > self.outlier_threshold_std * std_change:
                    result.add_issue(
                        "statistical_outlier",
                        ValidationSeverity.MEDIUM,
                        f"Statistical outlier detected: {change:.4f} change at index {i+1}",
                        "close_price"
                    )

    def _calculate_quality_score(self, data: PriceData, result: ValidationResult) -> int:
        """Calculate quality score for price data"""
        score = 100

        # Deduct points for issues by severity
        severity_penalties = {
            "critical": 30,
            "high": 15,
            "medium": 10,
            "low": 5
        }

        for issue in result.issues:
            penalty = severity_penalties.get(issue["severity"], 5)
            score -= penalty

        # Bonus for data freshness (within last hour)
        if data.timestamp:
            age_hours = (datetime.now() - data.timestamp).total_seconds() / 3600
            if age_hours < 1:
                score += 5
            elif age_hours > 24:
                score -= 5

        return max(0, min(100, score))

    def _calculate_current_price_quality_score(self, price: CurrentPrice, result: ValidationResult) -> int:
        """Calculate quality score for current price data"""
        score = 100

        # Deduct points for issues
        severity_penalties = {
            "critical": 30,
            "high": 15,
            "medium": 10,
            "low": 5
        }

        for issue in result.issues:
            penalty = severity_penalties.get(issue["severity"], 5)
            score -= penalty

        # Bonus for having bid/ask data
        if price.bid and price.ask:
            score += 5

        # Bonus for having volume data
        if price.volume:
            score += 5

        return max(0, min(100, score))