"""
Data Quality Controller for Bulk Data Loading
Advanced validation, quality scoring, and data cleansing for market data imports
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import re
import statistics
from decimal import Decimal, InvalidOperation
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"      # 90-100% quality score
    GOOD = "good"               # 70-89% quality score
    FAIR = "fair"               # 50-69% quality score
    POOR = "poor"               # 30-49% quality score
    UNACCEPTABLE = "unacceptable"  # 0-29% quality score


class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    CRITICAL = "critical"       # Data integrity issues
    WARNING = "warning"         # Quality concerns
    INFO = "info"              # Minor issues or suggestions


@dataclass
class ValidationRule:
    """Configuration for a validation rule"""
    name: str
    description: str
    severity: ValidationSeverity
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationIssue:
    """Represents a validation issue found in data"""
    rule_name: str
    severity: ValidationSeverity
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None
    suggestion: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class QualityMetrics:
    """Data quality metrics for a record or dataset"""
    completeness_score: float = 0.0      # % of required fields present
    accuracy_score: float = 0.0          # % of accurate values
    consistency_score: float = 0.0       # % of consistent values
    validity_score: float = 0.0          # % of valid format values
    timeliness_score: float = 0.0        # % of timely data
    uniqueness_score: float = 0.0        # % of unique records

    @property
    def overall_score(self) -> float:
        """Calculate overall quality score"""
        scores = [
            self.completeness_score,
            self.accuracy_score,
            self.consistency_score,
            self.validity_score,
            self.timeliness_score,
            self.uniqueness_score
        ]
        return sum(scores) / len(scores)

    @property
    def quality_level(self) -> QualityLevel:
        """Get quality level based on overall score"""
        score = self.overall_score
        if score >= 90:
            return QualityLevel.EXCELLENT
        elif score >= 70:
            return QualityLevel.GOOD
        elif score >= 50:
            return QualityLevel.FAIR
        elif score >= 30:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNACCEPTABLE


@dataclass
class DataProfile:
    """Statistical profile of a dataset"""
    total_records: int = 0
    total_fields: int = 0

    # Field completeness
    field_completeness: Dict[str, float] = field(default_factory=dict)

    # Numeric field statistics
    numeric_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # String field statistics
    string_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Date field statistics
    date_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Value distributions
    value_distributions: Dict[str, Dict[Any, int]] = field(default_factory=dict)

    # Outliers
    outliers: Dict[str, List[Any]] = field(default_factory=dict)


class MarketDataValidator:
    """Market data specific validation rules"""

    def __init__(self):
        self.symbol_pattern = re.compile(r'^[A-Z]{1,10}$')
        self.known_exchanges = {'NYSE', 'NASDAQ', 'AMEX', 'TSX', 'LSE'}

    async def validate_symbol(self, symbol: str) -> List[ValidationIssue]:
        """Validate market symbol format"""
        issues = []

        if not symbol:
            issues.append(ValidationIssue(
                rule_name="symbol_required",
                severity=ValidationSeverity.CRITICAL,
                message="Symbol is required",
                field="symbol"
            ))
            return issues

        # Check format
        if not self.symbol_pattern.match(symbol.upper()):
            issues.append(ValidationIssue(
                rule_name="symbol_format",
                severity=ValidationSeverity.CRITICAL,
                message="Symbol must be 1-10 uppercase letters",
                field="symbol",
                value=symbol,
                suggestion="Use uppercase letters only (e.g., AAPL, MSFT)"
            ))

        # Check length
        if len(symbol) > 10:
            issues.append(ValidationIssue(
                rule_name="symbol_length",
                severity=ValidationSeverity.WARNING,
                message="Symbol is unusually long",
                field="symbol",
                value=symbol
            ))

        return issues

    async def validate_price_data(self, record: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate OHLCV price data"""
        issues = []

        # Extract prices
        try:
            open_price = float(record.get('open', 0))
            high_price = float(record.get('high', 0))
            low_price = float(record.get('low', 0))
            close_price = float(record.get('close', 0))
            volume = int(record.get('volume', 0))
        except (ValueError, TypeError) as e:
            issues.append(ValidationIssue(
                rule_name="price_format",
                severity=ValidationSeverity.CRITICAL,
                message=f"Invalid price format: {str(e)}",
                field="prices"
            ))
            return issues

        # Validate positive prices
        for field, price in [('open', open_price), ('high', high_price),
                            ('low', low_price), ('close', close_price)]:
            if price <= 0:
                issues.append(ValidationIssue(
                    rule_name="positive_price",
                    severity=ValidationSeverity.CRITICAL,
                    message=f"{field.capitalize()} price must be positive",
                    field=field,
                    value=price
                ))

        # Validate volume
        if volume < 0:
            issues.append(ValidationIssue(
                rule_name="positive_volume",
                severity=ValidationSeverity.CRITICAL,
                message="Volume cannot be negative",
                field="volume",
                value=volume
            ))

        # Validate price relationships
        if high_price < low_price:
            issues.append(ValidationIssue(
                rule_name="high_low_relationship",
                severity=ValidationSeverity.CRITICAL,
                message="High price cannot be less than low price",
                field="prices",
                value={"high": high_price, "low": low_price}
            ))

        if not (low_price <= open_price <= high_price):
            issues.append(ValidationIssue(
                rule_name="open_price_range",
                severity=ValidationSeverity.CRITICAL,
                message="Open price must be between high and low prices",
                field="open",
                value=open_price,
                suggestion=f"Open should be between {low_price} and {high_price}"
            ))

        if not (low_price <= close_price <= high_price):
            issues.append(ValidationIssue(
                rule_name="close_price_range",
                severity=ValidationSeverity.CRITICAL,
                message="Close price must be between high and low prices",
                field="close",
                value=close_price,
                suggestion=f"Close should be between {low_price} and {high_price}"
            ))

        # Check for suspicious price movements
        price_range = high_price - low_price
        avg_price = (open_price + close_price) / 2

        if price_range > (avg_price * 0.5):  # 50% daily range
            issues.append(ValidationIssue(
                rule_name="excessive_volatility",
                severity=ValidationSeverity.WARNING,
                message="Unusually high price volatility detected",
                field="prices",
                value={"range": price_range, "avg": avg_price}
            ))

        return issues

    async def validate_timestamp(self, timestamp: Any) -> List[ValidationIssue]:
        """Validate timestamp format and reasonableness"""
        issues = []

        if not timestamp:
            issues.append(ValidationIssue(
                rule_name="timestamp_required",
                severity=ValidationSeverity.CRITICAL,
                message="Timestamp is required",
                field="timestamp"
            ))
            return issues

        # Parse timestamp
        try:
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            elif isinstance(timestamp, (int, float)):
                dt = datetime.fromtimestamp(timestamp)
            elif isinstance(timestamp, datetime):
                dt = timestamp
            else:
                raise ValueError("Unsupported timestamp format")

        except (ValueError, OSError) as e:
            issues.append(ValidationIssue(
                rule_name="timestamp_format",
                severity=ValidationSeverity.CRITICAL,
                message=f"Invalid timestamp format: {str(e)}",
                field="timestamp",
                value=timestamp
            ))
            return issues

        # Check timestamp reasonableness
        now = datetime.now()
        min_date = datetime(1970, 1, 1)  # Unix epoch
        max_future_date = now + timedelta(days=1)  # Allow 1 day in future

        if dt < min_date:
            issues.append(ValidationIssue(
                rule_name="timestamp_too_old",
                severity=ValidationSeverity.WARNING,
                message="Timestamp is before Unix epoch",
                field="timestamp",
                value=dt
            ))

        if dt > max_future_date:
            issues.append(ValidationIssue(
                rule_name="timestamp_future",
                severity=ValidationSeverity.WARNING,
                message="Timestamp is in the future",
                field="timestamp",
                value=dt
            ))

        # Check for business day (optional warning)
        if dt.weekday() > 4:  # Saturday or Sunday
            issues.append(ValidationIssue(
                rule_name="weekend_trading",
                severity=ValidationSeverity.INFO,
                message="Trading data on weekend",
                field="timestamp",
                value=dt
            ))

        return issues


class DataQualityController:
    """Main data quality controller"""

    def __init__(self):
        self.validators = {
            'market_data': MarketDataValidator()
        }

        self.validation_rules: List[ValidationRule] = []
        self.custom_validators: List[Callable] = []

        # Quality thresholds
        self.quality_thresholds = {
            QualityLevel.EXCELLENT: 90.0,
            QualityLevel.GOOD: 70.0,
            QualityLevel.FAIR: 50.0,
            QualityLevel.POOR: 30.0
        }

        # Initialize default rules
        self._initialize_default_rules()

    def _initialize_default_rules(self):
        """Initialize default validation rules"""
        default_rules = [
            ValidationRule(
                name="required_fields",
                description="Check for required fields",
                severity=ValidationSeverity.CRITICAL
            ),
            ValidationRule(
                name="data_types",
                description="Validate data types",
                severity=ValidationSeverity.CRITICAL
            ),
            ValidationRule(
                name="value_ranges",
                description="Check value ranges",
                severity=ValidationSeverity.WARNING
            ),
            ValidationRule(
                name="business_rules",
                description="Apply business logic validation",
                severity=ValidationSeverity.WARNING
            ),
            ValidationRule(
                name="outlier_detection",
                description="Detect statistical outliers",
                severity=ValidationSeverity.INFO
            )
        ]

        self.validation_rules.extend(default_rules)

    async def validate_record(self, record: Dict[str, Any], record_type: str = 'market_data') -> Tuple[List[ValidationIssue], QualityMetrics]:
        """Validate a single record and calculate quality metrics"""
        issues = []

        # Get appropriate validator
        validator = self.validators.get(record_type)
        if not validator:
            issues.append(ValidationIssue(
                rule_name="unknown_record_type",
                severity=ValidationSeverity.WARNING,
                message=f"No validator for record type: {record_type}"
            ))
            return issues, QualityMetrics()

        # Run validation rules
        if record_type == 'market_data':
            # Symbol validation
            symbol_issues = await validator.validate_symbol(record.get('symbol', ''))
            issues.extend(symbol_issues)

            # Price data validation
            price_issues = await validator.validate_price_data(record)
            issues.extend(price_issues)

            # Timestamp validation
            timestamp_issues = await validator.validate_timestamp(record.get('timestamp'))
            issues.extend(timestamp_issues)

        # Run custom validators
        for custom_validator in self.custom_validators:
            try:
                custom_issues = await custom_validator(record)
                if custom_issues:
                    issues.extend(custom_issues)
            except Exception as e:
                issues.append(ValidationIssue(
                    rule_name="custom_validator_error",
                    severity=ValidationSeverity.WARNING,
                    message=f"Custom validator failed: {str(e)}"
                ))

        # Calculate quality metrics
        metrics = await self._calculate_quality_metrics(record, issues)

        return issues, metrics

    async def _calculate_quality_metrics(self, record: Dict[str, Any], issues: List[ValidationIssue]) -> QualityMetrics:
        """Calculate quality metrics for a record"""
        metrics = QualityMetrics()

        # Completeness: percentage of required fields present
        required_fields = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        present_fields = sum(1 for field in required_fields if record.get(field) is not None)
        metrics.completeness_score = (present_fields / len(required_fields)) * 100

        # Accuracy: inverse of critical issues
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        metrics.accuracy_score = max(0, 100 - (len(critical_issues) * 20))  # -20 points per critical issue

        # Validity: format and type validation
        format_issues = [i for i in issues if 'format' in i.rule_name]
        metrics.validity_score = max(0, 100 - (len(format_issues) * 25))

        # Consistency: logical relationships
        relationship_issues = [i for i in issues if 'relationship' in i.rule_name or 'range' in i.rule_name]
        metrics.consistency_score = max(0, 100 - (len(relationship_issues) * 30))

        # Timeliness: timestamp reasonableness
        timestamp_issues = [i for i in issues if 'timestamp' in i.rule_name]
        metrics.timeliness_score = max(0, 100 - (len(timestamp_issues) * 25))

        # Uniqueness: assume 100% for individual records (calculated at batch level)
        metrics.uniqueness_score = 100.0

        return metrics

    async def validate_batch(self, records: List[Dict[str, Any]], record_type: str = 'market_data') -> Tuple[List[List[ValidationIssue]], List[QualityMetrics], QualityMetrics]:
        """Validate a batch of records"""
        all_issues = []
        all_metrics = []

        # Validate individual records
        for record in records:
            issues, metrics = await self.validate_record(record, record_type)
            all_issues.append(issues)
            all_metrics.append(metrics)

        # Calculate batch-level metrics
        batch_metrics = await self._calculate_batch_metrics(records, all_metrics)

        return all_issues, all_metrics, batch_metrics

    async def _calculate_batch_metrics(self, records: List[Dict[str, Any]], individual_metrics: List[QualityMetrics]) -> QualityMetrics:
        """Calculate batch-level quality metrics"""
        if not individual_metrics:
            return QualityMetrics()

        batch_metrics = QualityMetrics()

        # Average individual metrics
        batch_metrics.completeness_score = statistics.mean(m.completeness_score for m in individual_metrics)
        batch_metrics.accuracy_score = statistics.mean(m.accuracy_score for m in individual_metrics)
        batch_metrics.consistency_score = statistics.mean(m.consistency_score for m in individual_metrics)
        batch_metrics.validity_score = statistics.mean(m.validity_score for m in individual_metrics)
        batch_metrics.timeliness_score = statistics.mean(m.timeliness_score for m in individual_metrics)

        # Calculate uniqueness at batch level
        if records:
            unique_records = len(set(
                (r.get('symbol', ''), r.get('timestamp', ''))
                for r in records
            ))
            batch_metrics.uniqueness_score = (unique_records / len(records)) * 100

        return batch_metrics

    async def profile_data(self, records: List[Dict[str, Any]]) -> DataProfile:
        """Create statistical profile of dataset"""
        if not records:
            return DataProfile()

        profile = DataProfile()
        profile.total_records = len(records)

        # Get all field names
        all_fields = set()
        for record in records:
            all_fields.update(record.keys())

        profile.total_fields = len(all_fields)

        # Calculate field completeness
        for field in all_fields:
            non_null_count = sum(1 for r in records if r.get(field) is not None)
            profile.field_completeness[field] = (non_null_count / len(records)) * 100

        # Analyze each field
        for field in all_fields:
            values = [r.get(field) for r in records if r.get(field) is not None]

            if not values:
                continue

            # Determine field type and create appropriate statistics
            if all(isinstance(v, (int, float)) for v in values):
                profile.numeric_stats[field] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'std_dev': statistics.stdev(values) if len(values) > 1 else 0
                }

                # Detect outliers (values beyond 2 standard deviations)
                if len(values) > 1:
                    mean = statistics.mean(values)
                    std_dev = statistics.stdev(values)
                    outliers = [v for v in values if abs(v - mean) > 2 * std_dev]
                    if outliers:
                        profile.outliers[field] = outliers

            elif all(isinstance(v, str) for v in values):
                profile.string_stats[field] = {
                    'min_length': min(len(v) for v in values),
                    'max_length': max(len(v) for v in values),
                    'avg_length': statistics.mean(len(v) for v in values),
                    'unique_values': len(set(values))
                }

            # Value distribution (top 10 most common values)
            from collections import Counter
            value_counts = Counter(values)
            profile.value_distributions[field] = dict(value_counts.most_common(10))

        return profile

    async def detect_anomalies(self, records: List[Dict[str, Any]], field: str) -> List[Dict[str, Any]]:
        """Detect anomalies in a specific field using statistical methods"""
        values = [r.get(field) for r in records if r.get(field) is not None]

        if not values or not all(isinstance(v, (int, float)) for v in values):
            return []

        anomalies = []

        # Z-score based anomaly detection
        if len(values) > 1:
            mean = statistics.mean(values)
            std_dev = statistics.stdev(values)

            for i, record in enumerate(records):
                value = record.get(field)
                if value is not None and isinstance(value, (int, float)):
                    z_score = abs(value - mean) / std_dev if std_dev > 0 else 0

                    if z_score > 3:  # 3 standard deviations
                        anomalies.append({
                            'record_index': i,
                            'field': field,
                            'value': value,
                            'z_score': z_score,
                            'type': 'statistical_outlier'
                        })

        return anomalies

    async def suggest_data_corrections(self, record: Dict[str, Any], issues: List[ValidationIssue]) -> Dict[str, Any]:
        """Suggest corrections for data quality issues"""
        suggestions = {}

        for issue in issues:
            if issue.suggestion:
                suggestions[issue.field or 'general'] = issue.suggestion
            elif issue.rule_name == 'symbol_format' and issue.field == 'symbol':
                # Suggest uppercase conversion
                if issue.value:
                    suggestions['symbol'] = str(issue.value).upper()
            elif issue.rule_name == 'positive_price' and issue.field in ['open', 'high', 'low', 'close']:
                # Suggest taking absolute value for negative prices
                if issue.value and issue.value < 0:
                    suggestions[issue.field] = abs(issue.value)

        return suggestions

    def add_custom_validator(self, validator: Callable[[Dict[str, Any]], List[ValidationIssue]]) -> None:
        """Add custom validation function"""
        self.custom_validators.append(validator)

    def get_quality_summary(self, metrics_list: List[QualityMetrics]) -> Dict[str, Any]:
        """Get summary of quality metrics"""
        if not metrics_list:
            return {}

        overall_scores = [m.overall_score for m in metrics_list]

        quality_levels = [m.quality_level for m in metrics_list]
        level_counts = {}
        for level in QualityLevel:
            level_counts[level.value] = sum(1 for ql in quality_levels if ql == level)

        return {
            'total_records': len(metrics_list),
            'average_quality_score': statistics.mean(overall_scores),
            'min_quality_score': min(overall_scores),
            'max_quality_score': max(overall_scores),
            'quality_level_distribution': level_counts,
            'records_above_threshold': {
                'excellent': sum(1 for s in overall_scores if s >= 90),
                'good': sum(1 for s in overall_scores if s >= 70),
                'acceptable': sum(1 for s in overall_scores if s >= 50)
            }
        }