"""Data Quality Assessment Implementation

Comprehensive data quality assessment that analyzes data completeness, accuracy,
consistency, timeliness, and validity to provide actionable quality insights.
"""

import statistics
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

from ..data_sources.base import CurrentPrice, PriceData

logger = logging.getLogger(__name__)


class QualityDimension(Enum):
    """Data quality dimensions"""
    COMPLETENESS = "completeness"     # Data presence and coverage
    ACCURACY = "accuracy"             # Correctness of data values
    CONSISTENCY = "consistency"       # Data consistency across sources/time
    TIMELINESS = "timeliness"        # Data freshness and delivery speed
    VALIDITY = "validity"            # Data conforms to business rules
    UNIQUENESS = "uniqueness"        # No duplicate or redundant data


class QualityIssueType(Enum):
    """Types of quality issues"""
    MISSING_DATA = "missing_data"
    STALE_DATA = "stale_data"
    DUPLICATE_DATA = "duplicate_data"
    INCONSISTENT_VALUES = "inconsistent_values"
    INVALID_FORMAT = "invalid_format"
    OUT_OF_RANGE = "out_of_range"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    CROSS_SOURCE_DISAGREEMENT = "cross_source_disagreement"


@dataclass
class QualityIssue:
    """Represents a specific data quality issue"""
    issue_type: QualityIssueType
    dimension: QualityDimension
    severity: str  # critical, high, medium, low
    description: str
    affected_records: int
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    first_detected: datetime = field(default_factory=datetime.now)
    last_detected: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "issue_type": self.issue_type.value,
            "dimension": self.dimension.value,
            "severity": self.severity,
            "description": self.description,
            "affected_records": self.affected_records,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "first_detected": self.first_detected.isoformat(),
            "last_detected": self.last_detected.isoformat()
        }


@dataclass
class QualityReport:
    """Comprehensive data quality assessment report"""
    overall_score: float  # 0.0 to 100.0
    dimension_scores: Dict[QualityDimension, float]
    issues: List[QualityIssue]
    recommendations: List[str]
    assessment_period: Tuple[datetime, datetime]
    total_records_analyzed: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "overall_score": self.overall_score,
            "dimension_scores": {dim.value: score for dim, score in self.dimension_scores.items()},
            "issues": [issue.to_dict() for issue in self.issues],
            "recommendations": self.recommendations,
            "assessment_period": {
                "start": self.assessment_period[0].isoformat(),
                "end": self.assessment_period[1].isoformat()
            },
            "total_records_analyzed": self.total_records_analyzed,
            "metadata": self.metadata,
            "generated_at": self.generated_at.isoformat()
        }


class DataQualityAssessor:
    """Comprehensive data quality assessment engine

    Features:
    - Multi-dimensional quality analysis
    - Automated issue detection and classification
    - Trend analysis and quality degradation detection
    - Actionable recommendations for quality improvement
    - Historical quality tracking and reporting
    """

    def __init__(self,
                 freshness_threshold_seconds: int = 300,
                 accuracy_tolerance: float = 0.01,
                 consistency_threshold: float = 0.05):
        """Initialize data quality assessor

        Args:
            freshness_threshold_seconds: Maximum age for fresh data
            accuracy_tolerance: Tolerance for accuracy checks
            consistency_threshold: Threshold for consistency validation
        """
        self.freshness_threshold = freshness_threshold_seconds
        self.accuracy_tolerance = accuracy_tolerance
        self.consistency_threshold = consistency_threshold

        # Quality tracking
        self.quality_history: List[QualityReport] = []
        self.issue_tracking: Dict[str, QualityIssue] = {}

        # Assessment cache
        self.cached_assessments: Dict[str, Tuple[datetime, QualityReport]] = {}
        self.cache_ttl = 300  # 5 minutes

    def assess_current_prices(self, price_data: Dict[str, Dict[str, CurrentPrice]]) -> QualityReport:
        """Assess quality of current price data across sources

        Args:
            price_data: Nested dict {symbol: {source: CurrentPrice}}

        Returns:
            QualityReport with comprehensive quality assessment
        """
        assessment_start = datetime.now()
        issues = []
        total_records = sum(len(sources) for sources in price_data.values())

        if total_records == 0:
            return self._create_empty_report(assessment_start)

        # Assess each quality dimension
        completeness_score, completeness_issues = self._assess_completeness(price_data)
        accuracy_score, accuracy_issues = self._assess_accuracy(price_data)
        consistency_score, consistency_issues = self._assess_consistency(price_data)
        timeliness_score, timeliness_issues = self._assess_timeliness(price_data)
        validity_score, validity_issues = self._assess_validity(price_data)
        uniqueness_score, uniqueness_issues = self._assess_uniqueness(price_data)

        # Collect all issues
        all_issues = (completeness_issues + accuracy_issues + consistency_issues +
                     timeliness_issues + validity_issues + uniqueness_issues)

        # Calculate dimension scores
        dimension_scores = {
            QualityDimension.COMPLETENESS: completeness_score,
            QualityDimension.ACCURACY: accuracy_score,
            QualityDimension.CONSISTENCY: consistency_score,
            QualityDimension.TIMELINESS: timeliness_score,
            QualityDimension.VALIDITY: validity_score,
            QualityDimension.UNIQUENESS: uniqueness_score
        }

        # Calculate overall score (weighted average)
        weights = {
            QualityDimension.COMPLETENESS: 0.2,
            QualityDimension.ACCURACY: 0.25,
            QualityDimension.CONSISTENCY: 0.2,
            QualityDimension.TIMELINESS: 0.15,
            QualityDimension.VALIDITY: 0.15,
            QualityDimension.UNIQUENESS: 0.05
        }

        overall_score = sum(dimension_scores[dim] * weight for dim, weight in weights.items())

        # Generate recommendations
        recommendations = self._generate_recommendations(dimension_scores, all_issues)

        # Create report
        assessment_end = datetime.now()
        report = QualityReport(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            issues=all_issues,
            recommendations=recommendations,
            assessment_period=(assessment_start, assessment_end),
            total_records_analyzed=total_records,
            metadata={
                "symbols_analyzed": len(price_data),
                "sources_analyzed": len(set(
                    source for sources in price_data.values() for source in sources.keys()
                )),
                "assessment_duration_ms": (assessment_end - assessment_start).total_seconds() * 1000
            }
        )

        # Update tracking
        self._update_quality_tracking(report)

        return report

    def assess_historical_data(self, historical_data: Dict[str, List[PriceData]],
                             time_window: Optional[timedelta] = None) -> QualityReport:
        """Assess quality of historical data

        Args:
            historical_data: Dict mapping source names to historical data lists
            time_window: Time window to analyze (None for all data)

        Returns:
            QualityReport for historical data quality
        """
        assessment_start = datetime.now()

        # Filter data by time window if specified
        if time_window:
            cutoff_time = assessment_start - time_window
            filtered_data = {}
            for source, data_list in historical_data.items():
                filtered_data[source] = [
                    data for data in data_list if data.timestamp >= cutoff_time
                ]
        else:
            filtered_data = historical_data

        total_records = sum(len(data_list) for data_list in filtered_data.values())

        if total_records == 0:
            return self._create_empty_report(assessment_start)

        issues = []

        # Assess temporal consistency
        temporal_score, temporal_issues = self._assess_temporal_consistency(filtered_data)
        issues.extend(temporal_issues)

        # Assess data gaps
        gaps_score, gaps_issues = self._assess_data_gaps(filtered_data)
        issues.extend(gaps_issues)

        # Assess value consistency over time
        value_consistency_score, value_issues = self._assess_historical_value_consistency(filtered_data)
        issues.extend(value_issues)

        # Calculate dimension scores for historical data
        dimension_scores = {
            QualityDimension.COMPLETENESS: gaps_score,
            QualityDimension.ACCURACY: 85.0,  # Placeholder - would need reference data
            QualityDimension.CONSISTENCY: value_consistency_score,
            QualityDimension.TIMELINESS: temporal_score,
            QualityDimension.VALIDITY: 90.0,  # Placeholder
            QualityDimension.UNIQUENESS: 95.0  # Historical data typically doesn't have duplicates
        }

        # Calculate overall score
        overall_score = statistics.mean(dimension_scores.values())

        # Generate recommendations
        recommendations = self._generate_recommendations(dimension_scores, issues)

        assessment_end = datetime.now()
        return QualityReport(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            issues=issues,
            recommendations=recommendations,
            assessment_period=(assessment_start, assessment_end),
            total_records_analyzed=total_records,
            metadata={
                "data_type": "historical",
                "sources_analyzed": len(filtered_data),
                "time_span_days": (
                    max(data.timestamp for data_list in filtered_data.values() for data in data_list) -
                    min(data.timestamp for data_list in filtered_data.values() for data in data_list)
                ).days if total_records > 0 else 0
            }
        )

    def _assess_completeness(self, price_data: Dict[str, Dict[str, CurrentPrice]]) -> Tuple[float, List[QualityIssue]]:
        """Assess data completeness"""
        issues = []
        all_sources = set()
        symbol_source_counts = {}

        # Collect all sources and count availability per symbol
        for symbol, sources in price_data.items():
            symbol_source_counts[symbol] = len(sources)
            all_sources.update(sources.keys())

        if not all_sources:
            return 0.0, issues

        # Calculate completeness score
        expected_total = len(price_data) * len(all_sources)
        actual_total = sum(symbol_source_counts.values())
        completeness_score = (actual_total / expected_total) * 100 if expected_total > 0 else 0

        # Identify symbols with missing data
        missing_data_symbols = [
            symbol for symbol, count in symbol_source_counts.items()
            if count < len(all_sources)
        ]

        if missing_data_symbols:
            missing_percentage = len(missing_data_symbols) / len(price_data)
            severity = "critical" if missing_percentage > 0.5 else "high" if missing_percentage > 0.2 else "medium"

            issues.append(QualityIssue(
                issue_type=QualityIssueType.MISSING_DATA,
                dimension=QualityDimension.COMPLETENESS,
                severity=severity,
                description=f"{len(missing_data_symbols)} symbols missing data from some sources",
                affected_records=len(missing_data_symbols),
                confidence=0.9,
                metadata={
                    "missing_symbols": missing_data_symbols[:10],  # Limit for readability
                    "expected_sources": len(all_sources),
                    "average_sources_per_symbol": statistics.mean(symbol_source_counts.values())
                }
            ))

        return completeness_score, issues

    def _assess_accuracy(self, price_data: Dict[str, Dict[str, CurrentPrice]]) -> Tuple[float, List[QualityIssue]]:
        """Assess data accuracy using cross-source comparison"""
        issues = []
        accuracy_scores = []

        for symbol, sources in price_data.items():
            if len(sources) < 2:
                continue

            prices = [price_obj.price for price_obj in sources.values()]
            mean_price = statistics.mean(prices)

            # Calculate relative deviations
            deviations = [abs(price - mean_price) / mean_price for price in prices if mean_price > 0]

            if deviations:
                max_deviation = max(deviations)
                avg_deviation = statistics.mean(deviations)

                # Score based on deviation (lower deviation = higher accuracy)
                symbol_accuracy = max(0, 100 - (avg_deviation * 1000))  # Scale appropriately
                accuracy_scores.append(symbol_accuracy)

                # Flag significant deviations
                if max_deviation > self.accuracy_tolerance:
                    outlier_sources = [
                        source for source, price_obj in sources.items()
                        if abs(price_obj.price - mean_price) / mean_price > self.accuracy_tolerance
                    ]

                    severity = "high" if max_deviation > 0.1 else "medium"

                    issues.append(QualityIssue(
                        issue_type=QualityIssueType.INCONSISTENT_VALUES,
                        dimension=QualityDimension.ACCURACY,
                        severity=severity,
                        description=f"Price deviation detected for {symbol}: {max_deviation:.2%}",
                        affected_records=len(outlier_sources),
                        confidence=0.8,
                        metadata={
                            "symbol": symbol,
                            "mean_price": mean_price,
                            "max_deviation": max_deviation,
                            "outlier_sources": outlier_sources,
                            "price_range": {"min": min(prices), "max": max(prices)}
                        }
                    ))

        overall_accuracy = statistics.mean(accuracy_scores) if accuracy_scores else 85.0  # Default score
        return overall_accuracy, issues

    def _assess_consistency(self, price_data: Dict[str, Dict[str, CurrentPrice]]) -> Tuple[float, List[QualityIssue]]:
        """Assess data consistency across sources"""
        issues = []
        consistency_scores = []

        for symbol, sources in price_data.items():
            if len(sources) < 2:
                continue

            # Check price consistency
            prices = [price_obj.price for price_obj in sources.values()]
            price_cv = statistics.stdev(prices) / statistics.mean(prices) if statistics.mean(prices) > 0 else 0

            # Check timestamp consistency
            timestamps = [price_obj.timestamp for price_obj in sources.values() if price_obj.timestamp]
            if timestamps:
                time_spread = (max(timestamps) - min(timestamps)).total_seconds()
                time_consistency = max(0, 100 - time_spread)  # Penalize large time spreads
            else:
                time_consistency = 50  # Neutral score if no timestamps

            # Combine consistency measures
            symbol_consistency = max(0, 100 - (price_cv * 1000)) * 0.7 + time_consistency * 0.3
            consistency_scores.append(symbol_consistency)

            # Flag consistency issues
            if price_cv > self.consistency_threshold:
                issues.append(QualityIssue(
                    issue_type=QualityIssueType.CROSS_SOURCE_DISAGREEMENT,
                    dimension=QualityDimension.CONSISTENCY,
                    severity="medium" if price_cv < 0.1 else "high",
                    description=f"Price inconsistency for {symbol}: CV={price_cv:.3f}",
                    affected_records=len(sources),
                    confidence=0.8,
                    metadata={
                        "symbol": symbol,
                        "coefficient_of_variation": price_cv,
                        "price_spread": max(prices) - min(prices),
                        "time_spread_seconds": time_spread if timestamps else None
                    }
                ))

        overall_consistency = statistics.mean(consistency_scores) if consistency_scores else 80.0
        return overall_consistency, issues

    def _assess_timeliness(self, price_data: Dict[str, Dict[str, CurrentPrice]]) -> Tuple[float, List[QualityIssue]]:
        """Assess data timeliness and freshness"""
        issues = []
        timeliness_scores = []
        now = datetime.now()

        stale_count = 0
        total_records = 0

        for symbol, sources in price_data.items():
            for source, price_obj in sources.items():
                total_records += 1

                if price_obj.timestamp:
                    age_seconds = (now - price_obj.timestamp).total_seconds()

                    # Calculate freshness score
                    if age_seconds <= self.freshness_threshold:
                        freshness_score = 100
                    elif age_seconds <= self.freshness_threshold * 2:
                        freshness_score = 75
                    elif age_seconds <= self.freshness_threshold * 5:
                        freshness_score = 50
                    else:
                        freshness_score = 0
                        stale_count += 1

                    timeliness_scores.append(freshness_score)
                else:
                    # No timestamp = unknown freshness
                    timeliness_scores.append(25)

        if stale_count > 0:
            stale_percentage = stale_count / total_records
            severity = "critical" if stale_percentage > 0.5 else "high" if stale_percentage > 0.2 else "medium"

            issues.append(QualityIssue(
                issue_type=QualityIssueType.STALE_DATA,
                dimension=QualityDimension.TIMELINESS,
                severity=severity,
                description=f"{stale_count} records are stale (>{self.freshness_threshold}s old)",
                affected_records=stale_count,
                confidence=0.9,
                metadata={
                    "stale_percentage": stale_percentage,
                    "freshness_threshold": self.freshness_threshold,
                    "total_records": total_records
                }
            ))

        overall_timeliness = statistics.mean(timeliness_scores) if timeliness_scores else 50.0
        return overall_timeliness, issues

    def _assess_validity(self, price_data: Dict[str, Dict[str, CurrentPrice]]) -> Tuple[float, List[QualityIssue]]:
        """Assess data validity against business rules"""
        issues = []
        validity_scores = []

        for symbol, sources in price_data.items():
            for source, price_obj in sources.items():
                score = 100  # Start with perfect score

                # Price must be positive
                if price_obj.price <= 0:
                    score = 0
                    issues.append(QualityIssue(
                        issue_type=QualityIssueType.INVALID_FORMAT,
                        dimension=QualityDimension.VALIDITY,
                        severity="critical",
                        description=f"Invalid price: {price_obj.price} for {symbol}/{source}",
                        affected_records=1,
                        confidence=1.0,
                        metadata={"price": price_obj.price, "symbol": symbol, "source": source}
                    ))

                # Price should be reasonable
                elif price_obj.price > 100000:  # $100k seems unreasonable for most stocks
                    score -= 20
                    issues.append(QualityIssue(
                        issue_type=QualityIssueType.OUT_OF_RANGE,
                        dimension=QualityDimension.VALIDITY,
                        severity="medium",
                        description=f"Unusually high price: ${price_obj.price:,.2f} for {symbol}/{source}",
                        affected_records=1,
                        confidence=0.7,
                        metadata={"price": price_obj.price, "symbol": symbol, "source": source}
                    ))

                # Volume validation (if present)
                if price_obj.volume is not None:
                    if price_obj.volume < 0:
                        score -= 30
                        issues.append(QualityIssue(
                            issue_type=QualityIssueType.INVALID_FORMAT,
                            dimension=QualityDimension.VALIDITY,
                            severity="high",
                            description=f"Invalid volume: {price_obj.volume} for {symbol}/{source}",
                            affected_records=1,
                            confidence=1.0,
                            metadata={"volume": price_obj.volume, "symbol": symbol, "source": source}
                        ))

                validity_scores.append(max(0, score))

        overall_validity = statistics.mean(validity_scores) if validity_scores else 90.0
        return overall_validity, issues

    def _assess_uniqueness(self, price_data: Dict[str, Dict[str, CurrentPrice]]) -> Tuple[float, List[QualityIssue]]:
        """Assess data uniqueness (no duplicates)"""
        issues = []
        duplicate_count = 0
        total_records = sum(len(sources) for sources in price_data.values())

        # Check for duplicate prices from same source
        for symbol, sources in price_data.items():
            price_counts = {}
            for source, price_obj in sources.items():
                price_key = (price_obj.price, price_obj.timestamp)
                if price_key in price_counts:
                    duplicate_count += 1
                    price_counts[price_key].append(source)
                else:
                    price_counts[price_key] = [source]

            # Report duplicates
            duplicates = {key: sources for key, sources in price_counts.items() if len(sources) > 1}
            if duplicates:
                issues.append(QualityIssue(
                    issue_type=QualityIssueType.DUPLICATE_DATA,
                    dimension=QualityDimension.UNIQUENESS,
                    severity="low",
                    description=f"Duplicate prices detected for {symbol}",
                    affected_records=len(duplicates),
                    confidence=0.8,
                    metadata={
                        "symbol": symbol,
                        "duplicates": {f"{price}@{timestamp}": sources
                                     for (price, timestamp), sources in duplicates.items()}
                    }
                ))

        uniqueness_score = max(0, 100 - (duplicate_count / max(1, total_records) * 100))
        return uniqueness_score, issues

    def _assess_temporal_consistency(self, historical_data: Dict[str, List[PriceData]]) -> Tuple[float, List[QualityIssue]]:
        """Assess temporal consistency in historical data"""
        issues = []
        consistency_scores = []

        for source, data_list in historical_data.items():
            if len(data_list) < 2:
                continue

            # Sort by timestamp
            sorted_data = sorted(data_list, key=lambda x: x.timestamp)

            # Check for temporal anomalies
            temporal_issues = 0
            for i in range(1, len(sorted_data)):
                prev_data = sorted_data[i-1]
                curr_data = sorted_data[i]

                # Check for backwards time
                if curr_data.timestamp <= prev_data.timestamp:
                    temporal_issues += 1

                # Check for unreasonable time gaps
                time_gap = (curr_data.timestamp - prev_data.timestamp).total_seconds()
                if time_gap > 86400 * 7:  # More than a week gap
                    temporal_issues += 1

            consistency_score = max(0, 100 - (temporal_issues / len(sorted_data) * 100))
            consistency_scores.append(consistency_score)

            if temporal_issues > 0:
                issues.append(QualityIssue(
                    issue_type=QualityIssueType.TEMPORAL_INCONSISTENCY,
                    dimension=QualityDimension.TIMELINESS,
                    severity="medium" if temporal_issues < len(sorted_data) * 0.1 else "high",
                    description=f"Temporal inconsistencies in {source}: {temporal_issues} issues",
                    affected_records=temporal_issues,
                    confidence=0.9,
                    metadata={
                        "source": source,
                        "total_records": len(sorted_data),
                        "temporal_issues": temporal_issues
                    }
                ))

        overall_score = statistics.mean(consistency_scores) if consistency_scores else 85.0
        return overall_score, issues

    def _assess_data_gaps(self, historical_data: Dict[str, List[PriceData]]) -> Tuple[float, List[QualityIssue]]:
        """Assess data gaps in historical data"""
        issues = []
        gap_scores = []

        for source, data_list in historical_data.items():
            if len(data_list) < 2:
                continue

            # Sort by timestamp
            sorted_data = sorted(data_list, key=lambda x: x.timestamp)

            # Expected interval (assume daily data if gaps > 1 hour)
            total_time_span = (sorted_data[-1].timestamp - sorted_data[0].timestamp).total_seconds()
            expected_intervals = max(1, total_time_span / 86400)  # Daily intervals
            actual_intervals = len(sorted_data) - 1

            completeness = min(100, (actual_intervals / expected_intervals) * 100) if expected_intervals > 0 else 100
            gap_scores.append(completeness)

            if completeness < 90:
                missing_intervals = expected_intervals - actual_intervals
                issues.append(QualityIssue(
                    issue_type=QualityIssueType.MISSING_DATA,
                    dimension=QualityDimension.COMPLETENESS,
                    severity="high" if completeness < 70 else "medium",
                    description=f"Data gaps in {source}: {missing_intervals:.0f} missing intervals",
                    affected_records=int(missing_intervals),
                    confidence=0.8,
                    metadata={
                        "source": source,
                        "completeness_percentage": completeness,
                        "expected_intervals": expected_intervals,
                        "actual_intervals": actual_intervals
                    }
                ))

        overall_score = statistics.mean(gap_scores) if gap_scores else 85.0
        return overall_score, issues

    def _assess_historical_value_consistency(self, historical_data: Dict[str, List[PriceData]]) -> Tuple[float, List[QualityIssue]]:
        """Assess value consistency in historical data"""
        issues = []
        consistency_scores = []

        for source, data_list in historical_data.items():
            if len(data_list) < 10:
                continue

            # Check for impossible values
            prices = [data.close_price for data in data_list if data.close_price > 0]
            if not prices:
                continue

            # Statistical analysis
            mean_price = statistics.mean(prices)
            std_price = statistics.stdev(prices) if len(prices) > 1 else 0

            outliers = 0
            for price in prices:
                if std_price > 0:
                    z_score = abs(price - mean_price) / std_price
                    if z_score > 5:  # More than 5 standard deviations
                        outliers += 1

            consistency_score = max(0, 100 - (outliers / len(prices) * 100))
            consistency_scores.append(consistency_score)

            if outliers > 0:
                issues.append(QualityIssue(
                    issue_type=QualityIssueType.OUT_OF_RANGE,
                    dimension=QualityDimension.CONSISTENCY,
                    severity="medium",
                    description=f"Statistical outliers in {source}: {outliers} outliers detected",
                    affected_records=outliers,
                    confidence=0.7,
                    metadata={
                        "source": source,
                        "outlier_count": outliers,
                        "total_records": len(prices),
                        "mean_price": mean_price,
                        "std_deviation": std_price
                    }
                ))

        overall_score = statistics.mean(consistency_scores) if consistency_scores else 85.0
        return overall_score, issues

    def _generate_recommendations(self, dimension_scores: Dict[QualityDimension, float],
                                issues: List[QualityIssue]) -> List[str]:
        """Generate actionable recommendations based on quality assessment"""
        recommendations = []

        # Dimension-based recommendations
        for dimension, score in dimension_scores.items():
            if score < 60:
                if dimension == QualityDimension.COMPLETENESS:
                    recommendations.append("Add more data sources to improve coverage")
                elif dimension == QualityDimension.ACCURACY:
                    recommendations.append("Implement cross-source validation to detect inaccurate data")
                elif dimension == QualityDimension.CONSISTENCY:
                    recommendations.append("Review data source configurations for consistency")
                elif dimension == QualityDimension.TIMELINESS:
                    recommendations.append("Optimize data ingestion pipelines for faster delivery")
                elif dimension == QualityDimension.VALIDITY:
                    recommendations.append("Strengthen data validation rules and constraints")

        # Issue-based recommendations
        critical_issues = [issue for issue in issues if issue.severity == "critical"]
        if critical_issues:
            recommendations.append("Address critical data quality issues immediately")

        high_issues = [issue for issue in issues if issue.severity == "high"]
        if len(high_issues) > 5:
            recommendations.append("Implement automated monitoring for high-severity issues")

        # Generic recommendations
        if len(issues) > 10:
            recommendations.append("Consider implementing a data quality monitoring dashboard")

        return recommendations

    def _create_empty_report(self, assessment_start: datetime) -> QualityReport:
        """Create an empty quality report when no data is available"""
        return QualityReport(
            overall_score=0.0,
            dimension_scores={dim: 0.0 for dim in QualityDimension},
            issues=[],
            recommendations=["No data available for quality assessment"],
            assessment_period=(assessment_start, datetime.now()),
            total_records_analyzed=0,
            metadata={"error": "No data provided for assessment"}
        )

    def _update_quality_tracking(self, report: QualityReport) -> None:
        """Update quality tracking and history"""
        self.quality_history.append(report)

        # Keep last 100 reports
        if len(self.quality_history) > 100:
            self.quality_history = self.quality_history[-100:]

        # Update issue tracking
        for issue in report.issues:
            issue_key = f"{issue.issue_type.value}_{issue.dimension.value}"
            if issue_key in self.issue_tracking:
                self.issue_tracking[issue_key].last_detected = issue.last_detected
                self.issue_tracking[issue_key].affected_records += issue.affected_records
            else:
                self.issue_tracking[issue_key] = issue

    def get_quality_trends(self, days: int = 7) -> Dict[str, Any]:
        """Get quality trends over time"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_reports = [
            report for report in self.quality_history
            if report.generated_at >= cutoff_date
        ]

        if not recent_reports:
            return {"error": "No recent quality reports available"}

        # Calculate trends
        scores_over_time = []
        for report in recent_reports:
            scores_over_time.append({
                "timestamp": report.generated_at,
                "overall_score": report.overall_score,
                "dimension_scores": report.dimension_scores
            })

        return {
            "period_days": days,
            "total_reports": len(recent_reports),
            "scores_over_time": scores_over_time,
            "average_score": statistics.mean(r.overall_score for r in recent_reports),
            "score_trend": "improving" if len(recent_reports) >= 2 and
                          recent_reports[-1].overall_score > recent_reports[0].overall_score else "stable",
            "persistent_issues": len(self.issue_tracking)
        }