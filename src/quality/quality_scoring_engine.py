"""Advanced Quality Scoring Engine

Implements a comprehensive A-F grading system for market data quality with
multi-dimensional analysis, weighted scoring, and actionable recommendations.
"""

import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

from ..validation.data_quality_assessor import (
    DataQualityAssessor, QualityReport, QualityDimension, QualityIssue
)
from ..data_sources.base import CurrentPrice, PriceData

logger = logging.getLogger(__name__)


class QualityGrade(Enum):
    """Quality grade levels"""
    A_PLUS = "A+"      # 97-100: Exceptional quality
    A = "A"            # 93-96: Excellent quality
    A_MINUS = "A-"     # 90-92: Very good quality
    B_PLUS = "B+"      # 87-89: Good quality
    B = "B"            # 83-86: Above average quality
    B_MINUS = "B-"     # 80-82: Average quality
    C_PLUS = "C+"      # 77-79: Below average quality
    C = "C"            # 73-76: Poor quality
    C_MINUS = "C-"     # 70-72: Very poor quality
    D_PLUS = "D+"      # 67-69: Unacceptable quality
    D = "D"            # 60-66: Severely deficient quality
    F = "F"            # 0-59: Failed quality standards


class SeverityLevel(Enum):
    """Issue severity levels for scoring"""
    CRITICAL = "critical"      # -20 to -50 points
    HIGH = "high"             # -10 to -20 points
    MEDIUM = "medium"         # -5 to -10 points
    LOW = "low"              # -1 to -5 points
    INFO = "info"            # 0 points


@dataclass
class DimensionWeight:
    """Weight configuration for quality dimensions"""
    completeness: float = 0.25      # 25% - Data presence and coverage
    accuracy: float = 0.20          # 20% - Correctness of values
    consistency: float = 0.15       # 15% - Internal consistency
    timeliness: float = 0.20        # 20% - Data freshness
    validity: float = 0.15          # 15% - Business rule compliance
    uniqueness: float = 0.05        # 5% - Duplicate detection

    def __post_init__(self):
        """Validate weights sum to 1.0"""
        total = (self.completeness + self.accuracy + self.consistency +
                self.timeliness + self.validity + self.uniqueness)
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Dimension weights must sum to 1.0, got {total}")


@dataclass
class ScoringConfig:
    """Configuration for the quality scoring engine"""
    dimension_weights: DimensionWeight = field(default_factory=DimensionWeight)

    # Grade thresholds
    grade_thresholds: Dict[QualityGrade, float] = field(default_factory=lambda: {
        QualityGrade.A_PLUS: 97.0,
        QualityGrade.A: 93.0,
        QualityGrade.A_MINUS: 90.0,
        QualityGrade.B_PLUS: 87.0,
        QualityGrade.B: 83.0,
        QualityGrade.B_MINUS: 80.0,
        QualityGrade.C_PLUS: 77.0,
        QualityGrade.C: 73.0,
        QualityGrade.C_MINUS: 70.0,
        QualityGrade.D_PLUS: 67.0,
        QualityGrade.D: 60.0,
        QualityGrade.F: 0.0
    })

    # Severity impact on scores
    severity_penalties: Dict[SeverityLevel, Tuple[float, float]] = field(default_factory=lambda: {
        SeverityLevel.CRITICAL: (20.0, 50.0),    # Min, Max penalty
        SeverityLevel.HIGH: (10.0, 20.0),
        SeverityLevel.MEDIUM: (5.0, 10.0),
        SeverityLevel.LOW: (1.0, 5.0),
        SeverityLevel.INFO: (0.0, 0.0)
    })

    # Bonus points for excellence
    excellence_bonuses: Dict[str, float] = field(default_factory=lambda: {
        "perfect_completeness": 2.0,
        "perfect_accuracy": 2.0,
        "exceptional_timeliness": 1.0,
        "zero_duplicates": 1.0,
        "consistent_quality": 1.0
    })

    # Time-based quality decay
    enable_time_decay: bool = True
    max_data_age_hours: float = 24.0
    time_decay_factor: float = 0.1      # Penalty per hour


@dataclass
class DetailedScore:
    """Detailed scoring breakdown for a quality dimension"""
    base_score: float                    # Base score (0-100)
    penalties: List[Tuple[str, float]]   # List of (reason, penalty) tuples
    bonuses: List[Tuple[str, float]]     # List of (reason, bonus) tuples
    final_score: float                   # Final score after adjustments
    weight: float                        # Dimension weight
    weighted_contribution: float        # Final score * weight

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "base_score": self.base_score,
            "penalties": self.penalties,
            "bonuses": self.bonuses,
            "final_score": self.final_score,
            "weight": self.weight,
            "weighted_contribution": self.weighted_contribution
        }


@dataclass
class QualityScoreCard:
    """Comprehensive quality score card with detailed breakdown"""
    symbol: str
    source: str
    overall_grade: QualityGrade
    overall_score: float

    # Detailed dimension scores
    dimension_scores: Dict[QualityDimension, DetailedScore]

    # Summary metrics
    total_issues: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int

    # Recommendations
    priority_recommendations: List[str]
    improvement_suggestions: List[str]

    # Metadata
    assessment_period: Tuple[datetime, datetime]
    data_points_analyzed: int
    confidence_level: float = 0.0
    generated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "symbol": self.symbol,
            "source": self.source,
            "overall_grade": self.overall_grade.value,
            "overall_score": self.overall_score,
            "dimension_scores": {
                dim.value: score.to_dict()
                for dim, score in self.dimension_scores.items()
            },
            "total_issues": self.total_issues,
            "critical_issues": self.critical_issues,
            "high_issues": self.high_issues,
            "medium_issues": self.medium_issues,
            "low_issues": self.low_issues,
            "priority_recommendations": self.priority_recommendations,
            "improvement_suggestions": self.improvement_suggestions,
            "assessment_period": {
                "start": self.assessment_period[0].isoformat(),
                "end": self.assessment_period[1].isoformat()
            },
            "data_points_analyzed": self.data_points_analyzed,
            "confidence_level": self.confidence_level,
            "generated_at": self.generated_at.isoformat(),
            "metadata": self.metadata
        }


class QualityScoringEngine:
    """Advanced quality scoring engine with A-F grading system

    Features:
    - Multi-dimensional weighted scoring
    - Intelligent penalty and bonus system
    - Time-based quality decay
    - Actionable recommendations
    - Trend analysis and improvement tracking
    """

    def __init__(self, config: Optional[ScoringConfig] = None):
        """Initialize the quality scoring engine

        Args:
            config: Scoring configuration, uses defaults if None
        """
        self.config = config or ScoringConfig()
        self.quality_assessor = DataQualityAssessor()

        # Historical scoring data
        self.score_history: Dict[str, Dict[str, List[QualityScoreCard]]] = {}  # symbol -> source -> scores
        self.trend_cache: Dict[str, Dict[str, Any]] = {}  # symbol -> source -> trend data

        # Performance tracking
        self.scoring_metrics = {
            "total_assessments": 0,
            "average_processing_time": 0.0,
            "grade_distribution": {grade.value: 0 for grade in QualityGrade}
        }

    def generate_score_card(self, symbol: str, source: str,
                          current_prices: List[CurrentPrice]) -> QualityScoreCard:
        """Generate comprehensive quality score card

        Args:
            symbol: Stock symbol
            source: Data source name
            current_prices: List of current price data

        Returns:
            QualityScoreCard with detailed scoring breakdown
        """
        start_time = datetime.now()

        try:
            # Convert current prices to format expected by assessor
            # DataQualityAssessor expects {symbol: {source: CurrentPrice}}
            assessment_data = {symbol: {source: current_prices[0] if current_prices else None}}
            if len(current_prices) > 1:
                # If multiple prices, create additional entries
                for i, price in enumerate(current_prices[1:], 1):
                    assessment_data[symbol][f"{source}_alt_{i}"] = price

            # Get base quality report
            quality_report = self.quality_assessor.assess_current_prices(assessment_data)

            # Calculate detailed dimension scores
            dimension_scores = self._calculate_dimension_scores(quality_report, current_prices)

            # Calculate overall score
            overall_score = self._calculate_overall_score(dimension_scores)

            # Determine grade
            overall_grade = self._determine_grade(overall_score)

            # Count issues by severity
            issue_counts = self._count_issues_by_severity(quality_report.issues)

            # Generate recommendations
            priority_recs, improvement_suggestions = self._generate_recommendations(
                dimension_scores, quality_report.issues
            )

            # Calculate confidence level
            confidence_level = self._calculate_confidence_level(
                current_prices, quality_report, dimension_scores
            )

            # Create score card
            score_card = QualityScoreCard(
                symbol=symbol,
                source=source,
                overall_grade=overall_grade,
                overall_score=overall_score,
                dimension_scores=dimension_scores,
                total_issues=len(quality_report.issues),
                critical_issues=issue_counts.get(SeverityLevel.CRITICAL, 0),
                high_issues=issue_counts.get(SeverityLevel.HIGH, 0),
                medium_issues=issue_counts.get(SeverityLevel.MEDIUM, 0),
                low_issues=issue_counts.get(SeverityLevel.LOW, 0),
                priority_recommendations=priority_recs,
                improvement_suggestions=improvement_suggestions,
                assessment_period=(
                    min(p.timestamp for p in current_prices) if current_prices else start_time,
                    max(p.timestamp for p in current_prices) if current_prices else start_time
                ),
                data_points_analyzed=len(current_prices),
                confidence_level=confidence_level
            )

            # Update history and metrics
            self._update_score_history(symbol, source, score_card)
            self._update_scoring_metrics(score_card, datetime.now() - start_time)

            return score_card

        except Exception as e:
            logger.error(f"Error generating score card for {symbol}/{source}: {e}")
            # Return minimal failing score card
            return self._create_error_score_card(symbol, source, str(e))

    def _calculate_dimension_scores(self, quality_report: QualityReport,
                                   current_prices: List[CurrentPrice]) -> Dict[QualityDimension, DetailedScore]:
        """Calculate detailed scores for each quality dimension"""
        dimension_scores = {}
        weights = self.config.dimension_weights

        for dimension in QualityDimension:
            # Get base score from quality report
            base_score = quality_report.dimension_scores.get(dimension, 0.0) * 100

            # Calculate penalties
            penalties = self._calculate_dimension_penalties(dimension, quality_report.issues)

            # Calculate bonuses
            bonuses = self._calculate_dimension_bonuses(dimension, base_score, current_prices)

            # Apply time decay if enabled
            if self.config.enable_time_decay:
                time_penalty = self._calculate_time_decay_penalty(current_prices)
                if time_penalty > 0:
                    penalties.append(("data_age_penalty", time_penalty))

            # Calculate final score
            total_penalties = sum(penalty for _, penalty in penalties)
            total_bonuses = sum(bonus for _, bonus in bonuses)
            final_score = max(0.0, min(100.0, base_score - total_penalties + total_bonuses))

            # Get dimension weight
            weight = getattr(weights, dimension.value)

            dimension_scores[dimension] = DetailedScore(
                base_score=base_score,
                penalties=penalties,
                bonuses=bonuses,
                final_score=final_score,
                weight=weight,
                weighted_contribution=final_score * weight
            )

        return dimension_scores

    def _calculate_dimension_penalties(self, dimension: QualityDimension,
                                     issues: List[QualityIssue]) -> List[Tuple[str, float]]:
        """Calculate penalties for a specific dimension"""
        penalties = []

        # Filter issues for this dimension
        dimension_issues = [issue for issue in issues if issue.dimension == dimension]

        for issue in dimension_issues:
            # Map issue severity to penalty
            severity = self._map_severity_to_enum(issue.severity)
            min_penalty, max_penalty = self.config.severity_penalties[severity]

            # Calculate penalty based on issue impact
            penalty = self._calculate_issue_penalty(issue, min_penalty, max_penalty)
            penalties.append((issue.description, penalty))

        return penalties

    def _calculate_dimension_bonuses(self, dimension: QualityDimension,
                                   base_score: float, current_prices: List[CurrentPrice]) -> List[Tuple[str, float]]:
        """Calculate bonus points for exceptional performance"""
        bonuses = []

        # Perfect scores get bonuses
        if base_score >= 99.5:
            if dimension == QualityDimension.COMPLETENESS:
                bonuses.append(("perfect_completeness", self.config.excellence_bonuses["perfect_completeness"]))
            elif dimension == QualityDimension.ACCURACY:
                bonuses.append(("perfect_accuracy", self.config.excellence_bonuses["perfect_accuracy"]))

        # Timeliness bonuses for very fresh data
        if dimension == QualityDimension.TIMELINESS and current_prices:
            latest_age = (datetime.now() - max(p.timestamp for p in current_prices)).total_seconds()
            if latest_age < 30:  # Less than 30 seconds old
                bonuses.append(("exceptional_timeliness", self.config.excellence_bonuses["exceptional_timeliness"]))

        # Uniqueness bonus for zero duplicates
        if dimension == QualityDimension.UNIQUENESS and base_score >= 99.9:
            bonuses.append(("zero_duplicates", self.config.excellence_bonuses["zero_duplicates"]))

        return bonuses

    def _calculate_time_decay_penalty(self, current_prices: List[CurrentPrice]) -> float:
        """Calculate penalty based on data age"""
        if not current_prices:
            return 0.0

        # Find oldest data point
        oldest_timestamp = min(p.timestamp for p in current_prices)
        age_hours = (datetime.now() - oldest_timestamp).total_seconds() / 3600

        if age_hours > self.config.max_data_age_hours:
            # Penalty increases with age beyond threshold
            excess_hours = age_hours - self.config.max_data_age_hours
            return min(20.0, excess_hours * self.config.time_decay_factor)

        return 0.0

    def _calculate_issue_penalty(self, issue: QualityIssue, min_penalty: float, max_penalty: float) -> float:
        """Calculate penalty for a specific issue"""
        # Use affected records and confidence to scale penalty
        base_penalty = min_penalty

        # Scale by affected records (if available)
        if hasattr(issue, 'affected_records') and issue.affected_records > 1:
            # More affected records = higher penalty
            scale_factor = min(2.0, 1.0 + (issue.affected_records - 1) * 0.1)
            base_penalty *= scale_factor

        # Scale by confidence (if available)
        if hasattr(issue, 'confidence') and issue.confidence > 0:
            # Higher confidence in issue = higher penalty
            base_penalty *= issue.confidence

        return min(max_penalty, base_penalty)

    def _calculate_overall_score(self, dimension_scores: Dict[QualityDimension, DetailedScore]) -> float:
        """Calculate weighted overall quality score"""
        return sum(score.weighted_contribution for score in dimension_scores.values())

    def _determine_grade(self, score: float) -> QualityGrade:
        """Determine letter grade from numerical score"""
        for grade, threshold in self.config.grade_thresholds.items():
            if score >= threshold:
                return grade
        return QualityGrade.F

    def _count_issues_by_severity(self, issues: List[QualityIssue]) -> Dict[SeverityLevel, int]:
        """Count issues by severity level"""
        counts = {}
        for issue in issues:
            severity = self._map_severity_to_enum(issue.severity)
            counts[severity] = counts.get(severity, 0) + 1
        return counts

    def _map_severity_to_enum(self, severity_str: str) -> SeverityLevel:
        """Map string severity to enum"""
        severity_mapping = {
            "critical": SeverityLevel.CRITICAL,
            "high": SeverityLevel.HIGH,
            "medium": SeverityLevel.MEDIUM,
            "low": SeverityLevel.LOW,
            "info": SeverityLevel.INFO
        }
        return severity_mapping.get(severity_str.lower(), SeverityLevel.MEDIUM)

    def _generate_recommendations(self, dimension_scores: Dict[QualityDimension, DetailedScore],
                                issues: List[QualityIssue]) -> Tuple[List[str], List[str]]:
        """Generate priority recommendations and improvement suggestions"""
        priority_recs = []
        improvement_suggestions = []

        # Priority recommendations for critical/high severity issues
        critical_issues = [issue for issue in issues if self._map_severity_to_enum(issue.severity) in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]]
        for issue in critical_issues[:3]:  # Top 3 critical issues
            priority_recs.append(f"URGENT: {issue.description}")

        # Improvement suggestions based on lowest scoring dimensions
        sorted_dimensions = sorted(dimension_scores.items(), key=lambda x: x[1].final_score)

        for dimension, score in sorted_dimensions[:3]:  # Bottom 3 dimensions
            if score.final_score < 80.0:  # Only if below B- grade
                improvement_suggestions.append(self._get_dimension_improvement_suggestion(dimension, score))

        # Add specific recommendations based on score patterns
        self._add_pattern_based_recommendations(dimension_scores, improvement_suggestions)

        return priority_recs, improvement_suggestions

    def _get_dimension_improvement_suggestion(self, dimension: QualityDimension,
                                            score: DetailedScore) -> str:
        """Get improvement suggestion for a specific dimension"""
        suggestions = {
            QualityDimension.COMPLETENESS: "Implement data completeness validation and automated gap detection",
            QualityDimension.ACCURACY: "Add cross-source validation and outlier detection algorithms",
            QualityDimension.CONSISTENCY: "Implement temporal consistency checks and drift detection",
            QualityDimension.TIMELINESS: "Optimize data ingestion pipeline for reduced latency",
            QualityDimension.VALIDITY: "Strengthen business rule validation and format checking",
            QualityDimension.UNIQUENESS: "Add duplicate detection and deduplication processes"
        }

        base_suggestion = suggestions.get(dimension, "Review and improve data quality processes")
        return f"{dimension.value.title()}: {base_suggestion} (Score: {score.final_score:.1f})"

    def _add_pattern_based_recommendations(self, dimension_scores: Dict[QualityDimension, DetailedScore],
                                         suggestions: List[str]) -> None:
        """Add recommendations based on scoring patterns"""
        scores = [score.final_score for score in dimension_scores.values()]

        # Check for consistently low scores
        if all(score < 70 for score in scores):
            suggestions.append("Overall quality is poor - consider comprehensive data quality overhaul")

        # Check for high variance in scores
        if len(scores) > 1 and statistics.stdev(scores) > 20:
            suggestions.append("Inconsistent quality across dimensions - focus on standardization")

        # Check for specific patterns
        timeliness_score = dimension_scores.get(QualityDimension.TIMELINESS, None)
        if timeliness_score and timeliness_score.final_score < 60:
            suggestions.append("Critical timeliness issues detected - review data pipeline latency")

    def _calculate_confidence_level(self, current_prices: List[CurrentPrice],
                                  quality_report: QualityReport,
                                  dimension_scores: Dict[QualityDimension, DetailedScore]) -> float:
        """Calculate confidence level in the scoring assessment"""
        confidence_factors = []

        # Data volume factor
        if len(current_prices) >= 10:
            confidence_factors.append(0.95)
        elif len(current_prices) >= 5:
            confidence_factors.append(0.80)
        else:
            confidence_factors.append(0.60)

        # Score consistency factor
        scores = [score.final_score for score in dimension_scores.values()]
        if len(scores) > 1:
            score_variance = statistics.stdev(scores)
            if score_variance < 10:
                confidence_factors.append(0.95)
            elif score_variance < 20:
                confidence_factors.append(0.80)
            else:
                confidence_factors.append(0.65)

        # Data freshness factor
        if current_prices:
            latest_age = (datetime.now() - max(p.timestamp for p in current_prices)).total_seconds()
            if latest_age < 300:  # 5 minutes
                confidence_factors.append(0.95)
            elif latest_age < 1800:  # 30 minutes
                confidence_factors.append(0.85)
            else:
                confidence_factors.append(0.70)

        return statistics.mean(confidence_factors) if confidence_factors else 0.5

    def _update_score_history(self, symbol: str, source: str, score_card: QualityScoreCard) -> None:
        """Update scoring history for trend analysis"""
        if symbol not in self.score_history:
            self.score_history[symbol] = {}
        if source not in self.score_history[symbol]:
            self.score_history[symbol][source] = []

        # Add new score card
        self.score_history[symbol][source].append(score_card)

        # Keep only recent history (last 100 assessments)
        if len(self.score_history[symbol][source]) > 100:
            self.score_history[symbol][source] = self.score_history[symbol][source][-100:]

    def _update_scoring_metrics(self, score_card: QualityScoreCard, processing_time: timedelta) -> None:
        """Update scoring performance metrics"""
        self.scoring_metrics["total_assessments"] += 1

        # Update average processing time
        total_time = (self.scoring_metrics["average_processing_time"] *
                     (self.scoring_metrics["total_assessments"] - 1) +
                     processing_time.total_seconds())
        self.scoring_metrics["average_processing_time"] = total_time / self.scoring_metrics["total_assessments"]

        # Update grade distribution
        self.scoring_metrics["grade_distribution"][score_card.overall_grade.value] += 1

    def _create_error_score_card(self, symbol: str, source: str, error: str) -> QualityScoreCard:
        """Create a minimal score card for error cases"""
        return QualityScoreCard(
            symbol=symbol,
            source=source,
            overall_grade=QualityGrade.F,
            overall_score=0.0,
            dimension_scores={},
            total_issues=1,
            critical_issues=1,
            high_issues=0,
            medium_issues=0,
            low_issues=0,
            priority_recommendations=[f"CRITICAL: Scoring error - {error}"],
            improvement_suggestions=["Review data quality assessment configuration"],
            assessment_period=(datetime.now(), datetime.now()),
            data_points_analyzed=0,
            confidence_level=0.0,
            metadata={"error": error}
        )

    def get_quality_trends(self, symbol: str, source: str,
                          period_days: int = 7) -> Dict[str, Any]:
        """Get quality trends for a symbol/source combination"""
        if (symbol not in self.score_history or
            source not in self.score_history[symbol]):
            return {"error": "No historical data available"}

        history = self.score_history[symbol][source]
        cutoff_date = datetime.now() - timedelta(days=period_days)
        recent_scores = [sc for sc in history if sc.generated_at >= cutoff_date]

        if not recent_scores:
            return {"error": "No recent data available"}

        # Calculate trends
        scores = [sc.overall_score for sc in recent_scores]
        grades = [sc.overall_grade.value for sc in recent_scores]

        return {
            "symbol": symbol,
            "source": source,
            "period_days": period_days,
            "total_assessments": len(recent_scores),
            "current_score": scores[-1] if scores else 0,
            "current_grade": grades[-1] if grades else "F",
            "average_score": statistics.mean(scores) if scores else 0,
            "score_trend": "improving" if len(scores) >= 2 and scores[-1] > scores[0] else "declining",
            "score_volatility": statistics.stdev(scores) if len(scores) > 1 else 0,
            "grade_distribution": {grade: grades.count(grade) for grade in set(grades)},
            "best_score": max(scores) if scores else 0,
            "worst_score": min(scores) if scores else 0
        }

    def get_scoring_summary(self) -> Dict[str, Any]:
        """Get overall scoring system performance summary"""
        return {
            "total_assessments": self.scoring_metrics["total_assessments"],
            "average_processing_time_ms": self.scoring_metrics["average_processing_time"] * 1000,
            "grade_distribution": self.scoring_metrics["grade_distribution"],
            "symbols_tracked": len(self.score_history),
            "total_sources": sum(len(sources) for sources in self.score_history.values()),
            "configuration": {
                "dimension_weights": {
                    "completeness": self.config.dimension_weights.completeness,
                    "accuracy": self.config.dimension_weights.accuracy,
                    "consistency": self.config.dimension_weights.consistency,
                    "timeliness": self.config.dimension_weights.timeliness,
                    "validity": self.config.dimension_weights.validity,
                    "uniqueness": self.config.dimension_weights.uniqueness
                },
                "time_decay_enabled": self.config.enable_time_decay,
                "max_data_age_hours": self.config.max_data_age_hours
            }
        }

    def update_configuration(self, config: ScoringConfig) -> None:
        """Update scoring configuration"""
        self.config = config
        # Clear trend cache as weights may have changed
        self.trend_cache.clear()
        logger.info("Quality scoring configuration updated")

    def reset_history(self, symbol: Optional[str] = None, source: Optional[str] = None) -> None:
        """Reset scoring history for debugging or cleanup"""
        if symbol and source:
            if symbol in self.score_history and source in self.score_history[symbol]:
                self.score_history[symbol][source].clear()
        elif symbol:
            if symbol in self.score_history:
                self.score_history[symbol].clear()
        else:
            self.score_history.clear()
            self.trend_cache.clear()
            self.scoring_metrics = {
                "total_assessments": 0,
                "average_processing_time": 0.0,
                "grade_distribution": {grade.value: 0 for grade in QualityGrade}
            }