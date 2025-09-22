"""Quality Dashboard

Interactive dashboard for comprehensive quality monitoring, trend analysis,
and performance tracking across all data sources and symbols.
"""

import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import logging

from .quality_scoring_engine import (
    QualityScoringEngine, QualityScoreCard, QualityGrade, DimensionWeight
)
from ..validation.data_quality_assessor import QualityDimension
from ..data_sources.base import CurrentPrice

logger = logging.getLogger(__name__)


@dataclass
class DashboardMetrics:
    """Dashboard summary metrics"""
    total_symbols: int
    total_sources: int
    total_assessments: int
    current_average_grade: str
    current_average_score: float

    # Grade distribution
    grade_distribution: Dict[str, int] = field(default_factory=dict)

    # Quality trends
    trend_direction: str = "stable"  # improving, declining, stable
    trend_strength: float = 0.0      # 0.0 to 1.0

    # Critical alerts
    failing_sources: List[Tuple[str, str, str]] = field(default_factory=list)  # (symbol, source, grade)
    degrading_quality: List[Tuple[str, str, float]] = field(default_factory=list)  # (symbol, source, decline)

    # Performance metrics
    assessment_frequency: float = 0.0  # assessments per hour
    average_processing_time_ms: float = 0.0

    # Data freshness
    freshest_data_age_seconds: float = 0.0
    oldest_data_age_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "total_symbols": self.total_symbols,
            "total_sources": self.total_sources,
            "total_assessments": self.total_assessments,
            "current_average_grade": self.current_average_grade,
            "current_average_score": self.current_average_score,
            "grade_distribution": self.grade_distribution,
            "trend_direction": self.trend_direction,
            "trend_strength": self.trend_strength,
            "failing_sources": [
                {"symbol": s, "source": src, "grade": g}
                for s, src, g in self.failing_sources
            ],
            "degrading_quality": [
                {"symbol": s, "source": src, "decline_percentage": d}
                for s, src, d in self.degrading_quality
            ],
            "assessment_frequency": self.assessment_frequency,
            "average_processing_time_ms": self.average_processing_time_ms,
            "freshest_data_age_seconds": self.freshest_data_age_seconds,
            "oldest_data_age_seconds": self.oldest_data_age_seconds
        }


@dataclass
class SourceQualityProfile:
    """Quality profile for a specific source"""
    source_name: str
    symbols_tracked: List[str]
    current_average_grade: QualityGrade
    current_average_score: float

    # Historical performance
    assessments_count: int
    best_score: float
    worst_score: float
    score_variance: float

    # Dimension strengths and weaknesses
    strongest_dimension: str
    weakest_dimension: str
    dimension_scores: Dict[str, float]

    # Issue patterns
    common_issues: List[Tuple[str, int]]  # (issue_type, frequency)
    critical_issues_count: int

    # Recommendations
    improvement_priority: str  # high, medium, low
    recommended_actions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "source_name": self.source_name,
            "symbols_tracked": self.symbols_tracked,
            "current_average_grade": self.current_average_grade.value,
            "current_average_score": self.current_average_score,
            "assessments_count": self.assessments_count,
            "best_score": self.best_score,
            "worst_score": self.worst_score,
            "score_variance": self.score_variance,
            "strongest_dimension": self.strongest_dimension,
            "weakest_dimension": self.weakest_dimension,
            "dimension_scores": self.dimension_scores,
            "common_issues": [
                {"issue_type": issue, "frequency": freq}
                for issue, freq in self.common_issues
            ],
            "critical_issues_count": self.critical_issues_count,
            "improvement_priority": self.improvement_priority,
            "recommended_actions": self.recommended_actions
        }


class QualityDashboard:
    """Comprehensive quality monitoring dashboard

    Features:
    - Real-time quality metrics and trends
    - Source-specific quality profiles
    - Alert management and prioritization
    - Performance analytics and optimization insights
    - Interactive quality improvement recommendations
    """

    def __init__(self, scoring_engine: QualityScoringEngine):
        """Initialize the quality dashboard

        Args:
            scoring_engine: Quality scoring engine instance
        """
        self.scoring_engine = scoring_engine

        # Alert thresholds
        self.alert_thresholds = {
            "critical_grade": QualityGrade.D,      # Grade D or below triggers alert
            "degradation_threshold": 15.0,         # 15 point decline triggers alert
            "processing_time_threshold": 1000.0,   # 1 second processing time alert
            "data_age_threshold": 3600.0          # 1 hour data age alert
        }

        # Dashboard cache
        self.metrics_cache: Optional[DashboardMetrics] = None
        self.cache_timestamp: Optional[datetime] = None
        self.cache_ttl_seconds = 300  # 5 minutes

    def get_dashboard_metrics(self, refresh: bool = False) -> DashboardMetrics:
        """Get comprehensive dashboard metrics

        Args:
            refresh: Force refresh of cached metrics

        Returns:
            DashboardMetrics with current system status
        """
        # Check cache
        if (not refresh and self.metrics_cache and self.cache_timestamp and
            (datetime.now() - self.cache_timestamp).total_seconds() < self.cache_ttl_seconds):
            return self.metrics_cache

        # Calculate fresh metrics
        metrics = self._calculate_dashboard_metrics()

        # Update cache
        self.metrics_cache = metrics
        self.cache_timestamp = datetime.now()

        return metrics

    def _calculate_dashboard_metrics(self) -> DashboardMetrics:
        """Calculate comprehensive dashboard metrics"""
        scoring_summary = self.scoring_engine.get_scoring_summary()
        score_history = self.scoring_engine.score_history

        # Basic counts
        total_symbols = len(score_history)
        total_sources = sum(len(sources) for sources in score_history.values())
        total_assessments = scoring_summary["total_assessments"]

        # Current quality metrics
        current_scores = []
        current_grades = []
        all_score_cards = []

        for symbol, sources in score_history.items():
            for source, cards in sources.items():
                if cards:  # Has score cards
                    latest_card = cards[-1]
                    current_scores.append(latest_card.overall_score)
                    current_grades.append(latest_card.overall_grade.value)
                    all_score_cards.append(latest_card)

        avg_score = statistics.mean(current_scores) if current_scores else 0.0
        avg_grade = self._calculate_average_grade(current_grades)

        # Grade distribution
        grade_distribution = scoring_summary["grade_distribution"]

        # Quality trends
        trend_direction, trend_strength = self._calculate_quality_trends()

        # Critical alerts
        failing_sources = self._identify_failing_sources(all_score_cards)
        degrading_quality = self._identify_degrading_quality()

        # Performance metrics
        assessment_frequency = self._calculate_assessment_frequency()
        avg_processing_time = scoring_summary["average_processing_time_ms"]

        # Data freshness
        freshest_age, oldest_age = self._calculate_data_freshness(all_score_cards)

        return DashboardMetrics(
            total_symbols=total_symbols,
            total_sources=total_sources,
            total_assessments=total_assessments,
            current_average_grade=avg_grade,
            current_average_score=avg_score,
            grade_distribution=grade_distribution,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            failing_sources=failing_sources,
            degrading_quality=degrading_quality,
            assessment_frequency=assessment_frequency,
            average_processing_time_ms=avg_processing_time,
            freshest_data_age_seconds=freshest_age,
            oldest_data_age_seconds=oldest_age
        )

    def _calculate_average_grade(self, grades: List[str]) -> str:
        """Calculate average grade from list of grades"""
        if not grades:
            return "F"

        # Map grades to numerical values
        grade_values = {
            "A+": 4.3, "A": 4.0, "A-": 3.7,
            "B+": 3.3, "B": 3.0, "B-": 2.7,
            "C+": 2.3, "C": 2.0, "C-": 1.7,
            "D+": 1.3, "D": 1.0, "F": 0.0
        }

        total_value = sum(grade_values.get(grade, 0.0) for grade in grades)
        avg_value = total_value / len(grades)

        # Map back to grade
        for grade, value in sorted(grade_values.items(), key=lambda x: x[1], reverse=True):
            if avg_value >= value:
                return grade

        return "F"

    def _calculate_quality_trends(self) -> Tuple[str, float]:
        """Calculate overall quality trend direction and strength"""
        recent_scores = []
        for symbol, sources in self.scoring_engine.score_history.items():
            for source, cards in sources.items():
                if len(cards) >= 2:  # Need at least 2 data points
                    # Compare recent vs older scores
                    recent = statistics.mean([c.overall_score for c in cards[-5:]])
                    older = statistics.mean([c.overall_score for c in cards[-10:-5]] if len(cards) >= 10 else cards[:-5])
                    if older > 0:  # Avoid division by zero
                        change = (recent - older) / older
                        recent_scores.append(change)

        if not recent_scores:
            return "stable", 0.0

        avg_change = statistics.mean(recent_scores)
        change_magnitude = abs(avg_change)

        if avg_change > 0.05:  # 5% improvement
            return "improving", min(1.0, change_magnitude * 10)
        elif avg_change < -0.05:  # 5% decline
            return "declining", min(1.0, change_magnitude * 10)
        else:
            return "stable", change_magnitude * 10

    def _identify_failing_sources(self, score_cards: List[QualityScoreCard]) -> List[Tuple[str, str, str]]:
        """Identify sources with failing grades"""
        failing = []
        threshold_grade = self.alert_thresholds["critical_grade"]

        for card in score_cards:
            if self._compare_grades(card.overall_grade, threshold_grade) <= 0:
                failing.append((card.symbol, card.source, card.overall_grade.value))

        return failing

    def _compare_grades(self, grade1: QualityGrade, grade2: QualityGrade) -> int:
        """Compare two grades (-1: grade1 < grade2, 0: equal, 1: grade1 > grade2)"""
        grade_order = [
            QualityGrade.F, QualityGrade.D, QualityGrade.D_PLUS,
            QualityGrade.C_MINUS, QualityGrade.C, QualityGrade.C_PLUS,
            QualityGrade.B_MINUS, QualityGrade.B, QualityGrade.B_PLUS,
            QualityGrade.A_MINUS, QualityGrade.A, QualityGrade.A_PLUS
        ]

        idx1 = grade_order.index(grade1)
        idx2 = grade_order.index(grade2)

        if idx1 < idx2:
            return -1
        elif idx1 > idx2:
            return 1
        else:
            return 0

    def _identify_degrading_quality(self) -> List[Tuple[str, str, float]]:
        """Identify sources with degrading quality"""
        degrading = []
        threshold = self.alert_thresholds["degradation_threshold"]

        for symbol, sources in self.scoring_engine.score_history.items():
            for source, cards in sources.items():
                if len(cards) >= 5:  # Need sufficient history
                    recent_avg = statistics.mean([c.overall_score for c in cards[-3:]])
                    older_avg = statistics.mean([c.overall_score for c in cards[-8:-3]] if len(cards) >= 8 else cards[:-3])

                    decline = older_avg - recent_avg
                    if decline > threshold:
                        degrading.append((symbol, source, decline))

        return degrading

    def _calculate_assessment_frequency(self) -> float:
        """Calculate assessments per hour"""
        total_assessments = 0
        earliest_time = datetime.now()
        latest_time = datetime.now() - timedelta(days=365)

        for symbol, sources in self.scoring_engine.score_history.items():
            for source, cards in sources.items():
                total_assessments += len(cards)
                if cards:
                    earliest_time = min(earliest_time, min(c.generated_at for c in cards))
                    latest_time = max(latest_time, max(c.generated_at for c in cards))

        if total_assessments == 0 or earliest_time >= latest_time:
            return 0.0

        time_span_hours = (latest_time - earliest_time).total_seconds() / 3600
        return total_assessments / max(1.0, time_span_hours)

    def _calculate_data_freshness(self, score_cards: List[QualityScoreCard]) -> Tuple[float, float]:
        """Calculate data freshness metrics"""
        if not score_cards:
            return 0.0, 0.0

        now = datetime.now()
        freshest_age = float('inf')
        oldest_age = 0.0

        for card in score_cards:
            # Use end of assessment period as data timestamp
            data_age = (now - card.assessment_period[1]).total_seconds()
            freshest_age = min(freshest_age, data_age)
            oldest_age = max(oldest_age, data_age)

        return max(0.0, freshest_age), oldest_age

    def get_source_quality_profiles(self) -> Dict[str, SourceQualityProfile]:
        """Get quality profiles for all data sources"""
        profiles = {}

        # Group by source across all symbols
        source_data = defaultdict(list)
        for symbol, sources in self.scoring_engine.score_history.items():
            for source, cards in sources.items():
                if cards:  # Has data
                    source_data[source].extend([(symbol, card) for card in cards])

        for source_name, data in source_data.items():
            symbols = list(set(symbol for symbol, _ in data))
            cards = [card for _, card in data]

            if not cards:
                continue

            # Current metrics
            latest_cards = {}
            for symbol, card in data:
                if symbol not in latest_cards or card.generated_at > latest_cards[symbol].generated_at:
                    latest_cards[symbol] = card

            current_scores = [card.overall_score for card in latest_cards.values()]
            current_grades = [card.overall_grade for card in latest_cards.values()]

            avg_score = statistics.mean(current_scores) if current_scores else 0.0
            avg_grade = self._get_mode_grade(current_grades) if current_grades else QualityGrade.F

            # Historical metrics
            all_scores = [card.overall_score for card in cards]
            best_score = max(all_scores) if all_scores else 0.0
            worst_score = min(all_scores) if all_scores else 0.0
            score_variance = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0

            # Dimension analysis
            dimension_scores = self._analyze_source_dimensions(cards)
            strongest_dim = max(dimension_scores.items(), key=lambda x: x[1])[0] if dimension_scores else "unknown"
            weakest_dim = min(dimension_scores.items(), key=lambda x: x[1])[0] if dimension_scores else "unknown"

            # Issue analysis
            common_issues = self._analyze_common_issues(cards)
            critical_issues = sum(card.critical_issues for card in cards)

            # Generate recommendations
            improvement_priority, actions = self._generate_source_recommendations(
                avg_score, dimension_scores, common_issues, critical_issues
            )

            profiles[source_name] = SourceQualityProfile(
                source_name=source_name,
                symbols_tracked=symbols,
                current_average_grade=avg_grade,
                current_average_score=avg_score,
                assessments_count=len(cards),
                best_score=best_score,
                worst_score=worst_score,
                score_variance=score_variance,
                strongest_dimension=strongest_dim,
                weakest_dimension=weakest_dim,
                dimension_scores=dimension_scores,
                common_issues=common_issues,
                critical_issues_count=critical_issues,
                improvement_priority=improvement_priority,
                recommended_actions=actions
            )

        return profiles

    def _get_mode_grade(self, grades: List[QualityGrade]) -> QualityGrade:
        """Get the most common grade"""
        if not grades:
            return QualityGrade.F

        grade_counts = {}
        for grade in grades:
            grade_counts[grade] = grade_counts.get(grade, 0) + 1

        return max(grade_counts.items(), key=lambda x: x[1])[0]

    def _analyze_source_dimensions(self, cards: List[QualityScoreCard]) -> Dict[str, float]:
        """Analyze average dimension scores for a source"""
        dimension_totals = defaultdict(list)

        for card in cards:
            for dimension, score in card.dimension_scores.items():
                dimension_totals[dimension.value].append(score.final_score)

        return {
            dim: statistics.mean(scores) if scores else 0.0
            for dim, scores in dimension_totals.items()
        }

    def _analyze_common_issues(self, cards: List[QualityScoreCard]) -> List[Tuple[str, int]]:
        """Analyze common issues across score cards"""
        issue_counts = defaultdict(int)

        for card in cards:
            # Count issues by type (simplified for now)
            if card.critical_issues > 0:
                issue_counts["critical_data_quality"] += card.critical_issues
            if card.high_issues > 0:
                issue_counts["high_priority_issues"] += card.high_issues
            if card.medium_issues > 0:
                issue_counts["medium_priority_issues"] += card.medium_issues

        # Return top 5 most common issues
        return sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    def _generate_source_recommendations(self, avg_score: float, dimension_scores: Dict[str, float],
                                       common_issues: List[Tuple[str, int]],
                                       critical_issues: int) -> Tuple[str, List[str]]:
        """Generate improvement recommendations for a source"""
        actions = []

        # Determine priority
        if avg_score < 60 or critical_issues > 10:
            priority = "high"
            actions.append("URGENT: Immediate quality improvement required")
        elif avg_score < 75 or critical_issues > 5:
            priority = "medium"
            actions.append("Quality improvement recommended within 30 days")
        else:
            priority = "low"
            actions.append("Monitor quality trends and optimize as needed")

        # Dimension-specific recommendations
        for dimension, score in dimension_scores.items():
            if score < 70:
                actions.append(f"Focus on improving {dimension} (current score: {score:.1f})")

        # Issue-specific recommendations
        for issue_type, count in common_issues[:3]:  # Top 3 issues
            if count > 5:
                actions.append(f"Address recurring {issue_type.replace('_', ' ')} ({count} occurrences)")

        return priority, actions

    def get_quality_alerts(self, severity: str = "all") -> List[Dict[str, Any]]:
        """Get current quality alerts

        Args:
            severity: Filter by severity (critical, high, medium, low, all)

        Returns:
            List of alert dictionaries
        """
        alerts = []
        metrics = self.get_dashboard_metrics()

        # Critical grade alerts
        for symbol, source, grade in metrics.failing_sources:
            alerts.append({
                "type": "failing_grade",
                "severity": "critical",
                "symbol": symbol,
                "source": source,
                "message": f"Quality grade {grade} below threshold",
                "details": {"current_grade": grade, "threshold": self.alert_thresholds["critical_grade"].value},
                "timestamp": datetime.now().isoformat()
            })

        # Degrading quality alerts
        for symbol, source, decline in metrics.degrading_quality:
            severity_level = "high" if decline > 25 else "medium"
            alerts.append({
                "type": "quality_degradation",
                "severity": severity_level,
                "symbol": symbol,
                "source": source,
                "message": f"Quality declined by {decline:.1f} points",
                "details": {"decline_points": decline, "threshold": self.alert_thresholds["degradation_threshold"]},
                "timestamp": datetime.now().isoformat()
            })

        # Performance alerts
        if metrics.average_processing_time_ms > self.alert_thresholds["processing_time_threshold"]:
            alerts.append({
                "type": "performance_degradation",
                "severity": "medium",
                "symbol": "system",
                "source": "quality_engine",
                "message": f"Processing time {metrics.average_processing_time_ms:.1f}ms exceeds threshold",
                "details": {
                    "current_time_ms": metrics.average_processing_time_ms,
                    "threshold_ms": self.alert_thresholds["processing_time_threshold"]
                },
                "timestamp": datetime.now().isoformat()
            })

        # Data freshness alerts
        if metrics.oldest_data_age_seconds > self.alert_thresholds["data_age_threshold"]:
            alerts.append({
                "type": "stale_data",
                "severity": "medium",
                "symbol": "system",
                "source": "data_pipeline",
                "message": f"Oldest data is {metrics.oldest_data_age_seconds/3600:.1f} hours old",
                "details": {
                    "age_hours": metrics.oldest_data_age_seconds / 3600,
                    "threshold_hours": self.alert_thresholds["data_age_threshold"] / 3600
                },
                "timestamp": datetime.now().isoformat()
            })

        # Filter by severity
        if severity != "all":
            alerts = [alert for alert in alerts if alert["severity"] == severity]

        return sorted(alerts, key=lambda x: x["timestamp"], reverse=True)

    def generate_quality_report(self, include_recommendations: bool = True) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        metrics = self.get_dashboard_metrics()
        profiles = self.get_source_quality_profiles()
        alerts = self.get_quality_alerts()

        report = {
            "report_timestamp": datetime.now().isoformat(),
            "executive_summary": {
                "overall_grade": metrics.current_average_grade,
                "overall_score": metrics.current_average_score,
                "total_sources_monitored": metrics.total_sources,
                "total_symbols_tracked": metrics.total_symbols,
                "quality_trend": metrics.trend_direction,
                "critical_alerts": len([a for a in alerts if a["severity"] == "critical"])
            },
            "detailed_metrics": metrics.to_dict(),
            "source_profiles": {name: profile.to_dict() for name, profile in profiles.items()},
            "active_alerts": alerts,
            "grade_distribution": metrics.grade_distribution
        }

        if include_recommendations:
            report["improvement_recommendations"] = self._generate_system_recommendations(metrics, profiles)

        return report

    def _generate_system_recommendations(self, metrics: DashboardMetrics,
                                       profiles: Dict[str, SourceQualityProfile]) -> List[str]:
        """Generate system-wide improvement recommendations"""
        recommendations = []

        # Overall quality recommendations
        if metrics.current_average_score < 70:
            recommendations.append("PRIORITY: System-wide quality below acceptable levels - implement comprehensive quality improvement plan")

        # Source-specific recommendations
        high_priority_sources = [name for name, profile in profiles.items() if profile.improvement_priority == "high"]
        if high_priority_sources:
            recommendations.append(f"Address critical quality issues in sources: {', '.join(high_priority_sources)}")

        # Trend-based recommendations
        if metrics.trend_direction == "declining" and metrics.trend_strength > 0.3:
            recommendations.append("Quality trend is declining - investigate root causes and implement corrective measures")

        # Performance recommendations
        if metrics.average_processing_time_ms > 500:
            recommendations.append("Optimize quality assessment performance - current processing time exceeds target")

        return recommendations

    def update_alert_thresholds(self, thresholds: Dict[str, Any]) -> None:
        """Update alert threshold configuration"""
        self.alert_thresholds.update(thresholds)
        # Clear cache to force recalculation
        self.metrics_cache = None
        logger.info(f"Updated alert thresholds: {thresholds}")

    def clear_cache(self) -> None:
        """Clear dashboard metrics cache"""
        self.metrics_cache = None
        self.cache_timestamp = None