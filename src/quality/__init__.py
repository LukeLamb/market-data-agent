"""Quality Management Package

Advanced quality scoring and grading system for market data with A-F grades,
multi-dimensional analysis, and actionable improvement recommendations.
"""

from .quality_scoring_engine import (
    QualityScoringEngine,
    QualityScoreCard,
    QualityGrade,
    SeverityLevel,
    DimensionWeight,
    ScoringConfig,
    DetailedScore
)

from .quality_dashboard import (
    QualityDashboard,
    DashboardMetrics,
    SourceQualityProfile
)

from .quality_manager import (
    QualityManager,
    QualityManagerConfig,
    QualityActionPlan
)

__all__ = [
    # Scoring Engine
    "QualityScoringEngine",
    "QualityScoreCard",
    "QualityGrade",
    "SeverityLevel",
    "DimensionWeight",
    "ScoringConfig",
    "DetailedScore",

    # Dashboard
    "QualityDashboard",
    "DashboardMetrics",
    "SourceQualityProfile",

    # Quality Manager
    "QualityManager",
    "QualityManagerConfig",
    "QualityActionPlan"
]