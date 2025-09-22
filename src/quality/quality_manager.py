"""Quality Manager

Unified quality management system that orchestrates scoring, monitoring,
and improvement workflows across all market data sources.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

from .quality_scoring_engine import (
    QualityScoringEngine, QualityScoreCard, QualityGrade, ScoringConfig
)
from .quality_dashboard import QualityDashboard, DashboardMetrics
from ..validation.validation_engine import ValidationEngine, ValidationSummary
from ..data_sources.base import CurrentPrice, PriceData

logger = logging.getLogger(__name__)


@dataclass
class QualityManagerConfig:
    """Configuration for the quality manager"""
    # Automatic assessment settings
    enable_auto_assessment: bool = True
    assessment_interval_minutes: float = 15.0
    batch_assessment_size: int = 100

    # Quality thresholds for actions
    critical_quality_threshold: float = 60.0      # Below this triggers immediate action
    degradation_alert_threshold: float = 15.0     # Point decline that triggers alert
    improvement_target_score: float = 85.0        # Target quality score

    # Integration settings
    enable_validation_integration: bool = True
    enable_real_time_scoring: bool = True
    enable_trend_analysis: bool = True

    # Performance settings
    max_concurrent_assessments: int = 10
    assessment_timeout_seconds: float = 30.0
    cache_assessment_results: bool = True

    # Alert and notification settings
    enable_quality_alerts: bool = True
    alert_cooldown_minutes: float = 60.0
    notification_channels: List[str] = field(default_factory=lambda: ["log", "dashboard"])


@dataclass
class QualityActionPlan:
    """Action plan for quality improvement"""
    symbol: str
    source: str
    current_grade: QualityGrade
    current_score: float
    target_score: float
    priority_level: str  # critical, high, medium, low

    # Specific actions
    immediate_actions: List[str]
    short_term_actions: List[str]  # Within 1 week
    long_term_actions: List[str]   # Within 1 month

    # Success metrics
    success_criteria: List[str]
    review_date: datetime
    estimated_improvement: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "symbol": self.symbol,
            "source": self.source,
            "current_grade": self.current_grade.value,
            "current_score": self.current_score,
            "target_score": self.target_score,
            "priority_level": self.priority_level,
            "immediate_actions": self.immediate_actions,
            "short_term_actions": self.short_term_actions,
            "long_term_actions": self.long_term_actions,
            "success_criteria": self.success_criteria,
            "review_date": self.review_date.isoformat(),
            "estimated_improvement": self.estimated_improvement
        }


class QualityManager:
    """Unified quality management system

    Features:
    - Automated quality assessment and scoring
    - Integration with validation engine
    - Real-time quality monitoring and alerting
    - Quality improvement action planning
    - Performance analytics and reporting
    """

    def __init__(self, config: Optional[QualityManagerConfig] = None,
                 validation_engine: Optional[ValidationEngine] = None):
        """Initialize the quality manager

        Args:
            config: Quality manager configuration
            validation_engine: Optional validation engine for integration
        """
        self.config = config or QualityManagerConfig()
        self.validation_engine = validation_engine

        # Initialize core components
        self.scoring_engine = QualityScoringEngine()
        self.dashboard = QualityDashboard(self.scoring_engine)

        # Quality tracking
        self.active_assessments: Dict[str, datetime] = {}  # symbol_source -> start_time
        self.assessment_queue: List[Tuple[str, str, List[CurrentPrice]]] = []  # (symbol, source, data)
        self.action_plans: Dict[str, QualityActionPlan] = {}  # symbol_source -> plan

        # Performance tracking
        self.manager_metrics = {
            "total_assessments_managed": 0,
            "auto_assessments_triggered": 0,
            "action_plans_created": 0,
            "quality_improvements_detected": 0,
            "alert_notifications_sent": 0
        }

        # Alert management
        self.recent_alerts: Dict[str, datetime] = {}  # alert_key -> last_sent_time
        self.alert_callbacks: List[Callable] = []

        # Background task management
        self.auto_assessment_task: Optional[asyncio.Task] = None
        self.is_running = False

    async def start_quality_management(self) -> None:
        """Start automated quality management processes"""
        if self.is_running:
            logger.warning("Quality management is already running")
            return

        self.is_running = True
        logger.info("Starting quality management system")

        if self.config.enable_auto_assessment:
            self.auto_assessment_task = asyncio.create_task(self._auto_assessment_loop())

    async def stop_quality_management(self) -> None:
        """Stop automated quality management processes"""
        if not self.is_running:
            return

        self.is_running = False
        logger.info("Stopping quality management system")

        if self.auto_assessment_task:
            self.auto_assessment_task.cancel()
            try:
                await self.auto_assessment_task
            except asyncio.CancelledError:
                pass

    async def assess_data_quality(self, symbol: str, source: str,
                                current_prices: List[CurrentPrice],
                                real_time: bool = False) -> QualityScoreCard:
        """Assess data quality and generate score card

        Args:
            symbol: Stock symbol
            source: Data source name
            current_prices: Current price data
            real_time: Whether this is a real-time assessment

        Returns:
            QualityScoreCard with comprehensive quality analysis
        """
        assessment_key = f"{symbol}_{source}"

        try:
            # Track assessment start
            self.active_assessments[assessment_key] = datetime.now()

            # Generate score card
            score_card = self.scoring_engine.generate_score_card(symbol, source, current_prices)

            # Integrate with validation engine if available
            if self.validation_engine and self.config.enable_validation_integration:
                validation_score = await self._integrate_validation_results(symbol, source, current_prices)
                if validation_score:
                    # Adjust score card based on validation results
                    score_card = self._adjust_score_with_validation(score_card, validation_score)

            # Check for quality issues and create action plans
            if score_card.overall_score < self.config.critical_quality_threshold:
                await self._create_quality_action_plan(score_card)

            # Send alerts if needed
            if self.config.enable_quality_alerts:
                await self._check_and_send_alerts(score_card)

            # Update metrics
            self.manager_metrics["total_assessments_managed"] += 1
            if real_time:
                self.manager_metrics["real_time_assessments"] = self.manager_metrics.get("real_time_assessments", 0) + 1

            return score_card

        except Exception as e:
            logger.error(f"Error assessing quality for {symbol}/{source}: {e}")
            raise
        finally:
            # Clean up tracking
            self.active_assessments.pop(assessment_key, None)

    async def _integrate_validation_results(self, symbol: str, source: str,
                                          current_prices: List[CurrentPrice]) -> Optional[ValidationSummary]:
        """Integrate with validation engine for comprehensive assessment"""
        if not self.validation_engine or not current_prices:
            return None

        try:
            # Use the most recent price for validation
            latest_price = max(current_prices, key=lambda p: p.timestamp)

            # Create additional sources dict for cross-validation if we have multiple prices
            additional_sources = {}
            if len(current_prices) > 1:
                for i, price in enumerate(current_prices[1:], 1):
                    additional_sources[f"{source}_alt_{i}"] = price

            validation_result = await self.validation_engine.validate_current_price(
                symbol, source, latest_price, additional_sources
            )

            return validation_result

        except Exception as e:
            logger.warning(f"Validation integration failed for {symbol}/{source}: {e}")
            return None

    def _adjust_score_with_validation(self, score_card: QualityScoreCard,
                                    validation_result: ValidationSummary) -> QualityScoreCard:
        """Adjust quality score based on validation results"""
        # Calculate validation impact on score
        validation_confidence = validation_result.overall_confidence
        validation_accepted = validation_result.data_accepted

        # Adjust overall score based on validation
        if not validation_accepted:
            # Failed validation significantly impacts score
            score_adjustment = -20.0 * (1.0 - validation_confidence)
        else:
            # Passed validation may provide small boost
            score_adjustment = 2.0 * validation_confidence

        # Apply adjustment
        adjusted_score = max(0.0, min(100.0, score_card.overall_score + score_adjustment))

        # Update score card
        score_card.overall_score = adjusted_score
        score_card.overall_grade = self.scoring_engine._determine_grade(adjusted_score)

        # Add validation info to metadata
        score_card.metadata.update({
            "validation_integrated": True,
            "validation_confidence": validation_confidence,
            "validation_accepted": validation_accepted,
            "validation_adjustment": score_adjustment
        })

        return score_card

    async def _create_quality_action_plan(self, score_card: QualityScoreCard) -> None:
        """Create quality improvement action plan"""
        action_key = f"{score_card.symbol}_{score_card.source}"

        # Determine priority level
        if score_card.overall_score < 40:
            priority = "critical"
        elif score_card.overall_score < 60:
            priority = "high"
        elif score_card.overall_score < 75:
            priority = "medium"
        else:
            priority = "low"

        # Generate actions based on score card
        immediate_actions = []
        short_term_actions = []
        long_term_actions = []

        # Use priority recommendations as immediate actions
        immediate_actions.extend(score_card.priority_recommendations[:3])

        # Use improvement suggestions for short-term actions
        short_term_actions.extend(score_card.improvement_suggestions[:5])

        # Add long-term strategic actions
        if score_card.overall_score < 70:
            long_term_actions.extend([
                "Implement comprehensive data quality monitoring",
                "Establish data quality SLAs with source provider",
                "Develop automated quality improvement workflows"
            ])

        # Estimate improvement potential
        estimated_improvement = min(25.0, (100.0 - score_card.overall_score) * 0.6)

        action_plan = QualityActionPlan(
            symbol=score_card.symbol,
            source=score_card.source,
            current_grade=score_card.overall_grade,
            current_score=score_card.overall_score,
            target_score=min(100.0, score_card.overall_score + estimated_improvement),
            priority_level=priority,
            immediate_actions=immediate_actions,
            short_term_actions=short_term_actions,
            long_term_actions=long_term_actions,
            success_criteria=[
                f"Achieve quality score above {score_card.overall_score + 10:.0f}",
                "Reduce critical issues to zero",
                "Maintain consistent quality for 7 days"
            ],
            review_date=datetime.now() + timedelta(days=7),
            estimated_improvement=estimated_improvement
        )

        self.action_plans[action_key] = action_plan
        self.manager_metrics["action_plans_created"] += 1

        logger.info(f"Created {priority} priority action plan for {score_card.symbol}/{score_card.source}")

    async def _check_and_send_alerts(self, score_card: QualityScoreCard) -> None:
        """Check for alert conditions and send notifications"""
        alert_key = f"{score_card.symbol}_{score_card.source}"

        # Check cooldown period
        if alert_key in self.recent_alerts:
            last_alert = self.recent_alerts[alert_key]
            if (datetime.now() - last_alert).total_seconds() < self.config.alert_cooldown_minutes * 60:
                return  # Still in cooldown

        # Check alert conditions
        alerts_to_send = []

        # Critical quality alert
        if score_card.overall_score < self.config.critical_quality_threshold:
            alerts_to_send.append({
                "type": "critical_quality",
                "severity": "critical",
                "message": f"Critical quality issue: {score_card.symbol}/{score_card.source} scored {score_card.overall_grade.value}",
                "score_card": score_card
            })

        # High critical issues alert
        if score_card.critical_issues > 5:
            alerts_to_send.append({
                "type": "high_critical_issues",
                "severity": "high",
                "message": f"High number of critical issues: {score_card.critical_issues} detected",
                "score_card": score_card
            })

        # Send alerts
        for alert in alerts_to_send:
            await self._send_alert(alert)
            self.recent_alerts[alert_key] = datetime.now()
            self.manager_metrics["alert_notifications_sent"] += 1

    async def _send_alert(self, alert: Dict[str, Any]) -> None:
        """Send quality alert through configured channels"""
        # Log alert
        if "log" in self.config.notification_channels:
            logger.warning(f"Quality Alert [{alert['severity']}]: {alert['message']}")

        # Execute alert callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    async def _auto_assessment_loop(self) -> None:
        """Main loop for automated quality assessments"""
        logger.info("Starting automated quality assessment loop")

        while self.is_running:
            try:
                # Process assessment queue
                if self.assessment_queue:
                    batch = self.assessment_queue[:self.config.batch_assessment_size]
                    self.assessment_queue = self.assessment_queue[self.config.batch_assessment_size:]

                    # Process batch concurrently
                    tasks = []
                    for symbol, source, data in batch:
                        task = self.assess_data_quality(symbol, source, data, real_time=False)
                        tasks.append(task)

                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
                        self.manager_metrics["auto_assessments_triggered"] += len(tasks)

                # Wait for next interval
                await asyncio.sleep(self.config.assessment_interval_minutes * 60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto assessment loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

        logger.info("Automated quality assessment loop stopped")

    def queue_assessment(self, symbol: str, source: str, current_prices: List[CurrentPrice]) -> None:
        """Queue data for automated quality assessment"""
        self.assessment_queue.append((symbol, source, current_prices))

    def get_quality_dashboard_metrics(self) -> DashboardMetrics:
        """Get comprehensive dashboard metrics"""
        return self.dashboard.get_dashboard_metrics()

    def get_quality_report(self, include_action_plans: bool = True) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        report = self.dashboard.generate_quality_report()

        # Add manager-specific information
        report["quality_management"] = {
            "auto_assessment_enabled": self.config.enable_auto_assessment,
            "assessments_queued": len(self.assessment_queue),
            "active_assessments": len(self.active_assessments),
            "manager_metrics": self.manager_metrics
        }

        if include_action_plans:
            report["active_action_plans"] = {
                key: plan.to_dict() for key, plan in self.action_plans.items()
            }

        return report

    def get_action_plan(self, symbol: str, source: str) -> Optional[QualityActionPlan]:
        """Get quality action plan for specific symbol/source"""
        action_key = f"{symbol}_{source}"
        return self.action_plans.get(action_key)

    def update_action_plan_progress(self, symbol: str, source: str,
                                  completed_actions: List[str],
                                  progress_notes: str) -> None:
        """Update progress on quality action plan"""
        action_key = f"{symbol}_{source}"
        if action_key not in self.action_plans:
            logger.warning(f"No action plan found for {symbol}/{source}")
            return

        plan = self.action_plans[action_key]

        # Add progress to metadata
        if "progress_updates" not in plan.__dict__:
            plan.__dict__["progress_updates"] = []

        plan.__dict__["progress_updates"].append({
            "timestamp": datetime.now().isoformat(),
            "completed_actions": completed_actions,
            "notes": progress_notes
        })

        logger.info(f"Updated action plan progress for {symbol}/{source}")

    def add_alert_callback(self, callback: Callable) -> None:
        """Add callback function for quality alerts"""
        self.alert_callbacks.append(callback)

    def remove_alert_callback(self, callback: Callable) -> None:
        """Remove alert callback function"""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)

    def update_configuration(self, config: QualityManagerConfig) -> None:
        """Update quality manager configuration"""
        self.config = config
        logger.info("Quality manager configuration updated")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get quality management performance metrics"""
        dashboard_metrics = self.dashboard.get_dashboard_metrics()
        scoring_summary = self.scoring_engine.get_scoring_summary()

        return {
            "manager_metrics": self.manager_metrics,
            "dashboard_metrics": dashboard_metrics.to_dict(),
            "scoring_metrics": scoring_summary,
            "active_assessments": len(self.active_assessments),
            "queued_assessments": len(self.assessment_queue),
            "active_action_plans": len(self.action_plans),
            "system_status": "running" if self.is_running else "stopped"
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of quality management system"""
        health_status = {
            "overall_status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }

        # Check scoring engine
        try:
            scoring_summary = self.scoring_engine.get_scoring_summary()
            health_status["components"]["scoring_engine"] = {
                "status": "healthy",
                "total_assessments": scoring_summary["total_assessments"]
            }
        except Exception as e:
            health_status["components"]["scoring_engine"] = {
                "status": "error",
                "error": str(e)
            }
            health_status["overall_status"] = "degraded"

        # Check dashboard
        try:
            dashboard_metrics = self.dashboard.get_dashboard_metrics()
            health_status["components"]["dashboard"] = {
                "status": "healthy",
                "symbols_tracked": dashboard_metrics.total_symbols
            }
        except Exception as e:
            health_status["components"]["dashboard"] = {
                "status": "error",
                "error": str(e)
            }
            health_status["overall_status"] = "degraded"

        # Check validation integration
        if self.validation_engine:
            health_status["components"]["validation_integration"] = {
                "status": "healthy",
                "enabled": self.config.enable_validation_integration
            }
        else:
            health_status["components"]["validation_integration"] = {
                "status": "not_configured",
                "enabled": False
            }

        # Check auto assessment
        health_status["components"]["auto_assessment"] = {
            "status": "running" if self.is_running else "stopped",
            "enabled": self.config.enable_auto_assessment,
            "queue_size": len(self.assessment_queue)
        }

        return health_status