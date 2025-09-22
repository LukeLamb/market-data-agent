"""Tests for Quality Manager

Comprehensive tests for the unified quality management system including
automated assessment, action planning, and alert management.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.quality.quality_manager import (
    QualityManager, QualityManagerConfig, QualityActionPlan
)
from src.quality.quality_scoring_engine import QualityScoreCard, QualityGrade
from src.validation.validation_engine import ValidationSummary
from src.data_sources.base import CurrentPrice


@pytest.fixture
def manager_config():
    """Quality manager configuration for testing"""
    return QualityManagerConfig(
        enable_auto_assessment=False,  # Disable for testing
        assessment_interval_minutes=1.0,
        critical_quality_threshold=60.0,
        enable_quality_alerts=True
    )


@pytest.fixture
def quality_manager(manager_config):
    """Quality manager instance for testing"""
    return QualityManager(manager_config)


@pytest.fixture
def mock_validation_engine():
    """Mock validation engine for testing"""
    engine = AsyncMock()
    engine.validate_current_price.return_value = ValidationSummary(
        data_accepted=True,
        overall_confidence=0.9,
        quality_score=85.0,
        validation_mode=Mock()
    )
    return engine


@pytest.fixture
def sample_current_prices():
    """Sample current price data for testing"""
    timestamp = datetime.now()
    return [
        CurrentPrice(
            symbol="AAPL",
            price=150.0,
            timestamp=timestamp,
            volume=1000000,
            source="test_source",
            quality_score=90
        )
    ]


@pytest.fixture
def sample_score_card():
    """Sample quality score card for testing"""
    return QualityScoreCard(
        symbol="AAPL",
        source="test_source",
        overall_grade=QualityGrade.B,
        overall_score=80.0,
        dimension_scores={},
        total_issues=2,
        critical_issues=0,
        high_issues=1,
        medium_issues=1,
        low_issues=0,
        priority_recommendations=["Fix timeliness"],
        improvement_suggestions=["Improve accuracy"],
        assessment_period=(datetime.now() - timedelta(hours=1), datetime.now()),
        data_points_analyzed=10,
        confidence_level=0.85
    )


class TestQualityManager:
    """Test suite for QualityManager"""

    def test_initialization_default_config(self):
        """Test manager initialization with default configuration"""
        manager = QualityManager()

        assert manager.config is not None
        assert isinstance(manager.config, QualityManagerConfig)
        assert manager.scoring_engine is not None
        assert manager.dashboard is not None
        assert manager.validation_engine is None
        assert not manager.is_running

    def test_initialization_with_validation_engine(self, manager_config, mock_validation_engine):
        """Test manager initialization with validation engine"""
        manager = QualityManager(manager_config, mock_validation_engine)

        assert manager.validation_engine == mock_validation_engine
        assert manager.config.enable_validation_integration

    @pytest.mark.asyncio
    async def test_start_stop_quality_management(self, quality_manager):
        """Test starting and stopping quality management"""
        # Start management
        await quality_manager.start_quality_management()
        assert quality_manager.is_running

        # Stop management
        await quality_manager.stop_quality_management()
        assert not quality_manager.is_running

    @pytest.mark.asyncio
    async def test_assess_data_quality_basic(self, quality_manager, sample_current_prices):
        """Test basic data quality assessment"""
        with patch.object(quality_manager.scoring_engine, 'generate_score_card') as mock_generate:
            mock_generate.return_value = QualityScoreCard(
                symbol="AAPL",
                source="test_source",
                overall_grade=QualityGrade.A,
                overall_score=90.0,
                dimension_scores={},
                total_issues=0,
                critical_issues=0,
                high_issues=0,
                medium_issues=0,
                low_issues=0,
                priority_recommendations=[],
                improvement_suggestions=[],
                assessment_period=(datetime.now() - timedelta(hours=1), datetime.now()),
                data_points_analyzed=1,
                confidence_level=0.95
            )

            score_card = await quality_manager.assess_data_quality(
                "AAPL", "test_source", sample_current_prices
            )

            assert isinstance(score_card, QualityScoreCard)
            assert score_card.symbol == "AAPL"
            assert score_card.source == "test_source"
            assert quality_manager.manager_metrics["total_assessments_managed"] == 1

    @pytest.mark.asyncio
    async def test_assess_data_quality_with_validation(self, manager_config, mock_validation_engine, sample_current_prices):
        """Test data quality assessment with validation integration"""
        manager = QualityManager(manager_config, mock_validation_engine)

        with patch.object(manager.scoring_engine, 'generate_score_card') as mock_generate:
            mock_generate.return_value = QualityScoreCard(
                symbol="AAPL",
                source="test_source",
                overall_grade=QualityGrade.B,
                overall_score=80.0,
                dimension_scores={},
                total_issues=1,
                critical_issues=0,
                high_issues=1,
                medium_issues=0,
                low_issues=0,
                priority_recommendations=[],
                improvement_suggestions=[],
                assessment_period=(datetime.now() - timedelta(hours=1), datetime.now()),
                data_points_analyzed=1,
                confidence_level=0.85
            )

            score_card = await manager.assess_data_quality(
                "AAPL", "test_source", sample_current_prices
            )

            # Should have called validation engine
            mock_validation_engine.validate_current_price.assert_called_once()

            # Should have validation metadata
            assert score_card.metadata.get("validation_integrated") is True

    @pytest.mark.asyncio
    async def test_create_quality_action_plan(self, quality_manager):
        """Test quality action plan creation"""
        # Create score card that triggers action plan
        poor_score_card = QualityScoreCard(
            symbol="AAPL",
            source="test_source",
            overall_grade=QualityGrade.D,
            overall_score=50.0,  # Below threshold
            dimension_scores={},
            total_issues=5,
            critical_issues=2,
            high_issues=2,
            medium_issues=1,
            low_issues=0,
            priority_recommendations=["URGENT: Fix critical issues"],
            improvement_suggestions=["Improve data pipeline"],
            assessment_period=(datetime.now() - timedelta(hours=1), datetime.now()),
            data_points_analyzed=10,
            confidence_level=0.60
        )

        await quality_manager._create_quality_action_plan(poor_score_card)

        action_key = "AAPL_test_source"
        assert action_key in quality_manager.action_plans

        action_plan = quality_manager.action_plans[action_key]
        assert action_plan.symbol == "AAPL"
        assert action_plan.source == "test_source"
        assert action_plan.priority_level == "high"
        assert len(action_plan.immediate_actions) > 0
        assert quality_manager.manager_metrics["action_plans_created"] == 1

    @pytest.mark.asyncio
    async def test_alert_management(self, quality_manager):
        """Test quality alert generation and management"""
        # Create score card that triggers alerts
        critical_score_card = QualityScoreCard(
            symbol="AAPL",
            source="test_source",
            overall_grade=QualityGrade.F,
            overall_score=30.0,  # Well below threshold
            dimension_scores={},
            total_issues=10,
            critical_issues=8,
            high_issues=2,
            medium_issues=0,
            low_issues=0,
            priority_recommendations=["CRITICAL: Immediate action required"],
            improvement_suggestions=[],
            assessment_period=(datetime.now() - timedelta(hours=1), datetime.now()),
            data_points_analyzed=10,
            confidence_level=0.95
        )

        # Mock alert callback
        alert_callback = Mock()
        quality_manager.add_alert_callback(alert_callback)

        await quality_manager._check_and_send_alerts(critical_score_card)

        # Should have sent alerts
        assert quality_manager.manager_metrics["alert_notifications_sent"] > 0
        assert alert_callback.called

    def test_queue_assessment(self, quality_manager, sample_current_prices):
        """Test assessment queuing"""
        initial_queue_size = len(quality_manager.assessment_queue)

        quality_manager.queue_assessment("AAPL", "test_source", sample_current_prices)

        assert len(quality_manager.assessment_queue) == initial_queue_size + 1

    def test_get_action_plan(self, quality_manager):
        """Test action plan retrieval"""
        # Create mock action plan
        action_plan = QualityActionPlan(
            symbol="AAPL",
            source="test_source",
            current_grade=QualityGrade.C,
            current_score=70.0,
            target_score=85.0,
            priority_level="medium",
            immediate_actions=["Action 1"],
            short_term_actions=["Action 2"],
            long_term_actions=["Action 3"],
            success_criteria=["Criteria 1"],
            review_date=datetime.now() + timedelta(days=7),
            estimated_improvement=15.0
        )

        quality_manager.action_plans["AAPL_test_source"] = action_plan

        retrieved_plan = quality_manager.get_action_plan("AAPL", "test_source")
        assert retrieved_plan == action_plan

        # Test non-existent plan
        non_existent = quality_manager.get_action_plan("NONEXISTENT", "source")
        assert non_existent is None

    def test_update_action_plan_progress(self, quality_manager):
        """Test action plan progress updates"""
        # Create action plan
        action_plan = QualityActionPlan(
            symbol="AAPL",
            source="test_source",
            current_grade=QualityGrade.C,
            current_score=70.0,
            target_score=85.0,
            priority_level="medium",
            immediate_actions=["Action 1"],
            short_term_actions=["Action 2"],
            long_term_actions=["Action 3"],
            success_criteria=["Criteria 1"],
            review_date=datetime.now() + timedelta(days=7),
            estimated_improvement=15.0
        )

        quality_manager.action_plans["AAPL_test_source"] = action_plan

        # Update progress
        quality_manager.update_action_plan_progress(
            "AAPL", "test_source",
            ["Action 1 completed"],
            "Good progress on immediate actions"
        )

        # Check progress was recorded
        assert "progress_updates" in action_plan.__dict__
        assert len(action_plan.__dict__["progress_updates"]) == 1

    def test_alert_callback_management(self, quality_manager):
        """Test alert callback management"""
        callback1 = Mock()
        callback2 = Mock()

        # Add callbacks
        quality_manager.add_alert_callback(callback1)
        quality_manager.add_alert_callback(callback2)
        assert len(quality_manager.alert_callbacks) == 2

        # Remove callback
        quality_manager.remove_alert_callback(callback1)
        assert len(quality_manager.alert_callbacks) == 1
        assert callback2 in quality_manager.alert_callbacks

    def test_configuration_update(self, quality_manager):
        """Test configuration updates"""
        new_config = QualityManagerConfig(
            critical_quality_threshold=50.0,
            enable_quality_alerts=False,
            assessment_interval_minutes=30.0
        )

        quality_manager.update_configuration(new_config)

        assert quality_manager.config.critical_quality_threshold == 50.0
        assert not quality_manager.config.enable_quality_alerts
        assert quality_manager.config.assessment_interval_minutes == 30.0

    def test_get_quality_report(self, quality_manager):
        """Test quality report generation"""
        report = quality_manager.get_quality_report()

        assert "quality_management" in report
        assert "assessments_queued" in report["quality_management"]
        assert "active_assessments" in report["quality_management"]
        assert "manager_metrics" in report["quality_management"]

    def test_get_performance_metrics(self, quality_manager):
        """Test performance metrics retrieval"""
        metrics = quality_manager.get_performance_metrics()

        assert "manager_metrics" in metrics
        assert "dashboard_metrics" in metrics
        assert "scoring_metrics" in metrics
        assert "system_status" in metrics

    @pytest.mark.asyncio
    async def test_health_check(self, quality_manager):
        """Test comprehensive health check"""
        health_status = await quality_manager.health_check()

        assert "overall_status" in health_status
        assert "timestamp" in health_status
        assert "components" in health_status

        # Check required components
        components = health_status["components"]
        assert "scoring_engine" in components
        assert "dashboard" in components
        assert "validation_integration" in components
        assert "auto_assessment" in components

    def test_quality_action_plan_serialization(self):
        """Test QualityActionPlan serialization"""
        action_plan = QualityActionPlan(
            symbol="AAPL",
            source="test_source",
            current_grade=QualityGrade.B,
            current_score=80.0,
            target_score=90.0,
            priority_level="medium",
            immediate_actions=["Action 1"],
            short_term_actions=["Action 2", "Action 3"],
            long_term_actions=["Action 4"],
            success_criteria=["Criteria 1", "Criteria 2"],
            review_date=datetime.now() + timedelta(days=7),
            estimated_improvement=10.0
        )

        serialized = action_plan.to_dict()

        assert serialized["symbol"] == "AAPL"
        assert serialized["source"] == "test_source"
        assert serialized["current_grade"] == "B"
        assert serialized["current_score"] == 80.0
        assert serialized["target_score"] == 90.0
        assert serialized["priority_level"] == "medium"
        assert serialized["immediate_actions"] == ["Action 1"]
        assert serialized["estimated_improvement"] == 10.0

    @pytest.mark.asyncio
    async def test_error_handling(self, quality_manager, sample_current_prices):
        """Test error handling in quality assessment"""
        with patch.object(quality_manager.scoring_engine, 'generate_score_card') as mock_generate:
            mock_generate.side_effect = Exception("Test error")

            with pytest.raises(Exception):
                await quality_manager.assess_data_quality(
                    "AAPL", "test_source", sample_current_prices
                )

    @pytest.mark.asyncio
    async def test_alert_cooldown(self, quality_manager, sample_score_card):
        """Test alert cooldown functionality"""
        # Set short cooldown for testing
        quality_manager.config.alert_cooldown_minutes = 0.1

        # Create critical score card
        critical_card = sample_score_card
        critical_card.overall_score = 30.0  # Below threshold

        # First alert should be sent
        await quality_manager._check_and_send_alerts(critical_card)
        initial_alert_count = quality_manager.manager_metrics["alert_notifications_sent"]

        # Immediate second alert should be blocked by cooldown
        await quality_manager._check_and_send_alerts(critical_card)
        assert quality_manager.manager_metrics["alert_notifications_sent"] == initial_alert_count

    def test_validation_score_adjustment(self, quality_manager, sample_score_card):
        """Test score adjustment based on validation results"""
        validation_result = ValidationSummary(
            data_accepted=False,  # Failed validation
            overall_confidence=0.3,
            quality_score=60.0,
            validation_mode=Mock()
        )

        adjusted_card = quality_manager._adjust_score_with_validation(
            sample_score_card, validation_result
        )

        # Score should be reduced due to failed validation
        assert adjusted_card.overall_score < sample_score_card.overall_score
        assert adjusted_card.metadata["validation_integrated"] is True
        assert adjusted_card.metadata["validation_accepted"] is False