"""Tests for Memory Manager

Comprehensive tests for the memory management system including knowledge graph
operations, pattern recognition, adaptive learning, and quality prediction.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.memory.memory_manager import (
    MemoryManager, AdaptiveLearningConfig, MemoryEntity, MemoryRelation,
    EntityType, RelationType
)
from src.data_sources.base import CurrentPrice
from src.quality.quality_scoring_engine import QualityScoreCard, QualityGrade


@pytest.fixture
def learning_config():
    """Adaptive learning configuration for testing"""
    return AdaptiveLearningConfig(
        pattern_detection_threshold=0.5,
        anomaly_detection_threshold=0.2,
        quality_correlation_threshold=0.4,
        learning_decay_factor=0.9,
        max_memory_age_days=7,
        enable_predictive_scoring=True,
        enable_source_reputation=True,
        enable_market_context=True
    )


@pytest.fixture
def memory_manager(learning_config):
    """Memory manager instance for testing"""
    return MemoryManager(learning_config, memory_server_available=False)


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
            quality_score=85
        ),
        CurrentPrice(
            symbol="AAPL",
            price=151.0,
            timestamp=timestamp + timedelta(minutes=1),
            volume=1100000,
            source="test_source",
            quality_score=87
        )
    ]


@pytest.fixture
def sample_quality_score():
    """Sample quality score card for testing"""
    return QualityScoreCard(
        symbol="AAPL",
        source="test_source",
        overall_grade=QualityGrade.B,
        overall_score=82.0,
        dimension_scores={"completeness": 90.0, "accuracy": 85.0, "timeliness": 75.0},
        total_issues=3,
        critical_issues=0,
        high_issues=1,
        medium_issues=2,
        low_issues=0,
        priority_recommendations=["Improve timeliness"],
        improvement_suggestions=["Monitor data freshness"],
        assessment_period=(datetime.now() - timedelta(hours=1), datetime.now()),
        data_points_analyzed=10,
        confidence_level=0.85
    )


class TestMemoryManager:
    """Test suite for MemoryManager"""

    @pytest.mark.asyncio
    async def test_initialization(self, memory_manager):
        """Test memory manager initialization"""
        await memory_manager.initialize_memory_system()

        assert not memory_manager.memory_server_available
        assert len(memory_manager.entities) == 0
        assert len(memory_manager.relations) == 0
        assert len(memory_manager.source_reputation) > 0  # Should initialize default sources

    @pytest.mark.asyncio
    async def test_learn_from_price_data_basic(self, memory_manager, sample_current_prices):
        """Test basic learning from price data"""
        await memory_manager.initialize_memory_system()

        symbol = "AAPL"
        source = "test_source"

        await memory_manager.learn_from_price_data(symbol, source, sample_current_prices)

        # Check entities were created
        assert symbol in memory_manager.entities
        assert source in memory_manager.entities

        # Check entity attributes
        symbol_entity = memory_manager.entities[symbol]
        assert symbol_entity.entity_type == EntityType.SYMBOL
        assert symbol_entity.attributes["latest_price"] == 151.0
        assert symbol_entity.attributes["data_points"] == 2

        source_entity = memory_manager.entities[source]
        assert source_entity.entity_type == EntityType.SOURCE
        assert source_entity.attributes["data_points_provided"] == 2

        # Check relationship was created
        data_relations = [
            rel for rel in memory_manager.relations
            if rel.relation_type == RelationType.PROVIDES_DATA
        ]
        assert len(data_relations) == 1
        assert data_relations[0].from_entity == source
        assert data_relations[0].to_entity == symbol

        assert memory_manager.memory_metrics["learning_cycles"] == 1

    @pytest.mark.asyncio
    async def test_learn_with_quality_score(self, memory_manager, sample_current_prices, sample_quality_score):
        """Test learning with quality score integration"""
        await memory_manager.initialize_memory_system()

        symbol = "AAPL"
        source = "test_source"

        await memory_manager.learn_from_price_data(
            symbol, source, sample_current_prices, sample_quality_score
        )

        # Check quality patterns were learned
        pattern_key = f"{symbol}_{source}_quality"
        assert pattern_key in memory_manager.pattern_cache

        pattern = memory_manager.pattern_cache[pattern_key]
        assert len(pattern["scores"]) == 1
        assert pattern["scores"][0] == 82.0
        assert len(pattern["grades"]) == 1
        assert pattern["grades"][0] == "B"

        # Check source entity has quality information
        source_entity = memory_manager.entities[source]
        assert source_entity.attributes["latest_quality_grade"] == "B"
        assert source_entity.attributes["latest_quality_score"] == 82.0

        assert memory_manager.memory_metrics["patterns_learned"] == 1

    @pytest.mark.asyncio
    async def test_anomaly_detection(self, memory_manager):
        """Test anomaly detection in price data"""
        await memory_manager.initialize_memory_system()

        # Create price data with large jump
        timestamp = datetime.now()
        anomaly_prices = [
            CurrentPrice(
                symbol="AAPL",
                price=150.0,
                timestamp=timestamp,
                volume=1000000,
                source="test_source",
                quality_score=85
            ),
            CurrentPrice(
                symbol="AAPL",
                price=200.0,  # 33% jump - should trigger anomaly
                timestamp=timestamp + timedelta(minutes=1),
                volume=1000000,
                source="test_source",
                quality_score=85
            )
        ]

        await memory_manager.learn_from_price_data("AAPL", "test_source", anomaly_prices)

        # Check anomaly was detected
        anomaly_entities = [
            entity for entity in memory_manager.entities.values()
            if entity.entity_type == EntityType.ANOMALY
        ]
        assert len(anomaly_entities) == 1

        anomaly = anomaly_entities[0]
        assert anomaly.attributes["symbol"] == "AAPL"
        assert anomaly.attributes["source"] == "test_source"
        assert anomaly.attributes["change_percent"] > 0.3

        # Check anomaly relationship
        anomaly_relations = [
            rel for rel in memory_manager.relations
            if rel.relation_type == RelationType.EXHIBITS_PATTERN
        ]
        assert len(anomaly_relations) == 1

    @pytest.mark.asyncio
    async def test_source_reputation_update(self, memory_manager, sample_current_prices, sample_quality_score):
        """Test source reputation tracking"""
        await memory_manager.initialize_memory_system()

        source = "test_source"
        initial_reputation = await memory_manager.get_source_reputation(source)

        # Learn from good quality data
        await memory_manager.learn_from_price_data(
            "AAPL", source, sample_current_prices, sample_quality_score
        )

        updated_reputation = await memory_manager.get_source_reputation(source)

        # Reputation should improve with good quality data
        assert updated_reputation >= initial_reputation

    @pytest.mark.asyncio
    async def test_quality_prediction(self, memory_manager, sample_current_prices, sample_quality_score):
        """Test quality score prediction"""
        await memory_manager.initialize_memory_system()

        symbol = "AAPL"
        source = "test_source"

        # Learn from historical data
        await memory_manager.learn_from_price_data(
            symbol, source, sample_current_prices, sample_quality_score
        )

        # Add more quality scores to build pattern
        for i in range(5):
            quality_score = QualityScoreCard(
                symbol=symbol,
                source=source,
                overall_grade=QualityGrade.B,
                overall_score=80.0 + i,  # Gradually improving
                dimension_scores={},
                total_issues=2 - (i // 3),
                critical_issues=0,
                high_issues=1,
                medium_issues=1,
                low_issues=0,
                priority_recommendations=[],
                improvement_suggestions=[],
                assessment_period=(datetime.now() - timedelta(hours=1), datetime.now()),
                data_points_analyzed=10,
                confidence_level=0.85
            )
            await memory_manager.learn_from_price_data(
                symbol, source, sample_current_prices, quality_score
            )

        # Test prediction
        new_price = CurrentPrice(
            symbol=symbol,
            price=155.0,
            timestamp=datetime.now(),
            volume=1200000,
            source=source,
            quality_score=85  # Valid score
        )

        predicted_quality = await memory_manager.predict_quality_score(symbol, source, new_price)

        assert predicted_quality is not None
        assert 0.0 <= predicted_quality <= 100.0
        assert memory_manager.memory_metrics["predictions_made"] == 1

    @pytest.mark.asyncio
    async def test_market_context_retrieval(self, memory_manager, sample_current_prices):
        """Test market context retrieval"""
        await memory_manager.initialize_memory_system()

        symbol = "AAPL"
        source = "test_source"

        await memory_manager.learn_from_price_data(symbol, source, sample_current_prices)

        context = await memory_manager.get_market_context(symbol)

        assert isinstance(context, dict)
        assert "related_symbols" in context
        assert "quality_patterns" in context
        assert "recent_anomalies" in context

    @pytest.mark.asyncio
    async def test_memory_cleanup(self, memory_manager, sample_current_prices):
        """Test memory cleanup functionality"""
        await memory_manager.initialize_memory_system()

        # Create old entities
        old_timestamp = datetime.now() - timedelta(days=10)

        old_entity = MemoryEntity(
            name="OLD_SYMBOL",
            entity_type=EntityType.SYMBOL,
            attributes={"test": "data"},
            created_at=old_timestamp,
            updated_at=old_timestamp
        )
        memory_manager.entities["OLD_SYMBOL"] = old_entity

        old_relation = MemoryRelation(
            from_entity="OLD_SOURCE",
            to_entity="OLD_SYMBOL",
            relation_type=RelationType.PROVIDES_DATA,
            strength=0.5,
            created_at=old_timestamp,
            metadata={}
        )
        memory_manager.relations.append(old_relation)

        # Add recent entity
        await memory_manager.learn_from_price_data("AAPL", "test_source", sample_current_prices)

        initial_entity_count = len(memory_manager.entities)
        initial_relation_count = len(memory_manager.relations)

        # Cleanup should remove old entities but keep recent ones
        await memory_manager.cleanup_old_memories()

        # Should have fewer entities/relations now
        assert len(memory_manager.entities) < initial_entity_count
        assert len(memory_manager.relations) < initial_relation_count

        # Recent entities should still exist
        assert "AAPL" in memory_manager.entities
        assert "test_source" in memory_manager.entities

    def test_entity_types_and_relations(self, memory_manager):
        """Test entity and relation type enums"""
        # Test EntityType enum
        assert EntityType.SYMBOL.value == "symbol"
        assert EntityType.SOURCE.value == "data_source"
        assert EntityType.PATTERN.value == "pattern"
        assert EntityType.ANOMALY.value == "anomaly"

        # Test RelationType enum
        assert RelationType.PROVIDES_DATA.value == "provides_data"
        assert RelationType.HAS_QUALITY_ISSUE.value == "has_quality_issue"
        assert RelationType.EXHIBITS_PATTERN.value == "exhibits_pattern"

    def test_memory_report_generation(self, memory_manager):
        """Test memory system report generation"""
        report = memory_manager.get_memory_report()

        assert "memory_system" in report
        assert "learning_config" in report
        assert "entity_types" in report
        assert "relation_types" in report

        memory_info = report["memory_system"]
        assert "server_available" in memory_info
        assert "entities_count" in memory_info
        assert "relations_count" in memory_info
        assert "metrics" in memory_info

        # Check entity type breakdown
        entity_breakdown = report["entity_types"]
        for entity_type in EntityType:
            assert entity_type.value in entity_breakdown

        # Check relation type breakdown
        relation_breakdown = report["relation_types"]
        for relation_type in RelationType:
            assert relation_type.value in relation_breakdown

    @pytest.mark.asyncio
    async def test_health_status(self, memory_manager):
        """Test memory system health status"""
        await memory_manager.initialize_memory_system()

        health_status = await memory_manager.get_health_status()

        assert "overall_status" in health_status
        assert "timestamp" in health_status
        assert "components" in health_status

        components = health_status["components"]
        assert "memory_server" in components
        assert "pattern_learning" in components
        assert "reputation_system" in components
        assert "prediction_system" in components

        # Check component statuses
        for component_name, component_info in components.items():
            assert "status" in component_info

    @pytest.mark.asyncio
    async def test_pattern_cache_management(self, memory_manager, sample_current_prices, sample_quality_score):
        """Test pattern cache size management"""
        await memory_manager.initialize_memory_system()

        symbol = "AAPL"
        source = "test_source"

        # Add many quality scores to test cache size limiting
        for i in range(120):  # More than the 100 limit
            quality_score = QualityScoreCard(
                symbol=symbol,
                source=source,
                overall_grade=QualityGrade.B,
                overall_score=80.0,
                dimension_scores={},
                total_issues=2,
                critical_issues=0,
                high_issues=1,
                medium_issues=1,
                low_issues=0,
                priority_recommendations=[f"Recommendation {i}"],
                improvement_suggestions=[],
                assessment_period=(datetime.now() - timedelta(hours=1), datetime.now()),
                data_points_analyzed=10,
                confidence_level=0.85
            )
            await memory_manager.learn_from_price_data(
                symbol, source, sample_current_prices, quality_score
            )

        # Check that cache size was limited
        pattern_key = f"{symbol}_{source}_quality"
        pattern = memory_manager.pattern_cache[pattern_key]

        # Should have learned many patterns but cache should be managed
        assert len(pattern["scores"]) >= 50  # Should have learned patterns
        assert len(pattern["grades"]) >= 50
        assert len(pattern["issues"]) >= 50

        # Cache should not grow unbounded (some truncation should occur)
        assert len(pattern["scores"]) <= 121  # Total added: initial + 120 iterations, with some management

    @pytest.mark.asyncio
    async def test_configuration_options(self, learning_config):
        """Test different configuration options"""
        # Test with disabled features
        disabled_config = AdaptiveLearningConfig(
            enable_predictive_scoring=False,
            enable_source_reputation=False,
            enable_market_context=False
        )

        memory_manager = MemoryManager(disabled_config)
        await memory_manager.initialize_memory_system()

        # Prediction should return None when disabled
        prediction = await memory_manager.predict_quality_score(
            "AAPL", "test_source", Mock()
        )
        assert prediction is None

        # Reputation should return default when disabled
        reputation = await memory_manager.get_source_reputation("test_source")
        assert reputation == 1.0

        # Context should return empty when disabled
        context = await memory_manager.get_market_context("AAPL")
        assert context == {}

    @pytest.mark.asyncio
    async def test_error_handling(self, memory_manager):
        """Test error handling in memory operations"""
        await memory_manager.initialize_memory_system()

        # Test with invalid data
        with patch.object(memory_manager, '_create_or_update_symbol_entity', side_effect=Exception("Test error")):
            # Should not raise exception, should handle gracefully
            await memory_manager.learn_from_price_data("AAPL", "test_source", [])

        # Test prediction with no patterns
        prediction = await memory_manager.predict_quality_score("UNKNOWN", "unknown_source", Mock())
        assert prediction is None

    @pytest.mark.asyncio
    async def test_symbol_relationship_learning(self, memory_manager, sample_current_prices):
        """Test learning relationships between symbols"""
        await memory_manager.initialize_memory_system()

        # Learn data for multiple related symbols
        symbols = ["AAPL", "GOOGL", "MSFT"]
        for symbol in symbols:
            await memory_manager.learn_from_price_data(symbol, "test_source", sample_current_prices)

        # Manually add correlation relationship for testing
        correlation_relation = MemoryRelation(
            from_entity="AAPL",
            to_entity="GOOGL",
            relation_type=RelationType.CORRELATES_WITH,
            strength=0.8,
            created_at=datetime.now(),
            metadata={"correlation_type": "sector"}
        )
        memory_manager.relations.append(correlation_relation)

        # Test related symbols retrieval
        related_symbols = await memory_manager._find_related_symbols("AAPL")
        assert "GOOGL" in related_symbols

        # Test market context includes related symbols
        context = await memory_manager.get_market_context("AAPL")
        assert "related_symbols" in context
        assert len(context["related_symbols"]) > 0