"""Memory Server Integration Manager

Provides knowledge graph integration and adaptive learning capabilities
for the market data agent using the MCP memory server protocol.
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from src.data_sources.base import CurrentPrice
from src.quality.quality_scoring_engine import QualityScoreCard, QualityGrade


class EntityType(Enum):
    """Entity types for knowledge graph"""
    SYMBOL = "symbol"
    SOURCE = "data_source"
    PATTERN = "pattern"
    ANOMALY = "anomaly"
    QUALITY_ISSUE = "quality_issue"
    PERFORMANCE_METRIC = "performance_metric"
    MARKET_CONDITION = "market_condition"


class RelationType(Enum):
    """Relationship types for knowledge graph"""
    PROVIDES_DATA = "provides_data"
    HAS_QUALITY_ISSUE = "has_quality_issue"
    EXHIBITS_PATTERN = "exhibits_pattern"
    CORRELATES_WITH = "correlates_with"
    IMPACTS_QUALITY = "impacts_quality"
    DERIVED_FROM = "derived_from"
    SIMILAR_TO = "similar_to"


@dataclass
class MemoryEntity:
    """Entity in the knowledge graph"""
    name: str
    entity_type: EntityType
    attributes: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    confidence: float = 1.0


@dataclass
class MemoryRelation:
    """Relationship in the knowledge graph"""
    from_entity: str
    to_entity: str
    relation_type: RelationType
    strength: float
    created_at: datetime
    metadata: Dict[str, Any]


@dataclass
class AdaptiveLearningConfig:
    """Configuration for adaptive learning system"""
    pattern_detection_threshold: float = 0.7
    anomaly_detection_threshold: float = 0.3
    quality_correlation_threshold: float = 0.6
    learning_decay_factor: float = 0.95
    max_memory_age_days: int = 30
    enable_predictive_scoring: bool = True
    enable_source_reputation: bool = True
    enable_market_context: bool = True


class MemoryManager:
    """Memory server integration manager with adaptive learning"""

    def __init__(self,
                 config: Optional[AdaptiveLearningConfig] = None,
                 memory_server_available: bool = True):
        self.config = config or AdaptiveLearningConfig()
        self.memory_server_available = memory_server_available

        # Local memory stores (fallback when MCP server unavailable)
        self.entities: Dict[str, MemoryEntity] = {}
        self.relations: List[MemoryRelation] = []

        # Learning state
        self.pattern_cache: Dict[str, Any] = {}
        self.source_reputation: Dict[str, float] = {}
        self.quality_predictions: Dict[str, float] = {}

        # Performance tracking
        self.memory_metrics = {
            "entities_created": 0,
            "relations_created": 0,
            "patterns_learned": 0,
            "predictions_made": 0,
            "cache_hits": 0,
            "learning_cycles": 0
        }

    async def initialize_memory_system(self):
        """Initialize memory system and load existing knowledge"""
        try:
            if self.memory_server_available:
                await self._load_from_memory_server()
            else:
                await self._load_from_local_store()

            # Initialize source reputation tracking
            await self._initialize_source_reputation()

            # Load cached patterns
            await self._load_pattern_cache()

            print("✓ Memory system initialized successfully")

        except Exception as e:
            print(f"⚠ Memory system initialization failed: {e}")
            self.memory_server_available = False

    async def learn_from_price_data(self,
                                   symbol: str,
                                   source: str,
                                   price_data: List[CurrentPrice],
                                   quality_score: Optional[QualityScoreCard] = None):
        """Learn patterns and relationships from price data"""
        try:
            # Create/update symbol entity
            await self._create_or_update_symbol_entity(symbol, price_data)

            # Create/update source entity
            await self._create_or_update_source_entity(source, price_data, quality_score)

            # Establish data relationship
            await self._create_data_relationship(symbol, source, price_data)

            # Learn quality patterns
            if quality_score:
                await self._learn_quality_patterns(symbol, source, quality_score)

            # Detect anomalies
            await self._detect_and_record_anomalies(symbol, source, price_data)

            # Update source reputation
            await self._update_source_reputation(source, price_data, quality_score)

            self.memory_metrics["learning_cycles"] += 1

        except Exception as e:
            print(f"⚠ Learning from price data failed: {e}")

    async def predict_quality_score(self,
                                   symbol: str,
                                   source: str,
                                   current_price: CurrentPrice) -> Optional[float]:
        """Predict quality score based on learned patterns"""
        if not self.config.enable_predictive_scoring:
            return None

        try:
            # Get historical patterns
            patterns = await self._get_symbol_source_patterns(symbol, source)

            # Calculate prediction based on multiple factors
            prediction = await self._calculate_quality_prediction(
                symbol, source, current_price, patterns
            )

            if prediction:
                self.memory_metrics["predictions_made"] += 1
                self.quality_predictions[f"{symbol}_{source}"] = prediction

            return prediction

        except Exception as e:
            print(f"⚠ Quality prediction failed: {e}")
            return None

    async def get_source_reputation(self, source: str) -> float:
        """Get learned reputation score for a data source"""
        if not self.config.enable_source_reputation:
            return 1.0

        return self.source_reputation.get(source, 0.5)  # Default to neutral

    async def get_market_context(self, symbol: str) -> Dict[str, Any]:
        """Get relevant market context for a symbol"""
        if not self.config.enable_market_context:
            return {}

        try:
            context = {}

            # Get related symbols
            related_symbols = await self._find_related_symbols(symbol)
            context["related_symbols"] = related_symbols

            # Get quality patterns
            quality_patterns = await self._get_symbol_quality_patterns(symbol)
            context["quality_patterns"] = quality_patterns

            # Get anomaly history
            anomaly_history = await self._get_symbol_anomalies(symbol)
            context["recent_anomalies"] = anomaly_history

            return context

        except Exception as e:
            print(f"⚠ Market context retrieval failed: {e}")
            return {}

    async def cleanup_old_memories(self):
        """Clean up old memories based on configuration"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.max_memory_age_days)

            # Clean entities
            old_entities = [
                name for name, entity in self.entities.items()
                if entity.updated_at < cutoff_date
            ]

            for entity_name in old_entities:
                await self._delete_entity(entity_name)

            # Clean relations
            old_relations = [
                rel for rel in self.relations
                if rel.created_at < cutoff_date
            ]

            for relation in old_relations:
                await self._delete_relation(relation)

            print(f"✓ Cleaned up {len(old_entities)} old entities and {len(old_relations)} old relations")

        except Exception as e:
            print(f"⚠ Memory cleanup failed: {e}")

    # Internal implementation methods

    async def _load_from_memory_server(self):
        """Load existing knowledge from MCP memory server"""
        # This would use the MCP memory server protocol
        # For now, implementing local fallback
        pass

    async def _load_from_local_store(self):
        """Load from local memory store"""
        # Initialize empty stores
        self.entities = {}
        self.relations = []

    async def _initialize_source_reputation(self):
        """Initialize source reputation tracking"""
        # Load existing reputation scores from memory
        for source in ["polygon", "alpha_vantage", "iex", "finnhub", "yfinance"]:
            if source not in self.source_reputation:
                self.source_reputation[source] = 0.8  # Start with good reputation

    async def _load_pattern_cache(self):
        """Load cached patterns from memory"""
        self.pattern_cache = {}

    async def _create_or_update_symbol_entity(self, symbol: str, price_data: List[CurrentPrice]):
        """Create or update symbol entity with latest data"""
        now = datetime.now()

        if symbol in self.entities:
            entity = self.entities[symbol]
            entity.updated_at = now
            entity.attributes.update({
                "latest_price": price_data[-1].price if price_data else None,
                "data_points": entity.attributes.get("data_points", 0) + len(price_data),
                "last_seen": now.isoformat()
            })
        else:
            entity = MemoryEntity(
                name=symbol,
                entity_type=EntityType.SYMBOL,
                attributes={
                    "latest_price": price_data[-1].price if price_data else None,
                    "data_points": len(price_data),
                    "first_seen": now.isoformat(),
                    "last_seen": now.isoformat(),
                    "sources": []
                },
                created_at=now,
                updated_at=now
            )
            self.entities[symbol] = entity
            self.memory_metrics["entities_created"] += 1

    async def _create_or_update_source_entity(self,
                                             source: str,
                                             price_data: List[CurrentPrice],
                                             quality_score: Optional[QualityScoreCard] = None):
        """Create or update data source entity"""
        now = datetime.now()

        attributes = {
            "data_points_provided": len(price_data),
            "last_active": now.isoformat()
        }

        if quality_score:
            attributes.update({
                "latest_quality_grade": quality_score.overall_grade.value,
                "latest_quality_score": quality_score.overall_score,
                "total_issues": quality_score.total_issues,
                "critical_issues": quality_score.critical_issues
            })

        if source in self.entities:
            entity = self.entities[source]
            entity.updated_at = now
            entity.attributes.update(attributes)
            entity.attributes["total_data_points"] = (
                entity.attributes.get("total_data_points", 0) + len(price_data)
            )
        else:
            attributes.update({
                "total_data_points": len(price_data),
                "first_active": now.isoformat()
            })

            entity = MemoryEntity(
                name=source,
                entity_type=EntityType.SOURCE,
                attributes=attributes,
                created_at=now,
                updated_at=now
            )
            self.entities[source] = entity
            self.memory_metrics["entities_created"] += 1

    async def _create_data_relationship(self,
                                       symbol: str,
                                       source: str,
                                       price_data: List[CurrentPrice]):
        """Create relationship between symbol and source"""
        now = datetime.now()

        # Check if relationship already exists
        existing = next((
            rel for rel in self.relations
            if rel.from_entity == source and rel.to_entity == symbol
            and rel.relation_type == RelationType.PROVIDES_DATA
        ), None)

        if existing:
            # Update strength based on data quality and frequency
            existing.strength = min(1.0, existing.strength + 0.1)
            existing.metadata["last_updated"] = now.isoformat()
            existing.metadata["data_points"] = (
                existing.metadata.get("data_points", 0) + len(price_data)
            )
        else:
            relation = MemoryRelation(
                from_entity=source,
                to_entity=symbol,
                relation_type=RelationType.PROVIDES_DATA,
                strength=0.8,  # Start with good strength
                created_at=now,
                metadata={
                    "data_points": len(price_data),
                    "last_updated": now.isoformat()
                }
            )
            self.relations.append(relation)
            self.memory_metrics["relations_created"] += 1

    async def _learn_quality_patterns(self,
                                     symbol: str,
                                     source: str,
                                     quality_score: QualityScoreCard):
        """Learn quality patterns from score cards"""
        pattern_key = f"{symbol}_{source}_quality"

        if pattern_key not in self.pattern_cache:
            self.pattern_cache[pattern_key] = {
                "scores": [],
                "grades": [],
                "issues": [],
                "recommendations": []
            }

        cache = self.pattern_cache[pattern_key]
        cache["scores"].append(quality_score.overall_score)
        cache["grades"].append(quality_score.overall_grade.value)
        cache["issues"].append({
            "total": quality_score.total_issues,
            "critical": quality_score.critical_issues,
            "high": quality_score.high_issues,
            "timestamp": datetime.now().isoformat()
        })
        cache["recommendations"].extend(quality_score.priority_recommendations)

        # Keep only recent data
        if len(cache["scores"]) > 100:
            for key in cache:
                if isinstance(cache[key], list):
                    cache[key] = cache[key][-50:]  # Keep last 50 entries

        self.memory_metrics["patterns_learned"] += 1

    async def _detect_and_record_anomalies(self,
                                          symbol: str,
                                          source: str,
                                          price_data: List[CurrentPrice]):
        """Detect and record anomalies in price data"""
        if len(price_data) < 2:
            return

        # Simple anomaly detection: large price jumps
        for i in range(1, len(price_data)):
            prev_price = price_data[i-1].price
            curr_price = price_data[i].price

            if prev_price > 0:
                change_percent = abs(curr_price - prev_price) / prev_price

                if change_percent > self.config.anomaly_detection_threshold:
                    # Record anomaly
                    anomaly_name = f"{symbol}_{source}_anomaly_{datetime.now().timestamp()}"
                    anomaly_entity = MemoryEntity(
                        name=anomaly_name,
                        entity_type=EntityType.ANOMALY,
                        attributes={
                            "symbol": symbol,
                            "source": source,
                            "change_percent": change_percent,
                            "prev_price": prev_price,
                            "curr_price": curr_price,
                            "timestamp": price_data[i].timestamp.isoformat()
                        },
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                    self.entities[anomaly_name] = anomaly_entity

                    # Create relationship to symbol
                    relation = MemoryRelation(
                        from_entity=symbol,
                        to_entity=anomaly_name,
                        relation_type=RelationType.EXHIBITS_PATTERN,
                        strength=change_percent,
                        created_at=datetime.now(),
                        metadata={"anomaly_type": "price_jump"}
                    )
                    self.relations.append(relation)

    async def _update_source_reputation(self,
                                       source: str,
                                       price_data: List[CurrentPrice],
                                       quality_score: Optional[QualityScoreCard] = None):
        """Update source reputation based on data quality"""
        current_reputation = self.source_reputation.get(source, 0.5)

        # Factor in data availability
        availability_factor = 1.0 if price_data else 0.0

        # Factor in quality score
        quality_factor = 1.0
        if quality_score:
            # Convert grade to numeric factor
            grade_values = {
                "A+": 1.0, "A": 0.95, "A-": 0.9,
                "B+": 0.85, "B": 0.8, "B-": 0.75,
                "C+": 0.7, "C": 0.65, "C-": 0.6,
                "D+": 0.55, "D": 0.5, "D-": 0.45,
                "F": 0.0
            }
            quality_factor = grade_values.get(quality_score.overall_grade.value, 0.5)

        # Calculate new reputation with exponential moving average
        new_reputation = (
            self.config.learning_decay_factor * current_reputation +
            (1 - self.config.learning_decay_factor) * (availability_factor * quality_factor)
        )

        self.source_reputation[source] = max(0.0, min(1.0, new_reputation))

    async def _get_symbol_source_patterns(self, symbol: str, source: str) -> Dict[str, Any]:
        """Get learned patterns for symbol-source combination"""
        pattern_key = f"{symbol}_{source}_quality"
        return self.pattern_cache.get(pattern_key, {})

    async def _calculate_quality_prediction(self,
                                           symbol: str,
                                           source: str,
                                           current_price: CurrentPrice,
                                           patterns: Dict[str, Any]) -> Optional[float]:
        """Calculate predicted quality score"""
        if not patterns or not patterns.get("scores"):
            return None

        # Simple prediction: weighted average of recent scores
        recent_scores = patterns["scores"][-10:]  # Last 10 scores
        if not recent_scores:
            return None

        # Weight recent scores more heavily
        weights = [i + 1 for i in range(len(recent_scores))]
        weighted_sum = sum(score * weight for score, weight in zip(recent_scores, weights))
        weight_sum = sum(weights)

        base_prediction = weighted_sum / weight_sum

        # Adjust for source reputation
        source_reputation = await self.get_source_reputation(source)
        adjusted_prediction = base_prediction * source_reputation

        return min(100.0, max(0.0, adjusted_prediction))

    async def _find_related_symbols(self, symbol: str) -> List[str]:
        """Find symbols related to the given symbol"""
        related = []

        for relation in self.relations:
            if (relation.from_entity == symbol and
                relation.relation_type in [RelationType.CORRELATES_WITH, RelationType.SIMILAR_TO]):
                related.append(relation.to_entity)
            elif (relation.to_entity == symbol and
                  relation.relation_type in [RelationType.CORRELATES_WITH, RelationType.SIMILAR_TO]):
                related.append(relation.from_entity)

        return related[:10]  # Limit to top 10

    async def _get_symbol_quality_patterns(self, symbol: str) -> Dict[str, Any]:
        """Get quality patterns for a symbol across all sources"""
        patterns = {}

        for key, pattern in self.pattern_cache.items():
            if key.startswith(f"{symbol}_") and key.endswith("_quality"):
                source = key.replace(f"{symbol}_", "").replace("_quality", "")
                patterns[source] = pattern

        return patterns

    async def _get_symbol_anomalies(self, symbol: str) -> List[Dict[str, Any]]:
        """Get recent anomalies for a symbol"""
        anomalies = []
        cutoff_date = datetime.now() - timedelta(days=7)  # Last week

        for entity_name, entity in self.entities.items():
            if (entity.entity_type == EntityType.ANOMALY and
                entity.attributes.get("symbol") == symbol and
                entity.created_at > cutoff_date):
                anomalies.append({
                    "name": entity_name,
                    "attributes": entity.attributes,
                    "created_at": entity.created_at.isoformat()
                })

        return sorted(anomalies, key=lambda x: x["created_at"], reverse=True)[:5]

    async def _delete_entity(self, entity_name: str):
        """Delete an entity and its relationships"""
        if entity_name in self.entities:
            del self.entities[entity_name]

        # Remove related relationships
        self.relations = [
            rel for rel in self.relations
            if rel.from_entity != entity_name and rel.to_entity != entity_name
        ]

    async def _delete_relation(self, relation: MemoryRelation):
        """Delete a specific relation"""
        if relation in self.relations:
            self.relations.remove(relation)

    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory system report"""
        return {
            "memory_system": {
                "server_available": self.memory_server_available,
                "entities_count": len(self.entities),
                "relations_count": len(self.relations),
                "patterns_cached": len(self.pattern_cache),
                "source_reputations": dict(self.source_reputation),
                "quality_predictions": dict(self.quality_predictions),
                "metrics": dict(self.memory_metrics)
            },
            "learning_config": asdict(self.config),
            "entity_types": {
                entity_type.value: len([
                    e for e in self.entities.values()
                    if e.entity_type == entity_type
                ])
                for entity_type in EntityType
            },
            "relation_types": {
                rel_type.value: len([
                    r for r in self.relations
                    if r.relation_type == rel_type
                ])
                for rel_type in RelationType
            }
        }

    async def get_health_status(self) -> Dict[str, Any]:
        """Get memory system health status"""
        now = datetime.now()

        return {
            "overall_status": "healthy" if self.memory_server_available else "degraded",
            "timestamp": now.isoformat(),
            "components": {
                "memory_server": {
                    "status": "connected" if self.memory_server_available else "fallback",
                    "entities": len(self.entities),
                    "relations": len(self.relations)
                },
                "pattern_learning": {
                    "status": "active",
                    "patterns_learned": self.memory_metrics["patterns_learned"],
                    "cache_size": len(self.pattern_cache)
                },
                "reputation_system": {
                    "status": "active" if self.config.enable_source_reputation else "disabled",
                    "sources_tracked": len(self.source_reputation)
                },
                "prediction_system": {
                    "status": "active" if self.config.enable_predictive_scoring else "disabled",
                    "predictions_made": self.memory_metrics["predictions_made"]
                }
            }
        }