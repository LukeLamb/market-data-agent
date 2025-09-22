"""Cross-Source Validation Implementation

Validates data consistency across multiple data sources using consensus algorithms,
weighted voting, and statistical correlation analysis to identify the most reliable data.
"""

import statistics
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

from ..data_sources.base import CurrentPrice, PriceData

logger = logging.getLogger(__name__)


class SourceAgreement(Enum):
    """Levels of agreement between data sources"""
    PERFECT = "perfect"           # All sources agree exactly
    STRONG = "strong"            # Sources agree within tight tolerance
    MODERATE = "moderate"        # Sources agree within moderate tolerance
    WEAK = "weak"               # Sources disagree but within acceptable range
    POOR = "poor"               # Significant disagreement between sources
    CONFLICT = "conflict"        # Major conflicts that cannot be resolved


@dataclass
class ConsensusResult:
    """Result of cross-source consensus validation"""
    consensus_value: Optional[float]      # Agreed-upon value
    confidence: float                     # Confidence in consensus (0.0-1.0)
    agreement_level: SourceAgreement     # Level of source agreement
    participating_sources: List[str]     # Sources that provided data
    outlier_sources: List[str]          # Sources that were outliers
    weights_used: Dict[str, float]       # Weights applied to each source
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "consensus_value": self.consensus_value,
            "confidence": self.confidence,
            "agreement_level": self.agreement_level.value,
            "participating_sources": self.participating_sources,
            "outlier_sources": self.outlier_sources,
            "weights_used": self.weights_used,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


class CrossSourceValidator:
    """Cross-source data validation using consensus algorithms

    Features:
    - Weighted consensus based on source reliability
    - Outlier detection and removal
    - Temporal correlation analysis
    - Quality-based source weighting
    - Adaptive tolerance based on market conditions
    """

    def __init__(self,
                 tolerance_tight: float = 0.001,    # 0.1% tolerance for tight agreement
                 tolerance_moderate: float = 0.005,  # 0.5% tolerance for moderate agreement
                 tolerance_loose: float = 0.02,     # 2% tolerance for loose agreement
                 min_sources: int = 2,              # Minimum sources for consensus
                 outlier_threshold: float = 2.0):   # Z-score threshold for outlier detection
        """Initialize cross-source validator

        Args:
            tolerance_tight: Tolerance for strong agreement
            tolerance_moderate: Tolerance for moderate agreement
            tolerance_loose: Tolerance for weak agreement
            min_sources: Minimum number of sources required
            outlier_threshold: Z-score threshold for outlier detection
        """
        self.tolerance_tight = tolerance_tight
        self.tolerance_moderate = tolerance_moderate
        self.tolerance_loose = tolerance_loose
        self.min_sources = min_sources
        self.outlier_threshold = outlier_threshold

        # Source reliability tracking
        self.source_reliability: Dict[str, float] = {}
        self.source_history: Dict[str, List[Tuple[datetime, float, float]]] = {}  # timestamp, value, consensus

        # Adaptive parameters
        self.market_volatility_factor = 1.0
        self.last_consensus_update = datetime.now()

    def validate_current_prices(self, price_data: Dict[str, CurrentPrice]) -> ConsensusResult:
        """Validate current price data from multiple sources

        Args:
            price_data: Dictionary mapping source names to CurrentPrice objects

        Returns:
            ConsensusResult with consensus value and validation details
        """
        if len(price_data) < self.min_sources:
            return ConsensusResult(
                consensus_value=None,
                confidence=0.0,
                agreement_level=SourceAgreement.POOR,
                participating_sources=list(price_data.keys()),
                outlier_sources=[],
                weights_used={},
                details={"error": f"Need at least {self.min_sources} sources, got {len(price_data)}"}
            )

        # Extract prices and quality scores
        source_data = {}
        for source_name, price_obj in price_data.items():
            source_data[source_name] = {
                "price": price_obj.price,
                "quality": price_obj.quality_score,
                "timestamp": price_obj.timestamp,
                "volume": price_obj.volume
            }

        return self._calculate_consensus(source_data, "price")

    def validate_historical_data(self, historical_data: Dict[str, List[PriceData]],
                               data_field: str = "close_price") -> List[ConsensusResult]:
        """Validate historical data across sources

        Args:
            historical_data: Dictionary mapping source names to historical data lists
            data_field: Field to validate (close_price, open_price, etc.)

        Returns:
            List of ConsensusResult objects for each time period
        """
        # Align data by timestamp
        aligned_data = self._align_historical_data(historical_data, data_field)

        results = []
        for timestamp, source_values in aligned_data.items():
            if len(source_values) >= self.min_sources:
                # Convert to format expected by consensus calculation
                formatted_data = {
                    source: {
                        "price": value,
                        "quality": 80.0,  # Default quality if not available
                        "timestamp": timestamp
                    }
                    for source, value in source_values.items()
                }

                consensus = self._calculate_consensus(formatted_data, "price")
                results.append(consensus)

        return results

    def _calculate_consensus(self, source_data: Dict[str, Dict[str, Any]], field: str) -> ConsensusResult:
        """Calculate consensus value from multiple sources

        Args:
            source_data: Dictionary of source data
            field: Field name to validate

        Returns:
            ConsensusResult with consensus calculation
        """
        # Extract values and weights
        values = []
        sources = []
        raw_weights = []

        for source_name, data in source_data.items():
            if field in data and data[field] is not None:
                values.append(float(data[field]))
                sources.append(source_name)

                # Calculate weight based on quality and reliability
                quality_weight = data.get("quality", 50.0) / 100.0
                reliability_weight = self.source_reliability.get(source_name, 0.5)
                combined_weight = (quality_weight + reliability_weight) / 2.0
                raw_weights.append(combined_weight)

        if len(values) < self.min_sources:
            return ConsensusResult(
                consensus_value=None,
                confidence=0.0,
                agreement_level=SourceAgreement.POOR,
                participating_sources=sources,
                outlier_sources=[],
                weights_used=dict(zip(sources, raw_weights))
            )

        # Detect and remove outliers
        clean_values, clean_sources, clean_weights, outliers = self._remove_outliers(values, sources, raw_weights)

        if len(clean_values) < self.min_sources:
            return ConsensusResult(
                consensus_value=statistics.median(values),
                confidence=0.3,
                agreement_level=SourceAgreement.CONFLICT,
                participating_sources=sources,
                outlier_sources=outliers,
                weights_used=dict(zip(sources, raw_weights)),
                details={"reason": "Too many outliers detected"}
            )

        # Calculate weighted consensus
        normalized_weights = self._normalize_weights(clean_weights)
        consensus_value = sum(v * w for v, w in zip(clean_values, normalized_weights))

        # Determine agreement level
        agreement_level = self._assess_agreement(clean_values, consensus_value)

        # Calculate confidence based on agreement and number of sources
        confidence = self._calculate_confidence(clean_values, consensus_value, agreement_level, len(clean_sources))

        # Update source reliability based on agreement
        self._update_source_reliability(clean_sources, clean_values, consensus_value)

        return ConsensusResult(
            consensus_value=consensus_value,
            confidence=confidence,
            agreement_level=agreement_level,
            participating_sources=clean_sources,
            outlier_sources=outliers,
            weights_used=dict(zip(clean_sources, normalized_weights)),
            details={
                "raw_values": dict(zip(sources, values)),
                "clean_values": dict(zip(clean_sources, clean_values)),
                "standard_deviation": statistics.stdev(clean_values) if len(clean_values) > 1 else 0,
                "median": statistics.median(clean_values),
                "range": max(clean_values) - min(clean_values) if clean_values else 0
            }
        )

    def _remove_outliers(self, values: List[float], sources: List[str],
                        weights: List[float]) -> Tuple[List[float], List[str], List[float], List[str]]:
        """Remove statistical outliers from the dataset

        Args:
            values: List of values
            sources: List of source names
            weights: List of weights

        Returns:
            Tuple of (clean_values, clean_sources, clean_weights, outlier_sources)
        """
        if len(values) <= 3:  # Don't remove outliers from small datasets
            return values, sources, weights, []

        # Calculate z-scores
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0

        if std_val == 0:  # All values are the same
            return values, sources, weights, []

        z_scores = [abs(v - mean_val) / std_val for v in values]

        # Identify outliers
        clean_indices = [i for i, z in enumerate(z_scores) if z <= self.outlier_threshold]
        outlier_indices = [i for i, z in enumerate(z_scores) if z > self.outlier_threshold]

        clean_values = [values[i] for i in clean_indices]
        clean_sources = [sources[i] for i in clean_indices]
        clean_weights = [weights[i] for i in clean_indices]
        outlier_sources = [sources[i] for i in outlier_indices]

        return clean_values, clean_sources, clean_weights, outlier_sources

    def _normalize_weights(self, weights: List[float]) -> List[float]:
        """Normalize weights to sum to 1.0"""
        total_weight = sum(weights)
        if total_weight == 0:
            return [1.0 / len(weights)] * len(weights)
        return [w / total_weight for w in weights]

    def _assess_agreement(self, values: List[float], consensus: float) -> SourceAgreement:
        """Assess the level of agreement between sources

        Args:
            values: List of values from sources
            consensus: Consensus value

        Returns:
            SourceAgreement level
        """
        if len(values) <= 1:
            return SourceAgreement.PERFECT

        # Calculate relative deviations from consensus
        deviations = [abs(v - consensus) / consensus for v in values if consensus != 0]

        if not deviations:
            return SourceAgreement.PERFECT

        max_deviation = max(deviations)
        avg_deviation = statistics.mean(deviations)

        # Adjust tolerances based on market volatility
        tight_tolerance = self.tolerance_tight * self.market_volatility_factor
        moderate_tolerance = self.tolerance_moderate * self.market_volatility_factor
        loose_tolerance = self.tolerance_loose * self.market_volatility_factor

        # Determine agreement level
        if max_deviation <= tight_tolerance:
            return SourceAgreement.PERFECT if max_deviation <= tight_tolerance / 2 else SourceAgreement.STRONG
        elif max_deviation <= moderate_tolerance:
            return SourceAgreement.MODERATE
        elif max_deviation <= loose_tolerance:
            return SourceAgreement.WEAK
        elif avg_deviation <= loose_tolerance:
            return SourceAgreement.POOR
        else:
            return SourceAgreement.CONFLICT

    def _calculate_confidence(self, values: List[float], consensus: float,
                            agreement: SourceAgreement, num_sources: int) -> float:
        """Calculate confidence in the consensus

        Args:
            values: List of values
            consensus: Consensus value
            agreement: Agreement level
            num_sources: Number of sources

        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Base confidence from agreement level
        agreement_confidence = {
            SourceAgreement.PERFECT: 1.0,
            SourceAgreement.STRONG: 0.9,
            SourceAgreement.MODERATE: 0.7,
            SourceAgreement.WEAK: 0.5,
            SourceAgreement.POOR: 0.3,
            SourceAgreement.CONFLICT: 0.1
        }

        base_confidence = agreement_confidence[agreement]

        # Boost confidence with more sources
        source_factor = min(1.0, num_sources / 5.0)  # Max boost at 5 sources

        # Reduce confidence if values are very spread out
        if len(values) > 1 and consensus != 0:
            coefficient_of_variation = statistics.stdev(values) / abs(consensus)
            spread_penalty = min(0.5, coefficient_of_variation)
        else:
            spread_penalty = 0

        final_confidence = base_confidence * source_factor * (1 - spread_penalty)
        return max(0.0, min(1.0, final_confidence))

    def _update_source_reliability(self, sources: List[str], values: List[float], consensus: float) -> None:
        """Update source reliability scores based on agreement with consensus

        Args:
            sources: List of source names
            values: List of values from sources
            consensus: Consensus value
        """
        if consensus == 0:
            return

        for source, value in zip(sources, values):
            # Calculate how close this source was to consensus
            relative_error = abs(value - consensus) / abs(consensus)

            # Convert to reliability score (lower error = higher reliability)
            agreement_score = max(0.0, 1.0 - relative_error * 10)  # Scale error appropriately

            # Update reliability with exponential moving average
            if source in self.source_reliability:
                alpha = 0.1  # Learning rate
                self.source_reliability[source] = (
                    alpha * agreement_score + (1 - alpha) * self.source_reliability[source]
                )
            else:
                self.source_reliability[source] = agreement_score

            # Keep reliability in reasonable bounds
            self.source_reliability[source] = max(0.1, min(1.0, self.source_reliability[source]))

            # Update source history
            if source not in self.source_history:
                self.source_history[source] = []

            self.source_history[source].append((datetime.now(), value, consensus))

            # Trim history
            if len(self.source_history[source]) > 1000:
                self.source_history[source] = self.source_history[source][-1000:]

    def _align_historical_data(self, historical_data: Dict[str, List[PriceData]],
                             field: str) -> Dict[datetime, Dict[str, float]]:
        """Align historical data by timestamp for cross-source validation

        Args:
            historical_data: Dictionary mapping source names to historical data
            field: Field name to extract

        Returns:
            Dictionary mapping timestamps to source values
        """
        aligned = {}

        for source_name, data_list in historical_data.items():
            for data_point in data_list:
                timestamp = data_point.timestamp.replace(second=0, microsecond=0)  # Round to minute
                value = getattr(data_point, field, None)

                if value is not None:
                    if timestamp not in aligned:
                        aligned[timestamp] = {}
                    aligned[timestamp][source_name] = float(value)

        return aligned

    def get_source_reliability_scores(self) -> Dict[str, float]:
        """Get current reliability scores for all sources"""
        return dict(self.source_reliability)

    def set_source_reliability(self, source: str, reliability: float) -> None:
        """Manually set reliability score for a source

        Args:
            source: Source name
            reliability: Reliability score (0.0 to 1.0)
        """
        self.source_reliability[source] = max(0.0, min(1.0, reliability))

    def update_market_volatility(self, volatility_factor: float) -> None:
        """Update market volatility factor for adaptive tolerances

        Args:
            volatility_factor: Multiplier for tolerance thresholds (1.0 = normal, >1.0 = more volatile)
        """
        self.market_volatility_factor = max(0.5, min(3.0, volatility_factor))
        logger.info(f"Updated market volatility factor to {self.market_volatility_factor}")

    def reset_source_reliability(self, source: Optional[str] = None) -> None:
        """Reset reliability scores for one or all sources

        Args:
            source: Source name to reset, or None to reset all
        """
        if source:
            self.source_reliability.pop(source, None)
            self.source_history.pop(source, None)
        else:
            self.source_reliability.clear()
            self.source_history.clear()

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation performance"""
        if not self.source_history:
            return {"error": "No validation history available"}

        total_validations = sum(len(history) for history in self.source_history.values())

        return {
            "total_validations": total_validations,
            "sources_tracked": len(self.source_reliability),
            "reliability_scores": dict(self.source_reliability),
            "market_volatility_factor": self.market_volatility_factor,
            "last_update": self.last_consensus_update.isoformat(),
            "configuration": {
                "tolerance_tight": self.tolerance_tight,
                "tolerance_moderate": self.tolerance_moderate,
                "tolerance_loose": self.tolerance_loose,
                "min_sources": self.min_sources,
                "outlier_threshold": self.outlier_threshold
            }
        }