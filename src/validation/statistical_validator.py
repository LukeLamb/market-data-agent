"""Statistical Validation Implementation

Advanced statistical analysis for detecting anomalies, outliers, and data quality issues
in financial market data using multiple statistical methods and machine learning techniques.
"""

import numpy as np
import statistics
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Optional scipy import for advanced statistical methods
try:
    from scipy import stats
    from scipy.stats import zscore, iqr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available - using basic statistical methods only")


class AnomalyType(Enum):
    """Types of anomalies that can be detected"""
    OUTLIER = "outlier"                    # Statistical outlier
    SUDDEN_CHANGE = "sudden_change"        # Rapid price movement
    VOLUME_ANOMALY = "volume_anomaly"      # Unusual trading volume
    PRICE_GAP = "price_gap"               # Large gap between consecutive prices
    STALE_DATA = "stale_data"             # Data hasn't updated recently
    IMPOSSIBLE_VALUE = "impossible_value"  # Mathematically impossible values
    CONSISTENCY_ERROR = "consistency_error" # Internal data inconsistency
    TREND_REVERSAL = "trend_reversal"     # Unexpected trend change


@dataclass
class ValidationResult:
    """Result of statistical validation"""
    is_valid: bool
    anomaly_type: Optional[AnomalyType] = None
    confidence: float = 0.0  # 0.0 to 1.0
    severity: str = "info"   # info, warning, error, critical
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "is_valid": self.is_valid,
            "anomaly_type": self.anomaly_type.value if self.anomaly_type else None,
            "confidence": self.confidence,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


class StatisticalValidator:
    """Advanced statistical validator for financial market data

    Features:
    - Multiple anomaly detection algorithms
    - Adaptive thresholds based on market conditions
    - Time-series analysis and trend detection
    - Volume and price correlation analysis
    - Statistical significance testing
    """

    def __init__(self,
                 lookback_window: int = 50,
                 outlier_threshold: float = 3.0,
                 volume_threshold: float = 5.0,
                 gap_threshold: float = 0.1,
                 staleness_threshold: int = 300):
        """Initialize statistical validator

        Args:
            lookback_window: Number of historical data points to consider
            outlier_threshold: Z-score threshold for outlier detection
            volume_threshold: Threshold for volume anomaly detection
            gap_threshold: Threshold for price gap detection (as percentage)
            staleness_threshold: Maximum age in seconds before data is stale
        """
        self.lookback_window = lookback_window
        self.outlier_threshold = outlier_threshold
        self.volume_threshold = volume_threshold
        self.gap_threshold = gap_threshold
        self.staleness_threshold = staleness_threshold

        # Historical data for comparison
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.volume_history: Dict[str, List[Tuple[datetime, float]]] = {}

        # Statistics cache
        self.stats_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_expiry: Dict[str, datetime] = {}

    def validate_price_data(self, symbol: str, price: float,
                          timestamp: Optional[datetime] = None,
                          volume: Optional[float] = None) -> ValidationResult:
        """Validate a single price data point

        Args:
            symbol: Stock symbol
            price: Current price
            timestamp: Price timestamp (defaults to now)
            volume: Trading volume (optional)

        Returns:
            ValidationResult with validation status and details
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Basic sanity checks
        basic_result = self._validate_basic_constraints(symbol, price, timestamp, volume)
        if not basic_result.is_valid:
            return basic_result

        # Statistical analysis requires historical data
        if symbol not in self.price_history or len(self.price_history[symbol]) < 5:
            self._update_history(symbol, price, timestamp, volume)
            return ValidationResult(
                is_valid=True,
                message="Insufficient historical data for statistical validation",
                details={"data_points": len(self.price_history.get(symbol, []))}
            )

        # Perform comprehensive statistical validation
        results = []

        # Outlier detection
        results.append(self._detect_price_outliers(symbol, price))

        # Sudden change detection
        results.append(self._detect_sudden_changes(symbol, price, timestamp))

        # Volume anomaly detection (if volume provided)
        if volume is not None:
            results.append(self._detect_volume_anomalies(symbol, volume))

        # Price gap detection
        results.append(self._detect_price_gaps(symbol, price, timestamp))

        # Trend analysis
        results.append(self._analyze_trend_consistency(symbol, price))

        # Update history after analysis
        self._update_history(symbol, price, timestamp, volume)

        # Combine results
        return self._combine_validation_results(results)

    def _validate_basic_constraints(self, symbol: str, price: float,
                                  timestamp: datetime, volume: Optional[float]) -> ValidationResult:
        """Validate basic data constraints"""

        # Price must be positive
        if price <= 0:
            return ValidationResult(
                is_valid=False,
                anomaly_type=AnomalyType.IMPOSSIBLE_VALUE,
                confidence=1.0,
                severity="error",
                message=f"Price must be positive, got {price}",
                details={"price": price}
            )

        # Price must be reasonable (not extreme values)
        if price > 1000000:  # $1M per share seems unreasonable
            return ValidationResult(
                is_valid=False,
                anomaly_type=AnomalyType.IMPOSSIBLE_VALUE,
                confidence=0.9,
                severity="warning",
                message=f"Price seems unreasonably high: ${price:,.2f}",
                details={"price": price}
            )

        # Volume validation
        if volume is not None and volume < 0:
            return ValidationResult(
                is_valid=False,
                anomaly_type=AnomalyType.IMPOSSIBLE_VALUE,
                confidence=1.0,
                severity="error",
                message=f"Volume cannot be negative, got {volume}",
                details={"volume": volume}
            )

        # Staleness check
        age_seconds = (datetime.now() - timestamp).total_seconds()
        if age_seconds > self.staleness_threshold:
            return ValidationResult(
                is_valid=False,
                anomaly_type=AnomalyType.STALE_DATA,
                confidence=0.8,
                severity="warning",
                message=f"Data is {age_seconds:.0f} seconds old",
                details={"age_seconds": age_seconds, "threshold": self.staleness_threshold}
            )

        return ValidationResult(is_valid=True)

    def _detect_price_outliers(self, symbol: str, price: float) -> ValidationResult:
        """Detect if price is a statistical outlier"""
        history = [p for _, p in self.price_history[symbol][-self.lookback_window:]]

        if len(history) < 10:
            return ValidationResult(is_valid=True)

        # Calculate z-score
        mean_price = statistics.mean(history)
        std_price = statistics.stdev(history) if len(history) > 1 else 0

        if std_price == 0:
            return ValidationResult(is_valid=True)

        z_score = abs(price - mean_price) / std_price

        if z_score > self.outlier_threshold:
            return ValidationResult(
                is_valid=False,
                anomaly_type=AnomalyType.OUTLIER,
                confidence=min(1.0, z_score / self.outlier_threshold),
                severity="warning" if z_score < 5 else "error",
                message=f"Price ${price:.2f} is {z_score:.2f} standard deviations from mean ${mean_price:.2f}",
                details={
                    "z_score": z_score,
                    "mean": mean_price,
                    "std": std_price,
                    "threshold": self.outlier_threshold
                }
            )

        return ValidationResult(is_valid=True)

    def _detect_sudden_changes(self, symbol: str, price: float, timestamp: datetime) -> ValidationResult:
        """Detect sudden price changes"""
        if len(self.price_history[symbol]) == 0:
            return ValidationResult(is_valid=True)

        last_timestamp, last_price = self.price_history[symbol][-1]
        time_diff = (timestamp - last_timestamp).total_seconds()

        # Skip if too much time has passed (market closed, etc.)
        if time_diff > 3600:  # 1 hour
            return ValidationResult(is_valid=True)

        # Calculate percentage change
        price_change = abs(price - last_price) / last_price

        # Dynamic threshold based on time elapsed
        # Allow larger changes over longer periods
        time_factor = min(1.0, time_diff / 60)  # Normalize to minutes
        adjusted_threshold = 0.05 * (1 + time_factor)  # Base 5% threshold

        if price_change > adjusted_threshold:
            return ValidationResult(
                is_valid=False,
                anomaly_type=AnomalyType.SUDDEN_CHANGE,
                confidence=min(1.0, price_change / adjusted_threshold),
                severity="warning" if price_change < 0.2 else "error",
                message=f"Sudden price change: {price_change:.2%} in {time_diff:.0f} seconds",
                details={
                    "price_change": price_change,
                    "time_diff": time_diff,
                    "threshold": adjusted_threshold,
                    "previous_price": last_price
                }
            )

        return ValidationResult(is_valid=True)

    def _detect_volume_anomalies(self, symbol: str, volume: float) -> ValidationResult:
        """Detect volume anomalies"""
        if symbol not in self.volume_history or len(self.volume_history[symbol]) < 10:
            return ValidationResult(is_valid=True)

        history = [v for _, v in self.volume_history[symbol][-self.lookback_window:]]

        # Calculate volume statistics
        median_volume = statistics.median(history)

        if median_volume == 0:
            return ValidationResult(is_valid=True)

        # Use ratio to median for volume anomaly detection
        volume_ratio = volume / median_volume

        if volume_ratio > self.volume_threshold or volume_ratio < (1 / self.volume_threshold):
            anomaly_type = AnomalyType.VOLUME_ANOMALY
            severity = "warning"

            if volume_ratio > 10 or volume_ratio < 0.1:
                severity = "error"

            return ValidationResult(
                is_valid=False,
                anomaly_type=anomaly_type,
                confidence=min(1.0, max(volume_ratio, 1/volume_ratio) / self.volume_threshold),
                severity=severity,
                message=f"Volume {volume:,.0f} is {volume_ratio:.1f}x median volume {median_volume:,.0f}",
                details={
                    "volume": volume,
                    "median_volume": median_volume,
                    "ratio": volume_ratio,
                    "threshold": self.volume_threshold
                }
            )

        return ValidationResult(is_valid=True)

    def _detect_price_gaps(self, symbol: str, price: float, timestamp: datetime) -> ValidationResult:
        """Detect significant price gaps"""
        if len(self.price_history[symbol]) == 0:
            return ValidationResult(is_valid=True)

        last_timestamp, last_price = self.price_history[symbol][-1]
        time_diff = (timestamp - last_timestamp).total_seconds()

        # Only check for gaps if reasonable time has passed
        if time_diff < 60:  # Less than 1 minute
            return ValidationResult(is_valid=True)

        gap_percentage = abs(price - last_price) / last_price

        if gap_percentage > self.gap_threshold:
            return ValidationResult(
                is_valid=False,
                anomaly_type=AnomalyType.PRICE_GAP,
                confidence=min(1.0, gap_percentage / self.gap_threshold),
                severity="warning" if gap_percentage < 0.5 else "error",
                message=f"Price gap of {gap_percentage:.2%} detected",
                details={
                    "gap_percentage": gap_percentage,
                    "previous_price": last_price,
                    "current_price": price,
                    "time_diff": time_diff,
                    "threshold": self.gap_threshold
                }
            )

        return ValidationResult(is_valid=True)

    def _analyze_trend_consistency(self, symbol: str, price: float) -> ValidationResult:
        """Analyze trend consistency and detect reversals"""
        if len(self.price_history[symbol]) < 20:
            return ValidationResult(is_valid=True)

        recent_prices = [p for _, p in self.price_history[symbol][-20:]]

        # Calculate recent trend
        if SCIPY_AVAILABLE:
            # Use linear regression to determine trend
            x = np.arange(len(recent_prices))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_prices)

            # Predict next price based on trend
            predicted_price = slope * len(recent_prices) + intercept
            prediction_error = abs(price - predicted_price) / predicted_price

            # Check if actual price significantly deviates from trend
            if prediction_error > 0.15 and r_value ** 2 > 0.5:  # Strong trend with significant deviation
                return ValidationResult(
                    is_valid=False,
                    anomaly_type=AnomalyType.TREND_REVERSAL,
                    confidence=min(1.0, prediction_error * r_value ** 2),
                    severity="info",
                    message=f"Potential trend reversal: price {price:.2f} vs predicted {predicted_price:.2f}",
                    details={
                        "predicted_price": predicted_price,
                        "actual_price": price,
                        "prediction_error": prediction_error,
                        "trend_strength": r_value ** 2,
                        "slope": slope
                    }
                )
        else:
            # Simple trend analysis without scipy
            first_half = recent_prices[:10]
            second_half = recent_prices[10:]

            first_avg = statistics.mean(first_half)
            second_avg = statistics.mean(second_half)

            # Determine trend direction
            trend_up = second_avg > first_avg

            # Check if current price contradicts trend
            if trend_up and price < second_avg * 0.95:
                return ValidationResult(
                    is_valid=False,
                    anomaly_type=AnomalyType.TREND_REVERSAL,
                    confidence=0.6,
                    severity="info",
                    message=f"Downward move against upward trend",
                    details={
                        "trend_direction": "up",
                        "recent_average": second_avg,
                        "current_price": price
                    }
                )
            elif not trend_up and price > second_avg * 1.05:
                return ValidationResult(
                    is_valid=False,
                    anomaly_type=AnomalyType.TREND_REVERSAL,
                    confidence=0.6,
                    severity="info",
                    message=f"Upward move against downward trend",
                    details={
                        "trend_direction": "down",
                        "recent_average": second_avg,
                        "current_price": price
                    }
                )

        return ValidationResult(is_valid=True)

    def _update_history(self, symbol: str, price: float,
                       timestamp: datetime, volume: Optional[float]) -> None:
        """Update historical data for future analysis"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        if symbol not in self.volume_history:
            self.volume_history[symbol] = []

        # Add new data point
        self.price_history[symbol].append((timestamp, price))
        if volume is not None:
            self.volume_history[symbol].append((timestamp, volume))

        # Trim history to lookback window
        max_history = self.lookback_window * 2  # Keep extra for better statistics
        if len(self.price_history[symbol]) > max_history:
            self.price_history[symbol] = self.price_history[symbol][-max_history:]
        if len(self.volume_history[symbol]) > max_history:
            self.volume_history[symbol] = self.volume_history[symbol][-max_history:]

    def _combine_validation_results(self, results: List[ValidationResult]) -> ValidationResult:
        """Combine multiple validation results into one"""
        # Filter out valid results
        invalid_results = [r for r in results if not r.is_valid]

        if not invalid_results:
            return ValidationResult(
                is_valid=True,
                message="All statistical validations passed",
                details={"checks_performed": len(results)}
            )

        # Find most severe issue
        severity_order = {"info": 1, "warning": 2, "error": 3, "critical": 4}
        most_severe = max(invalid_results, key=lambda r: severity_order.get(r.severity, 0))

        # Combine details
        combined_details = {
            "validation_checks": len(results),
            "failed_checks": len(invalid_results),
            "issues": [
                {
                    "type": r.anomaly_type.value if r.anomaly_type else "unknown",
                    "confidence": r.confidence,
                    "severity": r.severity,
                    "message": r.message
                }
                for r in invalid_results
            ]
        }

        return ValidationResult(
            is_valid=False,
            anomaly_type=most_severe.anomaly_type,
            confidence=most_severe.confidence,
            severity=most_severe.severity,
            message=f"Statistical validation failed: {most_severe.message}",
            details=combined_details
        )

    def get_validation_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get validation statistics for a symbol"""
        if symbol not in self.price_history:
            return {"error": "No historical data available"}

        prices = [p for _, p in self.price_history[symbol]]

        if len(prices) < 2:
            return {"error": "Insufficient data for statistics"}

        stats = {
            "data_points": len(prices),
            "price_stats": {
                "mean": statistics.mean(prices),
                "median": statistics.median(prices),
                "min": min(prices),
                "max": max(prices),
                "std": statistics.stdev(prices) if len(prices) > 1 else 0
            }
        }

        if symbol in self.volume_history and self.volume_history[symbol]:
            volumes = [v for _, v in self.volume_history[symbol]]
            stats["volume_stats"] = {
                "mean": statistics.mean(volumes),
                "median": statistics.median(volumes),
                "min": min(volumes),
                "max": max(volumes)
            }

        return stats

    def clear_history(self, symbol: Optional[str] = None) -> None:
        """Clear historical data for validation"""
        if symbol:
            self.price_history.pop(symbol, None)
            self.volume_history.pop(symbol, None)
            self.stats_cache.pop(symbol, None)
            self.cache_expiry.pop(symbol, None)
        else:
            self.price_history.clear()
            self.volume_history.clear()
            self.stats_cache.clear()
            self.cache_expiry.clear()