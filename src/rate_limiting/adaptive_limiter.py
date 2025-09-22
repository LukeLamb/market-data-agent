"""Adaptive Rate Limiter Implementation

Machine learning-based rate limiting that adapts to API behavior patterns,
server load, and error rates to optimize request success rates.
"""

import asyncio
import time
import statistics
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AdaptationStrategy(Enum):
    """Strategies for rate limit adaptation"""
    CONSERVATIVE = "conservative"  # Slow adaptation, prioritize stability
    BALANCED = "balanced"         # Moderate adaptation
    AGGRESSIVE = "aggressive"     # Fast adaptation, maximize throughput


@dataclass
class ResponseMetrics:
    """Metrics for API response analysis"""
    timestamp: float
    success: bool
    response_time_ms: float
    http_status: int
    error_type: Optional[str] = None
    retry_after: Optional[int] = None


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive rate limiting"""
    source_name: str
    initial_rate: float  # Initial requests per second
    min_rate: float = 0.1  # Minimum rate to prevent complete shutdown
    max_rate: float = 100.0  # Maximum rate to prevent overload
    adaptation_strategy: AdaptationStrategy = AdaptationStrategy.BALANCED
    learning_window: int = 300  # Seconds of history to consider
    adaptation_factor: float = 0.1  # How aggressively to adapt (0.0-1.0)
    error_threshold: float = 0.1  # Error rate threshold for rate reduction
    success_threshold: float = 0.95  # Success rate threshold for rate increase
    response_time_threshold: float = 2000.0  # Response time threshold (ms)


class AdaptiveRateLimiter:
    """Adaptive rate limiter that learns from API behavior

    Features:
    - Machine learning-based rate adaptation
    - Response time and error rate analysis
    - Predictive rate adjustment based on patterns
    - Multiple adaptation strategies
    - Integration with existing rate limiting infrastructure
    """

    def __init__(self, config: AdaptiveConfig):
        self.config = config
        self.current_rate = config.initial_rate

        # Historical data
        self.response_history: List[ResponseMetrics] = []
        self.rate_history: List[Tuple[float, float]] = []  # (timestamp, rate)

        # Adaptation state
        self.last_adaptation = time.time()
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        self.adaptation_paused = False

        # Performance metrics
        self.current_error_rate = 0.0
        self.current_success_rate = 1.0
        self.average_response_time = 0.0
        self.server_load_indicator = 0.0

        # Learning parameters
        self.adaptation_cooldown = 30.0  # Minimum seconds between adaptations
        self.pattern_weights: Dict[str, float] = {
            "error_rate": 0.3,
            "response_time": 0.2,
            "success_rate": 0.3,
            "server_signals": 0.2
        }

    def record_response(self, success: bool, response_time_ms: float,
                       http_status: int, error_type: Optional[str] = None,
                       retry_after: Optional[int] = None) -> None:
        """Record an API response for learning

        Args:
            success: Whether the request was successful
            response_time_ms: Response time in milliseconds
            http_status: HTTP status code
            error_type: Type of error if unsuccessful
            retry_after: Retry-After header value if present
        """
        now = time.time()

        metrics = ResponseMetrics(
            timestamp=now,
            success=success,
            response_time_ms=response_time_ms,
            http_status=http_status,
            error_type=error_type,
            retry_after=retry_after
        )

        self.response_history.append(metrics)

        # Trim history to learning window
        cutoff = now - self.config.learning_window
        self.response_history = [m for m in self.response_history if m.timestamp > cutoff]

        # Update consecutive counters
        if success:
            self.consecutive_successes += 1
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            self.consecutive_successes = 0

        # Trigger adaptation if conditions are met
        self._check_adaptation_triggers()

    def _check_adaptation_triggers(self) -> None:
        """Check if rate adaptation should be triggered"""
        now = time.time()

        # Check cooldown
        if now - self.last_adaptation < self.adaptation_cooldown:
            return

        # Check if we have enough data
        if len(self.response_history) < 10:
            return

        # Calculate current metrics
        self._update_metrics()

        # Determine if adaptation is needed
        should_adapt = False
        adaptation_direction = 0  # -1 for decrease, +1 for increase

        # Error rate trigger
        if self.current_error_rate > self.config.error_threshold:
            should_adapt = True
            adaptation_direction = -1
            logger.info(f"Triggering rate decrease due to high error rate: {self.current_error_rate:.3f}")

        # Success rate trigger for increase
        elif (self.current_success_rate > self.config.success_threshold and
              self.average_response_time < self.config.response_time_threshold):
            should_adapt = True
            adaptation_direction = 1
            logger.debug(f"Triggering rate increase due to good performance")

        # Response time trigger
        elif self.average_response_time > self.config.response_time_threshold * 1.5:
            should_adapt = True
            adaptation_direction = -1
            logger.info(f"Triggering rate decrease due to slow response time: {self.average_response_time:.1f}ms")

        # Consecutive failures trigger
        elif self.consecutive_failures >= 5:
            should_adapt = True
            adaptation_direction = -1
            logger.warning(f"Triggering rate decrease due to consecutive failures: {self.consecutive_failures}")

        # Server load signals
        elif self._detect_server_overload():
            should_adapt = True
            adaptation_direction = -1
            logger.info("Triggering rate decrease due to server overload signals")

        if should_adapt and not self.adaptation_paused:
            self._adapt_rate(adaptation_direction)

    def _update_metrics(self) -> None:
        """Update current performance metrics"""
        if not self.response_history:
            return

        recent_responses = self.response_history[-50:]  # Last 50 responses

        # Calculate error rate
        errors = sum(1 for r in recent_responses if not r.success)
        self.current_error_rate = errors / len(recent_responses)

        # Calculate success rate
        self.current_success_rate = 1.0 - self.current_error_rate

        # Calculate average response time
        response_times = [r.response_time_ms for r in recent_responses if r.success]
        if response_times:
            self.average_response_time = statistics.mean(response_times)

        # Update server load indicator
        self.server_load_indicator = self._calculate_server_load()

    def _detect_server_overload(self) -> bool:
        """Detect server overload based on response patterns"""
        if len(self.response_history) < 20:
            return False

        recent = self.response_history[-20:]

        # Check for 429 (Too Many Requests) responses
        rate_limited = sum(1 for r in recent if r.http_status == 429)
        if rate_limited > 2:
            return True

        # Check for 503 (Service Unavailable) responses
        service_unavailable = sum(1 for r in recent if r.http_status == 503)
        if service_unavailable > 1:
            return True

        # Check for timeout patterns
        timeouts = sum(1 for r in recent if r.error_type == "timeout")
        if timeouts > 3:
            return True

        # Check for increasing response times
        if len(recent) >= 10:
            first_half = recent[:10]
            second_half = recent[10:]

            avg_first = statistics.mean(r.response_time_ms for r in first_half if r.success)
            avg_second = statistics.mean(r.response_time_ms for r in second_half if r.success)

            if avg_first > 0 and avg_second > avg_first * 2:
                return True

        return False

    def _calculate_server_load(self) -> float:
        """Calculate server load indicator (0.0 = low load, 1.0 = high load)"""
        if not self.response_history:
            return 0.0

        recent = self.response_history[-30:]
        load_factors = []

        # Response time factor
        response_times = [r.response_time_ms for r in recent if r.success]
        if response_times:
            avg_time = statistics.mean(response_times)
            time_factor = min(1.0, avg_time / self.config.response_time_threshold)
            load_factors.append(time_factor)

        # Error rate factor
        error_factor = self.current_error_rate / self.config.error_threshold
        load_factors.append(min(1.0, error_factor))

        # Rate limiting signals
        rate_limited = sum(1 for r in recent if r.http_status in [429, 503])
        rate_factor = min(1.0, rate_limited / 5.0)
        load_factors.append(rate_factor)

        return statistics.mean(load_factors) if load_factors else 0.0

    def _adapt_rate(self, direction: int) -> None:
        """Adapt the current rate based on learned patterns

        Args:
            direction: -1 to decrease rate, +1 to increase rate
        """
        old_rate = self.current_rate

        # Calculate adaptation amount based on strategy
        base_factor = self.config.adaptation_factor

        if self.config.adaptation_strategy == AdaptationStrategy.CONSERVATIVE:
            base_factor *= 0.5
        elif self.config.adaptation_strategy == AdaptationStrategy.AGGRESSIVE:
            base_factor *= 2.0

        # Calculate new rate
        if direction > 0:
            # Increase rate (more conservative)
            multiplier = 1.0 + (base_factor * 0.5)
            self.current_rate = min(self.config.max_rate, self.current_rate * multiplier)
        else:
            # Decrease rate (more aggressive for safety)
            multiplier = 1.0 - base_factor
            self.current_rate = max(self.config.min_rate, self.current_rate * multiplier)

        # Record rate change
        self.last_adaptation = time.time()
        self.rate_history.append((self.last_adaptation, self.current_rate))

        # Trim rate history
        cutoff = self.last_adaptation - (self.config.learning_window * 2)
        self.rate_history = [(t, r) for t, r in self.rate_history if t > cutoff]

        logger.info(f"Adapted rate for {self.config.source_name}: {old_rate:.3f} -> {self.current_rate:.3f}")

    def get_current_rate(self) -> float:
        """Get current adaptive rate limit"""
        return self.current_rate

    def get_recommended_delay(self) -> float:
        """Get recommended delay between requests in seconds"""
        if self.current_rate <= 0:
            return 60.0  # Default to 1 minute if rate is 0

        return 1.0 / self.current_rate

    def pause_adaptation(self) -> None:
        """Pause rate adaptation (useful during maintenance or known issues)"""
        self.adaptation_paused = True
        logger.info(f"Paused adaptation for {self.config.source_name}")

    def resume_adaptation(self) -> None:
        """Resume rate adaptation"""
        self.adaptation_paused = False
        logger.info(f"Resumed adaptation for {self.config.source_name}")

    def reset_learning(self) -> None:
        """Reset learning data and return to initial rate"""
        self.response_history.clear()
        self.rate_history.clear()
        self.current_rate = self.config.initial_rate
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        self.last_adaptation = time.time()
        logger.info(f"Reset learning for {self.config.source_name}")

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        self._update_metrics()

        # Calculate rate stability
        if len(self.rate_history) >= 2:
            rates = [r for _, r in self.rate_history[-10:]]
            rate_stability = 1.0 - (statistics.stdev(rates) / statistics.mean(rates)) if len(rates) > 1 else 1.0
        else:
            rate_stability = 1.0

        # Calculate adaptation frequency
        recent_adaptations = [t for t, _ in self.rate_history if time.time() - t < 3600]
        adaptations_per_hour = len(recent_adaptations)

        return {
            "source_name": self.config.source_name,
            "current_rate": self.current_rate,
            "initial_rate": self.config.initial_rate,
            "adaptation_strategy": self.config.adaptation_strategy.value,
            "performance_metrics": {
                "error_rate": self.current_error_rate,
                "success_rate": self.current_success_rate,
                "average_response_time_ms": self.average_response_time,
                "server_load_indicator": self.server_load_indicator
            },
            "learning_stats": {
                "total_responses": len(self.response_history),
                "consecutive_successes": self.consecutive_successes,
                "consecutive_failures": self.consecutive_failures,
                "rate_stability": rate_stability,
                "adaptations_per_hour": adaptations_per_hour
            },
            "current_state": {
                "adaptation_paused": self.adaptation_paused,
                "last_adaptation": datetime.fromtimestamp(self.last_adaptation).isoformat(),
                "recommended_delay_seconds": self.get_recommended_delay()
            }
        }

    def get_rate_trend(self, hours: int = 24) -> List[Tuple[datetime, float]]:
        """Get rate trend over specified time period

        Args:
            hours: Number of hours of history to return

        Returns:
            List of (timestamp, rate) tuples
        """
        cutoff = time.time() - (hours * 3600)
        return [
            (datetime.fromtimestamp(timestamp), rate)
            for timestamp, rate in self.rate_history
            if timestamp > cutoff
        ]

    def predict_optimal_rate(self) -> float:
        """Predict optimal rate based on historical patterns

        Returns:
            Predicted optimal rate
        """
        if len(self.response_history) < 50:
            return self.current_rate

        # Analyze patterns in successful periods
        successful_periods = []
        window_size = 10

        for i in range(len(self.response_history) - window_size):
            window = self.response_history[i:i + window_size]
            success_rate = sum(1 for r in window if r.success) / len(window)
            avg_response_time = statistics.mean(r.response_time_ms for r in window if r.success)

            if success_rate >= 0.9 and avg_response_time < self.config.response_time_threshold:
                # Find corresponding rate during this period
                window_time = window[0].timestamp
                rate_at_time = self._get_rate_at_time(window_time)
                if rate_at_time:
                    successful_periods.append(rate_at_time)

        if successful_periods:
            # Return the median of successful rates
            return statistics.median(successful_periods)
        else:
            # Conservative fallback
            return self.current_rate * 0.8

    def _get_rate_at_time(self, timestamp: float) -> Optional[float]:
        """Get the rate that was active at a specific timestamp"""
        applicable_rates = [(t, r) for t, r in self.rate_history if t <= timestamp]
        if applicable_rates:
            return applicable_rates[-1][1]  # Most recent rate before timestamp
        return None