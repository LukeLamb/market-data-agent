"""Cost Management System

Intelligent API cost tracking and optimization for multi-source data retrieval.
Helps minimize costs while maintaining data quality and availability.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)


class CostTier(Enum):
    """API cost tiers"""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


@dataclass
class APICostConfig:
    """Configuration for API cost tracking"""
    source_name: str
    tier: CostTier

    # Rate limits
    requests_per_second: Optional[int] = None
    requests_per_minute: Optional[int] = None
    requests_per_hour: Optional[int] = None
    requests_per_day: Optional[int] = None
    requests_per_month: Optional[int] = None

    # Cost structure
    cost_per_request: float = 0.0  # Base cost per request
    cost_per_symbol: float = 0.0   # Additional cost per symbol
    monthly_fee: float = 0.0       # Fixed monthly cost

    # Cost by endpoint type
    endpoint_costs: Dict[str, float] = field(default_factory=dict)

    # Premium features cost
    real_time_multiplier: float = 1.0    # Real-time data cost multiplier
    historical_multiplier: float = 1.0   # Historical data cost multiplier
    news_multiplier: float = 1.0         # News data cost multiplier

    # Overage costs (when limits exceeded)
    overage_cost_per_request: float = 0.0

    # Quality metrics (for cost vs quality optimization)
    data_quality_score: float = 80.0     # 0-100 quality rating
    average_latency_ms: float = 500.0    # Average response time
    reliability_score: float = 95.0      # 0-100 reliability rating


class CostManager:
    """Manages API costs across multiple data sources

    Features:
    - Real-time cost tracking and budgeting
    - Cost-aware source selection
    - Budget alerts and limits
    - Cost optimization recommendations
    - ROI analysis for different data sources
    """

    def __init__(self, monthly_budget: float = 1000.0):
        self.monthly_budget = monthly_budget
        self.daily_budget = monthly_budget / 30.0  # Simple daily allocation

        # Source configurations
        self.source_configs: Dict[str, APICostConfig] = {}

        # Usage tracking
        self.daily_usage: Dict[str, Dict[str, float]] = {}  # source -> date -> cost
        self.monthly_usage: Dict[str, float] = {}  # source -> total monthly cost
        self.request_counts: Dict[str, Dict[str, int]] = {}  # source -> endpoint -> count

        # Budget tracking
        self.budget_alerts_sent: Dict[str, datetime] = {}
        self.budget_exceeded: Dict[str, bool] = {}

        # Cost optimization
        self.cost_history: List[Dict] = []
        self.optimization_suggestions: List[Dict] = []

    def add_source_config(self, config: APICostConfig) -> None:
        """Add cost configuration for a data source"""
        self.source_configs[config.source_name] = config

        # Initialize tracking structures
        if config.source_name not in self.daily_usage:
            self.daily_usage[config.source_name] = {}
        if config.source_name not in self.monthly_usage:
            self.monthly_usage[config.source_name] = 0.0
        if config.source_name not in self.request_counts:
            self.request_counts[config.source_name] = {}

        logger.info(f"Added cost config for {config.source_name} ({config.tier.value} tier)")

    def calculate_request_cost(self, source_name: str, endpoint: str,
                             symbols: List[str] = None, is_real_time: bool = False,
                             is_historical: bool = False, is_news: bool = False) -> float:
        """Calculate the cost of a specific request

        Args:
            source_name: Name of the data source
            endpoint: API endpoint being called
            symbols: List of symbols being requested
            is_real_time: Whether this is real-time data
            is_historical: Whether this is historical data
            is_news: Whether this is news data

        Returns:
            Estimated cost of the request
        """
        if source_name not in self.source_configs:
            logger.warning(f"No cost config found for source {source_name}")
            return 0.0

        config = self.source_configs[source_name]

        # Base cost calculation
        base_cost = config.cost_per_request

        # Add per-symbol costs
        if symbols and config.cost_per_symbol > 0:
            base_cost += len(symbols) * config.cost_per_symbol

        # Add endpoint-specific costs
        if endpoint in config.endpoint_costs:
            base_cost += config.endpoint_costs[endpoint]

        # Apply multipliers for data type
        if is_real_time:
            base_cost *= config.real_time_multiplier
        elif is_historical:
            base_cost *= config.historical_multiplier
        elif is_news:
            base_cost *= config.news_multiplier

        return base_cost

    def record_request(self, source_name: str, endpoint: str, cost: float,
                      timestamp: Optional[datetime] = None) -> None:
        """Record a completed API request and its cost

        Args:
            source_name: Name of the data source
            endpoint: API endpoint that was called
            cost: Actual cost of the request
            timestamp: When the request was made (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        date_key = timestamp.strftime("%Y-%m-%d")

        # Update daily usage
        if source_name not in self.daily_usage:
            self.daily_usage[source_name] = {}
        if date_key not in self.daily_usage[source_name]:
            self.daily_usage[source_name][date_key] = 0.0
        self.daily_usage[source_name][date_key] += cost

        # Update monthly usage
        if source_name not in self.monthly_usage:
            self.monthly_usage[source_name] = 0.0
        self.monthly_usage[source_name] += cost

        # Update request counts
        if source_name not in self.request_counts:
            self.request_counts[source_name] = {}
        if endpoint not in self.request_counts[source_name]:
            self.request_counts[source_name][endpoint] = 0
        self.request_counts[source_name][endpoint] += 1

        # Record in cost history
        self.cost_history.append({
            "timestamp": timestamp.isoformat(),
            "source": source_name,
            "endpoint": endpoint,
            "cost": cost
        })

        # Trim cost history to last 1000 entries
        if len(self.cost_history) > 1000:
            self.cost_history = self.cost_history[-1000:]

        # Check budget limits
        self._check_budget_limits(source_name)

    def _check_budget_limits(self, source_name: str) -> None:
        """Check if budget limits have been exceeded"""
        today = datetime.now().strftime("%Y-%m-%d")

        # Check daily budget
        daily_cost = self.daily_usage.get(source_name, {}).get(today, 0.0)
        if daily_cost > self.daily_budget * 0.8:  # 80% of daily budget
            if source_name not in self.budget_alerts_sent or \
               (datetime.now() - self.budget_alerts_sent[source_name]).days >= 1:
                logger.warning(f"Source {source_name} approaching daily budget limit: ${daily_cost:.2f}")
                self.budget_alerts_sent[source_name] = datetime.now()

        # Check monthly budget
        monthly_cost = self.monthly_usage.get(source_name, 0.0)
        if monthly_cost > self.monthly_budget * 0.9:  # 90% of monthly budget
            if not self.budget_exceeded.get(source_name, False):
                logger.error(f"Source {source_name} approaching monthly budget limit: ${monthly_cost:.2f}")
                self.budget_exceeded[source_name] = True

    def get_cost_efficient_sources(self, required_symbols: List[str],
                                 endpoint_type: str = "current_price") -> List[Tuple[str, float]]:
        """Get sources ranked by cost efficiency for a given request

        Args:
            required_symbols: Symbols needed for the request
            endpoint_type: Type of endpoint (current_price, historical_data, etc.)

        Returns:
            List of (source_name, estimated_cost) tuples, sorted by cost efficiency
        """
        source_costs = []

        for source_name, config in self.source_configs.items():
            # Skip if budget exceeded
            if self.budget_exceeded.get(source_name, False):
                continue

            # Calculate cost for this source
            cost = self.calculate_request_cost(
                source_name, endpoint_type, required_symbols
            )

            # Adjust cost based on quality and reliability
            quality_factor = config.data_quality_score / 100.0
            reliability_factor = config.reliability_score / 100.0

            # Cost efficiency = cost / (quality * reliability)
            efficiency_score = cost / (quality_factor * reliability_factor)

            source_costs.append((source_name, cost, efficiency_score))

        # Sort by efficiency score (lower is better)
        source_costs.sort(key=lambda x: x[2])

        return [(source, cost) for source, cost, _ in source_costs]

    def suggest_optimizations(self) -> List[Dict[str, Any]]:
        """Generate cost optimization suggestions

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        # Analyze usage patterns
        total_monthly_cost = sum(self.monthly_usage.values())

        if total_monthly_cost > self.monthly_budget * 0.9:
            suggestions.append({
                "type": "budget_warning",
                "severity": "high",
                "message": f"Monthly costs (${total_monthly_cost:.2f}) approaching budget limit (${self.monthly_budget:.2f})",
                "recommendation": "Consider optimizing request patterns or upgrading to more cost-effective plans"
            })

        # Find most expensive sources
        if self.monthly_usage:
            most_expensive = max(self.monthly_usage.items(), key=lambda x: x[1])
            if most_expensive[1] > total_monthly_cost * 0.5:
                suggestions.append({
                    "type": "source_optimization",
                    "severity": "medium",
                    "message": f"Source '{most_expensive[0]}' accounts for {most_expensive[1]/total_monthly_cost*100:.1f}% of costs",
                    "recommendation": "Consider alternative sources or optimizing usage patterns for this source"
                })

        # Check for underutilized premium features
        for source_name, config in self.source_configs.items():
            if config.monthly_fee > 0 and config.tier in [CostTier.PREMIUM, CostTier.ENTERPRISE]:
                monthly_requests = sum(self.request_counts.get(source_name, {}).values())
                if monthly_requests < 1000:  # Arbitrary threshold
                    suggestions.append({
                        "type": "underutilization",
                        "severity": "low",
                        "message": f"Premium source '{source_name}' has low usage ({monthly_requests} requests)",
                        "recommendation": "Consider downgrading to a lower tier or increasing usage to justify cost"
                    })

        self.optimization_suggestions = suggestions
        return suggestions

    def get_usage_report(self) -> Dict[str, Any]:
        """Generate comprehensive usage and cost report

        Returns:
            Detailed usage report
        """
        total_monthly_cost = sum(self.monthly_usage.values())

        # Calculate daily averages
        daily_averages = {}
        for source_name, daily_costs in self.daily_usage.items():
            if daily_costs:
                daily_averages[source_name] = sum(daily_costs.values()) / len(daily_costs)
            else:
                daily_averages[source_name] = 0.0

        # Find most used endpoints
        all_endpoints = {}
        for source_name, endpoints in self.request_counts.items():
            for endpoint, count in endpoints.items():
                key = f"{source_name}:{endpoint}"
                all_endpoints[key] = count

        top_endpoints = sorted(all_endpoints.items(), key=lambda x: x[1], reverse=True)[:10]

        # Budget utilization
        budget_utilization = total_monthly_cost / self.monthly_budget if self.monthly_budget > 0 else 0

        return {
            "period": "monthly",
            "total_cost": total_monthly_cost,
            "budget": self.monthly_budget,
            "budget_utilization": budget_utilization,
            "remaining_budget": max(0, self.monthly_budget - total_monthly_cost),
            "cost_by_source": dict(self.monthly_usage),
            "daily_averages": daily_averages,
            "top_endpoints": dict(top_endpoints),
            "total_requests": sum(sum(endpoints.values()) for endpoints in self.request_counts.values()),
            "average_cost_per_request": total_monthly_cost / max(1, sum(sum(endpoints.values()) for endpoints in self.request_counts.values())),
            "optimization_suggestions": self.suggest_optimizations()
        }

    def reset_monthly_usage(self) -> None:
        """Reset monthly usage counters (typically called at month start)"""
        self.monthly_usage = {source: 0.0 for source in self.source_configs.keys()}
        self.budget_exceeded = {source: False for source in self.source_configs.keys()}
        self.budget_alerts_sent.clear()
        logger.info("Reset monthly usage counters")

    def export_usage_data(self, filepath: str) -> None:
        """Export usage data to JSON file for analysis"""
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "monthly_usage": self.monthly_usage,
            "daily_usage": self.daily_usage,
            "request_counts": self.request_counts,
            "cost_history": self.cost_history[-100:],  # Last 100 entries
            "source_configs": {
                name: {
                    "tier": config.tier.value,
                    "cost_per_request": config.cost_per_request,
                    "data_quality_score": config.data_quality_score,
                    "reliability_score": config.reliability_score
                }
                for name, config in self.source_configs.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported usage data to {filepath}")

    def load_predefined_configs(self) -> None:
        """Load predefined cost configurations for common data sources"""
        configs = [
            APICostConfig(
                source_name="yfinance",
                tier=CostTier.FREE,
                requests_per_second=2,
                data_quality_score=75.0,
                average_latency_ms=800.0,
                reliability_score=85.0
            ),
            APICostConfig(
                source_name="alpha_vantage",
                tier=CostTier.FREE,
                requests_per_minute=5,
                requests_per_day=500,
                cost_per_request=0.0,
                data_quality_score=85.0,
                average_latency_ms=600.0,
                reliability_score=90.0
            ),
            APICostConfig(
                source_name="iex_cloud",
                tier=CostTier.BASIC,
                requests_per_month=500000,
                cost_per_request=0.0001,  # $0.0001 per request
                endpoint_costs={"real_time": 0.0002},
                data_quality_score=90.0,
                average_latency_ms=300.0,
                reliability_score=95.0
            ),
            APICostConfig(
                source_name="twelve_data",
                tier=CostTier.BASIC,
                requests_per_day=800,
                requests_per_minute=8,
                cost_per_request=0.0,
                data_quality_score=88.0,
                average_latency_ms=400.0,
                reliability_score=92.0
            ),
            APICostConfig(
                source_name="finnhub",
                tier=CostTier.BASIC,
                requests_per_minute=60,
                cost_per_request=0.0,
                endpoint_costs={"news": 0.001},
                data_quality_score=92.0,
                average_latency_ms=350.0,
                reliability_score=94.0
            ),
            APICostConfig(
                source_name="polygon",
                tier=CostTier.PREMIUM,
                requests_per_minute=5,  # Free tier
                cost_per_request=0.002,  # Premium tier cost
                monthly_fee=99.0,  # Premium subscription
                real_time_multiplier=1.5,
                data_quality_score=95.0,
                average_latency_ms=200.0,
                reliability_score=98.0
            )
        ]

        for config in configs:
            self.add_source_config(config)