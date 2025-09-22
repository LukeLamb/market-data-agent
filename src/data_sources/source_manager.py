"""Data Source Manager

This module manages multiple data sources with intelligent failover, health monitoring,
and source prioritization for reliable market data collection.
"""

import asyncio
import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Type, Union
from enum import Enum
import logging

from .base import (
    BaseDataSource,
    PriceData,
    CurrentPrice,
    HealthStatus,
    DataSourceStatus,
    DataSourceError,
    SymbolNotFoundError,
    RateLimitError
)
from .yfinance_source import YFinanceSource
from .alpha_vantage_source import AlphaVantageSource
from ..validation.data_validator import DataValidator, ValidationResult

logger = logging.getLogger(__name__)


class SourcePriority(Enum):
    """Source priority levels"""
    PRIMARY = 1
    SECONDARY = 2
    BACKUP = 3
    DISABLED = 999


class SourceManagerError(Exception):
    """Base exception for source manager errors"""
    pass


class NoHealthySourceError(SourceManagerError):
    """Raised when no healthy sources are available"""
    pass


class DataSourceManager:
    """Manages multiple data sources with intelligent failover

    Features:
    - Multi-source management with priority-based selection
    - Intelligent failover and health monitoring
    - Data validation and quality scoring
    - Circuit breaker pattern for failed sources
    - Load balancing and rate limit management
    - Source reliability tracking
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize data source manager

        Args:
            config: Configuration dictionary with source settings
        """
        self.config = config or {}
        self.sources: Dict[str, BaseDataSource] = {}
        self.source_priorities: Dict[str, SourcePriority] = {}
        self.source_reliability: Dict[str, float] = {}
        self.circuit_breaker_states: Dict[str, Dict[str, Any]] = {}

        # Configuration
        self.max_failure_threshold = self.config.get("max_failure_threshold", 5)
        self.circuit_breaker_timeout = self.config.get("circuit_breaker_timeout", 300)  # 5 minutes
        self.health_check_interval = self.config.get("health_check_interval", 60)  # 1 minute
        self.validation_enabled = self.config.get("validation_enabled", True)

        # Initialize data validator
        self.validator = DataValidator(self.config.get("validation", {}))

        # Health monitoring
        self.last_health_check = {}
        self.health_check_task = None

        logger.info("Initialized data source manager")

    def register_source(
        self,
        name: str,
        source: BaseDataSource,
        priority: SourcePriority = SourcePriority.SECONDARY
    ) -> None:
        """Register a data source

        Args:
            name: Unique name for the source
            source: Data source instance
            priority: Source priority level
        """
        self.sources[name] = source
        self.source_priorities[name] = priority
        self.source_reliability[name] = 1.0  # Start with perfect reliability
        self.circuit_breaker_states[name] = {
            "state": "closed",  # closed, open, half-open
            "failure_count": 0,
            "last_failure": None,
            "next_attempt": None
        }

        logger.info(f"Registered data source: {name} (priority: {priority.name})")

    def register_yfinance_source(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Register YFinance data source

        Args:
            config: YFinance-specific configuration
        """
        yf_config = config or self.config.get("yfinance", {})
        source = YFinanceSource(yf_config)
        priority = SourcePriority(yf_config.get("priority", SourcePriority.PRIMARY.value))
        self.register_source("yfinance", source, priority)

    def register_alpha_vantage_source(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Register Alpha Vantage data source

        Args:
            config: Alpha Vantage-specific configuration
        """
        av_config = config or self.config.get("alpha_vantage", {})
        source = AlphaVantageSource(av_config)
        priority = SourcePriority(av_config.get("priority", SourcePriority.SECONDARY.value))
        self.register_source("alpha_vantage", source, priority)

    async def get_current_price(self, symbol: str) -> CurrentPrice:
        """Get current price with intelligent source selection

        Args:
            symbol: Stock symbol to fetch

        Returns:
            CurrentPrice object from the best available source

        Raises:
            NoHealthySourceError: If no healthy sources are available
            SymbolNotFoundError: If symbol is not found in any source
        """
        sources = self._get_prioritized_sources()

        last_error = None
        symbol_not_found_count = 0

        for source_name in sources:
            if not self._is_source_available(source_name):
                continue

            try:
                source = self.sources[source_name]
                price = await source.get_current_price(symbol)

                # Validate if enabled
                if self.validation_enabled:
                    validation_result = self.validator.validate_current_price(price)
                    if not validation_result.is_valid:
                        logger.warning(f"Invalid price data from {source_name}: {validation_result.issues}")
                        continue

                    # Update quality score from validation
                    price.quality_score = validation_result.quality_score

                # Update source reliability on success
                self._update_source_reliability(source_name, True)
                self._reset_circuit_breaker(source_name)

                logger.debug(f"Successfully fetched current price for {symbol} from {source_name}")
                return price

            except SymbolNotFoundError:
                symbol_not_found_count += 1
                logger.debug(f"Symbol {symbol} not found in {source_name}")
                continue

            except (RateLimitError, DataSourceError) as e:
                logger.warning(f"Failed to get current price from {source_name}: {e}")
                self._update_source_reliability(source_name, False)
                self._update_circuit_breaker(source_name, e)
                last_error = e
                continue

        # If all sources returned symbol not found, raise that error
        if symbol_not_found_count == len(sources):
            raise SymbolNotFoundError(f"Symbol {symbol} not found in any data source")

        # If we got here, no healthy sources available
        if last_error:
            raise last_error
        else:
            raise NoHealthySourceError("No healthy data sources available")

    async def get_historical_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        interval: str = "1d"
    ) -> List[PriceData]:
        """Get historical data with intelligent source selection

        Args:
            symbol: Stock symbol to fetch
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval

        Returns:
            List of PriceData objects from the best available source

        Raises:
            NoHealthySourceError: If no healthy sources are available
            SymbolNotFoundError: If symbol is not found in any source
        """
        sources = self._get_prioritized_sources()

        last_error = None
        symbol_not_found_count = 0

        for source_name in sources:
            if not self._is_source_available(source_name):
                continue

            try:
                source = self.sources[source_name]
                data = await source.get_historical_data(symbol, start_date, end_date, interval)

                # Validate if enabled
                if self.validation_enabled and data:
                    validation_result = self.validator.validate_price_history(data)
                    if not validation_result.is_valid:
                        logger.warning(f"Invalid historical data from {source_name}: {validation_result.issues}")
                        continue

                    # Update quality scores from validation
                    for item in data:
                        item.quality_score = validation_result.quality_score

                # Update source reliability on success
                self._update_source_reliability(source_name, True)
                self._reset_circuit_breaker(source_name)

                logger.debug(f"Successfully fetched {len(data)} historical records for {symbol} from {source_name}")
                return data

            except SymbolNotFoundError:
                symbol_not_found_count += 1
                logger.debug(f"Symbol {symbol} not found in {source_name}")
                continue

            except (RateLimitError, DataSourceError) as e:
                logger.warning(f"Failed to get historical data from {source_name}: {e}")
                self._update_source_reliability(source_name, False)
                self._update_circuit_breaker(source_name, e)
                last_error = e
                continue

        # If all sources returned symbol not found, raise that error
        if symbol_not_found_count == len(sources):
            raise SymbolNotFoundError(f"Symbol {symbol} not found in any data source")

        # If we got here, no healthy sources available
        if last_error:
            raise last_error
        else:
            raise NoHealthySourceError("No healthy data sources available")

    async def get_source_health_status(self) -> Dict[str, HealthStatus]:
        """Get health status for all registered sources

        Returns:
            Dictionary mapping source names to their health status
        """
        health_statuses = {}

        for source_name, source in self.sources.items():
            try:
                status = await source.get_health_status()
                health_statuses[source_name] = status
            except Exception as e:
                # Create a health status indicating failure
                health_statuses[source_name] = HealthStatus(
                    status=DataSourceStatus.UNHEALTHY,
                    error_count=self.circuit_breaker_states[source_name]["failure_count"],
                    message=f"Health check failed: {str(e)}"
                )

        return health_statuses

    async def validate_symbol(self, symbol: str) -> Dict[str, bool]:
        """Validate symbol across all available sources

        Args:
            symbol: Symbol to validate

        Returns:
            Dictionary mapping source names to validation results
        """
        results = {}

        for source_name, source in self.sources.items():
            if not self._is_source_available(source_name):
                results[source_name] = False
                continue

            try:
                is_valid = await source.validate_symbol(symbol)
                results[source_name] = is_valid
            except Exception as e:
                logger.warning(f"Symbol validation failed for {source_name}: {e}")
                results[source_name] = False

        return results

    def get_source_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all sources

        Returns:
            Dictionary with source statistics
        """
        stats = {}

        for source_name in self.sources:
            circuit_state = self.circuit_breaker_states[source_name]
            stats[source_name] = {
                "priority": self.source_priorities[source_name].name,
                "reliability": self.source_reliability[source_name],
                "circuit_breaker_state": circuit_state["state"],
                "failure_count": circuit_state["failure_count"],
                "last_failure": circuit_state["last_failure"],
                "is_available": self._is_source_available(source_name)
            }

        return stats

    def _get_prioritized_sources(self) -> List[str]:
        """Get sources ordered by priority and reliability

        Returns:
            List of source names in priority order
        """
        # Sort by priority first, then by reliability
        sources = list(self.sources.keys())
        sources.sort(key=lambda name: (
            self.source_priorities[name].value,
            -self.source_reliability[name]  # Higher reliability first
        ))

        return sources

    def _is_source_available(self, source_name: str) -> bool:
        """Check if a source is available (not in circuit breaker open state)

        Args:
            source_name: Name of the source to check

        Returns:
            True if source is available
        """
        if self.source_priorities[source_name] == SourcePriority.DISABLED:
            return False

        circuit_state = self.circuit_breaker_states[source_name]

        if circuit_state["state"] == "open":
            # Check if we should try half-open
            if circuit_state["next_attempt"] and datetime.now() >= circuit_state["next_attempt"]:
                circuit_state["state"] = "half-open"
                logger.info(f"Circuit breaker for {source_name} moved to half-open state")
                return True
            return False

        return True

    def _update_source_reliability(self, source_name: str, success: bool) -> None:
        """Update source reliability score

        Args:
            source_name: Name of the source
            success: Whether the operation was successful
        """
        current_reliability = self.source_reliability[source_name]

        if success:
            # Gradually increase reliability on success
            self.source_reliability[source_name] = min(1.0, current_reliability + 0.1)
        else:
            # Decrease reliability on failure
            self.source_reliability[source_name] = max(0.0, current_reliability - 0.2)

        logger.debug(f"Updated {source_name} reliability: {self.source_reliability[source_name]:.2f}")

    def _update_circuit_breaker(self, source_name: str, error: Exception) -> None:
        """Update circuit breaker state after an error

        Args:
            source_name: Name of the source
            error: The error that occurred
        """
        circuit_state = self.circuit_breaker_states[source_name]
        circuit_state["failure_count"] += 1
        circuit_state["last_failure"] = datetime.now()

        if circuit_state["failure_count"] >= self.max_failure_threshold:
            circuit_state["state"] = "open"
            circuit_state["next_attempt"] = datetime.now() + timedelta(seconds=self.circuit_breaker_timeout)
            logger.warning(f"Circuit breaker opened for {source_name} after {circuit_state['failure_count']} failures")

    def _reset_circuit_breaker(self, source_name: str) -> None:
        """Reset circuit breaker state after successful operation

        Args:
            source_name: Name of the source
        """
        circuit_state = self.circuit_breaker_states[source_name]

        if circuit_state["state"] in ["half-open", "open"]:
            logger.info(f"Circuit breaker reset for {source_name}")

        circuit_state["state"] = "closed"
        circuit_state["failure_count"] = 0
        circuit_state["last_failure"] = None
        circuit_state["next_attempt"] = None

    async def start_health_monitoring(self) -> None:
        """Start background health monitoring task"""
        if self.health_check_task and not self.health_check_task.done():
            return

        self.health_check_task = asyncio.create_task(self._health_monitor_loop())
        logger.info("Started health monitoring task")

    async def stop_health_monitoring(self) -> None:
        """Stop background health monitoring task"""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped health monitoring task")

    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")

    async def _perform_health_checks(self) -> None:
        """Perform health checks on all sources"""
        for source_name, source in self.sources.items():
            try:
                status = await source.get_health_status()

                # Update reliability based on health status
                if status.status == DataSourceStatus.HEALTHY:
                    self._update_source_reliability(source_name, True)
                elif status.status == DataSourceStatus.UNHEALTHY:
                    self._update_source_reliability(source_name, False)

                self.last_health_check[source_name] = datetime.now()

            except Exception as e:
                logger.warning(f"Health check failed for {source_name}: {e}")
                self._update_source_reliability(source_name, False)

    async def close(self) -> None:
        """Close all sources and cleanup"""
        await self.stop_health_monitoring()

        for source_name, source in self.sources.items():
            try:
                if hasattr(source, 'close'):
                    await source.close()
            except Exception as e:
                logger.error(f"Error closing source {source_name}: {e}")

        logger.info("Data source manager closed")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start_health_monitoring()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()