"""Base Data Source Interface

This module defines the abstract base class that all data sources must implement.
It ensures consistency across different financial data providers.
"""

from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import pandas as pd
from pydantic import BaseModel


class DataSourceStatus(Enum):
    """Data source health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class PriceData(BaseModel):
    """Standard price data model"""
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    adjusted_close: Optional[float] = None
    source: str
    quality_score: int = 100  # 0-100 score


class CurrentPrice(BaseModel):
    """Current/real-time price model"""
    symbol: str
    price: float
    timestamp: datetime
    volume: Optional[int] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    source: str
    quality_score: int = 100


class HealthStatus(BaseModel):
    """Health status response"""
    status: DataSourceStatus
    last_successful_call: Optional[datetime] = None
    error_count: int = 0
    response_time_ms: Optional[float] = None
    rate_limit_remaining: Optional[int] = None
    message: Optional[str] = None


class DataSourceError(Exception):
    """Base exception for data source errors"""
    pass


class RateLimitError(DataSourceError):
    """Raised when rate limit is exceeded"""
    pass


class AuthenticationError(DataSourceError):
    """Raised when authentication fails"""
    pass


class SymbolNotFoundError(DataSourceError):
    """Raised when symbol is not found"""
    pass


class BaseDataSource(ABC):
    """Abstract base class for all financial data sources

    This class defines the interface that all data sources must implement.
    It ensures consistency and enables easy swapping between different providers.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize the data source

        Args:
            name: Unique identifier for this data source
            config: Configuration dictionary with API keys, rate limits, etc.
        """
        self.name = name
        self.config = config
        self._health_status = HealthStatus(status=DataSourceStatus.UNKNOWN)
        self._error_count = 0
        self._last_successful_call = None

    @abstractmethod
    async def get_current_price(self, symbol: str) -> CurrentPrice:
        """Get current/real-time price for a symbol

        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')

        Returns:
            CurrentPrice object with latest price data

        Raises:
            SymbolNotFoundError: If symbol is not found
            RateLimitError: If rate limit is exceeded
            DataSourceError: For other API errors
        """
        pass

    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        interval: str = "1d"
    ) -> List[PriceData]:
        """Get historical OHLCV data for a symbol

        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            interval: Data interval ('1d', '1h', '5m', etc.)

        Returns:
            List of PriceData objects sorted by timestamp

        Raises:
            SymbolNotFoundError: If symbol is not found
            RateLimitError: If rate limit is exceeded
            DataSourceError: For other API errors
        """
        pass

    @abstractmethod
    async def get_health_status(self) -> HealthStatus:
        """Check the health status of this data source

        Returns:
            HealthStatus object with current status information
        """
        pass

    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """Get list of symbols supported by this data source

        Returns:
            List of supported symbol strings

        Note:
            This may return a subset for performance reasons.
            Use validate_symbol() to check specific symbols.
        """
        pass

    @abstractmethod
    async def validate_symbol(self, symbol: str) -> bool:
        """Check if a symbol is valid and tradeable

        Args:
            symbol: Symbol to validate

        Returns:
            True if symbol is valid and supported
        """
        pass

    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get rate limiting information

        Returns:
            Dictionary with rate limit details
        """
        return {
            "requests_per_minute": self.config.get("rate_limit", 60),
            "requests_per_day": self.config.get("daily_limit"),
            "current_usage": getattr(self, "_current_usage", 0)
        }

    def update_health_status(self, status: HealthStatus) -> None:
        """Update the health status of this data source

        Args:
            status: New health status
        """
        self._health_status = status
        if status.status == DataSourceStatus.HEALTHY:
            self._last_successful_call = datetime.now()

    def increment_error_count(self) -> None:
        """Increment error counter"""
        self._error_count += 1

    def reset_error_count(self) -> None:
        """Reset error counter to zero"""
        self._error_count = 0

    @property
    def error_count(self) -> int:
        """Get current error count"""
        return self._error_count

    @property
    def last_successful_call(self) -> Optional[datetime]:
        """Get timestamp of last successful call"""
        return self._last_successful_call

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', status='{self._health_status.status.value}')"