"""Alpha Vantage Data Source Implementation

This module implements the BaseDataSource interface using Alpha Vantage API.
Provides real-time and historical stock market data with API key authentication.
"""

import asyncio
import time
import json
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
import logging

import aiohttp
import pandas as pd

from .base import (
    BaseDataSource,
    PriceData,
    CurrentPrice,
    HealthStatus,
    DataSourceStatus,
    DataSourceError,
    SymbolNotFoundError,
    RateLimitError,
    AuthenticationError
)

logger = logging.getLogger(__name__)


class AlphaVantageSource(BaseDataSource):
    """Alpha Vantage data source implementation

    Features:
    - Real-time stock quotes
    - Historical daily data
    - API key authentication
    - Rate limiting (5 calls/minute, 500/day for free tier)
    - Comprehensive error handling
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, config: Dict[str, Any]):
        """Initialize Alpha Vantage data source

        Args:
            config: Configuration dictionary with API key and rate limits
        """
        super().__init__("alpha_vantage", config)

        self.api_key = config.get("api_key")
        if not self.api_key:
            raise AuthenticationError("Alpha Vantage API key is required")

        self._rate_limit_per_minute = config.get("rate_limit_per_minute", 5)
        self._daily_limit = config.get("daily_limit", 500)
        self._request_times = []
        self._daily_request_count = 0
        self._daily_reset_time = None
        self._session = None

        logger.info(f"Initialized Alpha Vantage source with limits: {self._rate_limit_per_minute}/min, {self._daily_limit}/day")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def get_current_price(self, symbol: str) -> CurrentPrice:
        """Get current price for a symbol

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            CurrentPrice object with latest price data

        Raises:
            SymbolNotFoundError: If symbol is not found
            RateLimitError: If rate limit is exceeded
            AuthenticationError: If API key is invalid
            DataSourceError: For other API errors
        """
        await self._check_rate_limits()

        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": self.api_key
        }

        try:
            session = await self._get_session()
            async with session.get(self.BASE_URL, params=params) as response:
                if response.status != 200:
                    raise DataSourceError(f"HTTP {response.status}: {await response.text()}")

                data = await response.json()

                # Check for API errors
                if "Error Message" in data:
                    raise SymbolNotFoundError(f"Symbol {symbol} not found: {data['Error Message']}")

                if "Note" in data:
                    raise RateLimitError(f"Alpha Vantage rate limit: {data['Note']}")

                if "Information" in data and "API call frequency" in data["Information"]:
                    raise RateLimitError(f"Rate limit exceeded: {data['Information']}")

                # Parse the response
                quote_data = data.get("Global Quote", {})
                if not quote_data:
                    raise DataSourceError(f"No quote data returned for {symbol}")

                price = float(quote_data.get("05. price", 0))
                if price <= 0:
                    raise SymbolNotFoundError(f"Invalid price data for symbol {symbol}")

                # Extract additional data
                volume = quote_data.get("06. volume")
                change_percent = quote_data.get("10. change percent", "").replace("%", "")

                self._record_request()
                self.reset_error_count()

                return CurrentPrice(
                    symbol=symbol.upper(),
                    price=price,
                    timestamp=datetime.now(),
                    volume=int(volume) if volume and volume.isdigit() else None,
                    source=self.name,
                    quality_score=95  # Alpha Vantage provides real-time data
                )

        except (SymbolNotFoundError, RateLimitError, AuthenticationError):
            raise
        except Exception as e:
            self.increment_error_count()
            logger.error(f"Error getting current price for {symbol}: {e}")
            raise DataSourceError(f"Failed to get current price for {symbol}: {e}")

    async def get_historical_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        interval: str = "1d"
    ) -> List[PriceData]:
        """Get historical OHLCV data for a symbol

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval (only '1d' supported by free tier)

        Returns:
            List of PriceData objects sorted by timestamp

        Raises:
            SymbolNotFoundError: If symbol is not found
            RateLimitError: If rate limit is exceeded
            DataSourceError: For other errors
        """
        await self._check_rate_limits()

        # Alpha Vantage free tier only supports daily data
        if interval != "1d":
            logger.warning(f"Alpha Vantage free tier only supports daily data, using 1d instead of {interval}")

        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full",  # Get full historical data
            "apikey": self.api_key
        }

        try:
            session = await self._get_session()
            async with session.get(self.BASE_URL, params=params) as response:
                if response.status != 200:
                    raise DataSourceError(f"HTTP {response.status}: {await response.text()}")

                data = await response.json()

                # Check for API errors
                if "Error Message" in data:
                    raise SymbolNotFoundError(f"Symbol {symbol} not found: {data['Error Message']}")

                if "Note" in data:
                    raise RateLimitError(f"Alpha Vantage rate limit: {data['Note']}")

                # Parse time series data
                time_series = data.get("Time Series (Daily)", {})
                if not time_series:
                    raise DataSourceError(f"No time series data returned for {symbol}")

                price_data = []
                for date_str, daily_data in time_series.items():
                    trade_date = datetime.strptime(date_str, "%Y-%m-%d").date()

                    # Filter by date range
                    if start_date <= trade_date <= end_date:
                        price_data.append(PriceData(
                            symbol=symbol.upper(),
                            timestamp=datetime.combine(trade_date, datetime.min.time()),
                            open_price=float(daily_data["1. open"]),
                            high_price=float(daily_data["2. high"]),
                            low_price=float(daily_data["3. low"]),
                            close_price=float(daily_data["4. close"]),
                            volume=int(daily_data["6. volume"]),
                            adjusted_close=float(daily_data["5. adjusted close"]),
                            source=self.name,
                            quality_score=95  # High quality historical data
                        ))

                if not price_data:
                    raise DataSourceError(f"No data found for {symbol} in date range {start_date} to {end_date}")

                self._record_request()
                self.reset_error_count()

                # Sort by timestamp
                price_data.sort(key=lambda x: x.timestamp)
                return price_data

        except (SymbolNotFoundError, RateLimitError):
            raise
        except Exception as e:
            self.increment_error_count()
            logger.error(f"Error getting historical data for {symbol}: {e}")
            raise DataSourceError(f"Failed to get historical data for {symbol}: {e}")

    async def get_health_status(self) -> HealthStatus:
        """Check health status by testing API connectivity

        Returns:
            HealthStatus object with current status
        """
        start_time = time.time()

        try:
            # Test with a simple API call
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": "AAPL",  # Use a reliable symbol
                "apikey": self.api_key
            }

            session = await self._get_session()
            async with session.get(self.BASE_URL, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    # Check if we got valid data (not rate limited)
                    if "Global Quote" in data and data["Global Quote"]:
                        response_time = (time.time() - start_time) * 1000

                        status = HealthStatus(
                            status=DataSourceStatus.HEALTHY,
                            last_successful_call=self.last_successful_call,
                            error_count=self.error_count,
                            response_time_ms=response_time,
                            rate_limit_remaining=self._get_remaining_requests(),
                            message="Alpha Vantage API is responding normally"
                        )

                        self.update_health_status(status)
                        return status
                    elif "Note" in data or "Information" in data:
                        # Rate limited but API is working
                        status = HealthStatus(
                            status=DataSourceStatus.DEGRADED,
                            last_successful_call=self.last_successful_call,
                            error_count=self.error_count,
                            response_time_ms=(time.time() - start_time) * 1000,
                            rate_limit_remaining=0,
                            message="Alpha Vantage API rate limited"
                        )

                        self.update_health_status(status)
                        return status

                raise Exception(f"HTTP {response.status}")

        except Exception as e:
            logger.warning(f"Alpha Vantage health check failed: {e}")

            # Determine status based on error count and type
            if "authentication" in str(e).lower() or "api key" in str(e).lower():
                status_level = DataSourceStatus.UNHEALTHY
                message = "Authentication failed - check API key"
            elif self.error_count > 5:
                status_level = DataSourceStatus.UNHEALTHY
                message = f"Multiple failures: {str(e)[:100]}"
            elif self.error_count > 2:
                status_level = DataSourceStatus.DEGRADED
                message = f"Some failures: {str(e)[:100]}"
            else:
                status_level = DataSourceStatus.DEGRADED
                message = f"Health check failed: {str(e)[:100]}"

            status = HealthStatus(
                status=status_level,
                last_successful_call=self.last_successful_call,
                error_count=self.error_count,
                response_time_ms=(time.time() - start_time) * 1000,
                message=message
            )

            self.update_health_status(status)
            return status

    def get_supported_symbols(self) -> List[str]:
        """Get list of commonly supported symbols

        Returns:
            List of popular US stock symbols supported by Alpha Vantage

        Note:
            Alpha Vantage supports most US stocks, ETFs, and many international symbols.
            This returns a subset of commonly traded symbols.
        """
        return [
            # Major US stocks
            "AAPL", "GOOGL", "GOOG", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX",
            "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "INTC",
            "CMCSA", "VZ", "ADBE", "CRM", "NKE", "PFE", "T", "ABT", "CVX", "KO",
            "WMT", "ORCL", "BAC", "XOM", "LLY", "BRK.B", "COST", "AVGO", "TMO",

            # Popular ETFs
            "SPY", "QQQ", "IWM", "VTI", "VOO", "VEA", "VWO", "AGG", "BND", "GLD",
            "SLV", "TLT", "HYG", "LQD", "EFA", "EEM", "VNQ", "XLF", "XLE", "XLK",

            # Other popular symbols
            "BTC-USD", "ETH-USD"  # Cryptocurrencies (if supported)
        ]

    async def validate_symbol(self, symbol: str) -> bool:
        """Check if a symbol is valid by making a test API call

        Args:
            symbol: Symbol to validate

        Returns:
            True if symbol is valid and supported
        """
        try:
            # Use a lightweight call to test symbol validity
            await self.get_current_price(symbol)
            return True
        except SymbolNotFoundError:
            return False
        except (RateLimitError, AuthenticationError, DataSourceError):
            # If we hit rate limits or other errors, we can't determine validity
            # Return True to be conservative (don't rule out valid symbols)
            return True

    async def _check_rate_limits(self) -> None:
        """Check and enforce rate limiting

        Raises:
            RateLimitError: If rate limit would be exceeded
        """
        await self._check_minute_rate_limit()
        await self._check_daily_rate_limit()

    async def _check_minute_rate_limit(self) -> None:
        """Check per-minute rate limit"""
        now = time.time()
        cutoff_time = now - 60  # 1 minute ago

        # Remove requests older than 1 minute
        self._request_times = [t for t in self._request_times if t > cutoff_time]

        if len(self._request_times) >= self._rate_limit_per_minute:
            oldest_request = min(self._request_times)
            wait_time = 60 - (now - oldest_request)

            if wait_time > 0:
                raise RateLimitError(
                    f"Minute rate limit exceeded. Wait {wait_time:.1f} seconds"
                )

    async def _check_daily_rate_limit(self) -> None:
        """Check daily rate limit"""
        now = datetime.now()

        # Reset daily counter if it's a new day
        if (self._daily_reset_time is None or
            now.date() > self._daily_reset_time.date()):
            self._daily_request_count = 0
            self._daily_reset_time = now

        if self._daily_request_count >= self._daily_limit:
            # Calculate time until next reset (midnight)
            tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            wait_hours = (tomorrow - now).total_seconds() / 3600

            raise RateLimitError(
                f"Daily rate limit exceeded. Resets in {wait_hours:.1f} hours"
            )

    def _record_request(self) -> None:
        """Record a successful request for rate limiting"""
        self._request_times.append(time.time())
        self._daily_request_count += 1

    def _get_remaining_requests(self) -> int:
        """Get remaining requests for the day"""
        return max(0, self._daily_limit - self._daily_request_count)

    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get current rate limiting information

        Returns:
            Dictionary with rate limit details
        """
        now = time.time()
        cutoff_time = now - 60
        current_minute_requests = len([t for t in self._request_times if t > cutoff_time])

        return {
            "requests_per_minute": self._rate_limit_per_minute,
            "requests_per_day": self._daily_limit,
            "current_minute_usage": current_minute_requests,
            "current_daily_usage": self._daily_request_count,
            "remaining_minute_requests": max(0, self._rate_limit_per_minute - current_minute_requests),
            "remaining_daily_requests": self._get_remaining_requests(),
            "daily_reset_time": self._daily_reset_time.isoformat() if self._daily_reset_time else None
        }

    async def close(self) -> None:
        """Close the HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()