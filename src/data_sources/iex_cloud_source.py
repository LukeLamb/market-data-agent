"""IEX Cloud Data Source Implementation

IEX Cloud provides real-time and historical market data with a generous free tier
and reliable API. This implementation includes proper rate limiting and error handling.

Free tier limits: 500,000 core data messages per month
Premium endpoints: charged per successful call
"""

import logging
import asyncio
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
import aiohttp
from decimal import Decimal

from .base import (
    BaseDataSource,
    CurrentPrice,
    PriceData,
    HealthStatus,
    DataSourceStatus,
    SymbolNotFoundError,
    RateLimitError,
    AuthenticationError,
    DataSourceError
)

logger = logging.getLogger(__name__)


class IEXCloudSource(BaseDataSource):
    """IEX Cloud data source implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("iex_cloud", config)

        # API configuration
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise AuthenticationError("IEX Cloud API key is required")

        self.base_url = config.get("base_url", "https://cloud.iexapis.com/stable")
        self.sandbox_url = "https://sandbox.iexapis.com/stable"
        self.use_sandbox = config.get("use_sandbox", False)
        self.version = config.get("version", "stable")

        # Rate limiting configuration
        self._rate_limit_per_second = config.get("rate_limit_per_second", 100)
        self._monthly_message_limit = config.get("monthly_message_limit", 500000)
        self._daily_limit = config.get("daily_limit", 25000)  # Conservative daily estimate

        # Request tracking
        self._requests_this_minute = 0
        self._requests_this_month = 0
        self._monthly_reset_date = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        self._last_request_time = None
        self._request_timestamps = []

        # Connection management
        self._session = None
        self._timeout = config.get("timeout", 30)
        self._max_retries = config.get("max_retries", 3)

        logger.info(f"IEX Cloud source initialized with sandbox={self.use_sandbox}")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        """Close the data source and cleanup resources"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _check_rate_limits(self):
        """Check and enforce rate limits"""
        now = datetime.now()

        # Clean old request timestamps (keep last second)
        cutoff = now - timedelta(seconds=1)
        self._request_timestamps = [ts for ts in self._request_timestamps if ts > cutoff]

        # Check per-second rate limit
        if len(self._request_timestamps) >= self._rate_limit_per_second:
            sleep_time = 1.0 - (now - self._request_timestamps[0]).total_seconds()
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)

        # Check monthly reset
        if now >= self._monthly_reset_date + timedelta(days=32):  # Next month
            self._monthly_reset_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            self._requests_this_month = 0
            logger.info("Monthly rate limit reset")

        # Check monthly limit
        if self._requests_this_month >= self._monthly_message_limit:
            raise RateLimitError(f"Monthly message limit of {self._monthly_message_limit} exceeded")

    def _record_request(self):
        """Record a successful request for rate limiting"""
        now = datetime.now()
        self._request_timestamps.append(now)
        self._requests_this_month += 1
        self._last_request_time = now

    def _get_base_url(self) -> str:
        """Get the appropriate base URL"""
        return self.sandbox_url if self.use_sandbox else self.base_url

    async def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make an authenticated request to IEX Cloud API"""
        await self._check_rate_limits()

        if params is None:
            params = {}

        # Add API token
        params["token"] = self.api_key

        url = f"{self._get_base_url()}/{endpoint}"

        session = await self._get_session()

        for attempt in range(self._max_retries):
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        self._record_request()
                        self.reset_error_count()
                        return data
                    elif response.status == 401:
                        raise AuthenticationError("Invalid API key")
                    elif response.status == 402:
                        raise RateLimitError("Payment required - credit limit exceeded")
                    elif response.status == 403:
                        raise AuthenticationError("Forbidden - check API key permissions")
                    elif response.status == 429:
                        # Rate limited
                        retry_after = int(response.headers.get("Retry-After", 60))
                        raise RateLimitError(f"Rate limited, retry after {retry_after} seconds")
                    elif response.status == 404:
                        raise SymbolNotFoundError(f"Symbol not found or endpoint not available")
                    else:
                        error_text = await response.text()
                        raise DataSourceError(f"HTTP {response.status}: {error_text}")

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == self._max_retries - 1:
                    raise DataSourceError(f"Request failed after {self._max_retries} attempts: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    async def get_current_price(self, symbol: str) -> CurrentPrice:
        """Get current price for a symbol

        Uses IEX Cloud's quote endpoint which provides real-time pricing data.
        """
        try:
            # Get quote data
            data = await self._make_request(f"stock/{symbol.upper()}/quote")

            if not data:
                raise SymbolNotFoundError(f"No data returned for symbol {symbol}")

            # Parse IEX Cloud quote response
            current_price = CurrentPrice(
                symbol=symbol.upper(),
                price=float(data.get("latestPrice", 0)),
                timestamp=datetime.fromtimestamp(data.get("latestUpdate", 0) / 1000) if data.get("latestUpdate") else datetime.now(),
                volume=data.get("latestVolume"),
                bid=data.get("iexBidPrice"),
                ask=data.get("iexAskPrice"),
                source=self.name,
                quality_score=self._calculate_quality_score(data)
            )

            return current_price

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

        IEX Cloud supports various intervals and historical ranges.
        Free tier includes up to 5 years of historical data.
        """
        try:
            # Determine the appropriate endpoint based on date range
            days_diff = (end_date - start_date).days

            if days_diff <= 5:
                # Use chart/date for recent data
                endpoint = f"stock/{symbol.upper()}/chart/5d"
            elif days_diff <= 30:
                endpoint = f"stock/{symbol.upper()}/chart/1m"
            elif days_diff <= 90:
                endpoint = f"stock/{symbol.upper()}/chart/3m"
            elif days_diff <= 180:
                endpoint = f"stock/{symbol.upper()}/chart/6m"
            elif days_diff <= 365:
                endpoint = f"stock/{symbol.upper()}/chart/1y"
            else:
                # For longer periods, use the max endpoint
                endpoint = f"stock/{symbol.upper()}/chart/max"

            # Add parameters for date filtering if needed
            params = {}
            if interval != "1d":
                logger.warning(f"IEX Cloud typically provides daily data, requested interval: {interval}")

            data = await self._make_request(endpoint, params)

            if not data:
                raise DataSourceError(f"No historical data returned for {symbol}")

            price_data = []
            for daily_data in data:
                trade_date = datetime.strptime(daily_data["date"], "%Y-%m-%d").date()

                # Filter by date range
                if start_date <= trade_date <= end_date:
                    # Skip entries with null prices
                    if daily_data.get("close") is None:
                        continue

                    price_data.append(PriceData(
                        symbol=symbol.upper(),
                        timestamp=datetime.combine(trade_date, datetime.min.time()),
                        open_price=float(daily_data.get("open", daily_data.get("close", 0))),
                        high_price=float(daily_data.get("high", daily_data.get("close", 0))),
                        low_price=float(daily_data.get("low", daily_data.get("close", 0))),
                        close_price=float(daily_data["close"]),
                        volume=daily_data.get("volume", 0),
                        source=self.name,
                        quality_score=95  # IEX Cloud provides high-quality historical data
                    ))

            if not price_data:
                raise DataSourceError(f"No valid data found for {symbol} in date range {start_date} to {end_date}")

            # Sort by timestamp
            price_data.sort(key=lambda x: x.timestamp)
            return price_data

        except (SymbolNotFoundError, RateLimitError, AuthenticationError):
            raise
        except Exception as e:
            self.increment_error_count()
            logger.error(f"Error getting historical data for {symbol}: {e}")
            raise DataSourceError(f"Failed to get historical data for {symbol}: {e}")

    async def get_health_status(self) -> HealthStatus:
        """Check the health of IEX Cloud API"""
        try:
            start_time = datetime.now()

            # Use a simple endpoint to check health
            await self._make_request("ref-data/symbols", {"format": "json"})

            response_time = (datetime.now() - start_time).total_seconds() * 1000

            return HealthStatus(
                status=DataSourceStatus.HEALTHY,
                last_successful_call=datetime.now(),
                error_count=self.error_count,
                response_time_ms=response_time,
                rate_limit_remaining=max(0, self._monthly_message_limit - self._requests_this_month),
                message="IEX Cloud API is healthy"
            )

        except Exception as e:
            logger.warning(f"IEX Cloud health check failed: {e}")
            return HealthStatus(
                status=DataSourceStatus.DEGRADED if self.error_count < 5 else DataSourceStatus.UNHEALTHY,
                last_successful_call=self._last_request_time,
                error_count=self.error_count,
                response_time_ms=None,
                rate_limit_remaining=max(0, self._monthly_message_limit - self._requests_this_month),
                message=f"Health check failed: {e}"
            )

    async def validate_symbol(self, symbol: str) -> bool:
        """Validate if a symbol is supported by IEX Cloud"""
        try:
            await self._make_request(f"stock/{symbol.upper()}/quote")
            return True
        except SymbolNotFoundError:
            return False
        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {e}")
            return False

    async def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols from IEX Cloud"""
        try:
            # IEX provides a symbols reference endpoint
            data = await self._make_request("ref-data/symbols", {"format": "json"})

            symbols = []
            for item in data:
                if item.get("isEnabled", False) and item.get("type") == "cs":  # Common stock
                    symbols.append(item["symbol"])

            return sorted(symbols)

        except Exception as e:
            logger.error(f"Error getting supported symbols: {e}")
            return []

    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get current rate limit information"""
        return {
            "requests_this_month": self._requests_this_month,
            "monthly_limit": self._monthly_message_limit,
            "monthly_remaining": max(0, self._monthly_message_limit - self._requests_this_month),
            "requests_last_second": len(self._request_timestamps),
            "per_second_limit": self._rate_limit_per_second,
            "last_request_time": self._last_request_time.isoformat() if self._last_request_time else None,
            "monthly_reset_date": self._monthly_reset_date.isoformat()
        }

    def _calculate_quality_score(self, quote_data: Dict[str, Any]) -> int:
        """Calculate quality score for IEX Cloud data"""
        score = 85  # Base score for IEX Cloud

        # Adjust based on data completeness
        if quote_data.get("latestPrice"):
            score += 5
        if quote_data.get("latestVolume"):
            score += 3
        if quote_data.get("iexBidPrice") and quote_data.get("iexAskPrice"):
            score += 4
        if quote_data.get("latestUpdate"):
            # Check data freshness
            update_time = datetime.fromtimestamp(quote_data["latestUpdate"] / 1000)
            age_minutes = (datetime.now() - update_time).total_seconds() / 60
            if age_minutes < 1:
                score += 3
            elif age_minutes < 5:
                score += 1

        return min(100, max(0, score))

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()