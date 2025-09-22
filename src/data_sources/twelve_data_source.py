"""Twelve Data Source Implementation

Twelve Data provides comprehensive financial market data with a generous free tier.
Free tier includes 800 API requests per day and supports real-time and historical data.
"""

import logging
import asyncio
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
import aiohttp

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


class TwelveDataSource(BaseDataSource):
    """Twelve Data source implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("twelve_data", config)

        # API configuration
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise AuthenticationError("Twelve Data API key is required")

        self.base_url = config.get("base_url", "https://api.twelvedata.com")

        # Rate limiting configuration
        self._daily_limit = config.get("daily_limit", 800)  # Free tier limit
        self._requests_per_minute = config.get("requests_per_minute", 8)  # Conservative estimate
        self._premium_tier = config.get("premium_tier", False)

        # Request tracking
        self._requests_today = 0
        self._daily_reset_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self._last_request_time = None
        self._request_timestamps = []

        # Connection management
        self._session = None
        self._timeout = config.get("timeout", 30)
        self._max_retries = config.get("max_retries", 3)

        logger.info(f"Twelve Data source initialized with daily limit: {self._daily_limit}")

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

        # Check for daily reset
        if now >= self._daily_reset_date + timedelta(days=1):
            self._daily_reset_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            self._requests_today = 0
            logger.info("Daily rate limit reset")

        # Check daily limit
        if self._requests_today >= self._daily_limit:
            raise RateLimitError(f"Daily limit of {self._daily_limit} requests exceeded")

        # Clean old request timestamps (keep last minute)
        cutoff = now - timedelta(minutes=1)
        self._request_timestamps = [ts for ts in self._request_timestamps if ts > cutoff]

        # Check per-minute rate limit
        if len(self._request_timestamps) >= self._requests_per_minute:
            sleep_time = 60.0 - (now - self._request_timestamps[0]).total_seconds()
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)

    def _record_request(self):
        """Record a successful request for rate limiting"""
        now = datetime.now()
        self._request_timestamps.append(now)
        self._requests_today += 1
        self._last_request_time = now

    async def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make an authenticated request to Twelve Data API"""
        await self._check_rate_limits()

        if params is None:
            params = {}

        # Add API key
        params["apikey"] = self.api_key

        url = f"{self.base_url}/{endpoint}"

        session = await self._get_session()

        for attempt in range(self._max_retries):
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Check for API error in response body
                        if isinstance(data, dict) and "status" in data and data["status"] == "error":
                            error_code = data.get("code", "unknown")
                            error_message = data.get("message", "Unknown error")

                            if error_code == 400:
                                raise SymbolNotFoundError(f"Symbol not found: {error_message}")
                            elif error_code == 429:
                                raise RateLimitError(f"Rate limit exceeded: {error_message}")
                            elif error_code in [401, 403]:
                                raise AuthenticationError(f"Authentication error: {error_message}")
                            else:
                                raise DataSourceError(f"API error ({error_code}): {error_message}")

                        self._record_request()
                        self.reset_error_count()
                        return data

                    elif response.status == 401:
                        raise AuthenticationError("Invalid API key")
                    elif response.status == 429:
                        # Rate limited
                        retry_after = int(response.headers.get("Retry-After", 60))
                        raise RateLimitError(f"Rate limited, retry after {retry_after} seconds")
                    elif response.status == 400:
                        error_text = await response.text()
                        raise SymbolNotFoundError(f"Bad request: {error_text}")
                    else:
                        error_text = await response.text()
                        raise DataSourceError(f"HTTP {response.status}: {error_text}")

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == self._max_retries - 1:
                    raise DataSourceError(f"Request failed after {self._max_retries} attempts: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    async def get_current_price(self, symbol: str) -> CurrentPrice:
        """Get current price for a symbol

        Uses Twelve Data's price endpoint for real-time pricing data.
        """
        try:
            # Get current price data
            params = {
                "symbol": symbol.upper(),
                "interval": "1min",
                "outputsize": 1
            }
            data = await self._make_request("price", params)

            if not data or "price" not in data:
                raise SymbolNotFoundError(f"No price data returned for symbol {symbol}")

            # Get additional data from quote endpoint for more complete information
            try:
                quote_data = await self._make_request("quote", {"symbol": symbol.upper()})
            except Exception:
                # If quote fails, use minimal data from price endpoint
                quote_data = {}

            current_price = CurrentPrice(
                symbol=symbol.upper(),
                price=float(data["price"]),
                timestamp=datetime.now(),  # Twelve Data doesn't always provide timestamp in price endpoint
                volume=quote_data.get("volume"),
                bid=quote_data.get("bid"),
                ask=quote_data.get("ask"),
                source=self.name,
                quality_score=self._calculate_quality_score(data, quote_data)
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

        Twelve Data supports various intervals and historical ranges.
        Free tier includes up to 5 years of historical data.
        """
        try:
            # Map interval format
            twelve_data_interval = self._map_interval(interval)

            params = {
                "symbol": symbol.upper(),
                "interval": twelve_data_interval,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "format": "JSON"
            }

            data = await self._make_request("time_series", params)

            if not data or "values" not in data:
                raise DataSourceError(f"No historical data returned for {symbol}")

            values = data["values"]
            if not values:
                raise DataSourceError(f"No valid data found for {symbol} in date range {start_date} to {end_date}")

            price_data = []
            for daily_data in values:
                try:
                    # Parse date - Twelve Data returns dates in various formats
                    date_str = daily_data.get("datetime", "")
                    if "T" in date_str:
                        trade_date = datetime.fromisoformat(date_str.replace("Z", "+00:00")).date()
                    else:
                        trade_date = datetime.strptime(date_str, "%Y-%m-%d").date()

                    # Filter by date range (API sometimes returns extra data)
                    if not (start_date <= trade_date <= end_date):
                        continue

                    # Skip entries with null prices
                    if not daily_data.get("close"):
                        continue

                    price_data.append(PriceData(
                        symbol=symbol.upper(),
                        timestamp=datetime.combine(trade_date, datetime.min.time()),
                        open_price=float(daily_data.get("open", daily_data.get("close", 0))),
                        high_price=float(daily_data.get("high", daily_data.get("close", 0))),
                        low_price=float(daily_data.get("low", daily_data.get("close", 0))),
                        close_price=float(daily_data["close"]),
                        volume=int(daily_data.get("volume", 0)) if daily_data.get("volume") else None,
                        source=self.name,
                        quality_score=90  # Twelve Data provides good quality historical data
                    ))

                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping invalid data point for {symbol}: {e}")
                    continue

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
        """Check the health of Twelve Data API"""
        try:
            start_time = datetime.now()

            # Use a simple endpoint to check health
            test_data = await self._make_request("price", {"symbol": "AAPL"})

            response_time = (datetime.now() - start_time).total_seconds() * 1000

            if test_data and "price" in test_data:
                status = DataSourceStatus.HEALTHY
                message = "Twelve Data API is healthy"
            else:
                status = DataSourceStatus.DEGRADED
                message = "Twelve Data API responding but data quality issues"

            return HealthStatus(
                status=status,
                last_successful_call=datetime.now(),
                error_count=self.error_count,
                response_time_ms=response_time,
                rate_limit_remaining=max(0, self._daily_limit - self._requests_today),
                message=message
            )

        except Exception as e:
            logger.warning(f"Twelve Data health check failed: {e}")
            return HealthStatus(
                status=DataSourceStatus.DEGRADED if self.error_count < 5 else DataSourceStatus.UNHEALTHY,
                last_successful_call=self._last_request_time,
                error_count=self.error_count,
                response_time_ms=None,
                rate_limit_remaining=max(0, self._daily_limit - self._requests_today),
                message=f"Health check failed: {e}"
            )

    async def validate_symbol(self, symbol: str) -> bool:
        """Validate if a symbol is supported by Twelve Data"""
        try:
            await self._make_request("price", {"symbol": symbol.upper()})
            return True
        except SymbolNotFoundError:
            return False
        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {e}")
            return False

    async def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols from Twelve Data"""
        try:
            # Twelve Data provides a symbols endpoint for supported stocks
            data = await self._make_request("stocks", {
                "country": "United States",
                "format": "JSON"
            })

            symbols = []
            if isinstance(data, dict) and "data" in data:
                for item in data["data"]:
                    if item.get("symbol") and item.get("access", {}).get("global", False):
                        symbols.append(item["symbol"])

            return sorted(symbols)

        except Exception as e:
            logger.error(f"Error getting supported symbols: {e}")
            # Return a basic list of common symbols if the endpoint fails
            return ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "SPY", "QQQ", "IWM"]

    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get current rate limit information"""
        return {
            "requests_today": self._requests_today,
            "daily_limit": self._daily_limit,
            "daily_remaining": max(0, self._daily_limit - self._requests_today),
            "requests_last_minute": len(self._request_timestamps),
            "per_minute_limit": self._requests_per_minute,
            "last_request_time": self._last_request_time.isoformat() if self._last_request_time else None,
            "daily_reset_date": self._daily_reset_date.isoformat(),
            "premium_tier": self._premium_tier
        }

    def _map_interval(self, interval: str) -> str:
        """Map standard interval to Twelve Data format"""
        interval_mapping = {
            "1min": "1min",
            "5min": "5min",
            "15min": "15min",
            "30min": "30min",
            "1h": "1h",
            "1d": "1day",
            "1w": "1week",
            "1m": "1month"
        }

        mapped = interval_mapping.get(interval, "1day")
        if mapped != interval:
            logger.info(f"Mapped interval {interval} to {mapped} for Twelve Data")

        return mapped

    def _calculate_quality_score(self, price_data: Dict[str, Any], quote_data: Dict[str, Any] = None) -> int:
        """Calculate quality score for Twelve Data"""
        score = 80  # Base score for Twelve Data

        # Adjust based on data completeness
        if price_data.get("price"):
            score += 10

        if quote_data:
            if quote_data.get("volume"):
                score += 3
            if quote_data.get("bid") and quote_data.get("ask"):
                score += 4
            if quote_data.get("change"):
                score += 2
            if quote_data.get("percent_change"):
                score += 1

        return min(100, max(0, score))

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()