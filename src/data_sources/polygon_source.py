"""Polygon.io Data Source Implementation

Polygon.io provides comprehensive financial data with real-time and historical market data.
Free tier includes 5 API calls per minute, paid tiers offer higher limits and more features.
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


class PolygonSource(BaseDataSource):
    """Polygon.io data source implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("polygon", config)

        # API configuration
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise AuthenticationError("Polygon.io API key is required")

        self.base_url = config.get("base_url", "https://api.polygon.io")

        # Rate limiting configuration
        self._calls_per_minute = config.get("calls_per_minute", 5)  # Free tier limit
        self._premium_tier = config.get("premium_tier", False)

        if self._premium_tier:
            self._calls_per_minute = config.get("premium_calls_per_minute", 1000)

        # Request tracking
        self._last_request_time = None
        self._request_timestamps = []

        # Connection management
        self._session = None
        self._timeout = config.get("timeout", 30)
        self._max_retries = config.get("max_retries", 3)

        logger.info(f"Polygon source initialized with {self._calls_per_minute} calls/minute limit")

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

        # Clean old request timestamps (keep last minute)
        cutoff = now - timedelta(minutes=1)
        self._request_timestamps = [ts for ts in self._request_timestamps if ts > cutoff]

        # Check per-minute rate limit
        if len(self._request_timestamps) >= self._calls_per_minute:
            sleep_time = 60.0 - (now - self._request_timestamps[0]).total_seconds()
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)

    def _record_request(self):
        """Record a successful request for rate limiting"""
        now = datetime.now()
        self._request_timestamps.append(now)
        self._last_request_time = now

    async def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make an authenticated request to Polygon.io API"""
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
                        if isinstance(data, dict):
                            status = data.get("status", "OK")
                            if status == "ERROR":
                                error_message = data.get("error", "Unknown error")
                                if "Invalid API Key" in error_message:
                                    raise AuthenticationError(f"Authentication error: {error_message}")
                                elif "limit" in error_message.lower():
                                    raise RateLimitError(f"Rate limit exceeded: {error_message}")
                                else:
                                    raise DataSourceError(f"API error: {error_message}")

                        self._record_request()
                        self.reset_error_count()
                        return data

                    elif response.status == 401:
                        raise AuthenticationError("Invalid API key")
                    elif response.status == 403:
                        raise AuthenticationError("Forbidden - check API key permissions")
                    elif response.status == 429:
                        # Rate limited
                        retry_after = int(response.headers.get("Retry-After", 60))
                        raise RateLimitError(f"Rate limited, retry after {retry_after} seconds")
                    elif response.status == 404:
                        # Some endpoints return 404 for invalid symbols
                        raise SymbolNotFoundError("Symbol not found")
                    else:
                        error_text = await response.text()
                        raise DataSourceError(f"HTTP {response.status}: {error_text}")

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == self._max_retries - 1:
                    raise DataSourceError(f"Request failed after {self._max_retries} attempts: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    async def get_current_price(self, symbol: str) -> CurrentPrice:
        """Get current price for a symbol

        Uses Polygon.io's last trade endpoint for current pricing data.
        """
        try:
            # Get last trade data
            endpoint = f"v2/last/trade/{symbol.upper()}"
            data = await self._make_request(endpoint)

            if not data or "results" not in data or not data["results"]:
                raise SymbolNotFoundError(f"No price data returned for symbol {symbol}")

            results = data["results"]

            # Get last quote for bid/ask if available
            quote_data = {}
            try:
                quote_endpoint = f"v2/last/nbbo/{symbol.upper()}"
                quote_response = await self._make_request(quote_endpoint)
                if quote_response and "results" in quote_response:
                    quote_data = quote_response["results"]
            except Exception:
                # Quote data is optional, continue without it
                pass

            # Polygon returns: p=price, s=size, t=timestamp
            current_price = CurrentPrice(
                symbol=symbol.upper(),
                price=float(results["p"]),  # Price
                timestamp=datetime.fromtimestamp(results.get("t", datetime.now().timestamp()) / 1000),
                volume=results.get("s"),  # Size/volume of last trade
                bid=quote_data.get("P") if quote_data else None,  # Bid price
                ask=quote_data.get("p") if quote_data else None,  # Ask price
                source=self.name,
                quality_score=self._calculate_quality_score(results, quote_data)
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

        Polygon.io supports daily and intraday data with various time spans.
        """
        try:
            # Map interval to Polygon format
            timespan, multiplier = self._map_interval(interval)

            # Format dates
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            params = {
                "adjusted": "true",
                "sort": "asc",
                "limit": 50000
            }

            endpoint = f"v2/aggs/ticker/{symbol.upper()}/range/{multiplier}/{timespan}/{start_str}/{end_str}"
            data = await self._make_request(endpoint, params)

            if not data or "results" not in data or not data["results"]:
                raise DataSourceError(f"No historical data returned for {symbol}")

            results = data["results"]

            price_data = []
            for candle in results:
                try:
                    # Polygon returns timestamps in milliseconds
                    trade_date = datetime.fromtimestamp(candle["t"] / 1000).date()

                    # Filter by date range
                    if not (start_date <= trade_date <= end_date):
                        continue

                    # Polygon returns: o=open, h=high, l=low, c=close, v=volume, t=timestamp
                    price_data.append(PriceData(
                        symbol=symbol.upper(),
                        timestamp=datetime.combine(trade_date, datetime.min.time()),
                        open_price=float(candle["o"]),
                        high_price=float(candle["h"]),
                        low_price=float(candle["l"]),
                        close_price=float(candle["c"]),
                        volume=int(candle["v"]) if candle.get("v") else None,
                        source=self.name,
                        quality_score=95  # Polygon provides high quality historical data
                    ))

                except (ValueError, TypeError, KeyError) as e:
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
        """Check the health of Polygon.io API"""
        try:
            start_time = datetime.now()

            # Use a simple endpoint to check health - get market status
            test_data = await self._make_request("v1/marketstatus/now")

            response_time = (datetime.now() - start_time).total_seconds() * 1000

            if test_data and "market" in test_data:
                status = DataSourceStatus.HEALTHY
                message = "Polygon.io API is healthy"
            else:
                status = DataSourceStatus.DEGRADED
                message = "Polygon.io API responding but data quality issues"

            return HealthStatus(
                status=status,
                last_successful_call=datetime.now(),
                error_count=self.error_count,
                response_time_ms=response_time,
                rate_limit_remaining=max(0, self._calls_per_minute - len(self._request_timestamps)),
                message=message
            )

        except Exception as e:
            logger.warning(f"Polygon health check failed: {e}")
            return HealthStatus(
                status=DataSourceStatus.DEGRADED if self.error_count < 5 else DataSourceStatus.UNHEALTHY,
                last_successful_call=self._last_request_time,
                error_count=self.error_count,
                response_time_ms=None,
                rate_limit_remaining=max(0, self._calls_per_minute - len(self._request_timestamps)),
                message=f"Health check failed: {e}"
            )

    async def validate_symbol(self, symbol: str) -> bool:
        """Validate if a symbol is supported by Polygon.io"""
        try:
            # Try to get ticker details
            endpoint = f"v3/reference/tickers/{symbol.upper()}"
            data = await self._make_request(endpoint)
            return bool(data and "results" in data and data["results"])
        except (SymbolNotFoundError, DataSourceError):
            return False
        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {e}")
            return False

    async def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols from Polygon.io"""
        try:
            # Get list of active stocks
            params = {
                "market": "stocks",
                "active": "true",
                "sort": "ticker",
                "order": "asc",
                "limit": 1000
            }

            data = await self._make_request("v3/reference/tickers", params)

            symbols = []
            if isinstance(data, dict) and "results" in data:
                for item in data["results"]:
                    if item.get("ticker") and item.get("active", True):
                        # Filter for common stocks and avoid complex instruments
                        ticker = item["ticker"]
                        if (len(ticker) <= 5 and
                            "." not in ticker and
                            "-" not in ticker and
                            ticker.isalpha()):
                            symbols.append(ticker)

            return sorted(symbols)

        except Exception as e:
            logger.error(f"Error getting supported symbols: {e}")
            # Return a basic list of common symbols if the endpoint fails
            return ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "SPY", "QQQ", "IWM"]

    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get current rate limit information"""
        return {
            "requests_last_minute": len(self._request_timestamps),
            "per_minute_limit": self._calls_per_minute,
            "minute_remaining": max(0, self._calls_per_minute - len(self._request_timestamps)),
            "last_request_time": self._last_request_time.isoformat() if self._last_request_time else None,
            "premium_tier": self._premium_tier
        }

    def _map_interval(self, interval: str) -> tuple[str, int]:
        """Map standard interval to Polygon timespan and multiplier format"""
        interval_mapping = {
            "1min": ("minute", 1),
            "5min": ("minute", 5),
            "15min": ("minute", 15),
            "30min": ("minute", 30),
            "1h": ("hour", 1),
            "1d": ("day", 1),
            "1w": ("week", 1),
            "1m": ("month", 1)
        }

        timespan, multiplier = interval_mapping.get(interval, ("day", 1))
        if (timespan, multiplier) != interval_mapping.get(interval, ("day", 1)):
            logger.info(f"Mapped interval {interval} to {multiplier} {timespan} for Polygon")

        return timespan, multiplier

    def _calculate_quality_score(self, trade_data: Dict[str, Any], quote_data: Dict[str, Any] = None) -> int:
        """Calculate quality score for Polygon data"""
        score = 90  # Base score for Polygon (high quality provider)

        # Adjust based on data completeness
        if trade_data.get("p") is not None:  # Price
            score += 5
        if trade_data.get("s") is not None:  # Size/volume
            score += 2
        if trade_data.get("t") is not None:  # Timestamp
            # Check data freshness
            trade_time = datetime.fromtimestamp(trade_data["t"] / 1000)
            age_minutes = (datetime.now() - trade_time).total_seconds() / 60
            if age_minutes < 5:  # Within 5 minutes
                score += 3

        # Quote data bonus
        if quote_data:
            if quote_data.get("P") and quote_data.get("p"):  # Bid and ask
                score += 2

        return min(100, max(0, score))

    async def get_market_status(self) -> Dict[str, Any]:
        """Get current market status (Polygon-specific feature)

        This is an additional feature that leverages Polygon's market status capabilities.
        """
        try:
            data = await self._make_request("v1/marketstatus/now")

            if isinstance(data, dict):
                return {
                    "market": data.get("market", "closed"),
                    "server_time": data.get("serverTime"),
                    "exchanges": data.get("exchanges", {}),
                    "currencies": data.get("currencies", {})
                }

            return {}

        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return {}

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()