"""Finnhub Data Source Implementation

Finnhub provides comprehensive financial data including real-time quotes, historical data,
and financial news. Free tier includes 60 API calls per minute.
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


class FinnhubSource(BaseDataSource):
    """Finnhub data source implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("finnhub", config)

        # API configuration
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise AuthenticationError("Finnhub API key is required")

        self.base_url = config.get("base_url", "https://finnhub.io/api/v1")

        # Rate limiting configuration
        self._calls_per_minute = config.get("calls_per_minute", 60)  # Free tier limit
        self._premium_tier = config.get("premium_tier", False)

        if self._premium_tier:
            self._calls_per_minute = config.get("premium_calls_per_minute", 300)

        # Request tracking
        self._last_request_time = None
        self._request_timestamps = []

        # Connection management
        self._session = None
        self._timeout = config.get("timeout", 30)
        self._max_retries = config.get("max_retries", 3)

        logger.info(f"Finnhub source initialized with {self._calls_per_minute} calls/minute limit")

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
        """Make an authenticated request to Finnhub API"""
        await self._check_rate_limits()

        if params is None:
            params = {}

        # Add API token
        params["token"] = self.api_key

        url = f"{self.base_url}/{endpoint}"

        session = await self._get_session()

        for attempt in range(self._max_retries):
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Check for API error in response body
                        if isinstance(data, dict) and "error" in data:
                            error_message = data["error"]
                            if "Invalid API key" in error_message:
                                raise AuthenticationError(f"Authentication error: {error_message}")
                            elif "API limit" in error_message or "rate limit" in error_message.lower():
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

        Uses Finnhub's quote endpoint for real-time pricing data.
        """
        try:
            # Get quote data
            data = await self._make_request("quote", {"symbol": symbol.upper()})

            if not data or data.get("c") is None:
                raise SymbolNotFoundError(f"No price data returned for symbol {symbol}")

            # Finnhub returns: c=current, h=high, l=low, o=open, pc=previous close, t=timestamp
            current_price = CurrentPrice(
                symbol=symbol.upper(),
                price=float(data["c"]),  # Current price
                timestamp=datetime.fromtimestamp(data.get("t", datetime.now().timestamp())),
                volume=None,  # Quote endpoint doesn't include volume
                bid=None,  # Quote endpoint doesn't include bid/ask
                ask=None,
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

        Finnhub supports daily and intraday data.
        """
        try:
            # Convert dates to Unix timestamps
            start_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
            end_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())

            # Map interval
            resolution = self._map_resolution(interval)

            params = {
                "symbol": symbol.upper(),
                "resolution": resolution,
                "from": start_timestamp,
                "to": end_timestamp
            }

            data = await self._make_request("stock/candle", params)

            if not data or data.get("s") != "ok":
                if data and data.get("s") == "no_data":
                    raise DataSourceError(f"No data available for {symbol} in the specified time range")
                raise DataSourceError(f"No historical data returned for {symbol}")

            # Finnhub returns arrays: c=close, h=high, l=low, o=open, t=timestamp, v=volume
            timestamps = data.get("t", [])
            opens = data.get("o", [])
            highs = data.get("h", [])
            lows = data.get("l", [])
            closes = data.get("c", [])
            volumes = data.get("v", [])

            if not timestamps:
                raise DataSourceError(f"No valid data found for {symbol} in date range {start_date} to {end_date}")

            price_data = []
            for i in range(len(timestamps)):
                try:
                    trade_date = datetime.fromtimestamp(timestamps[i]).date()

                    # Filter by date range
                    if not (start_date <= trade_date <= end_date):
                        continue

                    price_data.append(PriceData(
                        symbol=symbol.upper(),
                        timestamp=datetime.combine(trade_date, datetime.min.time()),
                        open_price=float(opens[i]) if i < len(opens) else 0,
                        high_price=float(highs[i]) if i < len(highs) else 0,
                        low_price=float(lows[i]) if i < len(lows) else 0,
                        close_price=float(closes[i]) if i < len(closes) else 0,
                        volume=int(volumes[i]) if i < len(volumes) and volumes[i] is not None else None,
                        source=self.name,
                        quality_score=90  # Finnhub provides reliable historical data
                    ))

                except (ValueError, TypeError, IndexError) as e:
                    logger.warning(f"Skipping invalid data point for {symbol} at index {i}: {e}")
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
        """Check the health of Finnhub API"""
        try:
            start_time = datetime.now()

            # Use a simple endpoint to check health - get quote for a known symbol
            test_data = await self._make_request("quote", {"symbol": "AAPL"})

            response_time = (datetime.now() - start_time).total_seconds() * 1000

            if test_data and test_data.get("c") is not None:
                status = DataSourceStatus.HEALTHY
                message = "Finnhub API is healthy"
            else:
                status = DataSourceStatus.DEGRADED
                message = "Finnhub API responding but data quality issues"

            return HealthStatus(
                status=status,
                last_successful_call=datetime.now(),
                error_count=self.error_count,
                response_time_ms=response_time,
                rate_limit_remaining=max(0, self._calls_per_minute - len(self._request_timestamps)),
                message=message
            )

        except Exception as e:
            logger.warning(f"Finnhub health check failed: {e}")
            return HealthStatus(
                status=DataSourceStatus.DEGRADED if self.error_count < 5 else DataSourceStatus.UNHEALTHY,
                last_successful_call=self._last_request_time,
                error_count=self.error_count,
                response_time_ms=None,
                rate_limit_remaining=max(0, self._calls_per_minute - len(self._request_timestamps)),
                message=f"Health check failed: {e}"
            )

    async def validate_symbol(self, symbol: str) -> bool:
        """Validate if a symbol is supported by Finnhub"""
        try:
            quote_data = await self._make_request("quote", {"symbol": symbol.upper()})
            # Check if we got valid price data
            return bool(quote_data and quote_data.get("c") is not None)
        except SymbolNotFoundError:
            return False
        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {e}")
            return False

    async def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols from Finnhub"""
        try:
            # Finnhub provides various stock symbols endpoints
            # Start with US stocks
            data = await self._make_request("stock/symbol", {"exchange": "US"})

            symbols = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get("symbol"):
                        # Filter for common stocks and avoid complex instruments
                        symbol = item["symbol"]
                        if (len(symbol) <= 5 and
                            "." not in symbol and
                            "-" not in symbol and
                            symbol.isalpha()):
                            symbols.append(symbol)

            return sorted(symbols[:1000])  # Limit to prevent oversized responses

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

    def _map_resolution(self, interval: str) -> str:
        """Map standard interval to Finnhub resolution format"""
        interval_mapping = {
            "1min": "1",
            "5min": "5",
            "15min": "15",
            "30min": "30",
            "1h": "60",
            "1d": "D",
            "1w": "W",
            "1m": "M"
        }

        mapped = interval_mapping.get(interval, "D")
        if mapped != interval:
            logger.info(f"Mapped interval {interval} to {mapped} for Finnhub")

        return mapped

    def _calculate_quality_score(self, quote_data: Dict[str, Any]) -> int:
        """Calculate quality score for Finnhub data"""
        score = 85  # Base score for Finnhub

        # Adjust based on data completeness
        if quote_data.get("c") is not None:  # Current price
            score += 5
        if quote_data.get("h") is not None:  # High
            score += 2
        if quote_data.get("l") is not None:  # Low
            score += 2
        if quote_data.get("o") is not None:  # Open
            score += 2
        if quote_data.get("pc") is not None:  # Previous close
            score += 2
        if quote_data.get("t") is not None:  # Timestamp
            # Check data freshness
            quote_time = datetime.fromtimestamp(quote_data["t"])
            age_minutes = (datetime.now() - quote_time).total_seconds() / 60
            if age_minutes < 15:  # Within 15 minutes
                score += 2

        return min(100, max(0, score))

    async def get_company_news(self, symbol: str, days_back: int = 7) -> List[Dict[str, Any]]:
        """Get recent company news (Finnhub-specific feature)

        This is an additional feature that leverages Finnhub's news capabilities.
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            params = {
                "symbol": symbol.upper(),
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d")
            }

            data = await self._make_request("company-news", params)

            if isinstance(data, list):
                return data[:10]  # Limit to 10 most recent articles

            return []

        except Exception as e:
            logger.error(f"Error getting company news for {symbol}: {e}")
            return []

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()