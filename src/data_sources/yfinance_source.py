"""YFinance Data Source Implementation

This module implements the BaseDataSource interface using Yahoo Finance data
through the yfinance library. Provides free access to stock market data with
15-20 minute delays for real-time prices.
"""

import asyncio
import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
import logging

import yfinance as yf
import pandas as pd

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

logger = logging.getLogger(__name__)


class YFinanceSource(BaseDataSource):
    """Yahoo Finance data source implementation

    Features:
    - Free access, no API key required
    - Historical data with good coverage
    - Real-time prices (15-20 minute delay)
    - Support for stocks, ETFs, indices
    - Rate limiting protection
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize YFinance data source

        Args:
            config: Configuration dictionary with rate limits and settings
        """
        super().__init__("yfinance", config)
        self._rate_limit = config.get("rate_limit", 120)  # requests per hour
        self._request_times = []
        self._last_health_check = None
        self._supported_symbols_cache = None
        self._cache_expiry = None

        logger.info(f"Initialized YFinance source with rate limit: {self._rate_limit}/hour")

    async def get_current_price(self, symbol: str) -> CurrentPrice:
        """Get current price for a symbol

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            CurrentPrice object with latest available data

        Raises:
            SymbolNotFoundError: If symbol is not found
            RateLimitError: If rate limit is exceeded
            DataSourceError: For other errors
        """
        await self._check_rate_limit()

        try:
            ticker = yf.Ticker(symbol)

            # Get current price using fast_info (most efficient)
            try:
                fast_info = ticker.fast_info
                current_price = fast_info.get('lastPrice')

                if current_price is None or current_price <= 0:
                    # Fallback to regular info
                    info = ticker.info
                    current_price = info.get('currentPrice') or info.get('regularMarketPrice')

                if current_price is None or current_price <= 0:
                    raise SymbolNotFoundError(f"No price data available for symbol {symbol}")

                # Always get full info for volume/bid/ask data
                info = ticker.info

                # Get additional data if available
                volume = (info.get('regularMarketVolume') or
                         info.get('volume') or
                         fast_info.get('regularMarketVolume'))

                bid = (info.get('bid') or
                      info.get('regularMarketBid'))

                ask = (info.get('ask') or
                      info.get('regularMarketAsk'))

                self._record_request()
                self.reset_error_count()

                return CurrentPrice(
                    symbol=symbol.upper(),
                    price=float(current_price),
                    timestamp=datetime.now(),
                    volume=int(volume) if volume else None,
                    bid=float(bid) if bid else None,
                    ask=float(ask) if ask else None,
                    source=self.name,
                    quality_score=85  # YFinance has ~15-20 min delay
                )

            except Exception as e:
                if "404" in str(e) or "not found" in str(e).lower():
                    raise SymbolNotFoundError(f"Symbol {symbol} not found")
                raise

        except SymbolNotFoundError:
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
            interval: Data interval ('1d', '1h', '5m', etc.)

        Returns:
            List of PriceData objects sorted by timestamp

        Raises:
            SymbolNotFoundError: If symbol is not found
            RateLimitError: If rate limit is exceeded
            DataSourceError: For other errors
        """
        await self._check_rate_limit()

        try:
            ticker = yf.Ticker(symbol)

            # Convert dates to strings for yfinance
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")  # Include end date

            # Download historical data
            hist = ticker.history(
                start=start_str,
                end=end_str,
                interval=interval,
                auto_adjust=True,
                prepost=True
            )

            if hist.empty:
                raise SymbolNotFoundError(f"No historical data found for symbol {symbol}")

            self._record_request()
            self.reset_error_count()

            # Convert to our standard format
            price_data = []
            for index, row in hist.iterrows():
                # Handle both timezone-aware and naive datetimes
                if hasattr(index, 'tz_localize'):
                    timestamp = index.tz_localize(None) if index.tz else index
                else:
                    timestamp = index

                price_data.append(PriceData(
                    symbol=symbol.upper(),
                    timestamp=timestamp,
                    open_price=float(row['Open']),
                    high_price=float(row['High']),
                    low_price=float(row['Low']),
                    close_price=float(row['Close']),
                    volume=int(row['Volume']) if pd.notna(row['Volume']) else 0,
                    adjusted_close=float(row['Close']),  # yfinance auto-adjusts
                    source=self.name,
                    quality_score=90  # Historical data is generally high quality
                ))

            # Sort by timestamp
            price_data.sort(key=lambda x: x.timestamp)

            return price_data

        except SymbolNotFoundError:
            raise
        except Exception as e:
            self.increment_error_count()
            logger.error(f"Error getting historical data for {symbol}: {e}")
            raise DataSourceError(f"Failed to get historical data for {symbol}: {e}")

    async def get_health_status(self) -> HealthStatus:
        """Check health status by testing a known symbol

        Returns:
            HealthStatus object with current status
        """
        start_time = time.time()

        try:
            # Test with a reliable symbol
            test_symbol = "AAPL"
            ticker = yf.Ticker(test_symbol)

            # Quick test using fast_info
            info = ticker.fast_info
            last_price = info.get('lastPrice')

            if last_price and last_price > 0:
                response_time = (time.time() - start_time) * 1000
                self._last_health_check = datetime.now()

                status = HealthStatus(
                    status=DataSourceStatus.HEALTHY,
                    last_successful_call=self.last_successful_call,
                    error_count=self.error_count,
                    response_time_ms=response_time,
                    message="YFinance API is responding normally"
                )

                self.update_health_status(status)
                return status
            else:
                raise Exception("No price data returned")

        except Exception as e:
            logger.warning(f"YFinance health check failed: {e}")

            # Determine status based on error count
            if self.error_count > 5:
                status_level = DataSourceStatus.UNHEALTHY
            elif self.error_count > 2:
                status_level = DataSourceStatus.DEGRADED
            else:
                status_level = DataSourceStatus.DEGRADED

            status = HealthStatus(
                status=status_level,
                last_successful_call=self.last_successful_call,
                error_count=self.error_count,
                response_time_ms=(time.time() - start_time) * 1000,
                message=f"Health check failed: {str(e)[:100]}"
            )

            self.update_health_status(status)
            return status

    def get_supported_symbols(self) -> List[str]:
        """Get list of commonly supported symbols

        Returns:
            List of popular symbol strings

        Note:
            This returns a subset of popular symbols. YFinance supports
            thousands of symbols but listing them all would be impractical.
        """
        # Cache commonly traded symbols
        if self._supported_symbols_cache is None or self._is_cache_expired():
            self._supported_symbols_cache = [
                # Major US stocks
                "AAPL", "GOOGL", "GOOG", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX",
                "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "INTC",
                "CMCSA", "VZ", "ADBE", "CRM", "NKE", "PFE", "T", "ABT", "CVX", "KO",

                # Major ETFs
                "SPY", "QQQ", "IWM", "VTI", "VOO", "VEA", "VWO", "AGG", "BND", "GLD",

                # Major indices (as tradeable symbols)
                "^GSPC", "^IXIC", "^DJI", "^RUT", "^VIX",

                # Popular international
                "TSM", "ASML", "SAP", "TM", "NVO", "AZN"
            ]
            self._cache_expiry = datetime.now() + timedelta(hours=24)

        return self._supported_symbols_cache.copy()

    async def validate_symbol(self, symbol: str) -> bool:
        """Check if a symbol is valid by attempting to fetch basic info

        Args:
            symbol: Symbol to validate

        Returns:
            True if symbol is valid and has data
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info

            # Check if we can get basic price info
            last_price = info.get('lastPrice')
            return last_price is not None and last_price > 0

        except Exception:
            return False

    async def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting

        Raises:
            RateLimitError: If rate limit would be exceeded
        """
        now = time.time()

        # Remove requests older than 1 hour
        cutoff_time = now - 3600  # 1 hour ago
        self._request_times = [t for t in self._request_times if t > cutoff_time]

        # Check if we're at the limit
        if len(self._request_times) >= self._rate_limit:
            oldest_request = min(self._request_times)
            wait_time = 3600 - (now - oldest_request)

            if wait_time > 0:
                raise RateLimitError(
                    f"Rate limit exceeded. Wait {wait_time:.1f} seconds before next request"
                )

    def _record_request(self) -> None:
        """Record a successful request for rate limiting"""
        self._request_times.append(time.time())

    def _is_cache_expired(self) -> bool:
        """Check if the symbols cache has expired"""
        return (
            self._cache_expiry is None or
            datetime.now() > self._cache_expiry
        )

    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get current rate limiting information

        Returns:
            Dictionary with rate limit details
        """
        now = time.time()
        cutoff_time = now - 3600
        current_requests = len([t for t in self._request_times if t > cutoff_time])

        return {
            "requests_per_hour": self._rate_limit,
            "current_usage": current_requests,
            "remaining_requests": max(0, self._rate_limit - current_requests),
            "reset_time": max(self._request_times) + 3600 if self._request_times else now
        }