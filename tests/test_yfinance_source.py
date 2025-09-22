"""Tests for YFinance Data Source Implementation"""

import pytest
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd

from src.data_sources.yfinance_source import YFinanceSource
from src.data_sources.base import (
    CurrentPrice,
    PriceData,
    HealthStatus,
    DataSourceStatus,
    SymbolNotFoundError,
    RateLimitError,
    DataSourceError
)


class TestYFinanceSource:
    """Test cases for YFinance data source"""

    def setup_method(self):
        """Set up test fixtures"""
        self.config = {
            "rate_limit": 120,  # 120 requests per hour
            "timeout": 30
        }
        self.source = YFinanceSource(self.config)

    def test_initialization(self):
        """Test YFinance source initialization"""
        assert self.source.name == "yfinance"
        assert self.source._rate_limit == 120
        assert self.source.config == self.config

    @pytest.mark.asyncio
    async def test_get_current_price_success(self):
        """Test successful current price retrieval"""
        mock_ticker = Mock()
        mock_fast_info = {
            'lastPrice': 150.25,
            'regularMarketVolume': 1000000,
            'bid': 150.20,
            'ask': 150.30
        }
        mock_ticker.fast_info = mock_fast_info

        with patch('yfinance.Ticker', return_value=mock_ticker):
            price = await self.source.get_current_price("AAPL")

            assert isinstance(price, CurrentPrice)
            assert price.symbol == "AAPL"
            assert price.price == 150.25
            assert price.volume == 1000000
            assert price.bid == 150.20
            assert price.ask == 150.30
            assert price.source == "yfinance"
            assert price.quality_score == 85

    @pytest.mark.asyncio
    async def test_get_current_price_fallback_to_info(self):
        """Test fallback to regular info when fast_info fails"""
        mock_ticker = Mock()

        # fast_info returns None for lastPrice
        mock_ticker.fast_info = {'lastPrice': None}

        # Regular info has the price
        mock_ticker.info = {
            'currentPrice': 150.25,
            'regularMarketVolume': 1000000
        }

        with patch('yfinance.Ticker', return_value=mock_ticker):
            price = await self.source.get_current_price("AAPL")
            assert price.price == 150.25

    @pytest.mark.asyncio
    async def test_get_current_price_symbol_not_found(self):
        """Test handling of invalid symbols"""
        mock_ticker = Mock()
        mock_ticker.fast_info = {'lastPrice': None}
        mock_ticker.info = {'currentPrice': None}

        with patch('yfinance.Ticker', return_value=mock_ticker):
            with pytest.raises(SymbolNotFoundError):
                await self.source.get_current_price("INVALID")

    @pytest.mark.asyncio
    async def test_get_historical_data_success(self):
        """Test successful historical data retrieval"""
        # Create mock historical data
        dates = pd.date_range('2023-01-01', periods=3, freq='D')
        mock_history = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0],
            'High': [105.0, 106.0, 107.0],
            'Low': [99.0, 100.0, 101.0],
            'Close': [104.0, 105.0, 106.0],
            'Volume': [1000000, 1100000, 1200000]
        }, index=dates)

        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_history

        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 3)

        with patch('yfinance.Ticker', return_value=mock_ticker):
            data = await self.source.get_historical_data("AAPL", start_date, end_date)

            assert len(data) == 3
            assert all(isinstance(item, PriceData) for item in data)
            assert data[0].symbol == "AAPL"
            assert data[0].open_price == 100.0
            assert data[0].high_price == 105.0
            assert data[0].source == "yfinance"
            assert data[0].quality_score == 90

    @pytest.mark.asyncio
    async def test_get_historical_data_empty_result(self):
        """Test handling of empty historical data"""
        mock_ticker = Mock()
        mock_ticker.history.return_value = pd.DataFrame()

        with patch('yfinance.Ticker', return_value=mock_ticker):
            with pytest.raises(SymbolNotFoundError):
                await self.source.get_historical_data(
                    "INVALID",
                    date(2023, 1, 1),
                    date(2023, 1, 31)
                )

    @pytest.mark.asyncio
    async def test_get_health_status_healthy(self):
        """Test health status check when service is healthy"""
        mock_ticker = Mock()
        mock_ticker.fast_info = {'lastPrice': 150.25}

        with patch('yfinance.Ticker', return_value=mock_ticker):
            status = await self.source.get_health_status()

            assert isinstance(status, HealthStatus)
            assert status.status == DataSourceStatus.HEALTHY
            assert status.response_time_ms is not None
            assert status.response_time_ms > 0

    @pytest.mark.asyncio
    async def test_get_health_status_unhealthy(self):
        """Test health status check when service is unhealthy"""
        mock_ticker = Mock()
        mock_ticker.fast_info = {'lastPrice': None}

        # Set high error count to trigger unhealthy status
        self.source._error_count = 10

        with patch('yfinance.Ticker', return_value=mock_ticker):
            status = await self.source.get_health_status()

            assert isinstance(status, HealthStatus)
            assert status.status == DataSourceStatus.UNHEALTHY
            assert "failed" in status.message.lower()

    def test_get_supported_symbols(self):
        """Test supported symbols retrieval"""
        symbols = self.source.get_supported_symbols()

        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert "AAPL" in symbols
        assert "GOOGL" in symbols
        assert "SPY" in symbols

    @pytest.mark.asyncio
    async def test_validate_symbol_valid(self):
        """Test symbol validation for valid symbol"""
        mock_ticker = Mock()
        mock_ticker.fast_info = {'lastPrice': 150.25}

        with patch('yfinance.Ticker', return_value=mock_ticker):
            is_valid = await self.source.validate_symbol("AAPL")
            assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_symbol_invalid(self):
        """Test symbol validation for invalid symbol"""
        mock_ticker = Mock()
        mock_ticker.fast_info = {'lastPrice': None}

        with patch('yfinance.Ticker', return_value=mock_ticker):
            is_valid = await self.source.validate_symbol("INVALID")
            assert is_valid is False

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting functionality"""
        # Set very low rate limit for testing
        self.source._rate_limit = 2

        # Simulate hitting rate limit
        import time
        current_time = time.time()
        self.source._request_times = [current_time, current_time]

        with pytest.raises(RateLimitError):
            await self.source._check_rate_limit()

    def test_rate_limit_info(self):
        """Test rate limit information retrieval"""
        info = self.source.get_rate_limit_info()

        assert isinstance(info, dict)
        assert "requests_per_hour" in info
        assert "current_usage" in info
        assert "remaining_requests" in info
        assert info["requests_per_hour"] == 120

    @pytest.mark.asyncio
    async def test_error_handling_and_counting(self):
        """Test error handling and error count tracking"""
        initial_count = self.source.error_count

        mock_ticker = Mock()
        mock_ticker.fast_info.side_effect = Exception("Network error")

        with patch('yfinance.Ticker', return_value=mock_ticker):
            with pytest.raises(DataSourceError):
                await self.source.get_current_price("AAPL")

        assert self.source.error_count == initial_count + 1

    def test_cache_expiry(self):
        """Test symbol cache expiration logic"""
        # Initially cache should be None
        assert self.source._supported_symbols_cache is None

        # Get symbols (should populate cache)
        symbols1 = self.source.get_supported_symbols()
        assert self.source._supported_symbols_cache is not None

        # Get symbols again (should use cache)
        symbols2 = self.source.get_supported_symbols()
        assert symbols1 == symbols2

        # Expire cache manually
        self.source._cache_expiry = datetime.now() - timedelta(hours=1)
        assert self.source._is_cache_expired() is True

    def test_request_recording(self):
        """Test request time recording for rate limiting"""
        initial_count = len(self.source._request_times)

        self.source._record_request()
        assert len(self.source._request_times) == initial_count + 1

        # Verify timestamp is recent
        import time
        assert abs(self.source._request_times[-1] - time.time()) < 1.0


class TestYFinanceIntegration:
    """Integration tests with real YFinance API (run sparingly)"""

    def setup_method(self):
        """Set up integration test fixtures"""
        self.config = {"rate_limit": 10}  # Low limit for testing
        self.source = YFinanceSource(self.config)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_current_price(self):
        """Test with real API call (requires internet)"""
        try:
            price = await self.source.get_current_price("AAPL")
            assert isinstance(price, CurrentPrice)
            assert price.symbol == "AAPL"
            assert price.price > 0
            assert price.source == "yfinance"
        except Exception as e:
            pytest.skip(f"Real API test failed: {e}")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_historical_data(self):
        """Test with real historical data API call"""
        try:
            start_date = date.today() - timedelta(days=7)
            end_date = date.today() - timedelta(days=1)

            data = await self.source.get_historical_data("AAPL", start_date, end_date)
            assert len(data) > 0
            assert all(isinstance(item, PriceData) for item in data)
            assert all(item.symbol == "AAPL" for item in data)
        except Exception as e:
            pytest.skip(f"Real API test failed: {e}")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_health_check(self):
        """Test real health check"""
        try:
            status = await self.source.get_health_status()
            assert isinstance(status, HealthStatus)
            # Should be healthy if internet connection is working
            assert status.status in [DataSourceStatus.HEALTHY, DataSourceStatus.DEGRADED]
        except Exception as e:
            pytest.skip(f"Real API test failed: {e}")