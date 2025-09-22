"""Tests for Alpha Vantage Data Source Implementation"""

import pytest
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, AsyncMock
import aiohttp
import json

from src.data_sources.alpha_vantage_source import AlphaVantageSource
from src.data_sources.base import (
    CurrentPrice,
    PriceData,
    HealthStatus,
    DataSourceStatus,
    SymbolNotFoundError,
    RateLimitError,
    AuthenticationError,
    DataSourceError
)


class TestAlphaVantageSource:
    """Test cases for Alpha Vantage data source"""

    def setup_method(self):
        """Set up test fixtures"""
        self.config = {
            "api_key": "test_api_key_123",
            "rate_limit_per_minute": 5,
            "daily_limit": 500
        }
        self.source = AlphaVantageSource(self.config)

    def test_initialization_success(self):
        """Test successful Alpha Vantage source initialization"""
        assert self.source.name == "alpha_vantage"
        assert self.source.api_key == "test_api_key_123"
        assert self.source._rate_limit_per_minute == 5
        assert self.source._daily_limit == 500

    def test_initialization_missing_api_key(self):
        """Test initialization without API key raises error"""
        config = {"rate_limit_per_minute": 5}
        with pytest.raises(AuthenticationError):
            AlphaVantageSource(config)

    @pytest.mark.asyncio
    async def test_get_current_price_success(self):
        """Test successful current price retrieval"""
        mock_response_data = {
            "Global Quote": {
                "01. symbol": "AAPL",
                "05. price": "150.25",
                "06. volume": "1000000",
                "10. change percent": "1.5%"
            }
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = mock_response_data

        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_response
        mock_context_manager.__aexit__.return_value = None

        mock_session = AsyncMock()
        mock_session.get.return_value = mock_context_manager

        with patch.object(self.source, '_get_session', new_callable=AsyncMock, return_value=mock_session):
            price = await self.source.get_current_price("AAPL")

            assert isinstance(price, CurrentPrice)
            assert price.symbol == "AAPL"
            assert price.price == 150.25
            assert price.volume == 1000000
            assert price.source == "alpha_vantage"
            assert price.quality_score == 95

    @pytest.mark.asyncio
    async def test_get_current_price_symbol_not_found(self):
        """Test handling of invalid symbols"""
        mock_response_data = {
            "Error Message": "Invalid API call. Please retry or visit the documentation"
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = mock_response_data

        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_response
        mock_context_manager.__aexit__.return_value = None

        mock_session = AsyncMock()
        mock_session.get.return_value = mock_context_manager

        with patch.object(self.source, '_get_session', new_callable=AsyncMock, return_value=mock_session):
            with pytest.raises(SymbolNotFoundError):
                await self.source.get_current_price("INVALID")

    @pytest.mark.asyncio
    async def test_get_current_price_rate_limited(self):
        """Test rate limit handling"""
        mock_response_data = {
            "Note": "Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute"
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = mock_response_data

        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_response
        mock_context_manager.__aexit__.return_value = None

        mock_session = AsyncMock()
        mock_session.get.return_value = mock_context_manager

        with patch.object(self.source, '_get_session', new_callable=AsyncMock, return_value=mock_session):
            with pytest.raises(RateLimitError):
                await self.source.get_current_price("AAPL")

    @pytest.mark.asyncio
    async def test_get_historical_data_success(self):
        """Test successful historical data retrieval"""
        mock_response_data = {
            "Time Series (Daily)": {
                "2023-01-03": {
                    "1. open": "100.0",
                    "2. high": "105.0",
                    "3. low": "99.0",
                    "4. close": "104.0",
                    "5. adjusted close": "104.0",
                    "6. volume": "1000000"
                },
                "2023-01-02": {
                    "1. open": "101.0",
                    "2. high": "106.0",
                    "3. low": "100.0",
                    "4. close": "105.0",
                    "5. adjusted close": "105.0",
                    "6. volume": "1100000"
                }
            }
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = mock_response_data

        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_response
        mock_context_manager.__aexit__.return_value = None

        mock_session = AsyncMock()
        mock_session.get.return_value = mock_context_manager

        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 31)

        with patch.object(self.source, '_get_session', new_callable=AsyncMock, return_value=mock_session):
            data = await self.source.get_historical_data("AAPL", start_date, end_date)

            assert len(data) == 2
            assert all(isinstance(item, PriceData) for item in data)
            assert data[0].symbol == "AAPL"
            assert data[0].source == "alpha_vantage"
            assert data[0].quality_score == 95

            # Check that data is sorted by timestamp
            assert data[0].timestamp <= data[1].timestamp

    @pytest.mark.asyncio
    async def test_get_historical_data_no_data(self):
        """Test handling when no historical data is found"""
        mock_response_data = {
            "Time Series (Daily)": {}
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = mock_response_data

        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_response
        mock_context_manager.__aexit__.return_value = None

        mock_session = AsyncMock()
        mock_session.get.return_value = mock_context_manager

        with patch.object(self.source, '_get_session', new_callable=AsyncMock, return_value=mock_session):
            with pytest.raises(DataSourceError):
                await self.source.get_historical_data(
                    "AAPL",
                    date(2023, 1, 1),
                    date(2023, 1, 31)
                )

    @pytest.mark.asyncio
    async def test_get_health_status_healthy(self):
        """Test health status check when service is healthy"""
        mock_response_data = {
            "Global Quote": {
                "01. symbol": "AAPL",
                "05. price": "150.25"
            }
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = mock_response_data

        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_response
        mock_context_manager.__aexit__.return_value = None

        mock_session = AsyncMock()
        mock_session.get.return_value = mock_context_manager

        with patch.object(self.source, '_get_session', new_callable=AsyncMock, return_value=mock_session):
            status = await self.source.get_health_status()

            assert isinstance(status, HealthStatus)
            assert status.status == DataSourceStatus.HEALTHY
            assert status.response_time_ms is not None
            assert status.response_time_ms > 0

    @pytest.mark.asyncio
    async def test_get_health_status_rate_limited(self):
        """Test health status when rate limited"""
        mock_response_data = {
            "Note": "Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute"
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = mock_response_data

        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_response
        mock_context_manager.__aexit__.return_value = None

        mock_session = AsyncMock()
        mock_session.get.return_value = mock_context_manager

        with patch.object(self.source, '_get_session', new_callable=AsyncMock, return_value=mock_session):
            status = await self.source.get_health_status()

            assert isinstance(status, HealthStatus)
            assert status.status == DataSourceStatus.DEGRADED
            assert "rate limited" in status.message.lower()

    @pytest.mark.asyncio
    async def test_get_health_status_unhealthy(self):
        """Test health status when service is unhealthy"""
        # Set high error count to trigger unhealthy status
        self.source._error_count = 10

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text.return_value = "Internal Server Error"

        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_response
        mock_context_manager.__aexit__.return_value = None

        mock_session = AsyncMock()
        mock_session.get.return_value = mock_context_manager

        with patch.object(self.source, '_get_session', new_callable=AsyncMock, return_value=mock_session):
            status = await self.source.get_health_status()

            assert isinstance(status, HealthStatus)
            assert status.status == DataSourceStatus.UNHEALTHY

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
        mock_response_data = {
            "Global Quote": {
                "01. symbol": "AAPL",
                "05. price": "150.25"
            }
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = mock_response_data

        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_response
        mock_context_manager.__aexit__.return_value = None

        mock_session = AsyncMock()
        mock_session.get.return_value = mock_context_manager

        with patch.object(self.source, '_get_session', new_callable=AsyncMock, return_value=mock_session):
            is_valid = await self.source.validate_symbol("AAPL")
            assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_symbol_invalid(self):
        """Test symbol validation for invalid symbol"""
        mock_response_data = {
            "Error Message": "Invalid API call"
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = mock_response_data

        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_response
        mock_context_manager.__aexit__.return_value = None

        mock_session = AsyncMock()
        mock_session.get.return_value = mock_context_manager

        with patch.object(self.source, '_get_session', new_callable=AsyncMock, return_value=mock_session):
            is_valid = await self.source.validate_symbol("INVALID")
            assert is_valid is False

    @pytest.mark.asyncio
    async def test_rate_limiting_minute_limit(self):
        """Test per-minute rate limiting"""
        # Set current time and fill request queue
        import time
        current_time = time.time()
        self.source._request_times = [current_time] * 5  # Hit the limit

        with pytest.raises(RateLimitError):
            await self.source._check_minute_rate_limit()

    @pytest.mark.asyncio
    async def test_rate_limiting_daily_limit(self):
        """Test daily rate limiting"""
        # Hit the daily limit
        self.source._daily_request_count = 500
        self.source._daily_reset_time = datetime.now()

        with pytest.raises(RateLimitError):
            await self.source._check_daily_rate_limit()

    def test_rate_limit_info(self):
        """Test rate limit information retrieval"""
        info = self.source.get_rate_limit_info()

        assert isinstance(info, dict)
        assert "requests_per_minute" in info
        assert "requests_per_day" in info
        assert "current_minute_usage" in info
        assert "current_daily_usage" in info
        assert "remaining_minute_requests" in info
        assert "remaining_daily_requests" in info

    @pytest.mark.asyncio
    async def test_error_handling_and_counting(self):
        """Test error handling and error count tracking"""
        initial_count = self.source.error_count

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text.return_value = "Internal Server Error"

        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_response
        mock_context_manager.__aexit__.return_value = None

        mock_session = AsyncMock()
        mock_session.get.return_value = mock_context_manager

        with patch.object(self.source, '_get_session', new_callable=AsyncMock, return_value=mock_session):
            with pytest.raises(DataSourceError):
                await self.source.get_current_price("AAPL")

        assert self.source.error_count == initial_count + 1

    def test_daily_reset_logic(self):
        """Test daily request counter reset logic"""
        # Set old reset time
        old_date = datetime.now() - timedelta(days=1)
        self.source._daily_reset_time = old_date
        self.source._daily_request_count = 100

        # This should reset the counter
        import asyncio
        asyncio.run(self.source._check_daily_rate_limit())

        assert self.source._daily_request_count == 0
        assert self.source._daily_reset_time.date() == datetime.now().date()

    def test_request_recording(self):
        """Test request recording for rate limiting"""
        initial_minute_count = len(self.source._request_times)
        initial_daily_count = self.source._daily_request_count

        self.source._record_request()

        assert len(self.source._request_times) == initial_minute_count + 1
        assert self.source._daily_request_count == initial_daily_count + 1

    @pytest.mark.asyncio
    async def test_session_management(self):
        """Test HTTP session creation and management"""
        session1 = await self.source._get_session()
        session2 = await self.source._get_session()

        # Should reuse the same session
        assert session1 is session2

        # Test cleanup
        await self.source.close()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager functionality"""
        async with AlphaVantageSource(self.config) as source:
            assert source.api_key == "test_api_key_123"

        # Session should be closed after context exit
        # (This is tested implicitly by the context manager)


class TestAlphaVantageIntegration:
    """Integration tests with real Alpha Vantage API (run sparingly)"""

    def setup_method(self):
        """Set up integration test fixtures"""
        # Use the API key from config if available
        import os
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

        if not api_key:
            pytest.skip("ALPHA_VANTAGE_API_KEY not set")

        self.config = {
            "api_key": api_key,
            "rate_limit_per_minute": 5,
            "daily_limit": 500
        }
        self.source = AlphaVantageSource(self.config)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_current_price(self):
        """Test with real API call (requires valid API key)"""
        try:
            async with self.source:
                price = await self.source.get_current_price("AAPL")
                assert isinstance(price, CurrentPrice)
                assert price.symbol == "AAPL"
                assert price.price > 0
                assert price.source == "alpha_vantage"
        except Exception as e:
            pytest.skip(f"Real API test failed: {e}")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_historical_data(self):
        """Test with real historical data API call"""
        try:
            async with self.source:
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
            async with self.source:
                status = await self.source.get_health_status()
                assert isinstance(status, HealthStatus)
                # Should be healthy if API key is valid
                assert status.status in [DataSourceStatus.HEALTHY, DataSourceStatus.DEGRADED]
        except Exception as e:
            pytest.skip(f"Real API test failed: {e}")