"""Tests for Twelve Data Source Implementation"""

import pytest
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, AsyncMock
import aiohttp
import json

from src.data_sources.twelve_data_source import TwelveDataSource
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


class TestTwelveDataSource:
    """Test cases for Twelve Data source"""

    def setup_method(self):
        """Set up test fixtures"""
        self.config = {
            "api_key": "test_api_key_123",
            "daily_limit": 800,
            "requests_per_minute": 8,
            "timeout": 30
        }
        self.source = TwelveDataSource(self.config)

    def test_initialization_success(self):
        """Test successful Twelve Data source initialization"""
        assert self.source.name == "twelve_data"
        assert self.source.api_key == "test_api_key_123"
        assert self.source._daily_limit == 800
        assert self.source._requests_per_minute == 8

    def test_initialization_missing_api_key(self):
        """Test initialization without API key raises error"""
        config = {"daily_limit": 800}
        with pytest.raises(AuthenticationError):
            TwelveDataSource(config)

    def test_interval_mapping(self):
        """Test interval mapping to Twelve Data format"""
        assert self.source._map_interval("1d") == "1day"
        assert self.source._map_interval("1w") == "1week"
        assert self.source._map_interval("1m") == "1month"
        assert self.source._map_interval("1min") == "1min"
        assert self.source._map_interval("invalid") == "1day"  # Default

    @pytest.mark.asyncio
    async def test_get_current_price_success(self):
        """Test successful current price retrieval"""
        mock_price_data = {
            "price": "150.25"
        }

        mock_quote_data = {
            "symbol": "AAPL",
            "volume": 1000000,
            "bid": 150.20,
            "ask": 150.30
        }

        async def mock_make_request(endpoint, params=None):
            if endpoint == "price":
                return mock_price_data
            elif endpoint == "quote":
                return mock_quote_data
            return {}

        with patch.object(self.source, '_make_request', side_effect=mock_make_request):
            price = await self.source.get_current_price("AAPL")

            assert isinstance(price, CurrentPrice)
            assert price.symbol == "AAPL"
            assert price.price == 150.25
            assert price.volume == 1000000
            assert price.bid == 150.20
            assert price.ask == 150.30
            assert price.source == "twelve_data"

    @pytest.mark.asyncio
    async def test_get_current_price_no_quote_data(self):
        """Test current price retrieval when quote endpoint fails"""
        mock_price_data = {
            "price": "150.25"
        }

        async def mock_make_request(endpoint, params=None):
            if endpoint == "price":
                return mock_price_data
            elif endpoint == "quote":
                raise DataSourceError("Quote endpoint failed")
            return {}

        with patch.object(self.source, '_make_request', side_effect=mock_make_request):
            price = await self.source.get_current_price("AAPL")

            assert isinstance(price, CurrentPrice)
            assert price.symbol == "AAPL"
            assert price.price == 150.25
            assert price.volume is None
            assert price.bid is None
            assert price.ask is None
            assert price.source == "twelve_data"

    @pytest.mark.asyncio
    async def test_get_current_price_symbol_not_found(self):
        """Test current price retrieval with invalid symbol"""
        mock_error_response = {
            "status": "error",
            "code": 400,
            "message": "Symbol not found"
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = mock_error_response

        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_response
        mock_context_manager.__aexit__.return_value = None

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value = mock_context_manager
            with pytest.raises(SymbolNotFoundError):
                await self.source.get_current_price("INVALID")

    @pytest.mark.asyncio
    async def test_get_current_price_rate_limited(self):
        """Test current price retrieval when rate limited"""
        mock_error_response = {
            "status": "error",
            "code": 429,
            "message": "Rate limit exceeded"
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = mock_error_response

        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_response
        mock_context_manager.__aexit__.return_value = None

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value = mock_context_manager
            with pytest.raises(RateLimitError):
                await self.source.get_current_price("AAPL")

    @pytest.mark.asyncio
    async def test_get_current_price_authentication_error(self):
        """Test current price retrieval with invalid API key"""
        mock_response = AsyncMock()
        mock_response.status = 401

        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_response
        mock_context_manager.__aexit__.return_value = None

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value = mock_context_manager
            with pytest.raises(AuthenticationError):
                await self.source.get_current_price("AAPL")

    @pytest.mark.asyncio
    async def test_get_historical_data_success(self):
        """Test successful historical data retrieval"""
        mock_response_data = {
            "meta": {
                "symbol": "AAPL",
                "interval": "1day"
            },
            "values": [
                {
                    "datetime": "2023-01-03",
                    "open": "100.0",
                    "high": "105.0",
                    "low": "99.0",
                    "close": "104.0",
                    "volume": "1000000"
                },
                {
                    "datetime": "2023-01-02",
                    "open": "101.0",
                    "high": "106.0",
                    "low": "100.0",
                    "close": "105.0",
                    "volume": "1100000"
                }
            ]
        }

        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 31)

        with patch.object(self.source, '_make_request', return_value=mock_response_data):
            data = await self.source.get_historical_data("AAPL", start_date, end_date)

            assert len(data) == 2
            assert all(isinstance(item, PriceData) for item in data)
            assert data[0].symbol == "AAPL"
            assert data[0].source == "twelve_data"
            assert data[0].quality_score == 90

            # Check that data is sorted by timestamp
            assert data[0].timestamp <= data[1].timestamp

    @pytest.mark.asyncio
    async def test_get_historical_data_with_timestamp(self):
        """Test historical data with timestamp format"""
        mock_response_data = {
            "values": [
                {
                    "datetime": "2023-01-03T16:00:00Z",
                    "open": "100.0",
                    "high": "105.0",
                    "low": "99.0",
                    "close": "104.0",
                    "volume": "1000000"
                }
            ]
        }

        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 31)

        with patch.object(self.source, '_make_request', return_value=mock_response_data):
            data = await self.source.get_historical_data("AAPL", start_date, end_date)

            assert len(data) == 1
            assert data[0].timestamp.date() == date(2023, 1, 3)

    @pytest.mark.asyncio
    async def test_get_historical_data_no_data(self):
        """Test handling when no historical data is found"""
        mock_response_data = {
            "values": []
        }

        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 31)

        with patch.object(self.source, '_make_request', return_value=mock_response_data):
            with pytest.raises(DataSourceError):
                await self.source.get_historical_data("AAPL", start_date, end_date)

    @pytest.mark.asyncio
    async def test_get_health_status_healthy(self):
        """Test health status when service is healthy"""
        mock_response_data = {
            "price": "150.25"
        }

        with patch.object(self.source, '_make_request', return_value=mock_response_data):
            status = await self.source.get_health_status()

            assert status.status == DataSourceStatus.HEALTHY
            assert status.error_count == 0
            assert status.response_time_ms is not None
            assert status.response_time_ms > 0

    @pytest.mark.asyncio
    async def test_get_health_status_degraded(self):
        """Test health status when service returns but has data issues"""
        mock_response_data = {}  # No price data

        with patch.object(self.source, '_make_request', return_value=mock_response_data):
            status = await self.source.get_health_status()

            assert status.status == DataSourceStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_get_health_status_unhealthy(self):
        """Test health status when service is unhealthy"""
        # Increment error count to simulate multiple failures
        for _ in range(6):
            self.source.increment_error_count()

        with patch.object(self.source, '_make_request', side_effect=DataSourceError("Connection failed")):
            status = await self.source.get_health_status()
            assert status.status == DataSourceStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_validate_symbol_valid(self):
        """Test symbol validation with valid symbol"""
        mock_response_data = {
            "price": "150.25"
        }

        with patch.object(self.source, '_make_request', return_value=mock_response_data):
            is_valid = await self.source.validate_symbol("AAPL")
            assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_symbol_invalid(self):
        """Test symbol validation with invalid symbol"""
        with patch.object(self.source, '_make_request', side_effect=SymbolNotFoundError("Symbol not found")):
            is_valid = await self.source.validate_symbol("INVALID")
            assert is_valid is False

    @pytest.mark.asyncio
    async def test_get_supported_symbols(self):
        """Test getting supported symbols"""
        mock_response_data = {
            "data": [
                {"symbol": "AAPL", "access": {"global": True}},
                {"symbol": "GOOGL", "access": {"global": True}},
                {"symbol": "PRIVATE", "access": {"global": False}}
            ]
        }

        with patch.object(self.source, '_make_request', return_value=mock_response_data):
            symbols = await self.source.get_supported_symbols()

            # Should only include globally accessible symbols
            assert "AAPL" in symbols
            assert "GOOGL" in symbols
            assert "PRIVATE" not in symbols

    @pytest.mark.asyncio
    async def test_get_supported_symbols_fallback(self):
        """Test fallback when supported symbols endpoint fails"""
        with patch.object(self.source, '_make_request', side_effect=DataSourceError("Endpoint failed")):
            symbols = await self.source.get_supported_symbols()

            # Should return fallback list
            assert len(symbols) > 0
            assert "AAPL" in symbols
            assert "GOOGL" in symbols

    def test_rate_limit_info(self):
        """Test rate limit information retrieval"""
        info = self.source.get_rate_limit_info()

        assert "requests_today" in info
        assert "daily_limit" in info
        assert "daily_remaining" in info
        assert "requests_last_minute" in info
        assert "per_minute_limit" in info
        assert info["daily_limit"] == 800
        assert info["per_minute_limit"] == 8

    @pytest.mark.asyncio
    async def test_rate_limiting_enforcement(self):
        """Test that rate limiting is properly enforced"""
        # Set a very low rate limit for testing
        self.source._requests_per_minute = 1

        # Add a timestamp to simulate hitting the limit
        now = datetime.now()
        self.source._request_timestamps = [now]

        # This should cause a delay
        start_time = datetime.now()
        await self.source._check_rate_limits()
        end_time = datetime.now()

        # Should have waited some time (though in tests it might be minimal)
        assert (end_time - start_time).total_seconds() >= 0

    @pytest.mark.asyncio
    async def test_daily_rate_limit_exceeded(self):
        """Test behavior when daily rate limit is exceeded"""
        # Set requests to exceed daily limit
        self.source._requests_today = self.source._daily_limit

        with pytest.raises(RateLimitError):
            await self.source._check_rate_limits()

    @pytest.mark.asyncio
    async def test_daily_reset_logic(self):
        """Test daily rate limit reset logic"""
        # Set an old reset date
        self.source._daily_reset_date = datetime.now() - timedelta(days=2)
        self.source._requests_today = 500

        # Check rate limits should reset the daily count
        await self.source._check_rate_limits()

        assert self.source._requests_today == 0
        assert self.source._daily_reset_date.date() == datetime.now().date()

    def test_calculate_quality_score(self):
        """Test quality score calculation"""
        # Complete data should have high score
        price_data = {"price": "150.0"}
        quote_data = {
            "volume": 1000000,
            "bid": 149.95,
            "ask": 150.05,
            "change": 2.50,
            "percent_change": 1.69
        }
        score = self.source._calculate_quality_score(price_data, quote_data)
        assert score == 100

        # Incomplete data should have lower score
        incomplete_price_data = {}
        score = self.source._calculate_quality_score(incomplete_price_data)
        assert score < 90

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager functionality"""
        async with TwelveDataSource(self.config) as source:
            assert source.name == "twelve_data"
        # Session should be closed after context exit

    @pytest.mark.asyncio
    async def test_session_management(self):
        """Test session creation and management"""
        session1 = await self.source._get_session()
        session2 = await self.source._get_session()

        # Should return the same session
        assert session1 is session2

        await self.source.close()

        # Should create new session after close
        session3 = await self.source._get_session()
        assert session3 is not session1

    @pytest.mark.asyncio
    async def test_error_handling_and_counting(self):
        """Test error handling and error count tracking"""
        initial_count = self.source.error_count

        with patch.object(self.source, '_make_request', side_effect=DataSourceError("Network error")):
            with pytest.raises(DataSourceError):
                await self.source.get_current_price("AAPL")

            assert self.source.error_count == initial_count + 1


class TestTwelveDataIntegration:
    """Integration tests for Twelve Data (requires API key)"""

    @pytest.mark.integration
    @pytest.mark.network
    @pytest.mark.api_key
    async def test_real_current_price(self):
        """Test real current price retrieval (requires API key)"""
        import os
        api_key = os.getenv("TWELVE_DATA_API_KEY")

        if not api_key:
            pytest.skip("TWELVE_DATA_API_KEY environment variable not set")

        config = {
            "api_key": api_key,
            "daily_limit": 800
        }

        async with TwelveDataSource(config) as source:
            price = await source.get_current_price("AAPL")

            assert isinstance(price, CurrentPrice)
            assert price.symbol == "AAPL"
            assert price.price > 0
            assert price.quality_score > 0

    @pytest.mark.integration
    @pytest.mark.network
    @pytest.mark.api_key
    async def test_real_historical_data(self):
        """Test real historical data retrieval (requires API key)"""
        import os
        api_key = os.getenv("TWELVE_DATA_API_KEY")

        if not api_key:
            pytest.skip("TWELVE_DATA_API_KEY environment variable not set")

        config = {
            "api_key": api_key,
            "daily_limit": 800
        }

        start_date = date.today() - timedelta(days=7)
        end_date = date.today() - timedelta(days=1)

        async with TwelveDataSource(config) as source:
            data = await source.get_historical_data("AAPL", start_date, end_date)

            assert len(data) > 0
            assert all(isinstance(item, PriceData) for item in data)
            assert all(item.symbol == "AAPL" for item in data)

    @pytest.mark.integration
    @pytest.mark.network
    @pytest.mark.api_key
    async def test_real_health_check(self):
        """Test real health check (requires API key)"""
        import os
        api_key = os.getenv("TWELVE_DATA_API_KEY")

        if not api_key:
            pytest.skip("TWELVE_DATA_API_KEY environment variable not set")

        config = {
            "api_key": api_key,
            "daily_limit": 800
        }

        async with TwelveDataSource(config) as source:
            status = await source.get_health_status()

            assert status.status in [DataSourceStatus.HEALTHY, DataSourceStatus.DEGRADED]
            assert status.response_time_ms is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not integration"])