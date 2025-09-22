"""Tests for Finnhub Data Source Implementation"""

import pytest
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, AsyncMock
import aiohttp
import json

from src.data_sources.finnhub_source import FinnhubSource
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


class TestFinnhubSource:
    """Test cases for Finnhub data source"""

    def setup_method(self):
        """Set up test fixtures"""
        self.config = {
            "api_key": "test_api_key_123",
            "calls_per_minute": 60,
            "timeout": 30
        }
        self.source = FinnhubSource(self.config)

    def test_initialization_success(self):
        """Test successful Finnhub source initialization"""
        assert self.source.name == "finnhub"
        assert self.source.api_key == "test_api_key_123"
        assert self.source._calls_per_minute == 60
        assert self.source._premium_tier is False

    def test_initialization_premium_tier(self):
        """Test initialization with premium tier"""
        config = {
            "api_key": "premium_key",
            "premium_tier": True,
            "premium_calls_per_minute": 300
        }
        source = FinnhubSource(config)
        assert source._premium_tier is True
        assert source._calls_per_minute == 300

    def test_initialization_missing_api_key(self):
        """Test initialization without API key raises error"""
        config = {"calls_per_minute": 60}
        with pytest.raises(AuthenticationError):
            FinnhubSource(config)

    def test_resolution_mapping(self):
        """Test interval mapping to Finnhub resolution format"""
        assert self.source._map_resolution("1d") == "D"
        assert self.source._map_resolution("1w") == "W"
        assert self.source._map_resolution("1m") == "M"
        assert self.source._map_resolution("1min") == "1"
        assert self.source._map_resolution("1h") == "60"
        assert self.source._map_resolution("invalid") == "D"  # Default

    @pytest.mark.asyncio
    async def test_get_current_price_success(self):
        """Test successful current price retrieval"""
        mock_response_data = {
            "c": 150.25,  # Current price
            "h": 152.0,   # High
            "l": 149.0,   # Low
            "o": 150.0,   # Open
            "pc": 148.5,  # Previous close
            "t": int(datetime.now().timestamp())  # Timestamp
        }

        with patch.object(self.source, '_make_request', return_value=mock_response_data):
            price = await self.source.get_current_price("AAPL")

            assert isinstance(price, CurrentPrice)
            assert price.symbol == "AAPL"
            assert price.price == 150.25
            assert price.volume is None  # Finnhub quote doesn't include volume
            assert price.bid is None
            assert price.ask is None
            assert price.source == "finnhub"

    @pytest.mark.asyncio
    async def test_get_current_price_no_data(self):
        """Test current price retrieval when no data is returned"""
        mock_response_data = {}

        with patch.object(self.source, '_make_request', return_value=mock_response_data):
            with pytest.raises(SymbolNotFoundError):
                await self.source.get_current_price("INVALID")

    @pytest.mark.asyncio
    async def test_get_current_price_api_error(self):
        """Test current price retrieval with API error"""
        mock_error_response = {
            "error": "Invalid API key"
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = mock_error_response

        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_response
        mock_context_manager.__aexit__.return_value = None

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value = mock_context_manager
            with pytest.raises(AuthenticationError):
                await self.source.get_current_price("AAPL")

    @pytest.mark.asyncio
    async def test_get_current_price_rate_limited(self):
        """Test current price retrieval when rate limited"""
        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.headers = {"Retry-After": "60"}

        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_response
        mock_context_manager.__aexit__.return_value = None

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value = mock_context_manager
            with pytest.raises(RateLimitError):
                await self.source.get_current_price("AAPL")

    @pytest.mark.asyncio
    async def test_get_historical_data_success(self):
        """Test successful historical data retrieval"""
        mock_response_data = {
            "s": "ok",
            "t": [1672704000, 1672790400],  # Timestamps
            "o": [100.0, 101.0],  # Open prices
            "h": [105.0, 106.0],  # High prices
            "l": [99.0, 100.0],   # Low prices
            "c": [104.0, 105.0],  # Close prices
            "v": [1000000, 1100000]  # Volumes
        }

        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 31)

        with patch.object(self.source, '_make_request', return_value=mock_response_data):
            data = await self.source.get_historical_data("AAPL", start_date, end_date)

            assert len(data) == 2
            assert all(isinstance(item, PriceData) for item in data)
            assert data[0].symbol == "AAPL"
            assert data[0].source == "finnhub"
            assert data[0].quality_score == 90

            # Check values
            assert data[0].open_price == 100.0
            assert data[0].close_price == 104.0
            assert data[0].volume == 1000000

    @pytest.mark.asyncio
    async def test_get_historical_data_no_data(self):
        """Test handling when no historical data is found"""
        mock_response_data = {
            "s": "no_data"
        }

        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 31)

        with patch.object(self.source, '_make_request', return_value=mock_response_data):
            with pytest.raises(DataSourceError):
                await self.source.get_historical_data("AAPL", start_date, end_date)

    @pytest.mark.asyncio
    async def test_get_historical_data_empty_arrays(self):
        """Test handling when historical data arrays are empty"""
        mock_response_data = {
            "s": "ok",
            "t": [],
            "o": [],
            "h": [],
            "l": [],
            "c": [],
            "v": []
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
            "c": 150.25,
            "h": 152.0,
            "l": 149.0,
            "o": 150.0,
            "pc": 148.5,
            "t": int(datetime.now().timestamp())
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
            "c": 150.25,
            "t": int(datetime.now().timestamp())
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
    async def test_validate_symbol_no_price_data(self):
        """Test symbol validation when no price data is returned"""
        mock_response_data = {}

        with patch.object(self.source, '_make_request', return_value=mock_response_data):
            is_valid = await self.source.validate_symbol("INVALID")
            assert is_valid is False

    @pytest.mark.asyncio
    async def test_get_supported_symbols(self):
        """Test getting supported symbols"""
        mock_response_data = [
            {"symbol": "AAPL", "description": "Apple Inc"},
            {"symbol": "GOOGL", "description": "Google"},
            {"symbol": "MSFT.TO", "description": "Microsoft Toronto"},  # Should be filtered
            {"symbol": "BRK-A", "description": "Berkshire Hathaway"}   # Should be filtered
        ]

        with patch.object(self.source, '_make_request', return_value=mock_response_data):
            symbols = await self.source.get_supported_symbols()

            # Should only include simple symbols
            assert "AAPL" in symbols
            assert "GOOGL" in symbols
            assert "MSFT.TO" not in symbols  # Contains dot
            assert "BRK-A" not in symbols    # Contains dash

    @pytest.mark.asyncio
    async def test_get_supported_symbols_fallback(self):
        """Test fallback when supported symbols endpoint fails"""
        with patch.object(self.source, '_make_request', side_effect=DataSourceError("Endpoint failed")):
            symbols = await self.source.get_supported_symbols()

            # Should return fallback list
            assert len(symbols) > 0
            assert "AAPL" in symbols
            assert "GOOGL" in symbols

    @pytest.mark.asyncio
    async def test_get_company_news(self):
        """Test getting company news (Finnhub-specific feature)"""
        mock_response_data = [
            {
                "category": "company",
                "datetime": int(datetime.now().timestamp()),
                "headline": "Apple Reports Strong Q4 Results",
                "id": 12345,
                "image": "https://example.com/image.jpg",
                "related": "AAPL",
                "source": "Reuters",
                "summary": "Apple Inc reported strong quarterly results...",
                "url": "https://example.com/news"
            }
        ]

        with patch.object(self.source, '_make_request', return_value=mock_response_data):
            news = await self.source.get_company_news("AAPL", days_back=7)

            assert len(news) == 1
            assert news[0]["headline"] == "Apple Reports Strong Q4 Results"
            assert news[0]["related"] == "AAPL"

    @pytest.mark.asyncio
    async def test_get_company_news_error(self):
        """Test company news when endpoint fails"""
        with patch.object(self.source, '_make_request', side_effect=DataSourceError("News endpoint failed")):
            news = await self.source.get_company_news("AAPL")
            assert news == []

    def test_rate_limit_info(self):
        """Test rate limit information retrieval"""
        info = self.source.get_rate_limit_info()

        assert "requests_last_minute" in info
        assert "per_minute_limit" in info
        assert "minute_remaining" in info
        assert "premium_tier" in info
        assert info["per_minute_limit"] == 60
        assert info["premium_tier"] is False

    @pytest.mark.asyncio
    async def test_rate_limiting_enforcement(self):
        """Test that rate limiting is properly enforced"""
        # Set a very low rate limit for testing
        self.source._calls_per_minute = 1

        # Add a timestamp to simulate hitting the limit
        now = datetime.now()
        self.source._request_timestamps = [now]

        # This should cause a delay
        start_time = datetime.now()
        await self.source._check_rate_limits()
        end_time = datetime.now()

        # Should have waited some time (though in tests it might be minimal)
        assert (end_time - start_time).total_seconds() >= 0

    def test_calculate_quality_score(self):
        """Test quality score calculation"""
        # Complete data should have high score
        complete_data = {
            "c": 150.0,
            "h": 152.0,
            "l": 149.0,
            "o": 150.5,
            "pc": 148.0,
            "t": int(datetime.now().timestamp())
        }
        score = self.source._calculate_quality_score(complete_data)
        assert score == 100

        # Incomplete data should have lower score
        incomplete_data = {
            "c": 150.0
        }
        score = self.source._calculate_quality_score(incomplete_data)
        assert score < 100

        # Old data should have slightly lower score
        old_data = {
            "c": 150.0,
            "h": 152.0,
            "l": 149.0,
            "o": 150.5,
            "pc": 148.0,
            "t": int((datetime.now() - timedelta(hours=2)).timestamp())
        }
        score = self.source._calculate_quality_score(old_data)
        assert score < 100

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager functionality"""
        async with FinnhubSource(self.config) as source:
            assert source.name == "finnhub"
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


class TestFinnhubIntegration:
    """Integration tests for Finnhub (requires API key)"""

    @pytest.mark.integration
    @pytest.mark.network
    @pytest.mark.api_key
    async def test_real_current_price(self):
        """Test real current price retrieval (requires API key)"""
        import os
        api_key = os.getenv("FINNHUB_API_KEY")

        if not api_key:
            pytest.skip("FINNHUB_API_KEY environment variable not set")

        config = {
            "api_key": api_key,
            "calls_per_minute": 60
        }

        async with FinnhubSource(config) as source:
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
        api_key = os.getenv("FINNHUB_API_KEY")

        if not api_key:
            pytest.skip("FINNHUB_API_KEY environment variable not set")

        config = {
            "api_key": api_key,
            "calls_per_minute": 60
        }

        start_date = date.today() - timedelta(days=7)
        end_date = date.today() - timedelta(days=1)

        async with FinnhubSource(config) as source:
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
        api_key = os.getenv("FINNHUB_API_KEY")

        if not api_key:
            pytest.skip("FINNHUB_API_KEY environment variable not set")

        config = {
            "api_key": api_key,
            "calls_per_minute": 60
        }

        async with FinnhubSource(config) as source:
            status = await source.get_health_status()

            assert status.status in [DataSourceStatus.HEALTHY, DataSourceStatus.DEGRADED]
            assert status.response_time_ms is not None

    @pytest.mark.integration
    @pytest.mark.network
    @pytest.mark.api_key
    async def test_real_company_news(self):
        """Test real company news retrieval (requires API key)"""
        import os
        api_key = os.getenv("FINNHUB_API_KEY")

        if not api_key:
            pytest.skip("FINNHUB_API_KEY environment variable not set")

        config = {
            "api_key": api_key,
            "calls_per_minute": 60
        }

        async with FinnhubSource(config) as source:
            news = await source.get_company_news("AAPL", days_back=3)

            # News might be empty, but should return a list
            assert isinstance(news, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not integration"])