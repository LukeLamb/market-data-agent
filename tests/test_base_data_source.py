"""Tests for Base Data Source Interface"""

import pytest
from datetime import datetime, date
from src.data_sources.base import (
    BaseDataSource,
    PriceData,
    CurrentPrice,
    HealthStatus,
    DataSourceStatus,
    DataSourceError,
    RateLimitError,
    SymbolNotFoundError
)


class MockDataSource(BaseDataSource):
    """Mock implementation for testing"""

    async def get_current_price(self, symbol: str) -> CurrentPrice:
        if symbol == "INVALID":
            raise SymbolNotFoundError(f"Symbol {symbol} not found")
        return CurrentPrice(
            symbol=symbol,
            price=150.0,
            timestamp=datetime.now(),
            source=self.name,
            quality_score=95
        )

    async def get_historical_data(self, symbol: str, start_date: date, end_date: date, interval: str = "1d"):
        if symbol == "INVALID":
            raise SymbolNotFoundError(f"Symbol {symbol} not found")
        return [
            PriceData(
                symbol=symbol,
                timestamp=datetime.now(),
                open_price=149.0,
                high_price=151.0,
                low_price=148.0,
                close_price=150.0,
                volume=1000000,
                source=self.name,
                quality_score=95
            )
        ]

    async def get_health_status(self) -> HealthStatus:
        return HealthStatus(
            status=DataSourceStatus.HEALTHY,
            response_time_ms=150.0,
            message="Mock source is healthy"
        )

    def get_supported_symbols(self):
        return ["AAPL", "GOOGL", "MSFT"]

    async def validate_symbol(self, symbol: str) -> bool:
        return symbol in self.get_supported_symbols()


class TestBaseDataSource:
    """Test cases for BaseDataSource functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.config = {
            "rate_limit": 60,
            "daily_limit": 1000
        }
        self.data_source = MockDataSource("test_source", self.config)

    def test_initialization(self):
        """Test data source initialization"""
        assert self.data_source.name == "test_source"
        assert self.data_source.config == self.config
        assert self.data_source.error_count == 0
        assert self.data_source.last_successful_call is None

    @pytest.mark.asyncio
    async def test_get_current_price_success(self):
        """Test successful current price retrieval"""
        price = await self.data_source.get_current_price("AAPL")
        assert isinstance(price, CurrentPrice)
        assert price.symbol == "AAPL"
        assert price.price == 150.0
        assert price.source == "test_source"
        assert price.quality_score == 95

    @pytest.mark.asyncio
    async def test_get_current_price_invalid_symbol(self):
        """Test current price with invalid symbol"""
        with pytest.raises(SymbolNotFoundError):
            await self.data_source.get_current_price("INVALID")

    @pytest.mark.asyncio
    async def test_get_historical_data_success(self):
        """Test successful historical data retrieval"""
        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 31)
        data = await self.data_source.get_historical_data("AAPL", start_date, end_date)

        assert isinstance(data, list)
        assert len(data) == 1
        assert isinstance(data[0], PriceData)
        assert data[0].symbol == "AAPL"
        assert data[0].source == "test_source"

    @pytest.mark.asyncio
    async def test_get_health_status(self):
        """Test health status check"""
        status = await self.data_source.get_health_status()
        assert isinstance(status, HealthStatus)
        assert status.status == DataSourceStatus.HEALTHY
        assert status.response_time_ms == 150.0

    def test_get_supported_symbols(self):
        """Test supported symbols retrieval"""
        symbols = self.data_source.get_supported_symbols()
        assert isinstance(symbols, list)
        assert "AAPL" in symbols
        assert "GOOGL" in symbols

    @pytest.mark.asyncio
    async def test_validate_symbol(self):
        """Test symbol validation"""
        assert await self.data_source.validate_symbol("AAPL") is True
        assert await self.data_source.validate_symbol("INVALID") is False

    def test_error_count_management(self):
        """Test error count tracking"""
        assert self.data_source.error_count == 0

        self.data_source.increment_error_count()
        assert self.data_source.error_count == 1

        self.data_source.increment_error_count()
        assert self.data_source.error_count == 2

        self.data_source.reset_error_count()
        assert self.data_source.error_count == 0

    def test_rate_limit_info(self):
        """Test rate limit information"""
        info = self.data_source.get_rate_limit_info()
        assert isinstance(info, dict)
        assert info["requests_per_minute"] == 60
        assert info["requests_per_day"] == 1000

    def test_string_representations(self):
        """Test string representations"""
        str_repr = str(self.data_source)
        assert "MockDataSource" in str_repr
        assert "test_source" in str_repr

        repr_str = repr(self.data_source)
        assert "MockDataSource" in repr_str
        assert "test_source" in repr_str


class TestDataModels:
    """Test data model validation"""

    def test_price_data_model(self):
        """Test PriceData model validation"""
        price_data = PriceData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open_price=100.0,
            high_price=105.0,
            low_price=99.0,
            close_price=103.0,
            volume=1000000,
            source="test",
            quality_score=95
        )
        assert price_data.symbol == "AAPL"
        assert price_data.quality_score == 95

    def test_current_price_model(self):
        """Test CurrentPrice model validation"""
        current_price = CurrentPrice(
            symbol="AAPL",
            price=150.0,
            timestamp=datetime.now(),
            source="test",
            quality_score=90
        )
        assert current_price.symbol == "AAPL"
        assert current_price.price == 150.0

    def test_health_status_model(self):
        """Test HealthStatus model validation"""
        status = HealthStatus(
            status=DataSourceStatus.HEALTHY,
            response_time_ms=100.0,
            rate_limit_remaining=950
        )
        assert status.status == DataSourceStatus.HEALTHY
        assert status.response_time_ms == 100.0