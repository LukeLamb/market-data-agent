"""Testing Utilities

This module provides common utilities, fixtures, and helpers for testing
the Market Data Agent components.
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import os
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import yaml

from src.data_sources.base import (
    BaseDataSource,
    PriceData,
    CurrentPrice,
    HealthStatus,
    DataSourceStatus
)
from src.data_sources.source_manager import DataSourceManager
from src.config.config_manager import ConfigManager
from src.error_handling import ErrorHandler


class MockDataSource(BaseDataSource):
    """Mock data source for testing"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config or {})
        self.should_fail = False
        self.symbol_not_found = False
        self.current_price_data = {}
        self.historical_data = {}
        self.supported_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        self.call_count = 0

    async def get_current_price(self, symbol: str) -> CurrentPrice:
        """Mock current price retrieval"""
        self.call_count += 1

        if self.should_fail:
            raise Exception("Mock data source failure")

        if self.symbol_not_found or symbol not in self.supported_symbols:
            from src.data_sources.base import SymbolNotFoundError
            raise SymbolNotFoundError(f"Symbol {symbol} not found")

        # Return mock data or pre-configured data
        if symbol in self.current_price_data:
            return self.current_price_data[symbol]

        return CurrentPrice(
            symbol=symbol,
            price=150.0 + self.call_count,
            timestamp=datetime.now(),
            volume=1000000,
            bid=149.95,
            ask=150.05,
            source=self.name,
            quality_score=95
        )

    async def get_historical_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        interval: str = "1d"
    ) -> List[PriceData]:
        """Mock historical data retrieval"""
        self.call_count += 1

        if self.should_fail:
            raise Exception("Mock data source failure")

        if self.symbol_not_found or symbol not in self.supported_symbols:
            from src.data_sources.base import SymbolNotFoundError
            raise SymbolNotFoundError(f"Symbol {symbol} not found")

        # Return mock data or pre-configured data
        if symbol in self.historical_data:
            return self.historical_data[symbol]

        # Generate sample historical data
        data = []
        current_date = start_date
        base_price = 150.0

        while current_date <= end_date:
            data.append(PriceData(
                symbol=symbol,
                timestamp=datetime.combine(current_date, datetime.min.time()),
                open_price=base_price,
                high_price=base_price + 5.0,
                low_price=base_price - 3.0,
                close_price=base_price + 2.0,
                volume=1000000,
                source=self.name,
                quality_score=95
            ))
            current_date += timedelta(days=1)
            base_price += 1.0  # Trend upward

        return data

    async def get_health_status(self) -> HealthStatus:
        """Mock health status"""
        if self.should_fail:
            return HealthStatus(
                status=DataSourceStatus.UNHEALTHY,
                error_count=5,
                message="Mock source unhealthy"
            )

        return HealthStatus(
            status=DataSourceStatus.HEALTHY,
            error_count=0,
            message="Mock source healthy",
            response_time_ms=100.0
        )

    async def validate_symbol(self, symbol: str) -> bool:
        """Mock symbol validation"""
        return symbol in self.supported_symbols and not self.symbol_not_found

    async def get_supported_symbols(self) -> List[str]:
        """Mock supported symbols"""
        return self.supported_symbols.copy()

    def set_current_price_data(self, symbol: str, price_data: CurrentPrice):
        """Set mock current price data for a symbol"""
        self.current_price_data[symbol] = price_data

    def set_historical_data(self, symbol: str, historical_data: List[PriceData]):
        """Set mock historical data for a symbol"""
        self.historical_data[symbol] = historical_data

    def set_failure_mode(self, should_fail: bool):
        """Set whether this mock should fail"""
        self.should_fail = should_fail

    def set_symbol_not_found(self, symbol_not_found: bool):
        """Set whether symbols should be reported as not found"""
        self.symbol_not_found = symbol_not_found


@pytest.fixture
def mock_data_source():
    """Fixture providing a mock data source"""
    return MockDataSource("mock_source")


@pytest_asyncio.fixture
async def mock_source_manager():
    """Fixture providing a mock source manager"""
    config = {
        "max_failure_threshold": 3,
        "circuit_breaker_timeout": 60,
        "validation_enabled": False
    }
    manager = DataSourceManager(config)

    # Register a mock source
    mock_source = MockDataSource("test_source")
    manager.register_source("test_source", mock_source, 1)  # Primary priority

    yield manager
    await manager.close()


@pytest.fixture
def mock_config_manager():
    """Fixture providing a mock configuration manager"""
    config_manager = ConfigManager()
    return config_manager


@pytest.fixture
def mock_error_handler():
    """Fixture providing a mock error handler"""
    return ErrorHandler()


@pytest.fixture
def temp_config_file():
    """Fixture providing a temporary config file"""
    config_content = {
        "environment": "test",
        "debug": True,
        "api": {
            "host": "127.0.0.1",
            "port": 8001
        },
        "yfinance": {
            "enabled": True,
            "priority": 1
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_content, f)
        temp_file = f.name

    yield temp_file

    # Cleanup
    os.unlink(temp_file)


@pytest.fixture
def sample_price_data():
    """Fixture providing sample price data"""
    return PriceData(
        symbol="AAPL",
        timestamp=datetime(2023, 1, 1, 9, 30, 0),
        open_price=150.0,
        high_price=155.0,
        low_price=149.0,
        close_price=154.0,
        volume=1000000,
        source="test_source",
        quality_score=95
    )


@pytest.fixture
def sample_current_price():
    """Fixture providing sample current price data"""
    return CurrentPrice(
        symbol="AAPL",
        price=150.0,
        timestamp=datetime(2023, 1, 1, 10, 0, 0),
        volume=1000000,
        bid=149.95,
        ask=150.05,
        source="test_source",
        quality_score=95
    )


@pytest.fixture
def sample_price_history():
    """Fixture providing sample price history"""
    return [
        PriceData(
            symbol="AAPL",
            timestamp=datetime(2023, 1, i, 9, 30, 0),
            open_price=150.0 + i,
            high_price=155.0 + i,
            low_price=149.0 + i,
            close_price=154.0 + i,
            volume=1000000,
            source="test_source",
            quality_score=95
        )
        for i in range(1, 6)  # 5 days of data
    ]


class TestDataBuilder:
    """Builder pattern for creating test data"""

    @staticmethod
    def create_price_data(
        symbol: str = "AAPL",
        timestamp: Optional[datetime] = None,
        open_price: float = 150.0,
        high_price: float = 155.0,
        low_price: float = 149.0,
        close_price: float = 154.0,
        volume: int = 1000000,
        source: str = "test_source",
        quality_score: int = 95
    ) -> PriceData:
        """Create price data with customizable fields"""
        return PriceData(
            symbol=symbol,
            timestamp=timestamp or datetime.now(),
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            volume=volume,
            source=source,
            quality_score=quality_score
        )

    @staticmethod
    def create_current_price(
        symbol: str = "AAPL",
        price: float = 150.0,
        timestamp: Optional[datetime] = None,
        volume: Optional[int] = 1000000,
        bid: Optional[float] = 149.95,
        ask: Optional[float] = 150.05,
        source: str = "test_source",
        quality_score: int = 95
    ) -> CurrentPrice:
        """Create current price data with customizable fields"""
        return CurrentPrice(
            symbol=symbol,
            price=price,
            timestamp=timestamp or datetime.now(),
            volume=volume,
            bid=bid,
            ask=ask,
            source=source,
            quality_score=quality_score
        )

    @staticmethod
    def create_price_series(
        symbol: str = "AAPL",
        start_date: date = date(2023, 1, 1),
        days: int = 5,
        base_price: float = 150.0,
        source: str = "test_source"
    ) -> List[PriceData]:
        """Create a series of price data"""
        data = []
        current_date = start_date

        for i in range(days):
            price = base_price + i
            data.append(TestDataBuilder.create_price_data(
                symbol=symbol,
                timestamp=datetime.combine(current_date, datetime.min.time()),
                open_price=price,
                high_price=price + 5.0,
                low_price=price - 3.0,
                close_price=price + 2.0,
                source=source
            ))
            current_date += timedelta(days=1)

        return data


class AsyncTestCase:
    """Base class for async test cases with common utilities"""

    @pytest_asyncio.fixture(autouse=True)
    async def setup_async(self):
        """Async setup for test cases"""
        # Override in subclasses for custom setup
        pass

    async def wait_for_condition(
        self,
        condition_func,
        timeout: float = 5.0,
        interval: float = 0.1
    ) -> bool:
        """Wait for a condition to be true with timeout"""
        start_time = asyncio.get_event_loop().time()

        while True:
            if condition_func():
                return True

            if asyncio.get_event_loop().time() - start_time > timeout:
                return False

            await asyncio.sleep(interval)


def assert_price_data_equal(actual: PriceData, expected: PriceData, tolerance: float = 0.01):
    """Assert that two PriceData objects are equal within tolerance"""
    assert actual.symbol == expected.symbol
    assert abs(actual.open_price - expected.open_price) <= tolerance
    assert abs(actual.high_price - expected.high_price) <= tolerance
    assert abs(actual.low_price - expected.low_price) <= tolerance
    assert abs(actual.close_price - expected.close_price) <= tolerance
    assert actual.volume == expected.volume
    assert actual.source == expected.source


def assert_current_price_equal(actual: CurrentPrice, expected: CurrentPrice, tolerance: float = 0.01):
    """Assert that two CurrentPrice objects are equal within tolerance"""
    assert actual.symbol == expected.symbol
    assert abs(actual.price - expected.price) <= tolerance
    assert actual.source == expected.source

    if actual.bid is not None and expected.bid is not None:
        assert abs(actual.bid - expected.bid) <= tolerance
    if actual.ask is not None and expected.ask is not None:
        assert abs(actual.ask - expected.ask) <= tolerance


def create_mock_response(status_code: int = 200, json_data: Optional[Dict] = None, text: str = ""):
    """Create a mock HTTP response"""
    mock_response = Mock()
    mock_response.status_code = status_code
    mock_response.json.return_value = json_data or {}
    mock_response.text = text
    mock_response.raise_for_status = Mock()

    if status_code >= 400:
        from requests.exceptions import HTTPError
        mock_response.raise_for_status.side_effect = HTTPError(f"HTTP {status_code}")

    return mock_response


def patch_datetime_now(mock_datetime):
    """Utility to patch datetime.now() for consistent testing"""
    mock_datetime.now.return_value = datetime(2023, 1, 1, 12, 0, 0)
    mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
    return mock_datetime


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Test markers for categorizing tests
def integration_test(func):
    """Mark a test as an integration test"""
    return pytest.mark.integration(func)


def slow_test(func):
    """Mark a test as slow"""
    return pytest.mark.slow(func)


def network_test(func):
    """Mark a test that requires network access"""
    return pytest.mark.network(func)


def requires_api_key(func):
    """Mark a test that requires API keys"""
    return pytest.mark.api_key(func)