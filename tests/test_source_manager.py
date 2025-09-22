"""Tests for Data Source Manager"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.data_sources.source_manager import (
    DataSourceManager,
    SourcePriority,
    SourceManagerError,
    NoHealthySourceError
)
from src.data_sources.base import (
    BaseDataSource,
    PriceData,
    CurrentPrice,
    HealthStatus,
    DataSourceStatus,
    DataSourceError,
    SymbolNotFoundError,
    RateLimitError
)


class MockDataSource(BaseDataSource):
    """Mock data source for testing"""

    def __init__(self, name: str, should_fail: bool = False, symbol_not_found: bool = False):
        super().__init__(name, {})
        self.should_fail = should_fail
        self.symbol_not_found = symbol_not_found
        self.call_count = 0

    async def get_current_price(self, symbol: str) -> CurrentPrice:
        self.call_count += 1

        if self.symbol_not_found:
            raise SymbolNotFoundError(f"Symbol {symbol} not found")

        if self.should_fail:
            raise DataSourceError("Mock data source failure")

        return CurrentPrice(
            symbol=symbol,
            price=150.0 + self.call_count,  # Slightly different prices
            timestamp=datetime.now(),
            source=self.name,
            quality_score=95
        )

    async def get_historical_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        interval: str = "1d"
    ) -> list[PriceData]:
        self.call_count += 1

        if self.symbol_not_found:
            raise SymbolNotFoundError(f"Symbol {symbol} not found")

        if self.should_fail:
            raise DataSourceError("Mock data source failure")

        return [
            PriceData(
                symbol=symbol,
                timestamp=datetime.combine(start_date, datetime.min.time()),
                open_price=150.0,
                high_price=155.0,
                low_price=149.0,
                close_price=154.0,
                volume=1000000,
                source=self.name,
                quality_score=95
            )
        ]

    async def get_health_status(self) -> HealthStatus:
        if self.should_fail:
            return HealthStatus(
                status=DataSourceStatus.UNHEALTHY,
                error_count=5,
                message="Mock source unhealthy"
            )

        return HealthStatus(
            status=DataSourceStatus.HEALTHY,
            error_count=0,
            message="Mock source healthy"
        )

    async def validate_symbol(self, symbol: str) -> bool:
        return not self.symbol_not_found

    async def get_supported_symbols(self) -> list[str]:
        """Return mock supported symbols"""
        if self.symbol_not_found:
            return []
        return ["AAPL", "GOOGL", "MSFT", "TSLA"]


class TestDataSourceManager:
    """Test cases for data source manager"""

    @pytest_asyncio.fixture
    async def manager(self):
        """Create manager instance for testing"""
        config = {
            "max_failure_threshold": 3,
            "circuit_breaker_timeout": 60,
            "validation_enabled": False  # Disable for cleaner testing
        }
        manager = DataSourceManager(config)
        yield manager
        await manager.close()

    def test_manager_initialization(self, manager):
        """Test manager initialization"""
        assert manager.max_failure_threshold == 3
        assert manager.circuit_breaker_timeout == 60
        assert len(manager.sources) == 0

    def test_register_source(self, manager):
        """Test source registration"""
        source = MockDataSource("test_source")
        manager.register_source("test", source, SourcePriority.PRIMARY)

        assert "test" in manager.sources
        assert manager.source_priorities["test"] == SourcePriority.PRIMARY
        assert manager.source_reliability["test"] == 1.0
        assert "test" in manager.circuit_breaker_states

    @pytest.mark.asyncio
    async def test_get_current_price_success(self, manager):
        """Test successful current price retrieval"""
        source = MockDataSource("test_source")
        manager.register_source("test", source, SourcePriority.PRIMARY)

        price = await manager.get_current_price("AAPL")

        assert price.symbol == "AAPL"
        assert price.price == 151.0  # 150.0 + 1 call
        assert price.source == "test_source"
        assert source.call_count == 1

    @pytest.mark.asyncio
    async def test_get_current_price_failover(self, manager):
        """Test failover to secondary source"""
        # Primary source that fails
        primary = MockDataSource("primary", should_fail=True)
        manager.register_source("primary", primary, SourcePriority.PRIMARY)

        # Secondary source that works
        secondary = MockDataSource("secondary")
        manager.register_source("secondary", secondary, SourcePriority.SECONDARY)

        price = await manager.get_current_price("AAPL")

        assert price.source == "secondary"
        assert primary.call_count == 1
        assert secondary.call_count == 1

    @pytest.mark.asyncio
    async def test_get_current_price_symbol_not_found(self, manager):
        """Test symbol not found in all sources"""
        source1 = MockDataSource("source1", symbol_not_found=True)
        source2 = MockDataSource("source2", symbol_not_found=True)

        manager.register_source("source1", source1, SourcePriority.PRIMARY)
        manager.register_source("source2", source2, SourcePriority.SECONDARY)

        with pytest.raises(SymbolNotFoundError):
            await manager.get_current_price("INVALID")

    @pytest.mark.asyncio
    async def test_get_current_price_no_healthy_sources(self, manager):
        """Test no healthy sources available"""
        source = MockDataSource("source", should_fail=True)
        manager.register_source("source", source, SourcePriority.PRIMARY)

        with pytest.raises(DataSourceError):
            await manager.get_current_price("AAPL")

    @pytest.mark.asyncio
    async def test_get_historical_data_success(self, manager):
        """Test successful historical data retrieval"""
        source = MockDataSource("test_source")
        manager.register_source("test", source, SourcePriority.PRIMARY)

        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 31)
        data = await manager.get_historical_data("AAPL", start_date, end_date)

        assert len(data) == 1
        assert data[0].symbol == "AAPL"
        assert data[0].source == "test_source"
        assert source.call_count == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, manager):
        """Test circuit breaker opens after failures"""
        source = MockDataSource("failing_source", should_fail=True)
        manager.register_source("failing", source, SourcePriority.PRIMARY)

        # Make enough failures to trigger circuit breaker
        for _ in range(3):
            try:
                await manager.get_current_price("AAPL")
            except DataSourceError:
                pass

        # Circuit breaker should be open
        circuit_state = manager.circuit_breaker_states["failing"]
        assert circuit_state["state"] == "open"
        assert circuit_state["failure_count"] == 3

        # Source should not be available
        assert not manager._is_source_available("failing")

    @pytest.mark.asyncio
    async def test_reliability_updates(self, manager):
        """Test source reliability score updates"""
        source = MockDataSource("test_source")
        manager.register_source("test", source, SourcePriority.PRIMARY)

        # Start with perfect reliability
        assert manager.source_reliability["test"] == 1.0

        # Success should maintain/increase reliability
        await manager.get_current_price("AAPL")
        reliability_after_success = manager.source_reliability["test"]
        assert reliability_after_success >= 1.0

        # Simulate failure
        source.should_fail = True
        try:
            await manager.get_current_price("AAPL")
        except DataSourceError:
            pass

        # Reliability should decrease after failure
        reliability_after_failure = manager.source_reliability["test"]
        assert reliability_after_failure < reliability_after_success

    @pytest.mark.asyncio
    async def test_source_prioritization(self, manager):
        """Test sources are tried in priority order"""
        # Register sources in different priority order
        secondary = MockDataSource("secondary")
        manager.register_source("secondary", secondary, SourcePriority.SECONDARY)

        primary = MockDataSource("primary")
        manager.register_source("primary", primary, SourcePriority.PRIMARY)

        backup = MockDataSource("backup")
        manager.register_source("backup", backup, SourcePriority.BACKUP)

        prioritized = manager._get_prioritized_sources()

        # Should be ordered by priority value (1, 2, 3)
        assert prioritized == ["primary", "secondary", "backup"]

    @pytest.mark.asyncio
    async def test_get_source_health_status(self, manager):
        """Test health status retrieval"""
        healthy_source = MockDataSource("healthy")
        unhealthy_source = MockDataSource("unhealthy", should_fail=True)

        manager.register_source("healthy", healthy_source, SourcePriority.PRIMARY)
        manager.register_source("unhealthy", unhealthy_source, SourcePriority.SECONDARY)

        health_statuses = await manager.get_source_health_status()

        assert health_statuses["healthy"].status == DataSourceStatus.HEALTHY
        assert health_statuses["unhealthy"].status == DataSourceStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_validate_symbol(self, manager):
        """Test symbol validation across sources"""
        valid_source = MockDataSource("valid_source")
        invalid_source = MockDataSource("invalid_source", symbol_not_found=True)

        manager.register_source("valid", valid_source, SourcePriority.PRIMARY)
        manager.register_source("invalid", invalid_source, SourcePriority.SECONDARY)

        results = await manager.validate_symbol("AAPL")

        assert results["valid"] is True
        assert results["invalid"] is False

    def test_get_source_statistics(self, manager):
        """Test source statistics retrieval"""
        source = MockDataSource("test_source")
        manager.register_source("test", source, SourcePriority.PRIMARY)

        stats = manager.get_source_statistics()

        assert "test" in stats
        assert stats["test"]["priority"] == "PRIMARY"
        assert stats["test"]["reliability"] == 1.0
        assert stats["test"]["circuit_breaker_state"] == "closed"
        assert stats["test"]["is_available"] is True

    def test_disabled_source_not_available(self, manager):
        """Test that disabled sources are not available"""
        source = MockDataSource("disabled_source")
        manager.register_source("disabled", source, SourcePriority.DISABLED)

        assert not manager._is_source_available("disabled")

    @pytest.mark.asyncio
    async def test_register_yfinance_source(self, manager):
        """Test YFinance source registration"""
        with patch('src.data_sources.source_manager.YFinanceSource') as mock_yf:
            mock_instance = AsyncMock()
            mock_yf.return_value = mock_instance

            config = {"priority": SourcePriority.PRIMARY.value}
            manager.register_yfinance_source(config)

            assert "yfinance" in manager.sources
            assert manager.source_priorities["yfinance"] == SourcePriority.PRIMARY
            mock_yf.assert_called_once_with(config)

    @pytest.mark.asyncio
    async def test_register_alpha_vantage_source(self, manager):
        """Test Alpha Vantage source registration"""
        with patch('src.data_sources.source_manager.AlphaVantageSource') as mock_av:
            mock_instance = AsyncMock()
            mock_av.return_value = mock_instance

            config = {"priority": SourcePriority.SECONDARY.value}
            manager.register_alpha_vantage_source(config)

            assert "alpha_vantage" in manager.sources
            assert manager.source_priorities["alpha_vantage"] == SourcePriority.SECONDARY
            mock_av.assert_called_once_with(config)

    @pytest.mark.asyncio
    async def test_health_monitoring_lifecycle(self, manager):
        """Test health monitoring start/stop"""
        await manager.start_health_monitoring()
        assert manager.health_check_task is not None
        assert not manager.health_check_task.done()

        await manager.stop_health_monitoring()
        assert manager.health_check_task.done()

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager functionality"""
        config = {"validation_enabled": False}

        async with DataSourceManager(config) as manager:
            source = MockDataSource("test_source")
            manager.register_source("test", source, SourcePriority.PRIMARY)

            price = await manager.get_current_price("AAPL")
            assert price.symbol == "AAPL"

        # Manager should be closed after context exit
        # Health monitoring task should be cancelled

    @pytest.mark.asyncio
    async def test_rate_limit_error_handling(self, manager):
        """Test rate limit error handling"""
        source = MockDataSource("rate_limited_source")
        manager.register_source("rate_limited", source, SourcePriority.PRIMARY)

        # Mock rate limit error
        async def mock_get_current_price(symbol):
            raise RateLimitError("Rate limit exceeded")

        source.get_current_price = mock_get_current_price

        with pytest.raises(RateLimitError):
            await manager.get_current_price("AAPL")

        # Should update reliability and circuit breaker
        assert manager.source_reliability["rate_limited"] < 1.0
        assert manager.circuit_breaker_states["rate_limited"]["failure_count"] == 1


if __name__ == "__main__":
    pytest.main([__file__])