"""Tests for API Endpoints"""

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, date
import json

from src.api.endpoints import app, get_data_manager
from src.data_sources.source_manager import DataSourceManager
from src.data_sources.base import (
    CurrentPrice,
    PriceData,
    HealthStatus,
    DataSourceStatus,
    SymbolNotFoundError,
    RateLimitError,
    DataSourceError
)


class TestAPIEndpoints:
    """Test cases for API endpoints"""

    @pytest_asyncio.fixture
    async def mock_manager(self):
        """Create a mock data manager for testing"""
        manager = AsyncMock(spec=DataSourceManager)

        # Mock current price response
        manager.get_current_price.return_value = CurrentPrice(
            symbol="AAPL",
            price=150.00,
            timestamp=datetime(2023, 1, 1, 10, 0, 0),
            volume=1000000,
            bid=149.95,
            ask=150.05,
            source="test_source",
            quality_score=95
        )

        # Mock historical data response
        manager.get_historical_data.return_value = [
            PriceData(
                symbol="AAPL",
                timestamp=datetime(2023, 1, 1, 9, 30, 0),
                open_price=149.50,
                high_price=151.00,
                low_price=149.00,
                close_price=150.50,
                volume=1000000,
                source="test_source",
                quality_score=95
            )
        ]

        # Mock health status response
        manager.get_source_health_status.return_value = {
            "test_source": HealthStatus(
                status=DataSourceStatus.HEALTHY,
                error_count=0,
                message="Test source healthy",
                response_time_ms=100.0
            )
        }

        # Mock source statistics
        manager.get_source_statistics.return_value = {
            "test_source": {
                "priority": "PRIMARY",
                "reliability": 0.95,
                "circuit_breaker_state": "closed",
                "failure_count": 0,
                "last_failure": None,
                "is_available": True
            }
        }

        # Mock symbol validation
        manager.validate_symbol.return_value = {
            "test_source": True
        }

        # Mock sources attribute
        manager.sources = {
            "test_source": MagicMock()
        }
        manager._is_source_available.return_value = True

        # Mock get_supported_symbols
        async def mock_get_supported_symbols():
            return ["AAPL", "GOOGL", "MSFT", "TSLA"]

        manager.sources["test_source"].get_supported_symbols = mock_get_supported_symbols

        return manager

    @pytest.fixture
    def client(self, mock_manager):
        """Create test client with mocked dependencies"""

        def override_get_data_manager():
            return mock_manager

        app.dependency_overrides[get_data_manager] = override_get_data_manager

        with TestClient(app) as client:
            yield client

        # Clean up
        app.dependency_overrides.clear()

    def test_root_endpoint(self, client):
        """Test root endpoint returns API information"""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "Market Data Agent API"
        assert data["version"] == "1.0.0"
        assert "docs" in data
        assert "health" in data

    def test_health_check_healthy(self, client):
        """Test health check with healthy sources"""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "sources" in data
        assert data["healthy_sources"] == 1
        assert data["total_sources"] == 1

    def test_get_sources(self, client):
        """Test get sources endpoint"""
        response = client.get("/sources")
        assert response.status_code == 200

        data = response.json()
        assert "sources" in data
        assert "total_sources" in data
        assert data["total_sources"] == 1
        assert "test_source" in data["sources"]

    def test_get_current_price_success(self, client):
        """Test successful current price retrieval"""
        response = client.get("/price/AAPL")
        assert response.status_code == 200

        data = response.json()
        assert data["symbol"] == "AAPL"
        assert data["price"] == 150.00
        assert data["source"] == "test_source"
        assert data["quality_score"] == 95
        assert "timestamp" in data

    def test_get_current_price_symbol_not_found(self, client, mock_manager):
        """Test current price with symbol not found"""
        mock_manager.get_current_price.side_effect = SymbolNotFoundError("Symbol not found")

        response = client.get("/price/INVALID")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_get_current_price_rate_limit(self, client, mock_manager):
        """Test current price with rate limit error"""
        mock_manager.get_current_price.side_effect = RateLimitError("Rate limit exceeded")

        response = client.get("/price/AAPL")
        assert response.status_code == 429
        assert "rate limit" in response.json()["detail"].lower()

    def test_get_current_price_data_source_error(self, client, mock_manager):
        """Test current price with data source error"""
        mock_manager.get_current_price.side_effect = DataSourceError("Data source error")

        response = client.get("/price/AAPL")
        assert response.status_code == 503
        assert "temporarily unavailable" in response.json()["detail"]

    def test_get_historical_data_success(self, client):
        """Test successful historical data retrieval"""
        response = client.get("/historical/AAPL?start_date=2023-01-01&end_date=2023-01-31")
        assert response.status_code == 200

        data = response.json()
        assert data["symbol"] == "AAPL"
        assert data["start_date"] == "2023-01-01"
        assert data["end_date"] == "2023-01-31"
        assert data["data_points"] == 1
        assert len(data["data"]) == 1
        assert data["data"][0]["open"] == 149.50

    def test_get_historical_data_invalid_date_range(self, client):
        """Test historical data with invalid date range"""
        response = client.get("/historical/AAPL?start_date=2023-01-31&end_date=2023-01-01")
        assert response.status_code == 400
        assert "Start date must be before end date" in response.json()["detail"]

    def test_get_historical_data_future_date(self, client):
        """Test historical data with future start date"""
        future_date = "2099-01-01"
        response = client.get(f"/historical/AAPL?start_date={future_date}&end_date={future_date}")
        assert response.status_code == 400
        assert "cannot be in the future" in response.json()["detail"]

    def test_get_historical_data_symbol_not_found(self, client, mock_manager):
        """Test historical data with symbol not found"""
        mock_manager.get_historical_data.side_effect = SymbolNotFoundError("Symbol not found")

        response = client.get("/historical/INVALID?start_date=2023-01-01&end_date=2023-01-31")
        assert response.status_code == 404

    def test_validate_symbol_success(self, client):
        """Test successful symbol validation"""
        response = client.get("/validate/AAPL")
        assert response.status_code == 200

        data = response.json()
        assert data["symbol"] == "AAPL"
        assert data["is_valid"] is True
        assert "test_source" in data["valid_sources"]
        assert data["total_sources_checked"] == 1

    def test_validate_symbol_invalid(self, client, mock_manager):
        """Test symbol validation for invalid symbol"""
        mock_manager.validate_symbol.return_value = {"test_source": False}

        response = client.get("/validate/INVALID")
        assert response.status_code == 200

        data = response.json()
        assert data["symbol"] == "INVALID"
        assert data["is_valid"] is False
        assert len(data["valid_sources"]) == 0

    def test_get_supported_symbols(self, client):
        """Test get supported symbols endpoint"""
        response = client.get("/symbols")
        assert response.status_code == 200

        data = response.json()
        assert "symbols" in data
        assert "total_unique_symbols" in data
        assert "returned_symbols" in data
        assert "source_counts" in data
        assert len(data["symbols"]) > 0

    def test_get_supported_symbols_with_limit(self, client):
        """Test get supported symbols with limit"""
        response = client.get("/symbols?limit=2")
        assert response.status_code == 200

        data = response.json()
        assert len(data["symbols"]) <= 2

    def test_symbol_case_insensitive(self, client):
        """Test that symbol lookups are case insensitive"""
        # Test lowercase symbol
        response = client.get("/price/aapl")
        assert response.status_code == 200

        data = response.json()
        assert data["symbol"] == "AAPL"  # Should be converted to uppercase

    def test_health_check_failure(self, client, mock_manager):
        """Test health check when data manager fails"""
        mock_manager.get_source_health_status.side_effect = Exception("Manager error")

        response = client.get("/health")
        assert response.status_code == 500

    def test_missing_query_parameters(self, client):
        """Test endpoints with missing required query parameters"""
        # Missing start_date and end_date
        response = client.get("/historical/AAPL")
        assert response.status_code == 422  # Validation error

    def test_invalid_query_parameters(self, client):
        """Test endpoints with invalid query parameters"""
        # Invalid date format
        response = client.get("/historical/AAPL?start_date=invalid&end_date=2023-01-31")
        assert response.status_code == 422  # Validation error

    def test_interval_parameter(self, client):
        """Test historical data with different interval"""
        response = client.get("/historical/AAPL?start_date=2023-01-01&end_date=2023-01-31&interval=1wk")
        assert response.status_code == 200

        data = response.json()
        assert data["interval"] == "1wk"

    def test_concurrent_requests(self, client, mock_manager):
        """Test handling of concurrent requests"""
        import concurrent.futures
        import threading

        def make_request():
            return client.get("/price/AAPL")

        # Make multiple concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in futures]

        # All requests should succeed
        for response in responses:
            assert response.status_code == 200

    def test_api_documentation_accessible(self, client):
        """Test that API documentation is accessible"""
        response = client.get("/docs")
        assert response.status_code == 200

        response = client.get("/redoc")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__])