"""
Tests for Real-Time Streaming Service
Tests streaming service functionality, client management, and performance
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal

from src.streaming.realtime_streaming_service import (
    RealtimeStreamingService,
    StreamingConfig,
    StreamingMode,
    StreamingMetrics,
    ClientSubscription
)
from src.data_sources.base import PriceData


class TestStreamingConfig:
    """Test streaming configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = StreamingConfig()

        assert config.websocket_host == "0.0.0.0"
        assert config.websocket_port == 8765
        assert config.max_connections == 10000
        assert config.default_mode == StreamingMode.SMART
        assert config.enable_compression is True
        assert config.enable_heartbeat is True

    def test_custom_config(self):
        """Test custom configuration"""
        config = StreamingConfig(
            websocket_port=9000,
            max_connections=5000,
            default_mode=StreamingMode.REAL_TIME,
            enable_compression=False
        )

        assert config.websocket_port == 9000
        assert config.max_connections == 5000
        assert config.default_mode == StreamingMode.REAL_TIME
        assert config.enable_compression is False


class TestClientSubscription:
    """Test client subscription functionality"""

    def test_subscription_creation(self):
        """Test client subscription initialization"""
        subscription = ClientSubscription(
            connection_id="test_conn_1",
            mode=StreamingMode.REAL_TIME
        )

        assert subscription.connection_id == "test_conn_1"
        assert subscription.mode == StreamingMode.REAL_TIME
        assert len(subscription.symbols) == 0
        assert subscription.rate_limit == 10.0

    def test_rate_limiting_logic(self):
        """Test rate limiting functionality"""
        subscription = ClientSubscription(
            connection_id="test_conn_1",
            rate_limit=2.0  # 2 updates per second
        )

        current_time = datetime.now()
        symbol = "AAPL"

        # First update should always be allowed
        assert subscription.should_send_update(symbol, current_time)

        # Mark first update
        subscription.last_update[symbol] = current_time

        # Immediate second update should be blocked
        assert not subscription.should_send_update(symbol, current_time)

        # Update after sufficient time should be allowed
        future_time = current_time + timedelta(seconds=0.6)  # > 0.5s (1/2.0)
        assert subscription.should_send_update(symbol, future_time)


class TestRealtimeStreamingService:
    """Test real-time streaming service functionality"""

    @pytest.fixture
    def mock_storage_service(self):
        """Mock storage service"""
        service = AsyncMock()
        service.get_price_history.return_value = []
        return service

    @pytest.fixture
    def mock_cache_manager(self):
        """Mock cache manager"""
        return AsyncMock()

    @pytest.fixture
    def streaming_config(self):
        """Test streaming configuration"""
        return StreamingConfig(
            websocket_port=8766,
            max_connections=100,
            heartbeat_interval=5.0
        )

    @pytest.fixture
    def streaming_service(self, mock_storage_service, mock_cache_manager, streaming_config):
        """Test streaming service instance"""
        return RealtimeStreamingService(
            mock_storage_service,
            mock_cache_manager,
            streaming_config
        )

    def test_service_initialization(self, streaming_service):
        """Test service initialization"""
        assert streaming_service.storage_service is not None
        assert streaming_service.cache_manager is not None
        assert streaming_service.config.websocket_port == 8766
        assert not streaming_service.is_running

    @pytest.mark.asyncio
    @patch('src.streaming.realtime_streaming_service.RealtimeWebSocketServer')
    @patch('src.streaming.realtime_streaming_service.MessageQueueManager')
    async def test_service_initialization_async(self, mock_queue_manager, mock_ws_server, streaming_service):
        """Test async service initialization"""
        # Mock the WebSocket server and queue manager
        mock_ws_instance = AsyncMock()
        mock_ws_server.return_value = mock_ws_instance

        mock_queue_instance = AsyncMock()
        mock_queue_manager.return_value = mock_queue_instance

        await streaming_service.initialize()

        assert streaming_service.is_running
        assert streaming_service.websocket_server == mock_ws_instance
        assert streaming_service.message_queue_manager == mock_queue_instance

        # Cleanup
        await streaming_service.close()

    @pytest.mark.asyncio
    async def test_client_connect_handling(self, streaming_service):
        """Test client connection handling"""
        # Mock WebSocket server
        streaming_service.websocket_server = AsyncMock()
        streaming_service.is_running = True

        connection_id = "test_client_1"
        websocket = AsyncMock()

        await streaming_service._handle_client_connect(connection_id, websocket)

        # Verify subscription was created
        assert connection_id in streaming_service.client_subscriptions
        subscription = streaming_service.client_subscriptions[connection_id]
        assert subscription.connection_id == connection_id
        assert subscription.mode == streaming_service.config.default_mode

    @pytest.mark.asyncio
    async def test_client_disconnect_handling(self, streaming_service):
        """Test client disconnection handling"""
        streaming_service.is_running = True

        # Set up test subscription
        connection_id = "test_client_1"
        subscription = ClientSubscription(connection_id=connection_id)
        subscription.symbols.add("AAPL")
        subscription.symbols.add("GOOGL")

        streaming_service.client_subscriptions[connection_id] = subscription
        streaming_service.symbol_subscribers = {
            "AAPL": {connection_id},
            "GOOGL": {connection_id}
        }

        await streaming_service._handle_client_disconnect(connection_id)

        # Verify cleanup
        assert connection_id not in streaming_service.client_subscriptions
        assert len(streaming_service.symbol_subscribers) == 0

    @pytest.mark.asyncio
    async def test_subscribe_request_handling(self, streaming_service):
        """Test symbol subscription request"""
        streaming_service.websocket_server = AsyncMock()
        streaming_service.is_running = True

        connection_id = "test_client_1"
        streaming_service.client_subscriptions[connection_id] = ClientSubscription(connection_id=connection_id)

        message = {
            "type": "subscribe",
            "symbols": ["AAPL", "GOOGL", "MSFT"]
        }

        await streaming_service._handle_subscribe_request(connection_id, message)

        # Verify subscription
        subscription = streaming_service.client_subscriptions[connection_id]
        assert "AAPL" in subscription.symbols
        assert "GOOGL" in subscription.symbols
        assert "MSFT" in subscription.symbols

        # Verify symbol subscribers
        assert "AAPL" in streaming_service.symbol_subscribers
        assert connection_id in streaming_service.symbol_subscribers["AAPL"]

    @pytest.mark.asyncio
    async def test_unsubscribe_request_handling(self, streaming_service):
        """Test symbol unsubscription request"""
        streaming_service.websocket_server = AsyncMock()
        streaming_service.is_running = True

        connection_id = "test_client_1"
        subscription = ClientSubscription(connection_id=connection_id)
        subscription.symbols.update(["AAPL", "GOOGL", "MSFT"])

        streaming_service.client_subscriptions[connection_id] = subscription
        streaming_service.symbol_subscribers = {
            "AAPL": {connection_id},
            "GOOGL": {connection_id},
            "MSFT": {connection_id}
        }

        message = {
            "type": "unsubscribe",
            "symbols": ["GOOGL", "MSFT"]
        }

        await streaming_service._handle_unsubscribe_request(connection_id, message)

        # Verify unsubscription
        assert "AAPL" in subscription.symbols
        assert "GOOGL" not in subscription.symbols
        assert "MSFT" not in subscription.symbols

        # Verify symbol subscribers cleanup
        assert "AAPL" in streaming_service.symbol_subscribers
        assert "GOOGL" not in streaming_service.symbol_subscribers
        assert "MSFT" not in streaming_service.symbol_subscribers

    @pytest.mark.asyncio
    async def test_configure_request_handling(self, streaming_service):
        """Test client configuration request"""
        streaming_service.websocket_server = AsyncMock()
        streaming_service.is_running = True

        connection_id = "test_client_1"
        streaming_service.client_subscriptions[connection_id] = ClientSubscription(connection_id=connection_id)

        message = {
            "type": "configure",
            "mode": "real_time",
            "rate_limit": 5.0,
            "filters": {"min_quality": 80}
        }

        await streaming_service._handle_configure_request(connection_id, message)

        # Verify configuration
        subscription = streaming_service.client_subscriptions[connection_id]
        assert subscription.mode == StreamingMode.REAL_TIME
        assert subscription.rate_limit == 5.0
        assert subscription.filters["min_quality"] == 80

    @pytest.mark.asyncio
    async def test_historical_request_handling(self, streaming_service):
        """Test historical data request"""
        streaming_service.websocket_server = AsyncMock()
        streaming_service.config.enable_historical_streaming = True
        streaming_service.is_running = True

        # Mock historical data
        historical_data = [
            PriceData(
                symbol="AAPL",
                timestamp=datetime.now() - timedelta(hours=i),
                open_price=Decimal("150.00"),
                high_price=Decimal("152.00"),
                low_price=Decimal("149.00"),
                close_price=Decimal("151.00"),
                volume=1000000,
                source="test"
            )
            for i in range(10)
        ]

        streaming_service.storage_service.get_price_history.return_value = historical_data

        connection_id = "test_client_1"
        message = {
            "type": "historical",
            "symbol": "AAPL",
            "start_time": (datetime.now() - timedelta(days=1)).isoformat(),
            "end_time": datetime.now().isoformat()
        }

        await streaming_service._handle_historical_request(connection_id, message)

        # Verify historical data retrieval was called
        streaming_service.storage_service.get_price_history.assert_called_once()

    @pytest.mark.asyncio
    async def test_price_update_streaming(self, streaming_service):
        """Test price update streaming to clients"""
        streaming_service.websocket_server = AsyncMock()
        streaming_service.is_running = True

        # Set up client subscription
        connection_id = "test_client_1"
        subscription = ClientSubscription(
            connection_id=connection_id,
            mode=StreamingMode.REAL_TIME
        )
        subscription.symbols.add("AAPL")

        streaming_service.client_subscriptions[connection_id] = subscription
        streaming_service.symbol_subscribers["AAPL"] = {connection_id}

        # Create test price data
        price_data = PriceData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open_price=Decimal("150.00"),
            high_price=Decimal("152.00"),
            low_price=Decimal("149.00"),
            close_price=Decimal("151.00"),
            volume=1000000,
            source="test",
            quality_score=95
        )

        await streaming_service.stream_price_update(price_data)

        # Verify message was sent
        streaming_service.websocket_server.send_message_to_connection.assert_called_once()

    @pytest.mark.asyncio
    async def test_data_quality_filtering(self, streaming_service):
        """Test data quality filtering"""
        streaming_service.websocket_server = AsyncMock()
        streaming_service.config.enable_data_validation = True
        streaming_service.config.min_quality_score = 80
        streaming_service.is_running = True

        # Set up client subscription
        connection_id = "test_client_1"
        subscription = ClientSubscription(connection_id=connection_id)
        subscription.symbols.add("AAPL")

        streaming_service.client_subscriptions[connection_id] = subscription
        streaming_service.symbol_subscribers["AAPL"] = {connection_id}

        # Create low-quality price data
        low_quality_data = PriceData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open_price=Decimal("150.00"),
            high_price=Decimal("152.00"),
            low_price=Decimal("149.00"),
            close_price=Decimal("151.00"),
            volume=1000000,
            source="test",
            quality_score=60  # Below minimum threshold
        )

        await streaming_service.stream_price_update(low_quality_data)

        # Verify no message was sent due to quality filtering
        streaming_service.websocket_server.send_message_to_connection.assert_not_called()
        assert streaming_service.metrics.quality_filtered == 1

    @pytest.mark.asyncio
    async def test_client_filters(self, streaming_service):
        """Test client-specific filters"""
        streaming_service.websocket_server = AsyncMock()
        streaming_service.is_running = True

        # Set up client subscription with filters
        connection_id = "test_client_1"
        subscription = ClientSubscription(connection_id=connection_id)
        subscription.symbols.add("AAPL")
        subscription.filters = {"min_quality": 90, "min_volume": 2000000}

        streaming_service.client_subscriptions[connection_id] = subscription
        streaming_service.symbol_subscribers["AAPL"] = {connection_id}

        # Create price data that doesn't meet filter criteria
        price_data = PriceData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open_price=Decimal("150.00"),
            high_price=Decimal("152.00"),
            low_price=Decimal("149.00"),
            close_price=Decimal("151.00"),
            volume=1000000,  # Below minimum volume
            source="test",
            quality_score=95
        )

        await streaming_service.stream_price_update(price_data)

        # Verify no message was sent due to filter
        streaming_service.websocket_server.send_message_to_connection.assert_not_called()

    @pytest.mark.asyncio
    async def test_streaming_modes(self, streaming_service):
        """Test different streaming modes"""
        streaming_service.websocket_server = AsyncMock()
        streaming_service.is_running = True

        # Test real-time mode
        connection_id_rt = "realtime_client"
        subscription_rt = ClientSubscription(
            connection_id=connection_id_rt,
            mode=StreamingMode.REAL_TIME
        )
        subscription_rt.symbols.add("AAPL")

        # Test batch mode
        connection_id_batch = "batch_client"
        subscription_batch = ClientSubscription(
            connection_id=connection_id_batch,
            mode=StreamingMode.BATCH
        )
        subscription_batch.symbols.add("AAPL")

        streaming_service.client_subscriptions[connection_id_rt] = subscription_rt
        streaming_service.client_subscriptions[connection_id_batch] = subscription_batch
        streaming_service.symbol_subscribers["AAPL"] = {connection_id_rt, connection_id_batch}

        price_data = PriceData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open_price=Decimal("150.00"),
            high_price=Decimal("152.00"),
            low_price=Decimal("149.00"),
            close_price=Decimal("151.00"),
            volume=1000000,
            source="test",
            quality_score=95
        )

        await streaming_service.stream_price_update(price_data)

        # Real-time client should get immediate message
        # Batch client should have update queued
        assert connection_id_batch in streaming_service.pending_updates

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, streaming_service):
        """Test metrics tracking functionality"""
        streaming_service.is_running = True

        # Add some test subscriptions
        for i in range(5):
            connection_id = f"client_{i}"
            subscription = ClientSubscription(
                connection_id=connection_id,
                mode=StreamingMode.REAL_TIME if i % 2 == 0 else StreamingMode.BATCH
            )
            if i < 3:  # Only first 3 have active subscriptions
                subscription.symbols.add("AAPL")

            streaming_service.client_subscriptions[connection_id] = subscription

        await streaming_service._update_metrics()

        metrics = streaming_service.metrics
        assert metrics.total_connections == 5
        assert metrics.active_connections == 3  # Only clients with symbols
        assert StreamingMode.REAL_TIME in metrics.connections_by_mode
        assert StreamingMode.BATCH in metrics.connections_by_mode

    @pytest.mark.asyncio
    async def test_subscription_info(self, streaming_service):
        """Test subscription information retrieval"""
        # Set up test subscriptions
        streaming_service.client_subscriptions = {
            "client_1": ClientSubscription("client_1", mode=StreamingMode.REAL_TIME),
            "client_2": ClientSubscription("client_2", mode=StreamingMode.BATCH)
        }

        streaming_service.symbol_subscribers = {
            "AAPL": {"client_1", "client_2"},
            "GOOGL": {"client_1"}
        }

        info = await streaming_service.get_subscription_info()

        assert info["total_clients"] == 2
        assert info["total_symbols"] == 2
        assert info["subscriptions_by_symbol"]["AAPL"] == 2
        assert info["subscriptions_by_symbol"]["GOOGL"] == 1

    @pytest.mark.asyncio
    async def test_price_data_conversion(self, streaming_service):
        """Test price data to dictionary conversion"""
        price_data = PriceData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open_price=Decimal("150.00"),
            high_price=Decimal("152.00"),
            low_price=Decimal("149.00"),
            close_price=Decimal("151.00"),
            volume=1000000,
            source="test",
            quality_score=95
        )

        data_dict = streaming_service._price_data_to_dict(price_data)

        assert data_dict["symbol"] == "AAPL"
        assert data_dict["open"] == 150.0
        assert data_dict["high"] == 152.0
        assert data_dict["low"] == 149.0
        assert data_dict["close"] == 151.0
        assert data_dict["volume"] == 1000000
        assert data_dict["source"] == "test"
        assert data_dict["quality_score"] == 95
        assert "timestamp" in data_dict

    @pytest.mark.asyncio
    async def test_error_handling(self, streaming_service):
        """Test error handling in various scenarios"""
        streaming_service.websocket_server = AsyncMock()
        streaming_service.is_running = True

        # Test invalid message type
        connection_id = "test_client"
        streaming_service.client_subscriptions[connection_id] = ClientSubscription(connection_id)

        invalid_message = {"type": "invalid_type"}
        await streaming_service._handle_client_message(connection_id, invalid_message)

        # Should send error message
        streaming_service.websocket_server.send_message_to_connection.assert_called()
        call_args = streaming_service.websocket_server.send_message_to_connection.call_args[0]
        assert call_args[1]["type"] == "error"

    @pytest.mark.asyncio
    async def test_service_closure(self, streaming_service):
        """Test proper service closure and cleanup"""
        streaming_service.is_running = True
        streaming_service.websocket_server = AsyncMock()
        streaming_service.message_queue_manager = AsyncMock()

        # Add some test data
        streaming_service.client_subscriptions["test"] = ClientSubscription("test")
        streaming_service.symbol_subscribers["AAPL"] = {"test"}

        await streaming_service.close()

        assert not streaming_service.is_running
        assert len(streaming_service.client_subscriptions) == 0
        assert len(streaming_service.symbol_subscribers) == 0
        streaming_service.websocket_server.close.assert_called_once()
        streaming_service.message_queue_manager.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_storage_service, mock_cache_manager, streaming_config):
        """Test streaming service as context manager"""
        with patch('src.streaming.realtime_streaming_service.RealtimeWebSocketServer'):
            with patch('src.streaming.realtime_streaming_service.MessageQueueManager'):
                async with RealtimeStreamingService(mock_storage_service, mock_cache_manager, streaming_config) as service:
                    assert service.is_running

                # Service should be closed after context exit
                assert not service.is_running


class TestStreamingMetrics:
    """Test streaming metrics data structure"""

    def test_metrics_initialization(self):
        """Test metrics initialization"""
        metrics = StreamingMetrics()

        assert metrics.total_connections == 0
        assert metrics.active_connections == 0
        assert metrics.total_messages_sent == 0
        assert metrics.avg_latency_ms == 0.0
        assert isinstance(metrics.last_updated, datetime)

    def test_metrics_updates(self):
        """Test metrics value updates"""
        metrics = StreamingMetrics()

        metrics.total_connections = 100
        metrics.active_connections = 75
        metrics.total_messages_sent = 50000
        metrics.avg_latency_ms = 2.5

        assert metrics.total_connections == 100
        assert metrics.active_connections == 75
        assert metrics.total_messages_sent == 50000
        assert metrics.avg_latency_ms == 2.5


class TestStreamingIntegration:
    """Integration tests for streaming service"""

    @pytest.mark.integration
    async def test_real_websocket_integration(self):
        """Integration test with real WebSocket connections"""
        # This test requires real WebSocket connections
        pytest.skip("Integration test requires real WebSocket setup")

    @pytest.mark.integration
    async def test_streaming_performance(self):
        """Test streaming service performance under load"""
        # This test would measure actual streaming performance
        pytest.skip("Integration test requires performance measurement setup")

    @pytest.mark.integration
    async def test_streaming_resilience(self):
        """Test streaming service resilience to failures"""
        # This test would simulate various failure scenarios
        pytest.skip("Integration test requires failure simulation setup")