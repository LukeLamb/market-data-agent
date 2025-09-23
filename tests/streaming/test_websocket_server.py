"""
Tests for WebSocket Server
Tests WebSocket server functionality, connection management, and message handling
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import json

from src.streaming.websocket_server import (
    RealtimeWebSocketServer,
    WebSocketConnection,
    ConnectionMetrics,
    ServerConfig,
    ConnectionState
)


class TestServerConfig:
    """Test WebSocket server configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = ServerConfig()

        assert config.host == "0.0.0.0"
        assert config.port == 8765
        assert config.max_connections == 10000
        assert config.enable_compression is True
        assert config.ping_interval == 20.0
        assert config.ping_timeout == 10.0

    def test_custom_config(self):
        """Test custom configuration"""
        config = ServerConfig(
            host="127.0.0.1",
            port=9000,
            max_connections=5000,
            enable_compression=False,
            ping_interval=30.0
        )

        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.max_connections == 5000
        assert config.enable_compression is False
        assert config.ping_interval == 30.0


class TestWebSocketConnection:
    """Test WebSocket connection functionality"""

    def test_connection_creation(self):
        """Test WebSocket connection initialization"""
        mock_websocket = AsyncMock()
        mock_websocket.remote_address = ("127.0.0.1", 12345)

        connection = WebSocketConnection(
            connection_id="test_conn_1",
            websocket=mock_websocket,
            client_info={"user_agent": "test_client"}
        )

        assert connection.connection_id == "test_conn_1"
        assert connection.websocket == mock_websocket
        assert connection.state == ConnectionState.CONNECTED
        assert connection.client_info["user_agent"] == "test_client"
        assert isinstance(connection.connected_at, datetime)

    def test_connection_properties(self):
        """Test connection calculated properties"""
        mock_websocket = AsyncMock()
        created_time = datetime.now() - timedelta(minutes=5)

        connection = WebSocketConnection(
            connection_id="test_conn_1",
            websocket=mock_websocket
        )
        connection.connected_at = created_time
        connection.last_activity = datetime.now() - timedelta(seconds=30)
        connection.messages_sent = 100
        connection.messages_received = 50
        connection.total_bytes_sent = 10000
        connection.total_bytes_received = 5000

        assert connection.connection_age_seconds > 290  # ~5 minutes
        assert connection.idle_time_seconds > 25  # ~30 seconds
        assert connection.total_messages == 150
        assert connection.total_bytes == 15000

    @pytest.mark.asyncio
    async def test_send_message(self):
        """Test sending message through connection"""
        mock_websocket = AsyncMock()
        connection = WebSocketConnection("test_conn_1", mock_websocket)

        message = {"type": "test", "data": "hello"}
        await connection.send_message(message)

        # Verify message was sent
        mock_websocket.send.assert_called_once()
        sent_data = mock_websocket.send.call_args[0][0]
        parsed_message = json.loads(sent_data)
        assert parsed_message["type"] == "test"
        assert parsed_message["data"] == "hello"

        # Verify metrics were updated
        assert connection.messages_sent == 1
        assert connection.total_bytes_sent > 0

    @pytest.mark.asyncio
    async def test_send_message_error_handling(self):
        """Test error handling during message sending"""
        mock_websocket = AsyncMock()
        mock_websocket.send.side_effect = Exception("Connection closed")

        connection = WebSocketConnection("test_conn_1", mock_websocket)

        message = {"type": "test"}
        result = await connection.send_message(message)

        assert result is False
        assert connection.error_count == 1

    @pytest.mark.asyncio
    async def test_close_connection(self):
        """Test connection closure"""
        mock_websocket = AsyncMock()
        connection = WebSocketConnection("test_conn_1", mock_websocket)

        await connection.close()

        assert connection.state == ConnectionState.DISCONNECTED
        mock_websocket.close.assert_called_once()


class TestConnectionMetrics:
    """Test connection metrics tracking"""

    def test_metrics_initialization(self):
        """Test metrics initialization"""
        metrics = ConnectionMetrics()

        assert metrics.total_connections == 0
        assert metrics.active_connections == 0
        assert metrics.max_concurrent_connections == 0
        assert metrics.total_messages_sent == 0
        assert metrics.avg_response_time_ms == 0.0

    def test_metrics_updates(self):
        """Test metrics value updates"""
        metrics = ConnectionMetrics()

        metrics.total_connections = 1000
        metrics.active_connections = 250
        metrics.max_concurrent_connections = 500
        metrics.total_messages_sent = 50000
        metrics.avg_response_time_ms = 15.5

        assert metrics.total_connections == 1000
        assert metrics.active_connections == 250
        assert metrics.max_concurrent_connections == 500
        assert metrics.total_messages_sent == 50000
        assert metrics.avg_response_time_ms == 15.5


class TestRealtimeWebSocketServer:
    """Test real-time WebSocket server functionality"""

    @pytest.fixture
    def server_config(self):
        """Test server configuration"""
        return ServerConfig(
            host="127.0.0.1",
            port=8766,
            max_connections=100
        )

    @pytest.fixture
    def websocket_server(self, server_config):
        """Test WebSocket server instance"""
        return RealtimeWebSocketServer(
            host=server_config.host,
            port=server_config.port,
            max_connections=server_config.max_connections
        )

    def test_server_initialization(self, websocket_server):
        """Test server initialization"""
        assert websocket_server.config.host == "127.0.0.1"
        assert websocket_server.config.port == 8766
        assert websocket_server.config.max_connections == 100
        assert not websocket_server.is_running
        assert len(websocket_server.connections) == 0

    @pytest.mark.asyncio
    @patch('websockets.serve')
    async def test_server_start(self, mock_serve, websocket_server):
        """Test server startup"""
        mock_server = AsyncMock()
        mock_serve.return_value = mock_server

        await websocket_server.initialize()

        assert websocket_server.is_running
        assert websocket_server.server == mock_server
        mock_serve.assert_called_once()

        # Cleanup
        await websocket_server.close()

    @pytest.mark.asyncio
    async def test_connection_handling(self, websocket_server):
        """Test connection handling"""
        websocket_server.is_running = True

        # Mock WebSocket connection
        mock_websocket = AsyncMock()
        mock_websocket.remote_address = ("127.0.0.1", 12345)
        mock_websocket.request_headers = {"User-Agent": "test_client"}

        # Simulate connection handling
        connection_id = await websocket_server._handle_new_connection(mock_websocket)

        assert connection_id is not None
        assert connection_id in websocket_server.connections
        assert websocket_server.metrics.total_connections == 1
        assert websocket_server.metrics.active_connections == 1

    @pytest.mark.asyncio
    async def test_connection_limit_enforcement(self, websocket_server):
        """Test connection limit enforcement"""
        websocket_server.config.max_connections = 2
        websocket_server.is_running = True

        # Add two connections (at limit)
        for i in range(2):
            mock_ws = AsyncMock()
            mock_ws.remote_address = ("127.0.0.1", 12345 + i)
            connection_id = await websocket_server._handle_new_connection(mock_ws)
            assert connection_id is not None

        # Try to add third connection (should be rejected)
        mock_ws_rejected = AsyncMock()
        mock_ws_rejected.remote_address = ("127.0.0.1", 12347)
        connection_id = await websocket_server._handle_new_connection(mock_ws_rejected)

        assert connection_id is None
        assert len(websocket_server.connections) == 2

    @pytest.mark.asyncio
    async def test_message_sending_to_connection(self, websocket_server):
        """Test sending message to specific connection"""
        websocket_server.is_running = True

        # Set up connection
        mock_websocket = AsyncMock()
        connection_id = await websocket_server._handle_new_connection(mock_websocket)

        message = {"type": "test", "data": "hello"}
        result = await websocket_server.send_message_to_connection(connection_id, message)

        assert result is True
        mock_websocket.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_message_sending_to_nonexistent_connection(self, websocket_server):
        """Test sending message to non-existent connection"""
        message = {"type": "test"}
        result = await websocket_server.send_message_to_connection("nonexistent", message)

        assert result is False

    @pytest.mark.asyncio
    async def test_broadcast_message(self, websocket_server):
        """Test broadcasting message to all connections"""
        websocket_server.is_running = True

        # Set up multiple connections
        mock_websockets = []
        connection_ids = []

        for i in range(3):
            mock_ws = AsyncMock()
            mock_ws.remote_address = ("127.0.0.1", 12345 + i)
            mock_websockets.append(mock_ws)

            connection_id = await websocket_server._handle_new_connection(mock_ws)
            connection_ids.append(connection_id)

        message = {"type": "broadcast", "data": "hello_all"}
        sent_count = await websocket_server.broadcast_message(message)

        assert sent_count == 3
        for mock_ws in mock_websockets:
            mock_ws.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_cleanup(self, websocket_server):
        """Test connection cleanup on disconnect"""
        websocket_server.is_running = True

        # Set up connection
        mock_websocket = AsyncMock()
        connection_id = await websocket_server._handle_new_connection(mock_websocket)

        assert len(websocket_server.connections) == 1
        assert websocket_server.metrics.active_connections == 1

        # Simulate disconnection
        await websocket_server._handle_connection_close(connection_id)

        assert len(websocket_server.connections) == 0
        assert websocket_server.metrics.active_connections == 0

    @pytest.mark.asyncio
    async def test_message_rate_limiting(self, websocket_server):
        """Test message rate limiting"""
        websocket_server.config.enable_rate_limiting = True
        websocket_server.config.rate_limit_per_second = 2
        websocket_server.is_running = True

        # Set up connection
        mock_websocket = AsyncMock()
        connection_id = await websocket_server._handle_new_connection(mock_websocket)

        # Send messages rapidly
        message = {"type": "test"}
        results = []

        for _ in range(5):
            result = await websocket_server.send_message_to_connection(connection_id, message)
            results.append(result)

        # Some messages should be rate limited
        successful_sends = sum(1 for r in results if r)
        assert successful_sends <= 2  # Rate limit should kick in

    @pytest.mark.asyncio
    async def test_connection_authentication(self, websocket_server):
        """Test connection authentication"""
        websocket_server.config.require_authentication = True
        websocket_server.is_running = True

        # Mock authentication handler
        async def mock_auth_handler(websocket, path):
            headers = getattr(websocket, 'request_headers', {})
            return headers.get('Authorization') == 'Bearer valid_token'

        websocket_server.auth_handler = mock_auth_handler

        # Test with valid authentication
        mock_websocket_valid = AsyncMock()
        mock_websocket_valid.request_headers = {'Authorization': 'Bearer valid_token'}
        mock_websocket_valid.remote_address = ("127.0.0.1", 12345)

        connection_id = await websocket_server._handle_new_connection(mock_websocket_valid)
        assert connection_id is not None

        # Test with invalid authentication
        mock_websocket_invalid = AsyncMock()
        mock_websocket_invalid.request_headers = {'Authorization': 'Bearer invalid_token'}
        mock_websocket_invalid.remote_address = ("127.0.0.1", 12346)

        connection_id = await websocket_server._handle_new_connection(mock_websocket_invalid)
        assert connection_id is None

    @pytest.mark.asyncio
    async def test_connection_metrics_tracking(self, websocket_server):
        """Test connection metrics tracking"""
        websocket_server.is_running = True

        # Add some connections
        for i in range(5):
            mock_ws = AsyncMock()
            mock_ws.remote_address = ("127.0.0.1", 12345 + i)
            await websocket_server._handle_new_connection(mock_ws)

        # Remove some connections
        connections_to_remove = list(websocket_server.connections.keys())[:2]
        for connection_id in connections_to_remove:
            await websocket_server._handle_connection_close(connection_id)

        metrics = await websocket_server.get_connection_metrics()

        assert metrics.total_connections == 5
        assert metrics.active_connections == 3
        assert metrics.max_concurrent_connections == 5

    @pytest.mark.asyncio
    async def test_connection_health_monitoring(self, websocket_server):
        """Test connection health monitoring"""
        websocket_server.is_running = True

        # Set up connection with health monitoring
        mock_websocket = AsyncMock()
        connection_id = await websocket_server._handle_new_connection(mock_websocket)

        connection = websocket_server.connections[connection_id]
        connection.last_ping = datetime.now() - timedelta(seconds=60)  # Old ping

        # Run health check
        await websocket_server._health_check_connections()

        # Connection should be marked as unhealthy or removed
        # (depending on implementation details)

    @pytest.mark.asyncio
    async def test_server_shutdown(self, websocket_server):
        """Test proper server shutdown"""
        websocket_server.is_running = True
        websocket_server.server = AsyncMock()

        # Add some connections
        for i in range(3):
            mock_ws = AsyncMock()
            mock_ws.remote_address = ("127.0.0.1", 12345 + i)
            await websocket_server._handle_new_connection(mock_ws)

        await websocket_server.close()

        assert not websocket_server.is_running
        assert len(websocket_server.connections) == 0

    @pytest.mark.asyncio
    async def test_error_handling_in_message_handling(self, websocket_server):
        """Test error handling during message processing"""
        websocket_server.is_running = True

        # Set up connection
        mock_websocket = AsyncMock()
        mock_websocket.send.side_effect = Exception("Connection error")
        connection_id = await websocket_server._handle_new_connection(mock_websocket)

        message = {"type": "test"}
        result = await websocket_server.send_message_to_connection(connection_id, message)

        assert result is False
        # Connection should be marked with error or removed

    @pytest.mark.asyncio
    async def test_websocket_protocol_handling(self, websocket_server):
        """Test WebSocket protocol specific handling"""
        websocket_server.is_running = True

        # Mock WebSocket with protocol handling
        mock_websocket = AsyncMock()
        mock_websocket.remote_address = ("127.0.0.1", 12345)

        # Test ping/pong handling
        connection_id = await websocket_server._handle_new_connection(mock_websocket)
        connection = websocket_server.connections[connection_id]

        # Simulate ping
        await websocket_server._send_ping(connection_id)
        mock_websocket.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_statistics(self, websocket_server):
        """Test connection statistics calculation"""
        websocket_server.is_running = True

        # Set up connections with various states
        for i in range(5):
            mock_ws = AsyncMock()
            mock_ws.remote_address = ("127.0.0.1", 12345 + i)
            connection_id = await websocket_server._handle_new_connection(mock_ws)

            # Set different activity levels
            connection = websocket_server.connections[connection_id]
            connection.messages_sent = i * 10
            connection.messages_received = i * 5
            connection.total_bytes_sent = i * 1000
            connection.last_activity = datetime.now() - timedelta(seconds=i * 10)

        stats = await websocket_server.get_connection_statistics()

        assert "total_connections" in stats
        assert "active_connections" in stats
        assert "total_messages_sent" in stats
        assert "total_bytes_transferred" in stats


class TestWebSocketIntegration:
    """Integration tests for WebSocket server"""

    @pytest.mark.integration
    async def test_real_websocket_connections(self):
        """Integration test with real WebSocket connections"""
        # This test requires real WebSocket client connections
        pytest.skip("Integration test requires real WebSocket client setup")

    @pytest.mark.integration
    async def test_websocket_performance_under_load(self):
        """Test WebSocket server performance under load"""
        # This test would measure actual performance metrics
        pytest.skip("Integration test requires performance measurement setup")

    @pytest.mark.integration
    async def test_websocket_connection_resilience(self):
        """Test WebSocket server resilience to connection failures"""
        # This test would simulate various connection failure scenarios
        pytest.skip("Integration test requires failure simulation setup")