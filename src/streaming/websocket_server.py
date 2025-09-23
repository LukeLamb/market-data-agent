"""
Real-Time WebSocket Server for Market Data Agent
Provides high-performance real-time data streaming with connection management
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import weakref
from contextlib import asynccontextmanager
import uuid

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from ..data_sources.base import PriceData, CurrentPrice
from ..storage.hybrid_storage_service import HybridStorageService

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """WebSocket message types"""
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    PRICE_UPDATE = "price_update"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    AUTHENTICATION = "auth"
    SYSTEM_STATUS = "system_status"


class ConnectionState(Enum):
    """WebSocket connection states"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"


@dataclass
class WebSocketConfig:
    """WebSocket server configuration"""
    host: str = "localhost"
    port: int = 8765
    max_connections: int = 10000
    heartbeat_interval: float = 30.0  # seconds
    connection_timeout: float = 60.0  # seconds
    max_message_size: int = 1024 * 1024  # 1MB
    compression: Optional[str] = "deflate"

    # Rate limiting
    max_messages_per_second: int = 100
    max_subscriptions_per_connection: int = 1000

    # Authentication
    require_authentication: bool = False
    auth_timeout: float = 10.0  # seconds

    # Performance tuning
    ping_interval: float = 20.0
    ping_timeout: float = 10.0
    close_timeout: float = 10.0


@dataclass
class ConnectionMetrics:
    """Metrics for individual WebSocket connections"""
    connection_id: str
    client_ip: str
    connected_at: datetime
    last_activity: datetime
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    subscriptions: Set[str] = field(default_factory=set)
    state: ConnectionState = ConnectionState.CONNECTING
    user_agent: Optional[str] = None

    @property
    def connection_duration(self) -> float:
        return (datetime.now() - self.connected_at).total_seconds()

    @property
    def idle_time(self) -> float:
        return (datetime.now() - self.last_activity).total_seconds()


@dataclass
class StreamingStats:
    """Real-time streaming statistics"""
    total_connections: int = 0
    active_connections: int = 0
    total_subscriptions: int = 0
    messages_per_second: float = 0.0
    bytes_per_second: float = 0.0
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0
    uptime_seconds: float = 0.0


class WebSocketConnection:
    """Enhanced WebSocket connection wrapper with metrics and state management"""

    def __init__(self, websocket, connection_id: str, client_ip: str):
        self.websocket = websocket
        self.connection_id = connection_id
        self.client_ip = client_ip
        self.metrics = ConnectionMetrics(
            connection_id=connection_id,
            client_ip=client_ip,
            connected_at=datetime.now(),
            last_activity=datetime.now()
        )
        self.message_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.rate_limiter = RateLimiter(max_messages_per_second=100)
        self.is_alive = True

    async def send_message(self, message: Dict[str, Any]) -> bool:
        """Send message to client with error handling"""
        try:
            if not self.is_alive:
                return False

            message_json = json.dumps(message)
            await self.websocket.send(message_json)

            # Update metrics
            self.metrics.messages_sent += 1
            self.metrics.bytes_sent += len(message_json.encode('utf-8'))
            self.metrics.last_activity = datetime.now()

            return True

        except (ConnectionClosed, WebSocketException) as e:
            logger.debug(f"Connection {self.connection_id} closed during send: {e}")
            self.is_alive = False
            return False
        except Exception as e:
            logger.error(f"Error sending message to {self.connection_id}: {e}")
            return False

    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive and parse message from client"""
        try:
            if not self.is_alive:
                return None

            message_raw = await self.websocket.recv()

            # Update metrics
            self.metrics.messages_received += 1
            self.metrics.bytes_received += len(message_raw.encode('utf-8'))
            self.metrics.last_activity = datetime.now()

            # Parse JSON message
            message = json.loads(message_raw)
            return message

        except (ConnectionClosed, WebSocketException):
            logger.debug(f"Connection {self.connection_id} closed during receive")
            self.is_alive = False
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON from {self.connection_id}: {e}")
            await self.send_error("Invalid JSON format")
            return None
        except Exception as e:
            logger.error(f"Error receiving message from {self.connection_id}: {e}")
            return None

    async def send_error(self, error_message: str) -> None:
        """Send error message to client"""
        error_msg = {
            "type": MessageType.ERROR.value,
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        }
        await self.send_message(error_msg)

    def add_subscription(self, symbol: str) -> bool:
        """Add symbol subscription"""
        if len(self.metrics.subscriptions) >= 1000:  # Max subscriptions limit
            return False
        self.metrics.subscriptions.add(symbol.upper())
        return True

    def remove_subscription(self, symbol: str) -> bool:
        """Remove symbol subscription"""
        symbol_upper = symbol.upper()
        if symbol_upper in self.metrics.subscriptions:
            self.metrics.subscriptions.remove(symbol_upper)
            return True
        return False

    async def close(self):
        """Close connection gracefully"""
        self.is_alive = False
        try:
            await self.websocket.close()
        except Exception as e:
            logger.debug(f"Error closing connection {self.connection_id}: {e}")


class RateLimiter:
    """Simple rate limiter for WebSocket connections"""

    def __init__(self, max_messages_per_second: int):
        self.max_messages_per_second = max_messages_per_second
        self.message_timestamps: List[float] = []

    def is_allowed(self) -> bool:
        """Check if message is allowed under rate limit"""
        now = time.time()

        # Remove old timestamps (older than 1 second)
        cutoff = now - 1.0
        self.message_timestamps = [ts for ts in self.message_timestamps if ts > cutoff]

        # Check if under limit
        if len(self.message_timestamps) < self.max_messages_per_second:
            self.message_timestamps.append(now)
            return True

        return False


class RealtimeWebSocketServer:
    """High-performance real-time WebSocket server for market data streaming"""

    def __init__(
        self,
        config: WebSocketConfig = None,
        storage_service: HybridStorageService = None
    ):
        self.config = config or WebSocketConfig()
        self.storage_service = storage_service

        # Connection management
        self.connections: Dict[str, WebSocketConnection] = {}
        self.symbol_subscriptions: Dict[str, Set[str]] = {}  # symbol -> set of connection_ids

        # Server state
        self.server = None
        self.is_running = False
        self.start_time = datetime.now()

        # Background tasks
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.stats_task: Optional[asyncio.Task] = None

        # Performance tracking
        self.stats = StreamingStats()
        self.message_count_history: List[Tuple[datetime, int]] = []

        # Event handlers
        self.message_handlers: Dict[str, Callable] = {
            MessageType.SUBSCRIBE.value: self._handle_subscribe,
            MessageType.UNSUBSCRIBE.value: self._handle_unsubscribe,
            MessageType.HEARTBEAT.value: self._handle_heartbeat,
            MessageType.AUTHENTICATION.value: self._handle_authentication,
        }

    async def start(self) -> None:
        """Start the WebSocket server"""
        try:
            logger.info(f"Starting WebSocket server on {self.config.host}:{self.config.port}")

            # Create WebSocket server
            self.server = await websockets.serve(
                self._handle_connection,
                self.config.host,
                self.config.port,
                ping_interval=self.config.ping_interval,
                ping_timeout=self.config.ping_timeout,
                close_timeout=self.config.close_timeout,
                max_size=self.config.max_message_size,
                compression=self.config.compression
            )

            self.is_running = True
            self.start_time = datetime.now()

            # Start background tasks
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            self.stats_task = asyncio.create_task(self._stats_loop())

            logger.info("WebSocket server started successfully")

        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            raise

    async def stop(self) -> None:
        """Stop the WebSocket server"""
        try:
            logger.info("Stopping WebSocket server...")

            self.is_running = False

            # Cancel background tasks
            for task in [self.heartbeat_task, self.cleanup_task, self.stats_task]:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            # Close all connections
            close_tasks = []
            for connection in list(self.connections.values()):
                close_tasks.append(connection.close())

            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)

            # Close server
            if self.server:
                self.server.close()
                await self.server.wait_closed()

            self.connections.clear()
            self.symbol_subscriptions.clear()

            logger.info("WebSocket server stopped")

        except Exception as e:
            logger.error(f"Error stopping WebSocket server: {e}")

    async def _handle_connection(self, websocket, path: str) -> None:
        """Handle new WebSocket connection"""
        connection_id = str(uuid.uuid4())
        client_ip = websocket.remote_address[0] if websocket.remote_address else "unknown"

        logger.debug(f"New connection: {connection_id} from {client_ip}")

        # Check connection limit
        if len(self.connections) >= self.config.max_connections:
            logger.warning(f"Connection limit reached, rejecting {connection_id}")
            await websocket.close(code=1013, reason="Server overloaded")
            return

        connection = WebSocketConnection(websocket, connection_id, client_ip)
        self.connections[connection_id] = connection

        try:
            connection.metrics.state = ConnectionState.CONNECTED

            # Send welcome message
            welcome_msg = {
                "type": "welcome",
                "connection_id": connection_id,
                "server_time": datetime.now().isoformat(),
                "heartbeat_interval": self.config.heartbeat_interval
            }
            await connection.send_message(welcome_msg)

            # Handle messages
            await self._message_loop(connection)

        except Exception as e:
            logger.error(f"Error handling connection {connection_id}: {e}")
        finally:
            await self._cleanup_connection(connection_id)

    async def _message_loop(self, connection: WebSocketConnection) -> None:
        """Main message handling loop for a connection"""
        try:
            while connection.is_alive and self.is_running:
                message = await connection.receive_message()

                if message is None:
                    break

                # Rate limiting
                if not connection.rate_limiter.is_allowed():
                    await connection.send_error("Rate limit exceeded")
                    continue

                # Handle message
                await self._handle_message(connection, message)

        except Exception as e:
            logger.error(f"Error in message loop for {connection.connection_id}: {e}")

    async def _handle_message(self, connection: WebSocketConnection, message: Dict[str, Any]) -> None:
        """Handle incoming message from client"""
        try:
            message_type = message.get("type")

            if message_type in self.message_handlers:
                await self.message_handlers[message_type](connection, message)
            else:
                await connection.send_error(f"Unknown message type: {message_type}")

        except Exception as e:
            logger.error(f"Error handling message from {connection.connection_id}: {e}")
            await connection.send_error("Internal server error")

    async def _handle_subscribe(self, connection: WebSocketConnection, message: Dict[str, Any]) -> None:
        """Handle symbol subscription"""
        symbols = message.get("symbols", [])

        if not isinstance(symbols, list):
            await connection.send_error("Invalid symbols format")
            return

        successful_subscriptions = []
        failed_subscriptions = []

        for symbol in symbols:
            if isinstance(symbol, str) and symbol:
                symbol_upper = symbol.upper()

                if connection.add_subscription(symbol_upper):
                    # Add to global subscriptions
                    if symbol_upper not in self.symbol_subscriptions:
                        self.symbol_subscriptions[symbol_upper] = set()
                    self.symbol_subscriptions[symbol_upper].add(connection.connection_id)

                    successful_subscriptions.append(symbol_upper)
                else:
                    failed_subscriptions.append(symbol_upper)

        # Send confirmation
        response = {
            "type": "subscription_response",
            "successful": successful_subscriptions,
            "failed": failed_subscriptions,
            "timestamp": datetime.now().isoformat()
        }
        await connection.send_message(response)

        logger.debug(f"Connection {connection.connection_id} subscribed to {len(successful_subscriptions)} symbols")

    async def _handle_unsubscribe(self, connection: WebSocketConnection, message: Dict[str, Any]) -> None:
        """Handle symbol unsubscription"""
        symbols = message.get("symbols", [])

        if not isinstance(symbols, list):
            await connection.send_error("Invalid symbols format")
            return

        unsubscribed = []

        for symbol in symbols:
            if isinstance(symbol, str) and symbol:
                symbol_upper = symbol.upper()

                if connection.remove_subscription(symbol_upper):
                    # Remove from global subscriptions
                    if symbol_upper in self.symbol_subscriptions:
                        self.symbol_subscriptions[symbol_upper].discard(connection.connection_id)
                        if not self.symbol_subscriptions[symbol_upper]:
                            del self.symbol_subscriptions[symbol_upper]

                    unsubscribed.append(symbol_upper)

        # Send confirmation
        response = {
            "type": "unsubscription_response",
            "unsubscribed": unsubscribed,
            "timestamp": datetime.now().isoformat()
        }
        await connection.send_message(response)

    async def _handle_heartbeat(self, connection: WebSocketConnection, message: Dict[str, Any]) -> None:
        """Handle heartbeat message"""
        response = {
            "type": MessageType.HEARTBEAT.value,
            "timestamp": datetime.now().isoformat()
        }
        await connection.send_message(response)

    async def _handle_authentication(self, connection: WebSocketConnection, message: Dict[str, Any]) -> None:
        """Handle authentication (placeholder implementation)"""
        # In a real implementation, validate credentials
        token = message.get("token")

        if self.config.require_authentication:
            # Simplified auth check
            if token and len(token) > 10:  # Basic validation
                connection.metrics.state = ConnectionState.AUTHENTICATED
                response = {"type": "auth_response", "status": "authenticated"}
            else:
                response = {"type": "auth_response", "status": "failed"}
        else:
            response = {"type": "auth_response", "status": "not_required"}

        await connection.send_message(response)

    async def broadcast_price_update(self, price_data: PriceData) -> int:
        """Broadcast price update to subscribed connections"""
        symbol = price_data.symbol.upper()

        if symbol not in self.symbol_subscriptions:
            return 0

        message = {
            "type": MessageType.PRICE_UPDATE.value,
            "symbol": symbol,
            "data": {
                "open": price_data.open_price,
                "high": price_data.high_price,
                "low": price_data.low_price,
                "close": price_data.close_price,
                "volume": price_data.volume,
                "timestamp": price_data.timestamp.isoformat(),
                "source": price_data.source,
                "quality_score": price_data.quality_score
            },
            "server_timestamp": datetime.now().isoformat()
        }

        # Send to all subscribed connections
        connection_ids = list(self.symbol_subscriptions[symbol])
        successful_sends = 0

        send_tasks = []
        for connection_id in connection_ids:
            if connection_id in self.connections:
                connection = self.connections[connection_id]
                send_tasks.append(connection.send_message(message))

        if send_tasks:
            results = await asyncio.gather(*send_tasks, return_exceptions=True)
            successful_sends = sum(1 for result in results if result is True)

        return successful_sends

    async def _cleanup_connection(self, connection_id: str) -> None:
        """Clean up disconnected connection"""
        if connection_id not in self.connections:
            return

        connection = self.connections[connection_id]

        # Remove from all symbol subscriptions
        for symbol in list(connection.metrics.subscriptions):
            if symbol in self.symbol_subscriptions:
                self.symbol_subscriptions[symbol].discard(connection_id)
                if not self.symbol_subscriptions[symbol]:
                    del self.symbol_subscriptions[symbol]

        # Remove connection
        del self.connections[connection_id]

        logger.debug(f"Cleaned up connection {connection_id}")

    async def _heartbeat_loop(self) -> None:
        """Background heartbeat loop"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)

                # Send heartbeat to all connections
                heartbeat_msg = {
                    "type": MessageType.HEARTBEAT.value,
                    "timestamp": datetime.now().isoformat()
                }

                dead_connections = []
                for connection_id, connection in self.connections.items():
                    if connection.is_alive:
                        success = await connection.send_message(heartbeat_msg)
                        if not success:
                            dead_connections.append(connection_id)
                    else:
                        dead_connections.append(connection_id)

                # Clean up dead connections
                for connection_id in dead_connections:
                    await self._cleanup_connection(connection_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop for idle connections"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Check every minute

                timeout_connections = []
                for connection_id, connection in self.connections.items():
                    if connection.metrics.idle_time > self.config.connection_timeout:
                        timeout_connections.append(connection_id)

                # Close timed out connections
                for connection_id in timeout_connections:
                    logger.debug(f"Closing idle connection {connection_id}")
                    if connection_id in self.connections:
                        await self.connections[connection_id].close()
                        await self._cleanup_connection(connection_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def _stats_loop(self) -> None:
        """Background statistics calculation loop"""
        while self.is_running:
            try:
                await asyncio.sleep(10)  # Update stats every 10 seconds
                await self._update_stats()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stats loop: {e}")

    async def _update_stats(self) -> None:
        """Update streaming statistics"""
        now = datetime.now()

        # Basic counts
        self.stats.total_connections = len(self.connections)
        self.stats.active_connections = sum(
            1 for conn in self.connections.values()
            if conn.metrics.idle_time < 300  # Active in last 5 minutes
        )
        self.stats.total_subscriptions = sum(
            len(conn.metrics.subscriptions)
            for conn in self.connections.values()
        )

        # Performance metrics
        total_messages = sum(conn.metrics.messages_sent for conn in self.connections.values())
        total_bytes = sum(conn.metrics.bytes_sent for conn in self.connections.values())

        self.stats.uptime_seconds = (now - self.start_time).total_seconds()

        if self.stats.uptime_seconds > 0:
            self.stats.messages_per_second = total_messages / self.stats.uptime_seconds
            self.stats.bytes_per_second = total_bytes / self.stats.uptime_seconds

    def get_connection_details(self) -> List[Dict[str, Any]]:
        """Get detailed information about all connections"""
        details = []

        for connection in self.connections.values():
            details.append({
                "connection_id": connection.connection_id,
                "client_ip": connection.client_ip,
                "connected_at": connection.metrics.connected_at.isoformat(),
                "connection_duration": connection.metrics.connection_duration,
                "idle_time": connection.metrics.idle_time,
                "state": connection.metrics.state.value,
                "messages_sent": connection.metrics.messages_sent,
                "messages_received": connection.metrics.messages_received,
                "bytes_sent": connection.metrics.bytes_sent,
                "bytes_received": connection.metrics.bytes_received,
                "subscriptions": list(connection.metrics.subscriptions),
                "subscription_count": len(connection.metrics.subscriptions)
            })

        return details

    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get comprehensive streaming statistics"""
        return {
            "total_connections": self.stats.total_connections,
            "active_connections": self.stats.active_connections,
            "total_subscriptions": self.stats.total_subscriptions,
            "messages_per_second": self.stats.messages_per_second,
            "bytes_per_second": self.stats.bytes_per_second,
            "avg_latency_ms": self.stats.avg_latency_ms,
            "error_rate": self.stats.error_rate,
            "uptime_seconds": self.stats.uptime_seconds,
            "symbol_subscriptions": len(self.symbol_subscriptions),
            "most_popular_symbols": self._get_popular_symbols(),
            "server_info": {
                "host": self.config.host,
                "port": self.config.port,
                "max_connections": self.config.max_connections,
                "is_running": self.is_running
            }
        }

    def _get_popular_symbols(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most popular symbols by subscriber count"""
        symbol_counts = [
            {"symbol": symbol, "subscribers": len(connection_ids)}
            for symbol, connection_ids in self.symbol_subscriptions.items()
        ]

        return sorted(symbol_counts, key=lambda x: x["subscribers"], reverse=True)[:limit]

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


# Global WebSocket server instance
websocket_server = None

async def get_websocket_server(
    config: WebSocketConfig = None,
    storage_service: HybridStorageService = None
) -> RealtimeWebSocketServer:
    """Get or create global WebSocket server instance"""
    global websocket_server
    if websocket_server is None:
        websocket_server = RealtimeWebSocketServer(config, storage_service)
    return websocket_server