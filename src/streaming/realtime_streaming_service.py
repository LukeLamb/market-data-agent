"""
Real-Time Streaming Service for Market Data Agent
Integrates WebSocket server with message queue for real-time data distribution
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from contextlib import asynccontextmanager

from .websocket_server import RealtimeWebSocketServer, ConnectionMetrics as WSConnectionMetrics
from .message_queue import MessageQueueManager, MessagePriority, QueueMetrics
from ..data_sources.base import PriceData
from ..storage.hybrid_storage_service import HybridStorageService
from ..caching.redis_cache_manager import RedisCacheManager

logger = logging.getLogger(__name__)


class StreamingMode(Enum):
    """Streaming mode configuration"""
    REAL_TIME = "real_time"           # Immediate streaming with minimal latency
    THROTTLED = "throttled"           # Rate-limited streaming for bandwidth management
    BATCH = "batch"                   # Batched updates for efficiency
    SMART = "smart"                   # Adaptive streaming based on client and data characteristics


@dataclass
class StreamingConfig:
    """Configuration for real-time streaming service"""
    # WebSocket server settings
    websocket_host: str = "0.0.0.0"
    websocket_port: int = 8765
    max_connections: int = 10000

    # Message queue settings
    queue_max_size: int = 100000
    queue_batch_size: int = 100
    queue_timeout: float = 1.0

    # Streaming behavior
    default_mode: StreamingMode = StreamingMode.SMART
    max_update_frequency: float = 10.0  # Max updates per second per symbol
    batch_interval: float = 0.1  # Batch processing interval in seconds

    # Data management
    symbol_buffer_size: int = 1000
    enable_historical_streaming: bool = True
    enable_aggregated_streaming: bool = True

    # Performance tuning
    enable_compression: bool = True
    enable_heartbeat: bool = True
    heartbeat_interval: float = 30.0

    # Quality control
    min_quality_score: int = 70  # Minimum quality score for streaming
    enable_data_validation: bool = True


@dataclass
class StreamingMetrics:
    """Metrics for the streaming service"""
    # Connection metrics
    total_connections: int = 0
    active_connections: int = 0
    connections_by_mode: Dict[StreamingMode, int] = field(default_factory=dict)

    # Data metrics
    total_messages_sent: int = 0
    messages_per_second: float = 0.0
    bytes_sent: int = 0
    compression_ratio: float = 0.0

    # Performance metrics
    avg_latency_ms: float = 0.0
    avg_throughput_mbps: float = 0.0
    queue_utilization: float = 0.0

    # Quality metrics
    messages_dropped: int = 0
    quality_filtered: int = 0
    error_rate: float = 0.0

    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ClientSubscription:
    """Client subscription configuration"""
    connection_id: str
    symbols: Set[str] = field(default_factory=set)
    mode: StreamingMode = StreamingMode.SMART
    filters: Dict[str, Any] = field(default_factory=dict)
    last_update: Dict[str, datetime] = field(default_factory=dict)
    rate_limit: float = 10.0  # Max updates per second

    def should_send_update(self, symbol: str, current_time: datetime) -> bool:
        """Check if update should be sent based on rate limiting"""
        if symbol not in self.last_update:
            return True

        time_since_last = (current_time - self.last_update[symbol]).total_seconds()
        min_interval = 1.0 / self.rate_limit
        return time_since_last >= min_interval


class RealtimeStreamingService:
    """Real-time streaming service for market data distribution"""

    def __init__(
        self,
        storage_service: HybridStorageService,
        cache_manager: RedisCacheManager,
        config: StreamingConfig = None
    ):
        self.storage_service = storage_service
        self.cache_manager = cache_manager
        self.config = config or StreamingConfig()

        # Core components
        self.websocket_server: Optional[RealtimeWebSocketServer] = None
        self.message_queue_manager: Optional[MessageQueueManager] = None

        # Client management
        self.client_subscriptions: Dict[str, ClientSubscription] = {}
        self.symbol_subscribers: Dict[str, Set[str]] = {}  # symbol -> connection_ids

        # Data management
        self.symbol_buffers: Dict[str, List[PriceData]] = {}
        self.pending_updates: Dict[str, List[Dict[str, Any]]] = {}

        # Performance tracking
        self.metrics = StreamingMetrics()
        self.start_time = datetime.now()

        # Background tasks
        self.batch_processor_task: Optional[asyncio.Task] = None
        self.metrics_updater_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None

        # Event handlers
        self.data_handlers: List[Callable[[PriceData], None]] = []

        self.is_running = False

    async def initialize(self) -> None:
        """Initialize the streaming service"""
        try:
            # Initialize WebSocket server
            self.websocket_server = RealtimeWebSocketServer(
                host=self.config.websocket_host,
                port=self.config.websocket_port,
                max_connections=self.config.max_connections
            )

            # Set up WebSocket event handlers
            self.websocket_server.on_connect = self._handle_client_connect
            self.websocket_server.on_disconnect = self._handle_client_disconnect
            self.websocket_server.on_message = self._handle_client_message

            await self.websocket_server.initialize()

            # Initialize message queue manager
            self.message_queue_manager = MessageQueueManager()
            await self.message_queue_manager.initialize()

            # Start background tasks
            self.batch_processor_task = asyncio.create_task(self._batch_processor_loop())
            self.metrics_updater_task = asyncio.create_task(self._metrics_updater_loop())

            if self.config.enable_heartbeat:
                self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            self.is_running = True
            logger.info(f"Real-time streaming service initialized on {self.config.websocket_host}:{self.config.websocket_port}")

        except Exception as e:
            logger.error(f"Failed to initialize streaming service: {e}")
            raise

    async def _handle_client_connect(self, connection_id: str, websocket) -> None:
        """Handle new client connection"""
        try:
            # Create default subscription
            subscription = ClientSubscription(
                connection_id=connection_id,
                mode=self.config.default_mode
            )
            self.client_subscriptions[connection_id] = subscription

            # Send welcome message
            welcome_message = {
                "type": "welcome",
                "connection_id": connection_id,
                "timestamp": datetime.now().isoformat(),
                "server_info": {
                    "version": "1.0.0",
                    "features": ["real_time", "historical", "aggregated"],
                    "max_symbols_per_connection": 1000
                }
            }

            await self._send_message_to_client(connection_id, welcome_message)

            logger.info(f"Client connected: {connection_id}")

        except Exception as e:
            logger.error(f"Error handling client connect: {e}")

    async def _handle_client_disconnect(self, connection_id: str) -> None:
        """Handle client disconnection"""
        try:
            # Remove client subscriptions
            if connection_id in self.client_subscriptions:
                subscription = self.client_subscriptions[connection_id]

                # Remove from symbol subscribers
                for symbol in subscription.symbols:
                    if symbol in self.symbol_subscribers:
                        self.symbol_subscribers[symbol].discard(connection_id)
                        if not self.symbol_subscribers[symbol]:
                            del self.symbol_subscribers[symbol]

                del self.client_subscriptions[connection_id]

            logger.info(f"Client disconnected: {connection_id}")

        except Exception as e:
            logger.error(f"Error handling client disconnect: {e}")

    async def _handle_client_message(self, connection_id: str, message: Dict[str, Any]) -> None:
        """Handle message from client"""
        try:
            message_type = message.get("type")

            if message_type == "subscribe":
                await self._handle_subscribe_request(connection_id, message)
            elif message_type == "unsubscribe":
                await self._handle_unsubscribe_request(connection_id, message)
            elif message_type == "configure":
                await self._handle_configure_request(connection_id, message)
            elif message_type == "historical":
                await self._handle_historical_request(connection_id, message)
            else:
                await self._send_error_to_client(connection_id, f"Unknown message type: {message_type}")

        except Exception as e:
            logger.error(f"Error handling client message: {e}")
            await self._send_error_to_client(connection_id, "Internal server error")

    async def _handle_subscribe_request(self, connection_id: str, message: Dict[str, Any]) -> None:
        """Handle symbol subscription request"""
        symbols = message.get("symbols", [])

        if not isinstance(symbols, list):
            symbols = [symbols]

        if connection_id not in self.client_subscriptions:
            return

        subscription = self.client_subscriptions[connection_id]

        for symbol in symbols:
            symbol = symbol.upper()
            subscription.symbols.add(symbol)

            # Add to symbol subscribers
            if symbol not in self.symbol_subscribers:
                self.symbol_subscribers[symbol] = set()
            self.symbol_subscribers[symbol].add(connection_id)

        # Send confirmation
        response = {
            "type": "subscription_confirmation",
            "symbols": list(subscription.symbols),
            "mode": subscription.mode.value,
            "timestamp": datetime.now().isoformat()
        }

        await self._send_message_to_client(connection_id, response)
        logger.info(f"Client {connection_id} subscribed to {len(symbols)} symbols")

    async def _handle_unsubscribe_request(self, connection_id: str, message: Dict[str, Any]) -> None:
        """Handle symbol unsubscription request"""
        symbols = message.get("symbols", [])

        if not isinstance(symbols, list):
            symbols = [symbols]

        if connection_id not in self.client_subscriptions:
            return

        subscription = self.client_subscriptions[connection_id]

        for symbol in symbols:
            symbol = symbol.upper()
            subscription.symbols.discard(symbol)

            # Remove from symbol subscribers
            if symbol in self.symbol_subscribers:
                self.symbol_subscribers[symbol].discard(connection_id)
                if not self.symbol_subscribers[symbol]:
                    del self.symbol_subscribers[symbol]

        # Send confirmation
        response = {
            "type": "unsubscription_confirmation",
            "symbols": symbols,
            "timestamp": datetime.now().isoformat()
        }

        await self._send_message_to_client(connection_id, response)

    async def _handle_configure_request(self, connection_id: str, message: Dict[str, Any]) -> None:
        """Handle client configuration request"""
        if connection_id not in self.client_subscriptions:
            return

        subscription = self.client_subscriptions[connection_id]

        # Update streaming mode
        if "mode" in message:
            try:
                subscription.mode = StreamingMode(message["mode"])
            except ValueError:
                await self._send_error_to_client(connection_id, f"Invalid streaming mode: {message['mode']}")
                return

        # Update rate limit
        if "rate_limit" in message:
            rate_limit = float(message["rate_limit"])
            if 0.1 <= rate_limit <= 100:
                subscription.rate_limit = rate_limit
            else:
                await self._send_error_to_client(connection_id, "Rate limit must be between 0.1 and 100")
                return

        # Update filters
        if "filters" in message:
            subscription.filters = message["filters"]

        # Send confirmation
        response = {
            "type": "configuration_confirmation",
            "mode": subscription.mode.value,
            "rate_limit": subscription.rate_limit,
            "filters": subscription.filters,
            "timestamp": datetime.now().isoformat()
        }

        await self._send_message_to_client(connection_id, response)

    async def _handle_historical_request(self, connection_id: str, message: Dict[str, Any]) -> None:
        """Handle historical data request"""
        if not self.config.enable_historical_streaming:
            await self._send_error_to_client(connection_id, "Historical streaming not enabled")
            return

        try:
            symbol = message.get("symbol", "").upper()
            start_time = datetime.fromisoformat(message.get("start_time"))
            end_time = datetime.fromisoformat(message.get("end_time", datetime.now().isoformat()))

            # Get historical data
            historical_data = await self.storage_service.get_price_history(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                limit=1000  # Limit to prevent overwhelming client
            )

            # Send historical data in chunks
            chunk_size = 100
            for i in range(0, len(historical_data), chunk_size):
                chunk = historical_data[i:i + chunk_size]

                response = {
                    "type": "historical_data",
                    "symbol": symbol,
                    "data": [self._price_data_to_dict(price) for price in chunk],
                    "chunk": i // chunk_size + 1,
                    "total_chunks": (len(historical_data) + chunk_size - 1) // chunk_size,
                    "timestamp": datetime.now().isoformat()
                }

                await self._send_message_to_client(connection_id, response)

                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)

        except Exception as e:
            logger.error(f"Error handling historical request: {e}")
            await self._send_error_to_client(connection_id, "Failed to retrieve historical data")

    async def stream_price_update(self, price_data: PriceData) -> None:
        """Stream price update to subscribed clients"""
        if not self.is_running:
            return

        try:
            # Validate data quality
            if self.config.enable_data_validation and price_data.quality_score < self.config.min_quality_score:
                self.metrics.quality_filtered += 1
                return

            symbol = price_data.symbol.upper()

            # Add to symbol buffer
            if symbol not in self.symbol_buffers:
                self.symbol_buffers[symbol] = []

            self.symbol_buffers[symbol].append(price_data)

            # Maintain buffer size
            if len(self.symbol_buffers[symbol]) > self.config.symbol_buffer_size:
                self.symbol_buffers[symbol] = self.symbol_buffers[symbol][-self.config.symbol_buffer_size:]

            # Get subscribers for this symbol
            if symbol not in self.symbol_subscribers:
                return

            current_time = datetime.now()
            message_data = self._price_data_to_dict(price_data)

            # Process each subscribed client
            for connection_id in self.symbol_subscribers[symbol].copy():
                try:
                    subscription = self.client_subscriptions.get(connection_id)
                    if not subscription:
                        continue

                    # Check rate limiting
                    if not subscription.should_send_update(symbol, current_time):
                        continue

                    # Apply filters
                    if not self._passes_filters(price_data, subscription.filters):
                        continue

                    # Prepare message based on streaming mode
                    if subscription.mode == StreamingMode.REAL_TIME:
                        await self._send_realtime_update(connection_id, symbol, message_data)
                    elif subscription.mode == StreamingMode.BATCH:
                        await self._queue_batch_update(connection_id, symbol, message_data)
                    elif subscription.mode == StreamingMode.THROTTLED:
                        await self._send_throttled_update(connection_id, symbol, message_data)
                    elif subscription.mode == StreamingMode.SMART:
                        await self._send_smart_update(connection_id, symbol, message_data, subscription)

                    # Update last update time
                    subscription.last_update[symbol] = current_time

                except Exception as e:
                    logger.error(f"Error sending update to client {connection_id}: {e}")
                    # Remove problematic client
                    self.symbol_subscribers[symbol].discard(connection_id)

            self.metrics.total_messages_sent += len(self.symbol_subscribers[symbol])

        except Exception as e:
            logger.error(f"Error streaming price update: {e}")

    async def _send_realtime_update(self, connection_id: str, symbol: str, data: Dict[str, Any]) -> None:
        """Send immediate real-time update"""
        message = {
            "type": "price_update",
            "symbol": symbol,
            "data": data,
            "mode": "real_time",
            "timestamp": datetime.now().isoformat()
        }

        await self._send_message_to_client(connection_id, message)

    async def _queue_batch_update(self, connection_id: str, symbol: str, data: Dict[str, Any]) -> None:
        """Queue update for batch processing"""
        if connection_id not in self.pending_updates:
            self.pending_updates[connection_id] = []

        update = {
            "symbol": symbol,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }

        self.pending_updates[connection_id].append(update)

    async def _send_throttled_update(self, connection_id: str, symbol: str, data: Dict[str, Any]) -> None:
        """Send throttled update with rate limiting"""
        # Simple throttling - could be enhanced with more sophisticated algorithms
        message = {
            "type": "price_update",
            "symbol": symbol,
            "data": data,
            "mode": "throttled",
            "timestamp": datetime.now().isoformat()
        }

        await self._send_message_to_client(connection_id, message)

    async def _send_smart_update(self, connection_id: str, symbol: str, data: Dict[str, Any], subscription: ClientSubscription) -> None:
        """Send smart update based on client characteristics"""
        # Smart mode adapts based on client behavior and data characteristics
        # For now, use real-time for active clients, throttled for others

        recent_activity = any(
            (datetime.now() - last_update).total_seconds() < 60
            for last_update in subscription.last_update.values()
        )

        if recent_activity:
            await self._send_realtime_update(connection_id, symbol, data)
        else:
            await self._send_throttled_update(connection_id, symbol, data)

    async def _batch_processor_loop(self) -> None:
        """Background task for processing batched updates"""
        while self.is_running:
            try:
                # Process pending batch updates
                for connection_id, updates in list(self.pending_updates.items()):
                    if not updates:
                        continue

                    # Group updates by symbol
                    symbol_updates = {}
                    for update in updates:
                        symbol = update["symbol"]
                        if symbol not in symbol_updates:
                            symbol_updates[symbol] = []
                        symbol_updates[symbol].append(update)

                    # Send batched updates
                    message = {
                        "type": "batch_update",
                        "updates": symbol_updates,
                        "count": len(updates),
                        "timestamp": datetime.now().isoformat()
                    }

                    await self._send_message_to_client(connection_id, message)

                    # Clear processed updates
                    self.pending_updates[connection_id] = []

                await asyncio.sleep(self.config.batch_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                await asyncio.sleep(self.config.batch_interval)

    async def _metrics_updater_loop(self) -> None:
        """Background task for updating metrics"""
        while self.is_running:
            try:
                await self._update_metrics()
                await asyncio.sleep(1.0)  # Update metrics every second

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                await asyncio.sleep(1.0)

    async def _heartbeat_loop(self) -> None:
        """Background task for sending heartbeats"""
        while self.is_running:
            try:
                heartbeat_message = {
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat(),
                    "server_time": time.time()
                }

                # Send heartbeat to all connected clients
                for connection_id in self.client_subscriptions:
                    await self._send_message_to_client(connection_id, heartbeat_message)

                await asyncio.sleep(self.config.heartbeat_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(self.config.heartbeat_interval)

    async def _update_metrics(self) -> None:
        """Update streaming service metrics"""
        self.metrics.total_connections = len(self.client_subscriptions)
        self.metrics.active_connections = sum(
            1 for sub in self.client_subscriptions.values()
            if sub.symbols  # Has active subscriptions
        )

        # Update connections by mode
        self.metrics.connections_by_mode = {}
        for subscription in self.client_subscriptions.values():
            mode = subscription.mode
            self.metrics.connections_by_mode[mode] = self.metrics.connections_by_mode.get(mode, 0) + 1

        # Update queue metrics if available
        if self.message_queue_manager:
            queue_metrics = await self.message_queue_manager.get_queue_metrics("streaming")
            if queue_metrics:
                self.metrics.queue_utilization = queue_metrics.utilization

        # Update WebSocket metrics if available
        if self.websocket_server:
            ws_metrics = await self.websocket_server.get_connection_metrics()
            if ws_metrics:
                self.metrics.avg_latency_ms = ws_metrics.avg_response_time_ms

        self.metrics.last_updated = datetime.now()

    def _price_data_to_dict(self, price_data: PriceData) -> Dict[str, Any]:
        """Convert PriceData to dictionary for JSON serialization"""
        return {
            "symbol": price_data.symbol,
            "timestamp": price_data.timestamp.isoformat(),
            "open": float(price_data.open_price),
            "high": float(price_data.high_price),
            "low": float(price_data.low_price),
            "close": float(price_data.close_price),
            "volume": int(price_data.volume),
            "source": price_data.source,
            "quality_score": price_data.quality_score
        }

    def _passes_filters(self, price_data: PriceData, filters: Dict[str, Any]) -> bool:
        """Check if price data passes client filters"""
        if not filters:
            return True

        # Quality filter
        if "min_quality" in filters:
            if price_data.quality_score < filters["min_quality"]:
                return False

        # Volume filter
        if "min_volume" in filters:
            if price_data.volume < filters["min_volume"]:
                return False

        # Price change filter
        if "min_price_change" in filters and hasattr(price_data, 'previous_close'):
            change_pct = abs(price_data.close_price - price_data.previous_close) / price_data.previous_close * 100
            if change_pct < filters["min_price_change"]:
                return False

        return True

    async def _send_message_to_client(self, connection_id: str, message: Dict[str, Any]) -> None:
        """Send message to specific client"""
        if self.websocket_server:
            await self.websocket_server.send_message_to_connection(connection_id, message)

    async def _send_error_to_client(self, connection_id: str, error_message: str) -> None:
        """Send error message to client"""
        error_response = {
            "type": "error",
            "message": error_message,
            "timestamp": datetime.now().isoformat()
        }
        await self._send_message_to_client(connection_id, error_response)

    async def get_streaming_metrics(self) -> StreamingMetrics:
        """Get comprehensive streaming metrics"""
        await self._update_metrics()
        return self.metrics

    async def get_subscription_info(self) -> Dict[str, Any]:
        """Get subscription information"""
        return {
            "total_clients": len(self.client_subscriptions),
            "total_symbols": len(self.symbol_subscribers),
            "subscriptions_by_symbol": {
                symbol: len(subscribers) for symbol, subscribers in self.symbol_subscribers.items()
            },
            "clients_by_mode": {
                mode.value: sum(1 for sub in self.client_subscriptions.values() if sub.mode == mode)
                for mode in StreamingMode
            }
        }

    async def add_data_handler(self, handler: Callable[[PriceData], None]) -> None:
        """Add custom data handler"""
        self.data_handlers.append(handler)

    async def close(self) -> None:
        """Shutdown the streaming service"""
        try:
            self.is_running = False

            # Cancel background tasks
            tasks_to_cancel = [
                self.batch_processor_task,
                self.metrics_updater_task,
                self.heartbeat_task
            ]

            for task in tasks_to_cancel:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            # Close WebSocket server
            if self.websocket_server:
                await self.websocket_server.close()

            # Close message queue manager
            if self.message_queue_manager:
                await self.message_queue_manager.close()

            # Clear data structures
            self.client_subscriptions.clear()
            self.symbol_subscribers.clear()
            self.symbol_buffers.clear()
            self.pending_updates.clear()

            logger.info("Real-time streaming service shut down")

        except Exception as e:
            logger.error(f"Error closing streaming service: {e}")

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Global streaming service instance
streaming_service = None

async def get_streaming_service(
    storage_service: HybridStorageService,
    cache_manager: RedisCacheManager,
    config: StreamingConfig = None
) -> RealtimeStreamingService:
    """Get or create global streaming service instance"""
    global streaming_service
    if streaming_service is None:
        streaming_service = RealtimeStreamingService(storage_service, cache_manager, config)
        await streaming_service.initialize()
    return streaming_service