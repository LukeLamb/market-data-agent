"""Market Data Agent - Main Entry Point"""

import asyncio
import logging
import os
import signal
import sys
from typing import Optional

from src.config.enhanced_config import (
    initialize_enhanced_config,
    shutdown_enhanced_config,
    get_current_config,
    is_development
)
from src.performance import start_performance_components, stop_performance_components


class MarketDataAgent:
    """Main Market Data Agent application"""

    def __init__(self):
        self.config = None
        self.logger = None
        self.running = False

    async def initialize(self, config_file: Optional[str] = None, environment: Optional[str] = None):
        """Initialize the Market Data Agent"""
        try:
            # Initialize enhanced configuration system
            self.config = await initialize_enhanced_config(
                base_config_file=config_file or "config.yaml",
                environment=environment,
                enable_hot_reload=is_development()
            )

            # Setup logging based on configuration
            self._setup_logging()
            self.logger = logging.getLogger(__name__)

            self.logger.info(f"Market Data Agent initializing...")
            self.logger.info(f"Environment: {self.config.environment}")
            self.logger.info(f"Debug mode: {self.config.debug}")

            # Initialize performance optimization components
            await start_performance_components()
            self.logger.info("Performance optimization components started")

            # TODO: Initialize other components
            # - Data sources
            # - Storage system
            # - API server
            # - Memory system
            # - Monitoring system

            self.running = True
            self.logger.info("Market Data Agent initialization complete")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to initialize Market Data Agent: {e}")
            else:
                print(f"Failed to initialize Market Data Agent: {e}")
            raise

    async def shutdown(self):
        """Shutdown the Market Data Agent"""
        if not self.running:
            return

        if self.logger:
            self.logger.info("Market Data Agent shutting down...")

        try:
            # Stop performance components
            await stop_performance_components()
            if self.logger:
                self.logger.info("Performance optimization components stopped")

            # TODO: Shutdown other components
            # - API server
            # - Data sources
            # - Storage system
            # - Memory system
            # - Monitoring system

            # Shutdown configuration system
            await shutdown_enhanced_config()
            if self.logger:
                self.logger.info("Configuration system stopped")

            self.running = False
            if self.logger:
                self.logger.info("Market Data Agent shutdown complete")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during shutdown: {e}")
            else:
                print(f"Error during shutdown: {e}")

    async def run(self):
        """Run the Market Data Agent"""
        if not self.running:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        self.logger.info("Market Data Agent running...")

        try:
            # Keep the agent running
            while self.running:
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            self.logger.info("Market Data Agent cancelled")
        except Exception as e:
            self.logger.error(f"Unexpected error in main loop: {e}")
            raise
        finally:
            await self.shutdown()

    def _setup_logging(self):
        """Setup logging based on configuration"""
        log_config = self.config.logging

        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, log_config.level.upper(), logging.INFO),
            format=log_config.format,
            force=True  # Override any existing configuration
        )

        # Setup file logging if enabled
        if log_config.file_enabled:
            from logging.handlers import RotatingFileHandler

            # Ensure log directory exists
            log_dir = os.path.dirname(log_config.file_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            file_handler = RotatingFileHandler(
                log_config.file_path,
                maxBytes=log_config.max_file_size,
                backupCount=log_config.backup_count
            )
            file_handler.setFormatter(logging.Formatter(log_config.format))
            logging.getLogger().addHandler(file_handler)


# Global agent instance
_agent: Optional[MarketDataAgent] = None


async def main():
    """Main application entry point"""
    global _agent

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Market Data Agent")
    parser.add_argument("--config", help="Configuration file path", default="config.yaml")
    parser.add_argument("--environment", help="Environment (development, staging, production)")
    args = parser.parse_args()

    _agent = MarketDataAgent()

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        if _agent and _agent.running:
            asyncio.create_task(_agent.shutdown())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Initialize and run the agent
        await _agent.initialize(
            config_file=args.config,
            environment=args.environment
        )

        await _agent.run()

    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        if _agent:
            await _agent.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)