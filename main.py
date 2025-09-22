"""Market Data Agent - Main Entry Point"""

import asyncio
import logging
from src.config.settings import get_settings


def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


async def main():
    """Main application entry point"""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting Market Data Agent...")

    # TODO: Initialize components after implementation
    # - Data sources
    # - Storage system
    # - API server

    logger.info("Market Data Agent ready")


if __name__ == "__main__":
    asyncio.run(main())