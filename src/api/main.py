"""Main Application Entry Point

This module provides the main entry point for running the Market Data Agent API server.
It can be run directly or used with uvicorn for production deployment.
"""

import uvicorn
import os
import logging
from dotenv import load_dotenv
from .endpoints import app
from ..config.config_manager import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Main function to run the API server"""
    try:
        # Load environment variables from .env file
        load_dotenv()

        # Load configuration
        config = load_config("config.yaml")
        api_config = config.api

        # Environment variables can still override config file
        host = os.getenv("API_HOST", api_config.host)
        port = int(os.getenv("API_PORT", str(api_config.port)))
        reload = os.getenv("API_RELOAD", str(api_config.reload)).lower() == "true"
        workers = int(os.getenv("API_WORKERS", str(api_config.workers)))
        log_level = os.getenv("API_LOG_LEVEL", api_config.log_level)

        logger.info(f"Starting Market Data Agent API on {host}:{port}")
        logger.info(f"Environment: {config.environment}")
        logger.info(f"Debug mode: {config.debug}")

        # Run with uvicorn
        uvicorn.run(
            "src.api.endpoints:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1,  # Multiple workers not supported with reload
            log_level=log_level.lower()
        )

    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        raise


if __name__ == "__main__":
    main()