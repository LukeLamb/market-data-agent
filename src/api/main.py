"""Main Application Entry Point

This module provides the main entry point for running the Market Data Agent API server.
It can be run directly or used with uvicorn for production deployment.
"""

import uvicorn
import os
import logging
from .endpoints import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Main function to run the API server"""
    # Configuration from environment variables
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    workers = int(os.getenv("API_WORKERS", "1"))

    logger.info(f"Starting Market Data Agent API on {host}:{port}")

    # Run with uvicorn
    uvicorn.run(
        "src.api.endpoints:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,  # Multiple workers not supported with reload
        log_level="info"
    )


if __name__ == "__main__":
    main()