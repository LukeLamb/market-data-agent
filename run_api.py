#!/usr/bin/env python3
"""
Market Data Agent API Runner

Simple script to start the Market Data Agent API server.
Usage:
    python run_api.py

Environment Variables:
    API_HOST: Host to bind to (default: 0.0.0.0)
    API_PORT: Port to bind to (default: 8000)
    API_RELOAD: Enable auto-reload in development (default: false)
    ALPHA_VANTAGE_API_KEY: Optional Alpha Vantage API key
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.api.main import main

if __name__ == "__main__":
    main()