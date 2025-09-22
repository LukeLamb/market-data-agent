"""API Endpoints

This module provides REST API endpoints for accessing market data using FastAPI.
It exposes current price data, historical data, and health monitoring capabilities.
"""

from fastapi import FastAPI, HTTPException, Depends, Query, Path
from fastapi.responses import JSONResponse
from datetime import date, datetime
from typing import Dict, List, Optional, Any
import logging
import os

from ..data_sources.source_manager import DataSourceManager, SourcePriority
from ..data_sources.base import (
    PriceData,
    CurrentPrice,
    HealthStatus,
    SymbolNotFoundError,
    RateLimitError,
    DataSourceError
)
from ..config.config_manager import load_config

logger = logging.getLogger(__name__)

# Global manager instance (will be initialized on startup)
data_manager: Optional[DataSourceManager] = None


def get_data_manager() -> DataSourceManager:
    """Dependency to get the data manager instance"""
    if data_manager is None:
        raise HTTPException(status_code=500, detail="Data manager not initialized")
    return data_manager


app = FastAPI(
    title="Market Data Agent API",
    description="REST API for accessing financial market data with intelligent source management",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.on_event("startup")
async def startup_event():
    """Initialize the data manager on startup"""
    global data_manager

    # Load configuration
    config = load_config("config.yaml")
    logger.info(f"Loaded configuration for environment: {config.environment}")

    # Configuration for data manager from config file
    manager_config = {
        "max_failure_threshold": config.source_manager.max_failure_threshold,
        "circuit_breaker_timeout": config.source_manager.circuit_breaker_timeout,
        "health_check_interval": config.source_manager.health_check_interval,
        "validation_enabled": config.source_manager.validation_enabled,
        "validation": {
            "max_price_change_percent": config.validation.max_price_change_percent,
            "min_volume": config.validation.min_volume,
            "min_price": config.validation.min_price,
            "max_price": config.validation.max_price
        }
    }

    data_manager = DataSourceManager(manager_config)

    # Register YFinance source if enabled
    if config.yfinance.enabled:
        yf_config = {
            "priority": config.yfinance.priority,
            "rate_limit_requests": config.yfinance.rate_limit_requests,
            "rate_limit_period": config.yfinance.rate_limit_period,
            "timeout": config.yfinance.timeout,
            "max_retries": config.yfinance.max_retries
        }
        data_manager.register_yfinance_source(yf_config)
        logger.info("Registered YFinance source")

    # Register Alpha Vantage source if enabled and API key is available
    if config.alpha_vantage.enabled:
        api_key = config.alpha_vantage.api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if api_key:
            av_config = {
                "api_key": api_key,
                "priority": config.alpha_vantage.priority,
                "rate_limit_requests": config.alpha_vantage.rate_limit_requests,
                "rate_limit_period": config.alpha_vantage.rate_limit_period,
                "timeout": config.alpha_vantage.timeout,
                "max_retries": config.alpha_vantage.max_retries
            }
            data_manager.register_alpha_vantage_source(av_config)
            logger.info("Registered Alpha Vantage source with API key")
        else:
            logger.info("Alpha Vantage enabled but no API key found, skipping registration")

    # Start health monitoring
    await data_manager.start_health_monitoring()
    logger.info("Market Data Agent API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global data_manager
    if data_manager:
        await data_manager.close()
        logger.info("Market Data Agent API shutdown completed")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Market Data Agent API",
        "version": "1.0.0",
        "description": "REST API for accessing financial market data",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check(manager: DataSourceManager = Depends(get_data_manager)):
    """Health check endpoint"""
    try:
        source_health = await manager.get_source_health_status()
        source_stats = manager.get_source_statistics()

        # Determine overall health
        healthy_sources = sum(1 for status in source_health.values()
                            if status.status.value == "healthy")
        total_sources = len(source_health)

        overall_status = "healthy" if healthy_sources > 0 else "unhealthy"

        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "sources": {
                name: {
                    "status": status.status.value,
                    "error_count": status.error_count,
                    "message": status.message,
                    "response_time_ms": status.response_time_ms,
                    "reliability": source_stats.get(name, {}).get("reliability", 0.0),
                    "is_available": source_stats.get(name, {}).get("is_available", False)
                }
                for name, status in source_health.items()
            },
            "healthy_sources": healthy_sources,
            "total_sources": total_sources
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.get("/sources")
async def get_sources(manager: DataSourceManager = Depends(get_data_manager)):
    """Get information about registered data sources"""
    try:
        stats = manager.get_source_statistics()
        return {
            "sources": stats,
            "total_sources": len(stats)
        }
    except Exception as e:
        logger.error(f"Failed to get sources: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve source information")


@app.get("/price/{symbol}")
async def get_current_price(
    symbol: str = Path(..., description="Stock symbol (e.g., AAPL, GOOGL)"),
    manager: DataSourceManager = Depends(get_data_manager)
) -> Dict[str, Any]:
    """Get current price for a symbol"""
    try:
        symbol = symbol.upper()
        price = await manager.get_current_price(symbol)

        return {
            "symbol": price.symbol,
            "price": price.price,
            "timestamp": price.timestamp.isoformat(),
            "volume": price.volume,
            "bid": price.bid,
            "ask": price.ask,
            "source": price.source,
            "quality_score": price.quality_score
        }

    except SymbolNotFoundError:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
    except RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit exceeded, please try again later")
    except DataSourceError as e:
        logger.error(f"Data source error for {symbol}: {e}")
        raise HTTPException(status_code=503, detail="Data source temporarily unavailable")
    except HTTPException:
        raise  # Re-raise HTTPExceptions as-is
    except Exception as e:
        logger.error(f"Unexpected error getting price for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/historical/{symbol}")
async def get_historical_data(
    symbol: str = Path(..., description="Stock symbol (e.g., AAPL, GOOGL)"),
    start_date: date = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: date = Query(..., description="End date (YYYY-MM-DD)"),
    interval: str = Query("1d", description="Data interval (1d, 1wk, 1mo)"),
    manager: DataSourceManager = Depends(get_data_manager)
) -> Dict[str, Any]:
    """Get historical data for a symbol"""
    try:
        symbol = symbol.upper()

        # Validate date range
        if start_date > end_date:
            raise HTTPException(status_code=400, detail="Start date must be before end date")

        if start_date > date.today():
            raise HTTPException(status_code=400, detail="Start date cannot be in the future")

        data = await manager.get_historical_data(symbol, start_date, end_date, interval)

        return {
            "symbol": symbol,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "interval": interval,
            "data_points": len(data),
            "data": [
                {
                    "timestamp": item.timestamp.isoformat(),
                    "open": item.open_price,
                    "high": item.high_price,
                    "low": item.low_price,
                    "close": item.close_price,
                    "volume": item.volume,
                    "adjusted_close": item.adjusted_close,
                    "source": item.source,
                    "quality_score": item.quality_score
                }
                for item in data
            ]
        }

    except SymbolNotFoundError:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
    except RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit exceeded, please try again later")
    except DataSourceError as e:
        logger.error(f"Data source error for {symbol}: {e}")
        raise HTTPException(status_code=503, detail="Data source temporarily unavailable")
    except HTTPException:
        raise  # Re-raise HTTPExceptions as-is
    except Exception as e:
        logger.error(f"Unexpected error getting historical data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/validate/{symbol}")
async def validate_symbol(
    symbol: str = Path(..., description="Stock symbol to validate"),
    manager: DataSourceManager = Depends(get_data_manager)
) -> Dict[str, Any]:
    """Validate if a symbol exists across data sources"""
    try:
        symbol = symbol.upper()
        validation_results = await manager.validate_symbol(symbol)

        valid_sources = [name for name, is_valid in validation_results.items() if is_valid]

        return {
            "symbol": symbol,
            "is_valid": len(valid_sources) > 0,
            "valid_sources": valid_sources,
            "validation_results": validation_results,
            "total_sources_checked": len(validation_results)
        }

    except HTTPException:
        raise  # Re-raise HTTPExceptions as-is
    except Exception as e:
        logger.error(f"Unexpected error validating symbol {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/symbols")
async def get_supported_symbols(
    manager: DataSourceManager = Depends(get_data_manager),
    limit: int = Query(100, description="Maximum number of symbols to return")
) -> Dict[str, Any]:
    """Get list of supported symbols from all sources"""
    try:
        all_symbols = set()
        source_symbols = {}

        for source_name, source in manager.sources.items():
            if manager._is_source_available(source_name):
                try:
                    symbols = await source.get_supported_symbols()
                    source_symbols[source_name] = len(symbols)
                    all_symbols.update(symbols)
                except Exception as e:
                    logger.warning(f"Failed to get symbols from {source_name}: {e}")
                    source_symbols[source_name] = 0

        # Convert to sorted list and limit
        sorted_symbols = sorted(list(all_symbols))[:limit]

        return {
            "symbols": sorted_symbols,
            "total_unique_symbols": len(all_symbols),
            "returned_symbols": len(sorted_symbols),
            "source_counts": source_symbols
        }

    except HTTPException:
        raise  # Re-raise HTTPExceptions as-is
    except Exception as e:
        logger.error(f"Unexpected error getting supported symbols: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )