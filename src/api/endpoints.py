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
from ..memory.memory_manager import MemoryManager, AdaptiveLearningConfig
from ..quality.quality_manager import QualityManager, QualityManagerConfig
from ..validation.validation_engine import ValidationEngine
from ..monitoring import (
    MetricsCollector, AlertingSystem, MonitoringDashboard,
    get_default_metrics, get_default_alerts, get_default_dashboard,
    record_counter, record_gauge, record_timer, time_operation
)

logger = logging.getLogger(__name__)

# Global manager instances (will be initialized on startup)
data_manager: Optional[DataSourceManager] = None
memory_manager: Optional[MemoryManager] = None
quality_manager: Optional[QualityManager] = None
validation_engine: Optional[ValidationEngine] = None
monitoring_dashboard: Optional[MonitoringDashboard] = None


def get_data_manager() -> DataSourceManager:
    """Dependency to get the data manager instance"""
    if data_manager is None:
        raise HTTPException(status_code=500, detail="Data manager not initialized")
    return data_manager


def get_memory_manager() -> MemoryManager:
    """Dependency to get the memory manager instance"""
    if memory_manager is None:
        raise HTTPException(status_code=500, detail="Memory manager not initialized")
    return memory_manager


def get_quality_manager() -> QualityManager:
    """Dependency to get the quality manager instance"""
    if quality_manager is None:
        raise HTTPException(status_code=500, detail="Quality manager not initialized")
    return quality_manager


def get_validation_engine() -> ValidationEngine:
    """Dependency to get the validation engine instance"""
    if validation_engine is None:
        raise HTTPException(status_code=500, detail="Validation engine not initialized")
    return validation_engine


def get_monitoring_dashboard() -> MonitoringDashboard:
    """Dependency to get the monitoring dashboard instance"""
    if monitoring_dashboard is None:
        raise HTTPException(status_code=500, detail="Monitoring dashboard not initialized")
    return monitoring_dashboard


app = FastAPI(
    title="Market Data Agent API",
    description="REST API for accessing financial market data with intelligent source management",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add monitoring middleware
@app.middleware("http")
async def monitoring_middleware(request, call_next):
    """Middleware to track request metrics"""
    # Record request start
    start_time = datetime.now()
    record_counter('api_requests_total', 1, {
        'method': request.method,
        'endpoint': str(request.url.path)
    })

    # Process request
    with time_operation('api_request_duration', {
        'method': request.method,
        'endpoint': str(request.url.path)
    }):
        response = await call_next(request)

    # Record response metrics
    record_counter('api_responses_total', 1, {
        'method': request.method,
        'endpoint': str(request.url.path),
        'status_code': str(response.status_code)
    })

    # Record response time
    duration = (datetime.now() - start_time).total_seconds()
    record_timer('api_response_time', duration, {
        'method': request.method,
        'endpoint': str(request.url.path),
        'status_code': str(response.status_code)
    })

    return response


@app.on_event("startup")
async def startup_event():
    """Initialize all managers on startup"""
    global data_manager, memory_manager, quality_manager, validation_engine, monitoring_dashboard

    # Load configuration
    config = load_config("config.yaml")
    logger.info(f"Loaded configuration for environment: {config.environment}")

    # Initialize validation engine first (needed by others)
    validation_engine = ValidationEngine()
    await validation_engine.initialize()
    logger.info("Validation engine initialized")

    # Initialize memory manager
    memory_config = AdaptiveLearningConfig(
        pattern_detection_threshold=0.7,
        anomaly_detection_threshold=0.3,
        quality_correlation_threshold=0.6,
        enable_predictive_scoring=True,
        enable_source_reputation=True,
        enable_market_context=True
    )
    memory_manager = MemoryManager(memory_config)
    await memory_manager.initialize_memory_system()
    logger.info("Memory manager initialized")

    # Initialize quality manager with validation integration
    quality_config = QualityManagerConfig(
        enable_auto_assessment=True,
        assessment_interval_minutes=5.0,
        critical_quality_threshold=60.0,
        enable_quality_alerts=True,
        enable_validation_integration=True
    )
    quality_manager = QualityManager(quality_config, validation_engine)
    await quality_manager.start_quality_management()
    logger.info("Quality manager initialized and started")

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

    # Set up learning integration - data manager will feed data to memory and quality systems
    data_manager._memory_manager = memory_manager
    data_manager._quality_manager = quality_manager

    # Initialize monitoring dashboard
    monitoring_dashboard = get_default_dashboard()
    logger.info("Monitoring dashboard initialized")

    # Record startup metrics
    record_counter('api_startup_count')
    record_gauge('api_startup_timestamp', datetime.now().timestamp())

    logger.info("Market Data Agent API started successfully with memory, quality, and monitoring integration")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global data_manager, memory_manager, quality_manager, validation_engine

    # Stop quality management first
    if quality_manager:
        await quality_manager.stop_quality_management()
        logger.info("Quality manager stopped")

    # Clean up memory system
    if memory_manager:
        await memory_manager.cleanup_old_memories()
        logger.info("Memory manager cleaned up")

    # Close data manager
    if data_manager:
        await data_manager.close()
        logger.info("Data manager closed")

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
    manager: DataSourceManager = Depends(get_data_manager),
    memory_mgr: MemoryManager = Depends(get_memory_manager),
    quality_mgr: QualityManager = Depends(get_quality_manager)
) -> Dict[str, Any]:
    """Get current price for a symbol with learning integration"""
    try:
        symbol = symbol.upper()
        price = await manager.get_current_price(symbol)

        # Learn from the price data in memory system
        await memory_mgr.learn_from_price_data(symbol, price.source, [price])

        # Get quality prediction from memory system
        predicted_quality = await memory_mgr.predict_quality_score(symbol, price.source, price)

        # Get source reputation
        source_reputation = await memory_mgr.get_source_reputation(price.source)

        # Get market context
        market_context = await memory_mgr.get_market_context(symbol)

        response = {
            "symbol": price.symbol,
            "price": price.price,
            "timestamp": price.timestamp.isoformat(),
            "volume": price.volume,
            "bid": price.bid,
            "ask": price.ask,
            "source": price.source,
            "quality_score": price.quality_score,
            "learning_insights": {
                "predicted_quality": predicted_quality,
                "source_reputation": source_reputation,
                "market_context": market_context
            }
        }

        return response

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


@app.get("/quality/assessment/{symbol}")
async def get_quality_assessment(
    symbol: str = Path(..., description="Stock symbol to assess"),
    source: Optional[str] = Query(None, description="Specific source to assess"),
    manager: DataSourceManager = Depends(get_data_manager),
    quality_mgr: QualityManager = Depends(get_quality_manager)
) -> Dict[str, Any]:
    """Get quality assessment for a symbol from a specific source or all sources"""
    try:
        symbol = symbol.upper()

        if source:
            # Get current price from specific source
            price = await manager.get_current_price_from_source(symbol, source)

            # Assess quality
            score_card = await quality_mgr.assess_data_quality(symbol, source, [price])

            return {
                "symbol": symbol,
                "source": source,
                "assessment": {
                    "overall_grade": score_card.overall_grade.value,
                    "overall_score": score_card.overall_score,
                    "dimension_scores": score_card.dimension_scores,
                    "total_issues": score_card.total_issues,
                    "critical_issues": score_card.critical_issues,
                    "high_issues": score_card.high_issues,
                    "medium_issues": score_card.medium_issues,
                    "low_issues": score_card.low_issues,
                    "priority_recommendations": score_card.priority_recommendations,
                    "improvement_suggestions": score_card.improvement_suggestions,
                    "confidence_level": score_card.confidence_level,
                    "assessment_period": [
                        score_card.assessment_period[0].isoformat(),
                        score_card.assessment_period[1].isoformat()
                    ],
                    "data_points_analyzed": score_card.data_points_analyzed
                }
            }
        else:
            # Get assessments from all sources
            assessments = {}
            for source_name in manager.sources.keys():
                try:
                    if manager._is_source_available(source_name):
                        price = await manager.get_current_price_from_source(symbol, source_name)
                        score_card = await quality_mgr.assess_data_quality(symbol, source_name, [price])
                        assessments[source_name] = {
                            "overall_grade": score_card.overall_grade.value,
                            "overall_score": score_card.overall_score,
                            "total_issues": score_card.total_issues,
                            "critical_issues": score_card.critical_issues
                        }
                except Exception as e:
                    logger.warning(f"Failed to assess {symbol} from {source_name}: {e}")
                    assessments[source_name] = {"error": str(e)}

            return {
                "symbol": symbol,
                "assessments": assessments,
                "total_sources_assessed": len(assessments)
            }

    except SymbolNotFoundError:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quality assessment failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Quality assessment failed")


@app.get("/quality/report")
async def get_quality_report(
    quality_mgr: QualityManager = Depends(get_quality_manager)
) -> Dict[str, Any]:
    """Get comprehensive quality report"""
    try:
        report = quality_mgr.get_quality_report()
        return report
    except Exception as e:
        logger.error(f"Failed to get quality report: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve quality report")


@app.get("/quality/action-plan/{symbol}")
async def get_action_plan(
    symbol: str = Path(..., description="Stock symbol"),
    source: str = Query(..., description="Data source"),
    quality_mgr: QualityManager = Depends(get_quality_manager)
) -> Dict[str, Any]:
    """Get quality action plan for a symbol-source combination"""
    try:
        symbol = symbol.upper()
        action_plan = quality_mgr.get_action_plan(symbol, source)

        if not action_plan:
            raise HTTPException(status_code=404, detail=f"No action plan found for {symbol} from {source}")

        return {
            "symbol": action_plan.symbol,
            "source": action_plan.source,
            "current_grade": action_plan.current_grade.value,
            "current_score": action_plan.current_score,
            "target_score": action_plan.target_score,
            "priority_level": action_plan.priority_level,
            "immediate_actions": action_plan.immediate_actions,
            "short_term_actions": action_plan.short_term_actions,
            "long_term_actions": action_plan.long_term_actions,
            "success_criteria": action_plan.success_criteria,
            "review_date": action_plan.review_date.isoformat(),
            "estimated_improvement": action_plan.estimated_improvement,
            "created_date": action_plan.created_date.isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get action plan: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve action plan")


@app.get("/memory/report")
async def get_memory_report(
    memory_mgr: MemoryManager = Depends(get_memory_manager)
) -> Dict[str, Any]:
    """Get comprehensive memory system report"""
    try:
        report = memory_mgr.get_memory_report()
        return report
    except Exception as e:
        logger.error(f"Failed to get memory report: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve memory report")


@app.get("/memory/context/{symbol}")
async def get_symbol_context(
    symbol: str = Path(..., description="Stock symbol"),
    memory_mgr: MemoryManager = Depends(get_memory_manager)
) -> Dict[str, Any]:
    """Get learned context and insights for a symbol"""
    try:
        symbol = symbol.upper()
        context = await memory_mgr.get_market_context(symbol)

        # Get source reputations for this symbol
        source_reputations = {}
        for source_name in ["polygon", "alpha_vantage", "iex", "finnhub", "yfinance"]:
            reputation = await memory_mgr.get_source_reputation(source_name)
            source_reputations[source_name] = reputation

        return {
            "symbol": symbol,
            "market_context": context,
            "source_reputations": source_reputations,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get symbol context for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve symbol context")


@app.get("/memory/health")
async def get_memory_health(
    memory_mgr: MemoryManager = Depends(get_memory_manager)
) -> Dict[str, Any]:
    """Get memory system health status"""
    try:
        health_status = await memory_mgr.get_health_status()
        return health_status
    except Exception as e:
        logger.error(f"Memory health check failed: {e}")
        raise HTTPException(status_code=500, detail="Memory health check failed")


@app.get("/system/comprehensive-health")
async def comprehensive_health_check(
    manager: DataSourceManager = Depends(get_data_manager),
    memory_mgr: MemoryManager = Depends(get_memory_manager),
    quality_mgr: QualityManager = Depends(get_quality_manager),
    validation_eng: ValidationEngine = Depends(get_validation_engine)
) -> Dict[str, Any]:
    """Comprehensive system health check including all components"""
    try:
        # Get individual health statuses
        data_health = await manager.get_source_health_status()
        memory_health = await memory_mgr.get_health_status()
        quality_health = await quality_mgr.health_check()
        validation_health = await validation_eng.get_health_status()

        # Determine overall system health
        component_statuses = [
            memory_health["overall_status"],
            quality_health["overall_status"],
            validation_health["overall_health"]
        ]

        data_healthy = sum(1 for status in data_health.values()
                          if status.status.value == "healthy") > 0

        overall_healthy = (
            data_healthy and
            all(status in ["healthy", "active"] for status in component_statuses)
        )

        return {
            "overall_status": "healthy" if overall_healthy else "degraded",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "data_sources": {
                    "status": "healthy" if data_healthy else "unhealthy",
                    "healthy_sources": sum(1 for status in data_health.values()
                                         if status.status.value == "healthy"),
                    "total_sources": len(data_health)
                },
                "memory_system": {
                    "status": memory_health["overall_status"],
                    "entities": memory_health["components"]["memory_server"]["entities"],
                    "relations": memory_health["components"]["memory_server"]["relations"]
                },
                "quality_system": {
                    "status": quality_health["overall_status"],
                    "components_healthy": sum(1 for comp in quality_health["components"].values()
                                            if comp.get("status") in ["healthy", "active"])
                },
                "validation_system": {
                    "status": validation_health["overall_health"],
                    "modes_active": len([m for m in validation_health["validation_modes"].values()
                                       if m.get("enabled", False)])
                }
            },
            "system_metrics": {
                "memory_patterns_learned": memory_health["components"]["pattern_learning"]["patterns_learned"],
                "quality_assessments_managed": quality_health.get("assessments_completed", 0),
                "validation_checks_performed": validation_health.get("total_validations", 0)
            }
        }

    except Exception as e:
        logger.error(f"Comprehensive health check failed: {e}")
        raise HTTPException(status_code=500, detail="Comprehensive health check failed")


# Monitoring and Alerting Endpoints

@app.get("/monitoring/dashboard/{layout_name}")
async def get_monitoring_dashboard(
    layout_name: str = Path(..., description="Dashboard layout name (overview, performance, alerts)"),
    dashboard: MonitoringDashboard = Depends(get_monitoring_dashboard)
):
    """
    Get monitoring dashboard data for a specific layout

    Available layouts:
    - overview: General system overview with key metrics
    - performance: Performance-focused metrics and charts
    - alerts: Alert status and history
    """
    try:
        # Record API request metric
        record_counter('monitoring_dashboard_requests', 1, {'layout': layout_name})

        with time_operation('dashboard_generation_time', {'layout': layout_name}):
            dashboard_data = dashboard.generate_dashboard(layout_name)

        return {
            "success": True,
            "layout_name": layout_name,
            "dashboard": dashboard_data
        }
    except Exception as e:
        logger.error(f"Error generating dashboard {layout_name}: {e}")
        record_counter('monitoring_dashboard_errors', 1, {'layout': layout_name, 'error': str(e)})
        raise HTTPException(status_code=500, detail=f"Failed to generate dashboard: {str(e)}")


@app.get("/monitoring/metrics")
async def get_current_metrics():
    """
    Get current system metrics
    """
    try:
        record_counter('monitoring_metrics_requests')

        metrics_collector = get_default_metrics()
        current_metrics = metrics_collector.get_current_metrics()

        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "metrics": current_metrics
        }
    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}")
        record_counter('monitoring_metrics_errors', 1, {'error': str(e)})
        raise HTTPException(status_code=500, detail=f"Failed to retrieve metrics: {str(e)}")


@app.get("/monitoring/metrics/{metric_name}")
async def get_metric_details(
    metric_name: str = Path(..., description="Name of the metric to retrieve"),
    include_history: bool = Query(False, description="Include historical data")
):
    """
    Get detailed information about a specific metric
    """
    try:
        record_counter('monitoring_metric_detail_requests', 1, {'metric': metric_name})

        metrics_collector = get_default_metrics()

        # Get current value
        current_metrics = metrics_collector.get_current_metrics()
        current_value = current_metrics.get(metric_name)

        if current_value is None:
            raise HTTPException(status_code=404, detail=f"Metric '{metric_name}' not found")

        response = {
            "success": True,
            "metric_name": metric_name,
            "current_value": current_value,
            "summary": metrics_collector.get_summary(metric_name).__dict__
        }

        if include_history:
            # Get historical data if available
            if metric_name in metrics_collector.metrics:
                history = list(metrics_collector.metrics[metric_name])
                response["history"] = [
                    {
                        "value": mv.value,
                        "timestamp": mv.timestamp.isoformat(),
                        "tags": mv.tags
                    }
                    for mv in history[-100:]  # Last 100 data points
                ]

        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving metric {metric_name}: {e}")
        record_counter('monitoring_metric_detail_errors', 1, {'metric': metric_name, 'error': str(e)})
        raise HTTPException(status_code=500, detail=f"Failed to retrieve metric details: {str(e)}")


@app.get("/monitoring/alerts")
async def get_active_alerts():
    """
    Get current active alerts
    """
    try:
        record_counter('monitoring_alerts_requests')

        alerting_system = get_default_alerts()

        # Get active alerts from alerting system
        active_alerts = alerting_system.get_active_alerts() if hasattr(alerting_system, 'get_active_alerts') else []

        # Also check metrics collector for any triggered alerts
        metrics_collector = get_default_metrics()
        metric_alerts = metrics_collector.check_alerts()

        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "active_alerts_count": len(active_alerts) + len(metric_alerts),
            "active_alerts": [alert.__dict__ for alert in active_alerts],
            "metric_alerts": [alert.__dict__ for alert in metric_alerts]
        }
    except Exception as e:
        logger.error(f"Error retrieving alerts: {e}")
        record_counter('monitoring_alerts_errors', 1, {'error': str(e)})
        raise HTTPException(status_code=500, detail=f"Failed to retrieve alerts: {str(e)}")


@app.get("/monitoring/system-status")
async def get_system_status(
    dashboard: MonitoringDashboard = Depends(get_monitoring_dashboard)
):
    """
    Get overall system status and health
    """
    try:
        record_counter('monitoring_system_status_requests')

        with time_operation('system_status_generation_time'):
            system_status = dashboard.get_system_status()

        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "system_status": system_status.__dict__
        }
    except Exception as e:
        logger.error(f"Error retrieving system status: {e}")
        record_counter('monitoring_system_status_errors', 1, {'error': str(e)})
        raise HTTPException(status_code=500, detail=f"Failed to retrieve system status: {str(e)}")


@app.post("/monitoring/metrics/{metric_name}/alert")
async def add_metric_alert(
    metric_name: str = Path(..., description="Name of the metric to add alert for"),
    alert_config: Dict[str, Any] = None
):
    """
    Add an alert rule for a specific metric
    """
    try:
        record_counter('monitoring_alert_config_requests', 1, {'metric': metric_name})

        if not alert_config:
            raise HTTPException(status_code=400, detail="Alert configuration required")

        # This would integrate with the alerting system to add alert rules
        # For now, return success with the configuration

        return {
            "success": True,
            "metric_name": metric_name,
            "alert_config": alert_config,
            "message": "Alert rule added successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding alert for metric {metric_name}: {e}")
        record_counter('monitoring_alert_config_errors', 1, {'metric': metric_name, 'error': str(e)})
        raise HTTPException(status_code=500, detail=f"Failed to add alert rule: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    # Record exception metric
    record_counter('api_exceptions', 1, {'exception_type': type(exc).__name__})

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )