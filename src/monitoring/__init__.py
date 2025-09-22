"""
Advanced Monitoring and Alerting System

This module provides comprehensive monitoring capabilities for the Market Data Agent,
including real-time metrics collection, intelligent alerting, and interactive dashboards.

Key Components:
- MetricsCollector: Advanced metrics collection with statistical analysis
- AlertingSystem: Intelligent alerting with multiple notification channels
- MonitoringDashboard: Real-time dashboard with interactive widgets

Usage:
    from monitoring import MetricsCollector, AlertingSystem, MonitoringDashboard

    # Initialize monitoring components
    metrics = MetricsCollector()
    alerts = AlertingSystem()
    dashboard = MonitoringDashboard(metrics, alerts)

    # Record metrics
    metrics.record_counter('api_requests')
    metrics.record_gauge('response_time', 150.5)

    # Check for alerts
    alerts.check_alerts(metrics.get_current_metrics())

    # Generate dashboard
    dashboard_data = dashboard.generate_dashboard('overview')
"""

from .metrics_collector import (
    MetricsCollector,
    MetricType,
    MetricValue,
    MetricSummary,
    AlertRule,
    Alert,
    AlertSeverity
)

from .alerting_system import (
    AlertingSystem,
    NotificationChannel,
    NotificationConfig,
    EscalationRule,
    AlertCorrelation
)

from .dashboard import (
    MonitoringDashboard,
    ChartType,
    DashboardWidget,
    DashboardLayout,
    SystemStatus
)

__version__ = "1.0.0"
__author__ = "Market Data Agent Team"

# Default monitoring instance for convenience
_default_metrics = None
_default_alerts = None
_default_dashboard = None

def get_default_metrics() -> MetricsCollector:
    """Get the default metrics collector instance."""
    global _default_metrics
    if _default_metrics is None:
        _default_metrics = MetricsCollector()
    return _default_metrics

def get_default_alerts() -> AlertingSystem:
    """Get the default alerting system instance."""
    global _default_alerts
    if _default_alerts is None:
        _default_alerts = AlertingSystem()
    return _default_alerts

def get_default_dashboard() -> MonitoringDashboard:
    """Get the default dashboard instance."""
    global _default_dashboard
    if _default_dashboard is None:
        metrics = get_default_metrics()
        alerts = get_default_alerts()
        _default_dashboard = MonitoringDashboard(metrics, alerts)
    return _default_dashboard

# Convenience functions for quick monitoring
def record_counter(name: str, value: int = 1, tags: dict = None) -> None:
    """Record a counter metric using the default collector."""
    get_default_metrics().record_counter(name, value, tags)

def record_gauge(name: str, value: float, tags: dict = None) -> None:
    """Record a gauge metric using the default collector."""
    get_default_metrics().record_gauge(name, value, tags)

def record_timer(name: str, duration: float, tags: dict = None) -> None:
    """Record a timer metric using the default collector."""
    get_default_metrics().record_timer(name, duration, tags)

def time_operation(name: str, tags: dict = None):
    """Context manager to time an operation using the default collector."""
    return get_default_metrics().time_operation(name, tags)

def check_alerts() -> list:
    """Check for alerts using the default alerting system."""
    metrics = get_default_metrics().get_current_metrics()
    return get_default_alerts().check_alerts(metrics)

def get_dashboard(layout_name: str = 'overview') -> dict:
    """Generate dashboard data using the default dashboard."""
    return get_default_dashboard().generate_dashboard(layout_name)

__all__ = [
    # Main classes
    'MetricsCollector',
    'AlertingSystem',
    'MonitoringDashboard',

    # Enums and types
    'MetricType',
    'AlertSeverity',
    'NotificationChannel',
    'ChartType',

    # Data classes
    'MetricValue',
    'MetricSummary',
    'AlertRule',
    'Alert',
    'NotificationConfig',
    'EscalationRule',
    'AlertCorrelation',
    'DashboardWidget',
    'DashboardLayout',
    'SystemStatus',

    # Utilities

    # Convenience functions
    'get_default_metrics',
    'get_default_alerts',
    'get_default_dashboard',
    'record_counter',
    'record_gauge',
    'record_timer',
    'time_operation',
    'check_alerts',
    'get_dashboard'
]