"""Real-Time Monitoring Dashboard

Interactive monitoring dashboard with real-time metrics visualization,
alert management, and system health overview for operational visibility.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from .metrics_collector import MetricsCollector, MetricSummary, metrics_collector
from .alerting_system import AlertingSystem, Alert, alerting_system

logger = logging.getLogger(__name__)


class DashboardTheme(Enum):
    """Dashboard visual themes"""
    LIGHT = "light"
    DARK = "dark"
    HIGH_CONTRAST = "high_contrast"


class ChartType(Enum):
    """Types of charts for metrics visualization"""
    LINE = "line"
    BAR = "bar"
    GAUGE = "gauge"
    PIE = "pie"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"


@dataclass
class DashboardWidget:
    """Configuration for dashboard widget"""
    id: str
    title: str
    chart_type: ChartType
    metric_name: str
    time_range_minutes: int = 60
    refresh_interval_seconds: int = 30
    labels_filter: Optional[Dict[str, str]] = None
    threshold_lines: Optional[List[Dict[str, Any]]] = None
    size: Tuple[int, int] = (400, 300)  # width, height
    position: Tuple[int, int] = (0, 0)  # x, y
    enabled: bool = True


@dataclass
class DashboardLayout:
    """Dashboard layout configuration"""
    name: str
    description: str
    widgets: List[DashboardWidget]
    theme: DashboardTheme = DashboardTheme.DARK
    auto_refresh: bool = True
    grid_size: Tuple[int, int] = (1200, 800)


@dataclass
class SystemStatus:
    """Overall system status"""
    status: str  # "healthy", "warning", "critical", "unknown"
    uptime_seconds: float
    total_requests: int
    success_rate: float
    error_rate: float
    response_time_p95: float
    active_alerts: int
    data_sources_healthy: int
    data_sources_total: int
    last_update: datetime


class MonitoringDashboard:
    """Real-time monitoring dashboard with interactive widgets"""

    def __init__(self,
                 metrics_collector: Optional[MetricsCollector] = None,
                 alerting_system: Optional[AlertingSystem] = None):
        """Initialize monitoring dashboard

        Args:
            metrics_collector: Metrics collector instance
            alerting_system: Alerting system instance
        """
        self.metrics_collector = metrics_collector or metrics_collector
        self.alerting_system = alerting_system or alerting_system

        # Dashboard state
        self.layouts: Dict[str, DashboardLayout] = {}
        self.current_layout: Optional[str] = None
        self.system_start_time = datetime.now()

        # Widget data cache
        self.widget_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl_seconds = 30

        # Real-time data
        self.real_time_data: Dict[str, List[Dict[str, Any]]] = {}
        self.max_real_time_points = 1000

        # Background tasks
        self.update_task: Optional[asyncio.Task] = None
        self.is_running = False

        # Default layouts
        self._setup_default_layouts()

        logger.info("Monitoring dashboard initialized")

    def _setup_default_layouts(self) -> None:
        """Setup default dashboard layouts"""
        # System Overview Layout
        system_widgets = [
            DashboardWidget(
                id="system_status",
                title="System Status",
                chart_type=ChartType.GAUGE,
                metric_name="system_health_score",
                position=(0, 0),
                size=(300, 200)
            ),
            DashboardWidget(
                id="response_time",
                title="Response Time (ms)",
                chart_type=ChartType.LINE,
                metric_name="response_time_ms",
                position=(300, 0),
                size=(600, 200)
            ),
            DashboardWidget(
                id="request_rate",
                title="Requests per Second",
                chart_type=ChartType.LINE,
                metric_name="requests_per_second",
                position=(0, 200),
                size=(450, 200)
            ),
            DashboardWidget(
                id="error_rate",
                title="Error Rate (%)",
                chart_type=ChartType.LINE,
                metric_name="error_rate_percent",
                position=(450, 200),
                size=(450, 200)
            ),
            DashboardWidget(
                id="data_sources",
                title="Data Sources Health",
                chart_type=ChartType.PIE,
                metric_name="data_source_health",
                position=(0, 400),
                size=(400, 300)
            ),
            DashboardWidget(
                id="active_alerts",
                title="Active Alerts",
                chart_type=ChartType.BAR,
                metric_name="alert_count_by_severity",
                position=(400, 400),
                size=(500, 300)
            )
        ]

        self.layouts["system_overview"] = DashboardLayout(
            name="System Overview",
            description="High-level system health and performance metrics",
            widgets=system_widgets
        )

        # Performance Details Layout
        performance_widgets = [
            DashboardWidget(
                id="response_time_histogram",
                title="Response Time Distribution",
                chart_type=ChartType.HISTOGRAM,
                metric_name="response_time_ms",
                position=(0, 0),
                size=(600, 300)
            ),
            DashboardWidget(
                id="throughput_trend",
                title="Throughput Trend",
                chart_type=ChartType.LINE,
                metric_name="throughput",
                position=(600, 0),
                size=(600, 300)
            ),
            DashboardWidget(
                id="memory_usage",
                title="Memory Usage",
                chart_type=ChartType.LINE,
                metric_name="memory_usage_mb",
                position=(0, 300),
                size=(400, 250)
            ),
            DashboardWidget(
                id="cpu_usage",
                title="CPU Usage (%)",
                chart_type=ChartType.LINE,
                metric_name="cpu_usage_percent",
                position=(400, 300),
                size=(400, 250)
            ),
            DashboardWidget(
                id="cache_hit_rate",
                title="Cache Hit Rate (%)",
                chart_type=ChartType.GAUGE,
                metric_name="cache_hit_rate_percent",
                position=(800, 300),
                size=(300, 250)
            )
        ]

        self.layouts["performance_details"] = DashboardLayout(
            name="Performance Details",
            description="Detailed performance metrics and resource utilization",
            widgets=performance_widgets
        )

        # Data Sources Layout
        sources_widgets = [
            DashboardWidget(
                id="source_response_times",
                title="Source Response Times",
                chart_type=ChartType.BAR,
                metric_name="source_response_time_ms",
                position=(0, 0),
                size=(600, 300)
            ),
            DashboardWidget(
                id="source_success_rates",
                title="Source Success Rates",
                chart_type=ChartType.BAR,
                metric_name="source_success_rate",
                position=(600, 0),
                size=(600, 300)
            ),
            DashboardWidget(
                id="api_usage",
                title="API Usage by Source",
                chart_type=ChartType.PIE,
                metric_name="api_calls_by_source",
                position=(0, 300),
                size=(400, 300)
            ),
            DashboardWidget(
                id="rate_limits",
                title="Rate Limit Utilization",
                chart_type=ChartType.BAR,
                metric_name="rate_limit_utilization",
                position=(400, 300),
                size=(500, 300)
            ),
            DashboardWidget(
                id="circuit_breaker_status",
                title="Circuit Breaker Status",
                chart_type=ChartType.HEATMAP,
                metric_name="circuit_breaker_states",
                position=(900, 300),
                size=(300, 300)
            )
        ]

        self.layouts["data_sources"] = DashboardLayout(
            name="Data Sources",
            description="Data source performance and health monitoring",
            widgets=sources_widgets
        )

        # Set default layout
        self.current_layout = "system_overview"

    def add_layout(self, layout: DashboardLayout) -> None:
        """Add custom dashboard layout

        Args:
            layout: Dashboard layout configuration
        """
        self.layouts[layout.name.lower().replace(" ", "_")] = layout
        logger.info(f"Added dashboard layout: {layout.name}")

    def set_current_layout(self, layout_name: str) -> bool:
        """Set current active layout

        Args:
            layout_name: Name of layout to activate

        Returns:
            True if layout was found and set
        """
        if layout_name in self.layouts:
            self.current_layout = layout_name
            logger.info(f"Set current layout: {layout_name}")
            return True
        return False

    async def get_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get data for a specific widget

        Args:
            widget: Widget configuration

        Returns:
            Widget data dictionary
        """
        # Check cache first
        cache_key = f"{widget.id}_{widget.metric_name}"
        if cache_key in self.widget_cache:
            cached_data, cached_time = self.widget_cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < self.cache_ttl_seconds:
                return cached_data

        # Get fresh data
        data = await self._fetch_widget_data(widget)

        # Cache the result
        self.widget_cache[cache_key] = (data, datetime.now())

        return data

    async def _fetch_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Fetch fresh data for widget

        Args:
            widget: Widget configuration

        Returns:
            Widget data dictionary
        """
        time_range = timedelta(minutes=widget.time_range_minutes)

        try:
            # Get metric summary
            summary = self.metrics_collector.get_metric_summary(
                widget.metric_name,
                time_range,
                widget.labels_filter
            )

            if not summary:
                return self._empty_widget_data(widget)

            # Format data based on chart type
            if widget.chart_type == ChartType.LINE:
                return await self._format_line_chart_data(widget, summary)
            elif widget.chart_type == ChartType.BAR:
                return await self._format_bar_chart_data(widget, summary)
            elif widget.chart_type == ChartType.GAUGE:
                return await self._format_gauge_data(widget, summary)
            elif widget.chart_type == ChartType.PIE:
                return await self._format_pie_chart_data(widget, summary)
            elif widget.chart_type == ChartType.HISTOGRAM:
                return await self._format_histogram_data(widget, summary)
            elif widget.chart_type == ChartType.HEATMAP:
                return await self._format_heatmap_data(widget, summary)
            else:
                return self._empty_widget_data(widget)

        except Exception as e:
            logger.error(f"Failed to fetch widget data for {widget.id}: {e}")
            return self._empty_widget_data(widget)

    def _empty_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Return empty widget data structure

        Args:
            widget: Widget configuration

        Returns:
            Empty widget data
        """
        return {
            "widget_id": widget.id,
            "title": widget.title,
            "chart_type": widget.chart_type.value,
            "data": [],
            "labels": [],
            "error": "No data available",
            "last_update": datetime.now().isoformat()
        }

    async def _format_line_chart_data(self, widget: DashboardWidget, summary: MetricSummary) -> Dict[str, Any]:
        """Format data for line chart

        Args:
            widget: Widget configuration
            summary: Metric summary

        Returns:
            Line chart data
        """
        # Get historical points
        time_range = timedelta(minutes=widget.time_range_minutes)
        cutoff_time = datetime.now() - time_range

        points = []
        if widget.metric_name in self.metrics_collector.metrics:
            for metric_value in self.metrics_collector.metrics[widget.metric_name]:
                if metric_value.timestamp > cutoff_time:
                    # Apply label filter
                    if widget.labels_filter:
                        if not all(metric_value.labels.get(k) == v for k, v in widget.labels_filter.items()):
                            continue

                    points.append({
                        "x": metric_value.timestamp.isoformat(),
                        "y": metric_value.value
                    })

        return {
            "widget_id": widget.id,
            "title": widget.title,
            "chart_type": widget.chart_type.value,
            "data": points[-100:],  # Limit to last 100 points
            "current_value": summary.last_value,
            "mean_value": summary.mean_value,
            "min_value": summary.min_value,
            "max_value": summary.max_value,
            "threshold_lines": widget.threshold_lines or [],
            "last_update": datetime.now().isoformat()
        }

    async def _format_bar_chart_data(self, widget: DashboardWidget, summary: MetricSummary) -> Dict[str, Any]:
        """Format data for bar chart

        Args:
            widget: Widget configuration
            summary: Metric summary

        Returns:
            Bar chart data
        """
        # Group data by labels
        label_groups = {}
        time_range = timedelta(minutes=widget.time_range_minutes)
        cutoff_time = datetime.now() - time_range

        if widget.metric_name in self.metrics_collector.metrics:
            for metric_value in self.metrics_collector.metrics[widget.metric_name]:
                if metric_value.timestamp > cutoff_time:
                    # Use first label as grouping key, or 'default' if no labels
                    group_key = list(metric_value.labels.values())[0] if metric_value.labels else "default"

                    if group_key not in label_groups:
                        label_groups[group_key] = []

                    label_groups[group_key].append(metric_value.value)

        # Calculate averages for each group
        bar_data = []
        labels = []
        for group, values in label_groups.items():
            if values:
                labels.append(group)
                bar_data.append(sum(values) / len(values))

        return {
            "widget_id": widget.id,
            "title": widget.title,
            "chart_type": widget.chart_type.value,
            "data": bar_data,
            "labels": labels,
            "last_update": datetime.now().isoformat()
        }

    async def _format_gauge_data(self, widget: DashboardWidget, summary: MetricSummary) -> Dict[str, Any]:
        """Format data for gauge chart

        Args:
            widget: Widget configuration
            summary: Metric summary

        Returns:
            Gauge chart data
        """
        # Determine gauge range based on historical data
        gauge_min = min(0, summary.min_value)
        gauge_max = max(100, summary.max_value * 1.2)

        return {
            "widget_id": widget.id,
            "title": widget.title,
            "chart_type": widget.chart_type.value,
            "value": summary.last_value,
            "min_value": gauge_min,
            "max_value": gauge_max,
            "mean_value": summary.mean_value,
            "threshold_lines": widget.threshold_lines or [],
            "last_update": datetime.now().isoformat()
        }

    async def _format_pie_chart_data(self, widget: DashboardWidget, summary: MetricSummary) -> Dict[str, Any]:
        """Format data for pie chart

        Args:
            widget: Widget configuration
            summary: Metric summary

        Returns:
            Pie chart data
        """
        # Group by labels for pie chart
        return await self._format_bar_chart_data(widget, summary)

    async def _format_histogram_data(self, widget: DashboardWidget, summary: MetricSummary) -> Dict[str, Any]:
        """Format data for histogram

        Args:
            widget: Widget configuration
            summary: Metric summary

        Returns:
            Histogram data
        """
        # Get raw values for histogram
        values = []
        time_range = timedelta(minutes=widget.time_range_minutes)
        cutoff_time = datetime.now() - time_range

        if widget.metric_name in self.metrics_collector.metrics:
            for metric_value in self.metrics_collector.metrics[widget.metric_name]:
                if metric_value.timestamp > cutoff_time:
                    # Apply label filter
                    if widget.labels_filter:
                        if not all(metric_value.labels.get(k) == v for k, v in widget.labels_filter.items()):
                            continue
                    values.append(metric_value.value)

        # Create histogram bins
        if not values:
            return self._empty_widget_data(widget)

        num_bins = 20
        min_val = min(values)
        max_val = max(values)
        bin_width = (max_val - min_val) / num_bins if max_val > min_val else 1

        bins = [0] * num_bins
        bin_labels = []

        for i in range(num_bins):
            bin_start = min_val + i * bin_width
            bin_end = min_val + (i + 1) * bin_width
            bin_labels.append(f"{bin_start:.1f}-{bin_end:.1f}")

        # Count values in each bin
        for value in values:
            if max_val > min_val:
                bin_index = min(int((value - min_val) / bin_width), num_bins - 1)
            else:
                bin_index = 0
            bins[bin_index] += 1

        return {
            "widget_id": widget.id,
            "title": widget.title,
            "chart_type": widget.chart_type.value,
            "data": bins,
            "labels": bin_labels,
            "total_samples": len(values),
            "last_update": datetime.now().isoformat()
        }

    async def _format_heatmap_data(self, widget: DashboardWidget, summary: MetricSummary) -> Dict[str, Any]:
        """Format data for heatmap

        Args:
            widget: Widget configuration
            summary: Metric summary

        Returns:
            Heatmap data
        """
        # For now, return empty heatmap data
        # This would need specific implementation based on use case
        return {
            "widget_id": widget.id,
            "title": widget.title,
            "chart_type": widget.chart_type.value,
            "data": [],
            "x_labels": [],
            "y_labels": [],
            "last_update": datetime.now().isoformat()
        }

    async def get_current_dashboard_data(self) -> Dict[str, Any]:
        """Get data for current dashboard layout

        Returns:
            Complete dashboard data
        """
        if not self.current_layout or self.current_layout not in self.layouts:
            return {"error": "No active layout"}

        layout = self.layouts[self.current_layout]

        # Get system status
        system_status = await self.get_system_status()

        # Get widget data
        widget_data = []
        for widget in layout.widgets:
            if widget.enabled:
                data = await self.get_widget_data(widget)
                widget_data.append(data)

        # Get active alerts
        active_alerts = self.alerting_system.get_active_alerts()

        return {
            "layout": {
                "name": layout.name,
                "description": layout.description,
                "theme": layout.theme.value,
                "grid_size": layout.grid_size,
                "auto_refresh": layout.auto_refresh
            },
            "system_status": asdict(system_status),
            "widgets": widget_data,
            "active_alerts": active_alerts,
            "alert_stats": self.alerting_system.get_notification_stats(),
            "last_update": datetime.now().isoformat()
        }

    async def get_system_status(self) -> SystemStatus:
        """Get overall system status

        Returns:
            System status summary
        """
        uptime = (datetime.now() - self.system_start_time).total_seconds()

        # Get metrics summaries
        response_time_summary = self.metrics_collector.get_metric_summary("response_time_ms", timedelta(minutes=5))
        request_summary = self.metrics_collector.get_metric_summary("requests_total", timedelta(minutes=5))
        error_summary = self.metrics_collector.get_metric_summary("errors_total", timedelta(minutes=5))

        # Calculate rates
        success_rate = 0.95  # Default
        error_rate = 0.05
        response_time_p95 = 0.0

        if response_time_summary:
            response_time_p95 = response_time_summary.percentile_95

        if request_summary and error_summary and request_summary.sum_value > 0:
            error_rate = error_summary.sum_value / request_summary.sum_value
            success_rate = 1.0 - error_rate

        # Get alert count
        active_alerts = len(self.alerting_system.get_active_alerts())

        # Determine overall status
        if active_alerts > 0:
            if any(alert["severity"] in ["critical", "high"] for alert in self.alerting_system.get_active_alerts()):
                status = "critical"
            else:
                status = "warning"
        elif error_rate > 0.1:
            status = "warning"
        elif success_rate > 0.95:
            status = "healthy"
        else:
            status = "warning"

        return SystemStatus(
            status=status,
            uptime_seconds=uptime,
            total_requests=int(request_summary.sum_value) if request_summary else 0,
            success_rate=success_rate,
            error_rate=error_rate,
            response_time_p95=response_time_p95,
            active_alerts=active_alerts,
            data_sources_healthy=4,  # Would come from actual data source manager
            data_sources_total=5,
            last_update=datetime.now()
        )

    async def start_real_time_updates(self) -> None:
        """Start real-time dashboard updates"""
        if self.is_running:
            return

        self.is_running = True
        self.update_task = asyncio.create_task(self._update_loop())
        logger.info("Started real-time dashboard updates")

    async def stop_real_time_updates(self) -> None:
        """Stop real-time dashboard updates"""
        self.is_running = False
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped real-time dashboard updates")

    async def _update_loop(self) -> None:
        """Background update loop for real-time data"""
        while self.is_running:
            try:
                # Update real-time metrics
                await self._update_real_time_metrics()

                # Clear expired cache entries
                self._cleanup_cache()

                await asyncio.sleep(5)  # Update every 5 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dashboard update loop error: {e}")
                await asyncio.sleep(5)

    async def _update_real_time_metrics(self) -> None:
        """Update real-time metrics data"""
        current_time = datetime.now()

        # Get current system metrics
        all_metrics = self.metrics_collector.get_all_metrics()

        # Store in real-time data
        for metric_name, value in all_metrics.get("gauges", {}).items():
            if metric_name not in self.real_time_data:
                self.real_time_data[metric_name] = []

            self.real_time_data[metric_name].append({
                "timestamp": current_time.isoformat(),
                "value": value
            })

            # Limit history
            if len(self.real_time_data[metric_name]) > self.max_real_time_points:
                self.real_time_data[metric_name] = self.real_time_data[metric_name][-500:]

    def _cleanup_cache(self) -> None:
        """Clean up expired cache entries"""
        current_time = datetime.now()
        expired_keys = []

        for cache_key, (_, cached_time) in self.widget_cache.items():
            if (current_time - cached_time).total_seconds() > self.cache_ttl_seconds * 2:
                expired_keys.append(cache_key)

        for key in expired_keys:
            del self.widget_cache[key]

    def get_available_layouts(self) -> List[Dict[str, str]]:
        """Get list of available dashboard layouts

        Returns:
            List of layout information
        """
        return [
            {
                "id": layout_id,
                "name": layout.name,
                "description": layout.description,
                "widget_count": len(layout.widgets)
            }
            for layout_id, layout in self.layouts.items()
        ]

    def export_dashboard_config(self) -> str:
        """Export dashboard configuration as JSON

        Returns:
            JSON string with dashboard configuration
        """
        config = {
            "layouts": {
                layout_id: {
                    "name": layout.name,
                    "description": layout.description,
                    "theme": layout.theme.value,
                    "auto_refresh": layout.auto_refresh,
                    "grid_size": layout.grid_size,
                    "widgets": [
                        {
                            "id": w.id,
                            "title": w.title,
                            "chart_type": w.chart_type.value,
                            "metric_name": w.metric_name,
                            "time_range_minutes": w.time_range_minutes,
                            "refresh_interval_seconds": w.refresh_interval_seconds,
                            "labels_filter": w.labels_filter,
                            "threshold_lines": w.threshold_lines,
                            "size": w.size,
                            "position": w.position,
                            "enabled": w.enabled
                        }
                        for w in layout.widgets
                    ]
                }
                for layout_id, layout in self.layouts.items()
            },
            "current_layout": self.current_layout,
            "export_timestamp": datetime.now().isoformat()
        }

        return json.dumps(config, indent=2)


# Global dashboard instance
monitoring_dashboard = MonitoringDashboard()