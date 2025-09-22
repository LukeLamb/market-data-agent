"""
Test suite for the MonitoringDashboard component

Tests cover all aspects of dashboard functionality including:
- Dashboard layout management and customization
- Widget creation and configuration
- Multiple chart types (line, bar, gauge, pie, heatmap, histogram)
- Real-time data visualization and updates
- System status monitoring
- Dashboard data generation and caching
- Performance and scalability
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import json

from src.monitoring.dashboard import (
    MonitoringDashboard,
    ChartType,
    DashboardWidget,
    DashboardLayout,
    SystemStatus
)

from src.monitoring.metrics_collector import (
    MetricsCollector,
    MetricType,
    MetricValue,
    Alert,
    AlertSeverity
)

from src.monitoring.alerting_system import AlertingSystem


class TestMonitoringDashboard:
    """Test suite for MonitoringDashboard functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.metrics_collector = MetricsCollector()
        self.alerting_system = AlertingSystem()
        self.dashboard = MonitoringDashboard(
            self.metrics_collector,
            self.alerting_system
        )

    def test_initialization(self):
        """Test dashboard initialization."""
        assert self.dashboard.metrics_collector is not None
        assert self.dashboard.alerting_system is not None
        assert len(self.dashboard.layouts) > 0  # Should have default layouts
        assert 'overview' in self.dashboard.layouts
        assert 'performance' in self.dashboard.layouts
        assert 'alerts' in self.dashboard.layouts

    def test_default_layouts_creation(self):
        """Test default layout creation."""
        # Check overview layout
        overview_layout = self.dashboard.layouts['overview']
        assert overview_layout.name == 'overview'
        assert len(overview_layout.widgets) > 0

        # Check performance layout
        performance_layout = self.dashboard.layouts['performance']
        assert performance_layout.name == 'performance'
        assert len(performance_layout.widgets) > 0

        # Check alerts layout
        alerts_layout = self.dashboard.layouts['alerts']
        assert alerts_layout.name == 'alerts'
        assert len(alerts_layout.widgets) > 0

    def test_add_custom_layout(self):
        """Test adding custom dashboard layout."""
        # Create custom widgets
        custom_widgets = [
            DashboardWidget(
                id='custom_metric',
                title='Custom Metric',
                chart_type=ChartType.LINE,
                metric_names=['custom_metric'],
                size={'width': 6, 'height': 4}
            )
        ]

        # Create custom layout
        custom_layout = DashboardLayout(
            name='custom',
            title='Custom Dashboard',
            widgets=custom_widgets,
            refresh_interval=30
        )

        # Add layout to dashboard
        self.dashboard.add_layout(custom_layout)

        # Verify layout was added
        assert 'custom' in self.dashboard.layouts
        assert self.dashboard.layouts['custom'].title == 'Custom Dashboard'

    def test_widget_creation(self):
        """Test dashboard widget creation."""
        widget = DashboardWidget(
            id='test_widget',
            title='Test Widget',
            chart_type=ChartType.BAR,
            metric_names=['test_metric'],
            size={'width': 8, 'height': 6},
            options={
                'show_legend': True,
                'color_scheme': 'blue'
            }
        )

        assert widget.id == 'test_widget'
        assert widget.title == 'Test Widget'
        assert widget.chart_type == ChartType.BAR
        assert widget.metric_names == ['test_metric']
        assert widget.size == {'width': 8, 'height': 6}
        assert widget.options['show_legend'] is True

    def test_generate_dashboard_overview(self):
        """Test generating overview dashboard."""
        # Add some test metrics
        self.metrics_collector.record_counter('api_requests', 100)
        self.metrics_collector.record_gauge('cpu_usage', 75.5)
        self.metrics_collector.record_timer('response_time', 0.150)

        # Generate dashboard
        dashboard_data = self.dashboard.generate_dashboard('overview')

        # Verify dashboard structure
        assert 'layout' in dashboard_data
        assert 'widgets' in dashboard_data
        assert 'system_status' in dashboard_data
        assert 'last_updated' in dashboard_data

        # Verify layout information
        layout = dashboard_data['layout']
        assert layout['name'] == 'overview'
        assert 'title' in layout

        # Verify widgets data
        widgets = dashboard_data['widgets']
        assert len(widgets) > 0

    def test_generate_dashboard_performance(self):
        """Test generating performance dashboard."""
        # Add performance metrics
        response_times = [0.100, 0.150, 0.200, 0.120, 0.180]
        for rt in response_times:
            self.metrics_collector.record_timer('response_time', rt)

        # Generate performance dashboard
        dashboard_data = self.dashboard.generate_dashboard('performance')

        # Verify dashboard structure
        assert 'layout' in dashboard_data
        assert dashboard_data['layout']['name'] == 'performance'

        # Should contain performance-specific widgets
        widgets = dashboard_data['widgets']
        widget_titles = [w.get('title', '') for w in widgets]

        # Should have widgets related to performance metrics
        assert len(widgets) > 0

    def test_generate_dashboard_alerts(self):
        """Test generating alerts dashboard."""
        # Create test alerts
        alerts = [
            Alert(
                name='high_cpu',
                severity=AlertSeverity.HIGH,
                message='High CPU usage detected',
                metric_name='cpu_usage',
                current_value=95.0,
                threshold=80.0
            ),
            Alert(
                name='low_memory',
                severity=AlertSeverity.MEDIUM,
                message='Low memory available',
                metric_name='memory_usage',
                current_value=85.0,
                threshold=90.0
            )
        ]

        # Mock alerting system to return test alerts
        with patch.object(self.alerting_system, 'get_active_alerts', return_value=alerts):
            dashboard_data = self.dashboard.generate_dashboard('alerts')

        # Verify alerts dashboard structure
        assert 'layout' in dashboard_data
        assert dashboard_data['layout']['name'] == 'alerts'

        # Should contain alert-specific data
        widgets = dashboard_data['widgets']
        assert len(widgets) > 0

    def test_widget_data_generation_line_chart(self):
        """Test line chart widget data generation."""
        # Add time series data
        for i in range(10):
            self.metrics_collector.record_histogram('metric_series', i * 10)

        # Create line chart widget
        widget = DashboardWidget(
            id='line_chart',
            title='Line Chart',
            chart_type=ChartType.LINE,
            metric_names=['metric_series']
        )

        # Generate widget data
        widget_data = self.dashboard.generate_widget_data(widget)

        # Verify line chart data structure
        assert 'type' in widget_data
        assert widget_data['type'] == 'line'
        assert 'data' in widget_data
        assert 'labels' in widget_data['data']
        assert 'datasets' in widget_data['data']

    def test_widget_data_generation_bar_chart(self):
        """Test bar chart widget data generation."""
        # Add categorical data
        categories = ['A', 'B', 'C', 'D']
        for i, cat in enumerate(categories):
            self.metrics_collector.record_counter(f'category_{cat}', (i + 1) * 10)

        # Create bar chart widget
        widget = DashboardWidget(
            id='bar_chart',
            title='Bar Chart',
            chart_type=ChartType.BAR,
            metric_names=[f'category_{cat}' for cat in categories]
        )

        # Generate widget data
        widget_data = self.dashboard.generate_widget_data(widget)

        # Verify bar chart data structure
        assert widget_data['type'] == 'bar'
        assert 'data' in widget_data
        assert len(widget_data['data']['labels']) == len(categories)

    def test_widget_data_generation_gauge_chart(self):
        """Test gauge chart widget data generation."""
        # Add gauge metric
        self.metrics_collector.record_gauge('cpu_usage', 75.5)

        # Create gauge widget
        widget = DashboardWidget(
            id='cpu_gauge',
            title='CPU Usage',
            chart_type=ChartType.GAUGE,
            metric_names=['cpu_usage'],
            options={
                'min_value': 0,
                'max_value': 100,
                'thresholds': [50, 80, 95]
            }
        )

        # Generate widget data
        widget_data = self.dashboard.generate_widget_data(widget)

        # Verify gauge data structure
        assert widget_data['type'] == 'gauge'
        assert 'value' in widget_data
        assert widget_data['value'] == 75.5
        assert 'options' in widget_data

    def test_widget_data_generation_pie_chart(self):
        """Test pie chart widget data generation."""
        # Add data for pie chart
        pie_data = {'success': 80, 'error': 15, 'timeout': 5}
        for category, count in pie_data.items():
            self.metrics_collector.record_counter(f'requests_{category}', count)

        # Create pie chart widget
        widget = DashboardWidget(
            id='pie_chart',
            title='Request Status Distribution',
            chart_type=ChartType.PIE,
            metric_names=list(pie_data.keys()),
            options={'show_percentages': True}
        )

        # Generate widget data
        widget_data = self.dashboard.generate_widget_data(widget)

        # Verify pie chart data structure
        assert widget_data['type'] == 'pie'
        assert 'data' in widget_data
        assert len(widget_data['data']['labels']) == len(pie_data)

    def test_widget_data_generation_histogram(self):
        """Test histogram widget data generation."""
        # Add histogram data
        for i in range(100):
            self.metrics_collector.record_histogram('response_times', i / 10.0)

        # Create histogram widget
        widget = DashboardWidget(
            id='histogram',
            title='Response Time Distribution',
            chart_type=ChartType.HISTOGRAM,
            metric_names=['response_times'],
            options={'bins': 10}
        )

        # Generate widget data
        widget_data = self.dashboard.generate_widget_data(widget)

        # Verify histogram data structure
        assert widget_data['type'] == 'histogram'
        assert 'data' in widget_data
        assert 'bins' in widget_data['data']

    def test_system_status_generation(self):
        """Test system status generation."""
        # Add system metrics
        self.metrics_collector.record_gauge('cpu_usage', 65.0)
        self.metrics_collector.record_gauge('memory_usage', 70.0)
        self.metrics_collector.record_gauge('disk_usage', 45.0)
        self.metrics_collector.record_counter('api_requests', 1000)

        # Generate system status
        system_status = self.dashboard.get_system_status()

        # Verify system status structure
        assert system_status.overall_health in ['healthy', 'warning', 'critical']
        assert 'cpu' in system_status.component_status
        assert 'memory' in system_status.component_status
        assert 'disk' in system_status.component_status

        # Verify metrics are included
        assert 'cpu_usage' in system_status.key_metrics
        assert system_status.key_metrics['cpu_usage'] == 65.0

    def test_dashboard_caching(self):
        """Test dashboard data caching."""
        # Enable caching
        self.dashboard.cache_ttl = 60  # 60 seconds

        # Generate dashboard (should cache result)
        dashboard_data1 = self.dashboard.generate_dashboard('overview')

        # Generate again immediately (should use cache)
        dashboard_data2 = self.dashboard.generate_dashboard('overview')

        # Should be identical due to caching
        assert dashboard_data1['last_updated'] == dashboard_data2['last_updated']

    def test_dashboard_cache_expiry(self):
        """Test dashboard cache expiry."""
        # Set very short cache TTL
        self.dashboard.cache_ttl = 0.1  # 0.1 seconds

        # Generate dashboard
        dashboard_data1 = self.dashboard.generate_dashboard('overview')

        # Wait for cache to expire
        import time
        time.sleep(0.2)

        # Generate again (should create new data)
        dashboard_data2 = self.dashboard.generate_dashboard('overview')

        # Should be different due to cache expiry
        assert dashboard_data1['last_updated'] != dashboard_data2['last_updated']

    def test_real_time_updates(self):
        """Test real-time dashboard updates."""
        # Add initial metrics
        self.metrics_collector.record_gauge('live_metric', 50.0)

        # Generate initial dashboard
        dashboard_data1 = self.dashboard.generate_dashboard('overview')

        # Update metrics
        self.metrics_collector.record_gauge('live_metric', 75.0)

        # Clear cache to force update
        self.dashboard.widget_cache.clear()

        # Generate updated dashboard
        dashboard_data2 = self.dashboard.generate_dashboard('overview')

        # Should reflect updated metrics
        assert dashboard_data1 != dashboard_data2

    def test_widget_filtering(self):
        """Test widget filtering by metric availability."""
        # Create widget for non-existent metric
        widget = DashboardWidget(
            id='missing_metric',
            title='Missing Metric',
            chart_type=ChartType.LINE,
            metric_names=['non_existent_metric']
        )

        # Generate widget data
        widget_data = self.dashboard.generate_widget_data(widget)

        # Should handle missing metrics gracefully
        assert 'error' in widget_data or len(widget_data.get('data', {}).get('datasets', [])) == 0

    def test_dashboard_export(self):
        """Test dashboard data export functionality."""
        # Add test data
        self.metrics_collector.record_counter('exports', 100)
        self.metrics_collector.record_gauge('export_rate', 25.5)

        # Generate dashboard
        dashboard_data = self.dashboard.generate_dashboard('overview')

        # Export to JSON
        json_export = json.dumps(dashboard_data, default=str)

        # Verify export is valid JSON
        parsed_data = json.loads(json_export)
        assert 'layout' in parsed_data
        assert 'widgets' in parsed_data

    def test_custom_widget_options(self):
        """Test custom widget options and configurations."""
        # Create widget with custom options
        custom_options = {
            'color_scheme': 'custom',
            'animation': True,
            'show_grid': False,
            'custom_threshold': 90.0
        }

        widget = DashboardWidget(
            id='custom_widget',
            title='Custom Widget',
            chart_type=ChartType.GAUGE,
            metric_names=['test_metric'],
            options=custom_options
        )

        # Add test data
        self.metrics_collector.record_gauge('test_metric', 85.0)

        # Generate widget data
        widget_data = self.dashboard.generate_widget_data(widget)

        # Verify custom options are preserved
        assert 'options' in widget_data
        widget_options = widget_data['options']
        assert widget_options.get('color_scheme') == 'custom'
        assert widget_options.get('animation') is True
        assert widget_options.get('show_grid') is False

    def test_multi_metric_widgets(self):
        """Test widgets with multiple metrics."""
        # Add multiple related metrics
        self.metrics_collector.record_gauge('cpu_user', 30.0)
        self.metrics_collector.record_gauge('cpu_system', 20.0)
        self.metrics_collector.record_gauge('cpu_iowait', 15.0)

        # Create multi-metric widget
        widget = DashboardWidget(
            id='cpu_breakdown',
            title='CPU Usage Breakdown',
            chart_type=ChartType.BAR,
            metric_names=['cpu_user', 'cpu_system', 'cpu_iowait']
        )

        # Generate widget data
        widget_data = self.dashboard.generate_widget_data(widget)

        # Verify all metrics are included
        assert 'data' in widget_data
        datasets = widget_data['data'].get('datasets', [])
        assert len(datasets) > 0

    def test_dashboard_performance(self):
        """Test dashboard generation performance."""
        # Add many metrics for performance testing
        for i in range(1000):
            self.metrics_collector.record_counter(f'metric_{i}', i)
            self.metrics_collector.record_gauge(f'gauge_{i}', float(i))

        # Measure dashboard generation time
        import time
        start_time = time.time()

        dashboard_data = self.dashboard.generate_dashboard('overview')

        end_time = time.time()
        generation_time = end_time - start_time

        # Should generate dashboard reasonably quickly (< 1 second)
        assert generation_time < 1.0

        # Verify dashboard was generated successfully
        assert 'widgets' in dashboard_data
        assert len(dashboard_data['widgets']) > 0

    def test_alert_integration(self):
        """Test dashboard integration with alerting system."""
        # Create test alerts
        alerts = [
            Alert(
                name='test_alert_1',
                severity=AlertSeverity.HIGH,
                message='Test alert 1',
                metric_name='test_metric',
                current_value=100.0,
                threshold=80.0
            ),
            Alert(
                name='test_alert_2',
                severity=AlertSeverity.CRITICAL,
                message='Test alert 2',
                metric_name='critical_metric',
                current_value=200.0,
                threshold=150.0
            )
        ]

        # Mock alerting system to return alerts
        with patch.object(self.alerting_system, 'get_active_alerts', return_value=alerts):
            # Generate dashboard with alerts
            dashboard_data = self.dashboard.generate_dashboard('alerts')

            # Verify alerts are included in system status
            system_status = dashboard_data['system_status']
            assert system_status['active_alerts'] >= len(alerts)

    def test_layout_validation(self):
        """Test dashboard layout validation."""
        # Test invalid layout (missing required fields)
        with pytest.raises((ValueError, AttributeError)):
            invalid_layout = DashboardLayout(
                name='',  # Empty name should be invalid
                title='Invalid Layout',
                widgets=[],
                refresh_interval=30
            )
            self.dashboard.add_layout(invalid_layout)

    def test_widget_size_validation(self):
        """Test widget size validation."""
        # Test valid widget sizes
        valid_widget = DashboardWidget(
            id='valid_widget',
            title='Valid Widget',
            chart_type=ChartType.LINE,
            metric_names=['test_metric'],
            size={'width': 6, 'height': 4}  # Valid grid size
        )

        # Should not raise exception
        assert valid_widget.size['width'] == 6
        assert valid_widget.size['height'] == 4


if __name__ == '__main__':
    pytest.main([__file__])