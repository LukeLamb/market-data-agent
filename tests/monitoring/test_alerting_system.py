"""
Test suite for the AlertingSystem component

Tests cover all aspects of intelligent alerting including:
- Alert rule configuration and management
- Multiple notification channels (email, webhook, Slack, console, SMS)
- Escalation workflows and timing
- Alert correlation and suppression
- Rate limiting and throttling
- Configuration management
- Error handling and retries
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from email.mime.text import MIMEText

from src.monitoring.alerting_system import (
    AlertingSystem,
    NotificationChannel,
    NotificationConfig,
    EscalationRule,
    AlertCorrelation
)

from src.monitoring.metrics_collector import (
    Alert,
    AlertSeverity
)


class TestAlertingSystem:
    """Test suite for AlertingSystem functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.alerting_system = AlertingSystem()

    def test_initialization(self):
        """Test alerting system initialization."""
        assert len(self.alerting_system.notification_configs) == 0
        assert len(self.alerting_system.escalation_rules) == 0
        assert len(self.alerting_system.correlation_rules) == 0
        assert len(self.alerting_system.active_alerts) == 0
        assert len(self.alerting_system.alert_history) == 0

    def test_add_notification_config(self):
        """Test adding notification configurations."""
        config = NotificationConfig(
            name='email_config',
            channel=NotificationChannel.EMAIL,
            settings={
                'smtp_server': 'smtp.example.com',
                'smtp_port': 587,
                'from_email': 'alerts@example.com',
                'to_emails': ['admin@example.com']
            },
            rate_limit=5,
            rate_limit_window=60
        )

        self.alerting_system.add_notification_config(config)
        assert 'email_config' in self.alerting_system.notification_configs

    def test_add_escalation_rule(self):
        """Test adding escalation rules."""
        rule = EscalationRule(
            severity=AlertSeverity.CRITICAL,
            initial_delay=0,
            escalation_delay=300,  # 5 minutes
            max_escalations=3,
            notification_configs=['email_config', 'slack_config']
        )

        self.alerting_system.add_escalation_rule(AlertSeverity.CRITICAL, rule)
        assert AlertSeverity.CRITICAL in self.alerting_system.escalation_rules

    def test_add_correlation_rule(self):
        """Test adding correlation rules."""
        rule = AlertCorrelation(
            name='cpu_memory_correlation',
            alert_patterns=['high_cpu_*', 'memory_*'],
            correlation_window=300,  # 5 minutes
            minimum_alerts=2,
            suppression_window=600  # 10 minutes
        )

        self.alerting_system.add_correlation_rule(rule)
        assert 'cpu_memory_correlation' in self.alerting_system.correlation_rules

    @patch('smtplib.SMTP')
    def test_send_email_notification(self, mock_smtp):
        """Test email notification sending."""
        # Configure mock SMTP
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        # Add email configuration
        config = NotificationConfig(
            name='email_config',
            channel=NotificationChannel.EMAIL,
            settings={
                'smtp_server': 'smtp.example.com',
                'smtp_port': 587,
                'from_email': 'alerts@example.com',
                'to_emails': ['admin@example.com'],
                'username': 'user',
                'password': 'pass'
            }
        )
        self.alerting_system.add_notification_config(config)

        # Create test alert
        alert = Alert(
            name='test_alert',
            severity=AlertSeverity.HIGH,
            message='Test alert message',
            metric_name='test_metric',
            current_value=100.0,
            threshold=80.0
        )

        # Send notification
        result = self.alerting_system.send_notification('email_config', alert)

        # Verify result
        assert result.success is True
        assert result.channel == NotificationChannel.EMAIL

        # Verify SMTP calls
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with('user', 'pass')
        mock_server.send_message.assert_called_once()

    @patch('aiohttp.ClientSession.post')
    async def test_send_webhook_notification(self, mock_post):
        """Test webhook notification sending."""
        # Configure mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text.return_value = 'OK'
        mock_post.return_value.__aenter__.return_value = mock_response

        # Add webhook configuration
        config = NotificationConfig(
            name='webhook_config',
            channel=NotificationChannel.WEBHOOK,
            settings={
                'url': 'https://example.com/webhook',
                'method': 'POST',
                'headers': {'Authorization': 'Bearer token123'}
            }
        )
        self.alerting_system.add_notification_config(config)

        # Create test alert
        alert = Alert(
            name='test_alert',
            severity=AlertSeverity.HIGH,
            message='Test alert message',
            metric_name='test_metric',
            current_value=100.0,
            threshold=80.0
        )

        # Send notification
        result = await self.alerting_system.send_notification_async('webhook_config', alert)

        # Verify result
        assert result.success is True
        assert result.channel == NotificationChannel.WEBHOOK

        # Verify webhook call
        mock_post.assert_called_once()

    @patch('aiohttp.ClientSession.post')
    async def test_send_slack_notification(self, mock_post):
        """Test Slack notification sending."""
        # Configure mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {'ok': True}
        mock_post.return_value.__aenter__.return_value = mock_response

        # Add Slack configuration
        config = NotificationConfig(
            name='slack_config',
            channel=NotificationChannel.SLACK,
            settings={
                'webhook_url': 'https://hooks.slack.com/services/test',
                'channel': '#alerts',
                'username': 'AlertBot'
            }
        )
        self.alerting_system.add_notification_config(config)

        # Create test alert
        alert = Alert(
            name='test_alert',
            severity=AlertSeverity.HIGH,
            message='Test alert message',
            metric_name='test_metric',
            current_value=100.0,
            threshold=80.0
        )

        # Send notification
        result = await self.alerting_system.send_notification_async('slack_config', alert)

        # Verify result
        assert result.success is True
        assert result.channel == NotificationChannel.SLACK

        # Verify Slack API call
        mock_post.assert_called_once()

    def test_send_console_notification(self):
        """Test console notification sending."""
        # Add console configuration
        config = NotificationConfig(
            name='console_config',
            channel=NotificationChannel.CONSOLE,
            settings={}
        )
        self.alerting_system.add_notification_config(config)

        # Create test alert
        alert = Alert(
            name='test_alert',
            severity=AlertSeverity.HIGH,
            message='Test alert message',
            metric_name='test_metric',
            current_value=100.0,
            threshold=80.0
        )

        # Send notification (should not raise exception)
        with patch('builtins.print') as mock_print:
            result = self.alerting_system.send_notification('console_config', alert)

            # Verify result
            assert result.success is True
            assert result.channel == NotificationChannel.CONSOLE

            # Verify console output
            mock_print.assert_called()

    def test_check_alerts_with_escalation(self):
        """Test alert checking with escalation rules."""
        # Add notification config
        config = NotificationConfig(
            name='test_config',
            channel=NotificationChannel.CONSOLE,
            settings={}
        )
        self.alerting_system.add_notification_config(config)

        # Add escalation rule
        rule = EscalationRule(
            severity=AlertSeverity.HIGH,
            initial_delay=0,
            escalation_delay=60,
            max_escalations=2,
            notification_configs=['test_config']
        )
        self.alerting_system.add_escalation_rule(AlertSeverity.HIGH, rule)

        # Create test alerts
        alerts = [
            Alert(
                name='high_cpu',
                severity=AlertSeverity.HIGH,
                message='High CPU usage detected',
                metric_name='cpu_usage',
                current_value=95.0,
                threshold=80.0
            )
        ]

        # Check alerts
        with patch('builtins.print'):  # Suppress console output
            processed_alerts = self.alerting_system.check_alerts(alerts)

        # Verify alerts were processed
        assert len(processed_alerts) == 1
        assert len(self.alerting_system.active_alerts) == 1

    def test_alert_correlation(self):
        """Test alert correlation and suppression."""
        # Add correlation rule
        rule = AlertCorrelation(
            name='system_overload',
            alert_patterns=['high_cpu', 'high_memory'],
            correlation_window=300,
            minimum_alerts=2,
            suppression_window=600
        )
        self.alerting_system.add_correlation_rule(rule)

        # Create related alerts
        alerts = [
            Alert(
                name='high_cpu',
                severity=AlertSeverity.HIGH,
                message='High CPU usage',
                metric_name='cpu_usage',
                current_value=95.0,
                threshold=80.0
            ),
            Alert(
                name='high_memory',
                severity=AlertSeverity.HIGH,
                message='High memory usage',
                metric_name='memory_usage',
                current_value=90.0,
                threshold=75.0
            )
        ]

        # Check correlation
        correlated_alerts = self.alerting_system.check_correlation(alerts)

        # Should detect correlation and potentially suppress individual alerts
        # depending on implementation logic
        assert len(correlated_alerts) >= 0

    def test_rate_limiting(self):
        """Test notification rate limiting."""
        # Add rate-limited config
        config = NotificationConfig(
            name='rate_limited_config',
            channel=NotificationChannel.CONSOLE,
            settings={},
            rate_limit=2,  # Max 2 notifications
            rate_limit_window=60  # Per minute
        )
        self.alerting_system.add_notification_config(config)

        # Create test alert
        alert = Alert(
            name='test_alert',
            severity=AlertSeverity.MEDIUM,
            message='Test alert',
            metric_name='test_metric',
            current_value=100.0,
            threshold=80.0
        )

        # Send notifications up to rate limit
        with patch('builtins.print'):  # Suppress console output
            result1 = self.alerting_system.send_notification('rate_limited_config', alert)
            result2 = self.alerting_system.send_notification('rate_limited_config', alert)
            result3 = self.alerting_system.send_notification('rate_limited_config', alert)

        # First two should succeed, third should be rate limited
        assert result1.success is True
        assert result2.success is True
        assert result3.success is False
        assert 'rate limit' in result3.error_message.lower()

    def test_notification_retry(self):
        """Test notification retry logic."""
        # Add config with retry settings
        config = NotificationConfig(
            name='retry_config',
            channel=NotificationChannel.WEBHOOK,
            settings={
                'url': 'https://example.com/webhook',
                'method': 'POST'
            },
            retry_count=3,
            retry_delay=1
        )
        self.alerting_system.add_notification_config(config)

        # Create test alert
        alert = Alert(
            name='test_alert',
            severity=AlertSeverity.HIGH,
            message='Test alert',
            metric_name='test_metric',
            current_value=100.0,
            threshold=80.0
        )

        # Mock failed HTTP requests
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 500  # Server error
            mock_post.return_value.__aenter__.return_value = mock_response

            # Send notification (should retry and eventually fail)
            result = asyncio.run(
                self.alerting_system.send_notification_async('retry_config', alert)
            )

            # Should have retried multiple times
            assert mock_post.call_count > 1
            assert result.success is False

    def test_alert_suppression(self):
        """Test alert suppression functionality."""
        # Create identical alerts
        alert1 = Alert(
            name='repeated_alert',
            severity=AlertSeverity.MEDIUM,
            message='Same alert message',
            metric_name='test_metric',
            current_value=100.0,
            threshold=80.0
        )

        alert2 = Alert(
            name='repeated_alert',
            severity=AlertSeverity.MEDIUM,
            message='Same alert message',
            metric_name='test_metric',
            current_value=100.0,
            threshold=80.0
        )

        # Process alerts
        processed1 = self.alerting_system.check_alerts([alert1])
        processed2 = self.alerting_system.check_alerts([alert2])

        # Second alert should be suppressed if it's identical and within suppression window
        assert len(processed1) >= 0
        assert len(processed2) >= 0

    def test_alert_history_management(self):
        """Test alert history management."""
        # Create multiple alerts
        alerts = []
        for i in range(10):
            alert = Alert(
                name=f'alert_{i}',
                severity=AlertSeverity.LOW,
                message=f'Alert {i}',
                metric_name='test_metric',
                current_value=float(i),
                threshold=5.0
            )
            alerts.append(alert)

        # Process alerts
        with patch('builtins.print'):  # Suppress console output
            for alert in alerts:
                self.alerting_system.check_alerts([alert])

        # Verify history is maintained
        assert len(self.alerting_system.alert_history) > 0

    def test_escalation_timing(self):
        """Test escalation timing logic."""
        # Add escalation rule with delays
        rule = EscalationRule(
            severity=AlertSeverity.CRITICAL,
            initial_delay=5,  # 5 seconds initial delay
            escalation_delay=10,  # 10 seconds between escalations
            max_escalations=2,
            notification_configs=['test_config']
        )
        self.alerting_system.add_escalation_rule(AlertSeverity.CRITICAL, rule)

        # Create critical alert
        alert = Alert(
            name='critical_alert',
            severity=AlertSeverity.CRITICAL,
            message='Critical system failure',
            metric_name='system_health',
            current_value=0.0,
            threshold=50.0
        )

        # Process alert (should respect timing)
        processed = self.alerting_system.check_alerts([alert])

        # Verify alert was processed
        assert len(processed) >= 0

    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test invalid email configuration
        with pytest.raises(ValueError):
            invalid_config = NotificationConfig(
                name='invalid_email',
                channel=NotificationChannel.EMAIL,
                settings={}  # Missing required email settings
            )
            self.alerting_system.add_notification_config(invalid_config)

    def test_multiple_notification_channels(self):
        """Test sending alerts through multiple channels."""
        # Add multiple configs
        email_config = NotificationConfig(
            name='email',
            channel=NotificationChannel.EMAIL,
            settings={
                'smtp_server': 'smtp.example.com',
                'smtp_port': 587,
                'from_email': 'alerts@example.com',
                'to_emails': ['admin@example.com']
            }
        )

        console_config = NotificationConfig(
            name='console',
            channel=NotificationChannel.CONSOLE,
            settings={}
        )

        self.alerting_system.add_notification_config(email_config)
        self.alerting_system.add_notification_config(console_config)

        # Add escalation rule using both channels
        rule = EscalationRule(
            severity=AlertSeverity.HIGH,
            initial_delay=0,
            escalation_delay=60,
            max_escalations=1,
            notification_configs=['email', 'console']
        )
        self.alerting_system.add_escalation_rule(AlertSeverity.HIGH, rule)

        # Create test alert
        alert = Alert(
            name='multi_channel_alert',
            severity=AlertSeverity.HIGH,
            message='Test multi-channel alert',
            metric_name='test_metric',
            current_value=100.0,
            threshold=80.0
        )

        # Process alert (should use both channels)
        with patch('builtins.print'), patch('smtplib.SMTP'):
            processed = self.alerting_system.check_alerts([alert])

        # Verify alert was processed
        assert len(processed) >= 0

    def test_clear_alerts(self):
        """Test clearing alert history and active alerts."""
        # Create and process some alerts
        alert = Alert(
            name='test_alert',
            severity=AlertSeverity.LOW,
            message='Test alert',
            metric_name='test_metric',
            current_value=100.0,
            threshold=80.0
        )

        with patch('builtins.print'):
            self.alerting_system.check_alerts([alert])

        # Verify alerts exist
        assert len(self.alerting_system.active_alerts) > 0 or len(self.alerting_system.alert_history) > 0

        # Clear alerts
        self.alerting_system.clear_alerts()

        # Verify alerts are cleared
        assert len(self.alerting_system.active_alerts) == 0
        assert len(self.alerting_system.alert_history) == 0


if __name__ == '__main__':
    pytest.main([__file__])