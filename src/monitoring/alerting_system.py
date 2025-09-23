"""Intelligent Alerting System

Advanced alerting with multiple notification channels, alert correlation,
intelligent suppression, and escalation workflows for operational excellence.
"""

import asyncio
import smtplib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

from .metrics_collector import Alert, AlertSeverity

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Types of notification channels"""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    CONSOLE = "console"
    SMS = "sms"
    CUSTOM = "custom"


class EscalationLevel(Enum):
    """Alert escalation levels"""
    LEVEL_1 = "level_1"  # Immediate team
    LEVEL_2 = "level_2"  # Management
    LEVEL_3 = "level_3"  # Executive
    LEVEL_4 = "level_4"  # External support


@dataclass
class NotificationConfig:
    """Configuration for notification channel"""
    channel: NotificationChannel
    enabled: bool = True
    severity_filter: List[AlertSeverity] = field(default_factory=lambda: list(AlertSeverity))
    config: Dict[str, Any] = field(default_factory=dict)
    rate_limit_minutes: int = 5
    template: Optional[str] = None


@dataclass
class EscalationRule:
    """Alert escalation configuration"""
    severity: AlertSeverity
    escalation_delays: List[int]  # Minutes before each escalation level
    channels_per_level: List[List[str]]  # Channels for each escalation level
    max_escalations: int = 3
    enabled: bool = True


@dataclass
class AlertCorrelation:
    """Alert correlation rule"""
    name: str
    metric_patterns: List[str]
    time_window_minutes: int
    min_alerts: int
    correlation_message: str
    suppress_individual: bool = True


@dataclass
class NotificationHistory:
    """History of sent notifications"""
    alert_id: str
    channel: NotificationChannel
    recipient: str
    sent_at: datetime
    success: bool
    error_message: Optional[str] = None


class AlertingSystem:
    """Intelligent alerting system with multiple notification channels"""

    def __init__(self):
        self.notification_configs: Dict[str, NotificationConfig] = {}
        self.escalation_rules: Dict[AlertSeverity, EscalationRule] = {}
        self.correlation_rules: Dict[str, AlertCorrelation] = {}

        # State tracking
        self.notification_history: List[NotificationHistory] = []
        self.escalation_state: Dict[str, Dict[str, Any]] = {}
        self.correlation_state: Dict[str, List[Alert]] = {}
        self.rate_limit_state: Dict[str, datetime] = {}

        # Alert processors
        self.alert_processors: List[Callable[[Alert], Alert]] = []
        self.notification_handlers: Dict[NotificationChannel, Callable] = {}

        # Configuration
        self.max_history_days = 30
        self.correlation_check_interval = 60  # seconds

        # Initialize default handlers
        self._setup_default_handlers()

        # Background tasks
        self._correlation_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        logger.info("Alerting system initialized")

    def add_notification_config(self, name: str, config: NotificationConfig) -> None:
        """Add notification channel configuration

        Args:
            name: Unique configuration name
            config: Notification configuration
        """
        self.notification_configs[name] = config
        logger.info(f"Added notification config: {name} ({config.channel.value})")

    def add_escalation_rule(self, severity: AlertSeverity, rule: EscalationRule) -> None:
        """Add escalation rule for severity level

        Args:
            severity: Alert severity level
            rule: Escalation rule configuration
        """
        self.escalation_rules[severity] = rule
        logger.info(f"Added escalation rule for {severity.value}")

    def add_correlation_rule(self, rule: AlertCorrelation) -> None:
        """Add alert correlation rule

        Args:
            rule: Correlation rule configuration
        """
        self.correlation_rules[rule.name] = rule
        logger.info(f"Added correlation rule: {rule.name}")

    def add_alert_processor(self, processor: Callable[[Alert], Alert]) -> None:
        """Add custom alert processor

        Args:
            processor: Function that processes and potentially modifies alerts
        """
        self.alert_processors.append(processor)
        logger.info("Added custom alert processor")

    async def process_alert(self, alert: Alert) -> None:
        """Process incoming alert through the alerting system

        Args:
            alert: Alert to process
        """
        try:
            # Apply alert processors
            processed_alert = alert
            for processor in self.alert_processors:
                try:
                    processed_alert = processor(processed_alert)
                except Exception as e:
                    logger.error(f"Alert processor failed: {e}")

            # Check for correlation
            correlated_alert = await self._check_correlation(processed_alert)
            if correlated_alert != processed_alert:
                logger.info(f"Alert correlated: {correlated_alert.message}")

            # Send notifications
            await self._send_notifications(correlated_alert)

            # Start escalation if needed
            await self._start_escalation(correlated_alert)

        except Exception as e:
            logger.error(f"Failed to process alert: {e}")

    async def _check_correlation(self, alert: Alert) -> Alert:
        """Check if alert should be correlated with others

        Args:
            alert: Alert to check

        Returns:
            Original or correlated alert
        """
        for rule_name, rule in self.correlation_rules.items():
            # Check if alert matches any pattern
            if not any(pattern in alert.rule.metric_name for pattern in rule.metric_patterns):
                continue

            # Initialize correlation state if needed
            if rule_name not in self.correlation_state:
                self.correlation_state[rule_name] = []

            # Add alert to correlation window
            self.correlation_state[rule_name].append(alert)

            # Remove old alerts outside time window
            cutoff_time = datetime.now() - timedelta(minutes=rule.time_window_minutes)
            self.correlation_state[rule_name] = [
                a for a in self.correlation_state[rule_name]
                if a.triggered_at > cutoff_time
            ]

            # Check if correlation threshold is met
            if len(self.correlation_state[rule_name]) >= rule.min_alerts:
                # Create correlated alert
                correlated_alert = Alert(
                    rule=alert.rule,
                    triggered_at=datetime.now(),
                    metric_value=alert.metric_value,
                    labels=alert.labels,
                    message=f"{rule.correlation_message} ({len(self.correlation_state[rule_name])} related alerts)"
                )

                # Clear correlation state to avoid duplicates
                self.correlation_state[rule_name] = []

                return correlated_alert

        return alert

    async def _send_notifications(self, alert: Alert) -> None:
        """Send notifications for alert

        Args:
            alert: Alert to send notifications for
        """
        for config_name, config in self.notification_configs.items():
            if not config.enabled:
                continue

            # Check severity filter
            if config.severity_filter and alert.rule.severity not in config.severity_filter:
                continue

            # Check rate limiting
            rate_limit_key = f"{config_name}_{alert.rule.metric_name}"
            if rate_limit_key in self.rate_limit_state:
                last_sent = self.rate_limit_state[rate_limit_key]
                if (datetime.now() - last_sent).total_seconds() < config.rate_limit_minutes * 60:
                    continue

            # Send notification
            success = await self._send_notification(config, alert)

            if success:
                self.rate_limit_state[rate_limit_key] = datetime.now()

    async def _send_notification(self, config: NotificationConfig, alert: Alert) -> bool:
        """Send single notification

        Args:
            config: Notification configuration
            alert: Alert to send

        Returns:
            True if notification was sent successfully
        """
        try:
            handler = self.notification_handlers.get(config.channel)
            if not handler:
                logger.warning(f"No handler for channel: {config.channel.value}")
                return False

            # Format message
            message = await self._format_alert_message(alert, config.template)

            # Send notification
            success = await handler(config, alert, message)

            # Record in history
            self.notification_history.append(NotificationHistory(
                alert_id=f"{alert.rule.metric_name}_{alert.triggered_at.timestamp()}",
                channel=config.channel,
                recipient=config.config.get("recipient", "unknown"),
                sent_at=datetime.now(),
                success=success
            ))

            return success

        except Exception as e:
            logger.error(f"Failed to send notification via {config.channel.value}: {e}")
            return False

    async def _format_alert_message(self, alert: Alert, template: Optional[str] = None) -> str:
        """Format alert message using template

        Args:
            alert: Alert to format
            template: Message template

        Returns:
            Formatted message string
        """
        if template:
            return template.format(
                severity=alert.rule.severity.value.upper(),
                metric=alert.rule.metric_name,
                value=alert.metric_value,
                threshold=alert.rule.threshold,
                time=alert.triggered_at.strftime("%Y-%m-%d %H:%M:%S"),
                message=alert.message
            )

        # Default template
        return (
            f"🚨 {alert.rule.severity.value.upper()} ALERT\n"
            f"Metric: {alert.rule.metric_name}\n"
            f"Current Value: {alert.metric_value}\n"
            f"Threshold: {alert.rule.threshold}\n"
            f"Time: {alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Message: {alert.message}"
        )

    async def _start_escalation(self, alert: Alert) -> None:
        """Start escalation process for alert

        Args:
            alert: Alert to escalate
        """
        escalation_rule = self.escalation_rules.get(alert.rule.severity)
        if not escalation_rule or not escalation_rule.enabled:
            return

        alert_key = f"{alert.rule.metric_name}_{hash(frozenset(alert.labels.items()))}"

        # Initialize escalation state
        self.escalation_state[alert_key] = {
            "alert": alert,
            "level": 0,
            "last_escalation": datetime.now(),
            "escalation_count": 0
        }

        # Schedule escalations
        asyncio.create_task(self._escalation_loop(alert_key, escalation_rule))

    async def _escalation_loop(self, alert_key: str, rule: EscalationRule) -> None:
        """Handle escalation timing and execution

        Args:
            alert_key: Alert identifier
            rule: Escalation rule
        """
        try:
            state = self.escalation_state.get(alert_key)
            if not state:
                return

            for level, delay_minutes in enumerate(rule.escalation_delays):
                if level >= rule.max_escalations:
                    break

                # Wait for escalation delay
                await asyncio.sleep(delay_minutes * 60)

                # Check if alert still active
                if alert_key not in self.escalation_state:
                    return

                # Check if alert was resolved
                alert = state["alert"]
                if alert.resolved:
                    del self.escalation_state[alert_key]
                    return

                # Escalate
                await self._execute_escalation(alert, level, rule)

                # Update escalation state
                state["level"] = level + 1
                state["last_escalation"] = datetime.now()
                state["escalation_count"] += 1

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Escalation loop failed: {e}")
        finally:
            # Clean up
            if alert_key in self.escalation_state:
                del self.escalation_state[alert_key]

    async def _execute_escalation(self, alert: Alert, level: int, rule: EscalationRule) -> None:
        """Execute escalation at specific level

        Args:
            alert: Alert to escalate
            level: Escalation level
            rule: Escalation rule
        """
        if level >= len(rule.channels_per_level):
            return

        channels = rule.channels_per_level[level]
        escalation_message = f"⬆️ ESCALATION LEVEL {level + 1}: {alert.message}"

        for channel_name in channels:
            config = self.notification_configs.get(channel_name)
            if config:
                # Create escalated alert
                escalated_alert = Alert(
                    rule=alert.rule,
                    triggered_at=alert.triggered_at,
                    metric_value=alert.metric_value,
                    labels=alert.labels,
                    message=escalation_message
                )

                await self._send_notification(config, escalated_alert)

    def _setup_default_handlers(self) -> None:
        """Setup default notification handlers"""
        self.notification_handlers[NotificationChannel.EMAIL] = self._handle_email
        self.notification_handlers[NotificationChannel.WEBHOOK] = self._handle_webhook
        self.notification_handlers[NotificationChannel.CONSOLE] = self._handle_console
        self.notification_handlers[NotificationChannel.SLACK] = self._handle_slack

    async def _handle_email(self, config: NotificationConfig, alert: Alert, message: str) -> bool:
        """Handle email notification

        Args:
            config: Email configuration
            alert: Alert to send
            message: Formatted message

        Returns:
            True if email was sent successfully
        """
        try:
            email_config = config.config

            # Create message
            msg = MIMEMultipart()
            msg['From'] = email_config.get('from_email', 'alerts@marketdata.com')
            msg['To'] = email_config.get('to_email', 'admin@marketdata.com')
            msg['Subject'] = f"Market Data Alert: {alert.rule.severity.value.upper()}"

            msg.attach(MIMEText(message, 'plain'))

            # Send email
            server = smtplib.SMTP(
                email_config.get('smtp_server', 'localhost'),
                email_config.get('smtp_port', 587)
            )

            if email_config.get('use_tls', True):
                server.starttls()

            if email_config.get('username'):
                server.login(email_config['username'], email_config['password'])

            server.send_message(msg)
            server.quit()

            logger.info(f"Email alert sent: {alert.rule.metric_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False

    async def _handle_webhook(self, config: NotificationConfig, alert: Alert, message: str) -> bool:
        """Handle webhook notification

        Args:
            config: Webhook configuration
            alert: Alert to send
            message: Formatted message

        Returns:
            True if webhook was successful
        """
        try:
            import aiohttp

            webhook_config = config.config
            url = webhook_config.get('url')
            if not url:
                logger.error("Webhook URL not configured")
                return False

            payload = {
                'alert': {
                    'severity': alert.rule.severity.value,
                    'metric': alert.rule.metric_name,
                    'value': alert.metric_value,
                    'threshold': alert.rule.threshold,
                    'time': alert.triggered_at.isoformat(),
                    'message': alert.message,
                    'labels': alert.labels
                },
                'text': message
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=webhook_config.get('headers', {}),
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status < 400:
                        logger.info(f"Webhook alert sent: {alert.rule.metric_name}")
                        return True
                    else:
                        logger.error(f"Webhook failed with status {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False

    async def _handle_console(self, config: NotificationConfig, alert: Alert, message: str) -> bool:
        """Handle console notification

        Args:
            config: Console configuration
            alert: Alert to send
            message: Formatted message

        Returns:
            Always True (console output doesn't fail)
        """
        print(f"\n{'='*50}")
        print("MARKET DATA ALERT")
        print(f"{'='*50}")
        print(message)
        print(f"{'='*50}\n")
        return True

    async def _handle_slack(self, config: NotificationConfig, alert: Alert, message: str) -> bool:
        """Handle Slack notification

        Args:
            config: Slack configuration
            alert: Alert to send
            message: Formatted message

        Returns:
            True if Slack message was sent successfully
        """
        try:
            import aiohttp

            slack_config = config.config
            webhook_url = slack_config.get('webhook_url')
            if not webhook_url:
                logger.error("Slack webhook URL not configured")
                return False

            # Format for Slack
            color_map = {
                AlertSeverity.CRITICAL: "#ff0000",
                AlertSeverity.HIGH: "#ff8800",
                AlertSeverity.MEDIUM: "#ffaa00",
                AlertSeverity.LOW: "#00aa00",
                AlertSeverity.INFO: "#0088ff"
            }

            payload = {
                "attachments": [
                    {
                        "color": color_map.get(alert.rule.severity, "#888888"),
                        "title": f"{alert.rule.severity.value.upper()} Alert: {alert.rule.metric_name}",
                        "text": message,
                        "footer": "Market Data Agent",
                        "ts": int(alert.triggered_at.timestamp())
                    }
                ]
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Slack alert sent: {alert.rule.metric_name}")
                        return True
                    else:
                        logger.error(f"Slack webhook failed with status {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False

    async def start(self) -> None:
        """Start background tasks"""
        if self._running:
            return

        self._running = True
        self._correlation_task = asyncio.create_task(self._correlation_cleanup_loop())
        self._cleanup_task = asyncio.create_task(self._history_cleanup_loop())
        logger.info("Alerting system started")

    async def stop(self) -> None:
        """Stop background tasks"""
        self._running = False

        if self._correlation_task:
            self._correlation_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()

        # Wait for tasks to complete
        tasks = [t for t in [self._correlation_task, self._cleanup_task] if t]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("Alerting system stopped")

    async def _correlation_cleanup_loop(self) -> None:
        """Background task to clean up correlation state"""
        while self._running:
            try:
                await asyncio.sleep(self.correlation_check_interval)

                # Clean up expired correlation data
                current_time = datetime.now()
                for rule_name, rule in self.correlation_rules.items():
                    if rule_name in self.correlation_state:
                        cutoff_time = current_time - timedelta(minutes=rule.time_window_minutes)
                        self.correlation_state[rule_name] = [
                            alert for alert in self.correlation_state[rule_name]
                            if alert.triggered_at > cutoff_time
                        ]

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Correlation cleanup failed: {e}")

    async def _history_cleanup_loop(self) -> None:
        """Background task to clean up notification history"""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Run every hour

                # Clean up old notification history
                cutoff_date = datetime.now() - timedelta(days=self.max_history_days)
                self.notification_history = [
                    record for record in self.notification_history
                    if record.sent_at > cutoff_date
                ]

                logger.debug(f"Cleaned up notification history, {len(self.notification_history)} records remaining")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"History cleanup failed: {e}")

    def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification statistics

        Returns:
            Dictionary with notification statistics
        """
        total_notifications = len(self.notification_history)
        successful_notifications = sum(1 for n in self.notification_history if n.success)

        # Get stats by channel
        channel_stats = {}
        for record in self.notification_history:
            channel = record.channel.value
            if channel not in channel_stats:
                channel_stats[channel] = {"total": 0, "successful": 0}

            channel_stats[channel]["total"] += 1
            if record.success:
                channel_stats[channel]["successful"] += 1

        return {
            "total_notifications": total_notifications,
            "successful_notifications": successful_notifications,
            "success_rate": successful_notifications / total_notifications if total_notifications > 0 else 0,
            "channel_stats": channel_stats,
            "active_escalations": len(self.escalation_state),
            "correlation_rules": len(self.correlation_rules),
            "notification_configs": len(self.notification_configs)
        }

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of currently active alerts

        Returns:
            List of active alert dictionaries
        """
        # Return empty list for now since we don't track active alerts yet
        # In a full implementation, this would return actual active alerts
        return []


# Global alerting system instance
alerting_system = AlertingSystem()