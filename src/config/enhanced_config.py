"""
Enhanced Configuration System

Complete configuration management system with hot reloading, environment configs,
validation, and integration with the Market Data Agent components.
"""

import asyncio
import os
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import logging

from .config_manager import ConfigManager, Config
from .hot_reload_config import (
    HotReloadConfigManager,
    EnvironmentOverride,
    ConfigValidationRule,
    start_hot_reload_config,
    stop_hot_reload_config,
    get_hot_reload_manager
)

logger = logging.getLogger(__name__)


class EnhancedConfigManager:
    """Enhanced configuration manager with all advanced features"""

    def __init__(self):
        self.hot_reload_manager: Optional[HotReloadConfigManager] = None
        self._config: Optional[Config] = None
        self._initialized = False

        # Configuration change callbacks
        self._component_callbacks: Dict[str, Callable[[Config, Config], None]] = {}

    async def initialize(
        self,
        base_config_file: str = "config.yaml",
        environment: Optional[str] = None,
        enable_hot_reload: bool = True
    ) -> Config:
        """Initialize the enhanced configuration system

        Args:
            base_config_file: Main configuration file
            environment: Environment to use (auto-detected if None)
            enable_hot_reload: Enable hot reloading

        Returns:
            Loaded configuration
        """
        if self._initialized:
            logger.warning("Enhanced configuration already initialized")
            return self._config

        try:
            # Determine environment
            if environment is None:
                environment = os.getenv('ENVIRONMENT', 'development')

            # Setup environment-specific config files
            environment_configs = {
                'development': 'config/development.yaml',
                'staging': 'config/staging.yaml',
                'production': 'config/production.yaml'
            }

            if enable_hot_reload:
                # Initialize hot reload system
                self._config = await start_hot_reload_config(
                    config_file=base_config_file,
                    environment_configs=environment_configs
                )

                self.hot_reload_manager = get_hot_reload_manager()

                # Setup default environment overrides
                self._setup_default_overrides()

                # Setup component integration callbacks
                self._setup_component_callbacks()

                logger.info("Hot reload configuration system initialized")

            else:
                # Initialize standard configuration
                config_manager = ConfigManager(base_config_file)
                self._config = config_manager.load()
                logger.info("Standard configuration system initialized")

            self._initialized = True
            logger.info(f"Enhanced configuration initialized for environment: {environment}")

            return self._config

        except Exception as e:
            logger.error(f"Failed to initialize enhanced configuration: {e}")
            raise

    async def shutdown(self):
        """Shutdown the enhanced configuration system"""
        if self.hot_reload_manager:
            await stop_hot_reload_config()
            self.hot_reload_manager = None

        self._initialized = False
        logger.info("Enhanced configuration system shut down")

    def get_config(self) -> Config:
        """Get current configuration"""
        if not self._initialized:
            raise RuntimeError("Enhanced configuration not initialized")

        if self.hot_reload_manager:
            return self.hot_reload_manager.get_config()
        else:
            return self._config

    def get_config_value(self, path: str, default: Any = None) -> Any:
        """Get specific configuration value by path"""
        if self.hot_reload_manager:
            return self.hot_reload_manager.get_config_value(path, default)
        else:
            # Simple path traversal for standard config
            config = self.get_config()
            parts = path.split('.')
            current = config

            try:
                for part in parts:
                    current = getattr(current, part)
                return current
            except AttributeError:
                return default

    def set_config_value(self, path: str, value: Any, environment: Optional[str] = None):
        """Set configuration value (only works with hot reload)"""
        if not self.hot_reload_manager:
            raise RuntimeError("Config value setting only available with hot reload enabled")

        self.hot_reload_manager.set_config_value(path, value, environment)

    def add_component_callback(self, component_name: str, callback: Callable[[Config, Config], None]):
        """Add configuration change callback for a component"""
        self._component_callbacks[component_name] = callback

        if self.hot_reload_manager:
            self.hot_reload_manager.add_reload_callback(callback)

    def remove_component_callback(self, component_name: str):
        """Remove configuration change callback for a component"""
        if component_name in self._component_callbacks:
            callback = self._component_callbacks.pop(component_name)

            if self.hot_reload_manager:
                self.hot_reload_manager.remove_reload_callback(callback)

    def add_validation_rule(self, rule: ConfigValidationRule):
        """Add configuration validation rule"""
        if self.hot_reload_manager:
            self.hot_reload_manager.add_validation_rule(rule)

    def get_config_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get configuration change history"""
        if self.hot_reload_manager:
            return self.hot_reload_manager.get_config_history(limit)
        else:
            return []

    def export_config(self, format: str = 'yaml') -> str:
        """Export current configuration"""
        if self.hot_reload_manager:
            return self.hot_reload_manager.export_config(format)
        else:
            # Basic export for standard config
            import yaml
            from dataclasses import asdict
            config_dict = asdict(self.get_config())
            return yaml.dump(config_dict, default_flow_style=False, indent=2)

    def _setup_default_overrides(self):
        """Setup default environment overrides"""
        if not self.hot_reload_manager:
            return

        # Development overrides
        dev_overrides = [
            EnvironmentOverride(
                environment="development",
                config_path="api.reload",
                value=True,
                priority=1
            ),
            EnvironmentOverride(
                environment="development",
                config_path="debug",
                value=True,
                priority=1
            ),
            EnvironmentOverride(
                environment="development",
                config_path="logging.level",
                value="DEBUG",
                priority=1
            )
        ]

        # Production overrides
        prod_overrides = [
            EnvironmentOverride(
                environment="production",
                config_path="api.reload",
                value=False,
                priority=1
            ),
            EnvironmentOverride(
                environment="production",
                config_path="debug",
                value=False,
                priority=1
            ),
            EnvironmentOverride(
                environment="production",
                config_path="logging.level",
                value="INFO",
                priority=1
            )
        ]

        for override in dev_overrides + prod_overrides:
            self.hot_reload_manager.add_environment_override(override)

    def _setup_component_callbacks(self):
        """Setup callbacks for component configuration changes"""
        if not self.hot_reload_manager:
            return

        # API server callback
        def api_config_changed(old_config: Config, new_config: Config):
            if old_config.api != new_config.api:
                logger.info("API configuration changed - restart may be required")

        # Database callback
        def database_config_changed(old_config: Config, new_config: Config):
            if old_config.database != new_config.database:
                logger.info("Database configuration changed - connection pool will be refreshed")

        # Data source callback
        def datasource_config_changed(old_config: Config, new_config: Config):
            changes = []
            if old_config.yfinance != new_config.yfinance:
                changes.append("yfinance")
            if old_config.alpha_vantage != new_config.alpha_vantage:
                changes.append("alpha_vantage")

            if changes:
                logger.info(f"Data source configuration changed: {', '.join(changes)}")

        # Validation callback
        def validation_config_changed(old_config: Config, new_config: Config):
            if old_config.validation != new_config.validation:
                logger.info("Validation configuration changed - new rules will apply immediately")

        self.add_component_callback("api", api_config_changed)
        self.add_component_callback("database", database_config_changed)
        self.add_component_callback("datasources", datasource_config_changed)
        self.add_component_callback("validation", validation_config_changed)


# Global enhanced configuration manager
_enhanced_config_manager: Optional[EnhancedConfigManager] = None


async def initialize_enhanced_config(
    base_config_file: str = "config.yaml",
    environment: Optional[str] = None,
    enable_hot_reload: bool = True
) -> Config:
    """Initialize the global enhanced configuration system"""
    global _enhanced_config_manager

    _enhanced_config_manager = EnhancedConfigManager()
    return await _enhanced_config_manager.initialize(
        base_config_file=base_config_file,
        environment=environment,
        enable_hot_reload=enable_hot_reload
    )


async def shutdown_enhanced_config():
    """Shutdown the global enhanced configuration system"""
    global _enhanced_config_manager

    if _enhanced_config_manager:
        await _enhanced_config_manager.shutdown()
        _enhanced_config_manager = None


def get_enhanced_config_manager() -> EnhancedConfigManager:
    """Get the global enhanced configuration manager"""
    if _enhanced_config_manager is None:
        raise RuntimeError("Enhanced configuration not initialized")
    return _enhanced_config_manager


def get_current_config() -> Config:
    """Get the current configuration"""
    return get_enhanced_config_manager().get_config()


def get_config_value(path: str, default: Any = None) -> Any:
    """Get specific configuration value by path"""
    return get_enhanced_config_manager().get_config_value(path, default)


def set_config_value(path: str, value: Any, environment: Optional[str] = None):
    """Set configuration value"""
    get_enhanced_config_manager().set_config_value(path, value, environment)


# Configuration decorators for easy component integration
def config_change_handler(component_name: str):
    """Decorator to register configuration change handlers"""
    def decorator(func: Callable[[Config, Config], None]):
        if _enhanced_config_manager:
            _enhanced_config_manager.add_component_callback(component_name, func)
        return func
    return decorator


# Utility functions for common configuration patterns
def is_development() -> bool:
    """Check if running in development environment"""
    try:
        return get_config_value("environment", "development") == "development"
    except:
        return True  # Default to development if config not available


def is_production() -> bool:
    """Check if running in production environment"""
    try:
        return get_config_value("environment", "development") == "production"
    except:
        return False


def is_debug_enabled() -> bool:
    """Check if debug mode is enabled"""
    try:
        return get_config_value("debug", False)
    except:
        return False


def get_api_config() -> Dict[str, Any]:
    """Get API configuration as dictionary"""
    try:
        config = get_current_config()
        return {
            "host": config.api.host,
            "port": config.api.port,
            "reload": config.api.reload,
            "workers": config.api.workers,
            "log_level": config.api.log_level,
            "cors_enabled": config.api.cors_enabled,
            "cors_origins": config.api.cors_origins
        }
    except:
        return {
            "host": "0.0.0.0",
            "port": 8000,
            "reload": False,
            "workers": 1,
            "log_level": "info",
            "cors_enabled": True,
            "cors_origins": ["*"]
        }