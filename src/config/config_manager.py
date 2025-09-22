"""Configuration Manager

This module provides centralized configuration management for the Market Data Agent.
It supports YAML configuration files, environment variable overrides, and validation.
"""

import os
import yaml
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, field
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)


@dataclass
class DataSourceConfig:
    """Configuration for a data source"""
    enabled: bool = True
    priority: int = 2  # 1=PRIMARY, 2=SECONDARY, 3=BACKUP
    api_key: Optional[str] = None
    rate_limit_requests: Optional[int] = None
    rate_limit_period: Optional[int] = None  # seconds
    timeout: int = 30
    max_retries: int = 3
    cache_ttl: int = 300  # 5 minutes


@dataclass
class SourceManagerConfig:
    """Configuration for the source manager"""
    max_failure_threshold: int = 5
    circuit_breaker_timeout: int = 300  # 5 minutes
    health_check_interval: int = 60  # 1 minute
    validation_enabled: bool = True


@dataclass
class ValidationConfig:
    """Configuration for data validation"""
    max_price_change_percent: float = 50.0
    min_volume: int = 0
    min_price: float = 0.01
    max_price: float = 100000.0
    anomaly_detection_enabled: bool = True
    quality_threshold: int = 70


@dataclass
class APIConfig:
    """Configuration for the API server"""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1
    log_level: str = "info"
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class DatabaseConfig:
    """Configuration for the database"""
    url: str = "sqlite:///market_data.db"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30


@dataclass
class LoggingConfig:
    """Configuration for logging"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = False
    file_path: str = "logs/market_data_agent.log"
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5


@dataclass
class Config:
    """Main configuration class"""
    # Data sources
    yfinance: DataSourceConfig = field(default_factory=lambda: DataSourceConfig(
        priority=1,  # PRIMARY
        rate_limit_requests=120,
        rate_limit_period=3600  # 1 hour
    ))
    alpha_vantage: DataSourceConfig = field(default_factory=lambda: DataSourceConfig(
        priority=2,  # SECONDARY
        rate_limit_requests=500,
        rate_limit_period=86400  # 1 day
    ))

    # Core components
    source_manager: SourceManagerConfig = field(default_factory=SourceManagerConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    api: APIConfig = field(default_factory=APIConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Global settings
    environment: str = "development"
    debug: bool = False


class ConfigManager:
    """Manages application configuration from multiple sources"""

    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration manager

        Args:
            config_file: Path to YAML configuration file
        """
        self.config_file = config_file
        self._config: Optional[Config] = None

    def load(self) -> Config:
        """Load configuration from file and environment variables

        Returns:
            Loaded configuration object
        """
        # Start with default configuration
        config_dict = self._get_default_config()

        # Load from YAML file if specified
        if self.config_file and os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        config_dict = self._merge_configs(config_dict, file_config)
                logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file {self.config_file}: {e}")

        # Override with environment variables
        config_dict = self._apply_env_overrides(config_dict)

        # Validate and create config object
        self._config = self._create_config_object(config_dict)
        self._validate_config(self._config)

        logger.info(f"Configuration loaded for environment: {self._config.environment}")
        return self._config

    def get_config(self) -> Config:
        """Get the current configuration

        Returns:
            Current configuration object

        Raises:
            RuntimeError: If configuration hasn't been loaded
        """
        if self._config is None:
            raise RuntimeError("Configuration not loaded. Call load() first.")
        return self._config

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration as dictionary"""
        default_config = Config()
        return self._dataclass_to_dict(default_config)

    def _dataclass_to_dict(self, obj) -> Dict[str, Any]:
        """Convert dataclass to dictionary recursively"""
        result = {}
        for field_name, field_value in obj.__dict__.items():
            if hasattr(field_value, '__dict__'):
                # Nested dataclass
                result[field_name] = self._dataclass_to_dict(field_value)
            else:
                result[field_name] = field_value
        return result

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries"""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides"""
        # Environment mapping for common variables
        env_mappings = {
            # Global
            'ENVIRONMENT': ['environment'],
            'DEBUG': ['debug'],

            # API
            'API_HOST': ['api', 'host'],
            'API_PORT': ['api', 'port'],
            'API_RELOAD': ['api', 'reload'],
            'API_WORKERS': ['api', 'workers'],
            'API_LOG_LEVEL': ['api', 'log_level'],

            # Data sources
            'ALPHA_VANTAGE_API_KEY': ['alpha_vantage', 'api_key'],
            'YFINANCE_ENABLED': ['yfinance', 'enabled'],
            'ALPHA_VANTAGE_ENABLED': ['alpha_vantage', 'enabled'],

            # Database
            'DATABASE_URL': ['database', 'url'],
            'DATABASE_ECHO': ['database', 'echo'],

            # Logging
            'LOG_LEVEL': ['logging', 'level'],
            'LOG_FILE_ENABLED': ['logging', 'file_enabled'],
            'LOG_FILE_PATH': ['logging', 'file_path'],

            # Source manager
            'SOURCE_MANAGER_MAX_FAILURES': ['source_manager', 'max_failure_threshold'],
            'SOURCE_MANAGER_CIRCUIT_TIMEOUT': ['source_manager', 'circuit_breaker_timeout'],
            'SOURCE_MANAGER_HEALTH_INTERVAL': ['source_manager', 'health_check_interval'],

            # Validation
            'VALIDATION_ENABLED': ['source_manager', 'validation_enabled'],
            'VALIDATION_MAX_PRICE_CHANGE': ['validation', 'max_price_change_percent'],
            'VALIDATION_MIN_VOLUME': ['validation', 'min_volume'],
        }

        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                self._set_nested_value(config, config_path, self._convert_env_value(env_value))

        return config

    def _set_nested_value(self, config: Dict[str, Any], path: List[str], value: Any):
        """Set a nested configuration value"""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type"""
        # Boolean conversion
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        elif value.lower() in ('false', 'no', '0', 'off'):
            return False

        # Number conversion
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

    def _create_config_object(self, config_dict: Dict[str, Any]) -> Config:
        """Create Config object from dictionary"""
        try:
            # Create nested dataclass objects
            yfinance_config = DataSourceConfig(**config_dict.get('yfinance', {}))
            alpha_vantage_config = DataSourceConfig(**config_dict.get('alpha_vantage', {}))
            source_manager_config = SourceManagerConfig(**config_dict.get('source_manager', {}))
            validation_config = ValidationConfig(**config_dict.get('validation', {}))
            api_config = APIConfig(**config_dict.get('api', {}))
            database_config = DatabaseConfig(**config_dict.get('database', {}))
            logging_config = LoggingConfig(**config_dict.get('logging', {}))

            # Create main config object
            return Config(
                yfinance=yfinance_config,
                alpha_vantage=alpha_vantage_config,
                source_manager=source_manager_config,
                validation=validation_config,
                api=api_config,
                database=database_config,
                logging=logging_config,
                environment=config_dict.get('environment', 'development'),
                debug=config_dict.get('debug', False)
            )

        except Exception as e:
            logger.error(f"Failed to create config object: {e}")
            raise ValueError(f"Invalid configuration: {e}")

    def _validate_config(self, config: Config):
        """Validate configuration values"""
        errors = []

        # Validate API configuration
        if config.api.port < 1 or config.api.port > 65535:
            errors.append("API port must be between 1 and 65535")

        if config.api.workers < 1:
            errors.append("API workers must be at least 1")

        # Validate source manager configuration
        if config.source_manager.max_failure_threshold < 1:
            errors.append("Max failure threshold must be at least 1")

        if config.source_manager.circuit_breaker_timeout < 1:
            errors.append("Circuit breaker timeout must be at least 1 second")

        if config.source_manager.health_check_interval < 1:
            errors.append("Health check interval must be at least 1 second")

        # Validate data source priorities
        if config.yfinance.priority < 1 or config.yfinance.priority > 999:
            errors.append("YFinance priority must be between 1 and 999")

        if config.alpha_vantage.priority < 1 or config.alpha_vantage.priority > 999:
            errors.append("Alpha Vantage priority must be between 1 and 999")

        # Validate validation configuration
        if config.validation.max_price_change_percent < 0:
            errors.append("Max price change percent must be non-negative")

        if config.validation.min_price < 0:
            errors.append("Min price must be non-negative")

        if config.validation.max_price <= config.validation.min_price:
            errors.append("Max price must be greater than min price")

        if errors:
            raise ValueError("Configuration validation errors: " + "; ".join(errors))

        logger.info("Configuration validation passed")

    def save_sample_config(self, file_path: str):
        """Save a sample configuration file

        Args:
            file_path: Path where to save the sample config
        """
        sample_config = self._get_default_config()

        # Add comments and examples
        sample_yaml = """# Market Data Agent Configuration
# This file contains the main configuration for the Market Data Agent

# Global settings
environment: development  # development, staging, production
debug: false

# Data source configurations
yfinance:
  enabled: true
  priority: 1  # 1=PRIMARY, 2=SECONDARY, 3=BACKUP
  rate_limit_requests: 120
  rate_limit_period: 3600  # seconds (1 hour)
  timeout: 30
  max_retries: 3

alpha_vantage:
  enabled: true
  priority: 2
  api_key: null  # Set via environment variable ALPHA_VANTAGE_API_KEY
  rate_limit_requests: 500
  rate_limit_period: 86400  # seconds (1 day)
  timeout: 30
  max_retries: 3

# Source manager configuration
source_manager:
  max_failure_threshold: 5
  circuit_breaker_timeout: 300  # seconds (5 minutes)
  health_check_interval: 60     # seconds (1 minute)
  validation_enabled: true

# Data validation configuration
validation:
  max_price_change_percent: 50.0
  min_volume: 0
  min_price: 0.01
  max_price: 100000.0
  anomaly_detection_enabled: true
  quality_threshold: 70

# API server configuration
api:
  host: "0.0.0.0"
  port: 8000
  reload: false  # Set to true for development
  workers: 1
  log_level: "info"
  cors_enabled: true
  cors_origins: ["*"]

# Database configuration
database:
  url: "sqlite:///market_data.db"
  echo: false
  pool_size: 5
  max_overflow: 10
  pool_timeout: 30

# Logging configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file_enabled: false
  file_path: "logs/market_data_agent.log"
  max_file_size: 10485760  # 10MB
  backup_count: 5
"""

        try:
            # Ensure directory exists
            directory = os.path.dirname(file_path)
            if directory:  # Only create directory if file_path has a directory component
                os.makedirs(directory, exist_ok=True)

            with open(file_path, 'w') as f:
                f.write(sample_yaml)

            logger.info(f"Sample configuration saved to {file_path}")

        except Exception as e:
            logger.error(f"Failed to save sample config: {e}")
            raise


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> Config:
    """Get the current configuration

    Returns:
        Current configuration object
    """
    return config_manager.get_config()


def load_config(config_file: Optional[str] = None) -> Config:
    """Load configuration from file and environment

    Args:
        config_file: Optional path to configuration file

    Returns:
        Loaded configuration object
    """
    global config_manager
    config_manager = ConfigManager(config_file)
    return config_manager.load()