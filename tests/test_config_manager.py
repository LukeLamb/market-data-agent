"""Tests for Configuration Manager"""

import pytest
import os
import tempfile
import yaml
from unittest.mock import patch

from src.config.config_manager import (
    ConfigManager,
    Config,
    DataSourceConfig,
    SourceManagerConfig,
    ValidationConfig,
    APIConfig,
    DatabaseConfig,
    LoggingConfig,
    load_config,
    get_config
)


class TestConfigManager:
    """Test cases for configuration manager"""

    def test_default_config_creation(self):
        """Test creation of default configuration"""
        config_manager = ConfigManager()
        config = config_manager.load()

        assert isinstance(config, Config)
        assert config.environment == "development"
        assert config.debug is False
        assert config.yfinance.enabled is True
        assert config.yfinance.priority == 1
        assert config.alpha_vantage.priority == 2
        assert config.api.port == 8000
        assert config.source_manager.max_failure_threshold == 5

    def test_yaml_config_loading(self):
        """Test loading configuration from YAML file"""
        # Create temporary YAML file
        config_data = {
            'environment': 'testing',
            'debug': True,
            'api': {
                'port': 9000,
                'host': '127.0.0.1'
            },
            'yfinance': {
                'enabled': False,
                'priority': 3
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_file = f.name

        try:
            config_manager = ConfigManager(temp_file)
            config = config_manager.load()

            assert config.environment == 'testing'
            assert config.debug is True
            assert config.api.port == 9000
            assert config.api.host == '127.0.0.1'
            assert config.yfinance.enabled is False
            assert config.yfinance.priority == 3

        finally:
            os.unlink(temp_file)

    def test_environment_variable_overrides(self):
        """Test environment variable overrides"""
        env_vars = {
            'ENVIRONMENT': 'production',
            'DEBUG': 'true',
            'API_HOST': '192.168.1.100',
            'API_PORT': '8080',
            'API_WORKERS': '4',
            'ALPHA_VANTAGE_API_KEY': 'test_key_123',
            'DATABASE_URL': 'postgresql://localhost/test',
            'LOG_LEVEL': 'DEBUG',
            'SOURCE_MANAGER_MAX_FAILURES': '10'
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config_manager = ConfigManager()
            config = config_manager.load()

            assert config.environment == 'production'
            assert config.debug is True
            assert config.api.host == '192.168.1.100'
            assert config.api.port == 8080
            assert config.api.workers == 4
            assert config.alpha_vantage.api_key == 'test_key_123'
            assert config.database.url == 'postgresql://localhost/test'
            assert config.logging.level == 'DEBUG'
            assert config.source_manager.max_failure_threshold == 10

    def test_boolean_environment_conversion(self):
        """Test boolean environment variable conversion"""
        test_cases = [
            ('true', True),
            ('True', True),
            ('TRUE', True),
            ('yes', True),
            ('1', True),
            ('on', True),
            ('false', False),
            ('False', False),
            ('FALSE', False),
            ('no', False),
            ('0', False),
            ('off', False),
        ]

        for env_value, expected in test_cases:
            with patch.dict(os.environ, {'DEBUG': env_value}, clear=False):
                config_manager = ConfigManager()
                config = config_manager.load()
                assert config.debug == expected

    def test_numeric_environment_conversion(self):
        """Test numeric environment variable conversion"""
        with patch.dict(os.environ, {'API_PORT': '8080'}, clear=False):
            config_manager = ConfigManager()
            config = config_manager.load()
            assert config.api.port == 8080
            assert isinstance(config.api.port, int)

        with patch.dict(os.environ, {'VALIDATION_MAX_PRICE_CHANGE': '25.5'}, clear=False):
            config_manager = ConfigManager()
            config = config_manager.load()
            assert config.validation.max_price_change_percent == 25.5
            assert isinstance(config.validation.max_price_change_percent, float)

    def test_config_validation_success(self):
        """Test successful configuration validation"""
        config_manager = ConfigManager()
        config = config_manager.load()
        # Should not raise any exceptions
        config_manager._validate_config(config)

    def test_config_validation_errors(self):
        """Test configuration validation errors"""
        config_manager = ConfigManager()

        # Test invalid API port
        config = config_manager.load()
        config.api.port = -1
        with pytest.raises(ValueError, match="API port must be between"):
            config_manager._validate_config(config)

        # Test invalid max failure threshold
        config = config_manager.load()
        config.source_manager.max_failure_threshold = 0
        with pytest.raises(ValueError, match="Max failure threshold must be at least"):
            config_manager._validate_config(config)

        # Test invalid price range
        config = config_manager.load()
        config.validation.max_price = 10.0
        config.validation.min_price = 20.0
        with pytest.raises(ValueError, match="Max price must be greater than min price"):
            config_manager._validate_config(config)

    def test_nested_config_merge(self):
        """Test merging of nested configuration"""
        config_manager = ConfigManager()

        base_config = {
            'api': {'host': 'localhost', 'port': 8000},
            'yfinance': {'enabled': True, 'priority': 1}
        }

        override_config = {
            'api': {'port': 9000},  # Only override port, keep host
            'alpha_vantage': {'enabled': False}  # Add new section
        }

        merged = config_manager._merge_configs(base_config, override_config)

        assert merged['api']['host'] == 'localhost'  # Preserved
        assert merged['api']['port'] == 9000  # Overridden
        assert merged['yfinance']['enabled'] is True  # Preserved
        assert merged['alpha_vantage']['enabled'] is False  # Added

    def test_invalid_yaml_file(self):
        """Test handling of invalid YAML file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_file = f.name

        try:
            config_manager = ConfigManager(temp_file)
            config = config_manager.load()  # Should still load with defaults
            assert config.environment == "development"

        finally:
            os.unlink(temp_file)

    def test_nonexistent_config_file(self):
        """Test handling of nonexistent config file"""
        config_manager = ConfigManager("nonexistent_file.yaml")
        config = config_manager.load()  # Should load with defaults
        assert config.environment == "development"

    def test_get_config_before_load(self):
        """Test getting configuration before loading"""
        config_manager = ConfigManager()
        with pytest.raises(RuntimeError, match="Configuration not loaded"):
            config_manager.get_config()

    def test_save_sample_config(self):
        """Test saving sample configuration file"""
        config_manager = ConfigManager()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_file = f.name

        try:
            os.unlink(temp_file)  # Remove the empty file
            config_manager.save_sample_config(temp_file)

            # Verify file was created and contains expected content
            assert os.path.exists(temp_file)

            with open(temp_file, 'r') as f:
                content = f.read()
                assert 'Market Data Agent Configuration' in content
                assert 'yfinance:' in content
                assert 'alpha_vantage:' in content
                assert 'api:' in content

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_dataclass_configurations(self):
        """Test individual dataclass configurations"""
        # Test DataSourceConfig
        ds_config = DataSourceConfig(
            enabled=True,
            priority=1,
            api_key="test_key",
            rate_limit_requests=100,
            timeout=60
        )
        assert ds_config.enabled is True
        assert ds_config.priority == 1
        assert ds_config.api_key == "test_key"

        # Test SourceManagerConfig
        sm_config = SourceManagerConfig(
            max_failure_threshold=10,
            circuit_breaker_timeout=600,
            health_check_interval=120
        )
        assert sm_config.max_failure_threshold == 10
        assert sm_config.circuit_breaker_timeout == 600

        # Test ValidationConfig
        val_config = ValidationConfig(
            max_price_change_percent=25.0,
            min_volume=1000,
            quality_threshold=80
        )
        assert val_config.max_price_change_percent == 25.0
        assert val_config.min_volume == 1000

        # Test APIConfig
        api_config = APIConfig(
            host="127.0.0.1",
            port=9000,
            workers=2,
            cors_enabled=False
        )
        assert api_config.host == "127.0.0.1"
        assert api_config.port == 9000
        assert api_config.cors_enabled is False

    def test_global_functions(self):
        """Test global configuration functions"""
        # Test load_config function
        config = load_config()
        assert isinstance(config, Config)

        # Test get_config function
        retrieved_config = get_config()
        assert retrieved_config is config  # Should be the same instance

    def test_config_with_yaml_and_env_override(self):
        """Test configuration with both YAML file and environment overrides"""
        # Create YAML config
        yaml_config = {
            'environment': 'staging',
            'api': {'port': 7000, 'host': 'yaml-host'},
            'yfinance': {'priority': 2}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_config, f)
            temp_file = f.name

        try:
            # Environment variables should override YAML
            with patch.dict(os.environ, {
                'ENVIRONMENT': 'production',
                'API_PORT': '8080'
            }, clear=False):
                config_manager = ConfigManager(temp_file)
                config = config_manager.load()

                # Environment should override YAML
                assert config.environment == 'production'
                assert config.api.port == 8080

                # YAML should override defaults where no env var exists
                assert config.api.host == 'yaml-host'
                assert config.yfinance.priority == 2

        finally:
            os.unlink(temp_file)

    def test_complex_nested_yaml_config(self):
        """Test complex nested YAML configuration"""
        complex_config = {
            'source_manager': {
                'max_failure_threshold': 3,
                'circuit_breaker_timeout': 120,
                'validation_enabled': False
            },
            'validation': {
                'max_price_change_percent': 75.0,
                'anomaly_detection_enabled': False,
                'quality_threshold': 90
            },
            'logging': {
                'level': 'DEBUG',
                'file_enabled': True,
                'file_path': '/tmp/test.log'
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(complex_config, f)
            temp_file = f.name

        try:
            config_manager = ConfigManager(temp_file)
            config = config_manager.load()

            assert config.source_manager.max_failure_threshold == 3
            assert config.source_manager.circuit_breaker_timeout == 120
            assert config.source_manager.validation_enabled is False

            assert config.validation.max_price_change_percent == 75.0
            assert config.validation.anomaly_detection_enabled is False
            assert config.validation.quality_threshold == 90

            assert config.logging.level == 'DEBUG'
            assert config.logging.file_enabled is True
            assert config.logging.file_path == '/tmp/test.log'

        finally:
            os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__])