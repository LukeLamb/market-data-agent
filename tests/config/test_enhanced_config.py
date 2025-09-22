"""
Tests for Enhanced Configuration System

Comprehensive tests for hot reloading, environment configs, validation,
and configuration management features.
"""

import pytest
import asyncio
import tempfile
import os
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.config.enhanced_config import (
    EnhancedConfigManager,
    initialize_enhanced_config,
    shutdown_enhanced_config,
    get_config_value,
    is_development,
    is_production
)
from src.config.hot_reload_config import (
    HotReloadConfigManager,
    EnvironmentOverride,
    ConfigValidationRule,
    ConfigChangeEvent
)
from src.config.config_manager import Config, ConfigManager


class TestEnhancedConfigManager:
    """Test suite for Enhanced Configuration Manager"""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary directory for config files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def sample_config(self):
        """Sample configuration data"""
        return {
            'environment': 'test',
            'debug': True,
            'api': {
                'host': '127.0.0.1',
                'port': 8001,
                'reload': True
            },
            'database': {
                'url': 'sqlite:///test.db'
            },
            'logging': {
                'level': 'DEBUG'
            }
        }

    @pytest.fixture
    def config_file(self, temp_config_dir, sample_config):
        """Create sample config file"""
        config_path = os.path.join(temp_config_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)
        return config_path

    @pytest.fixture
    def dev_config_file(self, temp_config_dir):
        """Create development config file"""
        dev_config = {
            'debug': True,
            'api': {
                'reload': True,
                'log_level': 'debug'
            },
            'logging': {
                'level': 'DEBUG'
            }
        }

        # Create config directory
        config_dir = os.path.join(temp_config_dir, 'config')
        os.makedirs(config_dir, exist_ok=True)

        dev_config_path = os.path.join(config_dir, 'development.yaml')
        with open(dev_config_path, 'w') as f:
            yaml.dump(dev_config, f)
        return dev_config_path

    def test_enhanced_config_manager_initialization(self):
        """Test enhanced config manager initialization"""
        manager = EnhancedConfigManager()
        assert manager.hot_reload_manager is None
        assert manager._config is None
        assert not manager._initialized

    @pytest.mark.asyncio
    async def test_initialize_with_hot_reload(self, config_file):
        """Test initialization with hot reload enabled"""
        manager = EnhancedConfigManager()

        # Change to config file directory for relative paths to work
        original_cwd = os.getcwd()
        config_dir = os.path.dirname(config_file)

        try:
            os.chdir(config_dir)
            config = await manager.initialize(
                base_config_file=os.path.basename(config_file),
                environment='test',
                enable_hot_reload=True
            )

            assert manager._initialized
            assert manager.hot_reload_manager is not None
            assert config is not None
            assert config.environment == 'test'

        finally:
            await manager.shutdown()
            os.chdir(original_cwd)

    @pytest.mark.asyncio
    async def test_initialize_without_hot_reload(self, config_file):
        """Test initialization without hot reload"""
        manager = EnhancedConfigManager()

        config = await manager.initialize(
            base_config_file=config_file,
            environment='test',
            enable_hot_reload=False
        )

        assert manager._initialized
        assert manager.hot_reload_manager is None
        assert config is not None
        assert config.environment == 'test'

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_get_config_value(self, config_file):
        """Test getting configuration values by path"""
        manager = EnhancedConfigManager()

        await manager.initialize(
            base_config_file=config_file,
            environment='test',
            enable_hot_reload=False
        )

        # Test getting nested values
        assert manager.get_config_value('api.host') == '127.0.0.1'
        assert manager.get_config_value('api.port') == 8001
        assert manager.get_config_value('environment') == 'test'
        assert manager.get_config_value('nonexistent.path', 'default') == 'default'

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_component_callbacks(self, config_file):
        """Test component configuration change callbacks"""
        manager = EnhancedConfigManager()

        callback_called = False
        old_config_ref = None
        new_config_ref = None

        def test_callback(old_config: Config, new_config: Config):
            nonlocal callback_called, old_config_ref, new_config_ref
            callback_called = True
            old_config_ref = old_config
            new_config_ref = new_config

        await manager.initialize(
            base_config_file=config_file,
            environment='test',
            enable_hot_reload=False
        )

        manager.add_component_callback('test_component', test_callback)
        assert 'test_component' in manager._component_callbacks

        manager.remove_component_callback('test_component')
        assert 'test_component' not in manager._component_callbacks

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_global_functions(self, config_file):
        """Test global configuration functions"""
        # Initialize global config
        config = await initialize_enhanced_config(
            base_config_file=config_file,
            environment='test',
            enable_hot_reload=False
        )

        assert config is not None

        # Test global getter functions
        assert get_config_value('environment') == 'test'
        assert get_config_value('api.host') == '127.0.0.1'

        # Shutdown global config
        await shutdown_enhanced_config()

    def test_utility_functions(self):
        """Test utility functions"""
        # Test without initialized config (should use defaults)
        assert is_development() == True  # Default
        assert is_production() == False


class TestHotReloadConfigManager:
    """Test suite for Hot Reload Configuration Manager"""

    @pytest.fixture
    def base_manager(self):
        """Create base configuration manager"""
        return ConfigManager()

    @pytest.fixture
    def hot_reload_manager(self, base_manager):
        """Create hot reload configuration manager"""
        return HotReloadConfigManager(base_manager)

    def test_hot_reload_initialization(self, hot_reload_manager):
        """Test hot reload manager initialization"""
        assert hot_reload_manager._config is None
        assert len(hot_reload_manager._watched_files) == 0
        assert len(hot_reload_manager._reload_callbacks) == 0
        assert not hot_reload_manager._running

    def test_add_watched_file(self, hot_reload_manager, tmp_path):
        """Test adding files to watch list"""
        # Create a temporary file
        test_file = tmp_path / "test_config.yaml"
        test_file.write_text("test: value")

        hot_reload_manager.add_watched_file(str(test_file))
        assert str(test_file.resolve()) in hot_reload_manager._watched_files

        # Test adding non-existent file
        non_existent = tmp_path / "non_existent.yaml"
        hot_reload_manager.add_watched_file(str(non_existent))
        assert str(non_existent.resolve()) not in hot_reload_manager._watched_files

    def test_environment_overrides(self, hot_reload_manager):
        """Test environment-specific overrides"""
        override = EnvironmentOverride(
            environment="test",
            config_path="api.port",
            value=9000,
            priority=1
        )

        hot_reload_manager.add_environment_override(override)
        assert override in hot_reload_manager._environment_overrides

    def test_validation_rules(self, hot_reload_manager):
        """Test configuration validation rules"""
        rule = ConfigValidationRule(
            path="api.port",
            validator=lambda x: isinstance(x, int) and 1000 <= x <= 9999,
            error_message="Port must be between 1000 and 9999"
        )

        hot_reload_manager.add_validation_rule(rule)
        assert rule in hot_reload_manager._validation_rules

    def test_config_change_tracking(self, hot_reload_manager):
        """Test configuration change tracking"""
        # Mock config objects
        old_config_dict = {"api": {"port": 8000}, "debug": False}
        new_config_dict = {"api": {"port": 9000}, "debug": True}

        changes = hot_reload_manager._find_config_differences(old_config_dict, new_config_dict)

        assert len(changes) == 2
        assert any(change['path'] == 'api.port' and change['type'] == 'modified' for change in changes)
        assert any(change['path'] == 'debug' and change['type'] == 'modified' for change in changes)

    def test_nested_value_operations(self, hot_reload_manager):
        """Test nested configuration value operations"""
        config_dict = {
            'level1': {
                'level2': {
                    'value': 'test'
                }
            }
        }

        # Test getting nested value
        value = hot_reload_manager._get_nested_value(config_dict, ['level1', 'level2', 'value'])
        assert value == 'test'

        # Test getting non-existent value
        value = hot_reload_manager._get_nested_value(config_dict, ['non', 'existent'], 'default')
        assert value == 'default'

        # Test setting nested value
        hot_reload_manager._set_nested_value(config_dict, ['level1', 'level2', 'new_value'], 'new')
        assert config_dict['level1']['level2']['new_value'] == 'new'

    def test_config_merge(self, hot_reload_manager):
        """Test configuration merging"""
        base_config = {
            'api': {'host': 'localhost', 'port': 8000},
            'debug': False
        }

        override_config = {
            'api': {'port': 9000},
            'debug': True,
            'new_setting': 'value'
        }

        merged = hot_reload_manager._merge_configs(base_config, override_config)

        assert merged['api']['host'] == 'localhost'  # Preserved
        assert merged['api']['port'] == 9000  # Overridden
        assert merged['debug'] == True  # Overridden
        assert merged['new_setting'] == 'value'  # Added


class TestConfigValidation:
    """Test configuration validation features"""

    def test_validation_rule_creation(self):
        """Test creating validation rules"""
        rule = ConfigValidationRule(
            path="test.value",
            validator=lambda x: x > 0,
            error_message="Value must be positive",
            required=True
        )

        assert rule.path == "test.value"
        assert rule.validator(5) == True
        assert rule.validator(-1) == False
        assert rule.required == True

    def test_environment_override_creation(self):
        """Test creating environment overrides"""
        override = EnvironmentOverride(
            environment="development",
            config_path="debug",
            value=True,
            priority=2
        )

        assert override.environment == "development"
        assert override.config_path == "debug"
        assert override.value == True
        assert override.priority == 2


class TestConfigChangeEvents:
    """Test configuration change event tracking"""

    def test_config_change_event_creation(self):
        """Test creating configuration change events"""
        event = ConfigChangeEvent(
            config_path="api.port",
            old_value=8000,
            new_value=9000,
            timestamp=datetime.now(),
            change_type="modified"
        )

        assert event.config_path == "api.port"
        assert event.old_value == 8000
        assert event.new_value == 9000
        assert event.change_type == "modified"


@pytest.mark.asyncio
async def test_integration_scenario(tmp_path):
    """Test complete integration scenario"""
    # Create config files
    base_config = {
        'environment': 'development',
        'debug': False,
        'api': {
            'host': '0.0.0.0',
            'port': 8000,
            'reload': False
        }
    }

    dev_config = {
        'debug': True,
        'api': {
            'reload': True,
            'log_level': 'debug'
        }
    }

    base_config_file = tmp_path / "config.yaml"
    dev_config_file = tmp_path / "development.yaml"

    with open(base_config_file, 'w') as f:
        yaml.dump(base_config, f)

    with open(dev_config_file, 'w') as f:
        yaml.dump(dev_config, f)

    # Initialize enhanced config
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        config = await initialize_enhanced_config(
            base_config_file=str(base_config_file.name),
            environment='development',
            enable_hot_reload=False  # Disable for testing
        )

        # Verify configuration loaded correctly
        assert config.environment == 'development'
        assert get_config_value('api.host') == '0.0.0.0'
        assert get_config_value('api.port') == 8000

        # Test utility functions
        assert is_development() == True
        assert is_production() == False

    finally:
        await shutdown_enhanced_config()
        os.chdir(original_cwd)


if __name__ == '__main__':
    pytest.main([__file__])