"""
Hot Reload Configuration Manager

Advanced configuration management with hot reloading, environment-specific configs,
configuration validation, and real-time configuration updates.
"""

import asyncio
import os
import yaml
import json
import hashlib
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Set, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
from concurrent.futures import ThreadPoolExecutor

from .config_manager import Config, ConfigManager

logger = logging.getLogger(__name__)


@dataclass
class ConfigChangeEvent:
    """Configuration change event"""
    config_path: str
    old_value: Any
    new_value: Any
    timestamp: datetime
    change_type: str  # 'modified', 'added', 'removed'


@dataclass
class EnvironmentOverride:
    """Environment-specific configuration override"""
    environment: str
    config_path: str
    value: Any
    priority: int = 1  # Higher priority overrides lower priority


@dataclass
class ConfigValidationRule:
    """Configuration validation rule"""
    path: str
    validator: Callable[[Any], bool]
    error_message: str
    required: bool = True


class ConfigFileWatcher(FileSystemEventHandler):
    """Watches configuration files for changes"""

    def __init__(self, hot_reload_manager: 'HotReloadConfigManager'):
        self.manager = hot_reload_manager
        self.debounce_time = 1.0  # seconds
        self.last_modified = {}

    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory:
            return

        file_path = event.src_path

        # Debounce rapid file changes
        now = datetime.now()
        if file_path in self.last_modified:
            if (now - self.last_modified[file_path]).total_seconds() < self.debounce_time:
                return

        self.last_modified[file_path] = now

        # Check if it's a watched configuration file
        if self.manager.is_watched_file(file_path):
            logger.info(f"Configuration file changed: {file_path}")
            asyncio.create_task(self.manager.reload_config_async(file_path))


class HotReloadConfigManager:
    """Enhanced configuration manager with hot reloading capabilities"""

    def __init__(self, base_config_manager: ConfigManager):
        self.base_manager = base_config_manager
        self._config: Optional[Config] = None
        self._config_lock = threading.RLock()

        # Hot reload settings
        self._watched_files: Set[str] = set()
        self._observer: Optional[Observer] = None
        self._file_watcher: Optional[ConfigFileWatcher] = None
        self._reload_callbacks: List[Callable[[Config, Config], None]] = []

        # Environment-specific configurations
        self._environment_overrides: List[EnvironmentOverride] = []
        self._environment_configs: Dict[str, str] = {}  # env -> config file path

        # Configuration validation
        self._validation_rules: List[ConfigValidationRule] = []

        # Configuration history and change tracking
        self._config_history: List[Dict[str, Any]] = []
        self._change_events: List[ConfigChangeEvent] = []
        self._max_history_size = 50

        # Performance and caching
        self._config_hash: Optional[str] = None
        self._config_cache: Dict[str, Any] = {}
        self._cache_ttl = timedelta(minutes=5)
        self._last_cache_update: Optional[datetime] = None

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        self._setup_default_validation_rules()

    def _setup_default_validation_rules(self):
        """Setup default configuration validation rules"""
        rules = [
            ConfigValidationRule(
                path="api.port",
                validator=lambda x: isinstance(x, int) and 1 <= x <= 65535,
                error_message="API port must be between 1 and 65535"
            ),
            ConfigValidationRule(
                path="api.workers",
                validator=lambda x: isinstance(x, int) and x >= 1,
                error_message="API workers must be at least 1"
            ),
            ConfigValidationRule(
                path="source_manager.max_failure_threshold",
                validator=lambda x: isinstance(x, int) and x >= 1,
                error_message="Max failure threshold must be at least 1"
            ),
            ConfigValidationRule(
                path="validation.max_price_change_percent",
                validator=lambda x: isinstance(x, (int, float)) and x >= 0,
                error_message="Max price change percent must be non-negative"
            ),
            ConfigValidationRule(
                path="validation.min_price",
                validator=lambda x: isinstance(x, (int, float)) and x >= 0,
                error_message="Min price must be non-negative"
            )
        ]

        self._validation_rules.extend(rules)

    async def start(self):
        """Start hot reload monitoring"""
        self._running = True

        # Start file system watcher
        if self._watched_files:
            self._file_watcher = ConfigFileWatcher(self)
            self._observer = Observer()

            # Watch directories containing config files
            watched_dirs = set()
            for file_path in self._watched_files:
                directory = os.path.dirname(os.path.abspath(file_path))
                if directory not in watched_dirs:
                    self._observer.schedule(self._file_watcher, directory, recursive=False)
                    watched_dirs.add(directory)

            self._observer.start()
            logger.info(f"Started watching {len(watched_dirs)} directories for config changes")

        # Start background cleanup task
        self._cleanup_task = asyncio.create_task(self._background_cleanup())

        logger.info("Hot reload configuration manager started")

    async def stop(self):
        """Stop hot reload monitoring"""
        self._running = False

        # Stop file system watcher
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
            self._file_watcher = None

        # Stop background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("Hot reload configuration manager stopped")

    def load_config(self, config_file: Optional[str] = None) -> Config:
        """Load configuration with hot reload support"""
        with self._config_lock:
            # Load base configuration
            config = self.base_manager.load()

            # Apply environment-specific overrides
            config = self._apply_environment_overrides(config)

            # Validate configuration
            self._validate_configuration(config)

            # Store configuration and update tracking
            old_config = self._config
            self._config = config

            # Update configuration hash for change detection
            self._update_config_hash(config)

            # Store in history
            self._add_to_history(config)

            # Trigger change callbacks
            if old_config and old_config != config:
                self._trigger_change_callbacks(old_config, config)
                self._track_config_changes(old_config, config)

            # Add config file to watched files
            if config_file and os.path.exists(config_file):
                self.add_watched_file(config_file)

            logger.info(f"Configuration loaded successfully (environment: {config.environment})")
            return config

    def get_config(self) -> Config:
        """Get current configuration"""
        with self._config_lock:
            if self._config is None:
                raise RuntimeError("Configuration not loaded. Call load_config() first.")
            return self._config

    def add_watched_file(self, file_path: str):
        """Add a file to the watch list for hot reloading"""
        abs_path = os.path.abspath(file_path)
        if os.path.exists(abs_path):
            self._watched_files.add(abs_path)
            logger.info(f"Added {abs_path} to watched files")
        else:
            logger.warning(f"Cannot watch non-existent file: {abs_path}")

    def remove_watched_file(self, file_path: str):
        """Remove a file from the watch list"""
        abs_path = os.path.abspath(file_path)
        self._watched_files.discard(abs_path)
        logger.info(f"Removed {abs_path} from watched files")

    def is_watched_file(self, file_path: str) -> bool:
        """Check if a file is being watched"""
        abs_path = os.path.abspath(file_path)
        return abs_path in self._watched_files

    def add_reload_callback(self, callback: Callable[[Config, Config], None]):
        """Add a callback to be called when configuration changes"""
        self._reload_callbacks.append(callback)

    def remove_reload_callback(self, callback: Callable[[Config, Config], None]):
        """Remove a reload callback"""
        if callback in self._reload_callbacks:
            self._reload_callbacks.remove(callback)

    def add_environment_config(self, environment: str, config_file: str):
        """Add environment-specific configuration file"""
        if os.path.exists(config_file):
            self._environment_configs[environment] = config_file
            self.add_watched_file(config_file)
            logger.info(f"Added environment config for {environment}: {config_file}")
        else:
            logger.warning(f"Environment config file does not exist: {config_file}")

    def add_environment_override(self, override: EnvironmentOverride):
        """Add environment-specific configuration override"""
        self._environment_overrides.append(override)
        logger.info(f"Added environment override for {override.environment}: {override.config_path}")

    def add_validation_rule(self, rule: ConfigValidationRule):
        """Add configuration validation rule"""
        self._validation_rules.append(rule)

    async def reload_config_async(self, changed_file: Optional[str] = None):
        """Asynchronously reload configuration"""
        try:
            logger.info(f"Reloading configuration due to file change: {changed_file}")

            # Small delay to ensure file write is complete
            await asyncio.sleep(0.5)

            # Reload configuration
            with self._config_lock:
                old_config = self._config
                new_config = self.load_config(self.base_manager.config_file)

                if old_config and old_config != new_config:
                    logger.info("Configuration reloaded successfully with changes")
                else:
                    logger.info("Configuration reloaded (no changes detected)")

        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")

    def get_config_value(self, path: str, default: Any = None) -> Any:
        """Get a specific configuration value by path"""
        config = self.get_config()
        return self._get_nested_value(asdict(config), path.split('.'), default)

    def set_config_value(self, path: str, value: Any, environment: Optional[str] = None):
        """Set a configuration value (creates an environment override)"""
        env = environment or self.get_config().environment
        override = EnvironmentOverride(
            environment=env,
            config_path=path,
            value=value,
            priority=1
        )
        self.add_environment_override(override)

    def get_config_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get configuration change history"""
        return self._config_history[-limit:]

    def get_change_events(self, limit: int = 20) -> List[ConfigChangeEvent]:
        """Get recent configuration change events"""
        return self._change_events[-limit:]

    def export_config(self, format: str = 'yaml') -> str:
        """Export current configuration in specified format"""
        config = self.get_config()
        config_dict = asdict(config)

        if format.lower() == 'yaml':
            return yaml.dump(config_dict, default_flow_style=False, indent=2)
        elif format.lower() == 'json':
            return json.dumps(config_dict, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _apply_environment_overrides(self, config: Config) -> Config:
        """Apply environment-specific overrides"""
        current_env = config.environment
        config_dict = asdict(config)

        # Load environment-specific config file if exists
        if current_env in self._environment_configs:
            env_config_file = self._environment_configs[current_env]
            if os.path.exists(env_config_file):
                try:
                    with open(env_config_file, 'r') as f:
                        env_config = yaml.safe_load(f)
                        if env_config:
                            config_dict = self._merge_configs(config_dict, env_config)
                            logger.info(f"Applied environment config from {env_config_file}")
                except Exception as e:
                    logger.warning(f"Failed to load environment config {env_config_file}: {e}")

        # Apply environment overrides
        applicable_overrides = [
            override for override in self._environment_overrides
            if override.environment == current_env
        ]

        # Sort by priority (higher priority first)
        applicable_overrides.sort(key=lambda x: x.priority, reverse=True)

        for override in applicable_overrides:
            self._set_nested_value(config_dict, override.config_path.split('.'), override.value)
            logger.debug(f"Applied override: {override.config_path} = {override.value}")

        # Recreate config object
        return self.base_manager._create_config_object(config_dict)

    def _validate_configuration(self, config: Config):
        """Validate configuration using defined rules"""
        config_dict = asdict(config)
        errors = []

        for rule in self._validation_rules:
            try:
                value = self._get_nested_value(config_dict, rule.path.split('.'))

                if value is None and rule.required:
                    errors.append(f"Required configuration missing: {rule.path}")
                elif value is not None and not rule.validator(value):
                    errors.append(f"{rule.error_message} (path: {rule.path}, value: {value})")

            except Exception as e:
                if rule.required:
                    errors.append(f"Cannot validate required configuration {rule.path}: {e}")

        if errors:
            raise ValueError("Configuration validation failed: " + "; ".join(errors))

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries"""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def _get_nested_value(self, config: Dict[str, Any], path: List[str], default: Any = None) -> Any:
        """Get nested configuration value"""
        current = config
        try:
            for key in path:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default

    def _set_nested_value(self, config: Dict[str, Any], path: List[str], value: Any):
        """Set nested configuration value"""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def _update_config_hash(self, config: Config):
        """Update configuration hash for change detection"""
        config_str = json.dumps(asdict(config), sort_keys=True)
        self._config_hash = hashlib.sha256(config_str.encode()).hexdigest()

    def _add_to_history(self, config: Config):
        """Add configuration to history"""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'config_hash': self._config_hash,
            'environment': config.environment,
            'config': asdict(config)
        }

        self._config_history.append(history_entry)

        # Trim history if too large
        if len(self._config_history) > self._max_history_size:
            self._config_history = self._config_history[-self._max_history_size:]

    def _track_config_changes(self, old_config: Config, new_config: Config):
        """Track configuration changes and create change events"""
        old_dict = asdict(old_config)
        new_dict = asdict(new_config)

        changes = self._find_config_differences(old_dict, new_dict)

        for change in changes:
            event = ConfigChangeEvent(
                config_path=change['path'],
                old_value=change['old_value'],
                new_value=change['new_value'],
                timestamp=datetime.now(),
                change_type=change['type']
            )
            self._change_events.append(event)

        # Trim change events if too many
        if len(self._change_events) > 100:
            self._change_events = self._change_events[-100:]

    def _find_config_differences(self, old: Dict[str, Any], new: Dict[str, Any], path: str = "") -> List[Dict[str, Any]]:
        """Find differences between configuration dictionaries"""
        differences = []

        # Check for modified and removed keys
        for key, old_value in old.items():
            current_path = f"{path}.{key}" if path else key

            if key not in new:
                differences.append({
                    'path': current_path,
                    'old_value': old_value,
                    'new_value': None,
                    'type': 'removed'
                })
            elif isinstance(old_value, dict) and isinstance(new[key], dict):
                differences.extend(self._find_config_differences(old_value, new[key], current_path))
            elif old_value != new[key]:
                differences.append({
                    'path': current_path,
                    'old_value': old_value,
                    'new_value': new[key],
                    'type': 'modified'
                })

        # Check for added keys
        for key, new_value in new.items():
            if key not in old:
                current_path = f"{path}.{key}" if path else key
                differences.append({
                    'path': current_path,
                    'old_value': None,
                    'new_value': new_value,
                    'type': 'added'
                })

        return differences

    def _trigger_change_callbacks(self, old_config: Config, new_config: Config):
        """Trigger configuration change callbacks"""
        for callback in self._reload_callbacks:
            try:
                callback(old_config, new_config)
            except Exception as e:
                logger.error(f"Configuration change callback failed: {e}")

    async def _background_cleanup(self):
        """Background task for cleanup and maintenance"""
        while self._running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes

                # Cleanup old cache entries
                self._cleanup_cache()

                # Cleanup old change events (keep only last 100)
                if len(self._change_events) > 100:
                    self._change_events = self._change_events[-100:]

                # Cleanup old history (keep only last 50)
                if len(self._config_history) > self._max_history_size:
                    self._config_history = self._config_history[-self._max_history_size:]

            except Exception as e:
                logger.error(f"Background cleanup error: {e}")

    def _cleanup_cache(self):
        """Cleanup expired cache entries"""
        if not self._last_cache_update:
            return

        if datetime.now() - self._last_cache_update > self._cache_ttl:
            self._config_cache.clear()
            self._last_cache_update = None
            logger.debug("Configuration cache cleared")


# Global hot reload configuration manager
_hot_reload_manager: Optional[HotReloadConfigManager] = None


def create_hot_reload_manager(base_config_manager: ConfigManager) -> HotReloadConfigManager:
    """Create hot reload configuration manager"""
    global _hot_reload_manager
    _hot_reload_manager = HotReloadConfigManager(base_config_manager)
    return _hot_reload_manager


def get_hot_reload_manager() -> HotReloadConfigManager:
    """Get the global hot reload configuration manager"""
    if _hot_reload_manager is None:
        raise RuntimeError("Hot reload manager not created. Call create_hot_reload_manager() first.")
    return _hot_reload_manager


async def start_hot_reload_config(config_file: Optional[str] = None, environment_configs: Optional[Dict[str, str]] = None) -> Config:
    """Start hot reload configuration system

    Args:
        config_file: Main configuration file path
        environment_configs: Dict of environment -> config file paths

    Returns:
        Loaded configuration
    """
    from .config_manager import ConfigManager

    # Create base config manager
    base_manager = ConfigManager(config_file)

    # Create hot reload manager
    hot_manager = create_hot_reload_manager(base_manager)

    # Add environment configs if provided
    if environment_configs:
        for env, env_config_file in environment_configs.items():
            hot_manager.add_environment_config(env, env_config_file)

    # Load configuration
    config = hot_manager.load_config(config_file)

    # Start hot reload monitoring
    await hot_manager.start()

    logger.info("Hot reload configuration system started")
    return config


async def stop_hot_reload_config():
    """Stop hot reload configuration system"""
    global _hot_reload_manager
    if _hot_reload_manager:
        await _hot_reload_manager.stop()
        _hot_reload_manager = None
    logger.info("Hot reload configuration system stopped")