#!/usr/bin/env python3
"""
Configuration Management CLI Tool

Command-line tool for managing Market Data Agent configuration,
including viewing, editing, validation, and hot reload monitoring.
"""

import asyncio
import argparse
import json
import sys
import os
from typing import Optional, Dict, Any
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config.enhanced_config import (
    initialize_enhanced_config,
    shutdown_enhanced_config,
    get_enhanced_config_manager,
    get_config_value,
    set_config_value
)
from src.config.hot_reload_config import ConfigValidationRule


class ConfigTool:
    """Configuration management CLI tool"""

    def __init__(self):
        self.config_manager = None

    async def initialize(self, config_file: str, environment: Optional[str] = None):
        """Initialize configuration system"""
        try:
            await initialize_enhanced_config(
                base_config_file=config_file,
                environment=environment,
                enable_hot_reload=True
            )
            self.config_manager = get_enhanced_config_manager()
            print(f"Configuration initialized from {config_file}")
            if environment:
                print(f"Environment: {environment}")
        except Exception as e:
            print(f"Failed to initialize configuration: {e}")
            sys.exit(1)

    async def shutdown(self):
        """Shutdown configuration system"""
        if self.config_manager:
            await shutdown_enhanced_config()

    async def view_config(self, path: Optional[str] = None, format: str = 'yaml'):
        """View configuration or specific value"""
        try:
            if path:
                # View specific configuration value
                value = get_config_value(path)
                if value is None:
                    print(f"Configuration path '{path}' not found")
                else:
                    print(f"{path}: {value}")
            else:
                # View entire configuration
                config_str = self.config_manager.export_config(format)
                print(config_str)
        except Exception as e:
            print(f"Error viewing configuration: {e}")

    async def set_config(self, path: str, value: str, environment: Optional[str] = None):
        """Set configuration value"""
        try:
            # Try to parse value as JSON first for complex types
            try:
                parsed_value = json.loads(value)
            except json.JSONDecodeError:
                # Treat as string if not valid JSON
                parsed_value = value

            set_config_value(path, parsed_value, environment)
            print(f"Set {path} = {parsed_value}")
            if environment:
                print(f"Environment: {environment}")
        except Exception as e:
            print(f"Error setting configuration: {e}")

    async def validate_config(self):
        """Validate current configuration"""
        try:
            config = self.config_manager.get_config()
            print("Configuration validation passed")
            print(f"Environment: {config.environment}")
            print(f"Debug mode: {config.debug}")
        except Exception as e:
            print(f"Configuration validation failed: {e}")

    async def show_history(self, limit: int = 10):
        """Show configuration change history"""
        try:
            history = self.config_manager.get_config_history(limit)
            if not history:
                print("No configuration history available")
                return

            print(f"Configuration History (last {len(history)} changes):")
            print("-" * 60)

            for i, entry in enumerate(reversed(history), 1):
                print(f"{i}. {entry['timestamp']}")
                print(f"   Environment: {entry['environment']}")
                print(f"   Hash: {entry['config_hash'][:12]}...")
                print()
        except Exception as e:
            print(f"Error showing history: {e}")

    async def show_changes(self, limit: int = 20):
        """Show recent configuration changes"""
        try:
            changes = self.config_manager.get_change_events(limit)
            if not changes:
                print("No configuration changes recorded")
                return

            print(f"Recent Configuration Changes (last {len(changes)}):")
            print("-" * 60)

            for i, change in enumerate(reversed(changes), 1):
                print(f"{i}. {change.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   Path: {change.config_path}")
                print(f"   Type: {change.change_type}")
                print(f"   Old: {change.old_value}")
                print(f"   New: {change.new_value}")
                print()
        except Exception as e:
            print(f"Error showing changes: {e}")

    async def watch_config(self):
        """Watch for configuration changes"""
        print("Watching for configuration changes... (Press Ctrl+C to stop)")
        print("-" * 60)

        def config_change_callback(old_config, new_config):
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Configuration changed")
            print(f"Environment: {new_config.environment}")

        self.config_manager.add_component_callback('cli_watcher', config_change_callback)

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping configuration watch...")
        finally:
            self.config_manager.remove_component_callback('cli_watcher')

    async def test_environments(self):
        """Test all environment configurations"""
        environments = ['development', 'staging', 'production']

        print("Testing environment configurations...")
        print("-" * 60)

        for env in environments:
            print(f"\nTesting {env} environment:")
            try:
                # Reinitialize with specific environment
                await shutdown_enhanced_config()

                await initialize_enhanced_config(
                    base_config_file='config.yaml',
                    environment=env,
                    enable_hot_reload=False
                )

                config_manager = get_enhanced_config_manager()
                config = config_manager.get_config()

                print(f"  ✓ Environment: {config.environment}")
                print(f"  ✓ Debug: {config.debug}")
                print(f"  ✓ API Port: {config.api.port}")
                print(f"  ✓ Log Level: {config.logging.level}")

            except Exception as e:
                print(f"  ✗ Error: {e}")

    async def create_sample_configs(self):
        """Create sample configuration files"""
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)

        # Sample environment configs
        sample_configs = {
            'development.yaml': {
                'debug': True,
                'api': {
                    'reload': True,
                    'log_level': 'debug'
                },
                'logging': {
                    'level': 'DEBUG',
                    'file_enabled': True
                }
            },
            'production.yaml': {
                'debug': False,
                'api': {
                    'reload': False,
                    'workers': 4,
                    'log_level': 'info'
                },
                'logging': {
                    'level': 'INFO',
                    'file_enabled': True,
                    'file_path': '/var/log/market_data_agent.log'
                },
                'database': {
                    'url': 'postgresql://user:pass@localhost/market_data_prod'
                }
            }
        }

        for filename, config_data in sample_configs.items():
            config_path = config_dir / filename

            if config_path.exists():
                print(f"Skipping {config_path} (already exists)")
                continue

            try:
                import yaml
                with open(config_path, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                print(f"Created {config_path}")
            except Exception as e:
                print(f"Error creating {config_path}: {e}")


async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Market Data Agent Configuration Tool")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    parser.add_argument("--env", help="Environment (development, staging, production)")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # View command
    view_parser = subparsers.add_parser("view", help="View configuration")
    view_parser.add_argument("--path", help="Specific configuration path to view")
    view_parser.add_argument("--format", choices=["yaml", "json"], default="yaml", help="Output format")

    # Set command
    set_parser = subparsers.add_parser("set", help="Set configuration value")
    set_parser.add_argument("path", help="Configuration path (e.g., api.port)")
    set_parser.add_argument("value", help="New value")
    set_parser.add_argument("--env", help="Target environment")

    # Validate command
    subparsers.add_parser("validate", help="Validate configuration")

    # History command
    history_parser = subparsers.add_parser("history", help="Show configuration history")
    history_parser.add_argument("--limit", type=int, default=10, help="Number of entries to show")

    # Changes command
    changes_parser = subparsers.add_parser("changes", help="Show configuration changes")
    changes_parser.add_argument("--limit", type=int, default=20, help="Number of changes to show")

    # Watch command
    subparsers.add_parser("watch", help="Watch for configuration changes")

    # Test command
    subparsers.add_parser("test-envs", help="Test all environment configurations")

    # Create samples command
    subparsers.add_parser("create-samples", help="Create sample configuration files")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    tool = ConfigTool()

    try:
        if args.command == "create-samples":
            await tool.create_sample_configs()
        elif args.command == "test-envs":
            await tool.test_environments()
        else:
            await tool.initialize(args.config, args.env)

            if args.command == "view":
                await tool.view_config(args.path, args.format)
            elif args.command == "set":
                await tool.set_config(args.path, args.value, getattr(args, 'env', None))
            elif args.command == "validate":
                await tool.validate_config()
            elif args.command == "history":
                await tool.show_history(args.limit)
            elif args.command == "changes":
                await tool.show_changes(args.limit)
            elif args.command == "watch":
                await tool.watch_config()

    except KeyboardInterrupt:
        print("\nOperation cancelled")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        await tool.shutdown()


if __name__ == "__main__":
    asyncio.run(main())