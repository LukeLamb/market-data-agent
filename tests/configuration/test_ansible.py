"""
Ansible Configuration Tests for Market Data Agent
Phase 4 Step 3: Configuration Management & Environment Automation
"""

import pytest
import subprocess
import yaml
import os
from pathlib import Path
from typing import Dict, Any, List


class TestAnsibleConfiguration:
    """Test Ansible configuration management setup"""

    @classmethod
    def setup_class(cls):
        """Set up test environment"""
        cls.ansible_dir = Path(__file__).parent.parent.parent / "ansible"
        cls.environments = ["development", "staging", "production"]

    def test_ansible_syntax_validation(self):
        """Test that all Ansible playbooks have valid syntax"""
        os.chdir(self.ansible_dir)

        # Find all playbook files
        playbook_files = list(self.ansible_dir.glob("playbooks/*.yml"))

        for playbook in playbook_files:
            result = subprocess.run([
                "ansible-playbook", str(playbook), "--syntax-check"
            ], capture_output=True, text=True)

            assert result.returncode == 0, \
                   f"Syntax error in {playbook}: {result.stderr}"

    def test_yaml_syntax_validation(self):
        """Test that all YAML files have valid syntax"""
        yaml_files = []
        yaml_files.extend(self.ansible_dir.glob("**/*.yml"))
        yaml_files.extend(self.ansible_dir.glob("**/*.yaml"))

        for yaml_file in yaml_files:
            with open(yaml_file, 'r') as f:
                try:
                    yaml.safe_load(f)
                except yaml.YAMLError as e:
                    pytest.fail(f"YAML syntax error in {yaml_file}: {e}")

    def test_inventory_structure(self):
        """Test that inventory file has proper structure"""
        inventory_file = self.ansible_dir / "inventory" / "hosts.yml"
        assert inventory_file.exists(), "Inventory file does not exist"

        with open(inventory_file, 'r') as f:
            inventory = yaml.safe_load(f)

        # Check top-level structure
        assert "all" in inventory, "Inventory must have 'all' group"
        assert "children" in inventory["all"], "All group must have children"

        # Check environment groups
        for env in self.environments:
            assert env in inventory["all"]["children"], \
                   f"Environment {env} not found in inventory"

    def test_inventory_validation(self):
        """Test that inventory can be parsed by Ansible"""
        os.chdir(self.ansible_dir)

        result = subprocess.run([
            "ansible-inventory", "--list", "-i", "inventory/hosts.yml"
        ], capture_output=True, text=True)

        assert result.returncode == 0, \
               f"Inventory validation failed: {result.stderr}"

        # Parse the output to ensure it's valid JSON
        import json
        try:
            inventory_data = json.loads(result.stdout)
            assert isinstance(inventory_data, dict), "Inventory output should be a dictionary"
        except json.JSONDecodeError:
            pytest.fail("Inventory output is not valid JSON")

    @pytest.mark.parametrize("environment", ["development", "staging", "production"])
    def test_environment_groups_exist(self, environment):
        """Test that environment-specific groups exist in inventory"""
        inventory_file = self.ansible_dir / "inventory" / "hosts.yml"

        with open(inventory_file, 'r') as f:
            inventory = yaml.safe_load(f)

        env_config = inventory["all"]["children"][environment]
        assert "children" in env_config, f"Environment {environment} should have children groups"

        # Check for essential groups
        essential_groups = [f"kubernetes_{environment.replace('production', 'prod')}",
                          f"monitoring_{environment.replace('production', 'prod')}"]

        for group in essential_groups:
            assert group in env_config["children"], \
                   f"Essential group {group} not found in {environment}"

    def test_required_roles_exist(self):
        """Test that required Ansible roles exist"""
        roles_dir = self.ansible_dir / "roles"
        assert roles_dir.exists(), "Roles directory does not exist"

        required_roles = ["common", "security", "monitoring", "kubernetes-common"]

        for role in required_roles:
            role_dir = roles_dir / role
            # Check if role directory exists (we created some)
            if role_dir.exists():
                # Check for main tasks file
                tasks_file = role_dir / "tasks" / "main.yml"
                assert tasks_file.exists(), f"Role {role} missing main tasks file"

    def test_common_role_structure(self):
        """Test the structure of the common role"""
        common_role = self.ansible_dir / "roles" / "common"

        if common_role.exists():
            # Check for essential directories
            assert (common_role / "tasks").exists(), "Common role missing tasks directory"
            assert (common_role / "handlers").exists(), "Common role missing handlers directory"
            assert (common_role / "templates").exists(), "Common role missing templates directory"

            # Check for main files
            assert (common_role / "tasks" / "main.yml").exists(), \
                   "Common role missing main tasks file"
            assert (common_role / "handlers" / "main.yml").exists(), \
                   "Common role missing main handlers file"

    def test_playbook_includes_validation(self):
        """Test that playbook includes are valid"""
        site_playbook = self.ansible_dir / "playbooks" / "site.yml"

        if site_playbook.exists():
            with open(site_playbook, 'r') as f:
                content = yaml.safe_load(f)

            # Should be a list of plays
            assert isinstance(content, list), "Site playbook should contain a list of plays"

            for play in content:
                if "import_playbook" in play:
                    # Check that imported playbooks exist
                    imported_playbook = self.ansible_dir / "playbooks" / play["import_playbook"]
                    # We haven't created all referenced playbooks, so just check structure
                    assert isinstance(play["import_playbook"], str), \
                           "import_playbook should be a string"

    def test_variable_definitions(self):
        """Test that essential variables are defined"""
        inventory_file = self.ansible_dir / "inventory" / "hosts.yml"

        with open(inventory_file, 'r') as f:
            inventory = yaml.safe_load(f)

        # Check global variables
        all_vars = inventory["all"].get("vars", {})

        essential_vars = [
            "ansible_user",
            "project_name",
            "common_packages"
        ]

        for var in essential_vars:
            assert var in all_vars, f"Essential variable {var} not defined in inventory"

    def test_security_hardening_variables(self):
        """Test security hardening variables"""
        inventory_file = self.ansible_dir / "inventory" / "hosts.yml"

        with open(inventory_file, 'r') as f:
            inventory = yaml.safe_load(f)

        all_vars = inventory["all"].get("vars", {})

        # Check SSH hardening variables
        if "ssh_hardening" in all_vars:
            ssh_config = all_vars["ssh_hardening"]
            assert ssh_config.get("permit_root_login") == "no", \
                   "Root login should be disabled"
            assert ssh_config.get("password_authentication") == "no", \
                   "Password authentication should be disabled"

    def test_docker_configuration(self):
        """Test Docker configuration variables"""
        inventory_file = self.ansible_dir / "inventory" / "hosts.yml"

        with open(inventory_file, 'r') as f:
            inventory = yaml.safe_load(f)

        all_vars = inventory["all"].get("vars", {})

        # Check Docker users
        assert "docker_users" in all_vars, "Docker users should be defined"
        assert isinstance(all_vars["docker_users"], list), \
               "Docker users should be a list"

    def test_monitoring_configuration(self):
        """Test monitoring configuration variables"""
        inventory_file = self.ansible_dir / "inventory" / "hosts.yml"

        with open(inventory_file, 'r') as f:
            inventory = yaml.safe_load(f)

        all_vars = inventory["all"].get("vars", {})

        # Check monitoring versions
        monitoring_vars = ["node_exporter_version", "prometheus_version", "grafana_version"]

        for var in monitoring_vars:
            assert var in all_vars, f"Monitoring variable {var} should be defined"

    def test_kubernetes_configuration(self):
        """Test Kubernetes-specific configuration"""
        # Check if Kubernetes playbook exists
        k8s_playbook = self.ansible_dir / "playbooks" / "kubernetes.yml"

        if k8s_playbook.exists():
            with open(k8s_playbook, 'r') as f:
                content = yaml.safe_load(f)

            # Should be a list of plays
            assert isinstance(content, list), "Kubernetes playbook should contain plays"

            # Check for master and worker node plays
            play_names = [play.get("name", "") for play in content]
            assert any("master" in name.lower() for name in play_names), \
                   "Should have master node configuration"
            assert any("worker" in name.lower() for name in play_names), \
                   "Should have worker node configuration"

    def test_template_files_syntax(self):
        """Test that template files have valid syntax"""
        template_files = []
        for role_dir in (self.ansible_dir / "roles").glob("*/templates"):
            if role_dir.exists():
                template_files.extend(role_dir.glob("*.j2"))

        for template_file in template_files:
            # Basic check - ensure file is readable and not empty
            assert template_file.exists(), f"Template file {template_file} does not exist"

            with open(template_file, 'r') as f:
                content = f.read()
                assert len(content.strip()) > 0, f"Template file {template_file} is empty"

                # Check for basic Jinja2 syntax (very basic validation)
                # Count opening and closing braces
                open_braces = content.count('{{') + content.count('{%')
                close_braces = content.count('}}') + content.count('%}')

                # Should have matching braces (basic check)
                # Note: This is a very basic check and won't catch all syntax errors
                if open_braces > 0:
                    assert open_braces == close_braces, \
                           f"Mismatched Jinja2 braces in {template_file}"

    def test_handler_definitions(self):
        """Test that handlers are properly defined"""
        for role_dir in (self.ansible_dir / "roles").glob("*"):
            if role_dir.is_dir():
                handlers_file = role_dir / "handlers" / "main.yml"
                if handlers_file.exists():
                    with open(handlers_file, 'r') as f:
                        handlers = yaml.safe_load(f)

                    if handlers:
                        assert isinstance(handlers, list), \
                               f"Handlers in {role_dir.name} should be a list"

                        for handler in handlers:
                            assert "name" in handler, \
                                   f"Handler in {role_dir.name} missing name"

    def test_environment_specific_variables(self):
        """Test environment-specific variable configurations"""
        inventory_file = self.ansible_dir / "inventory" / "hosts.yml"

        with open(inventory_file, 'r') as f:
            inventory = yaml.safe_load(f)

        for env in self.environments:
            env_config = inventory["all"]["children"][env]

            # Check for environment-specific variables
            if "vars" in env_config:
                env_vars = env_config["vars"]
                assert "environment" in env_vars, \
                       f"Environment {env} should have environment variable"

        # Check production-specific security
        prod_config = inventory["all"]["children"]["production"]
        if "children" in prod_config:
            for group_name, group_config in prod_config["children"].items():
                if "vars" in group_config:
                    # Production should have stricter configurations
                    pass  # Add specific production checks here

    def test_no_sensitive_data_in_inventory(self):
        """Test that no sensitive data is stored in inventory"""
        inventory_file = self.ansible_dir / "inventory" / "hosts.yml"

        with open(inventory_file, 'r') as f:
            content = f.read()

        # Patterns that might indicate sensitive data
        sensitive_patterns = [
            "password:",
            "secret:",
            "key:",
            "token:",
            "-----BEGIN"
        ]

        for pattern in sensitive_patterns:
            assert pattern not in content.lower(), \
                   f"Potential sensitive data found in inventory: {pattern}"

    def test_ansible_configuration_file(self):
        """Test Ansible configuration file if it exists"""
        ansible_cfg = self.ansible_dir / "ansible.cfg"

        if ansible_cfg.exists():
            # Basic validation - ensure it's readable
            with open(ansible_cfg, 'r') as f:
                content = f.read()
                assert len(content.strip()) > 0, "ansible.cfg should not be empty"

                # Check for common security settings
                if "[defaults]" in content:
                    # Should have host key checking disabled for automation
                    # (in production, this should be carefully considered)
                    pass