"""
Configuration Management Integration Tests for Market Data Agent
Phase 4 Step 3: Configuration Management & Environment Automation
"""

import pytest
import subprocess
import json
import yaml
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List


class TestConfigurationIntegration:
    """Test integration between configuration management components"""

    @classmethod
    def setup_class(cls):
        """Set up test environment"""
        cls.project_root = Path(__file__).parent.parent.parent
        cls.terraform_dir = cls.project_root / "terraform"
        cls.ansible_dir = cls.project_root / "ansible"
        cls.vault_dir = cls.project_root / "vault"
        cls.scripts_dir = cls.project_root / "scripts"

    def test_validation_script_exists(self):
        """Test that infrastructure validation script exists and is executable"""
        validation_script = self.scripts_dir / "validate-infrastructure.sh"
        assert validation_script.exists(), "Infrastructure validation script does not exist"

        # Check if script is executable (on Unix systems)
        if os.name != 'nt':  # Not Windows
            assert os.access(validation_script, os.X_OK), \
                   "Validation script should be executable"

    def test_drift_detection_script_exists(self):
        """Test that drift detection script exists"""
        drift_script = self.scripts_dir / "drift-detection.py"
        assert drift_script.exists(), "Drift detection script does not exist"

        # Check if it's a valid Python script
        with open(drift_script, 'r') as f:
            content = f.read()
            assert content.startswith("#!/usr/bin/env python3"), \
                   "Script should have Python shebang"

    def test_terraform_ansible_integration(self):
        """Test integration between Terraform and Ansible configurations"""
        # Get Terraform outputs that should match Ansible inventory
        terraform_vars = self._get_terraform_variables()
        ansible_inventory = self._get_ansible_inventory()

        # Check environment consistency
        for env in ["dev", "staging", "prod"]:
            # Check that environments exist in both
            assert self._terraform_has_environment(env), \
                   f"Environment {env} not configured in Terraform"
            assert self._ansible_has_environment(env), \
                   f"Environment {env} not configured in Ansible"

    def test_vault_kubernetes_integration(self):
        """Test integration between Vault and Kubernetes configurations"""
        # Check Vault namespace in Kubernetes manifests
        vault_namespace_file = self.vault_dir / "kubernetes" / "vault-namespace.yaml"

        if vault_namespace_file.exists():
            with open(vault_namespace_file, 'r') as f:
                manifests = list(yaml.safe_load_all(f))

            namespace_manifest = None
            for manifest in manifests:
                if manifest.get("kind") == "Namespace":
                    namespace_manifest = manifest
                    break

            assert namespace_manifest is not None, "Vault namespace manifest not found"
            assert namespace_manifest["metadata"]["name"] == "vault", \
                   "Vault namespace should be named 'vault'"

        # Check ServiceAccount integration
        vault_sa_file = self.vault_dir / "kubernetes" / "vault-serviceaccount.yaml"
        if vault_sa_file.exists():
            with open(vault_sa_file, 'r') as f:
                manifests = list(yaml.safe_load_all(f))

            # Check for market-data namespace ServiceAccount
            market_data_sa = None
            for manifest in manifests:
                if (manifest.get("kind") == "ServiceAccount" and
                    manifest.get("metadata", {}).get("namespace") == "market-data"):
                    market_data_sa = manifest
                    break

            if market_data_sa:
                annotations = market_data_sa.get("metadata", {}).get("annotations", {})
                assert "vault.hashicorp.com/role" in annotations, \
                       "Market data ServiceAccount should have Vault role annotation"

    def test_environment_consistency(self):
        """Test consistency across all environment configurations"""
        environments = ["dev", "staging", "prod"]

        for env in environments:
            # Check Terraform environment file
            terraform_env_file = self.terraform_dir / "environments" / f"{env}.tfvars"
            assert terraform_env_file.exists(), f"Terraform environment file for {env} missing"

            # Check backend configuration
            backend_file = self.terraform_dir / "environments" / f"backend-{env}.conf"
            assert backend_file.exists(), f"Terraform backend file for {env} missing"

            # Validate environment-specific values
            with open(terraform_env_file, 'r') as f:
                content = f.read()
                assert f'environment = "{env}"' in content, \
                       f"Environment variable not set correctly in {env}.tfvars"

    def test_security_consistency(self):
        """Test security configuration consistency across components"""
        # Check production security settings
        prod_tfvars = self.terraform_dir / "environments" / "prod.tfvars"

        if prod_tfvars.exists():
            with open(prod_tfvars, 'r') as f:
                content = f.read()

            # Production should have stricter security
            assert 'cluster_endpoint_public_access  = false' in content or \
                   'cluster_endpoint_public_access = false' in content, \
                   "Production cluster should not have public endpoint access"

        # Check Ansible security settings
        ansible_inventory = self.ansible_dir / "inventory" / "hosts.yml"
        if ansible_inventory.exists():
            with open(ansible_inventory, 'r') as f:
                inventory = yaml.safe_load(f)

            all_vars = inventory.get("all", {}).get("vars", {})
            if "ssh_hardening" in all_vars:
                ssh_config = all_vars["ssh_hardening"]
                assert ssh_config.get("permit_root_login") == "no", \
                       "SSH root login should be disabled"

    def test_monitoring_integration(self):
        """Test monitoring configuration integration"""
        # Check that monitoring is configured consistently
        ansible_inventory = self.ansible_dir / "inventory" / "hosts.yml"

        if ansible_inventory.exists():
            with open(ansible_inventory, 'r') as f:
                inventory = yaml.safe_load(f)

            # Check for monitoring variables
            all_vars = inventory.get("all", {}).get("vars", {})
            monitoring_vars = ["node_exporter_version", "prometheus_version"]

            for var in monitoring_vars:
                assert var in all_vars, f"Monitoring variable {var} should be defined"

    def test_network_configuration_consistency(self):
        """Test network configuration consistency"""
        # Check VPC CIDR consistency
        environments = ["dev", "staging", "prod"]
        cidrs = {}

        for env in environments:
            tfvars_file = self.terraform_dir / "environments" / f"{env}.tfvars"
            if tfvars_file.exists():
                with open(tfvars_file, 'r') as f:
                    content = f.read()

                # Extract VPC CIDR
                for line in content.split('\n'):
                    if 'vpc_cidr' in line and '=' in line:
                        cidr = line.split('=')[1].strip().strip('"')
                        cidrs[env] = cidr
                        break

        # Ensure all CIDRs are unique
        if len(cidrs) > 1:
            cidr_values = list(cidrs.values())
            assert len(set(cidr_values)) == len(cidr_values), \
                   f"VPC CIDRs must be unique across environments: {cidrs}"

    def test_backup_configuration_consistency(self):
        """Test backup configuration consistency"""
        environments = ["dev", "staging", "prod"]

        for env in environments:
            tfvars_file = self.terraform_dir / "environments" / f"{env}.tfvars"
            if tfvars_file.exists():
                with open(tfvars_file, 'r') as f:
                    content = f.read()

                # Check backup settings based on environment
                if env == "prod":
                    # Production should have longer retention
                    assert "backup_retention_days   = 30" in content or \
                           "rds_backup_retention_period = 30" in content, \
                           "Production should have 30-day backup retention"
                elif env == "dev":
                    # Development can have shorter retention
                    assert "backup_retention_days   = 3" in content or \
                           "rds_backup_retention_period = 3" in content, \
                           "Development can have 3-day backup retention"

    def test_scaling_configuration_consistency(self):
        """Test that scaling configurations are appropriate for environments"""
        environments = ["dev", "staging", "prod"]

        for env in environments:
            tfvars_file = self.terraform_dir / "environments" / f"{env}.tfvars"
            if tfvars_file.exists():
                with open(tfvars_file, 'r') as f:
                    content = f.read()

                if env == "dev":
                    # Dev should have smaller instances and lower counts
                    assert "t3.small" in content or "t3.micro" in content, \
                           "Development should use smaller instances"
                elif env == "prod":
                    # Production should have larger instances
                    assert any(instance in content for instance in
                              ["t3.large", "t3.xlarge", "c5.large", "r5.large"]), \
                           "Production should use appropriately sized instances"

    def test_configuration_file_permissions(self):
        """Test that configuration files have appropriate permissions"""
        # Check that sensitive files don't have world-readable permissions
        sensitive_files = [
            self.vault_dir / "vault-config.hcl",
        ]

        for file_path in sensitive_files:
            if file_path.exists() and os.name != 'nt':  # Not Windows
                file_stat = file_path.stat()
                # Check that file is not world-readable (others can't read)
                assert not (file_stat.st_mode & 0o004), \
                       f"Sensitive file {file_path} should not be world-readable"

    def test_documentation_consistency(self):
        """Test that documentation is consistent with configuration"""
        # Check that README or documentation mentions all environments
        readme_files = list(self.project_root.glob("README*")) + \
                      list(self.project_root.glob("docs/README*"))

        if readme_files:
            for readme in readme_files:
                with open(readme, 'r') as f:
                    content = f.read().lower()

                # Should mention key components
                key_components = ["terraform", "ansible", "vault", "kubernetes"]
                for component in key_components:
                    assert component in content, \
                           f"README should mention {component}"

    def _get_terraform_variables(self) -> Dict[str, Any]:
        """Get Terraform variables for analysis"""
        variables = {}
        variables_file = self.terraform_dir / "variables.tf"

        if variables_file.exists():
            # This is a simplified parser - in real scenarios you might use HCL parser
            with open(variables_file, 'r') as f:
                content = f.read()
                # Extract variable names (simplified)
                import re
                var_matches = re.findall(r'variable "([^"]+)"', content)
                for var in var_matches:
                    variables[var] = True

        return variables

    def _get_ansible_inventory(self) -> Dict[str, Any]:
        """Get Ansible inventory for analysis"""
        inventory_file = self.ansible_dir / "inventory" / "hosts.yml"

        if inventory_file.exists():
            with open(inventory_file, 'r') as f:
                return yaml.safe_load(f)

        return {}

    def _terraform_has_environment(self, environment: str) -> bool:
        """Check if Terraform has configuration for environment"""
        env_file = self.terraform_dir / "environments" / f"{environment}.tfvars"
        return env_file.exists()

    def _ansible_has_environment(self, environment: str) -> bool:
        """Check if Ansible has configuration for environment"""
        inventory = self._get_ansible_inventory()
        env_mapping = {
            "dev": "development",
            "staging": "staging",
            "prod": "production"
        }

        ansible_env = env_mapping.get(environment, environment)
        return ansible_env in inventory.get("all", {}).get("children", {})

    def test_end_to_end_validation(self):
        """Test end-to-end configuration validation"""
        # This would run the actual validation script if available
        validation_script = self.scripts_dir / "validate-infrastructure.sh"

        if validation_script.exists():
            # Test script exists and has proper structure
            with open(validation_script, 'r') as f:
                content = f.read()

            # Check for essential functions
            essential_functions = [
                "validate_terraform",
                "validate_ansible",
                "validate_kubernetes",
                "validate_vault"
            ]

            for function in essential_functions:
                assert function in content, \
                       f"Validation script should have {function} function"

    def test_drift_detection_capabilities(self):
        """Test drift detection script capabilities"""
        drift_script = self.scripts_dir / "drift-detection.py"

        if drift_script.exists():
            with open(drift_script, 'r') as f:
                content = f.read()

            # Check for essential classes and methods
            essential_components = [
                "class ConfigurationDriftDetector",
                "detect_terraform_drift",
                "detect_kubernetes_drift",
                "detect_vault_drift",
                "generate_report"
            ]

            for component in essential_components:
                assert component in content, \
                       f"Drift detection script should have {component}"