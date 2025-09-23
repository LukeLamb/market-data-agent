"""
Vault Configuration Tests for Market Data Agent
Phase 4 Step 3: Configuration Management & Environment Automation
"""

import pytest
import subprocess
import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, List


class TestVaultConfiguration:
    """Test HashiCorp Vault configuration"""

    @classmethod
    def setup_class(cls):
        """Set up test environment"""
        cls.vault_dir = Path(__file__).parent.parent.parent / "vault"
        cls.environments = ["dev", "staging", "prod"]

    def test_vault_config_file_exists(self):
        """Test that Vault configuration file exists"""
        vault_config = self.vault_dir / "vault-config.hcl"
        assert vault_config.exists(), "Vault configuration file does not exist"

    def test_vault_config_syntax(self):
        """Test Vault configuration syntax"""
        vault_config = self.vault_dir / "vault-config.hcl"

        # Basic syntax validation using vault validate command
        result = subprocess.run([
            "vault", "validate", str(vault_config)
        ], capture_output=True, text=True)

        # If vault command is available, check syntax
        if result.returncode != 127:  # Command not found
            assert result.returncode == 0, \
                   f"Vault configuration syntax error: {result.stderr}"

    def test_vault_config_security_settings(self):
        """Test Vault security configurations"""
        vault_config = self.vault_dir / "vault-config.hcl"

        with open(vault_config, 'r') as f:
            content = f.read()

        # Check for TLS configuration
        assert 'listener "tcp"' in content, "Should have TCP listener configured"
        assert 'tls_cert_file' in content, "Should have TLS certificate configured"
        assert 'tls_key_file' in content, "Should have TLS key configured"

        # Check for seal configuration (AWS KMS recommended for production)
        assert 'seal "awskms"' in content, "Should have AWS KMS seal configured"

        # Check UI is enabled
        assert 'ui = true' in content, "Vault UI should be enabled"

        # Check that mlock is disabled for containers
        assert 'disable_mlock = true' in content, \
               "mlock should be disabled for containerized deployments"

    def test_vault_storage_configuration(self):
        """Test Vault storage backend configuration"""
        vault_config = self.vault_dir / "vault-config.hcl"

        with open(vault_config, 'r') as f:
            content = f.read()

        # Should have at least one storage backend configured
        storage_backends = ['storage "consul"', 'storage "file"', 'storage "s3"']
        has_storage = any(backend in content for backend in storage_backends)

        assert has_storage, "At least one storage backend should be configured"

    def test_vault_kubernetes_manifests(self):
        """Test Vault Kubernetes manifest files"""
        k8s_dir = self.vault_dir / "kubernetes"
        assert k8s_dir.exists(), "Vault Kubernetes manifests directory should exist"

        # Check for essential manifest files
        essential_files = [
            "vault-namespace.yaml",
            "vault-serviceaccount.yaml"
        ]

        for filename in essential_files:
            manifest_file = k8s_dir / filename
            assert manifest_file.exists(), f"Essential manifest {filename} is missing"

    def test_vault_manifest_syntax(self):
        """Test Vault Kubernetes manifest YAML syntax"""
        k8s_dir = self.vault_dir / "kubernetes"

        if k8s_dir.exists():
            for yaml_file in k8s_dir.glob("*.yaml"):
                with open(yaml_file, 'r') as f:
                    try:
                        yaml.safe_load_all(f)
                    except yaml.YAMLError as e:
                        pytest.fail(f"YAML syntax error in {yaml_file}: {e}")

    def test_vault_kubernetes_dry_run(self):
        """Test Vault Kubernetes manifests with dry-run"""
        k8s_dir = self.vault_dir / "kubernetes"

        if k8s_dir.exists():
            for yaml_file in k8s_dir.glob("*.yaml"):
                result = subprocess.run([
                    "kubectl", "apply", "--dry-run=client", "-f", str(yaml_file)
                ], capture_output=True, text=True)

                # If kubectl is available, check dry-run
                if result.returncode != 127:  # Command not found
                    assert result.returncode == 0, \
                           f"Kubernetes manifest validation failed for {yaml_file}: {result.stderr}"

    def test_vault_rbac_configuration(self):
        """Test Vault RBAC configuration"""
        serviceaccount_file = self.vault_dir / "kubernetes" / "vault-serviceaccount.yaml"

        if serviceaccount_file.exists():
            with open(serviceaccount_file, 'r') as f:
                manifests = list(yaml.safe_load_all(f))

            # Check for ServiceAccount
            service_accounts = [m for m in manifests if m.get("kind") == "ServiceAccount"]
            assert len(service_accounts) > 0, "Should have ServiceAccount defined"

            # Check for ClusterRole
            cluster_roles = [m for m in manifests if m.get("kind") == "ClusterRole"]
            assert len(cluster_roles) > 0, "Should have ClusterRole defined"

            # Check for ClusterRoleBinding
            cluster_role_bindings = [m for m in manifests if m.get("kind") == "ClusterRoleBinding"]
            assert len(cluster_role_bindings) > 0, "Should have ClusterRoleBinding defined"

    def test_vault_namespace_configuration(self):
        """Test Vault namespace configuration"""
        namespace_file = self.vault_dir / "kubernetes" / "vault-namespace.yaml"

        if namespace_file.exists():
            with open(namespace_file, 'r') as f:
                manifests = list(yaml.safe_load_all(f))

            # Check for Namespace
            namespaces = [m for m in manifests if m.get("kind") == "Namespace"]
            assert len(namespaces) > 0, "Should have Namespace defined"

            # Check namespace name
            vault_namespace = namespaces[0]
            assert vault_namespace["metadata"]["name"] == "vault", \
                   "Vault namespace should be named 'vault'"

            # Check for ResourceQuota
            resource_quotas = [m for m in manifests if m.get("kind") == "ResourceQuota"]
            assert len(resource_quotas) > 0, "Should have ResourceQuota for vault namespace"

            # Check for LimitRange
            limit_ranges = [m for m in manifests if m.get("kind") == "LimitRange"]
            assert len(limit_ranges) > 0, "Should have LimitRange for vault namespace"

    def test_vault_helm_values(self):
        """Test Vault Helm values configuration"""
        helm_values = self.vault_dir / "helm" / "values.yaml"

        if helm_values.exists():
            with open(helm_values, 'r') as f:
                try:
                    values = yaml.safe_load(f)
                except yaml.YAMLError as e:
                    pytest.fail(f"YAML syntax error in Helm values: {e}")

            # Check essential Helm values
            assert "server" in values, "Should have server configuration"
            assert "injector" in values, "Should have injector configuration"

            # Check high availability configuration
            server_config = values.get("server", {})
            if "ha" in server_config:
                ha_config = server_config["ha"]
                assert ha_config.get("enabled") is True, "HA should be enabled"
                assert ha_config.get("replicas", 0) >= 3, "Should have at least 3 replicas for HA"

    def test_vault_init_script(self):
        """Test Vault initialization script"""
        init_script = self.vault_dir / "scripts" / "vault-init.sh"

        if init_script.exists():
            # Basic syntax check - ensure it's a valid shell script
            with open(init_script, 'r') as f:
                content = f.read()

            # Check shebang
            assert content.startswith("#!/bin/bash"), "Script should have bash shebang"

            # Check for error handling
            assert "set -euo pipefail" in content, "Script should have error handling"

            # Check for essential functions
            essential_functions = [
                "initialize_vault",
                "configure_kubernetes_auth",
                "create_policies",
                "create_roles"
            ]

            for function in essential_functions:
                assert function in content, f"Script should have {function} function"

    def test_vault_policy_definitions(self):
        """Test that Vault policies are properly defined in init script"""
        init_script = self.vault_dir / "scripts" / "vault-init.sh"

        if init_script.exists():
            with open(init_script, 'r') as f:
                content = f.read()

            # Check for market-data-agent policy
            assert "market-data-agent" in content, \
                   "Should define market-data-agent policy"

            # Check for monitoring policy
            assert "monitoring" in content, "Should define monitoring policy"

            # Check for secret paths
            secret_paths = [
                "secret/data/market-data/",
                "secret/data/database/",
                "secret/data/api-keys/"
            ]

            for path in secret_paths:
                assert path in content, f"Should define access to {path}"

    def test_vault_auth_methods(self):
        """Test Vault authentication methods configuration"""
        init_script = self.vault_dir / "scripts" / "vault-init.sh"

        if init_script.exists():
            with open(init_script, 'r') as f:
                content = f.read()

            # Check for Kubernetes auth method
            assert "vault auth enable kubernetes" in content, \
                   "Should enable Kubernetes auth method"

            # Check for auth configuration
            assert "vault write auth/kubernetes/config" in content, \
                   "Should configure Kubernetes auth"

    def test_vault_secret_engines(self):
        """Test Vault secret engines configuration"""
        init_script = self.vault_dir / "scripts" / "vault-init.sh"

        if init_script.exists():
            with open(init_script, 'r') as f:
                content = f.read()

            # Check for KV v2 secret engine
            assert "vault secrets enable" in content and "kv-v2" in content, \
                   "Should enable KV v2 secret engine"

            # Check for database secret engine
            assert "vault secrets enable database" in content, \
                   "Should enable database secret engine"

    def test_vault_security_practices(self):
        """Test Vault security best practices"""
        vault_config = self.vault_dir / "vault-config.hcl"

        with open(vault_config, 'r') as f:
            content = f.read()

        # Check for telemetry configuration
        assert "telemetry" in content, "Should have telemetry configured"

        # Check for log level configuration
        assert "log_level" in content, "Should have log level configured"

        # Check for cluster address
        assert "cluster_addr" in content, "Should have cluster address configured"

        # Check for API address
        assert "api_addr" in content, "Should have API address configured"

    def test_vault_kubernetes_integration(self):
        """Test Vault Kubernetes integration configuration"""
        serviceaccount_file = self.vault_dir / "kubernetes" / "vault-serviceaccount.yaml"

        if serviceaccount_file.exists():
            with open(serviceaccount_file, 'r') as f:
                manifests = list(yaml.safe_load_all(f))

            # Check for market-data-agent service account
            market_data_sa = None
            for manifest in manifests:
                if (manifest.get("kind") == "ServiceAccount" and
                    manifest.get("metadata", {}).get("name") == "market-data-agent-vault"):
                    market_data_sa = manifest
                    break

            assert market_data_sa is not None, \
                   "Should have market-data-agent-vault ServiceAccount"

            # Check for proper annotations
            annotations = market_data_sa.get("metadata", {}).get("annotations", {})
            assert "vault.hashicorp.com/auth-path" in annotations, \
                   "ServiceAccount should have Vault auth path annotation"
            assert "vault.hashicorp.com/role" in annotations, \
                   "ServiceAccount should have Vault role annotation"

    def test_vault_high_availability_config(self):
        """Test Vault high availability configuration"""
        helm_values = self.vault_dir / "helm" / "values.yaml"

        if helm_values.exists():
            with open(helm_values, 'r') as f:
                values = yaml.safe_load(f)

            server_config = values.get("server", {})

            # Check HA configuration
            if "ha" in server_config:
                ha_config = server_config["ha"]
                assert ha_config.get("enabled") is True, "HA should be enabled"

                # Check Raft storage
                if "raft" in ha_config:
                    raft_config = ha_config["raft"]
                    assert raft_config.get("enabled") is True, "Raft should be enabled"
                    assert "config" in raft_config, "Raft should have configuration"

    def test_vault_monitoring_integration(self):
        """Test Vault monitoring integration"""
        vault_config = self.vault_dir / "vault-config.hcl"

        with open(vault_config, 'r') as f:
            content = f.read()

        # Check for Prometheus telemetry
        assert "prometheus_retention_time" in content, \
               "Should have Prometheus telemetry configured"

        # Check Helm values for ServiceMonitor
        helm_values = self.vault_dir / "helm" / "values.yaml"
        if helm_values.exists():
            with open(helm_values, 'r') as f:
                values = yaml.safe_load(f)

            server_config = values.get("server", {})
            if "serviceMonitor" in server_config:
                service_monitor = server_config["serviceMonitor"]
                assert service_monitor.get("enabled") is True, \
                       "ServiceMonitor should be enabled for monitoring"

    def test_vault_backup_considerations(self):
        """Test Vault backup and recovery considerations"""
        # Check if there are backup-related configurations
        init_script = self.vault_dir / "scripts" / "vault-init.sh"

        if init_script.exists():
            with open(init_script, 'r') as f:
                content = f.read()

            # Check for key storage warnings
            assert "IMPORTANT" in content or "WARNING" in content, \
                   "Script should have warnings about key storage"

            # Check for secure key handling
            assert "vault-init-keys.json" in content, \
                   "Script should handle init keys securely"

    def test_vault_environment_considerations(self):
        """Test environment-specific Vault considerations"""
        vault_config = self.vault_dir / "vault-config.hcl"

        with open(vault_config, 'r') as f:
            content = f.read()

        # Should have comments about production considerations
        if "production" in content.lower() or "prod" in content.lower():
            # Production configurations should be more secure
            assert "awskms" in content.lower(), \
                   "Production should use AWS KMS for seal"