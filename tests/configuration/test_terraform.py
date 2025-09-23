"""
Terraform Configuration Tests for Market Data Agent
Phase 4 Step 3: Configuration Management & Environment Automation
"""

import pytest
import subprocess
import tempfile
import json
import os
from pathlib import Path
from typing import Dict, Any, List


class TestTerraformConfiguration:
    """Test Terraform infrastructure configuration"""

    @classmethod
    def setup_class(cls):
        """Set up test environment"""
        cls.terraform_dir = Path(__file__).parent.parent.parent / "terraform"
        cls.environments = ["dev", "staging", "prod"]

    def test_terraform_syntax_validation(self):
        """Test that all Terraform files have valid syntax"""
        os.chdir(self.terraform_dir)

        # Initialize Terraform
        result = subprocess.run(
            ["terraform", "init", "-backend=false"],
            capture_output=True, text=True
        )
        assert result.returncode == 0, f"Terraform init failed: {result.stderr}"

        # Validate syntax
        result = subprocess.run(
            ["terraform", "validate"],
            capture_output=True, text=True
        )
        assert result.returncode == 0, f"Terraform validation failed: {result.stderr}"

    def test_terraform_formatting(self):
        """Test that Terraform files are properly formatted"""
        os.chdir(self.terraform_dir)

        result = subprocess.run(
            ["terraform", "fmt", "-check=true", "-diff=true"],
            capture_output=True, text=True
        )
        assert result.returncode == 0, f"Terraform files are not properly formatted: {result.stdout}"

    @pytest.mark.parametrize("environment", ["dev", "staging", "prod"])
    def test_environment_tfvars_exists(self, environment):
        """Test that environment-specific tfvars files exist"""
        tfvars_file = self.terraform_dir / "environments" / f"{environment}.tfvars"
        assert tfvars_file.exists(), f"Missing tfvars file for {environment}"

    @pytest.mark.parametrize("environment", ["dev", "staging", "prod"])
    def test_terraform_plan_generation(self, environment):
        """Test that Terraform plans can be generated for each environment"""
        os.chdir(self.terraform_dir)

        tfvars_file = f"environments/{environment}.tfvars"
        plan_file = f"/tmp/terraform-{environment}-test.plan"

        # Initialize if needed
        subprocess.run(
            ["terraform", "init", "-backend=false"],
            capture_output=True, text=True, check=True
        )

        # Generate plan
        result = subprocess.run([
            "terraform", "plan",
            f"-var-file={tfvars_file}",
            f"-out={plan_file}"
        ], capture_output=True, text=True)

        assert result.returncode == 0, f"Terraform plan failed for {environment}: {result.stderr}"

        # Clean up
        if os.path.exists(plan_file):
            os.remove(plan_file)

    def test_required_variables_defined(self):
        """Test that all required variables are defined"""
        variables_file = self.terraform_dir / "variables.tf"

        # Read variables file
        with open(variables_file, 'r') as f:
            content = f.read()

        required_variables = [
            "environment",
            "project_name",
            "aws_region",
            "vpc_cidr",
            "kubernetes_version"
        ]

        for var in required_variables:
            assert f'variable "{var}"' in content, f"Required variable {var} not defined"

    @pytest.mark.parametrize("environment", ["dev", "staging", "prod"])
    def test_environment_specific_values(self, environment):
        """Test that environment-specific values are appropriate"""
        tfvars_file = self.terraform_dir / "environments" / f"{environment}.tfvars"

        with open(tfvars_file, 'r') as f:
            content = f.read()

        # Check environment name
        assert f'environment = "{environment}"' in content

        # Check security configurations based on environment
        if environment == "prod":
            # Production should have stricter security
            assert 'cluster_endpoint_public_access  = false' in content or \
                   'cluster_endpoint_public_access = false' in content, \
                   "Production cluster should not have public access"
        else:
            # Dev/staging can have public access for convenience
            assert 'cluster_endpoint_public_access' in content

    def test_security_configurations(self):
        """Test security-related configurations"""
        # Check main.tf for security settings
        main_file = self.terraform_dir / "main.tf"

        with open(main_file, 'r') as f:
            content = f.read()

        # Check for encryption settings
        assert "storage_encrypted" in content, "RDS storage should be encrypted"
        assert "at_rest_encryption_enabled = true" in content, "Redis should have encryption at rest"
        assert "transit_encryption_enabled = true" in content, "Redis should have encryption in transit"

    def test_no_hardcoded_secrets(self):
        """Test that no secrets are hardcoded in Terraform files"""
        # Patterns that might indicate hardcoded secrets
        secret_patterns = [
            "password =",
            "secret =",
            "api_key =",
            "private_key =",
            "access_key =",
        ]

        for tf_file in self.terraform_dir.glob("**/*.tf"):
            with open(tf_file, 'r') as f:
                content = f.read()

            for pattern in secret_patterns:
                # Look for patterns but exclude variable declarations and comments
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    if pattern in line and not line.strip().startswith('#') and \
                       'var.' not in line and 'variable' not in line:
                        pytest.fail(f"Potential hardcoded secret in {tf_file}:{i}: {line.strip()}")

    def test_resource_naming_convention(self):
        """Test that resources follow naming conventions"""
        main_file = self.terraform_dir / "main.tf"

        with open(main_file, 'r') as f:
            content = f.read()

        # Check that cluster name uses local variable
        assert "local.cluster_name" in content, "Should use local.cluster_name for consistent naming"

        # Check tag usage
        assert "local.common_tags" in content, "Should use common tags"

    def test_backup_and_retention_settings(self):
        """Test backup and retention configurations"""
        main_file = self.terraform_dir / "main.tf"

        with open(main_file, 'r') as f:
            content = f.read()

        # Check RDS backup settings
        assert "backup_retention_period" in content, "RDS should have backup retention configured"
        assert "backup_window" in content, "RDS should have backup window configured"

        # Check Redis backup settings
        assert "snapshot_retention_limit" in content, "Redis should have snapshot retention configured"

    def test_monitoring_configuration(self):
        """Test monitoring and observability configurations"""
        main_file = self.terraform_dir / "main.tf"

        with open(main_file, 'r') as f:
            content = f.read()

        # Check for monitoring role
        assert "monitoring_role_name" in content, "Should have monitoring role for RDS"
        assert "performance_insights_enabled = true" in content, "Should enable Performance Insights"

    @pytest.mark.parametrize("environment", ["dev", "staging", "prod"])
    def test_environment_resource_scaling(self, environment):
        """Test that resource scaling is appropriate for environment"""
        tfvars_file = self.terraform_dir / "environments" / f"{environment}.tfvars"

        with open(tfvars_file, 'r') as f:
            content = f.read()

        if environment == "dev":
            # Dev should use smaller instances
            assert "t3.micro" in content or "t3.small" in content, "Dev should use smaller instances"
        elif environment == "prod":
            # Prod should use larger instances
            assert any(instance in content for instance in ["t3.large", "t3.xlarge", "c5.large"]), \
                   "Production should use appropriately sized instances"

    def test_vpc_cidr_uniqueness(self):
        """Test that VPC CIDRs are unique across environments"""
        cidrs = {}

        for env in self.environments:
            tfvars_file = self.terraform_dir / "environments" / f"{env}.tfvars"

            with open(tfvars_file, 'r') as f:
                content = f.read()

            # Extract VPC CIDR
            for line in content.split('\n'):
                if 'vpc_cidr' in line and '=' in line:
                    cidr = line.split('=')[1].strip().strip('"')
                    if env in cidrs:
                        assert cidrs[env] != cidr, f"Duplicate CIDR found for {env}"
                    cidrs[env] = cidr

        # Ensure we found CIDRs for all environments
        assert len(cidrs) == len(self.environments), "Not all environment CIDRs found"

        # Ensure all CIDRs are different
        cidr_values = list(cidrs.values())
        assert len(set(cidr_values)) == len(cidr_values), "VPC CIDRs must be unique across environments"

    def test_security_groups_configuration(self):
        """Test security groups configuration"""
        sg_file = self.terraform_dir / "security-groups.tf"

        with open(sg_file, 'r') as f:
            content = f.read()

        # Check for essential security groups
        required_sgs = ["alb", "rds", "redis", "monitoring"]
        for sg in required_sgs:
            assert f'resource "aws_security_group" "{sg}"' in content, \
                   f"Missing security group: {sg}"

        # Check for least privilege principle
        assert "from_port" in content and "to_port" in content, \
               "Security groups should specify port ranges"

    def test_state_backend_configuration(self):
        """Test that state backend is properly configured"""
        main_file = self.terraform_dir / "main.tf"

        with open(main_file, 'r') as f:
            content = f.read()

        # Check for S3 backend
        assert 'backend "s3"' in content, "Should use S3 backend for state"

        # Check for backend config files
        for env in self.environments:
            backend_file = self.terraform_dir / "environments" / f"backend-{env}.conf"
            assert backend_file.exists(), f"Missing backend config for {env}"

    def test_output_definitions(self):
        """Test that required outputs are defined"""
        outputs_file = self.terraform_dir / "outputs.tf"

        with open(outputs_file, 'r') as f:
            content = f.read()

        required_outputs = [
            "cluster_name",
            "cluster_endpoint",
            "vpc_id",
            "db_instance_endpoint",
            "load_balancer_dns_name"
        ]

        for output in required_outputs:
            assert f'output "{output}"' in content, f"Missing required output: {output}"