#!/usr/bin/env python3
"""
Configuration Drift Detection for Market Data Agent
Phase 4 Step 3: Configuration Management & Environment Automation
"""

import json
import yaml
import argparse
import logging
import sys
import subprocess
import tempfile
import os
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import difflib


@dataclass
class DriftResult:
    """Result of drift detection"""
    component: str
    environment: str
    status: str  # 'clean', 'drift', 'error'
    changes: List[Dict[str, Any]]
    severity: str  # 'low', 'medium', 'high', 'critical'
    recommendations: List[str]
    timestamp: str


class ConfigurationDriftDetector:
    """Detect configuration drift across infrastructure components"""

    def __init__(self, environment: str, config_dir: str):
        self.environment = environment
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger(__name__)

        # Drift detection results
        self.results: List[DriftResult] = []

        # Configuration baselines
        self.baselines_dir = self.config_dir / "baselines" / environment
        self.baselines_dir.mkdir(parents=True, exist_ok=True)

    def detect_terraform_drift(self) -> DriftResult:
        """Detect drift in Terraform-managed infrastructure"""
        self.logger.info("Detecting Terraform configuration drift...")

        changes = []
        severity = "low"
        recommendations = []
        status = "clean"

        try:
            terraform_dir = self.config_dir / "terraform"
            tfvars_file = terraform_dir / "environments" / f"{self.environment}.tfvars"

            if not tfvars_file.exists():
                return DriftResult(
                    component="terraform",
                    environment=self.environment,
                    status="error",
                    changes=[{"error": f"Variables file not found: {tfvars_file}"}],
                    severity="high",
                    recommendations=["Create environment-specific tfvars file"],
                    timestamp=datetime.now(timezone.utc).isoformat()
                )

            # Run terraform plan to detect drift
            os.chdir(terraform_dir)

            # Initialize if needed
            subprocess.run(["terraform", "init", "-backend=false"],
                         check=True, capture_output=True, text=True)

            # Generate plan
            plan_result = subprocess.run([
                "terraform", "plan",
                f"-var-file={tfvars_file}",
                "-out=/tmp/terraform-drift.plan",
                "-detailed-exitcode"
            ], capture_output=True, text=True)

            # Exit code 2 means changes detected
            if plan_result.returncode == 2:
                status = "drift"
                severity = "medium"

                # Get plan details
                show_result = subprocess.run([
                    "terraform", "show", "-json", "/tmp/terraform-drift.plan"
                ], capture_output=True, text=True, check=True)

                plan_data = json.loads(show_result.stdout)

                # Analyze changes
                for change in plan_data.get("resource_changes", []):
                    action = change.get("change", {}).get("actions", [])
                    resource_address = change.get("address", "unknown")

                    change_info = {
                        "resource": resource_address,
                        "actions": action,
                        "type": change.get("type", "unknown")
                    }

                    # Determine severity based on action type
                    if "delete" in action:
                        severity = "high"
                        change_info["severity"] = "high"
                        recommendations.append(f"Review deletion of {resource_address}")
                    elif "create" in action:
                        change_info["severity"] = "medium"
                        recommendations.append(f"Verify creation of {resource_address}")
                    else:
                        change_info["severity"] = "low"

                    changes.append(change_info)

                if not recommendations:
                    recommendations.append("Review and apply Terraform changes if expected")

            elif plan_result.returncode == 1:
                status = "error"
                severity = "high"
                changes.append({"error": plan_result.stderr})
                recommendations.append("Fix Terraform configuration errors")

            # Clean up
            if os.path.exists("/tmp/terraform-drift.plan"):
                os.remove("/tmp/terraform-drift.plan")

        except subprocess.CalledProcessError as e:
            status = "error"
            severity = "high"
            changes.append({"error": str(e)})
            recommendations.append("Fix Terraform execution errors")
        except Exception as e:
            status = "error"
            severity = "high"
            changes.append({"error": str(e)})
            recommendations.append("Investigate drift detection errors")

        return DriftResult(
            component="terraform",
            environment=self.environment,
            status=status,
            changes=changes,
            severity=severity,
            recommendations=recommendations,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    def detect_kubernetes_drift(self) -> DriftResult:
        """Detect drift in Kubernetes configurations"""
        self.logger.info("Detecting Kubernetes configuration drift...")

        changes = []
        severity = "low"
        recommendations = []
        status = "clean"

        try:
            # Check if kubectl is available and configured
            subprocess.run(["kubectl", "cluster-info"],
                         check=True, capture_output=True, text=True)

            # Get current cluster state
            cluster_resources = self._get_cluster_resources()

            # Compare with baseline
            baseline_file = self.baselines_dir / "kubernetes-resources.json"
            if baseline_file.exists():
                with open(baseline_file, 'r') as f:
                    baseline_resources = json.load(f)

                # Compare resources
                drift_found = self._compare_kubernetes_resources(
                    baseline_resources, cluster_resources
                )

                if drift_found:
                    status = "drift"
                    severity = "medium"
                    changes.extend(drift_found)
                    recommendations.append("Review Kubernetes resource changes")
                    recommendations.append("Update baseline if changes are intentional")
            else:
                # Create baseline
                with open(baseline_file, 'w') as f:
                    json.dump(cluster_resources, f, indent=2)
                recommendations.append("Baseline created for future drift detection")

            # Check for critical namespace issues
            critical_namespaces = ["market-data", "monitoring", "vault"]
            for namespace in critical_namespaces:
                ns_result = subprocess.run([
                    "kubectl", "get", "namespace", namespace
                ], capture_output=True, text=True)

                if ns_result.returncode != 0:
                    status = "drift"
                    severity = "high"
                    changes.append({
                        "type": "missing_namespace",
                        "namespace": namespace,
                        "severity": "high"
                    })
                    recommendations.append(f"Create missing namespace: {namespace}")

            # Check resource quotas
            quota_result = subprocess.run([
                "kubectl", "get", "resourcequota", "--all-namespaces", "-o", "json"
            ], capture_output=True, text=True, check=True)

            quotas = json.loads(quota_result.stdout)
            if not quotas.get("items"):
                changes.append({
                    "type": "missing_resource_quotas",
                    "severity": "medium"
                })
                recommendations.append("Configure resource quotas for namespaces")
                if status == "clean":
                    status = "drift"
                    severity = "medium"

        except subprocess.CalledProcessError as e:
            status = "error"
            severity = "high"
            changes.append({"error": f"kubectl error: {e}"})
            recommendations.append("Fix kubectl connectivity or configuration")
        except Exception as e:
            status = "error"
            severity = "high"
            changes.append({"error": str(e)})
            recommendations.append("Investigate Kubernetes drift detection errors")

        return DriftResult(
            component="kubernetes",
            environment=self.environment,
            status=status,
            changes=changes,
            severity=severity,
            recommendations=recommendations,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    def detect_vault_drift(self) -> DriftResult:
        """Detect drift in Vault configurations"""
        self.logger.info("Detecting Vault configuration drift...")

        changes = []
        severity = "low"
        recommendations = []
        status = "clean"

        try:
            # Check Vault status
            vault_result = subprocess.run([
                "vault", "status", "-format=json"
            ], capture_output=True, text=True)

            if vault_result.returncode == 0:
                vault_status = json.loads(vault_result.stdout)

                # Check if Vault is sealed
                if vault_status.get("sealed", True):
                    status = "drift"
                    severity = "high"
                    changes.append({
                        "type": "vault_sealed",
                        "severity": "high"
                    })
                    recommendations.append("Unseal Vault cluster")

                # Check cluster status
                if vault_status.get("ha_enabled", False):
                    if not vault_status.get("is_self", False):
                        changes.append({
                            "type": "vault_not_leader",
                            "severity": "low"
                        })
                        recommendations.append("Check Vault cluster leadership")

                # Get current policies
                policies_result = subprocess.run([
                    "vault", "policy", "list", "-format=json"
                ], capture_output=True, text=True)

                if policies_result.returncode == 0:
                    policies = json.loads(policies_result.stdout)

                    # Check for required policies
                    required_policies = ["market-data-agent", "monitoring"]
                    missing_policies = [p for p in required_policies if p not in policies]

                    if missing_policies:
                        status = "drift"
                        severity = "medium"
                        changes.append({
                            "type": "missing_policies",
                            "policies": missing_policies,
                            "severity": "medium"
                        })
                        recommendations.append(f"Create missing policies: {missing_policies}")

                # Check auth methods
                auth_result = subprocess.run([
                    "vault", "auth", "list", "-format=json"
                ], capture_output=True, text=True)

                if auth_result.returncode == 0:
                    auth_methods = json.loads(auth_result.stdout)

                    if "kubernetes/" not in auth_methods:
                        status = "drift"
                        severity = "medium"
                        changes.append({
                            "type": "missing_auth_method",
                            "method": "kubernetes",
                            "severity": "medium"
                        })
                        recommendations.append("Enable Kubernetes authentication method")

            else:
                status = "error"
                severity = "high"
                changes.append({"error": "Cannot connect to Vault"})
                recommendations.append("Check Vault connectivity and authentication")

        except subprocess.CalledProcessError as e:
            status = "error"
            severity = "high"
            changes.append({"error": f"Vault CLI error: {e}"})
            recommendations.append("Check Vault CLI configuration")
        except Exception as e:
            status = "error"
            severity = "high"
            changes.append({"error": str(e)})
            recommendations.append("Investigate Vault drift detection errors")

        return DriftResult(
            component="vault",
            environment=self.environment,
            status=status,
            changes=changes,
            severity=severity,
            recommendations=recommendations,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    def detect_configuration_files_drift(self) -> DriftResult:
        """Detect drift in configuration files"""
        self.logger.info("Detecting configuration file drift...")

        changes = []
        severity = "low"
        recommendations = []
        status = "clean"

        try:
            # Check important configuration files
            config_files = [
                self.config_dir / "terraform" / "environments" / f"{self.environment}.tfvars",
                self.config_dir / "ansible" / "inventory" / "hosts.yml",
                self.config_dir / "vault" / "vault-config.hcl",
            ]

            for config_file in config_files:
                if not config_file.exists():
                    continue

                # Calculate current file hash
                current_hash = self._calculate_file_hash(config_file)

                # Compare with baseline
                baseline_hash_file = self.baselines_dir / f"{config_file.name}.hash"

                if baseline_hash_file.exists():
                    with open(baseline_hash_file, 'r') as f:
                        baseline_hash = f.read().strip()

                    if current_hash != baseline_hash:
                        status = "drift"
                        severity = "medium"

                        # Get diff
                        baseline_content_file = self.baselines_dir / f"{config_file.name}.content"
                        if baseline_content_file.exists():
                            with open(baseline_content_file, 'r') as f:
                                baseline_content = f.readlines()

                            with open(config_file, 'r') as f:
                                current_content = f.readlines()

                            diff = list(difflib.unified_diff(
                                baseline_content,
                                current_content,
                                fromfile=f"baseline/{config_file.name}",
                                tofile=f"current/{config_file.name}",
                                lineterm=""
                            ))

                            changes.append({
                                "type": "file_changed",
                                "file": str(config_file),
                                "diff_lines": len([l for l in diff if l.startswith(('+', '-'))]),
                                "severity": "medium"
                            })

                            recommendations.append(f"Review changes to {config_file.name}")
                else:
                    # Create baseline
                    with open(baseline_hash_file, 'w') as f:
                        f.write(current_hash)

                    # Store content for future diffs
                    baseline_content_file = self.baselines_dir / f"{config_file.name}.content"
                    with open(config_file, 'r') as src, open(baseline_content_file, 'w') as dst:
                        dst.write(src.read())

                    recommendations.append(f"Baseline created for {config_file.name}")

        except Exception as e:
            status = "error"
            severity = "high"
            changes.append({"error": str(e)})
            recommendations.append("Investigate configuration file drift detection errors")

        return DriftResult(
            component="configuration_files",
            environment=self.environment,
            status=status,
            changes=changes,
            severity=severity,
            recommendations=recommendations,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    def _get_cluster_resources(self) -> Dict[str, Any]:
        """Get current Kubernetes cluster resources"""
        resources = {}

        # Get namespaces
        ns_result = subprocess.run([
            "kubectl", "get", "namespaces", "-o", "json"
        ], capture_output=True, text=True, check=True)

        resources["namespaces"] = json.loads(ns_result.stdout)

        # Get deployments
        deploy_result = subprocess.run([
            "kubectl", "get", "deployments", "--all-namespaces", "-o", "json"
        ], capture_output=True, text=True, check=True)

        resources["deployments"] = json.loads(deploy_result.stdout)

        # Get services
        svc_result = subprocess.run([
            "kubectl", "get", "services", "--all-namespaces", "-o", "json"
        ], capture_output=True, text=True, check=True)

        resources["services"] = json.loads(svc_result.stdout)

        return resources

    def _compare_kubernetes_resources(self, baseline: Dict[str, Any],
                                    current: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Compare Kubernetes resources and return differences"""
        changes = []

        for resource_type in ["namespaces", "deployments", "services"]:
            baseline_items = {
                item["metadata"]["name"]: item
                for item in baseline.get(resource_type, {}).get("items", [])
            }
            current_items = {
                item["metadata"]["name"]: item
                for item in current.get(resource_type, {}).get("items", [])
            }

            # Check for added resources
            added = set(current_items.keys()) - set(baseline_items.keys())
            for name in added:
                changes.append({
                    "type": f"{resource_type}_added",
                    "name": name,
                    "severity": "medium"
                })

            # Check for removed resources
            removed = set(baseline_items.keys()) - set(current_items.keys())
            for name in removed:
                changes.append({
                    "type": f"{resource_type}_removed",
                    "name": name,
                    "severity": "high"
                })

        return changes

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def run_all_checks(self) -> List[DriftResult]:
        """Run all drift detection checks"""
        self.logger.info(f"Starting drift detection for {self.environment} environment...")

        # Run all drift detection methods
        checks = [
            self.detect_terraform_drift,
            self.detect_kubernetes_drift,
            self.detect_vault_drift,
            self.detect_configuration_files_drift
        ]

        for check in checks:
            try:
                result = check()
                self.results.append(result)
                self.logger.info(f"Completed {result.component} drift check: {result.status}")
            except Exception as e:
                self.logger.error(f"Failed to run {check.__name__}: {e}")
                error_result = DriftResult(
                    component=check.__name__.replace("detect_", "").replace("_drift", ""),
                    environment=self.environment,
                    status="error",
                    changes=[{"error": str(e)}],
                    severity="high",
                    recommendations=["Investigate drift detection failure"],
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
                self.results.append(error_result)

        return self.results

    def generate_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate drift detection report"""
        # Calculate overall status
        has_drift = any(r.status == "drift" for r in self.results)
        has_errors = any(r.status == "error" for r in self.results)

        if has_errors:
            overall_status = "error"
        elif has_drift:
            overall_status = "drift"
        else:
            overall_status = "clean"

        # Calculate severity
        severities = [r.severity for r in self.results]
        if "critical" in severities:
            overall_severity = "critical"
        elif "high" in severities:
            overall_severity = "high"
        elif "medium" in severities:
            overall_severity = "medium"
        else:
            overall_severity = "low"

        report = {
            "environment": self.environment,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": overall_status,
            "overall_severity": overall_severity,
            "summary": {
                "total_components": len(self.results),
                "clean": len([r for r in self.results if r.status == "clean"]),
                "drift": len([r for r in self.results if r.status == "drift"]),
                "errors": len([r for r in self.results if r.status == "error"])
            },
            "components": [asdict(result) for result in self.results],
            "recommendations": list(set(
                rec for result in self.results for rec in result.recommendations
            ))
        }

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"Drift detection report saved to: {output_file}")

        return report


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Detect configuration drift in Market Data Agent infrastructure")
    parser.add_argument("-e", "--environment", required=True, choices=["dev", "staging", "prod"],
                       help="Environment to check for drift")
    parser.add_argument("-c", "--config-dir", default=".",
                       help="Configuration directory (default: current directory)")
    parser.add_argument("-o", "--output", help="Output report file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run drift detection
    detector = ConfigurationDriftDetector(args.environment, args.config_dir)
    results = detector.run_all_checks()

    # Generate report
    output_file = args.output or f"drift-report-{args.environment}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    report = detector.generate_report(output_file)

    # Print summary
    print(f"\n=== DRIFT DETECTION SUMMARY ===")
    print(f"Environment: {args.environment}")
    print(f"Overall Status: {report['overall_status'].upper()}")
    print(f"Overall Severity: {report['overall_severity'].upper()}")
    print(f"Components Checked: {report['summary']['total_components']}")
    print(f"Clean: {report['summary']['clean']}")
    print(f"Drift Detected: {report['summary']['drift']}")
    print(f"Errors: {report['summary']['errors']}")

    # Exit with appropriate code
    if report['overall_status'] == "error":
        sys.exit(2)
    elif report['overall_status'] == "drift":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()