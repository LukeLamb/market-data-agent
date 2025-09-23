"""
Tests for Kubernetes manifests
Phase 4 Step 1: Container Orchestration & Deployment Automation
"""

import pytest
import yaml
import os
from pathlib import Path
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import tempfile
import subprocess


class TestKubernetesManifests:
    """Test suite for Kubernetes manifests"""

    def setup_method(self):
        """Setup for each test method"""
        self.project_root = Path(__file__).parent.parent.parent
        self.k8s_base_path = self.project_root / "k8s" / "base"
        self.helm_path = self.project_root / "helm" / "market-data-agent"

    def test_k8s_base_directory_exists(self):
        """Test that k8s base directory exists"""
        assert self.k8s_base_path.exists(), "k8s/base directory should exist"

    def test_namespace_manifest(self):
        """Test namespace manifest is valid"""
        namespace_file = self.k8s_base_path / "namespace.yaml"
        assert namespace_file.exists(), "namespace.yaml should exist"

        with open(namespace_file, 'r') as f:
            manifests = list(yaml.safe_load_all(f))

        # Should have namespace, resource quota, and limit range
        assert len(manifests) == 3, "Should have 3 manifests in namespace.yaml"

        namespace = manifests[0]
        assert namespace['kind'] == 'Namespace'
        assert namespace['metadata']['name'] == 'market-data-agent'

        resource_quota = manifests[1]
        assert resource_quota['kind'] == 'ResourceQuota'
        assert 'hard' in resource_quota['spec']

        limit_range = manifests[2]
        assert limit_range['kind'] == 'LimitRange'
        assert 'limits' in limit_range['spec']

    def test_deployment_manifest(self):
        """Test deployment manifest is valid"""
        deployment_file = self.k8s_base_path / "deployment.yaml"
        assert deployment_file.exists(), "deployment.yaml should exist"

        with open(deployment_file, 'r') as f:
            deployment = yaml.safe_load(f)

        assert deployment['kind'] == 'Deployment'
        assert deployment['metadata']['name'] == 'market-data-agent'

        # Check security context
        spec = deployment['spec']['template']['spec']
        assert 'securityContext' in spec
        assert spec['securityContext']['runAsNonRoot'] is True

        # Check container configuration
        container = spec['containers'][0]
        assert container['name'] == 'market-data-agent'
        assert 'securityContext' in container
        assert container['securityContext']['allowPrivilegeEscalation'] is False

        # Check resource limits
        assert 'resources' in container
        assert 'requests' in container['resources']
        assert 'limits' in container['resources']

        # Check probes
        assert 'livenessProbe' in container
        assert 'readinessProbe' in container
        assert 'startupProbe' in container

    def test_service_manifest(self):
        """Test service manifest is valid"""
        service_file = self.k8s_base_path / "service.yaml"
        assert service_file.exists(), "service.yaml should exist"

        with open(service_file, 'r') as f:
            manifests = list(yaml.safe_load_all(f))

        # Should have multiple services
        assert len(manifests) >= 2, "Should have at least 2 services"

        for manifest in manifests:
            assert manifest['kind'] == 'Service'
            assert 'ports' in manifest['spec']
            assert 'selector' in manifest['spec']

    def test_configmap_manifest(self):
        """Test configmap manifest is valid"""
        configmap_file = self.k8s_base_path / "configmap.yaml"
        assert configmap_file.exists(), "configmap.yaml should exist"

        with open(configmap_file, 'r') as f:
            manifests = list(yaml.safe_load_all(f))

        for manifest in manifests:
            assert manifest['kind'] == 'ConfigMap'
            assert 'data' in manifest

    def test_secret_manifest(self):
        """Test secret manifest is valid"""
        secret_file = self.k8s_base_path / "secret.yaml"
        assert secret_file.exists(), "secret.yaml should exist"

        with open(secret_file, 'r') as f:
            manifests = list(yaml.safe_load_all(f))

        for manifest in manifests:
            assert manifest['kind'] == 'Secret'
            assert manifest['type'] in ['Opaque', 'kubernetes.io/tls']

    def test_serviceaccount_manifest(self):
        """Test service account manifest is valid"""
        sa_file = self.k8s_base_path / "serviceaccount.yaml"
        assert sa_file.exists(), "serviceaccount.yaml should exist"

        with open(sa_file, 'r') as f:
            manifests = list(yaml.safe_load_all(f))

        # Should have ServiceAccount, Role, and RoleBinding
        kinds = [manifest['kind'] for manifest in manifests]
        assert 'ServiceAccount' in kinds
        assert 'Role' in kinds
        assert 'RoleBinding' in kinds

    def test_pvc_manifest(self):
        """Test PVC manifest is valid"""
        pvc_file = self.k8s_base_path / "pvc.yaml"
        assert pvc_file.exists(), "pvc.yaml should exist"

        with open(pvc_file, 'r') as f:
            manifests = list(yaml.safe_load_all(f))

        for manifest in manifests:
            assert manifest['kind'] == 'PersistentVolumeClaim'
            assert 'spec' in manifest
            assert 'accessModes' in manifest['spec']
            assert 'resources' in manifest['spec']

    def test_kustomization_manifest(self):
        """Test kustomization manifest is valid"""
        kustomization_file = self.k8s_base_path / "kustomization.yaml"
        assert kustomization_file.exists(), "kustomization.yaml should exist"

        with open(kustomization_file, 'r') as f:
            kustomization = yaml.safe_load(f)

        assert kustomization['kind'] == 'Kustomization'
        assert 'resources' in kustomization
        assert 'namespace' in kustomization

    def test_manifest_labels_consistency(self):
        """Test that all manifests have consistent labels"""
        manifest_files = [
            "namespace.yaml",
            "deployment.yaml",
            "service.yaml",
            "configmap.yaml",
            "secret.yaml",
            "serviceaccount.yaml",
            "pvc.yaml"
        ]

        expected_labels = {
            'app.kubernetes.io/name': 'market-data-agent',
            'app.kubernetes.io/part-of': 'market-data-platform',
            'app.kubernetes.io/managed-by': 'kustomize'
        }

        for file_name in manifest_files:
            file_path = self.k8s_base_path / file_name
            if file_path.exists():
                with open(file_path, 'r') as f:
                    manifests = list(yaml.safe_load_all(f))

                for manifest in manifests:
                    if manifest and 'metadata' in manifest:
                        labels = manifest['metadata'].get('labels', {})
                        for key, value in expected_labels.items():
                            if key in labels:
                                assert labels[key] == value, f"Label {key} mismatch in {file_name}"

    def test_kustomize_build(self):
        """Test that kustomize can build the manifests"""
        try:
            result = subprocess.run(
                ['kustomize', 'build', str(self.k8s_base_path)],
                capture_output=True,
                text=True,
                check=True
            )

            # Parse the output to ensure it's valid YAML
            manifests = list(yaml.safe_load_all(result.stdout))
            assert len(manifests) > 0, "Kustomize should generate manifests"

            # Check that all generated manifests are valid
            for manifest in manifests:
                if manifest:  # Skip empty documents
                    assert 'apiVersion' in manifest
                    assert 'kind' in manifest
                    assert 'metadata' in manifest

        except subprocess.CalledProcessError as e:
            pytest.fail(f"Kustomize build failed: {e.stderr}")
        except FileNotFoundError:
            pytest.skip("Kustomize not installed")

    def test_kubernetes_dry_run(self):
        """Test manifests with Kubernetes dry-run (if cluster is available)"""
        try:
            # Try to load kubeconfig
            config.load_kube_config()
            api_client = client.ApiClient()

            # Build manifests with kustomize
            result = subprocess.run(
                ['kustomize', 'build', str(self.k8s_base_path)],
                capture_output=True,
                text=True,
                check=True
            )

            # Create temporary file with manifests
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(result.stdout)
                temp_file = f.name

            try:
                # Apply with dry-run
                kubectl_result = subprocess.run(
                    ['kubectl', 'apply', '-f', temp_file, '--dry-run=client'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                assert kubectl_result.returncode == 0, "Kubectl dry-run should succeed"

            finally:
                os.unlink(temp_file)

        except (config.ConfigException, FileNotFoundError):
            pytest.skip("Kubernetes cluster not available or tools not installed")
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Kubernetes dry-run failed: {e.stderr}")


class TestHelmChart:
    """Test Helm chart configuration"""

    def setup_method(self):
        """Setup for each test method"""
        self.project_root = Path(__file__).parent.parent.parent
        self.helm_path = self.project_root / "helm" / "market-data-agent"

    def test_helm_chart_structure(self):
        """Test Helm chart has proper structure"""
        assert self.helm_path.exists(), "Helm chart directory should exist"
        assert (self.helm_path / "Chart.yaml").exists(), "Chart.yaml should exist"
        assert (self.helm_path / "values.yaml").exists(), "values.yaml should exist"
        assert (self.helm_path / "templates").exists(), "templates directory should exist"

    def test_chart_yaml(self):
        """Test Chart.yaml is valid"""
        chart_file = self.helm_path / "Chart.yaml"
        with open(chart_file, 'r') as f:
            chart = yaml.safe_load(f)

        assert chart['apiVersion'] == 'v2'
        assert chart['name'] == 'market-data-agent'
        assert chart['type'] == 'application'
        assert 'version' in chart
        assert 'appVersion' in chart

    def test_values_yaml(self):
        """Test values.yaml is valid"""
        values_file = self.helm_path / "values.yaml"
        with open(values_file, 'r') as f:
            values = yaml.safe_load(f)

        # Check required sections
        assert 'image' in values
        assert 'service' in values
        assert 'resources' in values
        assert 'config' in values

        # Check image configuration
        assert 'repository' in values['image']
        assert 'tag' in values['image']
        assert 'pullPolicy' in values['image']

    def test_helm_template(self):
        """Test Helm template rendering"""
        try:
            result = subprocess.run(
                ['helm', 'template', 'test-release', str(self.helm_path)],
                capture_output=True,
                text=True,
                check=True
            )

            # Parse the output to ensure it's valid YAML
            manifests = list(yaml.safe_load_all(result.stdout))
            assert len(manifests) > 0, "Helm should generate manifests"

            # Check that all generated manifests are valid
            for manifest in manifests:
                if manifest:  # Skip empty documents
                    assert 'apiVersion' in manifest
                    assert 'kind' in manifest
                    assert 'metadata' in manifest

        except subprocess.CalledProcessError as e:
            pytest.fail(f"Helm template failed: {e.stderr}")
        except FileNotFoundError:
            pytest.skip("Helm not installed")

    def test_helm_lint(self):
        """Test Helm chart passes linting"""
        try:
            result = subprocess.run(
                ['helm', 'lint', str(self.helm_path)],
                capture_output=True,
                text=True,
                check=True
            )
            assert "1 chart(s) linted, 0 chart(s) failed" in result.stdout

        except subprocess.CalledProcessError as e:
            pytest.fail(f"Helm lint failed: {e.stderr}")
        except FileNotFoundError:
            pytest.skip("Helm not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])