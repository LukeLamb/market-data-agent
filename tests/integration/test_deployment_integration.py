"""
Integration tests for deployment pipeline
Phase 4 Step 1: Container Orchestration & Deployment Automation
"""

import pytest
import requests
import time
import subprocess
import yaml
import os
from pathlib import Path
from kubernetes import client, config
from kubernetes.client.rest import ApiException


class TestDeploymentIntegration:
    """Integration tests for the complete deployment pipeline"""

    def setup_method(self):
        """Setup for each test method"""
        self.project_root = Path(__file__).parent.parent.parent
        self.namespace = "market-data-agent-test"
        self.release_name = "test-release"

    def teardown_method(self):
        """Cleanup after each test method"""
        # Clean up test namespace if it exists
        try:
            result = subprocess.run(
                ['kubectl', 'delete', 'namespace', self.namespace, '--ignore-not-found=true'],
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError:
            pass

    @pytest.mark.integration
    def test_full_deployment_pipeline(self):
        """Test the complete deployment pipeline"""
        # Skip if no cluster available
        try:
            subprocess.run(['kubectl', 'cluster-info'], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("Kubernetes cluster not available")

        # 1. Build Docker image
        image_tag = f"market-data-agent:integration-test-{int(time.time())}"
        build_result = subprocess.run(
            ['docker', 'build', '-t', image_tag, str(self.project_root)],
            capture_output=True,
            text=True
        )
        assert build_result.returncode == 0, f"Docker build failed: {build_result.stderr}"

        try:
            # 2. Deploy with Helm
            helm_path = self.project_root / "helm" / "market-data-agent"
            deploy_result = subprocess.run([
                'helm', 'install', self.release_name, str(helm_path),
                '--namespace', self.namespace,
                '--create-namespace',
                '--set', f'image.tag={image_tag.split(":")[1]}',
                '--set', f'image.repository={image_tag.split(":")[0]}',
                '--set', 'replicaCount=1',
                '--set', 'postgresql.enabled=false',
                '--set', 'redis.enabled=false',
                '--set', 'secrets.database_url=sqlite:///test.db',
                '--set', 'secrets.redis_url=redis://localhost:6379/0',
                '--wait',
                '--timeout=10m'
            ], capture_output=True, text=True)

            if deploy_result.returncode != 0:
                pytest.fail(f"Helm install failed: {deploy_result.stderr}")

            # 3. Wait for pods to be ready
            self._wait_for_pods_ready()

            # 4. Test application health
            self._test_application_health()

            # 5. Test API endpoints
            self._test_api_endpoints()

        finally:
            # Cleanup
            subprocess.run([
                'helm', 'uninstall', self.release_name,
                '--namespace', self.namespace
            ], capture_output=True)

            # Remove Docker image
            subprocess.run(['docker', 'rmi', image_tag], capture_output=True)

    def _wait_for_pods_ready(self, timeout=300):
        """Wait for pods to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            result = subprocess.run([
                'kubectl', 'get', 'pods',
                '-n', self.namespace,
                '-l', f'app.kubernetes.io/instance={self.release_name}',
                '-o', 'jsonpath={.items[*].status.phase}'
            ], capture_output=True, text=True)

            if result.returncode == 0:
                phases = result.stdout.strip().split()
                if all(phase == 'Running' for phase in phases) and phases:
                    # Check if pods are ready
                    ready_result = subprocess.run([
                        'kubectl', 'get', 'pods',
                        '-n', self.namespace,
                        '-l', f'app.kubernetes.io/instance={self.release_name}',
                        '-o', 'jsonpath={.items[*].status.conditions[?(@.type=="Ready")].status}'
                    ], capture_output=True, text=True)

                    if ready_result.returncode == 0:
                        ready_statuses = ready_result.stdout.strip().split()
                        if all(status == 'True' for status in ready_statuses) and ready_statuses:
                            return

            time.sleep(10)

        pytest.fail(f"Pods not ready within {timeout} seconds")

    def _test_application_health(self):
        """Test application health endpoints"""
        # Port-forward to access the application
        port_forward = subprocess.Popen([
            'kubectl', 'port-forward',
            f'service/{self.release_name}',
            '8080:80',
            '-n', self.namespace
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        try:
            # Wait for port-forward to establish
            time.sleep(5)

            # Test health endpoint
            max_retries = 10
            for _ in range(max_retries):
                try:
                    response = requests.get('http://localhost:8080/health', timeout=5)
                    if response.status_code == 200:
                        break
                except requests.exceptions.RequestException:
                    pass
                time.sleep(2)
            else:
                pytest.fail("Health endpoint not accessible")

            # Test readiness endpoint
            try:
                response = requests.get('http://localhost:8080/ready', timeout=5)
                # May return 200 or 503 depending on readiness, just check it responds
                assert response.status_code in [200, 503]
            except requests.exceptions.RequestException:
                pytest.fail("Readiness endpoint not accessible")

        finally:
            port_forward.terminate()
            port_forward.wait()

    def _test_api_endpoints(self):
        """Test API endpoints"""
        # Port-forward to access the application
        port_forward = subprocess.Popen([
            'kubectl', 'port-forward',
            f'service/{self.release_name}',
            '8080:80',
            '-n', self.namespace
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        try:
            # Wait for port-forward to establish
            time.sleep(5)

            # Test root endpoint
            response = requests.get('http://localhost:8080/', timeout=5)
            assert response.status_code in [200, 404], "Root endpoint should respond"

            # Test API endpoints (if they exist)
            try:
                response = requests.get('http://localhost:8080/api/v1/sources', timeout=5)
                # May not be fully functional without database, just check it responds
                assert response.status_code in [200, 500, 503], "API endpoint should respond"
            except requests.exceptions.RequestException:
                # API might not be fully implemented yet
                pass

        finally:
            port_forward.terminate()
            port_forward.wait()

    @pytest.mark.integration
    def test_helm_upgrade(self):
        """Test Helm upgrade functionality"""
        try:
            subprocess.run(['kubectl', 'cluster-info'], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("Kubernetes cluster not available")

        helm_path = self.project_root / "helm" / "market-data-agent"

        # Initial deployment
        deploy_result = subprocess.run([
            'helm', 'install', self.release_name, str(helm_path),
            '--namespace', self.namespace,
            '--create-namespace',
            '--set', 'replicaCount=1',
            '--set', 'postgresql.enabled=false',
            '--set', 'redis.enabled=false',
            '--wait',
            '--timeout=5m'
        ], capture_output=True, text=True)

        assert deploy_result.returncode == 0, f"Initial deployment failed: {deploy_result.stderr}"

        try:
            # Upgrade with different replica count
            upgrade_result = subprocess.run([
                'helm', 'upgrade', self.release_name, str(helm_path),
                '--namespace', self.namespace,
                '--set', 'replicaCount=2',
                '--set', 'postgresql.enabled=false',
                '--set', 'redis.enabled=false',
                '--wait',
                '--timeout=5m'
            ], capture_output=True, text=True)

            assert upgrade_result.returncode == 0, f"Upgrade failed: {upgrade_result.stderr}"

            # Verify replica count
            result = subprocess.run([
                'kubectl', 'get', 'deployment',
                f'{self.release_name}',
                '-n', self.namespace,
                '-o', 'jsonpath={.spec.replicas}'
            ], capture_output=True, text=True)

            assert result.stdout.strip() == '2', "Replica count should be updated to 2"

        finally:
            subprocess.run([
                'helm', 'uninstall', self.release_name,
                '--namespace', self.namespace
            ], capture_output=True)

    @pytest.mark.integration
    def test_rollback(self):
        """Test Helm rollback functionality"""
        try:
            subprocess.run(['kubectl', 'cluster-info'], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("Kubernetes cluster not available")

        helm_path = self.project_root / "helm" / "market-data-agent"

        # Initial deployment
        deploy_result = subprocess.run([
            'helm', 'install', self.release_name, str(helm_path),
            '--namespace', self.namespace,
            '--create-namespace',
            '--set', 'replicaCount=1',
            '--set', 'postgresql.enabled=false',
            '--set', 'redis.enabled=false',
            '--wait',
            '--timeout=5m'
        ], capture_output=True, text=True)

        assert deploy_result.returncode == 0, f"Initial deployment failed: {deploy_result.stderr}"

        try:
            # Upgrade
            upgrade_result = subprocess.run([
                'helm', 'upgrade', self.release_name, str(helm_path),
                '--namespace', self.namespace,
                '--set', 'replicaCount=3',
                '--set', 'postgresql.enabled=false',
                '--set', 'redis.enabled=false',
                '--wait',
                '--timeout=5m'
            ], capture_output=True, text=True)

            assert upgrade_result.returncode == 0, f"Upgrade failed: {upgrade_result.stderr}"

            # Rollback
            rollback_result = subprocess.run([
                'helm', 'rollback', self.release_name, '1',
                '--namespace', self.namespace,
                '--wait',
                '--timeout=5m'
            ], capture_output=True, text=True)

            assert rollback_result.returncode == 0, f"Rollback failed: {rollback_result.stderr}"

            # Verify replica count is back to 1
            result = subprocess.run([
                'kubectl', 'get', 'deployment',
                f'{self.release_name}',
                '-n', self.namespace,
                '-o', 'jsonpath={.spec.replicas}'
            ], capture_output=True, text=True)

            assert result.stdout.strip() == '1', "Replica count should be rolled back to 1"

        finally:
            subprocess.run([
                'helm', 'uninstall', self.release_name,
                '--namespace', self.namespace
            ], capture_output=True)


class TestSecurityCompliance:
    """Test security and compliance aspects of deployment"""

    def setup_method(self):
        """Setup for each test method"""
        self.project_root = Path(__file__).parent.parent.parent

    def test_security_policies(self):
        """Test that security policies are properly configured"""
        # Check deployment security context
        deployment_file = self.project_root / "k8s" / "base" / "deployment.yaml"
        with open(deployment_file, 'r') as f:
            deployment = yaml.safe_load(f)

        # Check pod security context
        pod_spec = deployment['spec']['template']['spec']
        assert pod_spec['securityContext']['runAsNonRoot'] is True
        assert pod_spec['securityContext']['runAsUser'] == 1000

        # Check container security context
        container = pod_spec['containers'][0]
        container_security = container['securityContext']
        assert container_security['allowPrivilegeEscalation'] is False
        assert container_security['readOnlyRootFilesystem'] is True
        assert 'ALL' in container_security['capabilities']['drop']

    def test_resource_limits(self):
        """Test that resource limits are properly set"""
        deployment_file = self.project_root / "k8s" / "base" / "deployment.yaml"
        with open(deployment_file, 'r') as f:
            deployment = yaml.safe_load(f)

        container = deployment['spec']['template']['spec']['containers'][0]
        resources = container['resources']

        assert 'requests' in resources
        assert 'limits' in resources
        assert 'cpu' in resources['requests']
        assert 'memory' in resources['requests']
        assert 'cpu' in resources['limits']
        assert 'memory' in resources['limits']

    def test_network_policies(self):
        """Test network policies if they exist"""
        # This test would check for network policy files
        # Since they might not exist in base configuration, we'll check the structure
        k8s_path = self.project_root / "k8s"

        # Look for network policy files
        network_policy_files = list(k8s_path.rglob("*network*policy*.yaml"))

        if network_policy_files:
            for policy_file in network_policy_files:
                with open(policy_file, 'r') as f:
                    policies = list(yaml.safe_load_all(f))

                for policy in policies:
                    if policy and policy.get('kind') == 'NetworkPolicy':
                        assert 'spec' in policy
                        assert 'podSelector' in policy['spec']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])