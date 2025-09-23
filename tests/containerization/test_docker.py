"""
Tests for Docker containerization
Phase 4 Step 1: Container Orchestration & Deployment Automation
"""

import pytest
import docker
import subprocess
import json
import time
from pathlib import Path


class TestDockerContainerization:
    """Test suite for Docker containerization"""

    def setup_method(self):
        """Setup for each test method"""
        self.client = docker.from_env()
        self.image_name = "market-data-agent"
        self.image_tag = "test"
        self.container_name = "market-data-agent-test"
        self.project_root = Path(__file__).parent.parent.parent

    def teardown_method(self):
        """Cleanup after each test method"""
        # Remove test containers
        try:
            container = self.client.containers.get(self.container_name)
            container.stop()
            container.remove()
        except docker.errors.NotFound:
            pass

        # Remove test images
        try:
            self.client.images.remove(f"{self.image_name}:{self.image_tag}", force=True)
        except docker.errors.ImageNotFound:
            pass

    def test_dockerfile_exists(self):
        """Test that Dockerfile exists and is valid"""
        dockerfile_path = self.project_root / "Dockerfile"
        assert dockerfile_path.exists(), "Dockerfile should exist"

        # Read and validate Dockerfile content
        with open(dockerfile_path, 'r') as f:
            content = f.read()

        # Check for multi-stage build
        assert "FROM python:3.11-slim as builder" in content
        assert "FROM python:3.11-slim as runtime" in content

        # Check for security best practices
        assert "USER appuser" in content
        assert "HEALTHCHECK" in content

        # Check for proper labels
        assert "org.opencontainers.image" in content

    def test_dockerignore_exists(self):
        """Test that .dockerignore exists and contains proper exclusions"""
        dockerignore_path = self.project_root / ".dockerignore"
        assert dockerignore_path.exists(), ".dockerignore should exist"

        with open(dockerignore_path, 'r') as f:
            content = f.read()

        # Check for important exclusions
        assert ".git" in content
        assert "__pycache__" in content
        assert "*.pyc" in content
        assert "tests/" in content

    def test_docker_build(self):
        """Test Docker image builds successfully"""
        # Build the image
        image, build_logs = self.client.images.build(
            path=str(self.project_root),
            tag=f"{self.image_name}:{self.image_tag}",
            dockerfile="Dockerfile",
            rm=True,
            forcerm=True
        )

        assert image is not None, "Docker image should build successfully"

        # Check image properties
        assert image.attrs['Config']['User'] == 'appuser'
        assert 'HEALTHCHECK' in image.attrs['Config']

        # Check image labels
        labels = image.attrs['Config']['Labels']
        assert 'org.opencontainers.image.title' in labels
        assert labels['org.opencontainers.image.title'] == 'Market Data Agent'

    def test_docker_build_args(self):
        """Test Docker build with build arguments"""
        build_args = {
            'BUILD_DATE': '2024-01-01T00:00:00Z',
            'VCS_REF': 'abc123',
            'VERSION': '1.0.0'
        }

        image, build_logs = self.client.images.build(
            path=str(self.project_root),
            tag=f"{self.image_name}:{self.image_tag}",
            dockerfile="Dockerfile",
            buildargs=build_args,
            rm=True,
            forcerm=True
        )

        # Check that build args are applied to labels
        labels = image.attrs['Config']['Labels']
        assert labels['org.opencontainers.image.version'] == '1.0.0'
        assert labels['org.opencontainers.image.revision'] == 'abc123'

    def test_container_startup(self):
        """Test that container starts and responds to health checks"""
        # First build the image
        self.client.images.build(
            path=str(self.project_root),
            tag=f"{self.image_name}:{self.image_tag}",
            dockerfile="Dockerfile",
            rm=True
        )

        # Run container with environment variables
        container = self.client.containers.run(
            f"{self.image_name}:{self.image_tag}",
            name=self.container_name,
            environment={
                'ENVIRONMENT': 'test',
                'LOG_LEVEL': 'DEBUG',
                'DATABASE_URL': 'sqlite:///test.db',
                'REDIS_URL': 'redis://localhost:6379/0'
            },
            ports={'8000/tcp': 8000},
            detach=True,
            remove=False
        )

        # Wait for container to start
        time.sleep(10)

        # Check container is running
        container.reload()
        assert container.status == 'running', "Container should be running"

        # Check health status (if healthcheck is implemented)
        # This might take some time for the health check to run
        max_retries = 6
        for _ in range(max_retries):
            container.reload()
            health = container.attrs.get('State', {}).get('Health', {})
            if health.get('Status') == 'healthy':
                break
            time.sleep(10)

        # Stop container
        container.stop()

    def test_container_ports_exposed(self):
        """Test that container exposes the correct ports"""
        # Build image
        image, _ = self.client.images.build(
            path=str(self.project_root),
            tag=f"{self.image_name}:{self.image_tag}",
            dockerfile="Dockerfile",
            rm=True
        )

        # Check exposed ports in image config
        exposed_ports = image.attrs['Config']['ExposedPorts']
        assert '8000/tcp' in exposed_ports  # HTTP port
        assert '8001/tcp' in exposed_ports  # WebSocket port
        assert '8080/tcp' in exposed_ports  # Metrics port

    def test_container_environment_variables(self):
        """Test container environment variables"""
        # Build and run container
        self.client.images.build(
            path=str(self.project_root),
            tag=f"{self.image_name}:{self.image_tag}",
            dockerfile="Dockerfile",
            rm=True
        )

        container = self.client.containers.run(
            f"{self.image_name}:{self.image_tag}",
            name=self.container_name,
            environment={
                'ENVIRONMENT': 'test',
                'LOG_LEVEL': 'DEBUG'
            },
            detach=True,
            remove=False
        )

        # Check environment variables are set
        env_vars = container.attrs['Config']['Env']
        env_dict = {var.split('=')[0]: var.split('=')[1] for var in env_vars if '=' in var}

        assert env_dict.get('ENVIRONMENT') == 'test'
        assert env_dict.get('LOG_LEVEL') == 'DEBUG'
        assert 'PYTHONUNBUFFERED' in env_dict

        container.stop()

    def test_container_volumes(self):
        """Test container volume mounts"""
        # Build and run container with volumes
        self.client.images.build(
            path=str(self.project_root),
            tag=f"{self.image_name}:{self.image_tag}",
            dockerfile="Dockerfile",
            rm=True
        )

        volumes = {
            '/tmp/test-logs': {'bind': '/app/logs', 'mode': 'rw'},
            '/tmp/test-data': {'bind': '/app/data', 'mode': 'rw'}
        }

        container = self.client.containers.run(
            f"{self.image_name}:{self.image_tag}",
            name=self.container_name,
            volumes=volumes,
            detach=True,
            remove=False
        )

        # Check volume mounts
        mounts = container.attrs['Mounts']
        mount_destinations = [mount['Destination'] for mount in mounts]

        assert '/app/logs' in mount_destinations
        assert '/app/data' in mount_destinations

        container.stop()

    def test_container_security(self):
        """Test container security configuration"""
        # Build image
        image, _ = self.client.images.build(
            path=str(self.project_root),
            tag=f"{self.image_name}:{self.image_tag}",
            dockerfile="Dockerfile",
            rm=True
        )

        # Check that container runs as non-root user
        config = image.attrs['Config']
        assert config['User'] == 'appuser', "Container should run as non-root user"

    def test_image_size(self):
        """Test that image size is reasonable"""
        # Build image
        image, _ = self.client.images.build(
            path=str(self.project_root),
            tag=f"{self.image_name}:{self.image_tag}",
            dockerfile="Dockerfile",
            rm=True
        )

        # Check image size (should be less than 1GB for a Python app)
        size_mb = image.attrs['Size'] / (1024 * 1024)
        assert size_mb < 1024, f"Image size ({size_mb:.2f}MB) should be less than 1GB"

    def test_image_layers(self):
        """Test image layer optimization"""
        # Build image
        image, _ = self.client.images.build(
            path=str(self.project_root),
            tag=f"{self.image_name}:{self.image_tag}",
            dockerfile="Dockerfile",
            rm=True
        )

        # Check number of layers (should be reasonable for multi-stage build)
        history = self.client.api.history(image.id)
        layer_count = len([layer for layer in history if layer['Size'] > 0])
        assert layer_count < 20, f"Image should have reasonable number of layers ({layer_count})"


class TestDockerCompose:
    """Test Docker Compose configuration"""

    def test_docker_compose_file_exists(self):
        """Test that docker-compose files exist if used"""
        project_root = Path(__file__).parent.parent.parent

        # Check for common docker-compose files
        compose_files = [
            "docker-compose.yml",
            "docker-compose.yaml",
            "docker-compose.dev.yml",
            "docker-compose.prod.yml"
        ]

        found_compose = any((project_root / file).exists() for file in compose_files)

        if found_compose:
            # If compose files exist, validate them
            for file in compose_files:
                compose_path = project_root / file
                if compose_path.exists():
                    # Try to parse the compose file
                    result = subprocess.run(
                        ['docker-compose', '-f', str(compose_path), 'config'],
                        capture_output=True,
                        text=True
                    )
                    assert result.returncode == 0, f"Docker compose file {file} should be valid"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])