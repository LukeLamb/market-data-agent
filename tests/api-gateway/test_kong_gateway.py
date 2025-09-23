"""
Kong API Gateway Tests for Market Data Agent
Phase 4 Step 4: API Gateway & Inter-Agent Communication
"""

import pytest
import requests
import jwt
import time
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import patch, Mock


class TestKongGateway:
    """Test Kong API Gateway deployment and configuration"""

    @classmethod
    def setup_class(cls):
        """Set up test environment"""
        cls.base_url = "https://api.market-data.example.com"
        cls.internal_url = "https://internal-api.market-data.local"
        cls.admin_url = "http://kong-admin.kong.svc.cluster.local:8001"

        # Test JWT token for authentication
        cls.test_jwt_payload = {
            "iss": "https://auth.market-data.example.com",
            "aud": "market-data-agents",
            "sub": "test-user",
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
            "scope": "read write market-data:read market-data:write",
            "role": "trader",
            "agent_id": "test-agent-001"
        }

    def test_kong_admin_api_accessible(self):
        """Test that Kong Admin API is accessible"""
        try:
            response = requests.get(f"{self.admin_url}/status", timeout=5)
            assert response.status_code == 200

            status_data = response.json()
            assert "database" in status_data
            assert "server" in status_data
        except requests.exceptions.RequestException:
            pytest.skip("Kong Admin API not accessible in test environment")

    def test_kong_proxy_health(self):
        """Test Kong proxy health endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5, verify=False)
            # Should return 404 or redirect since no health endpoint configured at root
            assert response.status_code in [404, 301, 302]
        except requests.exceptions.RequestException:
            pytest.skip("Kong proxy not accessible in test environment")

    def test_kong_services_configured(self):
        """Test that Kong services are properly configured"""
        try:
            response = requests.get(f"{self.admin_url}/services", timeout=5)
            assert response.status_code == 200

            services = response.json()
            service_names = [s["name"] for s in services.get("data", [])]

            expected_services = [
                "market-data-service",
                "analytics-service",
                "inter-agent-router"
            ]

            for service in expected_services:
                assert service in service_names, f"Service {service} not found"
        except requests.exceptions.RequestException:
            pytest.skip("Kong Admin API not accessible")

    def test_kong_routes_configured(self):
        """Test that Kong routes are properly configured"""
        try:
            response = requests.get(f"{self.admin_url}/routes", timeout=5)
            assert response.status_code == 200

            routes = response.json()
            route_paths = []
            for route in routes.get("data", []):
                if "paths" in route:
                    route_paths.extend(route["paths"])

            expected_paths = [
                "/api/v1/market-data",
                "/api/v1/analytics"
            ]

            for path in expected_paths:
                assert any(path in rpath for rpath in route_paths), f"Route {path} not found"
        except requests.exceptions.RequestException:
            pytest.skip("Kong Admin API not accessible")

    def test_kong_plugins_configured(self):
        """Test that Kong plugins are properly configured"""
        try:
            response = requests.get(f"{self.admin_url}/plugins", timeout=5)
            assert response.status_code == 200

            plugins = response.json()
            plugin_names = [p["name"] for p in plugins.get("data", [])]

            expected_plugins = [
                "rate-limiting",
                "jwt",
                "cors",
                "prometheus"
            ]

            for plugin in expected_plugins:
                assert plugin in plugin_names, f"Plugin {plugin} not configured"
        except requests.exceptions.RequestException:
            pytest.skip("Kong Admin API not accessible")

    def test_rate_limiting_plugin(self):
        """Test rate limiting plugin configuration"""
        try:
            response = requests.get(f"{self.admin_url}/plugins", timeout=5)
            assert response.status_code == 200

            plugins = response.json()
            rate_limit_plugins = [p for p in plugins.get("data", []) if p["name"] == "rate-limiting"]

            assert len(rate_limit_plugins) > 0, "No rate limiting plugins found"

            for plugin in rate_limit_plugins:
                config = plugin["config"]
                assert "minute" in config or "hour" in config, "Rate limits not configured"
                assert config.get("policy") == "redis", "Should use Redis for rate limiting"
        except requests.exceptions.RequestException:
            pytest.skip("Kong Admin API not accessible")

    def test_jwt_plugin_configuration(self):
        """Test JWT plugin configuration"""
        try:
            response = requests.get(f"{self.admin_url}/plugins", timeout=5)
            assert response.status_code == 200

            plugins = response.json()
            jwt_plugins = [p for p in plugins.get("data", []) if p["name"] == "jwt"]

            assert len(jwt_plugins) > 0, "No JWT plugins found"

            for plugin in jwt_plugins:
                config = plugin["config"]
                assert "header_names" in config, "JWT header names not configured"
                assert "Authorization" in config["header_names"], "Authorization header not configured"
        except requests.exceptions.RequestException:
            pytest.skip("Kong Admin API not accessible")

    def test_cors_plugin_configuration(self):
        """Test CORS plugin configuration"""
        try:
            response = requests.get(f"{self.admin_url}/plugins", timeout=5)
            assert response.status_code == 200

            plugins = response.json()
            cors_plugins = [p for p in plugins.get("data", []) if p["name"] == "cors"]

            assert len(cors_plugins) > 0, "No CORS plugins found"

            for plugin in cors_plugins:
                config = plugin["config"]
                assert "origins" in config, "CORS origins not configured"
                assert "methods" in config, "CORS methods not configured"
                assert "GET" in config["methods"], "GET method not allowed"
                assert "POST" in config["methods"], "POST method not allowed"
        except requests.exceptions.RequestException:
            pytest.skip("Kong Admin API not accessible")

    def test_prometheus_plugin_configured(self):
        """Test Prometheus metrics plugin"""
        try:
            response = requests.get(f"{self.admin_url}/plugins", timeout=5)
            assert response.status_code == 200

            plugins = response.json()
            prometheus_plugins = [p for p in plugins.get("data", []) if p["name"] == "prometheus"]

            assert len(prometheus_plugins) > 0, "Prometheus plugin not found"

            # Test metrics endpoint
            metrics_response = requests.get(f"{self.admin_url}/metrics", timeout=5)
            assert metrics_response.status_code == 200
            assert "kong_" in metrics_response.text, "Kong metrics not found"
        except requests.exceptions.RequestException:
            pytest.skip("Kong Admin API not accessible")

    def test_kong_consumer_configuration(self):
        """Test Kong consumer configuration"""
        try:
            response = requests.get(f"{self.admin_url}/consumers", timeout=5)
            assert response.status_code == 200

            consumers = response.json()
            consumer_usernames = [c["username"] for c in consumers.get("data", [])]

            expected_consumers = [
                "market-data-internal",
                "market-data-external",
                "analytics-user"
            ]

            for consumer in expected_consumers:
                assert consumer in consumer_usernames, f"Consumer {consumer} not found"
        except requests.exceptions.RequestException:
            pytest.skip("Kong Admin API not accessible")

    def test_kong_upstream_health(self):
        """Test Kong upstream health checks"""
        try:
            response = requests.get(f"{self.admin_url}/upstreams", timeout=5)
            assert response.status_code == 200

            upstreams = response.json()

            for upstream in upstreams.get("data", []):
                upstream_name = upstream["name"]
                health_response = requests.get(
                    f"{self.admin_url}/upstreams/{upstream_name}/health",
                    timeout=5
                )
                assert health_response.status_code == 200

                health_data = health_response.json()
                assert "data" in health_data
        except requests.exceptions.RequestException:
            pytest.skip("Kong Admin API not accessible")

    def test_market_data_service_routing(self):
        """Test market data service routing through Kong"""
        try:
            # Test without authentication (should fail)
            response = requests.get(
                f"{self.base_url}/api/v1/market-data/quotes/AAPL",
                timeout=10,
                verify=False
            )
            assert response.status_code in [401, 403], "Should require authentication"

        except requests.exceptions.RequestException:
            pytest.skip("Kong proxy not accessible")

    def test_analytics_service_routing(self):
        """Test analytics service routing through Kong"""
        try:
            # Test without authentication (should fail)
            response = requests.get(
                f"{self.base_url}/api/v1/analytics/reports",
                timeout=10,
                verify=False
            )
            assert response.status_code in [401, 403], "Should require authentication"

        except requests.exceptions.RequestException:
            pytest.skip("Kong proxy not accessible")

    def test_inter_agent_communication_routing(self):
        """Test inter-agent communication routing"""
        try:
            # Test without authentication (should fail)
            response = requests.post(
                f"{self.internal_url}/agent/health-check",
                json={"agent_id": "test-agent"},
                timeout=10,
                verify=False
            )
            assert response.status_code in [401, 403], "Should require authentication"

        except requests.exceptions.RequestException:
            pytest.skip("Inter-agent routing not accessible")

    def test_security_headers(self):
        """Test that security headers are added by Kong"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/health",
                timeout=10,
                verify=False
            )

            # Check for security headers
            expected_headers = [
                "X-Frame-Options",
                "X-Content-Type-Options",
                "X-XSS-Protection",
                "Strict-Transport-Security"
            ]

            for header in expected_headers:
                assert header in response.headers, f"Security header {header} missing"

        except requests.exceptions.RequestException:
            pytest.skip("Kong proxy not accessible")

    def test_request_id_injection(self):
        """Test that request ID is injected by Kong"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/health",
                timeout=10,
                verify=False
            )

            # Check for request ID header
            assert "X-Request-ID" in response.headers or "X-Correlation-ID" in response.headers

        except requests.exceptions.RequestException:
            pytest.skip("Kong proxy not accessible")

    def test_kong_configuration_validation(self):
        """Test Kong configuration validation"""
        try:
            # Test configuration endpoint
            response = requests.get(f"{self.admin_url}/config", timeout=5)
            assert response.status_code == 200

            config = response.json()

            # Validate important configuration settings
            assert config.get("database") == "postgres", "Should use PostgreSQL"
            assert "ssl" in config.get("admin_listen", ""), "Admin API should use SSL"

        except requests.exceptions.RequestException:
            pytest.skip("Kong Admin API not accessible")

    @pytest.mark.parametrize("plugin_name", [
        "rate-limiting", "jwt", "cors", "prometheus",
        "request-transformer", "response-transformer"
    ])
    def test_essential_plugins_enabled(self, plugin_name):
        """Test that essential plugins are enabled"""
        try:
            response = requests.get(f"{self.admin_url}/plugins", timeout=5)
            assert response.status_code == 200

            plugins = response.json()
            plugin_names = [p["name"] for p in plugins.get("data", [])]

            assert plugin_name in plugin_names, f"Essential plugin {plugin_name} not enabled"

        except requests.exceptions.RequestException:
            pytest.skip("Kong Admin API not accessible")

    def test_kong_clustering_setup(self):
        """Test Kong clustering setup if applicable"""
        try:
            response = requests.get(f"{self.admin_url}/clustering/status", timeout=5)

            if response.status_code == 200:
                clustering = response.json()
                # If clustering is enabled, validate configuration
                if clustering.get("enabled"):
                    assert "cluster_cert" in clustering
                    assert "cluster_cert_key" in clustering
            else:
                # Clustering might not be enabled, which is fine for single-node setups
                pytest.skip("Clustering not enabled or not supported")

        except requests.exceptions.RequestException:
            pytest.skip("Kong Admin API not accessible")

    def test_kong_database_connectivity(self):
        """Test Kong database connectivity"""
        try:
            response = requests.get(f"{self.admin_url}/status", timeout=5)
            assert response.status_code == 200

            status = response.json()
            db_status = status.get("database", {})

            assert db_status.get("reachable") == True, "Database should be reachable"

        except requests.exceptions.RequestException:
            pytest.skip("Kong Admin API not accessible")

    def test_kong_load_balancing_configuration(self):
        """Test Kong load balancing configuration"""
        try:
            response = requests.get(f"{self.admin_url}/services", timeout=5)
            assert response.status_code == 200

            services = response.json()

            for service in services.get("data", []):
                service_name = service["name"]

                # Check if service has load balancing configured
                if "host" in service:
                    # Test service health
                    health_response = requests.get(
                        f"{self.admin_url}/services/{service_name}/health",
                        timeout=5
                    )
                    # Health endpoint might not exist, but service should be configured
                    assert health_response.status_code in [200, 404]

        except requests.exceptions.RequestException:
            pytest.skip("Kong Admin API not accessible")

    def test_api_versioning_support(self):
        """Test API versioning support in Kong"""
        try:
            response = requests.get(f"{self.admin_url}/routes", timeout=5)
            assert response.status_code == 200

            routes = response.json()

            # Check that versioned routes exist
            version_paths = []
            for route in routes.get("data", []):
                if "paths" in route:
                    version_paths.extend([p for p in route["paths"] if "/v1/" in p])

            assert len(version_paths) > 0, "No versioned API paths found"

        except requests.exceptions.RequestException:
            pytest.skip("Kong Admin API not accessible")

    def test_kong_performance_configuration(self):
        """Test Kong performance-related configuration"""
        try:
            response = requests.get(f"{self.admin_url}/config", timeout=5)
            assert response.status_code == 200

            config = response.json()

            # Check worker processes configuration
            nginx_worker_processes = config.get("nginx_worker_processes")
            if nginx_worker_processes:
                assert nginx_worker_processes == "auto" or int(nginx_worker_processes) > 0

        except requests.exceptions.RequestException:
            pytest.skip("Kong Admin API not accessible")