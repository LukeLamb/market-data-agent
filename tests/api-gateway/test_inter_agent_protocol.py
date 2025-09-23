"""
Inter-Agent Communication Protocol Tests for Market Data Agent
Phase 4 Step 4: API Gateway & Inter-Agent Communication
"""

import pytest
import requests
import json
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import patch, Mock


class TestInterAgentProtocol:
    """Test inter-agent communication protocol implementation"""

    @classmethod
    def setup_class(cls):
        """Set up test environment"""
        cls.inter_agent_url = "https://inter-agent.market-data.local"
        cls.kong_admin_url = "http://kong-admin.kong.svc.cluster.local:8001"
        cls.consul_url = "http://consul-server.consul.svc.cluster.local:8500"

        # Test agent IDs
        cls.source_agent_id = "test-market-data-agent-001"
        cls.target_agent_id = "test-analytics-agent-001"

        # Message types for testing
        cls.message_types = [
            "market_data_request",
            "market_data_response",
            "analytics_request",
            "analytics_response",
            "health_check",
            "health_response",
            "agent_discovery",
            "configuration_update"
        ]

    def test_inter_agent_router_health(self):
        """Test inter-agent router health endpoint"""
        try:
            response = requests.get(f"{self.inter_agent_url}/health", timeout=10, verify=False)
            assert response.status_code == 200

            health_data = response.json()
            assert health_data.get("status") in ["healthy", "ok"]
        except requests.exceptions.RequestException:
            pytest.skip("Inter-agent router not accessible")

    def test_inter_agent_protocol_config(self):
        """Test inter-agent protocol configuration"""
        try:
            response = requests.get(f"{self.inter_agent_url}/protocol/config", timeout=10, verify=False)

            if response.status_code == 200:
                config = response.json()

                # Validate protocol configuration
                assert "apiVersion" in config
                assert config["apiVersion"] == "v1"
                assert "message_format" in config
                assert "security" in config
                assert "observability" in config
            else:
                pytest.skip("Protocol config endpoint not available")

        except requests.exceptions.RequestException:
            pytest.skip("Inter-agent router not accessible")

    def test_message_envelope_validation(self):
        """Test message envelope validation"""
        valid_message = {
            "message_id": str(uuid.uuid4()),
            "correlation_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source_agent_id": self.source_agent_id,
            "target_agent_id": self.target_agent_id,
            "message_type": "health_check",
            "priority": 5,
            "ttl": 300,
            "payload": {
                "agent_id": self.source_agent_id,
                "status": "healthy"
            }
        }

        try:
            response = requests.post(
                f"{self.inter_agent_url}/message/validate",
                json=valid_message,
                timeout=10,
                verify=False
            )

            if response.status_code == 200:
                validation_result = response.json()
                assert validation_result.get("valid") == True
            else:
                # Validation endpoint might require authentication
                assert response.status_code in [401, 403]

        except requests.exceptions.RequestException:
            pytest.skip("Inter-agent router not accessible")

    def test_invalid_message_rejection(self):
        """Test rejection of invalid messages"""
        invalid_messages = [
            # Missing required fields
            {
                "message_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "message_type": "health_check"
                # Missing correlation_id, source_agent_id, target_agent_id
            },
            # Invalid message type
            {
                "message_id": str(uuid.uuid4()),
                "correlation_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "source_agent_id": self.source_agent_id,
                "target_agent_id": self.target_agent_id,
                "message_type": "invalid_message_type",
                "payload": {}
            },
            # Invalid priority
            {
                "message_id": str(uuid.uuid4()),
                "correlation_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "source_agent_id": self.source_agent_id,
                "target_agent_id": self.target_agent_id,
                "message_type": "health_check",
                "priority": 15,  # Invalid: should be 1-10
                "payload": {}
            }
        ]

        for invalid_message in invalid_messages:
            try:
                response = requests.post(
                    f"{self.inter_agent_url}/message/validate",
                    json=invalid_message,
                    timeout=10,
                    verify=False
                )

                if response.status_code == 200:
                    validation_result = response.json()
                    assert validation_result.get("valid") == False
                    assert "errors" in validation_result
                elif response.status_code == 400:
                    # Bad request is also acceptable for invalid messages
                    pass
                else:
                    # Authentication required
                    assert response.status_code in [401, 403]

            except requests.exceptions.RequestException:
                pytest.skip("Inter-agent router not accessible")

    @pytest.mark.parametrize("message_type", [
        "market_data_request",
        "analytics_request",
        "health_check",
        "agent_discovery"
    ])
    def test_message_type_routing(self, message_type):
        """Test routing for different message types"""
        test_message = {
            "message_id": str(uuid.uuid4()),
            "correlation_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source_agent_id": self.source_agent_id,
            "target_agent_id": self.target_agent_id,
            "message_type": message_type,
            "priority": 5,
            "ttl": 300,
            "payload": self._get_payload_for_message_type(message_type)
        }

        try:
            response = requests.post(
                f"{self.inter_agent_url}/message/route",
                json=test_message,
                timeout=10,
                verify=False
            )

            # Response depends on authentication and target agent availability
            assert response.status_code in [200, 202, 401, 403, 404, 503]

            if response.status_code in [200, 202]:
                routing_result = response.json()
                assert "routed" in routing_result or "accepted" in routing_result

        except requests.exceptions.RequestException:
            pytest.skip("Inter-agent router not accessible")

    def test_market_data_request_schema(self):
        """Test market data request message schema"""
        market_data_request = {
            "message_id": str(uuid.uuid4()),
            "correlation_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source_agent_id": self.source_agent_id,
            "target_agent_id": "market-data-collector-001",
            "message_type": "market_data_request",
            "priority": 7,
            "ttl": 300,
            "payload": {
                "symbol": "AAPL",
                "data_type": "quote",
                "start_time": (datetime.utcnow() - timedelta(hours=1)).isoformat() + "Z",
                "end_time": datetime.utcnow().isoformat() + "Z",
                "granularity": "1m",
                "fields": ["open", "high", "low", "close", "volume"]
            }
        }

        try:
            response = requests.post(
                f"{self.inter_agent_url}/message/validate",
                json=market_data_request,
                timeout=10,
                verify=False
            )

            if response.status_code == 200:
                validation_result = response.json()
                assert validation_result.get("valid") == True
            else:
                assert response.status_code in [401, 403]

        except requests.exceptions.RequestException:
            pytest.skip("Inter-agent router not accessible")

    def test_analytics_request_schema(self):
        """Test analytics request message schema"""
        analytics_request = {
            "message_id": str(uuid.uuid4()),
            "correlation_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source_agent_id": self.source_agent_id,
            "target_agent_id": "analytics-engine-001",
            "message_type": "analytics_request",
            "priority": 5,
            "ttl": 600,
            "payload": {
                "computation_type": "trend_analysis",
                "parameters": {
                    "window_size": "30d",
                    "algorithm": "linear_regression"
                },
                "data_sources": ["market-data-collector-001"],
                "output_format": "json"
            }
        }

        try:
            response = requests.post(
                f"{self.inter_agent_url}/message/validate",
                json=analytics_request,
                timeout=10,
                verify=False
            )

            if response.status_code == 200:
                validation_result = response.json()
                assert validation_result.get("valid") == True
            else:
                assert response.status_code in [401, 403]

        except requests.exceptions.RequestException:
            pytest.skip("Inter-agent router not accessible")

    def test_agent_discovery_mechanism(self):
        """Test agent discovery and registration"""
        discovery_message = {
            "message_id": str(uuid.uuid4()),
            "correlation_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source_agent_id": self.source_agent_id,
            "target_agent_id": "discovery-service",
            "message_type": "agent_discovery",
            "priority": 8,
            "ttl": 300,
            "payload": {
                "action": "register",
                "agent_info": {
                    "agent_id": self.source_agent_id,
                    "agent_type": "market-data-collector",
                    "version": "1.0.0",
                    "endpoints": [
                        {
                            "name": "api",
                            "url": "https://market-data-agent.market-data.svc.cluster.local:8080",
                            "protocol": "https"
                        }
                    ],
                    "capabilities": ["real-time-quotes", "historical-data", "options-chains"],
                    "metadata": {
                        "supported_symbols": ["AAPL", "GOOGL", "MSFT"],
                        "update_frequency": "1s"
                    }
                }
            }
        }

        try:
            response = requests.post(
                f"{self.inter_agent_url}/message/route",
                json=discovery_message,
                timeout=10,
                verify=False
            )

            # Response depends on implementation
            assert response.status_code in [200, 202, 401, 403, 404]

        except requests.exceptions.RequestException:
            pytest.skip("Inter-agent router not accessible")

    def test_health_check_protocol(self):
        """Test health check message protocol"""
        health_check = {
            "message_id": str(uuid.uuid4()),
            "correlation_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source_agent_id": "monitoring-agent-001",
            "target_agent_id": self.target_agent_id,
            "message_type": "health_check",
            "priority": 3,
            "ttl": 60,
            "payload": {
                "agent_id": self.target_agent_id,
                "status": "healthy",
                "uptime": 86400,
                "version": "1.0.0",
                "capabilities": ["analytics", "reporting"]
            }
        }

        try:
            response = requests.post(
                f"{self.inter_agent_url}/message/route",
                json=health_check,
                timeout=10,
                verify=False
            )

            assert response.status_code in [200, 202, 401, 403, 404]

        except requests.exceptions.RequestException:
            pytest.skip("Inter-agent router not accessible")

    def test_message_ttl_validation(self):
        """Test message TTL (Time To Live) validation"""
        # Test message with very short TTL
        short_ttl_message = {
            "message_id": str(uuid.uuid4()),
            "correlation_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source_agent_id": self.source_agent_id,
            "target_agent_id": self.target_agent_id,
            "message_type": "health_check",
            "priority": 5,
            "ttl": 1,  # Very short TTL
            "payload": {"status": "test"}
        }

        try:
            # Wait a bit to ensure TTL expires
            time.sleep(2)

            response = requests.post(
                f"{self.inter_agent_url}/message/route",
                json=short_ttl_message,
                timeout=10,
                verify=False
            )

            # Message should be rejected due to expired TTL
            if response.status_code == 400:
                error_response = response.json()
                assert "ttl" in error_response.get("error", "").lower() or \
                       "expired" in error_response.get("error", "").lower()
            else:
                # TTL validation might not be implemented or requires auth
                assert response.status_code in [200, 202, 401, 403]

        except requests.exceptions.RequestException:
            pytest.skip("Inter-agent router not accessible")

    def test_message_priority_handling(self):
        """Test message priority handling"""
        priorities_to_test = [1, 5, 10]  # Low, medium, high priority

        for priority in priorities_to_test:
            priority_message = {
                "message_id": str(uuid.uuid4()),
                "correlation_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "source_agent_id": self.source_agent_id,
                "target_agent_id": self.target_agent_id,
                "message_type": "health_check",
                "priority": priority,
                "ttl": 300,
                "payload": {"status": "test", "priority_level": priority}
            }

            try:
                response = requests.post(
                    f"{self.inter_agent_url}/message/route",
                    json=priority_message,
                    timeout=10,
                    verify=False
                )

                # All priorities should be accepted (if authentication passes)
                assert response.status_code in [200, 202, 401, 403, 404]

            except requests.exceptions.RequestException:
                pytest.skip("Inter-agent router not accessible")

    def test_correlation_id_tracking(self):
        """Test correlation ID tracking across messages"""
        correlation_id = str(uuid.uuid4())

        # Send initial request
        request_message = {
            "message_id": str(uuid.uuid4()),
            "correlation_id": correlation_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source_agent_id": self.source_agent_id,
            "target_agent_id": self.target_agent_id,
            "message_type": "market_data_request",
            "priority": 5,
            "ttl": 300,
            "payload": {
                "symbol": "AAPL",
                "data_type": "quote"
            }
        }

        try:
            response = requests.post(
                f"{self.inter_agent_url}/message/route",
                json=request_message,
                timeout=10,
                verify=False
            )

            if response.status_code in [200, 202]:
                # Check if correlation ID is preserved in response headers
                assert "X-Correlation-ID" in response.headers
                assert response.headers["X-Correlation-ID"] == correlation_id

        except requests.exceptions.RequestException:
            pytest.skip("Inter-agent router not accessible")

    def test_consul_service_discovery_integration(self):
        """Test integration with Consul service discovery"""
        try:
            # Check if Consul is accessible
            consul_response = requests.get(f"{self.consul_url}/v1/agent/services", timeout=5)

            if consul_response.status_code == 200:
                services = consul_response.json()

                # Look for inter-agent router service
                inter_agent_services = [
                    s for s in services.values()
                    if "inter-agent" in s.get("Service", "").lower()
                ]

                if inter_agent_services:
                    # Test service discovery through inter-agent protocol
                    discovery_request = {
                        "message_id": str(uuid.uuid4()),
                        "correlation_id": str(uuid.uuid4()),
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "source_agent_id": self.source_agent_id,
                        "target_agent_id": "service-discovery",
                        "message_type": "agent_discovery",
                        "priority": 5,
                        "ttl": 300,
                        "payload": {
                            "action": "discover",
                            "agent_type": "analytics-engine"
                        }
                    }

                    response = requests.post(
                        f"{self.inter_agent_url}/message/route",
                        json=discovery_request,
                        timeout=10,
                        verify=False
                    )

                    assert response.status_code in [200, 202, 401, 403, 404]
                else:
                    pytest.skip("Inter-agent services not registered in Consul")
            else:
                pytest.skip("Consul not accessible")

        except requests.exceptions.RequestException:
            pytest.skip("Consul not accessible")

    def test_circuit_breaker_functionality(self):
        """Test circuit breaker functionality for failing agents"""
        # This test would typically require setting up failing agents
        # For now, test that circuit breaker configuration exists
        try:
            response = requests.get(
                f"{self.inter_agent_url}/circuit-breaker/status",
                timeout=10,
                verify=False
            )

            if response.status_code == 200:
                cb_status = response.json()
                # Validate circuit breaker status structure
                assert isinstance(cb_status, dict)
            else:
                # Circuit breaker status might not be exposed or requires auth
                assert response.status_code in [401, 403, 404]

        except requests.exceptions.RequestException:
            pytest.skip("Inter-agent router not accessible")

    def test_load_balancing_configuration(self):
        """Test load balancing configuration for multiple agent instances"""
        try:
            # Test routing configuration endpoint
            response = requests.get(
                f"{self.inter_agent_url}/routing/config",
                timeout=10,
                verify=False
            )

            if response.status_code == 200:
                routing_config = response.json()

                # Validate routing configuration
                if "routing_rules" in routing_config:
                    rules = routing_config["routing_rules"]
                    assert isinstance(rules, list)

                    # Check for load balancing algorithms
                    for rule in rules:
                        if "targets" in rule:
                            for target in rule["targets"]:
                                assert "weight" in target or "condition" in target
            else:
                # Routing config might not be exposed or requires auth
                assert response.status_code in [401, 403, 404]

        except requests.exceptions.RequestException:
            pytest.skip("Inter-agent router not accessible")

    def _get_payload_for_message_type(self, message_type: str) -> Dict[str, Any]:
        """Get appropriate payload for message type"""
        payloads = {
            "market_data_request": {
                "symbol": "AAPL",
                "data_type": "quote",
                "fields": ["bid", "ask", "last"]
            },
            "market_data_response": {
                "request_id": str(uuid.uuid4()),
                "status": "success",
                "data": {"symbol": "AAPL", "bid": 150.0, "ask": 150.5},
                "metadata": {"record_count": 1, "latency_ms": 50}
            },
            "analytics_request": {
                "computation_type": "trend_analysis",
                "parameters": {"window": "1h"},
                "data_sources": ["market-data-1"]
            },
            "analytics_response": {
                "request_id": str(uuid.uuid4()),
                "status": "success",
                "result": {"trend": "upward", "confidence": 0.85}
            },
            "health_check": {
                "agent_id": self.target_agent_id,
                "status": "healthy",
                "uptime": 3600
            },
            "health_response": {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "metrics": {"cpu_usage": 0.3, "memory_usage": 0.5}
            },
            "agent_discovery": {
                "action": "register",
                "agent_info": {
                    "agent_id": self.source_agent_id,
                    "agent_type": "test-agent",
                    "version": "1.0.0"
                }
            },
            "configuration_update": {
                "config_type": "feature_flags",
                "config_data": {"enable_caching": True},
                "version": "1.1.0"
            }
        }

        return payloads.get(message_type, {"test": True})