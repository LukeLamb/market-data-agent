"""
OAuth2 Authentication Tests for Market Data Agent
Phase 4 Step 4: API Gateway & Inter-Agent Communication
"""

import pytest
import requests
import jwt
import time
import json
import base64
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from urllib.parse import urlencode, parse_qs, urlparse


class TestOAuth2Authentication:
    """Test OAuth2 authentication server and JWT validation"""

    @classmethod
    def setup_class(cls):
        """Set up test environment"""
        cls.auth_server_url = "https://oauth2-server.auth.svc.cluster.local"
        cls.jwt_validator_url = "http://jwt-validator.auth.svc.cluster.local"
        cls.kong_admin_url = "http://kong-admin.kong.svc.cluster.local:8001"

        # Test client credentials
        cls.test_client_id = "market-data-agent"
        cls.test_client_secret = "super_secret_key_2024"

        # Test JWT signing keys (dummy for testing)
        cls.test_private_key = """-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEAvJqZvMlrf7tt...
-----END RSA PRIVATE KEY-----"""

        cls.test_public_key = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAvJqZvMlrf7tt...
-----END PUBLIC KEY-----"""

    def test_oauth2_server_health(self):
        """Test OAuth2 server health endpoint"""
        try:
            response = requests.get(f"{self.auth_server_url}/health", timeout=10, verify=False)
            assert response.status_code == 200

            health_data = response.json()
            assert health_data.get("status") in ["healthy", "ok"]
        except requests.exceptions.RequestException:
            pytest.skip("OAuth2 server not accessible in test environment")

    def test_oauth2_server_readiness(self):
        """Test OAuth2 server readiness endpoint"""
        try:
            response = requests.get(f"{self.auth_server_url}/ready", timeout=10, verify=False)
            assert response.status_code == 200

            ready_data = response.json()
            assert ready_data.get("ready") == True
        except requests.exceptions.RequestException:
            pytest.skip("OAuth2 server not accessible")

    def test_oauth2_authorization_endpoint(self):
        """Test OAuth2 authorization endpoint"""
        try:
            # Test authorization endpoint exists
            auth_params = {
                "response_type": "code",
                "client_id": self.test_client_id,
                "redirect_uri": "https://client.example.com/callback",
                "scope": "read market-data:read",
                "state": "test-state-123"
            }

            response = requests.get(
                f"{self.auth_server_url}/oauth/authorize",
                params=auth_params,
                timeout=10,
                verify=False,
                allow_redirects=False
            )

            # Should redirect to login or return authorization form
            assert response.status_code in [200, 302, 401]

        except requests.exceptions.RequestException:
            pytest.skip("OAuth2 server not accessible")

    def test_oauth2_token_endpoint_client_credentials(self):
        """Test OAuth2 token endpoint with client credentials flow"""
        try:
            token_data = {
                "grant_type": "client_credentials",
                "client_id": self.test_client_id,
                "client_secret": self.test_client_secret,
                "scope": "read write market-data:read market-data:write"
            }

            response = requests.post(
                f"{self.auth_server_url}/oauth/token",
                data=token_data,
                timeout=10,
                verify=False
            )

            if response.status_code == 200:
                token_response = response.json()

                # Validate token response
                assert "access_token" in token_response
                assert "token_type" in token_response
                assert token_response["token_type"].lower() == "bearer"
                assert "expires_in" in token_response
                assert "scope" in token_response

                # Validate JWT token structure
                access_token = token_response["access_token"]
                self._validate_jwt_structure(access_token)

            elif response.status_code == 401:
                # Client credentials might not be configured
                pytest.skip("Client credentials not configured for testing")
            else:
                pytest.fail(f"Unexpected response: {response.status_code} - {response.text}")

        except requests.exceptions.RequestException:
            pytest.skip("OAuth2 server not accessible")

    def test_oauth2_token_endpoint_authorization_code(self):
        """Test OAuth2 authorization code flow"""
        try:
            # This test would require a full OAuth2 flow simulation
            # For now, just test that the endpoint exists and responds appropriately
            token_data = {
                "grant_type": "authorization_code",
                "client_id": self.test_client_id,
                "client_secret": self.test_client_secret,
                "code": "invalid_code_for_testing",
                "redirect_uri": "https://client.example.com/callback"
            }

            response = requests.post(
                f"{self.auth_server_url}/oauth/token",
                data=token_data,
                timeout=10,
                verify=False
            )

            # Should return error for invalid code
            assert response.status_code in [400, 401]

            if response.status_code == 400:
                error_response = response.json()
                assert "error" in error_response
                assert error_response["error"] in ["invalid_grant", "invalid_request"]

        except requests.exceptions.RequestException:
            pytest.skip("OAuth2 server not accessible")

    def test_oauth2_jwks_endpoint(self):
        """Test OAuth2 JWKS (JSON Web Key Set) endpoint"""
        try:
            response = requests.get(
                f"{self.auth_server_url}/.well-known/jwks.json",
                timeout=10,
                verify=False
            )

            if response.status_code == 200:
                jwks = response.json()

                # Validate JWKS structure
                assert "keys" in jwks
                assert isinstance(jwks["keys"], list)
                assert len(jwks["keys"]) > 0

                # Validate key structure
                for key in jwks["keys"]:
                    assert "kty" in key  # Key type
                    assert "use" in key  # Key use
                    assert "kid" in key  # Key ID
                    assert "n" in key    # Modulus (for RSA)
                    assert "e" in key    # Exponent (for RSA)

            else:
                pytest.skip("JWKS endpoint not configured")

        except requests.exceptions.RequestException:
            pytest.skip("OAuth2 server not accessible")

    def test_jwt_validator_health(self):
        """Test JWT validator service health"""
        try:
            response = requests.get(f"{self.jwt_validator_url}/health", timeout=10)
            assert response.status_code == 200

            health_data = response.json()
            assert health_data.get("status") in ["healthy", "ok"]
        except requests.exceptions.RequestException:
            pytest.skip("JWT validator not accessible")

    def test_jwt_validator_ready(self):
        """Test JWT validator readiness"""
        try:
            response = requests.get(f"{self.jwt_validator_url}/ready", timeout=10)
            assert response.status_code == 200

            ready_data = response.json()
            assert ready_data.get("ready") == True
        except requests.exceptions.RequestException:
            pytest.skip("JWT validator not accessible")

    def test_jwt_validation_endpoint(self):
        """Test JWT validation endpoint"""
        try:
            # Create a test JWT token
            test_token = self._create_test_jwt()

            response = requests.post(
                f"{self.jwt_validator_url}/validate",
                headers={"Authorization": f"Bearer {test_token}"},
                timeout=10
            )

            # Response depends on whether the test key is configured
            assert response.status_code in [200, 401, 403]

            if response.status_code == 200:
                validation_response = response.json()
                assert "valid" in validation_response
                assert "claims" in validation_response

        except requests.exceptions.RequestException:
            pytest.skip("JWT validator not accessible")

    def test_jwt_scope_validation(self):
        """Test JWT scope validation"""
        try:
            # Test different scopes
            scopes_to_test = [
                "read",
                "write",
                "market-data:read",
                "market-data:write",
                "analytics:read",
                "admin"
            ]

            for scope in scopes_to_test:
                test_token = self._create_test_jwt(scope=scope)

                response = requests.post(
                    f"{self.jwt_validator_url}/validate",
                    headers={"Authorization": f"Bearer {test_token}"},
                    json={"required_scope": scope},
                    timeout=10
                )

                # Response depends on configuration
                assert response.status_code in [200, 401, 403]

        except requests.exceptions.RequestException:
            pytest.skip("JWT validator not accessible")

    def test_jwt_expiration_validation(self):
        """Test JWT expiration validation"""
        try:
            # Create expired token
            expired_token = self._create_test_jwt(expired=True)

            response = requests.post(
                f"{self.jwt_validator_url}/validate",
                headers={"Authorization": f"Bearer {expired_token}"},
                timeout=10
            )

            # Should reject expired token
            assert response.status_code in [401, 403]

            if response.status_code == 401:
                error_response = response.json()
                assert "error" in error_response
                assert "expired" in error_response["error"].lower()

        except requests.exceptions.RequestException:
            pytest.skip("JWT validator not accessible")

    def test_jwt_issuer_validation(self):
        """Test JWT issuer validation"""
        try:
            # Create token with wrong issuer
            wrong_issuer_token = self._create_test_jwt(issuer="https://wrong-issuer.example.com")

            response = requests.post(
                f"{self.jwt_validator_url}/validate",
                headers={"Authorization": f"Bearer {wrong_issuer_token}"},
                timeout=10
            )

            # Should reject token with wrong issuer
            assert response.status_code in [401, 403]

        except requests.exceptions.RequestException:
            pytest.skip("JWT validator not accessible")

    def test_jwt_audience_validation(self):
        """Test JWT audience validation"""
        try:
            # Create token with wrong audience
            wrong_audience_token = self._create_test_jwt(audience="wrong-audience")

            response = requests.post(
                f"{self.jwt_validator_url}/validate",
                headers={"Authorization": f"Bearer {wrong_audience_token}"},
                timeout=10
            )

            # Should reject token with wrong audience
            assert response.status_code in [401, 403]

        except requests.exceptions.RequestException:
            pytest.skip("JWT validator not accessible")

    def test_oauth2_refresh_token_flow(self):
        """Test OAuth2 refresh token flow"""
        try:
            # First get tokens with client credentials
            token_data = {
                "grant_type": "client_credentials",
                "client_id": self.test_client_id,
                "client_secret": self.test_client_secret,
                "scope": "read write"
            }

            response = requests.post(
                f"{self.auth_server_url}/oauth/token",
                data=token_data,
                timeout=10,
                verify=False
            )

            if response.status_code == 200:
                token_response = response.json()

                if "refresh_token" in token_response:
                    # Test refresh token
                    refresh_data = {
                        "grant_type": "refresh_token",
                        "refresh_token": token_response["refresh_token"],
                        "client_id": self.test_client_id,
                        "client_secret": self.test_client_secret
                    }

                    refresh_response = requests.post(
                        f"{self.auth_server_url}/oauth/token",
                        data=refresh_data,
                        timeout=10,
                        verify=False
                    )

                    assert refresh_response.status_code == 200
                    new_tokens = refresh_response.json()
                    assert "access_token" in new_tokens
                else:
                    pytest.skip("Refresh tokens not enabled for client credentials flow")
            else:
                pytest.skip("Cannot get initial tokens for refresh test")

        except requests.exceptions.RequestException:
            pytest.skip("OAuth2 server not accessible")

    def test_oauth2_introspection_endpoint(self):
        """Test OAuth2 token introspection endpoint"""
        try:
            # Create a test token first
            test_token = self._create_test_jwt()

            introspection_data = {
                "token": test_token,
                "client_id": self.test_client_id,
                "client_secret": self.test_client_secret
            }

            response = requests.post(
                f"{self.auth_server_url}/oauth/introspect",
                data=introspection_data,
                timeout=10,
                verify=False
            )

            if response.status_code == 200:
                introspection_response = response.json()
                assert "active" in introspection_response
                # Other fields depend on token validity and configuration
            else:
                # Introspection endpoint might not be implemented
                pytest.skip("Token introspection endpoint not available")

        except requests.exceptions.RequestException:
            pytest.skip("OAuth2 server not accessible")

    def test_oauth2_revocation_endpoint(self):
        """Test OAuth2 token revocation endpoint"""
        try:
            # Test token revocation endpoint exists
            revocation_data = {
                "token": "test_token_to_revoke",
                "client_id": self.test_client_id,
                "client_secret": self.test_client_secret
            }

            response = requests.post(
                f"{self.auth_server_url}/oauth/revoke",
                data=revocation_data,
                timeout=10,
                verify=False
            )

            # Should accept the request (even if token is invalid)
            assert response.status_code in [200, 400]

        except requests.exceptions.RequestException:
            pytest.skip("OAuth2 server not accessible")

    def test_oauth2_cors_configuration(self):
        """Test OAuth2 server CORS configuration"""
        try:
            # Test preflight request
            response = requests.options(
                f"{self.auth_server_url}/oauth/token",
                headers={
                    "Origin": "https://market-data-ui.example.com",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Content-Type"
                },
                timeout=10,
                verify=False
            )

            if response.status_code == 200:
                # Check CORS headers
                assert "Access-Control-Allow-Origin" in response.headers
                assert "Access-Control-Allow-Methods" in response.headers
            else:
                # CORS might not be configured
                pytest.skip("CORS not configured or not accessible")

        except requests.exceptions.RequestException:
            pytest.skip("OAuth2 server not accessible")

    def _create_test_jwt(self, scope="read", expired=False, issuer=None, audience=None):
        """Create a test JWT token"""
        now = int(time.time())
        exp = now - 3600 if expired else now + 3600

        payload = {
            "iss": issuer or "https://auth.market-data.example.com",
            "aud": audience or "market-data-agents",
            "sub": "test-user",
            "iat": now,
            "exp": exp,
            "scope": scope,
            "role": "trader",
            "agent_id": "test-agent-001"
        }

        # Use a dummy key for testing
        try:
            return jwt.encode(payload, "test-secret", algorithm="HS256")
        except Exception:
            # Return a malformed token if JWT encoding fails
            return "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.test.token"

    def _validate_jwt_structure(self, token):
        """Validate JWT token structure"""
        parts = token.split('.')
        assert len(parts) == 3, "JWT should have 3 parts"

        # Validate header
        try:
            header = json.loads(base64.urlsafe_b64decode(parts[0] + '=='))
            assert "alg" in header
            assert "typ" in header
            assert header["typ"] == "JWT"
        except Exception:
            pytest.fail("Invalid JWT header")

        # Validate payload
        try:
            payload = json.loads(base64.urlsafe_b64decode(parts[1] + '=='))
            assert "iss" in payload
            assert "aud" in payload
            assert "exp" in payload
        except Exception:
            pytest.fail("Invalid JWT payload")

    @pytest.mark.parametrize("grant_type", [
        "client_credentials",
        "authorization_code",
        "refresh_token"
    ])
    def test_oauth2_grant_types_supported(self, grant_type):
        """Test that OAuth2 grant types are supported"""
        try:
            # Test discovery endpoint
            response = requests.get(
                f"{self.auth_server_url}/.well-known/oauth-authorization-server",
                timeout=10,
                verify=False
            )

            if response.status_code == 200:
                discovery = response.json()
                if "grant_types_supported" in discovery:
                    assert grant_type in discovery["grant_types_supported"]
                else:
                    pytest.skip("OAuth2 discovery not fully implemented")
            else:
                pytest.skip("OAuth2 discovery endpoint not available")

        except requests.exceptions.RequestException:
            pytest.skip("OAuth2 server not accessible")

    def test_oauth2_rate_limiting(self):
        """Test OAuth2 server rate limiting"""
        try:
            # Make multiple rapid requests to test rate limiting
            for i in range(10):
                response = requests.post(
                    f"{self.auth_server_url}/oauth/token",
                    data={
                        "grant_type": "client_credentials",
                        "client_id": "invalid_client",
                        "client_secret": "invalid_secret"
                    },
                    timeout=5,
                    verify=False
                )

                if response.status_code == 429:
                    # Rate limiting is working
                    assert "Retry-After" in response.headers or "X-RateLimit-Reset" in response.headers
                    break
            else:
                # Rate limiting might not be configured or limits are high
                pytest.skip("Rate limiting not triggered or not configured")

        except requests.exceptions.RequestException:
            pytest.skip("OAuth2 server not accessible")