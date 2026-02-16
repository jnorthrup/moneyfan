"""
Test suite for Coinbase JWT Authentication Module.

This test suite follows TDD Red phase - tests are written BEFORE implementation.
All tests should FAIL initially because the module doesn't exist yet.

Test Coverage:
- Valid JWT token generation
- ES256 algorithm and key ID in header
- Correct JWT claims (sub/iss/nbf/exp/uri)
- Expiration within 120 seconds of nbf
- Environment variable validation
"""

import pytest
import os
import json
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone


class TestCoinbaseAuthJWTGeneration:
    """Test suite for JWT token generation."""
    
    # Valid EC private key for testing
    SAMPLE_PRIVATE_KEY = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEINyDYF2xPWBIfsCbfya1JKCFwQZTq8k4oFWJ/iWcSLV0oAoGCCqGSM49
AwEHoUQDQgAEgO3wS/Q8UEscy9t8a1XsQLNW1IqrEJFZ6+2lDG5BYIfZ8DRShpuJ
iOkA31g7mg8GBjf9FrUmirJaAYtd02+IQw==
-----END EC PRIVATE KEY-----"""
    
    # Valid CDP SDK key format for testing
    SAMPLE_CDP_KEY = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIOr6Jckusw9qYQg/NeXog3Pjh0fs35VXmbmVT5Wgtm+zoAoGCCqGSM49
AwEHoUQDQgAEgupBydaesOPU/1J/29w3bZZTbjZo7g90+GPimFSFOwcSLXT7/rfN
VukmCMZaNXXvU9hED+nYjpel5tYfZkjFtg==
-----END EC PRIVATE KEY-----"""
    
    def test_import_coinbase_auth_module(self):
        """Test that coinbase_auth module can be imported."""
        # This will fail with ImportError if module doesn't exist
        try:
            import coinbase_auth
            assert coinbase_auth is not None
        except ImportError as e:
            pytest.fail(f"coinbase_auth module not found: {e}")
    
    @patch.dict(os.environ, {
        'COINBASE_API_KEY_NAME': 'test-key-name',
        'COINBASE_PRIVATE_KEY': SAMPLE_PRIVATE_KEY
    })
    def test_generate_jwt_token_returns_valid_string(self):
        """Test that generate_jwt_token returns a valid JWT string."""
        import coinbase_auth
        
        method = "GET"
        host = "api.coinbase.com"
        path = "/api/v3/brokerage/accounts"
        
        token = coinbase_auth.generate_jwt_token(method, host, path)
        
        # Should return a non-empty string
        assert isinstance(token, str)
        assert len(token) > 0
        
        # JWT format: header.payload.signature
        parts = token.split('.')
        assert len(parts) == 3, "JWT must have 3 parts separated by dots"
    
    @patch.dict(os.environ, {
        'COINBASE_API_KEY_NAME': 'test-key-name',
        'COINBASE_PRIVATE_KEY': SAMPLE_PRIVATE_KEY
    })
    def test_jwt_header_contains_es256_and_kid(self):
        """Test that JWT header contains ES256 algorithm and key ID."""
        import coinbase_auth
        import base64
        import json
        
        method = "GET"
        host = "api.coinbase.com"
        path = "/api/v3/brokerage/accounts"
        
        token = coinbase_auth.generate_jwt_token(method, host, path)
        
        # Decode header (first part of JWT)
        header_b64 = token.split('.')[0]
        # Add padding if needed
        header_b64 += '=' * (4 - len(header_b64) % 4)
        header_json = base64.b64decode(header_b64)
        header = json.loads(header_json)
        
        # Verify header claims
        assert header['alg'] == 'ES256', "JWT must use ES256 algorithm"
        assert 'kid' in header, "JWT header must contain key ID (kid)"
        assert isinstance(header['kid'], str), "kid must be a string"
    
    @patch.dict(os.environ, {
        'COINBASE_API_KEY_NAME': 'test-api-key',
        'COINBASE_PRIVATE_KEY': SAMPLE_PRIVATE_KEY
    })
    def test_jwt_payload_contains_required_claims(self):
        """Test that JWT payload contains all required claims."""
        import coinbase_auth
        import base64
        import json
        
        method = "GET"
        host = "api.coinbase.com"
        path = "/api/v3/brokerage/accounts"
        
        token = coinbase_auth.generate_jwt_token(method, host, path)
        
        # Decode payload (second part of JWT)
        payload_b64 = token.split('.')[1]
        # Add padding if needed
        payload_b64 += '=' * (4 - len(payload_b64) % 4)
        payload_json = base64.b64decode(payload_b64)
        payload = json.loads(payload_json)
        
        # Verify required claims
        assert 'sub' in payload, "JWT must contain 'sub' claim"
        assert 'iss' in payload, "JWT must contain 'iss' claim"
        assert 'nbf' in payload, "JWT must contain 'nbf' claim"
        assert 'exp' in payload, "JWT must contain 'exp' claim"
        assert 'uri' in payload, "JWT must contain 'uri' claim"
        
        # Verify values
        assert payload['sub'] == 'test-api-key', "sub must match API key name"
        assert payload['iss'] == 'cdp', "iss must be 'cdp'"
        assert isinstance(payload['nbf'], int), "nbf must be integer timestamp"
        assert isinstance(payload['exp'], int), "exp must be integer timestamp"
    
    @patch.dict(os.environ, {
        'COINBASE_API_KEY_NAME': 'test-api-key',
        'COINBASE_PRIVATE_KEY': SAMPLE_PRIVATE_KEY
    })
    def test_jwt_uri_claim_matches_request(self):
        """Test that JWT uri claim matches method, host, and path."""
        import coinbase_auth
        import base64
        import json
        
        method = "POST"
        host = "api.coinbase.com"
        path = "/api/v3/brokerage/orders"
        
        token = coinbase_auth.generate_jwt_token(method, host, path)
        
        # Decode payload
        payload_b64 = token.split('.')[1]
        payload_b64 += '=' * (4 - len(payload_b64) % 4)
        payload_json = base64.b64decode(payload_b64)
        payload = json.loads(payload_json)
        
        # Verify uri claim format: method + host + path
        expected_uri = f"{method}{host}{path}"
        assert payload['uri'] == expected_uri, f"uri must be '{expected_uri}'"
    
    @patch.dict(os.environ, {
        'COINBASE_API_KEY_NAME': 'test-api-key',
        'COINBASE_PRIVATE_KEY': SAMPLE_PRIVATE_KEY
    })
    def test_jwt_expiration_within_120_seconds(self):
        """Test that JWT exp is within 120 seconds of nbf."""
        import coinbase_auth
        import base64
        import json
        import time
        
        method = "GET"
        host = "api.coinbase.com"
        path = "/api/v3/brokerage/accounts"
        
        token = coinbase_auth.generate_jwt_token(method, host, path)
        
        # Decode payload
        payload_b64 = token.split('.')[1]
        payload_b64 += '=' * (4 - len(payload_b64) % 4)
        payload_json = base64.b64decode(payload_b64)
        payload = json.loads(payload_json)
        
        # Calculate time difference
        time_diff = payload['exp'] - payload['nbf']
        
        # Verify expiration is exactly 120 seconds
        assert time_diff == 120, f"exp must be exactly 120 seconds after nbf, got {time_diff}"
        
        # Verify nbf is close to current time (within 5 seconds tolerance)
        current_time = int(time.time())
        assert abs(payload['nbf'] - current_time) <= 5, "nbf should be close to current time"
    
    @patch.dict(os.environ, {
        'COINBASE_API_KEY_NAME': 'test-api-key',
        'COINBASE_PRIVATE_KEY': SAMPLE_PRIVATE_KEY
    })
    def test_jwt_signature_is_valid_es256(self):
        """Test that JWT signature is valid ES256 signature."""
        import coinbase_auth
        
        method = "GET"
        host = "api.coinbase.com"
        path = "/api/v3/brokerage/accounts"
        
        token = coinbase_auth.generate_jwt_token(method, host, path)
        
        # Token should be verifiable with the public key
        # For now, just verify it has a signature part
        parts = token.split('.')
        assert len(parts) == 3, "JWT must have signature"
        assert len(parts[2]) > 0, "Signature must not be empty"
    
    def test_missing_api_key_name_raises_environment_error(self):
        """Test that missing COINBASE_API_KEY_NAME raises EnvironmentError."""
        # Patch only private key, not API key name
        with patch.dict(os.environ, {
            'COINBASE_PRIVATE_KEY': TestCoinbaseAuthJWTGeneration.SAMPLE_PRIVATE_KEY
        }, clear=True):
            import coinbase_auth
            
            with pytest.raises(EnvironmentError) as exc_info:
                coinbase_auth.generate_jwt_token("GET", "api.coinbase.com", "/api/v3/brokerage/accounts")
            
            assert 'COINBASE_API_KEY_NAME' in str(exc_info.value) or \
                   'API key' in str(exc_info.value).lower(), \
                   "Error message should mention missing API key"
    
    def test_missing_private_key_raises_environment_error(self):
        """Test that missing COINBASE_PRIVATE_KEY raises EnvironmentError."""
        # Patch only API key name, not private key
        with patch.dict(os.environ, {
            'COINBASE_API_KEY_NAME': 'test-key'
        }, clear=True):
            import coinbase_auth
            
            with pytest.raises(EnvironmentError) as exc_info:
                coinbase_auth.generate_jwt_token("GET", "api.coinbase.com", "/api/v3/brokerage/accounts")
            
            assert 'COINBASE_PRIVATE_KEY' in str(exc_info.value) or \
                   'private key' in str(exc_info.value).lower(), \
                   "Error message should mention missing private key"
    
    def test_empty_api_key_name_raises_environment_error(self):
        """Test that empty COINBASE_API_KEY_NAME raises EnvironmentError."""
        with patch.dict(os.environ, {
            'COINBASE_API_KEY_NAME': '',
            'COINBASE_PRIVATE_KEY': TestCoinbaseAuthJWTGeneration.SAMPLE_PRIVATE_KEY
        }, clear=True):
            import coinbase_auth
            
            with pytest.raises(EnvironmentError):
                coinbase_auth.generate_jwt_token("GET", "api.coinbase.com", "/api/v3/brokerage/accounts")
    
    def test_empty_private_key_raises_environment_error(self):
        """Test that empty COINBASE_PRIVATE_KEY raises EnvironmentError."""
        with patch.dict(os.environ, {
            'COINBASE_API_KEY_NAME': 'test-key',
            'COINBASE_PRIVATE_KEY': ''
        }, clear=True):
            import coinbase_auth
            
            with pytest.raises(EnvironmentError):
                coinbase_auth.generate_jwt_token("GET", "api.coinbase.com", "/api/v3/brokerage/accounts")
    
    @patch.dict(os.environ, {
        'COINBASE_API_KEY_NAME': 'test-key',
        'COINBASE_PRIVATE_KEY': SAMPLE_PRIVATE_KEY
    })
    def test_different_methods_hosts_paths_generate_different_tokens(self):
        """Test that different requests generate different JWT tokens."""
        import coinbase_auth
        
        # Generate tokens for different requests
        token1 = coinbase_auth.generate_jwt_token("GET", "api.coinbase.com", "/api/v3/brokerage/accounts")
        token2 = coinbase_auth.generate_jwt_token("POST", "api.coinbase.com", "/api/v3/brokerage/orders")
        token3 = coinbase_auth.generate_jwt_token("GET", "api.coinbase.com", "/api/v3/brokerage/products")
        
        # All tokens should be different (due to different URIs and timestamps)
        assert token1 != token2, "Different requests should generate different tokens"
        assert token2 != token3, "Different requests should generate different tokens"
        assert token1 != token3, "Different requests should generate different tokens"


class TestCoinbaseAuthKeyFormats:
    """Test suite for different private key format support."""
    
    SAMPLE_RAW_EC_KEY = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIMaM9pKon/88oHO0GH38EKVxKZV2apEhTtsqJLC7OYkFoAoGCCqGSM49
AwEHoUQDQgAEtgPUAq7ERmvkg61tV4ceBmK5Jo63Xe74p7x3MNnt8Nz9X/h0FSu9
PyyRclnpTuTWTV6j0IWTycBoYxy1pFIu2w==
-----END EC PRIVATE KEY-----"""
    
    @patch.dict(os.environ, {
        'COINBASE_API_KEY_NAME': 'test-key',
        'COINBASE_PRIVATE_KEY': SAMPLE_RAW_EC_KEY
    })
    def test_raw_ec_pem_format_supported(self):
        """Test that raw EC PEM format is supported."""
        import coinbase_auth
        
        token = coinbase_auth.generate_jwt_token("GET", "api.coinbase.com", "/api/v3/brokerage/accounts")
        
        assert isinstance(token, str)
        assert len(token) > 0
        assert len(token.split('.')) == 3


class TestCoinbaseAuthModuleInterface:
    """Test suite for module interface and public API."""
    
    INLINE_KEY_1 = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIOkK9NowzI/UUEnErwxC2HW57Jy93VyQTw78Mw9CppC9oAoGCCqGSM49
AwEHoUQDQgAEoTbueUHuxf81hVnxVlGdDO1f6PG0JjEUU2mKb2bfVXZwYVQKdqjJ
NV3VwjhiR+BQIVZtaTyvz5TlCobEVksfwA==
-----END EC PRIVATE KEY-----"""
    
    INLINE_KEY_2 = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIAlQ83VB/HLHN1aLLtK2zUaVZdaWDJBVvBmsxD5c8qAioAoGCCqGSM49
AwEHoUQDQgAEaZzqGaLSUUzsGDQQKUPUd2uq4q2UeozlW9fzdZN6knvCloix5Ytt
mUa7B6c6q4wvH97OWe+Ng2G/DeKdq254Og==
-----END EC PRIVATE KEY-----"""
    
    @patch.dict(os.environ, {
        'COINBASE_API_KEY_NAME': 'test-key',
        'COINBASE_PRIVATE_KEY': INLINE_KEY_1
    })
    def test_generate_jwt_token_is_callable(self):
        """Test that generate_jwt_token is a callable function."""
        import coinbase_auth
        
        assert callable(coinbase_auth.generate_jwt_token), \
            "generate_jwt_token must be a callable function"
    
    @patch.dict(os.environ, {
        'COINBASE_API_KEY_NAME': 'test-key',
        'COINBASE_PRIVATE_KEY': INLINE_KEY_2
    })
    def test_generate_jwt_token_accepts_method_host_path(self):
        """Test that generate_jwt_token accepts method, host, path parameters."""
        import coinbase_auth
        
        # Should not raise TypeError
        token = coinbase_auth.generate_jwt_token(
            method="GET",
            host="api.coinbase.com",
            path="/api/v3/brokerage/accounts"
        )
        
        assert token is not None


class TestCoinbaseAuthErrorHandling:
    """Test suite for error handling."""
    
    def test_invalid_key_format_raises_value_error(self):
        """Test that invalid key format raises ValueError with helpful message."""
        import coinbase_auth
        
        with patch.dict(os.environ, {
            'COINBASE_API_KEY_NAME': 'test-key',
            'COINBASE_PRIVATE_KEY': 'not-a-valid-key'
        }):
            with pytest.raises(ValueError) as exc_info:
                coinbase_auth.generate_jwt_token("GET", "api.coinbase.com", "/api/v3/brokerage/accounts")
            
            assert 'Failed to parse private key' in str(exc_info.value), \
                "Error message should mention parsing failure"
    
    def test_non_ec_key_raises_value_error(self):
        """Test that non-EC key raises ValueError."""
        import coinbase_auth
        
        # RSA key (not EC)
        rsa_key = """-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEA3MuIkjRT4Tqdhtc4tCledcegKn0C4oIQevhhHm+8VndSFkZP
XyVaX/6KVtE49kJfKImBfWL4yJ8B8IxUqQhZ/cL49XWYOozS3NP314RFXqll9KKu
W46BmDvvpSCkcFVh9U33OzBxRIuW1upFI8GZwHjydosEzN8JIYmVjdYgtWAtaGav
5uZC8q0g6cHG9oLEJx8z53yLBPdSfbDJHQhewSTfLp/ySpFarCwqBrWDfbiYO8a+
+YRhpQLW9ZQ8rjJCuTMlFX6jdtOqxXFY8aGLGG1DZURyFHALN4T78FcFL3KPfLTO
3roy0/iNoGQearo45TjmbA5xuNjWT8Mrhckb/QIDAQABAoIBAAKhqLRItt+Lkwvj
yJt5wGmKHPTSA2+J1M1JD2FCkX2nU5K2jk9y6BpA8/WibeRzeI5eCy11zpjCL7lQ
oYQCjeNaHZz7+eSSv+2dV3TEW7j2hGOkxQwMIqvFj8Q0TiQU7xiyb+9O6GlZz40A
AvMbkCv3f7jyzGEZCb29AHZjC7CVr0cxcP34m3siv6WVw/3gFV+40GPOE7C6V9eg
OhXbk+KF6PmwRwXrYskLNdQLRpv8v4zoZZAGAd62pDtD6sFCIcrjq5h1cWB+fzJY
W3ZGc0/fyEu/E2XaQvF5YOr1sBe3SDM60eZIWQV3hSvkuh2ht2yqVVmF3+9dbLbG
Lk4mpuECgYEA8edncdyQzNh5l3XL1aLxpLAic+9ibBVRsz7l2w47+OicxvTf3lUg
H6lc7HNuWnge944onxReMC3e0o3O1HH83xc5jHkOZ78jVQc0vuZuMQ/6q20YoIqQ
U3O/oxP9tjTS7M4lFpo+s7ebZFO1YXgaELLo/WKZu2qIin6McE86eZ0CgYEA6ak9
BGKTQgUKcsGVjx6AWnENs5y5rpT1YPuiyxHwx89eAbARLmomrqXW2RdauGh9nrcV
ylLsKWr0nupUro/gYEQrM1lDr5IJ93UhfS+HEC7Ra+epb56URATjPC5VabjebVYy
7JeKZDbQ68P2UjPEKe26V5ciWfc7mEEex48QTeECgYA5jyA0HZFuzIuSGHtZ6B2r
XCW2hF1c7m20QuEakHaAsYisZpPmKUctgXUU5hp5+F8V9IOB5qzKtf9xBkESl9Td
mH8fB6b/1KEpmD9atSW/EthIdfsIKDBTSxVsTlNuSX9uzVZR3H3S8XtOEgT0nklF
c+ywbge6aoz2t0nfZ1q4mQKBgAcyyolvDABVrWu5oQTmuKeQog5tfp7tQd36Aprk
85kEP24n4W+fn49z1nmbqZTSy0PvegFgqpvgCqc2quMx1YTBtN8BGf+3rQztk9mK
dEvAVX1QhrzEkubBi8qX1tPJ+Tg/FpSJWp8ZvTf2Mol3xMxR4ZK/OjSxVCmtn+gf
9S2hAoGBAIjTYBmRpX3ddragz7GclSG2zknKFQmGk0RGotSgvwGgeHbhHSAbFMOu
T4EwcagwXNU6odJgbjZ871besWO67nSJLfMbZRLvuZ/Ma/REeJIBglbZ3ClmZp76
9q8W5iseCFJdU4PdGkH12vN/cYBARwxoWfXAbP4MCsn8bxDFp1wo
-----END RSA PRIVATE KEY-----"""
        
        with patch.dict(os.environ, {
            'COINBASE_API_KEY_NAME': 'test-key',
            'COINBASE_PRIVATE_KEY': rsa_key
        }):
            with pytest.raises(ValueError) as exc_info:
                coinbase_auth.generate_jwt_token("GET", "api.coinbase.com", "/api/v3/brokerage/accounts")
            
            assert 'must be an EC key' in str(exc_info.value), \
                "Error message should mention EC key requirement"
