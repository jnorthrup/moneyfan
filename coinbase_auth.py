"""
Coinbase JWT Authentication Module

Generates JWT tokens for Coinbase Advanced Trade API authentication using ES256.
Supports both raw EC PEM format and CDP SDK key format.

Environment Variables:
    COINBASE_API_KEY_NAME: Your Coinbase API key name
    COINBASE_PRIVATE_KEY: Your EC private key in PEM format

Usage:
    import coinbase_auth
    token = coinbase_auth.generate_jwt_token("GET", "api.coinbase.com", "/api/v3/brokerage/accounts")
"""

import os
import time
import base64
from urllib.parse import urlparse

try:
    import requests
except ImportError:
    class _RequestsFallback:
        @staticmethod
        def Session():
            raise ImportError("requests is required. Install with: pip install requests")

    requests = _RequestsFallback()

try:
    import jwt
except ImportError:
    raise ImportError("PyJWT is required. Install with: pip install PyJWT")

try:
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization
except ImportError:
    raise ImportError("cryptography is required. Install with: pip install cryptography")


# Load environment variables from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _load_private_key() -> ec.EllipticCurvePrivateKey:
    """
    Load and parse the EC private key from environment variable.
    
    Supports:
    - Raw EC PEM format (both SEC1 EC PRIVATE KEY and PKCS8 PRIVATE KEY)
    - CDP SDK key format
    
    Returns:
        EllipticCurvePrivateKey: The parsed private key
        
    Raises:
        EnvironmentError: If COINBASE_PRIVATE_KEY is not set or empty
        ValueError: If the key format is invalid
    """
    private_key_pem = os.getenv("COINBASE_PRIVATE_KEY")
    
    if not private_key_pem:
        raise EnvironmentError(
            "COINBASE_PRIVATE_KEY environment variable is not set or empty"
        )
    
    # Handle escaped newlines (common in environment variables)
    private_key_pem = private_key_pem.replace('\\n', '\n').strip()
    
    try:
        # Load the private key from PEM format
        private_key = serialization.load_pem_private_key(
            private_key_pem.encode('utf-8'),
            password=None,
            backend=default_backend()
        )
        
        # Verify it's an EC key
        if not isinstance(private_key, ec.EllipticCurvePrivateKey):
            raise ValueError("Private key must be an EC key for ES256 algorithm")
            
        return private_key
        
    except Exception as e:
        raise ValueError(f"Failed to parse private key: {e}")


def _get_api_key_name() -> str:
    """
    Get the API key name from environment variable.
    
    Returns:
        str: The API key name
        
    Raises:
        EnvironmentError: If COINBASE_API_KEY_NAME is not set or empty
    """
    api_key_name = os.getenv("COINBASE_API_KEY_NAME")
    
    if not api_key_name:
        raise EnvironmentError(
            "COINBASE_API_KEY_NAME environment variable is not set or empty"
        )
    
    return api_key_name


def generate_jwt_token(method: str, host: str, path: str) -> str:
    """
    Generate a JWT token for Coinbase API authentication.
    
    The JWT includes:
    - Header: ES256 algorithm and key ID (kid)
    - Payload: sub (API key name), iss (cdp), nbf, exp, uri
    - Signature: ES256 signature using the private key
    
    Args:
        method: HTTP method (GET, POST, etc.)
        host: API host (e.g., "api.coinbase.com")
        path: API path (e.g., "/api/v3/brokerage/accounts")
        
    Returns:
        str: The JWT token
        
    Raises:
        EnvironmentError: If required environment variables are missing
        ValueError: If the private key is invalid
        
    Example:
        >>> token = generate_jwt_token("GET", "api.coinbase.com", "/api/v3/brokerage/accounts")
        >>> print(token)
        'eyJhbGciOiJFUzI1NiIsImtpZCI6I...'
    """
    # Load credentials
    api_key_name = _get_api_key_name()
    private_key = _load_private_key()
    
    # Calculate timestamps
    now = int(time.time())
    nbf = now
    exp = now + 120  # Token expires in 120 seconds
    
    # Build URI claim: method + host + path
    uri = f"{method}{host}{path}"
    
    # Extract key ID from private key (public key fingerprint)
    # For ES256, we use the base64url-encoded public key as kid
    public_key = private_key.public_key()
    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint
    )
    kid = base64.urlsafe_b64encode(public_bytes).decode('utf-8').rstrip('=')
    
    # Build JWT payload
    payload = {
        "sub": api_key_name,  # Subject: API key name
        "iss": "cdp",         # Issuer: Coinbase Cloud Platform
        "nbf": nbf,           # Not Before
        "exp": exp,           # Expiration
        "uri": uri            # Request URI
    }
    
    # Build JWT header
    headers = {
        "alg": "ES256",
        "kid": kid
    }
    
    # Sign and encode JWT
    token = jwt.encode(
        payload,
        private_key,
        algorithm="ES256",
        headers=headers
    )
    
    return token


class AuthenticationError(Exception):
    """Raised when authenticated Coinbase API calls are rejected."""


class CoinbaseClient:
    """Thin authenticated HTTP client for Coinbase Advanced Trade APIs."""

    def __init__(self, base_url: str | None = None, session: requests.Session | None = None):
        self.base_url = (base_url or os.getenv("COINBASE_API_URL", "https://api.coinbase.com")).rstrip("/")
        self._host = urlparse(self.base_url).netloc
        self._session = session or requests.Session()

    def _request(self, method: str, path: str, **kwargs):
        clean_path = path if path.startswith("/") else f"/{path}"
        token = generate_jwt_token(method.upper(), self._host, clean_path)
        headers = dict(kwargs.pop("headers", {}) or {})
        headers["Authorization"] = f"Bearer {token}"
        url = f"{self.base_url}{clean_path}"
        response = self._session.request(method=method.upper(), url=url, headers=headers, **kwargs)
        if response.status_code == 401:
            raise AuthenticationError(
                f"Coinbase authentication failed (401) for {method.upper()} {clean_path}: {getattr(response, 'text', '')}"
            )
        return response

    def get(self, path: str, **kwargs):
        return self._request("GET", path, **kwargs)

    def post(self, path: str, json: dict | None = None, **kwargs):
        return self._request("POST", path, json=json, **kwargs)
