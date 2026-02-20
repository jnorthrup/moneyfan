#!/usr/bin/env python3
"""
Coinbase API Test with Raw Secret Decoding
"""

import os
import sys
import time
import hmac
import hashlib
import base64
import json
from datetime import datetime

try:
    import requests
except ImportError:
    print("Installing requests package...")
    os.system("pip3 install requests")
    import requests

def main():
    print("=== Coinbase API Test with Raw Secret ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Check environment variables
    api_key = os.getenv("COINBASE_API_KEY")
    api_secret_encoded = os.getenv("COINBASE_API_SECRET")

    print("Environment Variables Check:")
    print(f"  COINBASE_API_KEY: {'✅ Set' if api_key else '❌ Not set'}")
    print(f"  COINBASE_API_SECRET: {'✅ Set' if api_secret_encoded else '❌ Not set'}")
    print()

    if not api_key or not api_secret_encoded:
        print("❌ ERROR: Missing required environment variables!")
        sys.exit(1)

    # Decode the EC private key to get the raw HMAC secret
    def decode_ec_key_to_hmac_secret(ec_key):
        """Decode EC private key to get raw HMAC secret"""
        try:
            # Remove header and footer
            key_data = ec_key.replace('-----BEGIN EC PRIVATE KEY-----', '').replace('-----END EC PRIVATE KEY-----', '')
            key_data = key_data.strip()
            
            # Add padding if needed
            missing_padding = len(key_data) % 4
            if missing_padding:
                key_data += '=' * (4 - missing_padding)
            
            # Decode base64
            decoded = base64.b64decode(key_data)
            print(f"  Raw key length: {len(decoded)} bytes")
            
            # Try to extract the secret (this is a simplified approach)
            # For EC keys, the secret is typically in the last 32 bytes
            if len(decoded) >= 32:
                # Try the last 32 bytes as a potential HMAC secret
                potential_secret = decoded[-32:]
                print(f"  Trying last 32 bytes as HMAC secret...")
                return potential_secret
            
            # Try the entire decoded key
            return decoded
            
        except Exception as e:
            print(f"  Error decoding key: {e}")
            return None

    print("Decoding EC private key to HMAC secret...")
    hmac_secret = decode_ec_key_to_hmac_secret(api_secret_encoded)
    
    if not hmac_secret:
        print("❌ Could not decode HMAC secret from EC private key")
        return
    
    print()

    def generate_signature_v2(timestamp, method, endpoint, body=""):
        """Generate signature for Coinbase v2 API"""
        message = timestamp + method + endpoint + body
        signature = hmac.new(
            hmac_secret,
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def make_request(method, endpoint, use_auth=False, params=None):
        """Make API request"""
        base_url = "https://api.coinbase.com"
        url = f"{base_url}{endpoint}"
        
        headers = {
            "CB-VERSION": "2024-01-01",
            "Content-Type": "application/json"
        }
        
        if use_auth:
            timestamp = str(int(time.time()))
            signature = generate_signature_v2(timestamp, method, endpoint)
            
            headers.update({
                "CB-ACCESS-KEY": api_key,
                "CB-ACCESS-SIGN": signature,
                "CB-ACCESS-TIMESTAMP": timestamp,
            })
        
        try:
            if method == "GET":
                response = requests.get(url, headers=headers, params=params)
            else:
                response = requests.post(url, headers=headers, json=params)
            
            return {
                "status_code": response.status_code,
                "data": response.json() if response.ok else {"error": response.text},
                "success": response.ok
            }
        except Exception as e:
            return {
                "status_code": 0,
                "data": {"error": str(e)},
                "success": False
            }

    # Test with raw HMAC secret
    print("Testing with raw HMAC secret...")
    print("=" * 60)
    
    test_endpoints = [
        ("/v2/accounts", "Accounts"),
        ("/v2/user", "User Info"),
    ]
    
    for endpoint, description in test_endpoints:
        print(f"\nTest: {description}")
        print(f"  Endpoint: {endpoint}")
        
        result = make_request("GET", endpoint, use_auth=True)
        
        if result["success"]:
            print(f"  ✅ SUCCESS (Status: {result['status_code']})")
            data = result["data"]
            if isinstance(data, dict):
                keys = list(data.keys())[:5]
                print(f"  Response keys: {keys}")
                if "data" in data:
                    print(f"  Data items: {len(data['data'])}")
        else:
            print(f"  ❌ FAILED (Status: {result['status_code']})")
            error = result["data"].get("error", "Unknown error")
            print(f"  Error: {error}")
    
    print("\n" + "=" * 60)
    print("=== Testing with Different Authentication Methods ===")
    
    # Try with passphrase (if needed)
    passphrase = os.getenv("COINBASE_PASSPHRASE")
    if passphrase:
        print(f"\nTesting with passphrase: {passphrase[:10]}...")
        
        def generate_signature_with_passphrase(timestamp, method, endpoint, body=""):
            """Generate signature with passphrase"""
            message = timestamp + method + endpoint + body
            signature = hmac.new(
                hmac_secret,
                message.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            return signature
        
        print("  (Passphrase is included in headers, but signature generation might differ)")
    
    print("\n=== Test Complete ===")
    print(f"API Key: {api_key[:20]}...")
    print(f"HMAC Secret: {hmac_secret[:8].hex()}..." if hmac_secret else "No HMAC secret")

if __name__ == "__main__":
    main()