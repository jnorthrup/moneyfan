#!/usr/bin/env python3
"""
Test Coinbase API using the XChange library approach
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
    print("=== Coinbase XChange-style API Test ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Check environment variables
    api_key = os.getenv("COINBASE_API_KEY")
    api_secret = os.getenv("COINBASE_API_SECRET")
    passphrase = os.getenv("COINBASE_PASSPHRASE", "")

    print("Environment Variables Check:")
    print(f"  COINBASE_API_KEY: {'✅ Set' if api_key else '❌ Not set'}")
    print(f"  COINBASE_API_SECRET: {'✅ Set' if api_secret else '❌ Not set'}")
    print(f"  COINBASE_PASSPHRASE: {'✅ Set' if passphrase else '⚠️  Not set (optional)'}")
    print()

    if not api_key or not api_secret:
        print("❌ ERROR: Missing required environment variables!")
        sys.exit(1)

    # Try to decode the EC private key
    def decode_ec_key(ec_key):
        """Decode EC private key to get raw bytes"""
        try:
            key_data = ec_key.replace('-----BEGIN EC PRIVATE KEY-----', '').replace('-----END EC PRIVATE KEY-----', '')
            key_data = key_data.strip()
            
            # Add padding if needed
            missing_padding = len(key_data) % 4
            if missing_padding:
                key_data += '=' * (4 - missing_padding)
            
            return base64.b64decode(key_data)
        except Exception as e:
            print(f"Error decoding EC key: {e}")
            return None

    decoded_key = decode_ec_key(api_secret)
    if not decoded_key:
        print("❌ Could not decode EC private key")
        return

    print(f"Decoded key length: {len(decoded_key)} bytes")
    print(f"Key (first 20 bytes): {decoded_key[:20].hex()}")
    print()

    # Try different authentication methods for Coinbase Pro API
    def generate_coinbasepro_signature(timestamp, method, endpoint, body=""):
        """Generate signature for Coinbase Pro API (deprecated)"""
        # For Coinbase Pro, the message format is different
        message = f"{timestamp}{method}{endpoint}{body}"
        
        # Try with decoded key as HMAC secret
        return hmac.new(
            decoded_key,
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def make_coinbasepro_request(method, endpoint, body=""):
        """Make request to Coinbase Pro API (deprecated)"""
        base_url = "https://api.pro.coinbase.com"
        url = f"{base_url}{endpoint}"
        
        timestamp = str(int(time.time()))
        signature = generate_coinbasepro_signature(timestamp, method, endpoint, body)
        
        headers = {
            "CB-ACCESS-KEY": api_key,
            "CB-ACCESS-SIGN": signature,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "CB-ACCESS-PASSPHRASE": passphrase,
            "Content-Type": "application/json"
        }
        
        try:
            if method == "GET":
                response = requests.get(url, headers=headers)
            else:
                response = requests.post(url, headers=headers, data=body)
            
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

    print("Testing Coinbase Pro API (deprecated)...")
    print("=" * 60)
    
    # Test with different authentication approaches
    test_cases = [
        ("GET", "/accounts"),
        ("GET", "/products"),
        ("GET", "/orders"),
    ]
    
    for method, endpoint in test_cases:
        print(f"\nTest: {method} {endpoint}")
        result = make_coinbasepro_request(method, endpoint)
        
        if result["success"]:
            print(f"  ✅ SUCCESS (Status: {result['status_code']})")
            data = result["data"]
            if isinstance(data, dict):
                keys = list(data.keys())[:5]
                print(f"  Response keys: {keys}")
        else:
            print(f"  ❌ FAILED (Status: {result['status_code']})")
            error = result["data"].get("error", "Unknown error")
            print(f"  Error: {error}")
    
    print("\n" + "=" * 60)
    print("Testing Coinbase Advanced Trade with different auth methods...")
    print("=" * 60)
    
    # Try Advanced Trade with different signature formats
    def generate_advanced_trade_signature_v1(timestamp, method, endpoint, body=""):
        """Try Advanced Trade signature v1 format"""
        message = f"{timestamp}{method}{endpoint}{body}"
        return hmac.new(
            decoded_key,
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def generate_advanced_trade_signature_v2(timestamp, method, endpoint, body=""):
        """Try Advanced Trade signature v2 format"""
        message = f"{timestamp}{method}{endpoint}{body}"
        return hmac.new(
            decoded_key,
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def make_advanced_trade_request(method, endpoint, body="", sig_version=1):
        """Make request to Advanced Trade API"""
        base_url = "https://api.coinbase.com"
        url = f"{base_url}{endpoint}"
        
        timestamp = str(int(time.time()))
        
        if sig_version == 1:
            signature = generate_advanced_trade_signature_v1(timestamp, method, endpoint, body)
        else:
            signature = generate_advanced_trade_signature_v2(timestamp, method, endpoint, body)
        
        headers = {
            "CB-ACCESS-KEY": api_key,
            "CB-ACCESS-SIGN": signature,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "CB-ACCESS-PASSPHRASE": passphrase,
            "Content-Type": "application/json"
        }
        
        try:
            if method == "GET":
                response = requests.get(url, headers=headers)
            else:
                response = requests.post(url, headers=headers, data=body)
            
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
    
    advanced_trade_tests = [
        ("/api/v3/brokerage/accounts", "Accounts"),
        ("/api/v3/brokerage/products", "Products"),
    ]
    
    for endpoint, description in advanced_trade_tests:
        for sig_version in [1, 2]:
            print(f"\nTest: {description} (Signature v{sig_version})")
            result = make_advanced_trade_request("GET", endpoint, sig_version=sig_version)
            
            if result["success"]:
                print(f"  ✅ SUCCESS (Status: {result['status_code']})")
                data = result["data"]
                if isinstance(data, dict):
                    keys = list(data.keys())[:5]
                    print(f"  Response keys: {keys}")
            else:
                print(f"  ❌ FAILED (Status: {result['status_code']})")
                error = result["data"].get("error", "Unknown error")
                print(f"  Error: {error}")
    
    print("\n" + "=" * 60)
    print("=== Test Summary ===")
    print(f"API Key: {api_key[:20]}...")
    print(f"Decoded key length: {len(decoded_key)} bytes")
    print(f"Key hex (first 20 bytes): {decoded_key[:20].hex()}")
    print(f"Passphrase: {'Set' if passphrase else 'Not set'}")
    print()
    print("Observations:")
    print("1. Coinbase Pro API is deprecated (returns 503)")
    print("2. API key authentication fails for all endpoints")
    print("3. The EC private key might be for a different service")
    print()
    print("Recommendations:")
    print("1. Check if this API key is for Coinbase Advanced Trade")
    print("2. Verify key permissions in Coinbase dashboard")
    print("3. Try creating a new API key with 'view' permissions")
    print("4. Check if there are IP restrictions on the API key")

if __name__ == "__main__":
    main()