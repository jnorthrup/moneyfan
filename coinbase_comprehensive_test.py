#!/usr/bin/env python3
"""
Comprehensive Coinbase API Test
Tests multiple API endpoints and authentication methods
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
    print("=== Comprehensive Coinbase API Test ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Check environment variables
    api_key = os.getenv("COINBASE_API_KEY")
    api_secret = os.getenv("COINBASE_API_SECRET")

    print("Environment Variables Check:")
    print(f"  COINBASE_API_KEY: {'✅ Set' if api_key else '❌ Not set'}")
    print(f"  COINBASE_API_SECRET: {'✅ Set' if api_secret else '❌ Not set'}")
    print()

    if not api_key or not api_secret:
        print("❌ ERROR: Missing required environment variables!")
        sys.exit(1)

    # Detect API secret type
    is_ec_key = "BEGIN EC PRIVATE KEY" in api_secret
    print(f"API Secret Type: {'EC Private Key' if is_ec_key else 'Standard HMAC'}")
    print()

    def get_hmac_secret():
        """Extract HMAC secret from API secret"""
        if is_ec_key:
            # Try to decode EC private key
            try:
                key_data = api_secret.replace('-----BEGIN EC PRIVATE KEY-----', '').replace('-----END EC PRIVATE KEY-----', '')
                key_data = key_data.strip()
                
                # Add padding if needed
                missing_padding = len(key_data) % 4
                if missing_padding:
                    key_data += '=' * (4 - missing_padding)
                
                decoded = base64.b64decode(key_data)
                print(f"  Decoded EC key: {len(decoded)} bytes")
                
                # Try different approaches to get HMAC secret
                # Approach 1: Last 32 bytes
                if len(decoded) >= 32:
                    return decoded[-32:]
                # Approach 2: Entire key
                return decoded
            except Exception as e:
                print(f"  Error decoding EC key: {e}")
                return None
        else:
            return api_secret.encode('utf-8')

    hmac_secret = get_hmac_secret()
    if not hmac_secret:
        print("❌ Could not extract HMAC secret")
        return

    print(f"HMAC Secret (hex): {hmac_secret.hex()[:32]}...")
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

    def make_request(method, endpoint, base_url="https://api.coinbase.com", params=None, add_auth=True):
        """Make API request"""
        url = f"{base_url}{endpoint}"
        
        headers = {
            "CB-VERSION": "2024-01-01",
            "Content-Type": "application/json"
        }
        
        if add_auth:
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
                "success": response.ok,
                "headers": dict(response.headers)
            }
        except Exception as e:
            return {
                "status_code": 0,
                "data": {"error": str(e)},
                "success": False,
                "headers": {}
            }

    print("Testing Multiple Coinbase APIs...")
    print("=" * 80)
    
    # Test different Coinbase API endpoints
    test_cases = [
        {
            "name": "Coinbase v2 - User Info",
            "method": "GET",
            "endpoint": "/v2/user",
            "base_url": "https://api.coinbase.com",
            "add_auth": True,
        },
        {
            "name": "Coinbase v2 - Accounts (Auth)",
            "method": "GET",
            "endpoint": "/v2/accounts",
            "base_url": "https://api.coinbase.com",
            "add_auth": True,
        },
        {
            "name": "Coinbase v2 - Accounts (No Auth)",
            "method": "GET",
            "endpoint": "/v2/accounts",
            "base_url": "https://api.coinbase.com",
            "add_auth": False,
        },
        {
            "name": "Coinbase Pro - Accounts",
            "method": "GET",
            "endpoint": "/accounts",
            "base_url": "https://api.pro.coinbase.com",
            "add_auth": True,
        },
        {
            "name": "Coinbase Advanced Trade - Accounts",
            "method": "GET",
            "endpoint": "/api/v3/brokerage/accounts",
            "base_url": "https://api.coinbase.com",
            "add_auth": True,
        },
    ]
    
    for test in test_cases:
        print(f"\n🔍 Test: {test['name']}")
        print(f"   Method: {test['method']}")
        print(f"   Endpoint: {test['endpoint']}")
        print(f"   Base URL: {test['base_url']}")
        print(f"   Auth: {'Yes' if test['add_auth'] else 'No'}")
        
        result = make_request(
            test['method'],
            test['endpoint'],
            test['base_url'],
            add_auth=test['add_auth']
        )
        
        if result["success"]:
            print(f"   ✅ SUCCESS (Status: {result['status_code']})")
            data = result["data"]
            if isinstance(data, dict):
                keys = list(data.keys())[:5]
                print(f"   Response keys: {keys}")
                if "data" in data:
                    items = data["data"]
                    if isinstance(items, list):
                        print(f"   Items count: {len(items)}")
                        if items:
                            sample = items[0]
                            if isinstance(sample, dict):
                                sample_keys = list(sample.keys())[:3]
                                print(f"   Sample keys: {sample_keys}")
        else:
            print(f"   ❌ FAILED (Status: {result['status_code']})")
            error = result["data"].get("error", "Unknown error")
            print(f"   Error: {error}")
            
            # Show response headers if available
            if result["headers"]:
                print(f"   Headers: {dict(result['headers'])}")

    print("\n" + "=" * 80)
    print("=== Public Market Data Tests (No Auth) ===")
    
    public_tests = [
        ("/v2/exchange-rates?currency=BTC", "BTC Exchange Rates"),
        ("/v2/prices/BTC-USD/spot", "BTC-USD Spot Price"),
        ("/v2/currencies", "Currencies"),
        ("/api/v3/brokerage/products", "Advanced Trade Products (No Auth)"),
    ]
    
    for endpoint, description in public_tests:
        try:
            response = requests.get(f"https://api.coinbase.com{endpoint}")
            if response.ok:
                print(f"✅ {description}: SUCCESS")
                data = response.json()
                if isinstance(data, dict):
                    if "data" in data and isinstance(data["data"], dict):
                        if "rates" in data["data"]:
                            btc_rate = data["data"]["rates"].get("USD", "N/A")
                            print(f"   BTC/USD: ${btc_rate}")
                        elif "amount" in data["data"]:
                            print(f"   Price: ${data['data']['amount']}")
                    elif "products" in data:
                        print(f"   Products count: {len(data['products'])}")
            else:
                print(f"❌ {description}: FAILED - {response.text}")
        except Exception as e:
            print(f"❌ {description}: ERROR - {e}")

    print("\n" + "=" * 80)
    print("=== Test Summary ===")
    print(f"API Key: {api_key[:20]}...")
    print(f"API Secret Type: {'EC Private Key' if is_ec_key else 'Standard HMAC'}")
    print(f"HMAC Secret (first 8 bytes): {hmac_secret[:8].hex() if hmac_secret else 'N/A'}")
    print()
    print("Next steps:")
    print("1. Check if your API key is for Coinbase Pro or Advanced Trade")
    print("2. Verify API key permissions in Coinbase settings")
    print("3. Check for IP restrictions in API key settings")
    print("4. Try creating a new API key with 'view' permissions only")

if __name__ == "__main__":
    main()