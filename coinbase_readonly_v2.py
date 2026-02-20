#!/usr/bin/env python3
"""
Coinbase Readonly API Test Script - Version 2
Testing different Coinbase API endpoints and authentication methods
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
    print("=== Coinbase Readonly API Test v2 ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Check environment variables
    api_key = os.getenv("COINBASE_API_KEY")
    api_secret = os.getenv("COINBASE_API_SECRET")
    passphrase = os.getenv("COINBASE_PASSPHRASE")

    print("Environment Variables Check:")
    print(f"  COINBASE_API_KEY: {'✅ Set' if api_key else '❌ Not set'}")
    print(f"  COINBASE_API_SECRET: {'✅ Set' if api_secret else '❌ Not set'}")
    print(f"  COINBASE_PASSPHRASE: {'✅ Set' if passphrase else '⚠️  Not set (optional)'}")
    print()

    if not api_key or not api_secret:
        print("❌ ERROR: Missing required environment variables!")
        print("   Please set COINBASE_API_KEY and COINBASE_API_SECRET")
        sys.exit(1)

    # Test different API endpoints
    endpoints_to_test = [
        ("GET", "/v2/accounts", False, "Coinbase v2 Accounts"),
        ("GET", "/v2/accounts", True, "Coinbase v2 Accounts with Auth"),
        ("GET", "/api/v3/brokerage/accounts", True, "Advanced Trade Accounts"),
        ("GET", "/api/v3/brokerage/products", True, "Advanced Trade Products"),
    ]

    def generate_signature_v2(timestamp, method, endpoint, body=""):
        """Generate signature for Coinbase v2 API"""
        message = timestamp + method + endpoint + body
        signature = hmac.new(
            api_secret.encode('utf-8'),
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
            
            if passphrase:
                headers["CB-ACCESS-PASSPHRASE"] = passphrase
        
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

    print("Testing various Coinbase API endpoints...")
    print("=" * 60)
    
    for method, endpoint, use_auth, description in endpoints_to_test:
        print(f"\nTest: {description}")
        print(f"  Method: {method}")
        print(f"  Endpoint: {endpoint}")
        print(f"  Auth: {'Yes' if use_auth else 'No'}")
        
        result = make_request(method, endpoint, use_auth)
        
        if result["success"]:
            print(f"  ✅ SUCCESS (Status: {result['status_code']})")
            data = result["data"]
            if isinstance(data, dict):
                keys = list(data.keys())[:5]
                print(f"  Response keys: {keys}")
                if "accounts" in data:
                    print(f"  Accounts count: {len(data['accounts'])}")
                    for account in data["accounts"][:3]:
                        name = account.get("name", "")
                        balance = account.get("balance", {}).get("amount", "0")
                        currency = account.get("balance", {}).get("currency", "")
                        print(f"    - {name}: {balance} {currency}")
                elif "products" in data:
                    print(f"  Products count: {len(data['products'])}")
        else:
            print(f"  ❌ FAILED (Status: {result['status_code']})")
            error = result["data"].get("error", "Unknown error")
            print(f"  Error: {error}")
    
    print("\n" + "=" * 60)
    print("=== Public Market Data Test (No Auth Required) ===")
    
    # Test public endpoints
    public_endpoints = [
        "/v2/exchange-rates?currency=BTC",
        "/v2/prices/BTC-USD/spot",
        "/v2/currencies",
    ]
    
    for endpoint in public_endpoints:
        try:
            response = requests.get(f"https://api.coinbase.com{endpoint}")
            if response.ok:
                print(f"✅ {endpoint}: SUCCESS")
                data = response.json()
                if isinstance(data, dict):
                    if "data" in data:
                        if "rates" in data["data"]:
                            btc_rate = data["data"]["rates"].get("USD", "N/A")
                            print(f"   BTC/USD: ${btc_rate}")
                        elif "amount" in data["data"]:
                            print(f"   Price: ${data['data']['amount']}")
            else:
                print(f"❌ {endpoint}: FAILED - {response.text}")
        except Exception as e:
            print(f"❌ {endpoint}: ERROR - {e}")
    
    print("\n=== Test Complete ===")
    print(f"API Credentials: {'Valid' if api_key and api_secret else 'Missing'}")

if __name__ == "__main__":
    main()