#!/usr/bin/env python3
"""
Coinbase Readonly API Test Script
This script attempts to connect to Coinbase using your API credentials
to perform readonly operations.
"""

import os
import sys
import time
from datetime import datetime

def main():
    print("=== Coinbase Readonly API Test ===")
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

    # Try to install required packages
    try:
        import requests
        import hmac
        import hashlib
        import base64
        import json
    except ImportError:
        print("Installing required packages...")
        os.system("pip3 install requests")
        import requests
        import hmac
        import hashlib
        import base64
        import json

    # Coinbase API endpoints
    BASE_URL = "https://api.coinbase.com/v2"
    ADVANCED_TRADE_URL = "https://api.coinbase.com/api/v3/brokerage"

    def generate_signature(timestamp, method, request_path, body=""):
        """Generate Coinbase API signature"""
        message = timestamp + method + request_path + body
        signature = hmac.new(
            api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def make_request(endpoint, method="GET", params=None, use_advanced_trade=False):
        """Make authenticated API request"""
        base_url = ADVANCED_TRADE_URL if use_advanced_trade else BASE_URL
        url = f"{base_url}{endpoint}"
        
        timestamp = str(int(time.time()))
        request_path = endpoint
        
        headers = {
            "CB-ACCESS-KEY": api_key,
            "CB-ACCESS-SIGN": generate_signature(timestamp, method, request_path),
            "CB-ACCESS-TIMESTAMP": timestamp,
            "CB-VERSION": "2024-01-01",
            "Content-Type": "application/json"
        }
        
        if passphrase:
            headers["CB-ACCESS-PASSPHRASE"] = passphrase
        
        try:
            if method == "GET":
                response = requests.get(url, headers=headers, params=params)
            else:
                response = requests.post(url, headers=headers, json=params)
            
            return response.json() if response.ok else {"error": response.text, "status": response.status_code}
        except Exception as e:
            return {"error": str(e)}

    print("Test 1: Get account balances (readonly)")
    try:
        # Get accounts (Advanced Trade API)
        result = make_request("/accounts", use_advanced_trade=True)
        if "error" not in result:
            print("✅ Account balances retrieved successfully!")
            accounts = result.get("accounts", [])
            print(f"   Number of accounts: {len(accounts)}")
            for account in accounts[:5]:  # Show first 5 accounts
                name = account.get("name", "")
                balance = account.get("available_balance", {}).get("value", "0")
                currency = account.get("available_balance", {}).get("currency", "")
                print(f"   - {name}: {balance} {currency}")
        else:
            print(f"❌ Error: {result['error']}")
    except Exception as e:
        print(f"❌ Account balances fetch failed: {e}")
    print()

    print("Test 2: Get market data (public API)")
    try:
        # Get BTC-USD ticker (public endpoint, no auth needed)
        response = requests.get("https://api.coinbase.com/v2/exchange-rates?currency=BTC")
        if response.ok:
            data = response.json()
            rates = data.get("data", {}).get("rates", {})
            btc_rate = rates.get("USD", "N/A")
            print(f"✅ BTC-USD exchange rate: ${btc_rate}")
        else:
            print(f"❌ Failed to get exchange rate: {response.text}")
    except Exception as e:
        print(f"❌ Market data fetch failed: {e}")
    print()

    print("Test 3: Get products (Advanced Trade)")
    try:
        # Get products list (Advanced Trade)
        result = make_request("/products", use_advanced_trade=True)
        if "error" not in result:
            products = result.get("products", [])
            print(f"✅ Products retrieved successfully!")
            print(f"   Total products: {len(products)}")
            sample_products = products[:5]
            for product in sample_products:
                product_id = product.get("product_id", "")
                price = product.get("price", "")
                print(f"   - {product_id}: ${price}")
        else:
            print(f"❌ Error: {result['error']}")
    except Exception as e:
        print(f"❌ Products fetch failed: {e}")
    print()

    print("=== Readonly API Test Complete ===")
    print(f"Status: {'API credentials valid' if api_key and api_secret else 'API credentials missing'}")

if __name__ == "__main__":
    main()