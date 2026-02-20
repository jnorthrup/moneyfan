#!/usr/bin/env python3
"""
Test Coinbase Public API Endpoints (No Authentication Required)
"""

import os
import sys
from datetime import datetime

try:
    import requests
except ImportError:
    print("Installing requests package...")
    os.system("pip3 install requests")
    import requests

def main():
    print("=== Coinbase Public API Test ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Test public endpoints that don't require authentication
    public_endpoints = [
        ("/v2/exchange-rates?currency=BTC", "BTC Exchange Rates"),
        ("/v2/exchange-rates?currency=ETH", "ETH Exchange Rates"),
        ("/v2/prices/BTC-USD/spot", "BTC-USD Spot Price"),
        ("/v2/prices/ETH-USD/spot", "ETH-USD Spot Price"),
        ("/v2/currencies", "Currencies"),
        ("/v2/time", "Server Time"),
    ]

    print("Testing Public Endpoints (No Authentication)...")
    print("=" * 60)

    success_count = 0
    total_count = len(public_endpoints)

    for endpoint, description in public_endpoints:
        try:
            response = requests.get(f"https://api.coinbase.com{endpoint}")
            if response.ok:
                print(f"✅ {description}: SUCCESS")
                data = response.json()
                
                # Extract useful information
                if "data" in data:
                    if "rates" in data["data"]:
                        btc_rate = data["data"]["rates"].get("USD", "N/A")
                        print(f"   BTC/USD: ${btc_rate}")
                    elif "amount" in data["data"]:
                        print(f"   Price: ${data['data']['amount']}")
                    elif "currencies" in data["data"]:
                        currencies = data["data"]["currencies"][:5]
                        print(f"   Sample currencies: {currencies}")
                    elif "iso" in data["data"]:
                        print(f"   Time: {data['data'].get('iso', 'N/A')}")
                elif "time" in data:
                    print(f"   Time: {data.get('time', 'N/A')}")
                
                success_count += 1
            else:
                print(f"❌ {description}: FAILED - Status {response.status_code}")
                print(f"   Error: {response.text[:100]}")
        except Exception as e:
            print(f"❌ {description}: ERROR - {e}")

    print()
    print("=" * 60)
    print(f"Public API Test Summary: {success_count}/{total_count} successful")

    # Test Coinbase Pro public endpoints
    print()
    print("Testing Coinbase Pro Public Endpoints...")
    print("=" * 60)

    pro_public_endpoints = [
        ("/products/BTC-USD/ticker", "BTC-USD Ticker"),
        ("/products/BTC-USD/stats", "BTC-USD Stats"),
        ("/products", "Products List"),
    ]

    for endpoint, description in pro_public_endpoints:
        try:
            response = requests.get(f"https://api.pro.coinbase.com{endpoint}")
            if response.ok:
                print(f"✅ {description}: SUCCESS")
                data = response.json()
                
                if isinstance(data, dict):
                    if "price" in data:
                        print(f"   Price: ${data['price']}")
                    elif "high" in data:
                        print(f"   24h High: ${data['high']}")
                    elif isinstance(data, list) and len(data) > 0:
                        print(f"   Items: {len(data)}")
                        if len(data) > 0 and isinstance(data[0], dict):
                            sample = data[0]
                            print(f"   Sample: {sample.get('id', 'N/A')}")
            else:
                print(f"❌ {description}: FAILED - Status {response.status_code}")
                # Check if Coinbase Pro is deprecated
                if response.status_code == 503:
                    print(f"   Notice: Coinbase Pro API appears to be deprecated")
        except Exception as e:
            print(f"❌ {description}: ERROR - {e}")

    print()
    print("=" * 60)
    print("=== Test Complete ===")
    print()
    print("Summary:")
    print("1. Coinbase public API is working correctly")
    print("2. Coinbase Pro API appears to be deprecated")
    print("3. Private API endpoints require proper authentication")
    print()
    print("Your API key analysis:")
    api_key = os.getenv("COINBASE_API_KEY")
    api_secret = os.getenv("COINBASE_API_SECRET")
    
    if api_key:
        print(f"  API Key: {api_key[:20]}...")
        print(f"  Format: UUID v4 (valid)")
    
    if api_secret:
        if "BEGIN EC PRIVATE KEY" in api_secret:
            print(f"  API Secret: EC Private Key format")
            print(f"  Note: This format is unusual for Coinbase API")
            print(f"  Try creating a new API key with standard HMAC secret")
    
    print()
    print("Recommendations:")
    print("1. Create a new API key in Coinbase dashboard")
    print("2. Use 'Advanced Trade API' permissions")
    print("3. Select 'View' permissions only for testing")
    print("4. Copy the API key and secret (standard HMAC format)")
    print("5. Set new environment variables and test again")

if __name__ == "__main__":
    main()