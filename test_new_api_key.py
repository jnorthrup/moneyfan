#!/usr/bin/env python3
import os
import time
import hmac
import hashlib
import requests

@pytest.mark.skip(reason="Requires live Coinbase credentials")
def test_coinbase_api(api_key, api_secret):
    """Test Coinbase API with new credentials"""
    print("Testing Coinbase API...")
    
    # Test public endpoint (should work without auth)
    response = requests.get("https://api.coinbase.com/v2/exchange-rates?currency=BTC")
    if response.ok:
        print("✅ Public API works")
        btc_rate = response.json()["data"]["rates"]["USD"]
        print(f"   BTC/USD: ${btc_rate}")
    else:
        print(f"❌ Public API failed: {response.status_code}")
    
    # Test private endpoint
    timestamp = str(int(time.time()))
    endpoint = "/v2/accounts"
    message = timestamp + "GET" + endpoint
    
    signature = hmac.new(
        api_secret.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    headers = {
        "CB-ACCESS-KEY": api_key,
        "CB-ACCESS-SIGN": signature,
        "CB-ACCESS-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }
    
    response = requests.get("https://api.coinbase.com" + endpoint, headers=headers)
    
    if response.ok:
        print("✅ Private API works!")
        data = response.json()
        if "data" in data:
            accounts = data["data"]
            print(f"   Found {len(accounts)} accounts")
            for account in accounts[:3]:
                name = account.get("name", "")
                balance = account.get("balance", {}).get("amount", "0")
                currency = account.get("balance", {}).get("currency", "")
                print(f"   - {name}: {balance} {currency}")
    else:
        print(f"❌ Private API failed: {response.status_code}")
        print(f"   Error: {response.text}")
    
    return response.ok

if __name__ == "__main__":
    api_key = os.getenv("COINBASE_API_KEY_NEW")
    api_secret = os.getenv("COINBASE_API_SECRET_NEW")
    
    if not api_key or not api_secret:
        print("❌ Set COINBASE_API_KEY_NEW and COINBASE_API_SECRET_NEW environment variables")
        sys.exit(1)
    
    success = test_coinbase_api(api_key, api_secret)
    sys.exit(0 if success else 1)
