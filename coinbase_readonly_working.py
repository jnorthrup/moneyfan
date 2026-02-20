#!/usr/bin/env python3
"""
Working Coinbase Readonly API Implementation
This script demonstrates a working readonly API client
"""

import os
import sys
import time
import hmac
import hashlib
import json
from typing import Optional, Dict, Any

try:
    import requests
except ImportError:
    os.system("pip3 install requests")
    import requests

def make_coinbase_request(api_key: str, api_secret: str, endpoint: str, method: str = "GET") -> Optional[Dict]:
    """Make authenticated request to Coinbase API"""
    
    # Ensure we have standard HMAC secret (not EC private key)
    if "BEGIN EC PRIVATE KEY" in api_secret:
        print("❌ EC private key format detected!")
        print("   Please create a new API key with standard HMAC secret")
        return None
    
    timestamp = str(int(time.time()))
    message = timestamp + method + endpoint
    
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
    
    url = f"https://api.coinbase.com{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        else:
            return None
        
        if response.ok:
            return response.json()
        else:
            print(f"Request failed: {response.status_code}")
            print(f"Error: {response.text}")
            return None
    except Exception as e:
        print(f"Error making request: {e}")
        return None

def main():
    print("=== Working Coinbase Readonly API ===")
    print()
    
    api_key = os.getenv("COINBASE_API_KEY_NEW")
    api_secret = os.getenv("COINBASE_API_SECRET_NEW")
    
    if not api_key or not api_secret:
        print("❌ Set COINBASE_API_KEY_NEW and COINBASE_API_SECRET_NEW")
        print()
        print("Get new API key from:")
        print("  https://www.coinbase.com/settings/api")
        print()
        print("Set environment variables:")
        print('  export COINBASE_API_KEY_NEW="your-api-key"')
        print('  export COINBASE_API_SECRET_NEW="your-api-secret"')
        return
    
    # Test public endpoint (no auth needed)
    print("1. Testing public endpoint...")
    try:
        response = requests.get("https://api.coinbase.com/v2/exchange-rates?currency=BTC")
        if response.ok:
            data = response.json()
            btc_rate = data["data"]["rates"]["USD"]
            print(f"✅ Public API works: BTC/USD = ${btc_rate}")
        else:
            print(f"❌ Public API failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test private endpoint
    print("\n2. Testing private endpoint...")
    accounts = make_coinbase_request(api_key, api_secret, "/v2/accounts")
    
    if accounts:
        print("✅ Private API works!")
        if "data" in accounts:
            account_list = accounts["data"]
            print(f"   Found {len(account_list)} accounts")
            for account in account_list[:5]:
                name = account.get("name", "")
                balance = account.get("balance", {}).get("amount", "0")
                currency = account.get("balance", {}).get("currency", "")
                print(f"   - {name}: {balance} {currency}")
    else:
        print("❌ Private API failed")
        print("   Check:")
        print("   - API key permissions (needs 'view' permissions)")
        print("   - API key is active")
        print("   - No IP restrictions")

if __name__ == "__main__":
    main()
