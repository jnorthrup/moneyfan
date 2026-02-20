#!/usr/bin/env python3
"""
Create a new API key setup for Coinbase
This script will help you create a working API key setup
"""

import os
import sys
import time
import hmac
import hashlib
import json
from datetime import datetime

try:
    import requests
except ImportError:
    print("Installing requests package...")
    os.system("pip3 install requests")
    import requests

def main():
    print("=== Coinbase API Key Setup Guide ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    print("Current Status:")
    print("❌ Your current API key is not authenticating properly")
    print("✅ Public APIs work fine (Coinbase is accessible)")
    print("❌ Private APIs return 401 Unauthorized")
    print()

    print("Root Cause Analysis:")
    print("1. API key may be invalid/expired")
    print("2. API key may have wrong permissions")
    print("3. API key may be for a different service")
    print("4. API secret format is unusual (EC private key)")
    print()

    print("=" * 80)
    print("SOLUTION: Create New API Key")
    print("=" * 80)
    print()

    print("Step 1: Create API Key in Coinbase Dashboard")
    print("-" * 40)
    print("1. Go to: https://www.coinbase.com/settings/api")
    print("2. Click 'Create API Key'")
    print("3. Set permissions:")
    print("   ✓ Wallet:accounts:read")
    print("   ✓ Wallet:transactions:read")
    print("   ✓ Wallet:buys:read")
    print("   ✓ Wallet:sells:read")
    print("   ✓ User:read")
    print("   ✗ Avoid write permissions for testing")
    print("4. Set IP restrictions (optional)")
    print("5. Click 'Create API Key'")
    print("6. Save the API Key and Secret (standard HMAC format)")
    print()

    print("Step 2: Test the New API Key")
    print("-" * 40)
    
    # Create a test script for the new API key
    test_script = '''#!/usr/bin/env python3
import os
import time
import hmac
import hashlib
import requests

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
'''

    # Save the test script
    with open("test_new_api_key.py", "w") as f:
        f.write(test_script)
    
    print("Step 3: Set New Environment Variables")
    print("-" * 40)
    print("Run these commands in your terminal:")
    print()
    print("  export COINBASE_API_KEY_NEW=\"your-new-api-key\"")
    print("  export COINBASE_API_SECRET_NEW=\"your-new-api-secret\"")
    print()
    print("Then test with:")
    print("  python3 test_new_api_key.py")
    print()

    print("Step 4: Update Your Application")
    print("-" * 40)
    print("Once the new API key works, update your environment:")
    print()
    print("  export COINBASE_API_KEY=\"$COINBASE_API_KEY_NEW\"")
    print("  export COINBASE_API_SECRET=\"$COINBASE_API_SECRET_NEW\"")
    print()

    print("=" * 80)
    print("Alternative: Try Different API Endpoint")
    print("=" * 80)
    print()

    # Test if the current API key works with different endpoints
    print("Testing current API key with different endpoints...")
    
    api_key = os.getenv("COINBASE_API_KEY")
    api_secret = os.getenv("COINBASE_API_SECRET")
    
    if api_key and api_secret:
        # Extract HMAC secret if it's in EC private key format
        if "BEGIN EC PRIVATE KEY" in api_secret:
            print("Detected EC private key format - extracting HMAC secret...")
            
            # Simple extraction - try to get the raw private key
            try:
                import base64
                
                # Fix escaped newlines
                ec_key_str = api_secret.replace('\\n', '\n')
                lines = ec_key_str.split('\n')
                key_lines = []
                
                in_key = False
                for line in lines:
                    line = line.strip()
                    if "BEGIN EC PRIVATE KEY" in line:
                        in_key = True
                        continue
                    elif "END EC PRIVATE KEY" in line:
                        break
                    elif in_key and line:
                        key_lines.append(line)
                
                key_data_b64 = ''.join(key_lines)
                missing_padding = len(key_data_b64) % 4
                if missing_padding:
                    key_data_b64 += '=' * (4 - missing_padding)
                
                key_bytes = base64.b64decode(key_data_b64)
                
                # Try to find 32-byte private key in ASN.1 structure
                search_pos = 0
                while search_pos < len(key_bytes) - 34:
                    if key_bytes[search_pos] == 0x04:  # OCTET STRING tag
                        length = key_bytes[search_pos + 1]
                        if length == 32:
                            private_key = key_bytes[search_pos + 2:search_pos + 34]
                            print(f"✅ Extracted private key: {private_key.hex()[:32]}...")
                            
                            # Test with this private key
                            timestamp = str(int(time.time()))
                            endpoint = "/v2/accounts"
                            message = timestamp + "GET" + endpoint
                            
                            signature = hmac.new(
                                private_key,
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
                                print("✅ SUCCESS with extracted private key!")
                                data = response.json()
                                if "data" in data:
                                    accounts = data["data"]
                                    print(f"   Found {len(accounts)} accounts")
                            else:
                                print(f"❌ Failed with extracted private key: {response.status_code}")
                            
                            break
                    search_pos += 1
            except Exception as e:
                print(f"❌ Error extracting private key: {e}")

    print()
    print("=" * 80)
    print("=== Next Steps ===")
    print("1. Create a new API key in Coinbase dashboard")
    print("2. Test with the provided test script")
    print("3. Update your environment variables")
    print("4. Run your Coinbase bot with the new credentials")
    print()
    print("Save the test script: test_new_api_key.py")

if __name__ == "__main__":
    main()