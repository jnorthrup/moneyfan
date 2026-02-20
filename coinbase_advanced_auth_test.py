#!/usr/bin/env python3
"""
Test different authentication methods for Coinbase Advanced Trade
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
    print("=== Coinbase Advanced Trade Authentication Test ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Get credentials
    api_key = os.getenv("COINBASE_API_KEY")
    api_secret_encoded = os.getenv("COINBASE_API_SECRET")
    passphrase = os.getenv("COINBASE_PASSPHRASE", "")

    print("Environment:")
    print(f"  API Key: {api_key[:20]}..." if api_key else "  API Key: ❌ Not set")
    print(f"  Passphrase: {'✅ Set' if passphrase else '⚠️  Not set'}")
    print()

    # Extract HMAC secret from EC private key
    def extract_hmac_secret(ec_key_str):
        """Extract HMAC secret from EC private key"""
        ec_key_str = ec_key_str.replace('\\n', '\n')
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
        
        # Find OCTET STRING tag (0x04) and extract 32 bytes
        search_pos = 0
        while search_pos < len(key_bytes) - 34:
            if key_bytes[search_pos] == 0x04:
                length = key_bytes[search_pos + 1]
                if length == 32:
                    return key_bytes[search_pos + 2:search_pos + 34]
            search_pos += 1
        
        return key_bytes

    hmac_secret = extract_hmac_secret(api_secret_encoded)
    print(f"✅ Extracted HMAC secret: {hmac_secret.hex()}")
    print()

    # Test different signature methods
    def test_signature_method(method_name, timestamp, method, endpoint, body, passphrase):
        """Test a specific signature method"""
        print(f"\n  Testing: {method_name}")
        
        if method_name == "v2-standard":
            # Coinbase v2 standard
            message = timestamp + method + endpoint + body
        elif method_name == "v2-with-body":
            # With body hash
            if body:
                body_hash = hashlib.sha256(body.encode()).hexdigest()
                message = timestamp + method + endpoint + body_hash
            else:
                message = timestamp + method + endpoint
        elif method_name == "v3-standard":
            # Advanced Trade v3 standard
            message = f"{timestamp}{method}{endpoint}{body}"
        elif method_name == "v3-with-passphrase":
            # v3 with passphrase
            message = f"{timestamp}{method}{endpoint}{body}{passphrase}"
        elif method_name == "v3-body-hash":
            # v3 with body hash
            if body:
                body_hash = hashlib.sha256(body.encode()).hexdigest()
                message = f"{timestamp}{method}{endpoint}{body_hash}"
            else:
                message = f"{timestamp}{method}{endpoint}"
        else:
            print(f"    ❌ Unknown method: {method_name}")
            return None
        
        signature = hmac.new(hmac_secret, message.encode(), hashlib.sha256).hexdigest()
        print(f"    Message: {message[:50]}...")
        print(f"    Signature: {signature[:32]}...")
        
        return signature

    # Test with different endpoints
    endpoints_to_test = [
        ("/api/v3/brokerage/accounts", "GET", ""),
        ("/v2/accounts", "GET", ""),
        ("/api/v3/brokerage/products", "GET", ""),
    ]

    signature_methods = [
        "v2-standard",
        "v2-with-body",
        "v3-standard",
        "v3-with-passphrase",
        "v3-body-hash",
    ]

    print("Testing Signature Methods...")
    print("=" * 80)

    for endpoint, method, body in endpoints_to_test:
        print(f"\nEndpoint: {endpoint}")
        timestamp = str(int(time.time()))
        
        for sig_method in signature_methods:
            signature = test_signature_method(sig_method, timestamp, method, endpoint, body, passphrase)
            
            if signature:
                # Try the request with this signature
                base_url = "https://api.coinbase.com" if endpoint.startswith("/v2") or endpoint.startswith("/api") else "https://api.pro.coinbase.com"
                url = f"{base_url}{endpoint}"
                
                headers = {
                    "CB-ACCESS-KEY": api_key,
                    "CB-ACCESS-SIGN": signature,
                    "CB-ACCESS-TIMESTAMP": timestamp,
                    "Content-Type": "application/json"
                }
                
                if passphrase:
                    headers["CB-ACCESS-PASSPHRASE"] = passphrase
                
                try:
                    response = requests.get(url, headers=headers)
                    if response.ok:
                        print(f"    ✅ SUCCESS with {sig_method}!")
                        data = response.json()
                        if isinstance(data, dict):
                            print(f"      Response keys: {list(data.keys())[:3]}")
                        break  # Stop testing other methods if one works
                    else:
                        print(f"    ❌ Failed with {sig_method}: {response.status_code}")
                except Exception as e:
                    print(f"    ❌ Error with {sig_method}: {e}")

    print("\n" + "=" * 80)
    print("=== Testing with Different Message Formats ===")

    # Test different message formats for the same endpoint
    endpoint = "/api/v3/brokerage/accounts"
    timestamp = str(int(time.time()))
    method = "GET"

    message_formats = [
        ("timestamp + method + endpoint + body", f"{timestamp}{method}{endpoint}"),
        ("endpoint + method + timestamp + body", f"{endpoint}{method}{timestamp}"),
        ("method + timestamp + endpoint + body", f"{method}{timestamp}{endpoint}"),
        ("with newlines", f"{timestamp}\n{method}\n{endpoint}\n"),
        ("with slashes", f"{timestamp}/{method}/{endpoint}/"),
    ]

    for desc, message in message_formats:
        signature = hmac.new(hmac_secret, message.encode(), hashlib.sha256).hexdigest()
        
        url = "https://api.coinbase.com" + endpoint
        headers = {
            "CB-ACCESS-KEY": api_key,
            "CB-ACCESS-SIGN": signature,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json"
        }
        
        if passphrase:
            headers["CB-ACCESS-PASSPHRASE"] = passphrase
        
        try:
            response = requests.get(url, headers=headers)
            status = "✅ SUCCESS" if response.ok else f"❌ FAILED ({response.status_code})"
            print(f"  {desc}: {status}")
            if response.ok:
                print(f"    Found working format!")
                break
        except Exception as e:
            print(f"  {desc}: ❌ ERROR - {e}")

    print("\n" + "=" * 80)
    print("=== Testing Raw API Key/Secret ===")

    # Try with raw API key/secret (without EC private key extraction)
    print("\nTrying with raw API key/secret (if different)...")
    
    # Check if the raw secret might already be an HMAC secret
    if "BEGIN EC PRIVATE KEY" not in api_secret_encoded:
        print("Raw secret doesn't contain EC private key marker - testing directly...")
        
        raw_secret = api_secret_encoded.encode('utf-8')
        test_hmac = hmac.new(raw_secret, "test".encode(), hashlib.sha256).hexdigest()
        print(f"  HMAC with raw secret: {test_hmac[:32]}...")
        
        # Test with raw secret
        timestamp = str(int(time.time()))
        message = f"{timestamp}GET/api/v3/brokerage/accounts"
        signature = hmac.new(raw_secret, message.encode(), hashlib.sha256).hexdigest()
        
        url = "https://api.coinbase.com/api/v3/brokerage/accounts"
        headers = {
            "CB-ACCESS-KEY": api_key,
            "CB-ACCESS-SIGN": signature,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "CB-ACCESS-PASSPHRASE": passphrase,
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(url, headers=headers)
            if response.ok:
                print("  ✅ SUCCESS with raw secret!")
            else:
                print(f"  ❌ Failed with raw secret: {response.status_code}")
        except Exception as e:
            print(f"  ❌ Error with raw secret: {e}")

    print("\n" + "=" * 80)
    print("=== Test Complete ===")
    print("\nNext steps to try:")
    print("1. Check if API key has proper permissions in Coinbase dashboard")
    print("2. Try creating a new API key with 'view' permissions only")
    print("3. Check if there are IP restrictions on the API key")
    print("4. Contact Coinbase support if authentication continues to fail")

if __name__ == "__main__":
    main()