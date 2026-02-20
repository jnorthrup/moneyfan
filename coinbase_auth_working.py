#!/usr/bin/env python3
import os
import time
import hmac
import hashlib
import requests

class CoinbaseAuth:
    """Working Coinbase authentication"""
    
    def __init__(self):
        self.api_key = "abc12d6a-f0d5-45ae-affe-e9139f06bb46"
        self.hmac_secret = bytes.fromhex("362e41c6515060ac17eeae5a0a303893c4d662300ed0a9a66f8b3ede98786575")
        self.passphrase = ""
        
    def make_request(self, method: str, endpoint: str, body: str = "") -> dict:
        """Make authenticated request"""
        timestamp = str(int(time.time()))
        
        # Try different signature formats
        signatures = [
            (timestamp + method + endpoint, "v2"),
            (f"{timestamp}{method}{endpoint}", "v3"),
        ]
        
        if self.passphrase:
            signatures.append((f"{timestamp}{method}{endpoint}{self.passphrase}", "v3_pass"))
        
        for message, method_name in signatures:
            signature = hmac.new(
                self.hmac_secret,
                message.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            headers = {
                "CB-ACCESS-KEY": self.api_key,
                "CB-ACCESS-SIGN": signature,
                "CB-ACCESS-TIMESTAMP": timestamp,
                "Content-Type": "application/json"
            }
            
            if method_name == "v3_pass":
                headers["CB-ACCESS-PASSPHRASE"] = self.passphrase
            
            url = f"https://api.coinbase.com{endpoint}"
            
            try:
                if method == "GET":
                    response = requests.get(url, headers=headers)
                else:
                    response = requests.post(url, headers=headers, data=body)
                
                if response.ok:
                    return {"success": True, "data": response.json(), "method": method_name}
                else:
                    continue
            except Exception as e:
                continue
        
        return {"success": False, "error": "All methods failed"}

if __name__ == "__main__":
    auth = CoinbaseAuth()
    
    # Test authentication
    result = auth.make_request("GET", "/v2/accounts")
    if result["success"]:
        print("✅ Authentication working!")
        print(f"   Method: {result['method']}")
        accounts = result["data"].get("data", [])
        print(f"   Accounts: {len(accounts)}")
    else:
        print("❌ Authentication failed")
