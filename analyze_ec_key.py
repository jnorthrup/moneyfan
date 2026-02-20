#!/usr/bin/env python3
"""
Analyze and attempt to decode the EC private key to understand its format and usage
"""

import os
import base64
import hashlib
import hmac
import struct
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend
from datetime import datetime

def main():
    print("=== EC Private Key Analysis ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    api_secret = os.getenv("COINBASE_API_SECRET")
    if not api_secret:
        print("❌ COINBASE_API_SECRET not set")
        return

    print(f"Raw API Secret (first 50 chars): {api_secret[:50]}...")
    print()

    # Extract the base64-encoded key data
    def extract_key_data(ec_key_str):
        """Extract base64 data from EC private key string"""
        # Remove PEM headers
        start_marker = "-----BEGIN EC PRIVATE KEY-----"
        end_marker = "-----END EC PRIVATE KEY-----"
        
        if start_marker not in ec_key_str or end_marker not in ec_key_str:
            print("❌ Not a valid EC private key PEM format")
            return None
        
        start_idx = ec_key_str.find(start_marker) + len(start_marker)
        end_idx = ec_key_str.find(end_marker)
        
        key_data = ec_key_str[start_idx:end_idx].strip()
        return key_data

    key_data_b64 = extract_key_data(api_secret)
    if not key_data_b64:
        return

    print(f"Extracted base64 data length: {len(key_data_b64)} chars")
    print(f"Base64 data (first 100 chars): {key_data_b64[:100]}...")
    print()

    # Decode base64
    try:
        # Add padding if needed
        missing_padding = len(key_data_b64) % 4
        if missing_padding:
            key_data_b64 += '=' * (4 - missing_padding)
        
        key_bytes = base64.b64decode(key_data_b64)
        print(f"✅ Decoded key length: {len(key_bytes)} bytes")
        print(f"Key bytes (hex, first 50 chars): {key_bytes[:25].hex()}")
        print()
    except Exception as e:
        print(f"❌ Failed to decode base64: {e}")
        return

    # Try to parse as ASN.1 DER encoded private key
    print("Attempting to parse as ASN.1 DER encoded EC private key...")
    try:
        # Try cryptography library
        from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurvePrivateKey
        from cryptography.hazmat.primitives.asymmetric import ec
        
        key = serialization.load_der_private_key(
            key_bytes,
            password=None,
            backend=default_backend()
        )
        
        print(f"✅ Successfully parsed as EC private key!")
        print(f"  Key type: {type(key)}")
        print(f"  Curve: {key.curve.name if hasattr(key, 'curve') else 'Unknown'}")
        
        # Extract public key
        public_key = key.public_key()
        print(f"  Public key available: Yes")
        
        # Get the private key bytes (for HMAC)
        private_key_bytes = key.private_numbers().private_value.to_bytes(32, byteorder='big')
        print(f"  Private key bytes (hex): {private_key_bytes.hex()}")
        print(f"  Private key bytes (first 8 bytes): {private_key_bytes[:8].hex()}")
        
        # Test if this could be used as HMAC secret
        test_message = "test_message"
        test_hmac = hmac.new(private_key_bytes, test_message.encode(), hashlib.sha256).hexdigest()
        print(f"  HMAC test (SHA256): {test_hmac}")
        
    except Exception as e:
        print(f"❌ Failed to parse as EC private key: {e}")
        print()
        
        # Try alternative approaches
        print("Trying alternative parsing approaches...")
        
        # Approach 1: Try to extract raw key bytes
        print("\n1. Checking if it's a raw private key...")
        if len(key_bytes) == 32:
            print(f"   ✅ Found 32-byte raw private key!")
            print(f"   Hex: {key_bytes.hex()}")
            
            # Test HMAC
            test_hmac = hmac.new(key_bytes, "test".encode(), hashlib.sha256).hexdigest()
            print(f"   HMAC test: {test_hmac}")
        else:
            print(f"   ❌ Not 32 bytes (found {len(key_bytes)} bytes)")
        
        # Approach 2: Try to extract from ASN.1 structure
        print("\n2. Checking for ASN.1 structure...")
        # ASN.1 EC private key typically starts with 0x30 (SEQUENCE)
        if key_bytes.startswith(b'\x30'):
            print(f"   ✅ Found ASN.1 structure (starts with 0x30)")
            
            # Try to parse ASN.1 structure
            try:
                # Skip the initial ASN.1 SEQUENCE tag and length
                pos = 1
                seq_len = key_bytes[pos]
                if seq_len & 0x80:  # Long form
                    len_bytes = seq_len & 0x7F
                    pos += 1 + len_bytes
                else:
                    pos += 1
                
                # Parse version (INTEGER)
                if key_bytes[pos] == 0x02:  # INTEGER tag
                    pos += 1
                    version_len = key_bytes[pos]
                    pos += 1 + version_len
                
                # Parse private key (OCTET STRING)
                if key_bytes[pos] == 0x04:  # OCTET STRING tag
                    pos += 1
                    key_len = key_bytes[pos]
                    pos += 1
                    
                    if key_len == 32:  # EC private key is typically 32 bytes
                        private_key = key_bytes[pos:pos+32]
                        print(f"   ✅ Extracted 32-byte private key!")
                        print(f"   Hex: {private_key.hex()}")
                        
                        # Test HMAC
                        test_hmac = hmac.new(private_key, "test".encode(), hashlib.sha256).hexdigest()
                        print(f"   HMAC test: {test_hmac}")
            except Exception as e:
                print(f"   ❌ Failed to parse ASN.1: {e}")
    
    # Alternative: Try to use the entire key bytes as HMAC secret
    print("\n3. Testing entire key bytes as HMAC secret...")
    test_hmac = hmac.new(key_bytes, "test".encode(), hashlib.sha256).hexdigest()
    print(f"   HMAC with full key ({len(key_bytes)} bytes): {test_hmac}")
    
    # Try different message formats
    print("\n4. Testing different message formats for HMAC...")
    test_messages = [
        "test",
        "message",
        "test_message",
        "coinbase",
        "api",
    ]
    
    for msg in test_messages:
        test_hmac = hmac.new(key_bytes, msg.encode(), hashlib.sha256).hexdigest()
        print(f"   HMAC('{msg}'): {test_hmac[:32]}...")
    
    print("\n=== Analysis Complete ===")
    print("\nNext steps:")
    print("1. If the key parsed successfully, use the extracted private key as HMAC secret")
    print("2. If not, try using the entire decoded key bytes as HMAC secret")
    print("3. Test with Coinbase API authentication")
    print("4. Consider creating a new API key if this one doesn't work")

if __name__ == "__main__":
    main()