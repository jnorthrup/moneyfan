#!/usr/bin/env python3
"""
Fixed EC private key decoder
"""

import os
import base64
import hashlib
import hmac
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend
from datetime import datetime

def main():
    print("=== EC Private Key Analysis (Fixed) ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    api_secret = os.getenv("COINBASE_API_SECRET")
    if not api_secret:
        print("❌ COINBASE_API_SECRET not set")
        return

    print(f"Raw API Secret (first 100 chars): {api_secret[:100]}...")
    print()

    # Extract the base64-encoded key data
    def extract_key_data(ec_key_str):
        """Extract base64 data from EC private key string"""
        # Remove PEM headers and get the base64 data
        lines = ec_key_str.split('\n')
        key_lines = []
        
        in_key = False
        for line in lines:
            if "BEGIN EC PRIVATE KEY" in line:
                in_key = True
                continue
            elif "END EC PRIVATE KEY" in line:
                break
            elif in_key:
                key_lines.append(line.strip())
        
        if not key_lines:
            print("❌ No key data found in PEM")
            return None
        
        key_data_b64 = ''.join(key_lines)
        return key_data_b64

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
        key = serialization.load_der_private_key(
            key_bytes,
            password=None,
            backend=default_backend()
        )
        
        print(f"✅ Successfully parsed as EC private key!")
        print(f"  Key type: {type(key)}")
        print(f"  Curve: {key.curve.name if hasattr(key, 'curve') else 'Unknown'}")
        
        # Extract private key value
        if hasattr(key, 'private_numbers'):
            private_numbers = key.private_numbers()
            private_value = private_numbers.private_value
            
            # Convert to bytes
            private_key_bytes = private_value.to_bytes(32, byteorder='big')
            print(f"  Private key bytes (hex): {private_key_bytes.hex()}")
            print(f"  Private key bytes (first 8 bytes): {private_key_bytes[:8].hex()}")
            
            # Test HMAC
            test_message = "test_message"
            test_hmac = hmac.new(private_key_bytes, test_message.encode(), hashlib.sha256).hexdigest()
            print(f"  HMAC test (SHA256): {test_hmac}")
            
            # Also test with the full key
            test_hmac_full = hmac.new(key_bytes, test_message.encode(), hashlib.sha256).hexdigest()
            print(f"  HMAC with full key: {test_hmac_full}")
        
    except Exception as e:
        print(f"❌ Failed to parse as EC private key: {e}")
        print(f"   Error details: {type(e).__name__}")
        
        # Try alternative approaches
        print("\nTrying alternative parsing approaches...")
        
        # Approach 1: Check for ASN.1 structure
        print("\n1. Checking ASN.1 structure...")
        print(f"   First 20 bytes (hex): {key_bytes[:20].hex()}")
        print(f"   First 20 bytes (ASCII): {key_bytes[:20]}")
        
        # Common ASN.1 structures for EC private keys
        # SEQUENCE {
        #   INTEGER version
        #   OCTET STRING privateKey
        #   [0] ECParameters (optional)
        # }
        
        if key_bytes and len(key_bytes) > 2:
            # Check if it starts with ASN.1 SEQUENCE (0x30)
            if key_bytes[0] == 0x30:
                print("   ✅ ASN.1 SEQUENCE detected")
                
                # Try to parse the structure manually
                try:
                    from cryptography.hazmat.asn1 import encode, decode
                    from cryptography.hazmat.asn1 import der
                    
                    # Try to decode the DER structure
                    sequence, remaining = der.decode(key_bytes)
                    print(f"   ✅ ASN.1 decoding successful!")
                    print(f"   Sequence type: {type(sequence)}")
                    print(f"   Sequence length: {len(sequence) if hasattr(sequence, '__len__') else 'N/A'}")
                    
                    # Check if it looks like EC private key
                    if hasattr(sequence, '__iter__'):
                        items = list(sequence)
                        print(f"   Sequence items: {len(items)}")
                        for i, item in enumerate(items[:5]):  # Show first 5 items
                            print(f"     Item {i}: {type(item)} - {str(item)[:50]}...")
                
                except Exception as e2:
                    print(f"   ❌ ASN.1 parsing failed: {e2}")
        
        # Approach 2: Try to extract 32-byte key from different positions
        print("\n2. Trying to extract 32-byte private key...")
        for start in range(0, max(0, len(key_bytes) - 32)):
            candidate = key_bytes[start:start+32]
            
            # Check if it looks like a valid private key (not all zeros)
            if candidate != b'\x00' * 32:
                test_hmac = hmac.new(candidate, "test".encode(), hashlib.sha256).hexdigest()
                print(f"   Position {start}: {candidate[:8].hex()}... (HMAC: {test_hmac[:16]}...)")

    # Test using the entire key as HMAC secret
    print("\n3. Testing entire key bytes as HMAC secret...")
    test_messages = ["test", "message", "coinbase", "api", "signature"]
    
    for msg in test_messages:
        test_hmac = hmac.new(key_bytes, msg.encode(), hashlib.sha256).hexdigest()
        print(f"   HMAC('{msg}'): {test_hmac[:32]}...")
    
    # Also test the last 32 bytes
    if len(key_bytes) >= 32:
        last_32 = key_bytes[-32:]
        print(f"\n4. Testing last 32 bytes as HMAC secret...")
        for msg in test_messages:
            test_hmac = hmac.new(last_32, msg.encode(), hashlib.sha256).hexdigest()
            print(f"   HMAC('{msg}'): {test_hmac[:32]}...")

if __name__ == "__main__":
    main()