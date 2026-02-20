#!/usr/bin/env python3
"""
EC private key decoder v2 - handles escaped newlines
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
    print("=== EC Private Key Analysis v2 ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    api_secret = os.getenv("COINBASE_API_SECRET")
    if not api_secret:
        print("❌ COINBASE_API_SECRET not set")
        return

    print(f"Raw API Secret length: {len(api_secret)} chars")
    print(f"First 200 chars: {api_secret[:200]}...")
    print()

    # Fix escaped newlines
    if '\\n' in api_secret:
        print("Found escaped newlines (\\n), converting to actual newlines...")
        api_secret = api_secret.replace('\\n', '\n')
        print(f"Converted length: {len(api_secret)} chars")
        print()

    # Extract the base64-encoded key data
    def extract_key_data(ec_key_str):
        """Extract base64 data from EC private key string"""
        # Remove PEM headers and get the base64 data
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
            elif in_key and line:  # Only add non-empty lines
                key_lines.append(line)
        
        if not key_lines:
            print("❌ No key data found in PEM")
            print(f"Lines found: {len(lines)}")
            print(f"First few lines: {lines[:5]}")
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
            
            # Save the private key for later use
            print(f"\n  Potential HMAC secrets to test:")
            print(f"    1. Extracted private key ({len(private_key_bytes)} bytes): {private_key_bytes.hex()[:32]}...")
            print(f"    2. Full key ({len(key_bytes)} bytes): {key_bytes.hex()[:32]}...")
        
    except Exception as e:
        print(f"❌ Failed to parse as EC private key: {e}")
        print(f"   Error details: {type(e).__name__}")
        
        # Try alternative approaches
        print("\nTrying alternative parsing approaches...")
        
        # Approach 1: Check if it's already a raw private key
        print("\n1. Checking if it's a raw private key...")
        if len(key_bytes) == 32:
            print(f"   ✅ Found 32-byte raw private key!")
            print(f"   Hex: {key_bytes.hex()}")
            
            # Test HMAC
            test_hmac = hmac.new(key_bytes, "test".encode(), hashlib.sha256).hexdigest()
            print(f"   HMAC test: {test_hmac}")
        else:
            print(f"   ❌ Not 32 bytes (found {len(key_bytes)} bytes)")
        
        # Approach 2: Try to parse ASN.1 structure manually
        print("\n2. Attempting manual ASN.1 parsing...")
        try:
            # Parse ASN.1 DER structure manually
            # EC private key format: SEQUENCE { INTEGER version, OCTET STRING privateKey, [0] ECParameters }
            
            if key_bytes[0] != 0x30:  # SEQUENCE tag
                print("   ❌ Doesn't start with ASN.1 SEQUENCE (0x30)")
            else:
                print("   ✅ ASN.1 SEQUENCE detected")
                
                # Try to find the OCTET STRING containing the private key
                # Look for 0x04 (OCTET STRING tag) followed by 32 bytes
                search_pos = 0
                while search_pos < len(key_bytes) - 32:
                    if key_bytes[search_pos] == 0x04:  # OCTET STRING tag
                        # Check if next byte is length
                        length = key_bytes[search_pos + 1]
                        if length == 32:  # EC private key length
                            candidate = key_bytes[search_pos + 2:search_pos + 34]
                            print(f"   ✅ Found candidate 32-byte private key at position {search_pos}")
                            print(f"   Hex: {candidate.hex()}")
                            
                            # Test HMAC
                            test_hmac = hmac.new(candidate, "test".encode(), hashlib.sha256).hexdigest()
                            print(f"   HMAC test: {test_hmac}")
                            
                            # Save this candidate
                            if 'private_key_bytes' not in locals():
                                private_key_bytes = candidate
                    search_pos += 1
                
                if 'private_key_bytes' not in locals():
                    print("   ❌ Could not find 32-byte private key in ASN.1 structure")
        
        except Exception as e:
            print(f"   ❌ Manual ASN.1 parsing failed: {e}")

    # Test various HMAC secret candidates
    print("\n3. Testing various HMAC secret candidates...")
    
    test_message = "test_message"
    candidates = []
    
    # Candidate 1: If we extracted a private key
    if 'private_key_bytes' in locals() and len(private_key_bytes) == 32:
        candidates.append(("Extracted private key", private_key_bytes))
    
    # Candidate 2: Full key
    candidates.append(("Full key", key_bytes))
    
    # Candidate 3: Last 32 bytes
    if len(key_bytes) >= 32:
        candidates.append(("Last 32 bytes", key_bytes[-32:]))
    
    # Candidate 4: First 32 bytes
    if len(key_bytes) >= 32:
        candidates.append(("First 32 bytes", key_bytes[:32]))
    
    # Candidate 5: Middle 32 bytes
    if len(key_bytes) >= 64:
        mid = len(key_bytes) // 2 - 16
        candidates.append(("Middle 32 bytes", key_bytes[mid:mid+32]))
    
    for name, secret in candidates:
        test_hmac = hmac.new(secret, test_message.encode(), hashlib.sha256).hexdigest()
        print(f"   {name}: {test_hmac[:32]}...")
        # Test a more realistic message format
        test_hmac2 = hmac.new(secret, b"GET/accounts", hashlib.sha256).hexdigest()
        print(f"     (GET/accounts): {test_hmac2[:32]}...")
    
    print("\n=== Analysis Complete ===")
    print("\nPotential HMAC secrets found:")
    for i, (name, secret) in enumerate(candidates, 1):
        print(f"  {i}. {name}: {secret.hex()[:32]}...")

if __name__ == "__main__":
    main()