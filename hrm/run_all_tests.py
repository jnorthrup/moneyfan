"""
Run all hierarchical codec tests
==================================

This script runs both parity and speed tests and displays results.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_parity_test():
    """Run parity tests."""
    print("\n" + "="*70)
    print("RUNNING PARITY TESTS")
    print("="*70)
    
    try:
        from hrm.parity_test_simple import main as parity_main
        result = parity_main()
        return result
    except Exception as e:
        print(f"Parity test failed: {e}")
        return False

def run_speed_test():
    """Run speed tests."""
    print("\n" + "="*70)
    print("RUNNING SPEED TESTS")
    print("="*70)
    
    try:
        from hrm.speed_test import main as speed_main
        result = speed_main()
        return result
    except Exception as e:
        print(f"Speed test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("HIERARCHICAL CODEC TEST SUITE")
    print("="*70)
    
    # Check dependencies
    try:
        import torch
        import mlx.core as mx
        print("✓ PyTorch available")
        print("✓ MLX available")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nPlease install required packages:")
        print("  pip install torch mlx")
        return False
    
    # Run tests
    parity_result = run_parity_test()
    speed_result = run_speed_test()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUITE SUMMARY")
    print("="*70)
    
    print(f"Parity Tests: {'✓ PASSED' if parity_result else '✗ FAILED'}")
    print(f"Speed Tests:  {'✓ PASSED' if speed_result else '✗ FAILED'}")
    
    if parity_result and speed_result:
        print("\n✓ ALL TESTS PASSED")
        return True
    else:
        print("\n✗ SOME TESTS FAILED")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
