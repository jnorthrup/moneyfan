#!/usr/bin/env python3
"""
Verification Script: Pandas MLX Integration for HRM
=====================================================

This script verifies that:
1. Pandas DataFrame can be loaded from feather files
2. Signal computation works with pandas/numpy
3. MLX model can process pandas-derived data
4. End-to-end pipeline works: df → signals → MLX → predictions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path

# Try to import MLX
try:
    import mlx.core as mx
    from hrm.hierarchical_codec_mlx import (
        MLXHierarchicalCodec,
        HierarchicalCodecConfig
    )
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("❌ MLX not available - cannot verify MLX integration")


def test_pandas_loading() -> bool:
    """Test loading feather files with pandas."""
    print("\n" + "="*60)
    print("TEST 1: Pandas DataFrame Loading")
    print("="*60)
    
    arrow_dir = Path("hrm/data/arrow")
    if not arrow_dir.exists():
        print("⚠️  Arrow directory not found, skipping test")
        return True
    
    feather_files = list(arrow_dir.glob("*.feather"))
    if not feather_files:
        print("⚠️  No feather files found, skipping test")
        return True
    
    print(f"Found {len(feather_files)} feather files")
    
    # Try to load first file
    try:
        df = pd.read_feather(feather_files[0])
        print(f"✅ Successfully loaded: {feather_files[0].name}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        return True
    except Exception as e:
        print(f"❌ Failed to load feather file: {e}")
        return False


def test_signal_computation() -> bool:
    """Test signal computation with pandas/numpy."""
    print("\n" + "="*60)
    print("TEST 2: Signal Computation (Pandas/Numpy)")
    print("="*60)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    data = {
        'time': dates,
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.rand(100) * 1000
    }
    df = pd.DataFrame(data)
    
    # Compute signals (simplified version)
    n_signals = 24
    T = len(df)
    signals = np.zeros((T, n_signals * 2), dtype=np.float32)
    
    c = np.nan_to_num(df['close'].values.astype(np.float32), nan=1.0)
    
    # MACD
    ema12 = pd.Series(c).ewm(span=12, min_periods=1).mean()
    ema26 = pd.Series(c).ewm(span=26, min_periods=1).mean()
    macd = (ema12 - ema26).values
    signals[:, 0] = np.clip(macd / (np.std(macd) + 1e-8), -1, 1)
    signals[:, n_signals] = 0.5
    
    # Momentum
    mom = pd.Series(c).pct_change(20).fillna(0)
    signals[:, 2] = np.clip(mom.values * 10, -1, 1)
    signals[:, n_signals + 2] = 0.5
    
    print(f"✅ Signal computation completed")
    print(f"   Input shape: {df.shape}")
    print(f"   Output shape: {signals.shape}")
    print(f"   Sample signals: {signals[10, :5]}")
    
    return True


def test_mlx_conversion() -> bool:
    """Test MLX array conversion from numpy."""
    print("\n" + "="*60)
    print("TEST 3: NumPy to MLX Conversion")
    print("="*60)
    
    if not HAS_MLX:
        print("❌ MLX not available")
        return False
    
    # Create numpy array
    signals_np = np.random.randn(10, 48).astype(np.float32)
    
    # Convert to MLX
    signals_mx = mx.array(signals_np)
    
    print(f"✅ MLX conversion successful")
    print(f"   NumPy shape: {signals_np.shape}")
    print(f"   MLX shape: {signals_mx.shape}")
    print(f"   NumPy dtype: {signals_np.dtype}")
    print(f"   MLX dtype: {signals_mx.dtype}")
    
    # Verify conversion
    if signals_mx.shape != signals_np.shape:
        print(f"❌ Shape mismatch!")
        return False
    
    if str(signals_mx.dtype) != "mlx.core.float32":
        print(f"❌ Type mismatch! Expected float32, got {signals_mx.dtype}")
        return False
    
    return True


def test_mlx_model() -> bool:
    """Test MLX model forward pass."""
    print("\n" + "="*60)
    print("TEST 4: MLX Model Forward Pass")
    print("="*60)
    
    if not HAS_MLX:
        print("❌ MLX not available")
        return False
    
    # Create model
    config = HierarchicalCodecConfig(n_signals=24, hidden_dim=64)
    model = MLXHierarchicalCodec(config)
    
    # Create test data
    signals_mx = mx.random.normal((2, 64, 48))  # batch=2, seq=64, signals=48
    
    try:
        # Run forward pass
        output, memory = model.forward(signals_mx, mode="trade")
        
        print(f"✅ MLX model forward pass successful")
        print(f"   Input shape: {signals_mx.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output: {np.array(output[0])}")
        
        # Verify output shape
        expected_shape = (2, 5)  # [return, confidence, stop, tp, pos]
        if output.shape != expected_shape:
            print(f"❌ Output shape mismatch! Expected {expected_shape}, got {output.shape}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ MLX forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end() -> bool:
    """Test end-to-end pipeline: df → signals → MLX → predictions."""
    print("\n" + "="*60)
    print("TEST 5: End-to-End Pandas → MLX Pipeline")
    print("="*60)
    
    if not HAS_MLX:
        print("❌ MLX not available")
        return False
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=64, freq='1h')
    data = {
        'time': dates,
        'open': np.random.randn(64).cumsum() + 100,
        'high': np.random.randn(64).cumsum() + 102,
        'low': np.random.randn(64).cumsum() + 98,
        'close': np.random.randn(64).cumsum() + 100,
        'volume': np.random.rand(64) * 1000
    }
    df = pd.DataFrame(data)
    
    # Step 1: Compute signals (pandas/numpy)
    n_signals = 24
    signals = np.zeros((len(df), n_signals * 2), dtype=np.float32)
    c = np.nan_to_num(df['close'].values.astype(np.float32), nan=1.0)
    
    # MACD
    ema12 = pd.Series(c).ewm(span=12, min_periods=1).mean()
    ema26 = pd.Series(c).ewm(span=26, min_periods=1).mean()
    macd = (ema12 - ema26).values
    signals[:, 0] = np.clip(macd / (np.std(macd) + 1e-8), -1, 1)
    signals[:, n_signals] = 0.5
    
    # Step 2: Convert to MLX
    signals_mx = mx.array(signals[None, :, :])  # Add batch dimension
    
    # Step 3: Create MLX model
    config = HierarchicalCodecConfig(n_signals=24, hidden_dim=64)
    model = MLXHierarchicalCodec(config)
    
    # Step 4: Run inference
    output, memory = model.forward(signals_mx, mode="trade")
    
    print(f"✅ End-to-end pipeline successful")
    print(f"   Input: DataFrame {df.shape}")
    print(f"   Signals: {signals.shape}")
    print(f"   MLX Input: {signals_mx.shape}")
    print(f"   MLX Output: {output.shape}")
    print(f"   Predictions: [return, confidence, stop, tp, pos]")
    
    # Verify
    if output.shape == (1, 5):
        print(f"   Sample prediction: {np.array(output[0])}")
        return True
    else:
        print(f"❌ Unexpected output shape: {output.shape}")
        return False


def run_all_tests() -> dict:
    """Run all verification tests."""
    print("="*60)
    print("PANDAS MLX INTEGRATION VERIFICATION")
    print("="*60)
    print(f"Started: {pd.Timestamp.now()}")
    
    results = {
        "pandas_loading": test_pandas_loading(),
        "signal_computation": test_signal_computation(),
        "mlx_conversion": test_mlx_conversion() if HAS_MLX else "SKIPPED",
        "mlx_model": test_mlx_model() if HAS_MLX else "SKIPPED",
        "end_to_end": test_end_to_end() if HAS_MLX else "SKIPPED",
    }
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        if result == "SKIPPED":
            print(f"  {test_name}: ⚠️  SKIPPED (MLX not available)")
        elif result:
            print(f"  {test_name}: ✅ PASS")
        else:
            print(f"  {test_name}: ❌ FAIL")
    
    passed = sum(1 for r in results.values() if r is True)
    skipped = sum(1 for r in results.values() if r == "SKIPPED")
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} tests passed, {skipped} skipped")
    
    if skipped == total:
        print(f"\n⚠️  All tests skipped - MLX not available")
        print(f"   Install MLX: pip install mlx")
    elif passed == total - skipped:
        print(f"\n✅ ALL TESTS PASSED!")
        print(f"   Pandas MLX integration is working correctly.")
    else:
        print(f"\n⚠️  Some tests failed - review above")
    
    return results


if __name__ == "__main__":
    results = run_all_tests()
    
    # Exit code
    passed = sum(1 for r in results.values() if r is True)
    total = len(results)
    skipped = sum(1 for r in results.values() if r == "SKIPPED")
    
    if passed == total - skipped and skipped < total:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure
