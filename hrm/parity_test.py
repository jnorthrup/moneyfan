"""
Parity Test: Verify PyTorch and MLX implementations produce matching outputs
===========================================================================

This script tests that both implementations are numerically equivalent within tolerance.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    from hrm.hierarchical_codec import HierarchicalCodec, HierarchicalCodecConfig as TorchConfig
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available")

try:
    import mlx.core as mx
    from hrm.hierarchical_codec_mlx import MLXHierarchicalCodec, HierarchicalCodecConfig as MLXConfig
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("MLX not available")


def set_same_weights(torch_model, mlx_model):
    """Copy weights from MLX model to PyTorch model for fair comparison."""
    # For parity testing, we need to ensure both models use the same weights
    # Since MLX and PyTorch have different attention implementations,
    # we'll focus on copying the weights we can match and test parity
    # for the parts we can control
    
    with torch.no_grad():
        # Copy input projection
        torch_model.input_proj.weight.copy_(torch.tensor(mlx_model.input_proj.weight, dtype=torch.float32))
        
        # Copy H/L init states
        torch_model.H_init.copy_(torch.tensor(mlx_model.H_init, dtype=torch.float32))
        torch_model.L_init.copy_(torch.tensor(mlx_model.L_init, dtype=torch.float32))
        
        # Copy heads
        torch_model.signal_head.weight.copy_(torch.tensor(mlx_model.signal_head.weight, dtype=torch.float32))
        torch_model.return_head.weight.copy_(torch.tensor(mlx_model.return_head.weight, dtype=torch.float32))
        torch_model.confidence_head.weight.copy_(torch.tensor(mlx_model.confidence_head.weight, dtype=torch.float32))
        torch_model.stop_head.weight.copy_(torch.tensor(mlx_model.stop_head.weight, dtype=torch.float32))
        torch_model.tp_head.weight.copy_(torch.tensor(mlx_model.tp_head.weight, dtype=torch.float32))
        torch_model.pos_head.weight.copy_(torch.tensor(mlx_model.pos_head.weight, dtype=torch.float32))
        
        # Copy MLP weights (these are simple linear layers)
        for i, (mlx_block, torch_block) in enumerate(zip(mlx_model.H_level.layers, torch_model.H_level.layers)):
            # MLP weights - both use Sequential with Linear layers
            torch_block.mlp[0].weight.copy_(torch.tensor(mlx_block.mlp.layers[0].weight, dtype=torch.float32))
            torch_block.mlp[2].weight.copy_(torch.tensor(mlx_block.mlp.layers[2].weight, dtype=torch.float32))
        
        for i, (mlx_block, torch_block) in enumerate(zip(mlx_model.L_level.layers, torch_model.L_level.layers)):
            # MLP weights
            torch_block.mlp[0].weight.copy_(torch.tensor(mlx_block.mlp.layers[0].weight, dtype=torch.float32))
            torch_block.mlp[2].weight.copy_(torch.tensor(mlx_block.mlp.layers[2].weight, dtype=torch.float32))


def test_pretrain_parity():
    """Test pretrain mode parity."""
    if not (HAS_TORCH and HAS_MLX):
        print("Skipping pretrain parity test - need both PyTorch and MLX")
        return False
    
    print("\n" + "="*60)
    print("TEST 1: Pretrain Mode Parity")
    print("="*60)
    
    # Same config
    config = MLXConfig()
    torch_config = TorchConfig(
        n_signals=config.n_signals,
        hidden_dim=config.hidden_dim,
        sparkline_frames=config.sparkline_frames,
        sparkline_horizon=config.sparkline_horizon,
        H_layers=config.H_layers,
        L_layers=config.L_layers,
        H_cycles=config.H_cycles,
        L_cycles=config.L_cycles,
        n_heads=config.n_heads,
        dropout=config.dropout
    )
    
    # Create models
    torch_model = HierarchicalCodec(torch_config)
    mlx_model = MLXHierarchicalCodec(config)
    
    # Set same weights
    set_same_weights(torch_model, mlx_model)
    
    # Test data
    np.random.seed(42)
    B, T = 4, 32
    signals_np = np.random.randn(B, T, config.n_signals * 2).astype(np.float32)
    signals_torch = torch.tensor(signals_np)
    signals_mlx = mx.array(signals_np)
    
    # Forward pass
    with torch.no_grad():
        torch_output, _ = torch_model(signals_torch, mode="pretrain")
    
    mlx_output, _ = mlx_model(signals_mlx, mode="pretrain")
    
    # Convert and compare
    torch_output_np = torch_output.numpy()
    mlx_output_np = np.array(mlx_output)
    
    diff = np.abs(torch_output_np - mlx_output_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    median_diff = np.median(diff)
    
    print(f"Batch size: {B}, Sequence length: {T}")
    print(f"Output shape: {torch_output_np.shape}")
    print(f"Max difference:    {max_diff:.10e}")
    print(f"Mean difference:   {mean_diff:.10e}")
    print(f"Median difference: {median_diff:.10e}")
    
    # Acceptance criteria: within 1e-5
    TOLERANCE = 1e-5
    passed = max_diff < TOLERANCE
    
    if passed:
        print(f"✓ PASSED: Max difference {max_diff:.2e} < tolerance {TOLERANCE}")
    else:
        print(f"✗ FAILED: Max difference {max_diff:.2e} >= tolerance {TOLERANCE}")
    
    return passed


def test_trade_parity():
    """Test trade mode parity."""
    if not (HAS_TORCH and HAS_MLX):
        print("Skipping trade parity test - need both PyTorch and MLX")
        return False
    
    print("\n" + "="*60)
    print("TEST 2: Trade Mode Parity")
    print("="*60)
    
    config = MLXConfig()
    torch_config = TorchConfig(
        n_signals=config.n_signals,
        hidden_dim=config.hidden_dim,
        sparkline_frames=config.sparkline_frames,
        sparkline_horizon=config.sparkline_horizon,
        H_layers=config.H_layers,
        L_layers=config.L_layers,
        H_cycles=config.H_cycles,
        L_cycles=config.L_cycles,
        n_heads=config.n_heads,
        dropout=config.dropout
    )
    
    torch_model = HierarchicalCodec(torch_config)
    mlx_model = MLXHierarchicalCodec(config)
    
    set_same_weights(torch_model, mlx_model)
    
    # Test data
    np.random.seed(42)
    B, T = 4, 32
    signals_np = np.random.randn(B, T, config.n_signals * 2).astype(np.float32)
    returns_np = np.random.randn(B, 1).astype(np.float32)
    
    signals_torch = torch.tensor(signals_np)
    returns_torch = torch.tensor(returns_np)
    signals_mlx = mx.array(signals_np)
    returns_mlx = mx.array(returns_np)
    
    # Forward pass
    with torch.no_grad():
        torch_output, _, torch_ret, torch_conf = torch_model.trade_loss(signals_torch, returns_torch)
    
    mlx_output, _, mlx_ret, mlx_conf = mlx_model.trade_loss(signals_mlx, returns_mlx)
    
    # Compare outputs
    torch_output_np = torch_output.numpy()
    mlx_output_np = np.array(mlx_output)
    
    torch_ret_np = torch_ret.numpy()
    mlx_ret_np = np.array(mlx_ret)
    
    torch_conf_np = torch_conf.numpy()
    mlx_conf_np = np.array(mlx_conf)
    
    # Check all outputs
    diff_output = np.abs(torch_output_np - mlx_output_np)
    diff_ret = np.abs(torch_ret_np - mlx_ret_np)
    diff_conf = np.abs(torch_conf_np - mlx_conf_np)
    
    max_diff_output = np.max(diff_output)
    max_diff_ret = np.max(diff_ret)
    max_diff_conf = np.max(diff_conf)
    
    print(f"Batch size: {B}, Sequence length: {T}")
    print(f"Trade output shape: {torch_output_np.shape}")
    print(f"Max difference (full output): {max_diff_output:.10e}")
    print(f"Max difference (return):      {max_diff_ret:.10e}")
    print(f"Max difference (confidence):  {max_diff_conf:.10e}")
    
    TOLERANCE = 1e-5
    passed = max_diff_output < TOLERANCE
    
    if passed:
        print(f"✓ PASSED: Max difference {max_diff_output:.2e} < tolerance {TOLERANCE}")
    else:
        print(f"✗ FAILED: Max difference {max_diff_output:.2e} >= tolerance {TOLERANCE}")
    
    return passed


def test_tiled_loops():
    """Test that tiled loops work correctly with larger batches/sequences."""
    if not (HAS_TORCH and HAS_MLX):
        print("Skipping tiled loops test - need both PyTorch and MLX")
        return False
    
    print("\n" + "="*60)
    print("TEST 3: Tiled Loops Correctness")
    print("="*60)
    
    config = MLXConfig()
    torch_config = TorchConfig(
        n_signals=config.n_signals,
        hidden_dim=config.hidden_dim,
        sparkline_frames=config.sparkline_frames,
        sparkline_horizon=config.sparkline_horizon,
        H_layers=config.H_layers,
        L_layers=config.L_layers,
        H_cycles=config.H_cycles,
        L_cycles=config.L_cycles,
        n_heads=config.n_heads,
        dropout=config.dropout
    )
    
    torch_model = HierarchicalCodec(torch_config)
    mlx_model = MLXHierarchicalCodec(config)
    
    set_same_weights(torch_model, mlx_model)
    
    # Test with larger batch that triggers tiling
    np.random.seed(42)
    B, T = 64, 128  # Large enough to trigger tiling
    signals_np = np.random.randn(B, T, config.n_signals * 2).astype(np.float32)
    signals_torch = torch.tensor(signals_np)
    signals_mlx = mx.array(signals_np)
    
    print(f"Testing with B={B}, T={T} (should trigger tiled execution)")
    
    # Forward pass
    with torch.no_grad():
        torch_output, _ = torch_model(signals_torch, mode="pretrain")
    
    mlx_output, _ = mlx_model(signals_mlx, mode="pretrain")
    
    torch_output_np = torch_output.numpy()
    mlx_output_np = np.array(mlx_output)
    
    diff = np.abs(torch_output_np - mlx_output_np)
    max_diff = np.max(diff)
    
    print(f"Max difference: {max_diff:.10e}")
    
    TOLERANCE = 1e-5
    passed = max_diff < TOLERANCE
    
    if passed:
        print(f"✓ PASSED: Tiled execution produces correct results")
    else:
        print(f"✗ FAILED: Tiled execution has numerical differences")
    
    return passed


def test_different_batch_sizes():
    """Test parity across different batch sizes."""
    if not (HAS_TORCH and HAS_MLX):
        print("Skipping batch size tests - need both PyTorch and MLX")
        return False
    
    print("\n" + "="*60)
    print("TEST 4: Multiple Batch Sizes")
    print("="*60)
    
    config = MLXConfig()
    torch_config = TorchConfig(
        n_signals=config.n_signals,
        hidden_dim=config.hidden_dim,
        sparkline_frames=config.sparkline_frames,
        sparkline_horizon=config.sparkline_horizon,
        H_layers=config.H_layers,
        L_layers=config.L_layers,
        H_cycles=config.H_cycles,
        L_cycles=config.L_cycles,
        n_heads=config.n_heads,
        dropout=config.dropout
    )
    
    batch_sizes = [1, 4, 8, 16, 32]
    seq_len = 32
    all_passed = True
    
    for B in batch_sizes:
        torch_model = HierarchicalCodec(torch_config)
        mlx_model = MLXHierarchicalCodec(config)
        set_same_weights(torch_model, mlx_model)
        
        np.random.seed(42)
        signals_np = np.random.randn(B, seq_len, config.n_signals * 2).astype(np.float32)
        signals_torch = torch.tensor(signals_np)
        signals_mlx = mx.array(signals_np)
        
        with torch.no_grad():
            torch_output, _ = torch_model(signals_torch, mode="pretrain")
        
        mlx_output, _ = mlx_model(signals_mlx, mode="pretrain")
        
        diff = np.max(np.abs(torch_output.numpy() - np.array(mlx_output)))
        passed = diff < 1e-5
        all_passed = all_passed and passed
        
        status = "✓" if passed else "✗"
        print(f"  B={B:2d}: {status} max_diff={diff:.2e}")
    
    if all_passed:
        print("✓ PASSED: All batch sizes produce correct results")
    else:
        print("✗ FAILED: Some batch sizes failed")
    
    return all_passed


def main():
    """Run all parity tests."""
    print("\n" + "="*70)
    print("PyTorch vs MLX HIERARCHICAL CODEC PARITY TESTS")
    print("="*70)
    
    if not (HAS_TORCH and HAS_MLX):
        print("\nMissing dependencies:")
        if not HAS_TORCH:
            print("  - PyTorch (pip install torch)")
        if not HAS_MLX:
            print("  - MLX (pip install mlx)")
        print("\nCannot run parity tests without both frameworks.")
        return False
    
    results = {
        'pretrain': test_pretrain_parity(),
        'trade': test_trade_parity(),
        'tiled': test_tiled_loops(),
        'batch_sizes': test_different_batch_sizes()
    }
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:15s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED - Outputs match within 1e-5 tolerance")
    else:
        print("\n✗ SOME TESTS FAILED")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
