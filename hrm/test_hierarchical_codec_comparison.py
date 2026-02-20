"""
Test script for Hierarchical Codec MLX implementation
Compares PyTorch vs MLX for speed and numerical parity
"""

import sys
import os

# Add venv to path for PyTorch
sys.path.insert(0, '/Users/jim/work/moneyfan/venv/lib/python3.14/site-packages')

# Add current directory to path
sys.path.append('/Users/jim/work/moneyfan')

import numpy as np
import time

# Import PyTorch
try:
    import torch
    print(f"PyTorch imported: {torch.__version__}")
    HAS_TORCH = True
except ImportError:
    print("PyTorch not available")
    HAS_TORCH = False

# Import MLX
try:
    import mlx.core as mx
    print("MLX imported")
    HAS_MLX = True
except ImportError:
    print("MLX not available")
    HAS_MLX = False

if HAS_TORCH and HAS_MLX:
    from hrm.hierarchical_codec import HierarchicalCodec, HierarchicalCodecConfig as TorchConfig
    from hrm.hierarchical_codec_mlx import MLXHierarchicalCodec, HierarchicalCodecConfig as MLXConfig
    
    print("\n" + "="*60)
    print("HIERARCHICAL CODEC COMPARISON TEST")
    print("="*60)
    
    # Configuration
    n_signals = 24
    hidden_dim = 64
    B, T = 4, 32
    
    # Create models
    torch_config = TorchConfig(n_signals=n_signals, hidden_dim=hidden_dim)
    torch_model = HierarchicalCodec(torch_config)
    
    mlx_config = MLXConfig(n_signals=n_signals, hidden_dim=hidden_dim)
    mlx_model = MLXHierarchicalCodec(mlx_config)
    
    print(f"PyTorch model parameters: {sum(p.numel() for p in torch_model.parameters()):,}")
    # MLX parameters() returns nested dictionaries
    def count_mlx_params(params_dict):
        count = 0
        for key, value in params_dict.items():
            if isinstance(value, dict):
                count += count_mlx_params(value)
            elif hasattr(value, 'size'):
                count += value.size
        return count
    
    mlx_param_count = count_mlx_params(mlx_model.parameters())
    print(f"MLX model parameters: {mlx_param_count:,}")
    
    # Create test data
    np.random.seed(42)
    signals_np = np.random.randn(B, T, n_signals * 2).astype(np.float32)
    signals_torch = torch.tensor(signals_np)
    signals_mlx = mx.array(signals_np)
    
    print(f"\nTest data: B={B}, T={T}, signals shape: {signals_np.shape}")
    
    # Test forward pass
    print("\n" + "-"*40)
    print("FORWARD PASS TEST")
    print("-"*40)
    
    # PyTorch
    torch_times = []
    for _ in range(5):
        start = time.time()
        with torch.no_grad():
            torch_output, torch_memory = torch_model(signals_torch, mode="pretrain")
        torch_times.append(time.time() - start)
    torch_time = np.mean(torch_times)
    
    # MLX
    mlx_times = []
    for _ in range(5):
        start = time.time()
        mlx_output, mlx_memory = mlx_model(signals_mlx, mode="pretrain")
        mlx_times.append(time.time() - start)
    mlx_time = np.mean(mlx_times)
    
    print(f"PyTorch time: {torch_time:.4f}s")
    print(f"MLX time: {mlx_time:.4f}s")
    print(f"Speedup: {torch_time / mlx_time:.2f}x")
    
    # Test numerical parity
    print("\n" + "-"*40)
    print("NUMERICAL PARITY TEST")
    print("-"*40)
    
    torch_output_np = torch_output.numpy()
    mlx_output_np = np.array(mlx_output)
    
    diff = np.abs(torch_output_np - mlx_output_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")
    print(f"Shapes match: {torch_output_np.shape == mlx_output_np.shape}")
    
    tolerance = 1e-5
    parity_pass = max_diff < tolerance
    print(f"{'✓' if parity_pass else '✗'} Outputs are {'numerically identical' if parity_pass else 'different'} within tolerance {tolerance}")
    
    # Test pretrain loss
    print("\n" + "-"*40)
    print("PRETRAIN LOSS TEST")
    print("-"*40)
    
    with torch.no_grad():
        torch_loss, _ = torch_model.pretrain_loss(signals_torch)
    mlx_loss, _ = mlx_model.pretrain_loss(signals_mlx)
    
    print(f"PyTorch loss: {float(torch_loss):.6f}")
    print(f"MLX loss: {float(mlx_loss):.6f}")
    loss_diff = abs(float(torch_loss) - float(mlx_loss))
    print(f"Loss difference: {loss_diff:.2e}")
    
    # Test trade loss
    print("\n" + "-"*40)
    print("TRADE LOSS TEST")
    print("-"*40)
    
    returns_np = np.random.randn(B, 1).astype(np.float32)
    returns_torch = torch.tensor(returns_np)
    returns_mlx = mx.array(returns_np)
    
    with torch.no_grad():
        torch_loss, _, torch_pred, torch_conf = torch_model.trade_loss(signals_torch, returns_torch)
    mlx_loss, _, mlx_pred, mlx_conf = mlx_model.trade_loss(signals_mlx, returns_mlx)
    
    print(f"PyTorch trade loss: {float(torch_loss):.6f}")
    print(f"MLX trade loss: {float(mlx_loss):.6f}")
    trade_loss_diff = abs(float(torch_loss) - float(mlx_loss))
    print(f"Trade loss difference: {trade_loss_diff:.2e}")
    
    torch_pred_np = torch_pred.detach().numpy()
    mlx_pred_np = np.array(mlx_pred)
    pred_diff = np.max(np.abs(torch_pred_np - mlx_pred_np))
    
    torch_conf_np = torch_conf.detach().numpy()
    mlx_conf_np = np.array(mlx_conf)
    conf_diff = np.max(np.abs(torch_conf_np - mlx_conf_np))
    
    print(f"Prediction max difference: {pred_diff:.2e}")
    print(f"Confidence max difference: {conf_diff:.2e}")
    
    # Test different batch sizes and sequence lengths
    print("\n" + "-"*40)
    print("SCALABILITY TEST")
    print("-"*40)
    
    test_configs = [(1, 16), (4, 32), (8, 64)]
    speedup_results = []
    
    for B_test, T_test in test_configs:
        print(f"\nB={B_test}, T={T_test}:")
        
        signals_np_test = np.random.randn(B_test, T_test, n_signals * 2).astype(np.float32)
        signals_torch_test = torch.tensor(signals_np_test)
        signals_mlx_test = mx.array(signals_np_test)
        
        # PyTorch
        torch_times_test = []
        for _ in range(3):
            start = time.time()
            with torch.no_grad():
                _ = torch_model(signals_torch_test, mode="pretrain")
            torch_times_test.append(time.time() - start)
        torch_time_test = np.mean(torch_times_test)
        
        # MLX
        mlx_times_test = []
        for _ in range(3):
            start = time.time()
            _ = mlx_model(signals_mlx_test, mode="pretrain")
            mlx_times_test.append(time.time() - start)
        mlx_time_test = np.mean(mlx_times_test)
        
        speedup = torch_time_test / mlx_time_test if mlx_time_test > 0 else 0
        speedup_results.append(speedup)
        
        print(f"  PyTorch: {torch_time_test:.4f}s, MLX: {mlx_time_test:.4f}s, Speedup: {speedup:.2f}x")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Numerical parity: {'PASS' if parity_pass else 'FAIL'} (max diff: {max_diff:.2e})")
    if speedup_results:
        avg_speedup = np.mean(speedup_results)
        max_speedup = max(speedup_results)
        min_speedup = min(speedup_results)
        print(f"✓ Speedup: {avg_speedup:.2f}x (avg), {max_speedup:.2f}x (max), {min_speedup:.2f}x (min)")
    
    # Test tiled vs non-tiled
    print("\n" + "-"*40)
    print("TILED vs NON-TILED TEST")
    print("-"*40)
    
    from hrm.hierarchical_codec_mlx import HierarchicalCodecConfig
    
    # Tiled config
    config_tiled = HierarchicalCodecConfig(n_signals=n_signals, hidden_dim=hidden_dim, tile_size=16, use_vmap=True)
    model_tiled = MLXHierarchicalCodec(config_tiled)
    
    # Non-tiled config (large tile size)
    config_non_tiled = HierarchicalCodecConfig(n_signals=n_signals, hidden_dim=hidden_dim, tile_size=256, use_vmap=False)
    model_non_tiled = MLXHierarchicalCodec(config_non_tiled)
    
    # Copy weights
    model_non_tiled.input_proj.weight = model_tiled.input_proj.weight
    model_non_tiled.H_init = model_tiled.H_init
    model_non_tiled.L_init = model_tiled.L_init
    model_non_tiled.signal_head.weight = model_tiled.signal_head.weight
    
    # Test
    signals_mlx_test = mx.array(signals_np)
    
    # Tiled
    tiled_times = []
    for _ in range(3):
        start = time.time()
        _ = model_tiled(signals_mlx_test, mode="pretrain")
        tiled_times.append(time.time() - start)
    tiled_time = np.mean(tiled_times)
    
    # Non-tiled
    non_tiled_times = []
    for _ in range(3):
        start = time.time()
        _ = model_non_tiled(signals_mlx_test, mode="pretrain")
        non_tiled_times.append(time.time() - start)
    non_tiled_time = np.mean(non_tiled_times)
    
    tiled_speedup = non_tiled_time / tiled_time if tiled_time > 0 else 0
    
    print(f"Tiled MLX: {tiled_time:.4f}s")
    print(f"Non-tiled MLX: {non_tiled_time:.4f}s")
    print(f"Tiling speedup: {tiled_speedup:.2f}x")
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    if parity_pass:
        print("✓ Numerical parity: PASSED")
    else:
        print("✗ Numerical parity: FAILED")
    
    if speedup_results:
        print(f"✓ PyTorch vs MLX speedup: {avg_speedup:.2f}x average")
    
    print(f"✓ Tiling speedup: {tiled_speedup:.2f}x")
    
elif not HAS_TORCH or not HAS_MLX:
    print("\nMissing dependencies:")
    if not HAS_TORCH:
        print("  - PyTorch not available")
    if not HAS_MLX:
        print("  - MLX not available")
    print("\nPlease install:")
    print("  pip install torch")
    print("  pip install mlx")