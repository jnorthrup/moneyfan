"""
Simple comparison test for PyTorch vs MLX hierarchical codec
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import time

# Try importing PyTorch from venv
try:
    sys.path.insert(0, '/Users/jim/work/moneyfan/venv/lib/python3.14/site-packages')
    import torch
    print(f"PyTorch imported: {torch.__version__}")
    HAS_TORCH = True
except ImportError:
    print("PyTorch not available")
    HAS_TORCH = False

# Try importing MLX from system
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
    print("COMPARISON TEST")
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
    print(f"MLX model parameters: {sum(p.numel() for p in mlx_model.parameters()):,}")
    
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
    if max_diff < tolerance:
        print(f"✓ Outputs are numerically identical within tolerance {tolerance}")
    else:
        print(f"✗ Outputs differ beyond tolerance {tolerance}")
    
    # Test pretrain loss
    print("\n" + "-"*40)
    print("PRETRAIN LOSS TEST")
    print("-"*40)
    
    torch_loss, _ = torch_model.pretrain_loss(signals_torch)
    mlx_loss, _ = mlx_model.pretrain_loss(signals_mlx)
    
    print(f"PyTorch loss: {float(torch_loss):.6f}")
    print(f"MLX loss: {float(mlx_loss):.6f}")
    print(f"Loss difference: {abs(float(torch_loss) - float(mlx_loss)):.2e}")
    
    # Test trade loss
    print("\n" + "-"*40)
    print("TRADE LOSS TEST")
    print("-"*40)
    
    returns_np = np.random.randn(B, 1).astype(np.float32)
    returns_torch = torch.tensor(returns_np)
    returns_mlx = mx.array(returns_np)
    
    torch_loss, _, torch_pred, torch_conf = torch_model.trade_loss(signals_torch, returns_torch)
    mlx_loss, _, mlx_pred, mlx_conf = mlx_model.trade_loss(signals_mlx, returns_mlx)
    
    print(f"PyTorch trade loss: {float(torch_loss):.6f}")
    print(f"MLX trade loss: {float(mlx_loss):.6f}")
    print(f"Trade loss difference: {abs(float(torch_loss) - float(mlx_loss)):.2e}")
    
    torch_pred_np = torch_pred.numpy()
    mlx_pred_np = np.array(mlx_pred)
    pred_diff = np.max(np.abs(torch_pred_np - mlx_pred_np))
    
    torch_conf_np = torch_conf.numpy()
    mlx_conf_np = np.array(mlx_conf)
    conf_diff = np.max(np.abs(torch_conf_np - mlx_conf_np))
    
    print(f"Prediction max difference: {pred_diff:.2e}")
    print(f"Confidence max difference: {conf_diff:.2e}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Numerical parity: {'PASS' if max_diff < tolerance else 'FAIL'}")
    print(f"✓ Speedup: {torch_time / mlx_time:.2f}x")
    
elif not HAS_TORCH or not HAS_MLX:
    print("\nMissing dependencies:")
    if not HAS_TORCH:
        print("  - PyTorch not available")
    if not HAS_MLX:
        print("  - MLX not available")
    print("\nPlease install:")
    print("  pip install torch")
    print("  pip install mlx")