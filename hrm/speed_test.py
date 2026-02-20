"""
Speed Test: Measure performance improvements of MLX implementation
===================================================================

This script measures and compares execution times between PyTorch and MLX implementations.
"""

import numpy as np
import time
import sys
from pathlib import Path
import matplotlib.pyplot as plt

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


def benchmark_forward_pass(model, signals, framework, num_runs=10, warmup=3):
    """Benchmark forward pass timing."""
    if framework == "torch":
        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = model(signals, mode="pretrain")
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(signals, mode="pretrain")
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append(time.perf_counter() - start)
        
    elif framework == "mlx":
        # Warmup
        for _ in range(warmup):
            _ = model(signals, mode="pretrain")
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(signals, mode="pretrain")
            mx.synchronize()
            times.append(time.perf_counter() - start)
    
    return np.mean(times), np.std(times)


def benchmark_pretrain(model, framework, B, T, n_signals=24, num_runs=10):
    """Benchmark pretrain mode."""
    if framework == "torch":
        if not HAS_TORCH:
            return None, None
        config = TorchConfig(n_signals=n_signals)
        signals_np = np.random.randn(B, T, n_signals * 2).astype(np.float32)
        signals = torch.tensor(signals_np)
        
    elif framework == "mlx":
        if not HAS_MLX:
            return None, None
        config = MLXConfig(n_signals=n_signals)
        signals_np = np.random.randn(B, T, n_signals * 2).astype(np.float32)
        signals = mx.array(signals_np)
    
    # Run benchmark
    mean_time, std_time = benchmark_forward_pass(model, signals, framework, num_runs)
    
    return mean_time, std_time


def run_comprehensive_benchmark():
    """Run comprehensive benchmark across different configurations."""
    if not (HAS_TORCH and HAS_MLX):
        print("Need both PyTorch and MLX for comparison")
        return None
    
    print("\n" + "="*70)
    print("COMPREHENSIVE SPEED BENCHMARK")
    print("="*70)
    
    # Test configurations
    test_configs = [
        # (B, T)
        (1, 16),   # Small batch, short sequence
        (1, 64),   # Small batch, long sequence
        (4, 32),   # Medium batch, medium sequence
        (8, 32),   # Medium batch, medium sequence
        (16, 32),  # Larger batch, medium sequence
        (32, 64),  # Larger batch, long sequence
        (64, 128), # Large batch, long sequence (should trigger tiling)
    ]
    
    results = []
    
    for B, T in test_configs:
        print(f"\nTesting B={B}, T={T}:")
        
        # PyTorch
        torch_config = TorchConfig()
        torch_model = HierarchicalCodec(torch_config)
        
        # MLX
        mlx_config = MLXConfig()
        mlx_model = MLXHierarchicalCodec(mlx_config)
        
        # Set same weights
        set_same_weights(torch_model, mlx_model)
        
        # Benchmark PyTorch
        torch_time, torch_std = benchmark_pretrain(torch_model, "torch", B, T)
        if torch_time is None:
            print("  PyTorch: Not available")
            continue
        
        # Benchmark MLX
        mlx_time, mlx_std = benchmark_pretrain(mlx_model, "mlx", B, T)
        if mlx_time is None:
            print("  MLX: Not available")
            continue
        
        # Calculate speedup
        speedup = torch_time / mlx_time if mlx_time > 0 else 0
        
        # Store results
        results.append({
            'B': B,
            'T': T,
            'torch_time': torch_time,
            'torch_std': torch_std,
            'mlx_time': mlx_time,
            'mlx_std': mlx_std,
            'speedup': speedup
        })
        
        # Print results
        print(f"  PyTorch: {torch_time:.4f} ± {torch_std:.4f}s")
        print(f"  MLX:     {mlx_time:.4f} ± {mlx_std:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")
    
    return results


def plot_results(results, filename="speedup_plot.png"):
    """Create speedup visualization."""
    if results is None or len(results) == 0:
        print("No results to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract data
    B = [r['B'] for r in results]
    T = [r['T'] for r in results]
    torch_times = [r['torch_time'] for r in results]
    mlx_times = [r['mlx_time'] for r in results]
    speedups = [r['speedup'] for r in results]
    
    # Plot 1: Time comparison
    ax = axes[0, 0]
    x = np.arange(len(B))
    width = 0.35
    ax.bar(x - width/2, torch_times, width, label='PyTorch', alpha=0.8)
    ax.bar(x + width/2, mlx_times, width, label='MLX', alpha=0.8)
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Execution Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f"B={b}, T={t}" for b, t in zip(B, T)], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Speedup
    ax = axes[0, 1]
    bars = ax.bar(range(len(speedups)), speedups, color='green', alpha=0.7)
    ax.axhline(y=5, color='red', linestyle='--', label='5x Target', alpha=0.7)
    ax.axhline(y=1, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Speedup (PyTorch/MLX)')
    ax.set_title('Speedup by Configuration')
    ax.set_xticks(range(len(speedups)))
    ax.set_xticklabels([f"B={b}, T={t}" for b, t in zip(B, T)], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add speedup values on bars
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{speedup:.1f}x', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Speedup vs Batch Size
    ax = axes[1, 0]
    unique_B = sorted(set(B))
    speedup_by_B = []
    for b in unique_B:
        speeds = [r['speedup'] for r in results if r['B'] == b]
        if speeds:
            speedup_by_B.append(np.mean(speeds))
        else:
            speedup_by_B.append(0)
    
    ax.plot(unique_B, speedup_by_B, 'o-', linewidth=2, markersize=8)
    ax.axhline(y=5, color='red', linestyle='--', label='5x Target', alpha=0.7)
    ax.axhline(y=1, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Batch Size (B)')
    ax.set_ylabel('Average Speedup')
    ax.set_title('Speedup vs Batch Size')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 4: Speedup vs Sequence Length
    ax = axes[1, 1]
    unique_T = sorted(set(T))
    speedup_by_T = []
    for t in unique_T:
        speeds = [r['speedup'] for r in results if r['T'] == t]
        if speeds:
            speedup_by_T.append(np.mean(speeds))
        else:
            speedup_by_T.append(0)
    
    ax.plot(unique_T, speedup_by_T, 's-', linewidth=2, markersize=8, color='orange')
    ax.axhline(y=5, color='red', linestyle='--', label='5x Target', alpha=0.7)
    ax.axhline(y=1, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Sequence Length (T)')
    ax.set_ylabel('Average Speedup')
    ax.set_title('Speedup vs Sequence Length')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to {filename}")
    plt.close()


def analyze_speedup_acceptance(results, target_speedup=5.0):
    """Analyze if MLX meets the 5x speedup acceptance criteria."""
    if results is None:
        return False
    
    print("\n" + "="*70)
    print("ACCEPTANCE CRITERIA ANALYSIS")
    print("="*70)
    
    print(f"Target: MLX should be at least {target_speedup}x faster")
    print()
    
    # Check each configuration
    all_passed = True
    speedups_above_target = []
    
    for i, result in enumerate(results):
        B, T = result['B'], result['T']
        speedup = result['speedup']
        torch_time = result['torch_time']
        mlx_time = result['mlx_time']
        
        passed = speedup >= target_speedup
        all_passed = all_passed and passed
        
        status = "✓" if passed else "✗"
        speedups_above_target.append(speedup)
        
        print(f"  Config {i+1}: B={B:2d}, T={T:3d}")
        print(f"    PyTorch: {torch_time:.4f}s, MLX: {mlx_time:.4f}s")
        print(f"    Speedup: {speedup:.2f}x {status}")
    
    # Overall analysis
    print()
    print("SUMMARY:")
    print(f"  Average speedup: {np.mean(speedups_above_target):.2f}x")
    print(f"  Max speedup:     {max(speedups_above_target):.2f}x")
    print(f"  Min speedup:     {min(speedups_above_target):.2f}x")
    
    # Count how many configs meet target
    configs_meeting_target = sum(1 for s in speedups_above_target if s >= target_speedup)
    total_configs = len(speedups_above_target)
    
    print(f"\n  Configurations meeting target: {configs_meeting_target}/{total_configs}")
    
    if all_passed:
        print(f"\n✓ ACCEPTANCE CRITERIA MET: MLX is at least {target_speedup}x faster in all configurations")
    else:
        print(f"\n✗ ACCEPTANCE CRITERIA NOT MET: MLX fails to meet {target_speedup}x speedup in some configurations")
    
    return all_passed


def benchmark_tiling_vs_non_tiling():
    """Benchmark tiling vs non-tiling execution."""
    if not (HAS_TORCH and HAS_MLX):
        return None
    
    print("\n" + "="*70)
    print("TILING IMPACT ANALYSIS")
    print("="*70)
    
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
    
    # Test different tile sizes
    tile_sizes = [8, 16, 32, 64]
    B, T = 64, 128  # Large enough to benefit from tiling
    
    print(f"Testing B={B}, T={T} with different tile sizes")
    print()
    
    results = []
    
    # Baseline: PyTorch (no tiling)
    torch_model = HierarchicalCodec(torch_config)
    mlx_baseline = MLXHierarchicalCodec(config)
    
    # Warmup and benchmark PyTorch
    signals_np = np.random.randn(B, T, config.n_signals * 2).astype(np.float32)
    signals_torch = torch.tensor(signals_np)
    
    torch_times = []
    for _ in range(5):
        start = time.perf_counter()
        with torch.no_grad():
            _ = torch_model(signals_torch, mode="pretrain")
        torch_times.append(time.perf_counter() - start)
    torch_time = np.mean(torch_times)
    
    print(f"PyTorch (baseline): {torch_time:.4f}s")
    
    # Test MLX with different tile sizes
    for tile_size in tile_sizes:
        # Create config with specific tile size
        config.tile_size = tile_size
        mlx_model = MLXHierarchicalCodec(config)
        
        # Set weights
        set_same_weights(torch_model, mlx_model)
        
        signals_mlx = mx.array(signals_np)
        
        # Warmup and benchmark
        mlx_times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = mlx_model(signals_mlx, mode="pretrain")
            mx.synchronize()
            mlx_times.append(time.perf_counter() - start)
        mlx_time = np.mean(mlx_times)
        
        speedup = torch_time / mlx_time if mlx_time > 0 else 0
        
        results.append({
            'tile_size': tile_size,
            'mlx_time': mlx_time,
            'speedup': speedup
        })
        
        print(f"  Tile size {tile_size:2d}: {mlx_time:.4f}s (speedup: {speedup:.2f}x)")
    
    # Find optimal tile size
    if results:
        best_result = max(results, key=lambda x: x['speedup'])
        print(f"\nOptimal tile size: {best_result['tile_size']} (speedup: {best_result['speedup']:.2f}x)")
    
    return results


def main():
    """Run all speed tests."""
    print("\n" + "="*70)
    print("MLX vs PyTorch HIERARCHICAL CODEC SPEED TESTS")
    print("="*70)
    
    if not (HAS_TORCH and HAS_MLX):
        print("\nMissing dependencies:")
        if not HAS_TORCH:
            print("  - PyTorch (pip install torch)")
        if not HAS_MLX:
            print("  - MLX (pip install mlx)")
        print("\nCannot run speed tests without both frameworks.")
        return False
    
    # Run comprehensive benchmark
    results = run_comprehensive_benchmark()
    
    if results:
        # Plot results
        plot_results(results)
        
        # Analyze acceptance criteria
        acceptance_passed = analyze_speedup_acceptance(results, target_speedup=5.0)
        
        # Benchmark tiling impact
        tiling_results = benchmark_tiling_vs_non_tiling()
        
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        
        print(f"Acceptance criteria (5x speedup): {'✓ PASSED' if acceptance_passed else '✗ FAILED'}")
        
        if tiling_results:
            avg_speedup = np.mean([r['speedup'] for r in results])
            print(f"Average speedup: {avg_speedup:.2f}x")
            print(f"Target: 5.0x")
            print(f"Percentage of target achieved: {(avg_speedup/5.0)*100:.1f}%")
        
        return acceptance_passed
    else:
        print("No results generated")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
