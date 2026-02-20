"""
Test script for Hierarchical Codec MLX implementation
======================================================

Compares PyTorch vs MLX implementations:
1. Numerical parity verification
2. Speedup measurements
3. Tiled vs non-tiled performance
"""

import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass
from typing import Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from hrm.hierarchical_codec import HierarchicalCodec, HierarchicalCodecConfig as TorchConfig
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available")

try:
    import mlx.core as mx
    import mlx.nn as nn_mlx
    from hrm.hierarchical_codec_mlx import (
        MLXHierarchicalCodec,
        MLXCodecTrainer,
        HierarchicalCodecConfig as MLXConfig
    )
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("MLX not available")


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

@dataclass
class TestConfig:
    batch_sizes: list = None
    seq_lengths: list = None
    tolerance: float = 1e-5
    warmup_runs: int = 3
    measure_runs: int = 5
    n_signals: int = 24
    hidden_dim: int = 64
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2, 4, 8]
        if self.seq_lengths is None:
            self.seq_lengths = [16, 32, 64]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def copy_torch_to_mlx(torch_model, mlx_model):
    """Copy weights from PyTorch to MLX model."""
    with torch.no_grad():
        # Copy input projection
        mlx_model.input_proj.weight = mx.array(torch_model.input_proj.weight.numpy().T)
        
        # Copy H/L init
        mlx_model.H_init = mx.array(torch_model.H_init.numpy())
        mlx_model.L_init = mx.array(torch_model.L_init.numpy())
        
        # Copy heads
        mlx_model.signal_head.weight = mx.array(torch_model.signal_head.weight.numpy().T)
        mlx_model.return_head.weight = mx.array(torch_model.return_head.weight.numpy().T)
        mlx_model.confidence_head.weight = mx.array(torch_model.confidence_head.weight.numpy().T)
        mlx_model.stop_head.weight = mx.array(torch_model.stop_head.weight.numpy().T)
        mlx_model.tp_head.weight = mx.array(torch_model.tp_head.weight.numpy().T)
        mlx_model.pos_head.weight = mx.array(torch_model.pos_head.weight.numpy().T)
    
    return mlx_model


def create_test_data(B: int, T: int, n_signals: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create test data with reproducible randomness."""
    np.random.seed(42)
    signals_np = np.random.randn(B, T, n_signals * 2).astype(np.float32)
    returns_np = np.random.randn(B, 1).astype(np.float32)
    return signals_np, returns_np


# =============================================================================
# NUMERICAL PARITY TESTS
# =============================================================================

def test_numerical_parity(test_config: TestConfig) -> bool:
    """Test numerical parity between PyTorch and MLX implementations."""
    if not HAS_TORCH or not HAS_MLX:
        print("Skipping numerical parity test - missing PyTorch or MLX")
        return False
    
    print("\n" + "="*60)
    print("NUMERICAL PARITY TESTS")
    print("="*60)
    
    all_passed = True
    
    for B in test_config.batch_sizes:
        for T in test_config.seq_lengths:
            print(f"\nTesting B={B}, T={T}...")
            
            # Create models
            torch_config = TorchConfig(
                n_signals=test_config.n_signals,
                hidden_dim=test_config.hidden_dim,
                H_cycles=2,
                L_cycles=3
            )
            torch_model = HierarchicalCodec(torch_config)
            
            mlx_config = MLXConfig(
                n_signals=test_config.n_signals,
                hidden_dim=test_config.hidden_dim,
                H_cycles=2,
                L_cycles=3
            )
            mlx_model = MLXHierarchicalCodec(mlx_config)
            
            # Copy weights
            mlx_model = copy_torch_to_mlx(torch_model, mlx_model)
            
            # Create test data
            signals_np, returns_np = create_test_data(B, T, test_config.n_signals)
            signals_torch = torch.tensor(signals_np)
            signals_mlx = mx.array(signals_np)
            
            # Test pretrain mode
            with torch.no_grad():
                torch_output, torch_memory = torch_model(signals_torch, mode="pretrain")
            mlx_output, mlx_memory = mlx_model(signals_mlx, mode="pretrain")
            
            torch_output_np = torch_output.numpy()
            mlx_output_np = np.array(mlx_output)
            
            diff = np.abs(torch_output_np - mlx_output_np)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            passed = max_diff < test_config.tolerance
            status = "✓ PASS" if passed else "✗ FAIL"
            
            print(f"  Pretrain: {status} (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")
            all_passed = all_passed and passed
            
            # Test trade mode
            returns_torch = torch.tensor(returns_np)
            returns_mlx = mx.array(returns_np)
            
            with torch.no_grad():
                torch_loss, _, torch_pred, torch_conf = torch_model.trade_loss(signals_torch, returns_torch)
            mlx_loss, _, mlx_pred, mlx_conf = mlx_model.trade_loss(signals_mlx, returns_mlx)
            
            torch_loss_np = float(torch_loss)
            mlx_loss_np = float(mlx_loss)
            
            loss_diff = abs(torch_loss_np - mlx_loss_np)
            pred_diff = np.max(np.abs(torch_pred.numpy() - np.array(mlx_pred)))
            conf_diff = np.max(np.abs(torch_conf.numpy() - np.array(mlx_conf)))
            
            loss_passed = loss_diff < test_config.tolerance
            pred_passed = pred_diff < test_config.tolerance
            conf_passed = conf_diff < test_config.tolerance
            
            status_loss = "✓ PASS" if loss_passed else "✗ FAIL"
            status_pred = "✓ PASS" if pred_passed else "✗ FAIL"
            status_conf = "✓ PASS" if conf_passed else "✗ FAIL"
            
            print(f"  Trade loss: {status_loss} (diff={loss_diff:.2e})")
            print(f"  Trade predictions: {status_pred} (max_diff={pred_diff:.2e})")
            print(f"  Trade confidence: {status_conf} (max_diff={conf_diff:.2e})")
            
            all_passed = all_passed and loss_passed and pred_passed and conf_passed
    
    print(f"\nOverall parity test: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
    return all_passed


# =============================================================================
# SPEEDUP MEASUREMENTS
# =============================================================================

def measure_speedup(test_config: TestConfig) -> dict:
    """Measure speedup of MLX vs PyTorch."""
    if not HAS_TORCH or not HAS_MLX:
        print("Skipping speedup measurement - missing PyTorch or MLX")
        return {}
    
    print("\n" + "="*60)
    print("SPEEDUP MEASUREMENTS")
    print("="*60)
    
    results = []
    
    for B in test_config.batch_sizes:
        for T in test_config.seq_lengths:
            print(f"\nB={B}, T={T}...")
            
            # Create models
            torch_config = TorchConfig(
                n_signals=test_config.n_signals,
                hidden_dim=test_config.hidden_dim,
                H_cycles=2,
                L_cycles=3
            )
            torch_model = HierarchicalCodec(torch_config)
            
            mlx_config = MLXConfig(
                n_signals=test_config.n_signals,
                hidden_dim=test_config.hidden_dim,
                H_cycles=2,
                L_cycles=3
            )
            mlx_model = MLXHierarchicalCodec(mlx_config)
            
            # Copy weights
            mlx_model = copy_torch_to_mlx(torch_model, mlx_model)
            
            # Create test data
            signals_np, returns_np = create_test_data(B, T, test_config.n_signals)
            signals_torch = torch.tensor(signals_np)
            signals_mlx = mx.array(signals_np)
            
            # Warm up
            for _ in range(test_config.warmup_runs):
                with torch.no_grad():
                    _ = torch_model(signals_torch, mode="pretrain")
                _ = mlx_model(signals_mlx, mode="pretrain")
            
            # Measure PyTorch
            torch_times = []
            for _ in range(test_config.measure_runs):
                start = time.time()
                with torch.no_grad():
                    _ = torch_model(signals_torch, mode="pretrain")
                torch_times.append(time.time() - start)
            torch_time = np.mean(torch_times)
            
            # Measure MLX
            mlx_times = []
            for _ in range(test_config.measure_runs):
                start = time.time()
                _ = mlx_model(signals_mlx, mode="pretrain")
                mlx_times.append(time.time() - start)
            mlx_time = np.mean(mlx_times)
            
            speedup = torch_time / mlx_time if mlx_time > 0 else 0
            
            results.append({
                'B': B,
                'T': T,
                'torch_time': torch_time,
                'mlx_time': mlx_time,
                'speedup': speedup
            })
            
            print(f"  PyTorch: {torch_time:.4f}s")
            print(f"  MLX:    {mlx_time:.4f}s")
            print(f"  Speedup: {speedup:.2f}x")
    
    # Summary
    if results:
        avg_speedup = np.mean([r['speedup'] for r in results])
        max_speedup = max([r['speedup'] for r in results])
        min_speedup = min([r['speedup'] for r in results])
        
        print(f"\n{'='*60}")
        print("SPEEDUP SUMMARY")
        print(f"{'='*60}")
        print(f"Average speedup: {avg_speedup:.2f}x")
        print(f"Max speedup: {max_speedup:.2f}x")
        print(f"Min speedup: {min_speedup:.2f}x")
        
        # Print results table
        print(f"\n{'='*60}")
        print("DETAILED RESULTS")
        print(f"{'='*60}")
        print(f"{'B':<4} {'T':<4} {'PyTorch':<10} {'MLX':<10} {'Speedup':<8}")
        print("-" * 40)
        for r in results:
            print(f"{r['B']:<4} {r['T']:<4} {r['torch_time']:<10.4f} {r['mlx_time']:<10.4f} {r['speedup']:<8.2f}x")
    
    return results


# =============================================================================
# TILED vs NON-TILED PERFORMANCE
# =============================================================================

def compare_tiled_vs_non_tiled(test_config: TestConfig) -> dict:
    """Compare tiled vs non-tiled MLX implementation."""
    if not HAS_MLX:
        print("Skipping tiled vs non-tiled test - MLX not available")
        return {}
    
    print("\n" + "="*60)
    print("TILED vs NON-TILED PERFORMANCE")
    print("="*60)
    
    results = []
    
    for B in test_config.batch_sizes:
        for T in test_config.seq_lengths:
            print(f"\nB={B}, T={T}...")
            
            # Create models
            config_tiled = MLXConfig(
                n_signals=test_config.n_signals,
                hidden_dim=test_config.hidden_dim,
                H_cycles=2,
                L_cycles=3,
                tile_size=16,
                use_vmap=True
            )
            config_non_tiled = MLXConfig(
                n_signals=test_config.n_signals,
                hidden_dim=test_config.hidden_dim,
                H_cycles=2,
                L_cycles=3,
                tile_size=256,  # Large tile size effectively disables tiling
                use_vmap=False
            )
            
            mlx_model_tiled = MLXHierarchicalCodec(config_tiled)
            mlx_model_non_tiled = MLXHierarchicalCodec(config_non_tiled)
            
            # Copy weights
            mlx_model_non_tiled.input_proj.weight = mlx_model_tiled.input_proj.weight
            mlx_model_non_tiled.H_init = mlx_model_tiled.H_init
            mlx_model_non_tiled.L_init = mlx_model_tiled.L_init
            mlx_model_non_tiled.signal_head.weight = mlx_model_tiled.signal_head.weight
            mlx_model_non_tiled.return_head.weight = mlx_model_tiled.return_head.weight
            mlx_model_non_tiled.confidence_head.weight = mlx_model_tiled.confidence_head.weight
            mlx_model_non_tiled.stop_head.weight = mlx_model_tiled.stop_head.weight
            mlx_model_non_tiled.tp_head.weight = mlx_model_tiled.tp_head.weight
            mlx_model_non_tiled.pos_head.weight = mlx_model_tiled.pos_head.weight
            
            # Create test data
            signals_np, _ = create_test_data(B, T, test_config.n_signals)
            signals_mlx = mx.array(signals_np)
            
            # Warm up
            for _ in range(test_config.warmup_runs):
                _ = mlx_model_tiled(signals_mlx, mode="pretrain")
                _ = mlx_model_non_tiled(signals_mlx, mode="pretrain")
            
            # Measure tiled
            tiled_times = []
            for _ in range(test_config.measure_runs):
                start = time.time()
                _ = mlx_model_tiled(signals_mlx, mode="pretrain")
                tiled_times.append(time.time() - start)
            tiled_time = np.mean(tiled_times)
            
            # Measure non-tiled
            non_tiled_times = []
            for _ in range(test_config.measure_runs):
                start = time.time()
                _ = mlx_model_non_tiled(signals_mlx, mode="pretrain")
                non_tiled_times.append(time.time() - start)
            non_tiled_time = np.mean(non_tiled_times)
            
            speedup = non_tiled_time / tiled_time if tiled_time > 0 else 0
            
            results.append({
                'B': B,
                'T': T,
                'tiled_time': tiled_time,
                'non_tiled_time': non_tiled_time,
                'speedup': speedup
            })
            
            print(f"  Tiled:     {tiled_time:.4f}s")
            print(f"  Non-tiled: {non_tiled_time:.4f}s")
            print(f"  Speedup:   {speedup:.2f}x")
    
    # Summary
    if results:
        avg_speedup = np.mean([r['speedup'] for r in results])
        
        print(f"\n{'='*60}")
        print("TILING SUMMARY")
        print(f"{'='*60}")
        print(f"Average tiling speedup: {avg_speedup:.2f}x")
        
        # Print results table
        print(f"\n{'='*60}")
        print("TILING DETAILED RESULTS")
        print(f"{'='*60}")
        print(f"{'B':<4} {'T':<4} {'Tiled':<10} {'Non-tiled':<10} {'Speedup':<8}")
        print("-" * 40)
        for r in results:
            print(f"{r['B']:<4} {r['T']:<4} {r['tiled_time']:<10.4f} {r['non_tiled_time']:<10.4f} {r['speedup']:<8.2f}x")
    
    return results


# =============================================================================
# BATCHING TESTS
# =============================================================================

def test_batching_performance(test_config: TestConfig) -> dict:
    """Test how MLX scales with batch size."""
    if not HAS_MLX:
        print("Skipping batching test - MLX not available")
        return {}
    
    print("\n" + "="*60)
    print("BATCHING PERFORMANCE TEST")
    print("="*60)
    
    results = []
    T = 32  # Fixed sequence length
    
    for B in [1, 2, 4, 8, 16, 32, 64]:
        print(f"\nTesting B={B}...")
        
        config = MLXConfig(
            n_signals=test_config.n_signals,
            hidden_dim=test_config.hidden_dim,
            H_cycles=2,
            L_cycles=3
        )
        mlx_model = MLXHierarchicalCodec(config)
        
        # Create test data
        signals_np, _ = create_test_data(B, T, test_config.n_signals)
        signals_mlx = mx.array(signals_np)
        
        # Warm up
        for _ in range(test_config.warmup_runs):
            _ = mlx_model(signals_mlx, mode="pretrain")
        
        # Measure
        times = []
        for _ in range(test_config.measure_runs):
            start = time.time()
            _ = mlx_model(signals_mlx, mode="pretrain")
            times.append(time.time() - start)
        avg_time = np.mean(times)
        
        # Compute throughput
        tokens = B * T
        throughput = tokens / avg_time if avg_time > 0 else 0
        
        results.append({
            'B': B,
            'T': T,
            'time': avg_time,
            'tokens': tokens,
            'throughput': throughput
        })
        
        print(f"  Time: {avg_time:.4f}s, Throughput: {throughput:.0f} tokens/s")
    
    # Summary
    if results:
        print(f"\n{'='*60}")
        print("BATCHING SUMMARY")
        print(f"{'='*60}")
        print(f"Best throughput: {max([r['throughput'] for r in results]):.0f} tokens/s")
        print(f"Worst throughput: {min([r['throughput'] for r in results]):.0f} tokens/s")
        
        # Print scalability
        print(f"\n{'='*60}")
        print("SCALABILITY ANALYSIS")
        print(f"{'='*60}")
        print(f"{'B':<6} {'Time':<10} {'Tokens':<8} {'Throughput':<12}")
        print("-" * 40)
        for r in results:
            print(f"{r['B']:<6} {r['time']:<10.4f} {r['tokens']:<8} {r['throughput']:<12.0f}")
    
    return results


# =============================================================================
# MEMORY USAGE TEST
# =============================================================================

def test_memory_usage(test_config: TestConfig) -> dict:
    """Test memory usage of MLX implementation."""
    if not HAS_MLX:
        print("Skipping memory test - MLX not available")
        return {}
    
    print("\n" + "="*60)
    print("MEMORY USAGE TEST")
    print("="*60)
    
    results = []
    
    for B in [1, 4, 8]:
        for T in [16, 32]:
            print(f"\nB={B}, T={T}...")
            
            config = MLXConfig(
                n_signals=test_config.n_signals,
                hidden_dim=test_config.hidden_dim,
                H_cycles=2,
                L_cycles=3
            )
            mlx_model = MLXHierarchicalCodec(config)
            
            # Create test data
            signals_np, _ = create_test_data(B, T, test_config.n_signals)
            signals_mlx = mx.array(signals_np)
            
            # Run forward pass
            output, memory = mlx_model(signals_mlx, mode="pretrain")
            
            # Estimate memory usage (approximate)
            total_params = sum(p.numel() for p in mlx_model.parameters())
            input_size = signals_np.nbytes
            output_size = output.nbytes
            
            # Memory for activations (simplified estimate)
            activation_memory = B * T * config.hidden_dim * 4 * 4  # 4 layers * 4 bytes per float32
            
            estimated_memory = total_params * 4 + input_size + output_size + activation_memory  # bytes
            
            results.append({
                'B': B,
                'T': T,
                'params': total_params,
                'input_size': input_size,
                'output_size': output_size,
                'estimated_memory_mb': estimated_memory / (1024 * 1024)
            })
            
            print(f"  Params: {total_params:,}")
            print(f"  Input: {input_size:,} bytes")
            print(f"  Output: {output_size:,} bytes")
            print(f"  Est. memory: {estimated_memory / (1024 * 1024):.2f} MB")
    
    # Summary
    if results:
        print(f"\n{'='*60}")
        print("MEMORY SUMMARY")
        print(f"{'='*60}")
        print(f"Max memory: {max([r['estimated_memory_mb'] for r in results]):.2f} MB")
        print(f"Min memory: {min([r['estimated_memory_mb'] for r in results]):.2f} MB")
    
    return results


# =============================================================================
# COMPREHENSIVE TEST SUITE
# =============================================================================

def run_comprehensive_tests():
    """Run all tests."""
    test_config = TestConfig(
        batch_sizes=[1, 2, 4, 8],
        seq_lengths=[16, 32],
        tolerance=1e-5,
        warmup_runs=2,
        measure_runs=3
    )
    
    print("="*60)
    print("HIERARCHICAL CODEC MLX - COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    results = {}
    
    # 1. Numerical parity
    if HAS_TORCH and HAS_MLX:
        results['parity'] = test_numerical_parity(test_config)
    
    # 2. Speedup measurements
    if HAS_TORCH and HAS_MLX:
        results['speedup'] = measure_speedup(test_config)
    
    # 3. Tiled vs non-tiled
    results['tiled_vs_non_tiled'] = compare_tiled_vs_non_tiled(test_config)
    
    # 4. Batching performance
    results['batching'] = test_batching_performance(test_config)
    
    # 5. Memory usage
    results['memory'] = test_memory_usage(test_config)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    if results.get('parity'):
        print(f"✓ Numerical parity: {'PASSED' if results['parity'] else 'FAILED'}")
    else:
        print("⊘ Numerical parity test skipped")
    
    if results.get('speedup'):
        avg_speedup = np.mean([r['speedup'] for r in results['speedup']])
        print(f"✓ PyTorch vs MLX speedup: {avg_speedup:.2f}x average")
    else:
        print("⊘ Speedup measurement skipped")
    
    if results.get('tiled_vs_non_tiled'):
        avg_tiled_speedup = np.mean([r['speedup'] for r in results['tiled_vs_non_tiled']])
        print(f"✓ Tiled vs non-tiled speedup: {avg_tiled_speedup:.2f}x average")
    else:
        print("⊘ Tiled vs non-tiled test skipped")
    
    if results.get('batching'):
        best_throughput = max([r['throughput'] for r in results['batching']])
        print(f"✓ Best throughput: {best_throughput:.0f} tokens/s")
    else:
        print("⊘ Batching test skipped")
    
    if results.get('memory'):
        max_memory = max([r['estimated_memory_mb'] for r in results['memory']])
        print(f"✓ Max memory usage: {max_memory:.2f} MB")
    else:
        print("⊘ Memory test skipped")
    
    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Check dependencies
    print("Checking dependencies...")
    print(f"PyTorch available: {HAS_TORCH}")
    print(f"MLX available: {HAS_MLX}")
    
    if not HAS_TORCH or not HAS_MLX:
        print("\nMissing dependencies. Please install:")
        if not HAS_TORCH:
            print("  - PyTorch: pip install torch")
        if not HAS_MLX:
            print("  - MLX: pip install mlx")
        sys.exit(1)
    
    # Run tests
    try:
        results = run_comprehensive_tests()
        print("\n✓ All tests completed successfully")
    except Exception as e:
        print(f"\n✗ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)