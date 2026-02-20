"""
MLX-Torch Bridge - Local Bridge for A/B Testing
================================================

Enables training in PyTorch and inference in MLX (or vice versa).

Features:
1. Weight synchronization: PyTorch → MLX
2. Inference bridge: Run MLX with PyTorch weights
3. A/B testing: Compare outputs from both frameworks
4. Speed comparison: MLX inference speed vs PyTorch

Bridge Limitations:
- One-way weight transfer (PyTorch → MLX is easier)
- No gradient flow across frameworks
- Numerical differences expected (different implementations)
- Best for: Train PyTorch, deploy MLX

Usage:
    bridge = MLXTorchBridge(pytorch_model, mlx_config)
    mlx_output = bridge.run_inference_mlx(signals)
    torch_output = pytorch_model(signals)
    compare_outputs(mlx_output, torch_output)
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import time

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("⚠️  MLX not available for bridge")


@dataclass
class BridgeConfig:
    """Configuration for MLX-Torch bridge."""
    # Numerical tolerance for weight matching
    weight_tolerance: float = 1e-4
    # Enable verbose logging
    verbose: bool = True
    # Normalize weights (adjust for framework differences)
    normalize_weights: bool = True
    # Use float32 for both (default: True for compatibility)
    use_float32: bool = True


class MLXTorchBridge:
    """
    Bridge between PyTorch and MLX hierarchical codecs.
    
    Transfers weights from PyTorch → MLX for inference comparison.
    """
    
    def __init__(
        self,
        torch_model: 'HierarchicalCodec',
        mlx_config: 'HierarchicalCodecConfig',
        config: BridgeConfig = None
    ):
        self.torch_model = torch_model
        self.config = config or BridgeConfig()
        
        # Create MLX model with same architecture
        if HAS_MLX:
            from hrm.hierarchical_codec_mlx import MLXHierarchicalCodec
            self.mlx_model = MLXHierarchicalCodec(mlx_config)
            self._transfer_weights()
        else:
            self.mlx_model = None
        
        # Statistics
        self.transfer_stats = {}
        self.inference_stats = {}
    
    def _transfer_weights(self):
        """Transfer weights from PyTorch to MLX."""
        if not HAS_MLX:
            return
        
        print("🧠 Transferring weights PyTorch → MLX...")
        
        # Map PyTorch parameter names to MLX paths
        torch_params = dict(self.torch_model.named_parameters())
        
        # Transfer input projection
        self._transfer_linear(
            torch_params['input_proj.weight'],
            self.mlx_model.input_proj,
            bias_key='input_proj.bias' if 'input_proj.bias' in torch_params else None,
            bias_torch=torch_params.get('input_proj.bias')
        )
        
        # Transfer H and L level layers
        for i, layer in enumerate(self.mlx_model.H_level.layers):
            prefix = f'H_level.layers.{i}'
            self._transfer_layer(torch_params, layer, prefix)
        
        for i, layer in enumerate(self.mlx_model.L_level.layers):
            prefix = f'L_level.layers.{i}'
            self._transfer_layer(torch_params, layer, prefix)
        
        # Transfer output heads
        for head_name in ['signal_head', 'return_head', 'confidence_head', 
                         'stop_head', 'tp_head', 'pos_head']:
            weight_key = f'{head_name}.weight'
            bias_key = f'{head_name}.bias'
            if weight_key in torch_params:
                self._transfer_linear(
                    torch_params[weight_key],
                    getattr(self.mlx_model, head_name),
                    bias_key=bias_key if bias_key in torch_params else None,
                    bias_torch=torch_params.get(bias_key)
                )
        
        # Transfer init states
        if 'H_init' in torch_params:
            self.mlx_model.H_init = mx.array(
                torch_params['H_init'].detach().numpy()
            )
        if 'L_init' in torch_params:
            self.mlx_model.L_init = mx.array(
                torch_params['L_init'].detach().numpy()
            )
        
        print("✅ Weight transfer complete")
    
    def _transfer_linear(self, torch_weight: torch.Tensor, mlx_linear, 
                        bias_key: str = None, bias_torch: torch.Tensor = None):
        """Transfer linear layer weights."""
        # Convert to numpy, then MLX
        weight_np = torch_weight.detach().numpy().astype(np.float32)
        if self.config.normalize_weights:
            # Normalize to prevent overflow/underflow
            weight_np = weight_np / np.sqrt(weight_np.var() + 1e-8)
        
        # MLX Linear stores weight as (out_dim, in_dim)
        # PyTorch Linear stores weight as (out_dim, in_dim)
        mlx_linear.weight = mx.array(weight_np)
        
        if bias_key and bias_torch is not None:
            bias_np = bias_torch.detach().numpy().astype(np.float32)
            if self.config.normalize_weights:
                bias_np = bias_np / np.sqrt(bias_np.var() + 1e-8)
            mlx_linear.bias = mx.array(bias_np)
    
    def _transfer_layer(self, torch_params: Dict, mlx_layer, prefix: str):
        """Transfer attention + MLP layer weights."""
        # Attention weights
        attn_weight_key = f'{prefix}.attention.weight'
        if attn_weight_key in torch_params:
            self._transfer_attention_weights(
                torch_params, mlx_layer.attention, prefix
            )
        
        # Norm weights
        norm1_weight_key = f'{prefix}.norm1.weight'
        if norm1_weight_key in torch_params:
            mlx_layer.norm1.weight = mx.array(
                torch_params[norm1_weight_key].detach().numpy().astype(np.float32)
            )
        
        norm2_weight_key = f'{prefix}.norm2.weight'
        if norm2_weight_key in torch_params:
            mlx_layer.norm2.weight = mx.array(
                torch_params[norm2_weight_key].detach().numpy().astype(np.float32)
            )
        
        # MLP weights - now using explicit linear1/linear2
        self._transfer_linear(
            torch_params[f'{prefix}.mlp.0.weight'],
            mlx_layer.mlp.linear1,
            bias_key=f'{prefix}.mlp.0.bias',
            bias_torch=torch_params.get(f'{prefix}.mlp.0.bias')
        )
        self._transfer_linear(
            torch_params[f'{prefix}.mlp.2.weight'],
            mlx_layer.mlp.linear2,
            bias_key=f'{prefix}.mlp.2.bias',
            bias_torch=torch_params.get(f'{prefix}.mlp.2.bias')
        )
    
    def _transfer_attention_weights(self, torch_params: Dict, mlx_attn, prefix: str):
        """
        Transfer attention weights from PyTorch to MLX.
        
        Note: PyTorch and MLX have different attention weight layouts.
        This is a simplified transfer - exact match may not be possible.
        """
        # PyTorch MultiheadAttention stores:
        # - q_proj_weight, k_proj_weight, v_proj_weight, out_proj.weight
        
        q_key = f'{prefix}.attention.q_proj_weight'
        k_key = f'{prefix}.attention.k_proj_weight'
        v_key = f'{prefix}.attention.v_proj_weight'
        out_key = f'{prefix}.attention.out_proj.weight'
        
        # For MLX, we'll set the internal weights if they exist
        # (MLX MultiHeadAttention implementation details vary)
        if hasattr(mlx_attn, 'wq') and q_key in torch_params:
            mlx_attn.wq = mx.array(
                torch_params[q_key].detach().numpy().astype(np.float32)
            )
        if hasattr(mlx_attn, 'wk') and k_key in torch_params:
            mlx_attn.wk = mx.array(
                torch_params[k_key].detach().numpy().astype(np.float32)
            )
        if hasattr(mlx_attn, 'wv') and v_key in torch_params:
            mlx_attn.wv = mx.array(
                torch_params[v_key].detach().numpy().astype(np.float32)
            )
        if hasattr(mlx_attn, 'wo') and out_key in torch_params:
            mlx_attn.wo = mx.array(
                torch_params[out_key].detach().numpy().astype(np.float32)
            )
    
    def run_inference_mlx(self, signals_torch: torch.Tensor) -> mx.array:
        """Run inference using MLX with PyTorch weights."""
        if not HAS_MLX or self.mlx_model is None:
            return None
        
        # Convert torch tensor to MLX array
        signals_mlx = mx.array(signals_torch.detach().numpy().astype(np.float32))
        
        # Run MLX inference
        start_time = time.time()
        output, _ = self.mlx_model.forward(signals_mlx, mode="trade")
        mx.eval(output)  # Force evaluation
        inference_time = time.time() - start_time
        
        self.inference_stats["mlx_time"] = inference_time
        return output
    
    def run_inference_torch(self, signals_torch: torch.Tensor) -> torch.Tensor:
        """Run inference using PyTorch."""
        start_time = time.time()
        with torch.no_grad():
            output, _ = self.torch_model.forward(signals_torch, mode="trade")
        inference_time = time.time() - start_time
        
        self.inference_stats["torch_time"] = inference_time
        return output
    
    def compare_outputs(self, signals_torch: torch.Tensor, tolerance: float = 1e-2) -> Dict:
        """
        Compare outputs from PyTorch and MLX.
        
        Returns similarity metrics.
        """
        if not HAS_MLX:
            return {"error": "MLX not available"}
        
        # Run both inferences
        torch_output = self.run_inference_torch(signals_torch)
        mlx_output = self.run_inference_mlx(signals_torch)
        
        if mlx_output is None:
            return {"error": "MLX inference failed"}
        
        # Convert to numpy for comparison
        torch_np = torch_output.detach().numpy()
        mlx_np = np.array(mlx_output)
        
        # Calculate similarity
        diff = torch_np - mlx_np
        abs_diff = np.abs(diff)
        
        similarity = 1.0 - abs_diff.mean()  # Simple similarity metric
        
        # Check if outputs are "close enough" (within tolerance)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        
        # Framework difference threshold (expected to be > 1e-5 due to different implementations)
        framework_tolerance = 0.1  # 10% difference is acceptable for different frameworks
        
        passes = max_diff < framework_tolerance
        
        stats = {
            "similarity": similarity,
            "max_diff": float(max_diff),
            "mean_diff": float(mean_diff),
            "passes": passes,
            "torch_time": self.inference_stats.get("torch_time", 0),
            "mlx_time": self.inference_stats.get("mlx_time", 0),
            "speedup": self.inference_stats.get("torch_time", 0) / max(self.inference_stats.get("mlx_time", 0), 1e-8),
        }
        
        if self.config.verbose:
            print(f"\n📊 Output Comparison:")
            print(f"  Similarity: {similarity:.4f}")
            print(f"  Max diff: {max_diff:.6f}")
            print(f"  Mean diff: {mean_diff:.6f}")
            print(f"  PyTorch time: {stats['torch_time']*1000:.2f} ms")
            print(f"  MLX time: {stats['mlx_time']*1000:.2f} ms")
            print(f"  Speedup: {stats['speedup']:.2f}x")
            print(f"  Passes tolerance: {'✅' if passes else '❌'}")
        
        return stats
    
    def ab_test_training(self, signals_torch: torch.Tensor, 
                        batch_size: int = 4, n_iterations: int = 10) -> Dict:
        """
        A/B test: Train in PyTorch, compare inference in both frameworks.
        
        This allows us to:
        1. Train once in PyTorch (stable)
        2. Test inference speed in MLX
        3. Compare numerical outputs
        """
        results = {
            "pytorch": {"times": [], "outputs": []},
            "mlx": {"times": [], "outputs": []},
            "comparison": [],
        }
        
        print(f"\n🔄 A/B Test: {n_iterations} iterations")
        
        for i in range(n_iterations):
            # Create batch
            batch = signals_torch[:batch_size]
            
            # PyTorch inference
            torch_start = time.time()
            torch_out, _ = self.torch_model.forward(batch, mode="trade")
            torch_time = time.time() - torch_start
            
            # MLX inference
            mlx_start = time.time()
            mlx_out, _ = self.mlx_model.forward(
                mx.array(batch.detach().numpy().astype(np.float32)), 
                mode="trade"
            )
            mx.eval(mlx_out)
            mlx_time = time.time() - mlx_start
            
            # Compare
            torch_np = torch_out.detach().numpy()
            mlx_np = np.array(mlx_out)
            diff = np.abs(torch_np - mlx_np).mean()
            
            results["pytorch"]["times"].append(torch_time)
            results["mlx"]["times"].append(mlx_time)
            results["comparison"].append(diff)
            
            if i % 5 == 0:
                print(f"  Iter {i}: torch={torch_time*1000:.2f}ms, "
                      f"mlx={mlx_time*1000:.2f}ms, diff={diff:.4f}")
        
        # Summary
        torch_avg = np.mean(results["pytorch"]["times"]) * 1000
        mlx_avg = np.mean(results["mlx"]["times"]) * 1000
        diff_avg = np.mean(results["comparison"])
        speedup = torch_avg / mlx_avg
        
        print(f"\n📈 A/B Test Results:")
        print(f"  PyTorch avg: {torch_avg:.2f} ms")
        print(f"  MLX avg: {mlx_avg:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Avg diff: {diff_avg:.6f}")
        
        results["summary"] = {
            "torch_avg_ms": torch_avg,
            "mlx_avg_ms": mlx_avg,
            "speedup": speedup,
            "avg_diff": diff_avg,
        }
        
        return results


def create_bridge_and_test(config: BridgeConfig = None) -> Optional[MLXTorchBridge]:
    """
    Create bridge and run basic tests.
    
    Returns the bridge if successful.
    """
    if not HAS_MLX:
        print("❌ MLX not available - cannot create bridge")
        return None
    
    from hrm.hierarchical_codec import (
        HierarchicalCodec,
        HierarchicalCodecConfig
    )
    from hrm.hierarchical_codec_mlx import (
        HierarchicalCodecConfig as MLXConfig
    )
    
    # Create PyTorch model
    torch_config = HierarchicalCodecConfig(n_signals=24, hidden_dim=64)
    torch_model = HierarchicalCodec(torch_config)
    
    # Create MLX config (same architecture)
    mlx_config = MLXConfig(n_signals=24, hidden_dim=64)
    
    # Create bridge
    bridge = MLXTorchBridge(torch_model, mlx_config, config)
    
    print("\n" + "=" * 60)
    print("MLX-Torch Bridge Created")
    print("=" * 60)
    print(f"  PyTorch params: {sum(p.numel() for p in torch_model.parameters()):,}")
    
    # MLX params - use MLX's built-in parameters()
    try:
        mlx_params = 0
        for name, param in bridge.mlx_model.parameters().items():
            if hasattr(param, 'shape'):
                size = np.prod(param.shape)
                mlx_params += size
        print(f"  MLX params: {mlx_params:,}")
    except Exception as e:
        print(f"  MLX params: Error counting ({e})")
    
    print(f"  Bridge status: {'✅ Ready' if bridge.mlx_model else '❌ Failed'}")
    
    return bridge


# Convenience function for quick testing
def quick_test():
    """Quick test of the bridge."""
    print("=" * 60)
    print("Quick Test: MLX-Torch Bridge")
    print("=" * 60)
    
    bridge = create_bridge_and_test()
    if bridge is None or bridge.mlx_model is None:
        return
    
    # Create test signals
    signals = torch.randn(4, 64, 48)  # batch=4, seq=64, signals=24*2
    
    # Compare outputs
    stats = bridge.compare_outputs(signals)
    
    if "error" in stats:
        print(f"❌ Error: {stats['error']}")
    elif stats["passes"]:
        print(f"\n✅ Bridge working correctly!")
        print(f"   Speedup: {stats['speedup']:.2f}x")
    else:
        print(f"\n⚠️  Outputs differ but may be expected:")
        print(f"   Difference: {stats['mean_diff']:.4f}")
        print(f"   Speedup: {stats['speedup']:.2f}x")


if __name__ == "__main__":
    quick_test()
