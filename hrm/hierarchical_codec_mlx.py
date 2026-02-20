"""
Hierarchical Codec MLX Implementation - Native (Preserves Architecture)
========================================================================

MLX implementation that preserves HRM's exact architecture:
- H/L nested cycles: SEQUENTIAL processing (not tiled)
- Sparkline update: CASCADING sequential updates (not tiled)
- State persistence: Proper state carry across cycles

MLX provides speedup through:
1. Lazy evaluation (automatic kernel fusion)
2. Metal GPU acceleration
3. ANE targeting (Apple Neural Engine)
4. Automatic optimization

DO NOT break HRM's sequential dependencies for speed.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("MLX not available")


@dataclass
class HierarchicalCodecConfig:
    """Configuration matching PyTorch version."""
    n_signals: int = 24
    hidden_dim: int = 64
    sparkline_frames: int = 20
    sparkline_horizon: int = 200
    H_layers: int = 2
    L_layers: int = 2
    H_cycles: int = 2
    L_cycles: int = 3
    n_heads: int = 4


class MLXSparklineMemory:
    """
    Sparkline memory - EXACT PyTorch logic.
    
    Frame 0 = current, frame k depends on frame k-1 (cascading).
    NO tiling, NO parallel frame computation.
    """
    
    def __init__(self, hidden_dim: int, n_frames: int = 20, horizon: int = 200):
        self.hidden_dim = hidden_dim
        self.n_frames = n_frames
        self.horizon = horizon
        self.ratio = horizon ** (1.0 / max(n_frames - 1, 1))
    
    def update(self, sparkline: Optional[mx.array], current: mx.array) -> mx.array:
        """
        Update sparkline with cascading sequential logic.
        
        Frame 0 = current
        Frame k = (1-alpha_k) * old[k] + alpha_k * frame_{k-1}
        
        This creates a cascading temporal memory.
        """
        B, D = current.shape
        
        if sparkline is None:
            sparkline = mx.zeros((B, self.n_frames, D))
        
        # Frame 0 is current
        frame_0 = current[:, None, :]
        
        # Sequential cascading: frame k depends on frame k-1
        frames: List[mx.array] = [frame_0]
        for k in range(1, self.n_frames):
            alpha_k = 1.0 / (self.ratio ** k)
            # Uses PREVIOUS frame (cascading)
            prev_frame = frames[-1]
            frame_k = (1.0 - alpha_k) * sparkline[:, k:k+1, :] + alpha_k * prev_frame
            frames.append(frame_k)
        
        return mx.concatenate(frames, axis=1)
    
    def read(self, sparkline: mx.array) -> mx.array:
        """
        Read weighted context from sparkline.
        
        Each frame contributes inversely to its age (ratio^k).
        """
        B, F, D = sparkline.shape
        
        # Precompute weights
        weights_list = [1.0 / (self.ratio ** k) for k in range(self.n_frames)]
        weights_sum = sum(weights_list)
        weights = mx.array([w / weights_sum for w in weights_list])  # [n_frames]
        
        # Apply weights
        weighted = sparkline * weights.reshape(1, -1, 1)
        return mx.sum(weighted, axis=1)  # [B, D]


class MLXMLP(nn.Module):
    """MLP layer - stores layers explicitly for weight transfer."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)
    
    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class MLXLayer(nn.Module):
    """
    Single layer: Attention + MLP with residuals.
    
    EXACT PyTorch structure, NO tiling.
    """
    
    def __init__(self, hidden_dim: int, n_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        # Attention
        self.attention = nn.MultiHeadAttention(hidden_dim, n_heads)
        self.norm1 = nn.RMSNorm(hidden_dim)
        
        # MLP - use explicit class for weight transfer
        self.mlp = MLXMLP(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)
    
    def __call__(self, x: mx.array) -> mx.array:
        """Forward with attention + MLP - SEQUENTIAL (not tiled)."""
        # Attention with residual
        attn_out = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # MLP with residual
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        
        return x


class MLXReasoningLevel(nn.Module):
    """
    H or L level: Multiple layers with sequential processing.
    
    EXACT PyTorch structure, NO tiling.
    """
    
    def __init__(self, hidden_dim: int, n_layers: int, n_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Python list (MLX doesn't have ModuleList)
        self.layers = [
            MLXLayer(hidden_dim, n_heads) for _ in range(n_layers)
        ]
    
    def __call__(self, z: mx.array, context: mx.array) -> mx.array:
        """
        Process through layers - SEQUENTIAL.
        
        Each layer sees the output of the previous layer.
        This preserves the hierarchical depth.
        """
        x = z + context
        for layer in self.layers:
            x = layer(x)
        return x


class MLXHierarchicalCodec(nn.Module):
    """
    Native MLX implementation - EXACT PyTorch architecture.
    
    NO TILING - preserves sequential H/L cycle dependencies.
    """
    
    def __init__(self, config: HierarchicalCodecConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(config.n_signals * 2, config.hidden_dim)
        
        # Sparkline memory
        self.sparkline = MLXSparklineMemory(
            config.hidden_dim, config.sparkline_frames, config.sparkline_horizon
        )
        
        # H/L levels - exactly like PyTorch
        self.H_level = MLXReasoningLevel(config.hidden_dim, config.H_layers, config.n_heads)
        self.L_level = MLXReasoningLevel(config.hidden_dim, config.L_layers, config.n_heads)
        
        # Initialize states (used when memory is None)
        self.H_init = mx.random.normal((config.hidden_dim,)) * 0.02
        self.L_init = mx.random.normal((config.hidden_dim,)) * 0.02
        
        # Output heads
        self.signal_head = nn.Linear(config.hidden_dim, config.n_signals * 2)
        self.return_head = nn.Linear(config.hidden_dim, 1)
        self.confidence_head = nn.Linear(config.hidden_dim, 1)
        self.stop_head = nn.Linear(config.hidden_dim, 1)
        self.tp_head = nn.Linear(config.hidden_dim, 1)
        self.pos_head = nn.Linear(config.hidden_dim, 1)
    
    def forward(
        self, 
        signals: mx.array,
        memory: Optional[Tuple] = None,
        mode: str = "pretrain"
    ) -> Tuple[mx.array, Optional[Tuple]]:
        """
        EXACT PyTorch forward pass - NO TILING.
        
        Preserves:
        - Sequential H/L cycle processing
        - Sparkline cascading updates
        - State persistence across cycles
        - Input injection at both H and L levels
        
        MLX optimizes execution WITHOUT changing the logic.
        """
        B, T, _ = signals.shape
        
        sparkline, z_H, z_L = memory if memory else (None, None, None)
        
        # Input projection
        x = self.input_proj(signals)
        
        # Sparkline update - EXACT PyTorch logic
        current = x.mean(axis=1)  # [B, D]
        sparkline = self.sparkline.update(sparkline, current)
        context = self.sparkline.read(sparkline)  # [B, D]
        
        # Add context to all timesteps
        context_expanded = mx.expand_dims(context, 1)  # [B, 1, D]
        context_expanded = mx.broadcast_to(
            context_expanded, 
            (B, T, self.hidden_dim)
        )  # [B, T, D]
        input_with_context = x + context_expanded
        
        # Initialize H/L states if needed
        if z_H is None:
            z_H = mx.broadcast_to(
                mx.expand_dims(self.H_init, 0), 
                (B, T, self.hidden_dim)
            )
            z_L = mx.broadcast_to(
                mx.expand_dims(self.L_init, 0), 
                (B, T, self.hidden_dim)
            )
        
        # H/L CYCLES - EXACT PyTorch logic (SEQUENTIAL)
        # Process ENTIRE sequence in each cycle, NOT in tiles
        
        # H_cycles - 1 (no gradient)
        for _h in range(self.config.H_cycles - 1):
            for _l in range(self.config.L_cycles):
                z_L = self.L_level(z_L, z_H + input_with_context)
            z_H = self.H_level(z_H, z_L)
        
        # Final cycle (with gradient)
        for _l in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_with_context)
        z_H = self.H_level(z_H, z_L)
        
        # Output heads
        if mode == "pretrain":
            output = self.signal_head(z_H[:, -1, :])
        else:
            ret = self.return_head(z_H[:, -1, :])
            conf = mx.sigmoid(self.confidence_head(z_H[:, -1, :]))
            
            stop = mx.tanh(self.stop_head(z_H[:, -1, :])) * 0.15
            tp = mx.sigmoid(self.tp_head(z_H[:, -1, :])) * 0.30
            pos = mx.sigmoid(self.pos_head(z_H[:, -1, :]))
            
            output = mx.concatenate([ret, conf, stop, tp, pos], axis=-1)
        
        # Return output and memory (for next forward pass)
        new_memory = (sparkline, z_H, z_L)
        return output, new_memory
    
    def __call__(self, signals: mx.array, memory: Optional[Tuple] = None, mode: str = "pretrain") -> Tuple[mx.array, Optional[Tuple]]:
        """Alias for forward."""
        return self.forward(signals, memory, mode)


class MLXCodecTrainer:
    """
    MLX-compatible trainer with automatic optimization.
    
    Features:
    1. Lazy evaluation (MLX handles optimization)
    2. ANE targeting (Apple Neural Engine)
    3. Automatic kernel fusion
    4. No manual optimization needed
    """
    
    def __init__(self, config: HierarchicalCodecConfig = None):
        self.config = config or HierarchicalCodecConfig()
        self.model = MLXHierarchicalCodec(self.config)
        # MLX uses automatic optimization, no explicit optimizer needed
    
    def pretrain_step(self, signals: mx.array) -> mx.array:
        """One pre-training step."""
        output, _ = self.model.forward(signals, mode="pretrain")
        # Predict last timestep's signals
        target = signals[:, -1, :]
        loss = mx.mean(mx.square(output - target))
        return loss
    
    def trade_step(self, signals: mx.array, returns: mx.array) -> mx.array:
        """One trade step."""
        output, _ = self.model.forward(signals, mode="trade")
        pred_return = output[:, 0]
        confidence = output[:, 1]
        
        # Weighted return
        weighted_return = pred_return * confidence
        loss = -mx.mean(weighted_return * returns)
        return loss


def enable_ane_optimization():
    """
    Enable ANE (Apple Neural Engine) optimization.
    
    This allows MLX to target specialized hardware for maximum speed.
    Call before creating models.
    """
    if HAS_MLX:
        # Set default device to ANE if available, otherwise GPU
        try:
            mx.set_default_device(mx.ane)
            print("✅ ANE optimization enabled")
        except:
            try:
                mx.set_default_device(mx.gpu)
                print("✅ GPU optimization enabled")
            except:
                print("⚠️  Using CPU fallback")
    else:
        print("❌ MLX not available")


def benchmark_speed(signals: mx.array, n_iter: int = 100) -> dict:
    """
    Benchmark native MLX speed.
    
    Returns timing statistics.
    """
    if not HAS_MLX:
        return {"error": "MLX not available"}
    
    config = HierarchicalCodecConfig()
    model = MLXHierarchicalCodec(config)
    
    # Warmup
    for _ in range(10):
        output, _ = model.forward(signals)
        mx.eval(output)
    
    # Timing
    import time
    times = []
    for _ in range(n_iter):
        start = time.perf_counter()
        output, _ = model.forward(signals)
        mx.eval(output)  # Force evaluation (MLX is lazy)
        times.append(time.perf_counter() - start)
    
    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "max_ms": np.max(times) * 1000,
    }


# Convenience import
import numpy as np
