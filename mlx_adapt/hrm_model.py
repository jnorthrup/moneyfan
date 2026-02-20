"""
MLX implementation of HRM model.

MLX-specific model with ANE optimizations.
"""
import math
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any


@dataclass
class HRMConfig:
    """HRM configuration for MLX"""
    n_assets: int = 128
    n_features: int = 15
    n_models: int = 5
    seq_len: int = 32
    hidden_dim: int = 128
    n_heads: int = 4
    H_cycles: int = 2
    L_cycles: int = 2
    H_layers: int = 2
    L_layers: int = 2
    lr: float = 1e-4
    weight_decay: float = 0.1
    batch_size: int = 32
    
    @property
    def input_dim(self) -> int:
        return self.n_features
    
    @property
    def output_dim(self) -> int:
        return self.n_models


def trunc_normal_init(shape: tuple, std: float = 1.0,
                      dtype: mx.Dtype = mx.float32) -> mx.array:
    """Truncated normal initialization for MLX."""
    x = mx.random.normal(shape, dtype=dtype)
    x = mx.clip(x, -2.0, 2.0)
    return x * std


def rms_norm(x: mx.array, eps: float = 1e-5) -> mx.array:
    """RMS normalization."""
    return x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps)


class Linear(nn.Module):
    """Linear layer with truncated normal init, ANE-friendly."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = False):
        super().__init__()
        self.weight = trunc_normal_init((out_dim, in_dim), std=1.0 / math.sqrt(in_dim))
        self.bias = mx.zeros((out_dim,)) if bias else None

    def __call__(self, x: mx.array) -> mx.array:
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


class Attention(nn.Module):
    """Multi-head attention with RoPE — matmul-only, ANE-friendly."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = Linear(dim, 3 * dim)
        self.out = Linear(dim, dim)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)

        q = qkv[:, :, 0].transpose(0, 2, 1, 3)   # [B, H, T, head_dim]
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)

        # RoPE — cos/sin shape [T, head_dim], broadcast to [1, 1, T, head_dim]
        cos4 = cos[None, None, :, :]
        sin4 = sin[None, None, :, :]
        q = (q * cos4) + (self._rotate_half(q) * sin4)
        k = (k * cos4) + (self._rotate_half(k) * sin4)

        # Scaled dot-product attention (matmul, no einsum)
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(q.dtype)
        out = attn @ v                                # [B, H, T, head_dim]

        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        return self.out(out)

    def _rotate_half(self, x: mx.array) -> mx.array:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return mx.concatenate([-x2, x1], axis=-1)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward — ANE-friendly."""

    def __init__(self, dim: int, mult: float = 4.0):
        super().__init__()
        hidden = int(mult * dim * 2 / 3)
        hidden = (hidden + 255) // 256 * 256   # round up to 256 multiple
        self.gate_up = Linear(dim, 2 * hidden)
        self.down = Linear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        gate_up = self.gate_up(x)
        half = gate_up.shape[-1] // 2
        gate = gate_up[..., :half]
        up = gate_up[..., half:]
        return self.down(nn.silu(gate) * up)


class Block(nn.Module):
    """Transformer block: pre-norm attention + SwiGLU FFN."""

    def __init__(self, dim: int, n_heads: int, mult: float = 4.0):
        super().__init__()
        self.attn = Attention(dim, n_heads)
        self.ffn = SwiGLU(dim, mult)
        self.eps = 1e-5

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x = rms_norm(x + self.attn(x, cos, sin), self.eps)
        x = rms_norm(x + self.ffn(x), self.eps)
        return x


class RoPE(nn.Module):
    """Rotary Position Embedding — precomputed, fixed seq_len (ANE-friendly)."""

    def __init__(self, head_dim: int, max_seq: int, base: float = 10000.0):
        super().__init__()
        freqs = 1.0 / (base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
        pos = mx.arange(max_seq, dtype=mx.float32)
        angles = mx.outer(pos, freqs)                       # [max_seq, head_dim//2]
        full = mx.concatenate([angles, angles], axis=-1)    # [max_seq, head_dim]
        self._cos = mx.cos(full).astype(mx.float32)
        self._sin = mx.sin(full).astype(mx.float32)

    def __call__(self, seq_len: int) -> Tuple[mx.array, mx.array]:
        return self._cos[:seq_len], self._sin[:seq_len]


class ReasoningModule(nn.Module):
    """
    Stack of transformer Blocks with input injection.
    """

    def __init__(self, dim: int, n_heads: int, n_layers: int, mult: float = 4.0):
        super().__init__()
        self.n_layers = n_layers
        for i in range(n_layers):
            setattr(self, f"layer_{i}", Block(dim, n_heads, mult))

    def __call__(self, x: mx.array, injection: mx.array,
                 cos: mx.array, sin: mx.array) -> mx.array:
        x = x + injection
        for i in range(self.n_layers):
            x = getattr(self, f"layer_{i}")(x, cos, sin)
        return x


class HRMInner(nn.Module):
    """
    Full H+L nested recurrent model for MLX.
    """

    def __init__(self, config: HRMConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.H_cycles = config.H_cycles
        self.L_cycles = config.L_cycles

        input_dim = config.input_dim

        self.input_proj = Linear(input_dim, config.hidden_dim)
        self.embed_scale = math.sqrt(config.hidden_dim)

        self.rope = RoPE(config.hidden_dim // config.n_heads, config.seq_len)

        self.H_level = ReasoningModule(config.hidden_dim, config.n_heads,
                                       config.H_layers)
        self.L_level = ReasoningModule(config.hidden_dim, config.n_heads,
                                       config.L_layers)

        self.output_head = Linear(config.hidden_dim, config.n_models, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [B, T, input_dim]  float32

        Returns:
            model_outputs: [B, n_models]  float32
        """
        B, T, _ = x.shape

        # 1. Project input
        x_emb = self.embed_scale * self.input_proj(x).astype(mx.float32)

        # 2. RoPE embeddings
        cos, sin = self.rope(T)

        # 3. Init hidden states
        z_H = mx.zeros((B, T, self.hidden_dim), dtype=mx.float32)
        z_L = mx.zeros((B, T, self.hidden_dim), dtype=mx.float32)

        # 4. Nested recurrent loop
        total_H = self.H_cycles
        for h in range(total_H):
            is_last_H = (h == total_H - 1)
            total_L = self.L_cycles
            for l in range(total_L):
                is_last_L = (l == total_L - 1)
                is_final = is_last_H and is_last_L

                new_z_L = self.L_level(z_L, z_H + x_emb, cos, sin)
                if not is_final:
                    new_z_L = mx.stop_gradient(new_z_L)
                z_L = new_z_L

            new_z_H = self.H_level(z_H, z_L, cos, sin)
            if not is_last_H:
                new_z_H = mx.stop_gradient(new_z_H)
            z_H = new_z_H

        # 5. Pool over T dimension -> [B, hidden_dim]
        pooled = z_H.mean(axis=1)

        # 6. Output head -> [B, n_models]
        logits = self.output_head(pooled)
        return logits


class HRMModel(nn.Module):
    """
    Public API wrapper for HRM MLX model.
    """

    def __init__(self, config: HRMConfig):
        super().__init__()
        self.config = config
        self.inner = HRMInner(config)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [B, T, input_dim]

        Returns:
            model_outputs: [B, n_models]
        """
        return self.inner(x)


class HRMTrainer:
    """
    Trainer for HRM MLX model.
    """

    def __init__(self, config: HRMConfig):
        self.config = config
        self.model = HRMModel(config)
        self.optimizer = optim.AdamW(
            learning_rate=config.lr,
            weight_decay=config.weight_decay,
        )

    def train_step(self, x: mx.array, targets: mx.array) -> mx.array:
        """
        Single training step.
        
        Args:
            x: Input features [B, T, input_dim]
            targets: Target outputs [B, n_models]
            
        Returns:
            loss: Scalar loss
        """
        def loss_fn(model_):
            predictions = model_(x)
            return mx.mean((predictions - targets) ** 2)

        loss, grads = mx.value_and_grad(loss_fn)(self.model)
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state)
        return loss