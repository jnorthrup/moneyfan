"""
HRM MLX Implementation — Apple Silicon / ANE
Hierarchical Reasoning Model targeting Apple Neural Engine via MLX.

Architecture:
  - Two recurrent modules: H_level (high-level, slow) and L_level (low-level, fast)
  - Nested loop: H_cycles × L_cycles with stop_gradient on non-final iterations
  - Input injection at each recurrence step
  - Output: softmax over n_regimes (regime weights for portfolio allocation)

ANE optimizations:
  - float32 everywhere (no float64)
  - Fixed seq_len (no dynamic shapes)
  - matmul-based attention (no einsum)
  - Batch size >= 1

Usage:
  from hrm.apple.hrm import HRMModel
  from hrm.apple.config import UNDERFIT_HRM
  model = HRMModel(UNDERFIT_HRM)
  out = model(mx.zeros((2, 16, config.n_assets * config.n_features)))
  # out.shape == [2, 6]
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pathlib
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

# Force ANE-friendly device (MLX uses ANE/GPU unified memory)
mx.set_default_device(mx.gpu)

N_REGIMES = 6  # TREND, MEAN_REVERSION, VOLATILITY, STAT_ARB, SYSTEMATIC, ML


# =============================================================================
# INITIALIZATION & UTILS
# =============================================================================

def trunc_normal_init(shape: tuple, std: float = 1.0,
                      dtype: mx.Dtype = mx.float32) -> mx.array:
    """Truncated normal initialization for MLX."""
    x = mx.random.normal(shape, dtype=dtype)
    x = mx.clip(x, -2.0, 2.0)
    return x * std


def rms_norm(x: mx.array, eps: float = 1e-5) -> mx.array:
    """RMS normalization."""
    return x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps)


# =============================================================================
# LAYERS
# =============================================================================

class Linear(nn.Module):
    """Linear layer with truncated normal init, ANE-friendly (no dynamic shapes)."""

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


# =============================================================================
# REASONING MODULE
# =============================================================================

class ReasoningModule(nn.Module):
    """
    Stack of transformer Blocks with input injection.

    At each call:
      x  = x + injection          (input injection)
      x  = Block_0(x) ... Block_n(x)
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


# =============================================================================
# HRM INNER — NESTED RECURRENT CORE
# =============================================================================

class HRMInner(nn.Module):
    """
    Full H+L nested recurrent model.

    Forward:
      1. Project input -> x_emb [B, T, hidden_dim]
      2. Init z_H, z_L as zeros [B, T, hidden_dim]
      3. Nested loop: H_cycles × L_cycles
           - All but final iteration use stop_gradient
      4. Pool over T (mean) -> [B, hidden_dim]
      5. Output head -> softmax -> [B, n_regimes]
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.H_cycles = config.H_cycles
        self.L_cycles = config.L_cycles

        input_dim = config.n_assets * config.n_features

        self.input_proj = Linear(input_dim, config.hidden_dim)
        self.embed_scale = math.sqrt(config.hidden_dim)

        self.rope = RoPE(config.hidden_dim // config.n_heads, config.seq_len)

        self.H_level = ReasoningModule(config.hidden_dim, config.n_heads,
                                       config.H_layers)
        self.L_level = ReasoningModule(config.hidden_dim, config.n_heads,
                                       config.L_layers)

        self.output_head = Linear(config.hidden_dim, N_REGIMES, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [B, T, n_assets * n_features]  float32

        Returns:
            regime_weights: [B, N_REGIMES]  float32, sums to 1
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

        # 6. Output head -> softmax
        logits = self.output_head(pooled)
        return mx.softmax(logits.astype(mx.float32), axis=-1)


# =============================================================================
# HRM MODEL — PUBLIC API
# =============================================================================

class HRMModel(nn.Module):
    """
    Public API wrapper for HRM.

    Usage:
        model = HRMModel(config)
        regime_weights = model(x)  # [B, 6]
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.inner = HRMInner(config)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [B, T, n_assets * n_features]

        Returns:
            regime_weights: [B, 6]  (softmax, sums to 1)
        """
        return self.inner(x)


# =============================================================================
# LOSS FUNCTION
# =============================================================================

def portfolio_loss(regime_weights: mx.array,
                   signals_tensor: mx.array,
                   returns_tensor: mx.array) -> mx.array:
    """
    Portfolio loss: negative return + turnover penalty.

    Args:
        regime_weights: [B, 6]  softmax regime weights
        signals_tensor: [B, 6]  pre-aggregated confidence-weighted signal per regime
        returns_tensor: [B]     actual next-bar returns

    Returns:
        scalar loss
    """
    # Combine regime weights with signal directions -> [B]
    alpha = (regime_weights * signals_tensor).sum(axis=-1)
    pnl = alpha * returns_tensor

    # Turnover penalty (consecutive batch items represent time steps)
    if regime_weights.shape[0] > 1:
        turnover = mx.abs(regime_weights[1:] - regime_weights[:-1]).mean()
    else:
        turnover = mx.array(0.0, dtype=mx.float32)

    return -pnl.mean() + 0.1 * turnover


# =============================================================================
# TRAINING STEP
# =============================================================================

def train_step(model: HRMModel,
               optimizer,
               x: mx.array,
               signals: mx.array,
               returns: mx.array) -> mx.array:
    """
    Single gradient update step.

    Args:
        model:     HRMModel
        optimizer: mlx.optimizers instance
        x:         [B, T, input_dim]
        signals:   [B, 6]
        returns:   [B]

    Returns:
        loss scalar
    """
    def loss_fn(model_):
        weights = model_(x)
        return portfolio_loss(weights, signals, returns)

    loss, grads = mx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    return loss


# =============================================================================
# HRM TRAINER
# =============================================================================

class HRMTrainer:
    """
    End-to-end trainer: loads data from DuckDB/Arrow, prepares batches, trains HRM.

    Usage:
        trainer = HRMTrainer(config, db_path="hrm/data/coinbase.duckdb")
        trainer.load_training_data()
        trainer.train(epochs=100)
        trainer.save_checkpoint("hrm/checkpoints/hrm_ane_v1.npz")
    """

    REGIME_NAMES = ["trend", "mean_reversion", "volatility",
                    "stat_arb", "systematic", "ml"]

    def __init__(self, config, db_path: str = "hrm/data/coinbase.duckdb"):
        self.config = config
        self.db_path = pathlib.Path(db_path) if db_path else None
        self.model = HRMModel(config)
        self.optimizer = optim.AdamW(
            learning_rate=config.lr,
            weight_decay=config.weight_decay,
        )
        self._candles_df = None
        self._features_df = None
        self._signals_df = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_training_data(self):
        """Load candles from DuckDB/Arrow and compute features + signals."""
        import pandas as pd
        from hrm.pipeline import load_candles, compute_features, compute_all_signals

        print(f"Loading candles from {self.db_path or 'DuckDB/Arrow'}...")
        self._candles_df = load_candles(self.db_path)
        n = len(self._candles_df)
        print(f"  Loaded {n:,} candle rows.")

        print("Computing features ...")
        self._features_df = compute_features(self._candles_df)
        print(f"  Features shape: {self._features_df.shape}")

        print("Computing signals ...")
        self._signals_df = compute_all_signals(self._candles_df, self._features_df)
        print(f"  Signals shape: {self._signals_df.shape}")

    # ------------------------------------------------------------------
    # Batch preparation
    # ------------------------------------------------------------------

    def prepare_batch(self, candles_df, features_df, signals_df
                      ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Convert DataFrames into MLX tensors for one training batch.

        Returns:
            x:       [B, T, n_assets * n_features]
            signals: [B, 6]  regime-aggregated signals
            returns: [B]     next-bar portfolio return
        """
        import pandas as pd

        cfg = self.config
        T = cfg.seq_len
        B = cfg.batch_size
        input_dim = cfg.n_assets * cfg.n_features

        # --- timestamps available in features ---
        timestamps = sorted(features_df.index.unique())
        n_ts = len(timestamps)
        if n_ts < T + B:
            raise ValueError(f"Not enough timestamps: {n_ts} < {T + B}")

        # Random start indices for the batch
        max_start = n_ts - T - 1
        starts = np.random.randint(0, max_start, size=B)

        x_list, sig_list, ret_list = [], [], []

        for start in starts:
            ts_window = timestamps[start: start + T]
            ts_next = timestamps[start + T]

            # ---- x: feature matrix ----
            feat_win = features_df.loc[features_df.index.isin(ts_window)]
            # Pivot to [T, n_assets * n_features]; fill NaN with 0
            if hasattr(feat_win.index, 'names') and 'time' in (feat_win.index.names or []):
                feat_win = feat_win.reset_index()

            # Flatten: take first T rows, first input_dim columns
            arr = feat_win.select_dtypes(include=[np.number]).values
            arr = arr[:T, :input_dim] if arr.shape[1] >= input_dim else np.pad(
                arr[:T], ((0, 0), (0, input_dim - arr.shape[1])))
            if arr.shape[0] < T:
                arr = np.pad(arr, ((0, T - arr.shape[0]), (0, 0)))
            x_list.append(arr[:T].astype(np.float32))

            # ---- signals: regime-aggregated ----
            if signals_df is not None and len(signals_df) > 0:
                sig_at = signals_df[signals_df.index <= ts_next].tail(1)
                regime_sigs = np.zeros(N_REGIMES, dtype=np.float32)
                for ri, rname in enumerate(self.REGIME_NAMES):
                    if rname in sig_at.columns:
                        val = float(sig_at[rname].iloc[0]) if len(sig_at) else 0.0
                        regime_sigs[ri] = np.clip(val, -1.0, 1.0)
            else:
                regime_sigs = np.zeros(N_REGIMES, dtype=np.float32)
            sig_list.append(regime_sigs)

            # ---- returns: next-bar mean close return ----
            close_col = [c for c in candles_df.columns if 'close' in c.lower()]
            if close_col and ts_next in candles_df.index:
                ret_row = candles_df.loc[ts_next, close_col[0]]
                prev_ts = timestamps[start + T - 1]
                if prev_ts in candles_df.index:
                    prev_close = candles_df.loc[prev_ts, close_col[0]]
                    ret = float((ret_row - prev_close) / (prev_close + 1e-8))
                else:
                    ret = 0.0
            else:
                ret = 0.0
            ret_list.append(np.float32(ret))

        x_np = np.stack(x_list, axis=0).astype(np.float32)         # [B, T, D]
        sig_np = np.stack(sig_list, axis=0).astype(np.float32)      # [B, 6]
        ret_np = np.array(ret_list, dtype=np.float32)               # [B]

        return mx.array(x_np), mx.array(sig_np), mx.array(ret_np)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, epochs: int = 100):
        """Run training loop."""
        import pandas as pd

        if self._features_df is None:
            raise RuntimeError("Call load_training_data() first.")

        print(f"\nTraining HRM for {epochs} epochs ...")
        print(f"  Model params: {self._count_params():,}")

        for epoch in range(1, epochs + 1):
            x, signals, returns = self.prepare_batch(
                self._candles_df, self._features_df, self._signals_df
            )
            loss = train_step(self.model, self.optimizer, x, signals, returns)
            loss_val = float(loss.item())

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:4d}/{epochs}  loss={loss_val:.6f}")

            if epoch % 50 == 0:
                ckpt = pathlib.Path("hrm/checkpoints") / f"hrm_ane_epoch{epoch}.npz"
                self.save_checkpoint(str(ckpt))

        # Show regime weight distribution on final batch
        self._show_regime_distribution(x)

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str):
        """Save model weights as .npz (MLX format)."""
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(p))
        print(f"  Saved checkpoint: {p}")

    def load_checkpoint(self, path: str):
        """Load model weights from .npz."""
        self.model.load_weights(str(path))
        print(f"  Loaded checkpoint: {path}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _count_params(self) -> int:
        def _count(d):
            total = 0
            for v in d.values():
                if isinstance(v, dict):
                    total += _count(v)
                elif hasattr(v, 'size'):
                    total += v.size
            return total
        return _count(self.model.parameters())

    def _show_regime_distribution(self, x: mx.array):
        """Print mean regime weight distribution."""
        with mx.no_grad():
            weights = self.model(x)
        mean_w = weights.mean(axis=0)
        mx.eval(mean_w)
        print("\nRegime weight distribution (mean over batch):")
        for i, name in enumerate(self.REGIME_NAMES):
            print(f"  {name:20s}: {float(mean_w[i].item()):.4f}")


# =============================================================================
# LEGACY COMPAT — keep HRM alias and HRMConfig alias from original skeleton
# =============================================================================

# Keep old HRMConfig dataclass (from original skeleton) so imports don't break
@dataclass
class _LegacyHRMConfig:
    """Legacy config — use hrm.apple.config.HRMConfig instead."""
    n_assets: int = 43
    n_features: int = 10
    n_models: int = 3
    seq_len: int = 16
    hidden_dim: int = 64
    n_heads: int = 4
    H_cycles: int = 2
    L_cycles: int = 2
    H_layers: int = 2
    L_layers: int = 2

    @property
    def input_dim(self) -> int:
        return self.n_assets * self.n_features

    @property
    def output_dim(self) -> int:
        return self.n_models


# compute_loss kept for backward compatibility
def compute_loss(weights: mx.array, returns: mx.array,
                 prev_weights: Optional[mx.array] = None) -> mx.array:
    """Compute loss — legacy compat."""
    portfolio_return = mx.sum(weights * returns, axis=-1)
    loss = -mx.mean(portfolio_return)
    if prev_weights is not None:
        turnover = mx.mean(mx.sum(mx.abs(weights - prev_weights), axis=-1))
        loss = loss + 0.1 * turnover
    return loss


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
    from hrm.apple.config import UNDERFIT_HRM

    print("=" * 60)
    print("HRM MLX Smoke Test")
    print("=" * 60)

    model = HRMModel(UNDERFIT_HRM)

    def _count(d):
        total = 0
        for v in d.values():
            if isinstance(v, dict):
                total += _count(v)
            elif hasattr(v, 'size'):
                total += v.size
        return total

    n_params = _count(model.parameters())
    print(f"Parameters: {n_params:,}")

    x = mx.zeros((2, UNDERFIT_HRM.seq_len,
                  UNDERFIT_HRM.n_assets * UNDERFIT_HRM.n_features))
    out = model(x)
    mx.eval(out)

    print(f"Input shape:   {x.shape}")
    print(f"Output shape:  {out.shape}")
    print(f"Weights sum:   {out.sum(axis=-1)}")

    # Test gradient
    signals = mx.ones((2, N_REGIMES), dtype=mx.float32) / N_REGIMES
    returns = mx.array([0.01, -0.005], dtype=mx.float32)

    def loss_fn(m):
        w = m(x)
        return portfolio_loss(w, signals, returns)

    loss, grads = mx.value_and_grad(loss_fn)(model)
    mx.eval(loss)
    print(f"Loss:          {loss.item():.6f}")
    print("Gradients computed successfully")
    print("=" * 60)
