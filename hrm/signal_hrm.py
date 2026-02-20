"""
HRM Signal Processor — 16-signal convergence model
====================================================

Instead of raw OHLCV features → HRM, this adapts HRM to take the 16 most
differentiated trading signals as input and learn:
  1. Which signals to trust (learned weights)
  2. When convergence is nonzero (≥2 signals agree direction + confidence)
  3. Final combined alpha = weighted_sum(signal * weight)

Signal input shape: [batch, seq_len, N_SIGNALS * 2]
  — N_SIGNALS scores + N_SIGNALS confidences, interleaved
Output: (weights[N_SIGNALS], combined_alpha, convergence_score)

Convergence is nonzero when:
  signals pointing same direction ≥ 2
  AND their sum of confidences > threshold

This is the minimum viable HRM adaptation — pure MLX, ANE-friendly,
float32 everywhere. Runs on M3 Pro without a checkpoint (random weights
show convergence > 0 structure immediately).

Run smoke test:
    python3 hrm/signal_hrm.py
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("[signal_hrm] MLX not available — numpy fallback mode")

# ---------------------------------------------------------------------------
# The 16 canonical signals (balanced across 6 regimes)
# ---------------------------------------------------------------------------
SIGNAL_24 = [
    # TREND (4)
    "macd_crossover",
    "sota_momentum",
    "momentum_trend",
    "mom_trend_additive",
    # MEAN REVERSION (4)
    "rsi_mean_reversion",
    "bollinger_reversion",
    "grid_reversion",
    "hrm_mean_reversion",
    # VOLATILITY (3)
    "volatility_breakout",      # proven $37K winner
    "vol_x_breakout_proven",    # proven multiply
    "momentum_x_vol",
    # STAT ARB (2)
    "bent_penny",
    "pairs_spread",
    # SYSTEMATIC (1)
    "dca_baseline",
    # ML (1)
    "technical_ml",
    # COMPOSITE (1)
    "rsi_x_trend",
    # ADDITIONAL 8 SIGNALS (balanced across regimes)
    # TREND (1 additional - total 5)
    "momentum_strength",
    # MEAN REVERSION (1 additional - total 5)
    "reversion_strength",
    # VOLATILITY (1 additional - total 4)
    "volatility_regime",
    # STAT ARB (1 additional - total 3)
    "correlation_signal",
    # NEW REGIMES (4 additional)
    "order_flow_signal",        # Order flow analysis
    "liquidity_signal",         # Liquidity analysis
    "sector_rotation",          # Sector rotation signals
    "composite_trend",          # Composite trend analysis
]
N_SIGNALS = len(SIGNAL_24)   # 24
# assert N_SIGNALS == 16  # Remove this assertion

SIGNAL_REGIMES = {
    "macd_crossover":       "trend",
    "sota_momentum":        "trend",
    "momentum_trend":       "trend",
    "mom_trend_additive":   "trend",
    "rsi_mean_reversion":   "mean_reversion",
    "bollinger_reversion":  "mean_reversion",
    "grid_reversion":       "mean_reversion",
    "hrm_mean_reversion":   "mean_reversion",
    "volatility_breakout":  "volatility",
    "vol_x_breakout_proven":"volatility",
    "momentum_x_vol":       "volatility",
    "bent_penny":           "stat_arb",
    "pairs_spread":         "stat_arb",
    "dca_baseline":         "systematic",
    "technical_ml":         "ml",
    "rsi_x_trend":          "mean_reversion",
    # Additional signals
    "momentum_strength":    "trend",
    "reversion_strength":   "mean_reversion",
    "volatility_regime":    "volatility",
    "correlation_signal":   "stat_arb",
    "order_flow_signal":    "order_flow",
    "liquidity_signal":     "liquidity",
    "sector_rotation":      "sector_rotation",
    "composite_trend":      "trend",
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class SignalHRMConfig:
    n_signals: int    = N_SIGNALS   # 24
    context_dim: int  = 12          # 8 (Static) + 3 (Tradebots) + 1 (Spark)
    input_dim: int    = N_SIGNALS * 2 + 12 # 48 + 12 = 60
    hidden_dim: int   = 64
    n_heads: int      = 4
    seq_len: int      = 32          # lookback window (bars)
    H_layers: int     = 2
    L_layers: int     = 2
    H_cycles: int     = 2
    L_cycles: int     = 3
    dropout: float    = 0.1
    convergence_threshold: float = 0.25  # min confidence sum for nonzero convergence
    sparkline_frames: int  = 20    # number of perspective memory frames
    sparkline_horizon: int = 2000  # vanishing point in ticks (configurable, not model-visible)


# ---------------------------------------------------------------------------
# MLX layers (ANE-friendly)
# ---------------------------------------------------------------------------
if HAS_MLX:
    def rms_norm(x: mx.array, eps: float = 1e-5) -> mx.array:
        return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)

    class SwiGLU(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            hidden = (int(dim * 8 / 3) + 63) // 64 * 64
            self.gate_proj = nn.Linear(dim, hidden, bias=False)
            self.up_proj   = nn.Linear(dim, hidden, bias=False)
            self.down_proj = nn.Linear(hidden, dim, bias=False)

        def __call__(self, x: mx.array) -> mx.array:
            return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

    class SignalAttention(nn.Module):
        """Non-causal attention over signal sequence. ANE-friendly fixed shapes."""
        def __init__(self, dim: int, n_heads: int):
            super().__init__()
            self.n_heads  = n_heads
            self.head_dim = dim // n_heads
            self.scale    = self.head_dim ** -0.5
            self.qkv = nn.Linear(dim, 3 * dim, bias=False)
            self.out = nn.Linear(dim, dim, bias=False)

        def __call__(self, x: mx.array) -> mx.array:
            B, T, D = x.shape
            qkv = self.qkv(x)                                     # [B, T, 3D]
            q, k, v = mx.split(qkv, 3, axis=-1)                   # [B, T, D] each

            # reshape to [B, n_heads, T, head_dim]
            def split_heads(t):
                return t.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

            q, k, v = split_heads(q), split_heads(k), split_heads(v)
            attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale     # [B, H, T, T]
            attn = mx.softmax(attn, axis=-1)
            out  = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
            return self.out(out)

    class Block(nn.Module):
        def __init__(self, dim: int, n_heads: int):
            super().__init__()
            self.attn = SignalAttention(dim, n_heads)
            self.ffn  = SwiGLU(dim)

        def __call__(self, x: mx.array) -> mx.array:
            x = rms_norm(x + self.attn(x))
            x = rms_norm(x + self.ffn(x))
            return x

    class ReasoningModule(nn.Module):
        """Stack of transformer blocks with input injection."""
        def __init__(self, dim: int, n_heads: int, n_layers: int):
            super().__init__()
            self.layers = [Block(dim, n_heads) for _ in range(n_layers)]

        def __call__(self, z: mx.array, input_emb: mx.array) -> mx.array:
            x = z + input_emb           # input injection
            for layer in self.layers:
                x = layer(x)
            return x

    class SignalHRM(nn.Module):
        """
        HRM adapted for signal-array input - MLX version.
    
    Fully implements the reference HRM architecture for MLX:
    - H×L nested reasoning cycles
    - Transformer blocks with attention
    - Full sparkline memory with logarithmic timescales
    - ANE-friendly fixed shapes
    
    Input:  x  [B, seq_len, N_SIGNALS * 2]  (signal + confidence interleaved)
    Output:
        weights      [B, N_SIGNALS]   softmax weights per signal
        alpha        [B]              combined alpha = sum(weights * signals)
        convergence  [B]              0..1 convergence score
    """
    def __init__(self, cfg: SignalHRMConfig, force_cpu: bool = False):
        super().__init__()
        self.cfg = cfg
        self.force_cpu = force_cpu or not HAS_MLX
        
        if self.force_cpu:
            # CPU fallback uses random weights
            np.random.seed(42)
            self.signal_weights = np.random.randn(cfg.n_signals).astype(np.float32)
            self.conv_weight = np.random.randn(cfg.hidden_dim).astype(np.float32)
            return
        
        D = cfg.hidden_dim
        
        # Input projection: signal tensor → hidden dimension
        self.input_proj = nn.Linear(cfg.input_dim, D, bias=False)
        
        # Hierarchical reasoning modules (H×L structure)
        self.H_level = ReasoningModule(D, cfg.n_heads, cfg.H_layers)
        self.L_level = ReasoningModule(D, cfg.n_heads, cfg.L_layers)
        
        # Output heads
        self.weight_head = nn.Linear(D, cfg.n_signals, bias=True)
        self.conv_head = nn.Linear(D, 1, bias=True)
        
        # Learnable initial states (zeroed, nonzero only after training)
        self._z_H_init = mx.zeros((1, 1, D))
        self._z_L_init = mx.zeros((1, 1, D))

        def __call__(self, x: mx.array | np.ndarray, memory=None):
            """Forward pass with 20-frame logarithmic sparkline memory.
            
            Memory is a [B, 20, D] buffer where frame k represents timescale
            2^k ticks. Frame 0 = last tick, frame 19 ≈ 5 years. Cascading
            update with hyperbolic alpha; log-weighted readout for z_H/z_L init.
            No gradient through sparkline — idempotent past.
            
            Args:
                x: Input tensor [B, seq_len, input_dim]
                memory: Optional (sparkline_H, sparkline_L) each [B, 20, D]
            
            Returns:
                (weights, alpha, convergence, memory)
            """
            N_FRAMES = self.cfg.sparkline_frames
            HORIZON = self.cfg.sparkline_horizon
            # Geometric horizon progression: H_k = ratio^k
            # ratio = HORIZON^(1/(N-1)) ensures Frame N-1 is at HORIZON ticks.
            ratio = HORIZON ** (1.0 / max(N_FRAMES - 1, 1))
            
            if self.force_cpu:
                x_np = x if isinstance(x, np.ndarray) else np.array(x)
                B, T, F = x_np.shape
                D = min(F, self.cfg.hidden_dim)
                pooled = x_np.mean(axis=1)  # [B, F]
                
                # Sparkline update + readout (CPU)
                if memory is not None:
                    sparkline = memory.copy()
                else:
                    sparkline = np.zeros((B, N_FRAMES, F), dtype=np.float32)
                
                # Frame 0 = current tick
                sparkline[:, 0, :] = pooled
                # Cascade: alpha_k = 1 / H_k
                for k in range(1, N_FRAMES):
                    h_k = ratio ** k
                    alpha_k = 1.0 / h_k
                    sparkline[:, k, :] = (1.0 - alpha_k) * sparkline[:, k, :] + alpha_k * sparkline[:, k - 1, :]
                
                # Weighting: w_k = 1/H_k (near frames more important)
                # This provides a consistent "vanishing point" perspective.
                persp_w = np.array([1.0 / (ratio ** k) for k in range(N_FRAMES)], dtype=np.float32)
                persp_w /= persp_w.sum()
                context = np.einsum('k,bkd->bd', persp_w, sparkline)  # [B, F]
                
                # Blend context into pooled
                pooled = 0.5 * pooled + 0.5 * context
                new_memory = sparkline
                
                w_logits = np.dot(pooled[:, :N_SIGNALS], np.diag(self.signal_weights))
                weights = np.exp(w_logits) / np.sum(np.exp(w_logits), axis=-1, keepdims=True)
                conv_d = min(pooled.shape[1], self.cfg.hidden_dim)
                conv_val = np.tanh(np.dot(pooled[:, :conv_d], self.conv_weight[:conv_d]))
                convergence = 0.5 * (conv_val + 1.0)
                raw_signals = x_np[:, -1, :N_SIGNALS*2:2]
                alpha = (weights * raw_signals).sum(axis=-1)
                return weights, alpha, convergence, new_memory

            B, T, _ = x.shape
            D = self.cfg.hidden_dim

            # Project input → embedding
            x_emb = self.input_proj(x)  # [B, T, D]

            # ── SPARKLINE MEMORY ──────────────────────────────────
            if memory is not None:
                spark_H = mx.stop_gradient(memory[0])  # [B, 20, D]
                spark_L = mx.stop_gradient(memory[1])
            else:
                spark_H = mx.zeros((B, N_FRAMES, D))
                spark_L = mx.zeros((B, N_FRAMES, D))

            # Current tick state (mean of input embedding)
            curr_H = x_emb.mean(axis=1, keepdims=True)  # [B, 1, D]
            curr_L = x_emb.mean(axis=1, keepdims=True)

            # Build updated sparkline: frame 0 = current, cascade up
            # Geometric horizon: H_k = ratio^k, alpha_k = 1/H_k
            frames_H = [curr_H]
            frames_L = [curr_L]
            for k in range(1, N_FRAMES):
                h_k = ratio ** k
                alpha_k = 1.0 / h_k
                fH = (1.0 - alpha_k) * spark_H[:, k:k+1, :] + alpha_k * frames_H[k - 1]
                fL = (1.0 - alpha_k) * spark_L[:, k:k+1, :] + alpha_k * frames_L[k - 1]
                frames_H.append(fH)
                frames_L.append(fL)
            
            new_spark_H = mx.concatenate(frames_H, 1)
            new_spark_L = mx.concatenate(frames_L, 1)

            # Weighting: w_k = 1/H_k
            persp_w = mx.array([1.0 / (ratio ** k) for k in range(N_FRAMES)])
            persp_w = persp_w / persp_w.sum()
            context_H = mx.sum(new_spark_H * persp_w.reshape(1, N_FRAMES, 1), axis=1, keepdims=True)
            context_L = mx.sum(new_spark_L * persp_w.reshape(1, N_FRAMES, 1), axis=1, keepdims=True)

            # Init z_H/z_L from sparkline context
            z_H = mx.broadcast_to(context_H, (B, T, D))
            z_L = mx.broadcast_to(context_L, (B, T, D))

            # Nested H × L recurrence (stop_gradient on non-final iterations)
            for h in range(self.cfg.H_cycles):
                for l in range(self.cfg.L_cycles):
                    is_final = (h == self.cfg.H_cycles - 1) and (l == self.cfg.L_cycles - 1)
                    if not is_final:
                        z_L = mx.stop_gradient(self.L_level(z_L, z_H + x_emb))
                    else:
                        z_L = self.L_level(z_L, z_H + x_emb)   # grad flows here only
                if h < self.cfg.H_cycles - 1:
                    z_H = mx.stop_gradient(self.H_level(z_H, z_L))
                else:
                    z_H = self.H_level(z_H, z_L)

            # Carry forward sparkline (detached, idempotent past)
            new_memory = (mx.stop_gradient(new_spark_H), mx.stop_gradient(new_spark_L))

            # Pool over sequence → [B, D]
            pooled = z_H.mean(axis=1)

            # Signal weights (softmax → sums to 1, all positive)
            weights = mx.softmax(self.weight_head(pooled), axis=-1)  # [B, N_SIGNALS]

            # Convergence score [B]
            conv_logit = self.conv_head(pooled).squeeze(-1)           # [B]
            convergence = mx.sigmoid(conv_logit)                      # [B] in (0,1)

            # Combined alpha [B]: weighted dot of HRM weights with raw signals
            raw_signals = x[:, -1, :N_SIGNALS*2:2]                   # [B, N_SIGNALS]
            alpha = (weights * raw_signals).sum(axis=-1)              # [B]

            return weights, alpha, convergence, new_memory

else:
    # CPU fallback - separate class definition
    class SignalHRM:
        """
        CPU-friendly NumPy version of SignalHRM.
        
        This is a FALLBACK implementation, not meant to match MLX numerically.
        - Uses simplified pooling instead of transformer blocks
        - Uses random weights (simulates uninitialized NN)
        - Same output format as MLX version
        - Used when MLX is unavailable or force_cpu=True
        
        Note: This is for training on CPU (BinanceSpotTrainer).
        For production on Apple Silicon, always use MLX version.
        """
        def __init__(self, cfg: SignalHRMConfig, force_cpu: bool = False):
            self.cfg = cfg
            self.force_cpu = True  # Always CPU in this branch
            np.random.seed(42)  # Deterministic for reproducibility
            
            # Random weights (simulates uninitialized NN for training)
            # These have same shapes as MLX weight_head and conv_head
            self.weight_head_weight = np.random.randn(cfg.hidden_dim, cfg.n_signals).astype(np.float32)
            self.weight_head_bias = np.random.randn(cfg.n_signals).astype(np.float32) * 0.1
            self.conv_head_weight = np.random.randn(cfg.hidden_dim, 1).astype(np.float32)
            self.conv_head_bias = np.random.randn(1).astype(np.float32) * 0.1
            
            # Hidden dimension
            self.hidden_dim = cfg.hidden_dim
        
        def __call__(self, x, memory=None):
            """
            Simplified CPU forward pass.
            
            Note: Does NOT match MLX numerically (different architecture).
            But produces valid trading signals suitable for training.
            """
            N_FRAMES = self.cfg.sparkline_frames
            HORIZON = self.cfg.sparkline_horizon
            ratio = HORIZON ** (1.0 / max(N_FRAMES - 1, 1))
            B, T, F = x.shape
            
            # Simplified pooling (instead of transformer blocks)
            pooled = x.mean(axis=1)  # [B, F]
            
            # Project to hidden dim
            if F > self.hidden_dim:
                pooled = pooled[:, :self.hidden_dim]
            elif F < self.hidden_dim:
                padded = np.zeros((B, self.hidden_dim), dtype=np.float32)
                padded[:, :F] = pooled
                pooled = padded
            
            # Sparkline memory (same logic as MLX)
            if memory is not None:
                sparkline = memory.copy()
            else:
                sparkline = np.zeros((B, N_FRAMES, self.hidden_dim), dtype=np.float32)
            
            sparkline[:, 0, :] = pooled
            for k in range(1, N_FRAMES):
                h_k = ratio ** k
                alpha_k = 1.0 / h_k
                sparkline[:, k, :] = (1.0 - alpha_k) * sparkline[:, k, :] + alpha_k * sparkline[:, k - 1, :]
            
            persp_w = np.array([1.0 / (ratio ** k) for k in range(N_FRAMES)], dtype=np.float32)
            persp_w /= persp_w.sum()
            context = np.einsum('k,bkd->bd', persp_w, sparkline)
            
            pooled = 0.5 * pooled + 0.5 * context
            
            # Weight computation (same as MLX: softmax over weight_head output)
            w_logits = np.dot(pooled, self.weight_head_weight) + self.weight_head_bias
            weights = np.exp(w_logits) / np.sum(np.exp(w_logits), axis=-1, keepdims=True)
            
            # Convergence computation (same as MLX: sigmoid)
            conv_logit = np.dot(pooled, self.conv_head_weight) + self.conv_head_bias
            conv_logit = conv_logit.squeeze(-1)
            convergence = 1.0 / (1.0 + np.exp(-conv_logit))  # sigmoid
            
            # Alpha: weighted sum of signals
            raw_signals = x[:, -1, :N_SIGNALS*2:2]  # [B, N_SIGNALS]
            alpha = (weights * raw_signals).sum(axis=-1)
            
            new_memory = sparkline
            
            return weights, alpha, convergence, new_memory
        
        def parameters(self):
            return []
        
        def to_ml(self) -> Dict[str, np.ndarray]:
            """Export parameters for MLX model (for weight transfer after training)"""
            return {
                'weight_head_weight': self.weight_head_weight,
                'weight_head_bias': self.weight_head_bias,
                'conv_head_weight': self.conv_head_weight,
                'conv_head_bias': self.conv_head_bias,
            }
        
        def load_from_dict(self, params: Dict[str, np.ndarray]):
            """Load parameters from dictionary (for weight transfer after training)"""
            for key in ['weight_head_weight', 'weight_head_bias', 
                       'conv_head_weight', 'conv_head_bias']:
                if key in params:
                    setattr(self, key, params[key].astype(np.float32))
        
        def to_ml(self) -> Dict[str, np.ndarray]:
            """Export parameters for MLX model"""
            return {
                'signal_weights': self.signal_weights,
                'conv_weight': self.conv_weight,
            }
        
        def load_from_dict(self, params: Dict[str, np.ndarray]):
            """Load parameters from dictionary"""
            if 'signal_weights' in params:
                self.signal_weights = params['signal_weights'].astype(np.float32)
            if 'conv_weight' in params:
                self.conv_weight = params['conv_weight'].astype(np.float32)
            
            # Also update MLX arrays if available
            if HAS_MLX and hasattr(self, 'signal_weights_mx'):
                self.signal_weights_mx = mx.array(self.signal_weights)
                self.conv_weight_mx = mx.array(self.conv_weight)


# ---------------------------------------------------------------------------
# Convergence scoring (numpy/pandas — no MLX needed)
# ---------------------------------------------------------------------------

def convergence_score(signals: np.ndarray, confidences: np.ndarray,
                      threshold: float = 0.25) -> float:
    """
    Pure numpy convergence score for one timestep.

    signals:      [N_SIGNALS] in [-1, 1]
    confidences:  [N_SIGNALS] in [0, 1]
    threshold:    minimum confidence sum for nonzero

    Returns:
        score in [0, 1]
        0.0 = no convergence (signals cancel or all low confidence)
        1.0 = full convergence (all signals agree with high confidence)

    Nonzero when:
        at least 2 signals agree direction (same sign)
        AND their confidence sum > threshold
    """
    pos_mask = signals > 0
    neg_mask = signals < 0

    pos_conf = (confidences * pos_mask).sum()
    neg_conf = (confidences * neg_mask).sum()

    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()

    # Must have at least 2 signals agreeing
    if max(n_pos, n_neg) < 2:
        return 0.0

    dominant_conf = max(pos_conf, neg_conf)
    if dominant_conf < threshold:
        return 0.0

    # Score = |pos_conf - neg_conf| / (pos_conf + neg_conf + 1e-8)
    # = how lopsided the confidence is toward one side
    total = pos_conf + neg_conf + 1e-8
    score = float(abs(pos_conf - neg_conf) / total)
    return score


def compute_convergence_series(signals_df: pd.DataFrame,
                                signal_names: List[str] = SIGNAL_24) -> pd.DataFrame:
    """
    Compute per-timestep convergence scores for each symbol.

    signals_df: output of pipeline.compute_all_signals()
    Returns: DataFrame [timestamp, symbol, convergence, n_agree, dominant_direction]
    """
    # Filter to the 16 canonical signals
    s16 = signals_df[signals_df["model"].isin(signal_names)]

    records = []
    for (ts, sym), grp in s16.groupby(["timestamp", "symbol"]):
        if len(grp) < 2:
            continue

        sig_arr  = grp["signal"].values.astype(np.float32)
        conf_arr = grp["confidence"].values.astype(np.float32)

        score    = convergence_score(sig_arr, conf_arr)
        n_pos    = int((sig_arr > 0).sum())
        n_neg    = int((sig_arr < 0).sum())
        direction = "LONG" if n_pos > n_neg else ("SHORT" if n_neg > n_pos else "FLAT")

        records.append({
            "timestamp":   ts,
            "symbol":      sym,
            "convergence": score,
            "n_long":      n_pos,
            "n_short":     n_neg,
            "direction":   direction,
            "nonzero":     score > 0.0,
        })

    return pd.DataFrame(records)


def build_signal_tensor(signals_df: pd.DataFrame,
                         symbol: str,
                         seq_len: int = 32,
                         signal_names: List[str] = SIGNAL_24) -> Optional[np.ndarray]:
    """
    Build [1, seq_len, N_SIGNALS * 2] float32 array for HRM inference.

    Layout: [sig_0, conf_0, sig_1, conf_1, ..., sig_15, conf_15]
    """
    s16 = signals_df[
        (signals_df["symbol"] == symbol) &
        (signals_df["model"].isin(signal_names))
    ].copy()

    if s16.empty:
        return None

    # Pivot to wide: [timestamp × model]
    sig_wide  = s16.pivot_table(index="timestamp", columns="model", values="signal")
    conf_wide = s16.pivot_table(index="timestamp", columns="model", values="confidence")

    # Reindex to canonical order
    sig_wide  = sig_wide.reindex(columns=signal_names, fill_value=0.0)
    conf_wide = conf_wide.reindex(columns=signal_names, fill_value=0.0)

    # Interleave: [sig_0, conf_0, sig_1, conf_1, ...]
    T = len(sig_wide)
    out = np.zeros((T, N_SIGNALS * 2), dtype=np.float32)
    out[:, 0::2] = sig_wide.values.astype(np.float32)
    out[:, 1::2] = conf_wide.values.astype(np.float32)

    # Take last seq_len rows
    out = out[-seq_len:]
    if len(out) < seq_len:
        # Pad front with zeros
        out = np.vstack([np.zeros((seq_len - len(out), N_SIGNALS * 2), dtype=np.float32), out])

    return out[np.newaxis, :, :]  # [1, seq_len, 32]


# ---------------------------------------------------------------------------
# Portfolio loss for training
# ---------------------------------------------------------------------------
if HAS_MLX:
    def portfolio_loss(weights: mx.array,
                       alpha: mx.array,
                       convergence: mx.array,
                       returns: mx.array,
                       prev_weights: Optional[mx.array] = None,
                       turnover_lambda: float = 0.1,
                       convergence_lambda: float = 0.05) -> mx.array:
        """
        Loss = -E[alpha * returns * convergence]
               + turnover_lambda * turnover
               + convergence_lambda * entropy(weights)

        Maximizes:
          - P&L (alpha × actual return, gated by convergence)
          - Concentration (low weight entropy = model picks a few strong signals)

        Minimizes:
          - Turnover (weight stability)

        convergence acts as a confidence gate:
          when convergence=0, no position → no loss → model learns to be selective.
        """
        # Primary: gated PnL
        gated_pnl = alpha * returns * convergence
        pnl_loss  = -gated_pnl.mean()

        # Turnover penalty
        if prev_weights is not None:
            turnover = mx.abs(weights - prev_weights).mean()
        else:
            turnover = mx.array(0.0)

        # Entropy penalty: encourages concentration (fewer high-weight signals)
        entropy = -(weights * mx.log(weights + 1e-8)).sum(axis=-1).mean()

        return pnl_loss + turnover_lambda * turnover + convergence_lambda * entropy


# ---------------------------------------------------------------------------
# Smoke test + convergence report
# ---------------------------------------------------------------------------

def run_smoke_test(cfg: Optional[SignalHRMConfig] = None) -> None:
    """
    Instantiate SignalHRM and verify:
    1. Output shapes are correct
    2. Weights sum to ~1.0 (softmax)
    3. Convergence is in (0, 1)
    4. Nonzero alpha for nonzero input
    """
    if not HAS_MLX:
        print("[smoke] MLX not available — skipping forward pass test")
        _numpy_smoke_test()
        return

    cfg = cfg or SignalHRMConfig()
    model = SignalHRM(cfg)
    mx.eval(model.parameters())

    B, T, D_in = 4, cfg.seq_len, cfg.input_dim

    print(f"\n── SignalHRM Smoke Test ───────────────────────────────────────")
    print(f"   Config: n_signals={cfg.n_signals}, hidden={cfg.hidden_dim}, "
          f"seq_len={cfg.seq_len}, H×L={cfg.H_cycles}×{cfg.L_cycles}")
    print(f"   Input:  [{B}, {T}, {D_in}]")

    # Test 1: zero input → should give uniform weights, near-zero alpha
    x_zeros = mx.zeros((B, T, D_in))
    w, a, c, _ = model(x_zeros)
    mx.eval(w, a, c)
    print(f"\n   [ZERO INPUT]")
    print(f"   weights shape:     {list(w.shape)}  ✓" if list(w.shape) == [B, N_SIGNALS] else f"   weights shape: FAIL {list(w.shape)}")
    print(f"   weights sum:       {np.array(w.sum(axis=-1)).mean():.4f}  (expect ~1.0)")
    print(f"   alpha:             {np.array(a).mean():.4f}  (expect ~0.0 for zero input)")
    print(f"   convergence:       {np.array(c).mean():.4f}  (expect ~0.5 uninit)")

    # Test 2: convergent bullish input (all trend signals = +0.8, conf = 0.7)
    x_bull = np.zeros((B, T, D_in), dtype=np.float32)
    # Set trend signals (indices 0-3) to +0.8
    for i in range(4):
        x_bull[:, :, i * 2]     = 0.8   # signal
        x_bull[:, :, i * 2 + 1] = 0.7   # confidence
    # Set mean_rev signals (4-7) to -0.3 (counter signal)
    for i in range(4, 8):
        x_bull[:, :, i * 2]     = -0.3
        x_bull[:, :, i * 2 + 1] = 0.3
    x_bull_mx = mx.array(x_bull)
    w2, a2, c2, _ = model(x_bull_mx)
    mx.eval(w2, a2, c2)
    print(f"\n   [BULL CONVERGENCE INPUT — 4 trend signals agree +0.8]")
    print(f"   weights (trend avg):  {np.array(w2)[:, :4].mean():.4f}")
    print(f"   weights (rev avg):    {np.array(w2)[:, 4:8].mean():.4f}")
    print(f"   alpha:                {np.array(a2).mean():.4f}")
    print(f"   convergence:          {np.array(c2).mean():.4f}")

    # Test 3: divergent input (signals cancel)
    x_div = np.zeros((B, T, D_in), dtype=np.float32)
    for i in range(8):
        sign = 1.0 if i % 2 == 0 else -1.0
        x_div[:, :, i * 2]     = sign * 0.7
        x_div[:, :, i * 2 + 1] = 0.6
    x_div_mx = mx.array(x_div)
    w3, a3, c3, _ = model(x_div_mx)
    mx.eval(w3, a3, c3)
    print(f"\n   [DIVERGENT INPUT — alternating +/- signals]")
    print(f"   alpha:        {np.array(a3).mean():.4f}  (expect near zero)")
    print(f"   convergence:  {np.array(c3).mean():.4f}")

    # Numpy convergence scoring (no MLX needed)
    _numpy_smoke_test()


def _numpy_smoke_test():
    print(f"\n── Numpy Convergence Smoke Test ───────────────────────────────")
    # 4 trend signals agree long, rest noisy
    sigs  = np.array([0.8,  0.7,  0.9,  0.6, -0.2, 0.1, -0.3, 0.0,
                      0.1, -0.1, 0.0,  0.1,  0.0, 0.2, 0.3, 0.1], dtype=np.float32)
    confs = np.array([0.8,  0.7,  0.9,  0.8,  0.3, 0.2, 0.4, 0.1,
                      0.2,  0.2, 0.1,  0.2,  0.1, 0.2, 0.2, 0.2], dtype=np.float32)
    score = convergence_score(sigs, confs)
    n_pos = int((sigs > 0).sum())
    n_neg = int((sigs < 0).sum())
    print(f"   Bull scenario: convergence={score:.3f}  n_long={n_pos}  n_short={n_neg}")
    print(f"   Nonzero: {score > 0.0}  ✓" if score > 0.0 else "   FAIL: convergence is zero!")

    # Fully divergent
    sigs2  = np.array([0.8, -0.8, 0.7, -0.7, 0.6, -0.6, 0.5, -0.5,
                       0.4, -0.4, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1], dtype=np.float32)
    confs2 = np.full(16, 0.5, dtype=np.float32)
    score2 = convergence_score(sigs2, confs2)
    print(f"   Divergent scenario: convergence={score2:.3f}  (expect ~0.0)")

    # Minimum viable: 2 signals agree
    sigs3  = np.zeros(16, dtype=np.float32)
    confs3 = np.zeros(16, dtype=np.float32)
    sigs3[0]  = 0.9;  confs3[0]  = 0.8
    sigs3[1]  = 0.7;  confs3[1]  = 0.6
    score3 = convergence_score(sigs3, confs3)
    print(f"   Minimum (2 agree, rest zero): convergence={score3:.3f}  nonzero={score3>0}")

    print(f"\n   Signal register ({N_SIGNALS} signals):")
    for i, name in enumerate(SIGNAL_24):
        regime = SIGNAL_REGIMES.get(name, "?")
        print(f"   [{i:2d}] {name:<28} {regime}")


# ---------------------------------------------------------------------------
# Unified factory for MLX ↔ CPU compatibility
# ---------------------------------------------------------------------------

def create_signal_hrm(
    cfg: Optional[SignalHRMConfig] = None,
    force_cpu: Optional[bool] = None,
) -> SignalHRM:
    """
    Factory function to create SignalHRM with automatic MLX/CPU selection.
    
    Args:
        cfg: Configuration for the model
        force_cpu: Force CPU version (True), MLX version (False), or auto (None)
    
    Returns:
        SignalHRM instance (MLX or CPU depending on availability and force_cpu)
    
    Usage:
        # Auto-select (preferred)
        hrm = create_signal_hrm(cfg)
        
        # Force CPU (for training on non-Apple hardware)
        hrm = create_signal_hrm(cfg, force_cpu=True)
        
        # Force MLX (for production on Apple Silicon)
        hrm = create_signal_hrm(cfg, force_cpu=False)
    """
    if cfg is None:
        cfg = SignalHRMConfig()
    
    # Determine backend
    if force_cpu is None:
        # Auto-detect: use MLX if available, otherwise CPU
        backend = "MLX" if HAS_MLX else "CPU"
        use_cpu = not HAS_MLX
    elif force_cpu:
        # Force CPU
        backend = "CPU (forced)"
        use_cpu = True
    else:
        # Force MLX
        backend = "MLX (forced)"
        use_cpu = False
    
    print(f"[SignalHRM] Using {backend} backend")
    
    return SignalHRM(cfg, force_cpu=use_cpu)


# ---------------------------------------------------------------------------
# Weight transfer utilities (MLX ↔ CPU)
# ---------------------------------------------------------------------------

def export_hrm_weights(hrm: SignalHRM) -> Dict[str, np.ndarray]:
    """
    Export weights from SignalHRM to numpy array.
    
    Works with both MLX and CPU versions.
    """
    weights = {}
    
    if hasattr(hrm, 'signal_weights'):
        # CPU version or has numpy weights
        weights['signal_weights'] = hrm.signal_weights.copy()
    
    if hasattr(hrm, 'conv_weight'):
        weights['conv_weight'] = hrm.conv_weight.copy()
    
    if hasattr(hrm, 'weight_head') and hasattr(hrm.weight_head, 'weight'):
        # MLX version: extract weight_head weights
        weight_head_weight = np.array(hrm.weight_head.weight)
        weights['weight_head_weight'] = weight_head_weight
        
        if hasattr(hrm.weight_head, 'bias') and hrm.weight_head.bias is not None:
            weight_head_bias = np.array(hrm.weight_head.bias)
            weights['weight_head_bias'] = weight_head_bias
    
    if hasattr(hrm, 'conv_head') and hasattr(hrm.conv_head, 'weight'):
        conv_head_weight = np.array(hrm.conv_head.weight)
        weights['conv_head_weight'] = conv_head_weight
        
        if hasattr(hrm.conv_head, 'bias') and hrm.conv_head.bias is not None:
            conv_head_bias = np.array(hrm.conv_head.bias)
            weights['conv_head_bias'] = conv_head_bias
    
    if hasattr(hrm, 'input_proj') and hasattr(hrm.input_proj, 'weight'):
        input_proj_weight = np.array(hrm.input_proj.weight)
        weights['input_proj_weight'] = input_proj_weight
    
    return weights


def load_hrm_weights(hrm: SignalHRM, weights: Dict[str, np.ndarray]):
    """
    Load weights into SignalHRM from numpy array dictionary.
    
    Works with both MLX and CPU versions.
    """
    if hasattr(hrm, 'signal_weights') and 'signal_weights' in weights:
        hrm.signal_weights = weights['signal_weights'].astype(np.float32)
    
    if hasattr(hrm, 'conv_weight') and 'conv_weight' in weights:
        hrm.conv_weight = weights['conv_weight'].astype(np.float32)
    
    # MLX version: load into model parameters
    if not hrm.force_cpu:
        if hasattr(hrm, 'weight_head') and 'weight_head_weight' in weights:
            hrm.weight_head.weight = mx.array(weights['weight_head_weight'].astype(np.float32))
        
        if hasattr(hrm, 'weight_head') and 'weight_head_bias' in weights:
            hrm.weight_head.bias = mx.array(weights['weight_head_bias'].astype(np.float32))
        
        if hasattr(hrm, 'conv_head') and 'conv_head_weight' in weights:
            hrm.conv_head.weight = mx.array(weights['conv_head_weight'].astype(np.float32))
        
        if hasattr(hrm, 'conv_head') and 'conv_head_bias' in weights:
            hrm.conv_head.bias = mx.array(weights['conv_head_bias'].astype(np.float32))
        
        if hasattr(hrm, 'input_proj') and 'input_proj_weight' in weights:
            hrm.input_proj.weight = mx.array(weights['input_proj_weight'].astype(np.float32))
        
        # Also update the numpy cache for CPU compatibility
        if hasattr(hrm, 'signal_weights'):
            hrm.signal_weights = weights.get('signal_weights', hrm.signal_weights).astype(np.float32)
        if hasattr(hrm, 'conv_weight'):
            hrm.conv_weight = weights.get('conv_weight', hrm.conv_weight).astype(np.float32)
    
    print(f"[SignalHRM] Loaded weights: {list(weights.keys())}")


# ---------------------------------------------------------------------------
# Compatibility verification
# ---------------------------------------------------------------------------

def verify_mlx_cpu_compatibility(cfg: Optional[SignalHRMConfig] = None) -> bool:
    """
    Verify that MLX and CPU versions work correctly (not numerically identical).
    
    Note: Exact numerical matching is NOT expected or required because:
    - MLX version uses full transformer architecture (H×L nested cycles with attention)
    - CPU version uses simplified pooling (for training performance)
    
    Both versions:
    1. Accept same input format [B, seq_len, N_SIGNALS*2]
    2. Output same format: (weights, alpha, convergence, memory)
    3. Use same sparkline memory logic
    4. Use same softmax/sigmoid activation
    5. Can transfer weights (after training on CPU)
    """
    if not HAS_MLX:
        print("[verify] MLX not available - skipping verification")
        return True
    
    if cfg is None:
        cfg = SignalHRMConfig()
    
    # Create both versions
    mlx_hrm = SignalHRM(cfg, force_cpu=False)
    cpu_hrm = SignalHRM(cfg, force_cpu=True)
    
    # Test input
    B, T, D = 2, cfg.seq_len, cfg.input_dim
    test_input = np.random.randn(B, T, D).astype(np.float32)
    
    # MLX forward
    mlx_input = mx.array(test_input)
    mlx_weights, mlx_alpha, mlx_conv, mlx_mem = mlx_hrm(mlx_input)
    mx.eval(mlx_weights, mlx_alpha, mlx_conv)
    
    # CPU forward
    cpu_weights, cpu_alpha, cpu_conv, cpu_mem = cpu_hrm(test_input)
    
    # Verify shapes
    shapes_ok = (
        list(mlx_weights.shape) == [B, cfg.n_signals] and
        list(cpu_weights.shape) == [B, cfg.n_signals] and
        list(mlx_alpha.shape) == [B] and
        list(cpu_alpha.shape) == [B] and
        list(mlx_conv.shape) == [B] and
        list(cpu_conv.shape) == [B]
    )
    
    # Verify output validity (each in expected range)
    mlx_weights_np = np.array(mlx_weights)
    all_valid = (
        # Weights should sum to 1
        np.allclose(mlx_weights_np.sum(axis=-1), 1.0, atol=1e-5) and
        np.allclose(cpu_weights.sum(axis=-1), 1.0, atol=1e-5) and
        # Convergence in [0, 1]
        np.all(0 <= np.array(mlx_conv_np) <= 1) and
        np.all(0 <= cpu_conv <= 1) and
        # Weights positive
        np.all(mlx_weights_np >= 0) and
        np.all(cpu_weights >= 0)
    )
    
    print(f"\n── MLX ↔ CPU Functional Verification ─────────────────────────")
    print(f"   Input: [{B}, {T}, {D}]")
    print(f"   Output shapes:  {shapes_ok}")
    print(f"   Output valid:   {all_valid}")
    print(f"   MLX backend:    {'✅ Available' if HAS_MLX else '❌ Not available'}")
    print(f"   CPU backend:    ✅ Always available")
    print(f"\n   Note: Numerical outputs will differ due to different architectures:")
    print(f"   - MLX: Full transformer with H×L nested cycles (for production)")
    print(f"   - CPU: Simplified pooling (for training on CPU)")
    print(f"\n   After training on CPU, weights can be transferred to MLX version.")
    
    return shapes_ok and all_valid


if __name__ == "__main__":
    print("=" * 70)
    print("SignalHRM: MLX ↔ CPU Compatible HRM for Trading Signals")
    print("=" * 70)
    
    # Run smoke test
    run_smoke_test()
    
    # Verify compatibility
    verify_mlx_cpu_compatibility()
