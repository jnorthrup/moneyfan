"""
HRM — Hierarchical Reasoning Model (PyTorch Reference)
=======================================================

Architecture:
    TemporalOrderBook (ob_depth_frames depth) → HRM → Trade Output

Naming convention (crypto-technical):
    TemporalOrderBook         — cascading exponential decay memory over bar embeddings
    ob_depth_frames           — temporal depth levels (decay frames) in the order book
    ob_lookback_horizon       — candle horizon for decay ratio calibration
    regime_state (z_H)        — working state of the macro regime layer
    tactical_state (z_L)      — working state of the tactical execution layer
    bar_codec_features [B,T+n,F] — extent input: T bar window + n prediction horizon
    codec_score_head          — world-model output (next-bar codec features)
    expected_return_head      — forward expected return ∈ [-1, 1]
    signal_conviction_head    — conviction score ∈ [0, 1]
    stop_loss_head            — SL offset ∈ [-0.15, 0]
    take_profit_head          — TP target ∈ [0, 0.30]
    position_size_head        — fraction of notional ∈ [0, 1]

Extent definition:
    extent = T + n  (bar window T + prediction horizon n)
    The model sees T bars in context and predicts n bars forward.

Training phases:
    pretrain : world-model loss — predict bar t+n codec features (self-supervised)
    trade    : alpha loss — maximise conviction-weighted expected return (supervised)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import math


@dataclass
class HRMConfig:
    """
    Configuration for the Hierarchical Reasoning Model (HRM).

    Crypto-technical field names:
      n_codec_outputs         : codec expert output channels (default 24)
      hidden_dim              : HRM embedding dimension
      ob_depth_frames         : temporal order book depth (decay frames)
      ob_lookback_horizon     : candle horizon for OB decay ratio calibration
      regime_attn_layers      : transformer layers in macro regime layer
      tactical_attn_layers    : transformer layers in tactical execution layer
      regime_update_cycles    : macro regime update cycles per forward pass
      tactical_update_cycles  : tactical cycles per regime cycle
      n_heads                 : attention heads
      dropout                 : dropout probability
    """
    n_codec_outputs: int = 24
    hidden_dim: int = 64
    ob_depth_frames: int = 20
    ob_lookback_horizon: int = 200
    regime_attn_layers: int = 2
    tactical_attn_layers: int = 2
    regime_update_cycles: int = 2
    tactical_update_cycles: int = 3
    n_heads: int = 4
    dropout: float = 0.1

    # Legacy aliases
    @property
    def n_signals(self): return self.n_codec_outputs
    @property
    def sparkline_frames(self): return self.ob_depth_frames
    @property
    def sparkline_horizon(self): return self.ob_lookback_horizon
    @property
    def H_layers(self): return self.regime_attn_layers
    @property
    def L_layers(self): return self.tactical_attn_layers
    @property
    def H_cycles(self): return self.regime_update_cycles
    @property
    def L_cycles(self): return self.tactical_update_cycles


HierarchicalCodecConfig = HRMConfig


class DepthAttentionBlock(nn.Module):
    """Single transformer block: MHA + FF with residuals."""
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(x, x, x, need_weights=False)[0])
        x = self.norm1(x)
        x = x + self.dropout(self.ff(x))
        x = self.norm2(x)
        return x


class MarketDepthLayer(nn.Module):
    """
    Market depth processing layer — stacked DepthAttentionBlocks.
    Used for both macro_regime_layer and tactical_execution_layer.
    """
    def __init__(self, hidden_dim: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList([
            DepthAttentionBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, state: torch.Tensor, bar_features_with_context: torch.Tensor) -> torch.Tensor:
        x = state + bar_features_with_context
        for block in self.blocks:
            x = block(x)
        return x


class TemporalOrderBook(nn.Module):
    """
    Temporal Order Book — cascading exponential decay memory.

    Frame 0 = most recent bar (highest weight), frame ob_depth_frames-1 = oldest.
    Frame k = (1 - alpha_k) * old[k] + alpha_k * frame_{k-1}
    Produces a recency-weighted market context vector.
    """
    def __init__(self, hidden_dim: int, ob_depth_frames: int = 20, ob_lookback_horizon: int = 200):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ob_depth_frames = ob_depth_frames
        self.decay_ratio = ob_lookback_horizon ** (1.0 / max(ob_depth_frames - 1, 1))

    def update(self, temporal_ob: Optional[torch.Tensor], current_bar: torch.Tensor) -> torch.Tensor:
        B = current_bar.shape[0]
        if temporal_ob is None:
            temporal_ob = torch.zeros(B, self.ob_depth_frames, self.hidden_dim, device=current_bar.device)
        frames = [current_bar.unsqueeze(1)]
        for k in range(1, self.ob_depth_frames):
            alpha_k = 1.0 / (self.decay_ratio ** k)
            frames.append((1.0 - alpha_k) * temporal_ob[:, k:k+1, :] + alpha_k * frames[k-1])
        return torch.cat(frames, dim=1)

    def read(self, temporal_ob: torch.Tensor) -> torch.Tensor:
        weights = torch.tensor(
            [1.0 / (self.decay_ratio ** k) for k in range(self.ob_depth_frames)],
            device=temporal_ob.device
        )
        weights = weights / weights.sum()
        return (temporal_ob * weights.view(1, -1, 1)).sum(dim=1)


class HierarchicalCodec(nn.Module):
    """
    HRM — PyTorch reference implementation.

    Accepts bar_codec_features of shape [B, T+n, n_codec_outputs*2] (extent = T+n).
    Predicts codec features at T+n (pretrain) or trade parameters (trade mode).
    """

    def __init__(self, config: HRMConfig):
        super().__init__()
        self.config = config
        self.bar_feature_proj = nn.Linear(config.n_codec_outputs * 2, config.hidden_dim)
        self.ob_memory = TemporalOrderBook(config.hidden_dim, config.ob_depth_frames, config.ob_lookback_horizon)
        self.macro_regime_layer = MarketDepthLayer(config.hidden_dim, config.regime_attn_layers, config.n_heads, config.dropout)
        self.tactical_execution_layer = MarketDepthLayer(config.hidden_dim, config.tactical_attn_layers, config.n_heads, config.dropout)
        self.regime_state_init = nn.Parameter(torch.randn(config.hidden_dim) * 0.02)
        self.tactical_state_init = nn.Parameter(torch.randn(config.hidden_dim) * 0.02)
        self.codec_score_head = nn.Linear(config.hidden_dim, config.n_codec_outputs * 2)
        self.expected_return_head = nn.Linear(config.hidden_dim, 1)
        self.signal_conviction_head = nn.Linear(config.hidden_dim, 1)
        self.stop_loss_head = nn.Linear(config.hidden_dim, 1)
        self.take_profit_head = nn.Linear(config.hidden_dim, 1)
        self.position_size_head = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        bar_codec_features: torch.Tensor,
        memory: Optional[Tuple] = None,
        mode: str = "pretrain"
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        HRM forward pass.

        Args:
            bar_codec_features : [B, T+n, n_codec_outputs*2] — extent input (T bar window + n forward horizon)
            memory             : (temporal_ob_state, regime_state, tactical_state) or None
            mode               : "pretrain" | "trade"
        """
        B, T_plus_n, _ = bar_codec_features.shape
        temporal_ob_state, regime_state, tactical_state = memory if memory else (None, None, None)

        bar_embed = self.bar_feature_proj(bar_codec_features)
        current_bar_embed = bar_embed.mean(dim=1)
        temporal_ob_state = self.ob_memory.update(temporal_ob_state, current_bar_embed)
        market_context = self.ob_memory.read(temporal_ob_state)

        bar_features_with_context = bar_embed + market_context.unsqueeze(1).expand(-1, T_plus_n, -1)

        if regime_state is None:
            regime_state = self.regime_state_init.view(1, 1, -1).expand(B, T_plus_n, -1)
            tactical_state = self.tactical_state_init.view(1, 1, -1).expand(B, T_plus_n, -1)

        with torch.no_grad():
            for _ in range(self.config.regime_update_cycles - 1):
                for _ in range(self.config.tactical_update_cycles):
                    tactical_state = self.tactical_execution_layer(tactical_state, regime_state + bar_features_with_context)
                regime_state = self.macro_regime_layer(regime_state, tactical_state)

        for _ in range(self.config.tactical_update_cycles):
            tactical_state = self.tactical_execution_layer(tactical_state, regime_state + bar_features_with_context)
        regime_state = self.macro_regime_layer(regime_state, tactical_state)

        new_memory = (temporal_ob_state.detach(), regime_state.detach(), tactical_state.detach())
        regime_final = regime_state[:, -1, :]  # read from last position of extent

        if mode == "pretrain":
            output = self.codec_score_head(regime_final)
        else:
            pred_fwd_return = self.expected_return_head(regime_final)
            signal_conviction = torch.sigmoid(self.signal_conviction_head(regime_final))
            stop_loss_pct = torch.tanh(self.stop_loss_head(regime_final)) * 0.15
            take_profit_pct = torch.sigmoid(self.take_profit_head(regime_final)) * 0.30
            position_fraction = torch.sigmoid(self.position_size_head(regime_final))
            output = torch.cat([pred_fwd_return, signal_conviction, stop_loss_pct, take_profit_pct, position_fraction], dim=-1)

        return output, new_memory

    def pretrain_loss(self, bar_codec_features: torch.Tensor, memory: Optional[Tuple] = None):
        """World-model pre-training: predict last timestep's codec features (self-supervised)."""
        target = bar_codec_features[:, -1, :]
        pred, new_memory = self.forward(bar_codec_features, memory, mode="pretrain")
        return F.mse_loss(pred, target), new_memory

    def trade_loss(self, bar_codec_features: torch.Tensor, realized_returns: torch.Tensor, memory: Optional[Tuple] = None):
        """Alpha-maximisation: maximise conviction-weighted forward return with SL/TP clamping."""
        output, new_memory = self.forward(bar_codec_features, memory, mode="trade")
        pred_fwd_return = output[:, 0:1]
        signal_conviction = output[:, 1:2]
        stop_loss_pct = output[:, 2:3]
        take_profit_pct = output[:, 3:4]
        position_fraction = output[:, 4:5]

        entry = torch.ones_like(realized_returns).unsqueeze(-1)
        exit_price = entry * (1 + realized_returns.unsqueeze(-1))
        pred_dir = torch.sign(pred_fwd_return)
        sl = torch.abs(stop_loss_pct)
        tp = take_profit_pct
        long_exit = torch.clamp(exit_price, entry * (1 - sl), entry * (1 + tp))
        short_exit = torch.clamp(exit_price, entry * (1 - tp), entry * (1 + sl))
        exit_final = torch.where(pred_dir > 0, long_exit, short_exit)
        raw_pnl = (exit_final - entry) * position_fraction * 100
        final_pnl = torch.where(pred_dir > 0, raw_pnl, -raw_pnl)
        alpha_loss = -torch.mean(final_pnl)
        return alpha_loss, new_memory, pred_fwd_return, signal_conviction


class HRMTrainer:
    """
    PyTorch trainer for the HRM.

    pretrain_step : world-model loss (self-supervised)
    finetune_step : alpha loss (supervised)
    """
    def __init__(self, config: HRMConfig = None):
        self.config = config or HRMConfig()
        self.model = HierarchicalCodec(self.config)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

    def pretrain_step(self, bar_codec_features: torch.Tensor):
        self.optimizer.zero_grad()
        loss, _ = self.model.pretrain_loss(bar_codec_features)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def finetune_step(self, bar_codec_features: torch.Tensor, realized_returns: torch.Tensor):
        self.optimizer.zero_grad()
        loss, _, pred_fwd_return, signal_conviction = self.model.trade_loss(bar_codec_features, realized_returns)
        loss.backward()
        self.optimizer.step()
        return loss.item(), pred_fwd_return.detach(), signal_conviction.detach()

    def save(self, path: str):
        torch.save({'config': self.config, 'model_state': self.model.state_dict(), 'optimizer_state': self.optimizer.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])


# Legacy alias
CodecTrainer = HRMTrainer


if __name__ == "__main__":
    config = HRMConfig()
    hrm = HierarchicalCodec(config)
    print(f"HRM parameters: {sum(p.numel() for p in hrm.parameters()):,}")

    B, T, n = 4, 30, 2  # extent = T + n = 32
    bar_codec_features = torch.randn(B, T + n, config.n_codec_outputs * 2)
    output, memory = hrm(bar_codec_features, mode="pretrain")
    print(f"Pre-train output shape: {output.shape}")

    world_model_loss, _ = hrm.pretrain_loss(bar_codec_features)
    print(f"World model loss: {world_model_loss.item():.4f}")

    realized_returns = torch.randn(B, 1)
    alpha_loss, _, pred_fwd_return, signal_conviction = hrm.trade_loss(bar_codec_features, realized_returns)
    print(f"Alpha loss: {alpha_loss.item():.4f}, conviction: {signal_conviction[0].item():.4f}")
