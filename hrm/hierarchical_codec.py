"""
Hierarchical Codec - H/L level reasoning with sparkline memory

Architecture:
    Sparkline (per-pair, 100-200 frames) → Codec (H-level + L-level) → Output

Key features from HRM reference:
1. H/L nested cycles: H guides, L executes
2. Input injection: raw signals visible at both levels
3. State carry: z_H, z_L persist (working memory)
4. 1-step gradient: efficient training

Pre-training: predict next signals (self-supervised)
Fine-tuning: predict returns (supervised)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import math


@dataclass
class HierarchicalCodecConfig:
    n_signals: int = 24
    hidden_dim: int = 64
    sparkline_frames: int = 20
    sparkline_horizon: int = 200
    
    H_layers: int = 2
    L_layers: int = 2
    H_cycles: int = 2
    L_cycles: int = 3
    
    n_heads: int = 4
    dropout: float = 0.1


class ReasoningBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
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
        x = x + self.dropout(self.mlp(x))
        x = self.norm2(x)
        return x


class ReasoningLevel(nn.Module):
    def __init__(self, hidden_dim: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            ReasoningBlock(hidden_dim, n_heads, dropout) 
            for _ in range(n_layers)
        ])
        
    def forward(self, z: torch.Tensor, input_injection: torch.Tensor) -> torch.Tensor:
        z = z + input_injection
        for layer in self.layers:
            z = layer(z)
        return z


class SparklineMemory(nn.Module):
    def __init__(self, hidden_dim: int, n_frames: int = 20, horizon: int = 200):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_frames = n_frames
        self.horizon = horizon
        self.ratio = horizon ** (1.0 / max(n_frames - 1, 1))
        
    def update(self, sparkline: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        """
        Update sparkline with current embedding.
        Frame 0 = current, cascade to higher frames.
        """
        B = current.shape[0]
        
        if sparkline is None:
            sparkline = torch.zeros(B, self.n_frames, self.hidden_dim, device=current.device)
        
        new_frames = [current.unsqueeze(1)]
        
        for k in range(1, self.n_frames):
            h_k = self.ratio ** k
            alpha_k = 1.0 / h_k
            frame_k = (1.0 - alpha_k) * sparkline[:, k:k+1, :] + alpha_k * new_frames[k - 1]
            new_frames.append(frame_k)
        
        return torch.cat(new_frames, dim=1)
    
    def read(self, sparkline: torch.Tensor) -> torch.Tensor:
        """Read weighted context from sparkline (vanishing point)."""
        weights = torch.tensor([1.0 / (self.ratio ** k) for k in range(self.n_frames)], 
                               device=sparkline.device)
        weights = weights / weights.sum()
        context = (sparkline * weights.view(1, -1, 1)).sum(dim=1)
        return context


class HierarchicalCodec(nn.Module):
    """
    Codec with hierarchical H/L reasoning.
    
    Architecture:
        signals[t] → embed → sparkline update
                            ↓
        H-level (slow) ←→ L-level (fast) with nested cycles
                            ↓
        Output: signal_prediction[t+1] OR return_prediction[t+1]
    """
    
    def __init__(self, config: HierarchicalCodecConfig):
        super().__init__()
        self.config = config
        
        self.input_proj = nn.Linear(config.n_signals * 2, config.hidden_dim)
        
        self.sparkline = SparklineMemory(config.hidden_dim, config.sparkline_frames, config.sparkline_horizon)
        
        self.H_level = ReasoningLevel(config.hidden_dim, config.H_layers, config.n_heads, config.dropout)
        self.L_level = ReasoningLevel(config.hidden_dim, config.L_layers, config.n_heads, config.dropout)
        
        self.H_init = nn.Parameter(torch.randn(config.hidden_dim) * 0.02)
        self.L_init = nn.Parameter(torch.randn(config.hidden_dim) * 0.02)
        
        self.signal_head = nn.Linear(config.hidden_dim, config.n_signals * 2)
        self.return_head = nn.Linear(config.hidden_dim, 1)
        self.confidence_head = nn.Linear(config.hidden_dim, 1)
        self.stop_head = nn.Linear(config.hidden_dim, 1)
        self.tp_head = nn.Linear(config.hidden_dim, 1)
        self.pos_head = nn.Linear(config.hidden_dim, 1)
        
    def forward(
        self, 
        signals: torch.Tensor,
        memory: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        mode: str = "pretrain"
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass with hierarchical reasoning.
        
        Args:
            signals: [B, T, n_signals*2] - signal scores + confidences
            memory: (sparkline, z_H, z_L) from previous step
            mode: "pretrain" (predict signals) or "trade" (predict returns)
        
        Returns:
            output: predicted signals or (return, confidence)
            target: ground truth for loss (signals[t+1] or returns)
            new_memory: (sparkline, z_H, z_L)
        """
        B, T, _ = signals.shape
        
        sparkline, z_H, z_L = memory if memory else (None, None, None)
        
        x = self.input_proj(signals)
        
        current = x.mean(dim=1)
        sparkline = self.sparkline.update(sparkline, current)
        context = self.sparkline.read(sparkline)
        
        context_expanded = context.unsqueeze(1).expand(-1, T, -1)
        input_with_context = x + context_expanded
        
        if z_H is None:
            z_H = self.H_init.view(1, 1, -1).expand(B, T, -1)
            z_L = self.L_init.view(1, 1, -1).expand(B, T, -1)
        
        with torch.no_grad():
            for _h in range(self.config.H_cycles - 1):
                for _l in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_with_context)
                z_H = self.H_level(z_H, z_L)
        
        for _l in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_with_context)
        z_H = self.H_level(z_H, z_L)
        
        new_memory = (
            sparkline.detach(),
            z_H.detach(),
            z_L.detach()
        )
        
        if mode == "pretrain":
            output = self.signal_head(z_H[:, -1, :])
        else:
            ret = self.return_head(z_H[:, -1, :])
            conf = torch.sigmoid(self.confidence_head(z_H[:, -1, :]))
            
            # Add trading order parameters
            # stop_loss: -0.15 to 0 (negative = stop below entry)
            stop = torch.tanh(self.stop_head(z_H[:, -1, :])) * 0.15
            # take_profit: 0 to 0.30 (positive = target above entry)
            tp = torch.sigmoid(self.tp_head(z_H[:, -1, :])) * 0.30
            # position_size: 0 to 1 (fraction of capital)
            pos = torch.sigmoid(self.pos_head(z_H[:, -1, :]))
            
            output = torch.cat([ret, conf, stop, tp, pos], dim=-1)
        
        return output, new_memory
    
    def pretrain_loss(self, signals: torch.Tensor, memory: Optional[Tuple] = None):
        """
        Pre-training: predict next timestep's signals.
        Self-supervised - no labels needed.
        """
        # The forward pass outputs prediction for the last timestep only
        # signals_target should be the last timestep's signals
        signals_target = signals[:, -1, :]
        
        pred, new_memory = self.forward(signals, memory, mode="pretrain")
        
        loss = F.mse_loss(pred, signals_target)
        
        return loss, new_memory
    
    def trade_loss(self, signals: torch.Tensor, returns: torch.Tensor, memory: Optional[Tuple] = None):
        """
        Fine-tuning: predict returns.
        Supervised - uses actual returns.
        """
        output, new_memory = self.forward(signals, memory, mode="trade")
        
        # Unpack outputs
        pred_return = output[:, 0:1]
        confidence = output[:, 1:2]
        stop_loss = output[:, 2:3]
        take_profit = output[:, 3:4]
        position_size = output[:, 4:5]
        
        # Trading logic with stops and position sizing
        # pred_return is [-1,1], confidence is [0,1]
        # stop_loss is [-0.15, 0], take_profit is [0, 0.30]
        # position_size is [0,1]
        
        # Expected move
        expected_move = torch.abs(pred_return) * confidence
        
        # Exit price simulation
        entry_price = torch.ones_like(returns).unsqueeze(-1)
        
        # Apply stop loss and take profit
        # Long position: stop below entry, TP above entry
        # Short position: stop above entry, TP below entry
        pred_dir = torch.sign(pred_return)
        actual_dir = torch.sign(returns.unsqueeze(-1))
        
        # Exit price with stops
        exit_price = entry_price * (1 + returns.unsqueeze(-1))
        
        # Stop loss and take profit bounds
        sl_pct = torch.abs(stop_loss)
        tp_pct = take_profit
        
        # For long: min(exit, entry*(1+tp_pct)) and max(exit, entry*(1-sl_pct))
        long_exit = torch.clamp(exit_price, entry_price * (1 - sl_pct), entry_price * (1 + tp_pct))
        
        # For short: inverted
        short_exit = torch.clamp(exit_price, entry_price * (1 - tp_pct), entry_price * (1 + sl_pct))
        
        # Choose exit based on position direction
        exit_price_final = torch.where(pred_dir > 0, long_exit, short_exit)
        
        # PnL calculation with position sizing
        # Base pnl = (exit - entry) * position_size * capital
        raw_pnl = (exit_price_final - entry_price) * position_size * 100
        
        # Final pnl (positive = win, negative = loss)
        final_pnl = torch.where(pred_dir > 0, raw_pnl, -raw_pnl)
        
        # Loss is negative of mean pnl to maximize
        loss = -torch.mean(final_pnl)
        
        return loss, new_memory, pred_return, confidence


class CodecTrainer:
    """
    Trainer for hierarchical codec.
    
    Two-phase training:
    1. Pre-train: predict signals (self-supervised)
    2. Fine-tune: predict returns (supervised)
    """
    
    def __init__(self, config: HierarchicalCodecConfig = None):
        self.config = config or HierarchicalCodecConfig()
        self.model = HierarchicalCodec(self.config)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
    def pretrain_step(self, signals_batch: torch.Tensor):
        """One pre-training step."""
        self.optimizer.zero_grad()
        loss, _ = self.model.pretrain_loss(signals_batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def finetune_step(self, signals_batch: torch.Tensor, returns_batch: torch.Tensor):
        """One fine-tuning step."""
        self.optimizer.zero_grad()
        loss, _, pred_ret, conf = self.model.trade_loss(signals_batch, returns_batch)
        loss.backward()
        self.optimizer.step()
        return loss.item(), pred_ret.detach(), conf.detach()
    
    def save(self, path: str):
        torch.save({
            'config': self.config,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }, path)
        
    def load(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])


if __name__ == "__main__":
    config = HierarchicalCodecConfig()
    codec = HierarchicalCodec(config)
    
    print(f"HierarchicalCodec parameters: {sum(p.numel() for p in codec.parameters()):,}")
    
    B, T = 4, 32
    signals = torch.randn(B, T, config.n_signals * 2)
    
    output, memory = codec(signals, mode="pretrain")
    print(f"Pre-train output shape: {output.shape}")
    
    loss, new_memory = codec.pretrain_loss(signals)
    print(f"Pre-train loss: {loss.item():.4f}")
    
    returns = torch.randn(B, 1)
    loss, _, pred_ret, conf = codec.trade_loss(signals, returns)
    print(f"Trade loss: {loss.item():.4f}, predicted return: {pred_ret[0].item():.4f}, confidence: {conf[0].item():.4f}")
