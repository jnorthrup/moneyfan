"""
Codecs - 24 separate signal codec models + HRM that emulates them

GOALS.md:
- 24 Codecs: Each a small 2-layer ML model
- HRM: Learns to emulate all 24 codec outputs
- Test-time: Only run HRM (no codecs)

Architecture:
1. 24 Codecs: small 2-layer, 32 hidden, outputs [confidence, direction, regime_fit]
2. 1 HRM: larger 2-layer, 128 hidden, outputs 24 x [confidence, direction, regime_fit]
3. HRM learns from same inputs as codecs (NOT codec outputs)
4. HRM replaces codecs in test-time
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class CodecConfig:
    """Configuration for codec models"""
    n_inputs: int = 15  # Number of instrument-metric inputs
    hidden_dim: int = 32  # Small 2-layer model
    n_layers: int = 2
    output_dim: int = 3  # [confidence, direction, regime_fit]


@dataclass
class HRMConfig:
    """Configuration for HRM model"""
    n_inputs: int = 15  # Same instrument-metric inputs as codecs
    hidden_dim: int = 128  # Larger than codecs
    n_codecs: int = 24  # Number of codecs to emulate
    n_layers: int = 2


class Codec(nn.Module):
    """
    A single codec model - small 2-layer ML model
    
    Each codec is trained independently for a specific SOTA strategy
    """
    def __init__(self, codec_id: str, config: CodecConfig):
        super().__init__()
        self.codec_id = codec_id
        self.config = config
        
        # Small 2-layer network
        self.net = nn.Sequential(
            nn.Linear(config.n_inputs, config.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.output_dim),
        )
        
        # Initialize
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [batch, n_inputs] - instrument-metric inputs
            
        Returns:
            [batch, 3] - [confidence, direction, regime_fit]
        """
        x = self.net(inputs)
        # Apply activation functions
        confidence = torch.sigmoid(x[..., 0:1])  # [0, 1]
        direction = torch.tanh(x[..., 1:2])      # [-1, 1]
        regime_fit = torch.sigmoid(x[..., 2:3])  # [0, 1]
        return torch.cat([confidence, direction, regime_fit], dim=-1)
    
    def compute_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Codec-specific loss (each codec has different reward surface)
        
        Args:
            inputs: [batch, n_inputs]
            targets: [batch, 3] - [expected_confidence, expected_direction, expected_regime_fit]
        """
        outputs = self.forward(inputs)
        
        # Different loss for each output component
        # Confidence: BCE loss (binary classification)
        confidence_loss = nn.BCELoss()(outputs[..., 0], targets[..., 0])
        
        # Direction: MSE loss (regression)
        direction_loss = nn.MSELoss()(outputs[..., 1], targets[..., 1])
        
        # Regime fit: BCE loss (binary classification)
        regime_loss = nn.BCELoss()(outputs[..., 2], targets[..., 2])
        
        # Weighted combination (can be codec-specific)
        total_loss = confidence_loss + direction_loss + regime_loss
        
        return total_loss


class CodecCollection:
    """
    Collection of 24 codec models - each trained independently
    """
    def __init__(self, n_codecs: int = 24, config: CodecConfig = None):
        self.n_codecs = n_codecs
        self.config = config or CodecConfig()
        self.codecs: Dict[str, Codec] = {}
        
        # Create 24 codec models with agent names from GOALS.md
        # Each codec is trained independently for different SOTA strategies
        agent_names = [
            "volatility_breakout",
            "momentum_trend",
            "mean_reversion",
            "trend_following",
            "pairs_trading",
            "grid_trading",
            "volume_profile",
            "order_flow",
            "correlation_trading",
            "liquidity_making",
            "sector_rotation",
            "composite_alpha",
            "rsi_reversal",
            "bollinger_bands",
            "macd_cross",
            "atr_breakout",
            "tick_momentum",
            "dca_baseline",
            "technical_ml",
            "hrm_mean_reversion",
            "volatility_x_momentum",
            "mean_reversion_v2",
            "sector_rotation_v2",
            "composite_trend",
        ]
        
        for i, agent_name in enumerate(agent_names):
            codec = Codec(agent_name, self.config)
            self.codecs[agent_name] = codec
    
    def forward_all(self, instrument_metrics: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run all codecs on same inputs
        
        Args:
            instrument_metrics: [batch, n_inputs]
            
        Returns:
            Dict[codec_id, [batch, 3]] - codec outputs
        """
        results = {}
        for codec_id, codec in self.codecs.items():
            results[codec_id] = codec(instrument_metrics)
        return results
    
    def train_all(self, 
                  instrument_metrics: torch.Tensor, 
                  targets: Dict[str, torch.Tensor],
                  n_epochs: int = 10) -> Dict[str, List[float]]:
        """
        Train all codecs independently
        
        Args:
            instrument_metrics: [batch, n_inputs]
            targets: Dict[codec_id, [batch, 3]] - target outputs for each codec
            n_epochs: number of training epochs
            
        Returns:
            Dict[codec_id, List[loss]] - training history per codec
        """
        history = {codec_id: [] for codec_id in self.codecs.keys()}
        
        for epoch in range(n_epochs):
            for codec_id, codec in self.codecs.items():
                codec.train()
                codec.optimizer.zero_grad()
                
                # Each codec uses same inputs but different targets
                loss = codec.compute_loss(instrument_metrics, targets[codec_id])
                loss.backward()
                codec.optimizer.step()
                
                history[codec_id].append(loss.item())
                
        return history


class HRM(nn.Module):
    """
    HRM meta-allocator - emulates 24 codec outputs
    
    GOALS.md: HRM learns to emulate those other 24 channels to replace them in test-time
    
    - Input: instrument-metric inputs (same as codecs)
    - Output: 24 x [confidence, direction, regime_fit] (72 values)
    - Goal: predict what each codec would output
    - Test-time: Only run HRM, not 24 codecs
    """
    def __init__(self, config: HRMConfig):
        super().__init__()
        self.config = config
        
        # Agent names for reference
        self.agent_names = [
            "volatility_breakout",
            "momentum_trend",
            "mean_reversion",
            "trend_following",
            "pairs_trading",
            "grid_trading",
            "volume_profile",
            "order_flow",
            "correlation_trading",
            "liquidity_making",
            "sector_rotation",
            "composite_alpha",
            "rsi_reversal",
            "bollinger_bands",
            "macd_cross",
            "atr_breakout",
            "tick_momentum",
            "dca_baseline",
            "technical_ml",
            "hrm_mean_reversion",
            "volatility_x_momentum",
            "mean_reversion_v2",
            "sector_rotation_v2",
            "composite_trend",
        ]
        
        # HRM network - larger than codecs
        # Output: n_codecs * 3 values
        self.net = nn.Sequential(
            nn.Linear(config.n_inputs, config.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.n_codecs * 3),  # 24 * 3 = 72 values
        )
        
        # Initialize
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
    
    def forward(self, instrument_metrics: torch.Tensor) -> torch.Tensor:
        """
        HRM predicts all 24 codec outputs from instrument-metrics
        
        Args:
            instrument_metrics: [batch, n_inputs] - same as codecs
            
        Returns:
            [batch, n_codecs, 3] - 24 codec outputs
        """
        batch_size = instrument_metrics.shape[0]
        output = self.net(instrument_metrics)  # [batch, n_codecs * 3]
        
        # Reshape to [batch, n_codecs, 3]
        output = output.view(batch_size, self.config.n_codecs, 3)
        
        # Apply activation functions
        confidence = torch.sigmoid(output[..., 0:1])  # [0, 1]
        direction = torch.tanh(output[..., 1:2])      # [-1, 1]
        regime_fit = torch.sigmoid(output[..., 2:3])  # [0, 1]
        
        return torch.cat([confidence, direction, regime_fit], dim=-1)
    
    def compute_loss(self, 
                     instrument_metrics: torch.Tensor,
                     codec_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        HRM loss: MSE between HRM outputs and ALL codec outputs
        
        Args:
            instrument_metrics: [batch, n_inputs]
            codec_outputs: Dict[agent_name, [batch, 3]]
        """
        # HRM predictions
        hrm_output = self.forward(instrument_metrics)  # [batch, n_codecs, 3]
        
        # Stack codec outputs in same order as agent_names
        codec_tensor = torch.stack([codec_outputs[agent_name] 
                                   for agent_name in self.agent_names], dim=1)
        
        # MSE loss between HRM and codec outputs
        loss = nn.MSELoss()(hrm_output, codec_tensor)
        
        return loss
    
    def forward_with_names(self, instrument_metrics: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        HRM predictions as dictionary with agent names
        
        Args:
            instrument_metrics: [batch, n_inputs]
            
        Returns:
            Dict[agent_name, [batch, 3]]
        """
        output = self.forward(instrument_metrics)
        results = {}
        for i, agent_name in enumerate(self.agent_names):
            results[agent_name] = output[:, i, :]
        return results
