"""
HRM Reference Implementation (PyTorch)
Manages Energy Quant Models swarm

Architecture:
- H_level: Sees global market patterns, determines regime
- L_level: Tactical decisions on model weights
- Output: Swarm model weights

Underfit config for fail-fast testing.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List


# =============================================================================
# INITIALIZATION & UTILS
# =============================================================================

def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, 
                       lower: float = -2.0, upper: float = 2.0) -> torch.Tensor:
    """Truncated normal initialization"""
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2
            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)
            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)
    return tensor


def rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMS normalization"""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


# =============================================================================
# LAYERS
# =============================================================================

class Linear(nn.Module):
    """Linear with truncated normal init"""
    def __init__(self, in_dim: int, out_dim: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty(out_dim, in_dim), std=1.0 / (in_dim ** 0.5))
        )
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class Attention(nn.Module):
    """Multi-head attention with RoPE"""
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = Linear(dim, 3 * dim)
        self.out = Linear(dim, dim)
    
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # Transpose for attention [B, n_heads, T, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply RoPE
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out(out)
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward"""
    def __init__(self, dim: int, mult: float = 4.0):
        super().__init__()
        hidden = int(mult * dim * 2 / 3)
        hidden = (hidden + 255) // 256 * 256  # Round to 256
        self.gate_up = Linear(dim, 2 * hidden)
        self.down = Linear(hidden, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Block(nn.Module):
    """Transformer block"""
    def __init__(self, dim: int, n_heads: int, mult: float = 4.0):
        super().__init__()
        self.attn = Attention(dim, n_heads)
        self.ffn = SwiGLU(dim, mult)
        self.eps = 1e-5
    
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = rms_norm(x + self.attn(x, cos, sin), self.eps)
        x = rms_norm(x + self.ffn(x), self.eps)
        return x


class RoPE(nn.Module):
    """Rotary Position Embedding"""
    def __init__(self, dim: int, max_seq: int, base: float = 10000.0):
        super().__init__()
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        pos = torch.arange(max_seq)
        angles = torch.outer(pos, freqs)
        self.register_buffer('cos', torch.cat([angles.cos(), angles.cos()], dim=-1))
        self.register_buffer('sin', torch.cat([angles.sin(), angles.sin()], dim=-1))
    
    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos[:seq_len], self.sin[:seq_len]


# =============================================================================
# HRM CONFIG
# =============================================================================

@dataclass
class HRMConfig:
    """HRM configuration - underfit for fast testing"""
    # Data dimensions
    n_assets: int = 43
    n_features: int = 10
    n_models: int = 3       # Number of swarm models
    seq_len: int = 16       # Underfit (small for testing)
    report_dim: int = 5     # Energy, entropy, confidence, perf, active_ratio
    
    # Model dimensions - underfit
    hidden_dim: int = 64
    n_heads: int = 4
    
    # HRM structure - underfit
    H_cycles: int = 2       # High-level iterations
    L_cycles: int = 2       # Low-level iterations  
    H_layers: int = 2       # Underfit
    L_layers: int = 2       # Underfit
    
    # Derived
    @property
    def input_dim(self) -> int:
        # Market state + model reports (full duplex)
        return self.n_assets * self.n_features + self.n_models * self.report_dim
    
    @property
    def output_dim(self) -> int:
        # Model weights + directives (full duplex)
        # weights: n_models, directive per model: 2 (regime_hint, risk_limit)
        return self.n_models + self.n_models * 2


# =============================================================================
# HRM MODEL
# =============================================================================

class ReasoningModule(nn.Module):
    """Stack of transformer blocks (H_level or L_level)"""
    def __init__(self, dim: int, n_heads: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            Block(dim, n_heads) for _ in range(n_layers)
        ])
    
    def forward(self, x: torch.Tensor, injection: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + injection
        for layer in self.layers:
            x = layer(x, cos, sin)
        return x


@dataclass  
class HRMCarry:
    """State carried through iterations"""
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class HRMOutput:
    """Full duplex output from HRM to swarm"""
    weights: torch.Tensor           # [batch, n_models]
    directives: torch.Tensor        # [batch, n_models, 2] (regime_hint, risk_limit)
    
    def get_weights(self) -> torch.Tensor:
        return self.weights
    
    def get_directives(self) -> List:
        """Convert to list of ModelDirective-like dicts"""
        B, M, _ = self.directives.shape
        result = []
        for b in range(B):
            batch_directives = []
            for m in range(M):
                batch_directives.append({
                    'regime_hint': self.directives[b, m, 0].item(),
                    'risk_limit': self.directives[b, m, 1].item(),
                })
            result.append(batch_directives)
        return result


class HRM(nn.Module):
    """
    Hierarchical Reasoning Model
    
    ONE HRM controls the swarm of energy quant models.
    Rolling marble energy: H_level → L_level → Swarm weights
    
    FULL DUPLEX I/O:
    
    INBOUND:
      market_state: [batch, seq_len, n_assets × n_features]
      model_reports: [batch, seq_len, n_models × report_dim]
    
    OUTBOUND:
      weights: [batch, n_models] - model selection weights
      directives: [batch, n_models, 2] - per-model control signals
    """
    
    def __init__(self, config: HRMConfig):
        super().__init__()
        self.config = config
        
        # Input/output projections
        self.embed_scale = math.sqrt(config.hidden_dim)
        self.input_proj = Linear(config.input_dim, config.hidden_dim)
        self.output_proj = Linear(config.hidden_dim, config.output_dim, bias=True)
        
        # Positional encoding
        self.rope = RoPE(config.hidden_dim // config.n_heads, config.seq_len)
        
        # Reasoning modules
        self.H_level = ReasoningModule(config.hidden_dim, config.n_heads, config.H_layers)
        self.L_level = ReasoningModule(config.hidden_dim, config.n_heads, config.L_layers)
        
        # Initial states (learnable)
        self.H_init = nn.Parameter(trunc_normal_init_(torch.empty(config.hidden_dim)))
        self.L_init = nn.Parameter(trunc_normal_init_(torch.empty(config.hidden_dim)))
    
    def forward(self, 
                market_state: torch.Tensor,
                model_reports: Optional[torch.Tensor] = None,
                carry: Optional[HRMCarry] = None) -> Tuple[HRMCarry, HRMOutput]:
        """
        Forward pass with full duplex I/O
        
        Args:
            market_state: [batch, seq_len, n_assets × n_features]
            model_reports: [batch, seq_len, n_models × report_dim] from swarm
            carry: Optional previous state
        
        Returns:
            carry: New state
            output: HRMOutput with weights and directives
        """
        B, T, _ = market_state.shape
        
        # Concatenate market state + model reports (full duplex inbound)
        if model_reports is None:
            model_reports = torch.zeros(B, T, self.config.n_models * self.config.report_dim,
                                       device=market_state.device, dtype=market_state.dtype)
        
        x = torch.cat([market_state, model_reports], dim=-1)
        
        # Initialize or use carry
        if carry is None:
            z_H = self.H_init.view(1, 1, -1).expand(B, T, -1)
            z_L = self.L_init.view(1, 1, -1).expand(B, T, -1)
        else:
            z_H, z_L = carry.z_H, carry.z_L
        
        # Input embedding
        x_embed = self.embed_scale * self.input_proj(x)
        
        # Positional encoding
        cos, sin = self.rope(T)
        
        # =================================================================
        # ITERATIVE REFINEMENT
        # =================================================================
        # Most iterations without gradient (faster)
        with torch.no_grad():
            for _ in range(self.config.H_cycles - 1):
                for _ in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + x_embed, cos, sin)
                z_H = self.H_level(z_H, z_L, cos, sin)
        
        # Last iteration with gradient
        for _ in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + x_embed, cos, sin)
        z_H = self.H_level(z_H, z_L, cos, sin)
        
        # New carry (detached)
        new_carry = HRMCarry(z_H.detach(), z_L.detach())
        
        # Output: take last position, project to full output
        output = self.output_proj(z_H[:, -1, :])  # [batch, n_models + n_models*2]
        
        # Split into weights and directives
        n_models = self.config.n_models
        weights_raw = output[:, :n_models]           # [batch, n_models]
        directives_raw = output[:, n_models:]        # [batch, n_models*2]
        
        # Softmax for weights
        weights = F.softmax(weights_raw, dim=-1)
        
        # Tanh for directives (regime_hint: -1 to 1, sigmoid for risk_limit: 0 to 1)
        directives = directives_raw.view(B, n_models, 2)
        directives[:, :, 0] = torch.tanh(directives[:, :, 0])      # regime_hint
        directives[:, :, 1] = torch.sigmoid(directives[:, :, 1])   # risk_limit
        
        return new_carry, HRMOutput(weights=weights, directives=directives)


# =============================================================================
# TRAINING UTILS
# =============================================================================

def compute_loss(hrm_output: HRMOutput, returns: torch.Tensor,
                 prev_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute loss for HRM training
    
    Args:
        hrm_output: HRMOutput with weights and directives
        returns: [batch, n_models] - actual returns from each model
        prev_weights: Previous weights (for turnover penalty)
    
    Returns:
        loss: Negative risk-adjusted return
    """
    weights = hrm_output.weights
    
    # Portfolio return
    portfolio_return = (weights * returns).sum(dim=-1)
    
    # Loss: negative return (we want to maximize)
    loss = -portfolio_return.mean()
    
    # Optional: turnover penalty
    if prev_weights is not None:
        turnover = (weights - prev_weights).abs().sum(dim=-1).mean()
        loss = loss + 0.1 * turnover
    
    return loss


if __name__ == "__main__":
    print("Testing HRM Reference Implementation with Full Duplex I/O...\n")
    
    # Underfit config
    config = HRMConfig(
        n_assets=43,
        n_features=10,
        n_models=3,
        seq_len=16,
        hidden_dim=64,
        n_heads=4,
        H_cycles=2,
        L_cycles=2,
        H_layers=2,
        L_layers=2,
    )
    
    model = HRM(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    print(f"Input dim: {config.input_dim} (market: {43*10} + reports: {3*5})")
    print(f"Output dim: {config.output_dim} (weights: 3 + directives: {3*2})")
    
    # Test forward
    market_state = torch.randn(2, config.seq_len, 43 * 10)
    model_reports = torch.randn(2, config.seq_len, 3 * 5)  # From swarm
    
    carry, output = model(market_state, model_reports)
    
    print(f"\nMarket state: {market_state.shape}")
    print(f"Model reports: {model_reports.shape}")
    print(f"Weights: {output.weights.shape}")
    print(f"Weights sample: {output.weights[0].detach().numpy()}")
    print(f"Directives: {output.directives.shape}")
    print(f"Directives sample: {output.directives[0].detach().numpy()}")
    
    # Test training step
    returns = torch.randn(2, config.n_models) * 0.1  # Fake returns
    loss = compute_loss(output, returns)
    loss.backward()
    
    print(f"\nLoss: {loss.item():.4f}")
    print("Gradients computed successfully")
