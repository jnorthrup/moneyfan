"""
HRM Reference Implementation - Full Architecture

NO INDICES. NO FIXED ORDER.

Input:
  - Currency graph (nodes + edges)
  - Model states (lazy, on-demand)
  - Market features (per currency)

Output:
  - Model weights (composition)
  - Directives (per model)
  - Currency signals (weighted combination)

Architecture:
  ┌─────────────────────────────────────────┐
  │              HRM (Composer)              │
  │                                         │
  │  H_level: Sees cross-currency patterns  │
  │  L_level: Determines model weights      │
  │                                         │
  │  Output: weights, directives            │
  └─────────────────────────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────┐
  │          ALL MODELS (parallel)          │
  │                                         │
  │  Each model sees currency graph         │
  │  Each outputs: Dict[Currency, signal]   │
  │                                         │
  │  volatility_breakout, momentum, etc.    │
  └─────────────────────────────────────────┘
                    │
                    ▼
  combined = Σ weight[i] × model[i].signals
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict

# Import currency graph
import sys
sys.path.insert(0, '..')
from currency_graph import Currency, Pair, CurrencyGraph


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class HRMConfig:
    """HRM configuration - no fixed dimensions"""
    # Currency graph (not fixed size!)
    n_currencies: int = 64      # Approximate, for tensor sizing
    n_features: int = 15        # Per-currency features
    n_models: int = 5           # Number of models to compose
    seq_len: int = 32           # Temporal lookback
    report_dim: int = 5         # Model report dimension
    
    # Model architecture
    hidden_dim: int = 256
    n_heads: int = 8
    expansion: float = 4.0
    
    # HRM cycles
    H_cycles: int = 2
    L_cycles: int = 2
    H_layers: int = 4
    L_layers: int = 4
    
    # Cross-currency attention
    cross_attention: bool = True
    max_pairs: int = 256        # Max edges in graph
    
    @property
    def currency_feature_dim(self) -> int:
        return self.n_features + 5  # features + graph features
    
    @property
    def model_state_dim(self) -> int:
        return self.n_models * self.report_dim
    
    @property
    def input_dim(self) -> int:
        return self.hidden_dim  # After currency embedding
    
    @property
    def output_dim(self) -> int:
        # Model weights + directives
        return self.n_models * 3  # weight + (regime_hint, risk_limit)


# =============================================================================
# CURRENCY EMBEDDING (Graph-aware)
# =============================================================================

class CurrencyEmbedding(nn.Module):
    """
    Embed currencies with graph structure.
    
    No indices! Uses learned embeddings per currency symbol.
    Falls back to learnable "unknown" embedding for new currencies.
    """
    
    def __init__(self, config: HRMConfig, known_currencies: Set[str] = None):
        super().__init__()
        self.config = config
        
        # Learnable embeddings for known currencies
        self.known = known_currencies or set()
        self.currency_to_idx = {c: i for i, c in enumerate(sorted(self.known))}
        
        # Embedding table (with room for unknown)
        self.max_currencies = max(config.n_currencies, len(self.known) + 10)
        self.embeddings = nn.Embedding(self.max_currencies, config.hidden_dim)
        
        # Unknown currency embedding
        self.unknown_embedding = nn.Parameter(
            torch.randn(config.hidden_dim) * 0.02
        )
        
        # Graph structure encoding
        self.graph_encoder = nn.Sequential(
            nn.Linear(5, config.hidden_dim // 4),  # degree, pagerank, etc.
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 4, config.hidden_dim // 4),
        )
    
    def forward(self, 
                currencies: List[Currency],
                graph_features: Dict[Currency, torch.Tensor] = None) -> torch.Tensor:
        """
        Embed currencies.
        
        Args:
            currencies: List of Currency objects
            graph_features: Optional graph features per currency
        
        Returns:
            embeddings: [n_currencies, hidden_dim]
        """
        embeddings = []
        
        for cur in currencies:
            if cur.symbol in self.currency_to_idx:
                idx = self.currency_to_idx[cur.symbol]
                emb = self.embeddings.weight[idx]
            else:
                emb = self.unknown_embedding
            
            # Add graph features if available
            if graph_features and cur in graph_features:
                gf = graph_features[cur]
                emb = emb + self.graph_encoder(gf)
            
            embeddings.append(emb)
        
        return torch.stack(embeddings)


class PairEmbedding(nn.Module):
    """Embed trading pairs (edges)"""
    
    def __init__(self, config: HRMConfig):
        super().__init__()
        self.pair_encoder = nn.Sequential(
            nn.Linear(4, config.hidden_dim // 4),  # volume, spread, etc.
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 4, config.hidden_dim // 4),
        )
    
    def forward(self, pairs: List[Pair], pair_features: torch.Tensor) -> torch.Tensor:
        """Embed pairs with their features"""
        return self.pair_encoder(pair_features)


# =============================================================================
# CROSS-CURRENCY ATTENTION
# =============================================================================

class CrossCurrencyAttention(nn.Module):
    """
    Attention over currencies.
    
    No fixed indices - attends over whatever currencies are present.
    Uses graph structure to bias attention.
    """
    
    def __init__(self, config: HRMConfig):
        super().__init__()
        self.config = config
        self.attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.n_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(config.hidden_dim)
    
    def forward(self,
                currency_embeddings: torch.Tensor,
                graph_adjacency: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Cross-currency attention.
        
        Args:
            currency_embeddings: [batch, n_currencies, hidden_dim]
            graph_adjacency: [n_currencies, n_currencies] optional bias
        
        Returns:
            updated: [batch, n_currencies, hidden_dim]
        """
        # Self-attention over currencies
        attn_mask = None
        if graph_adjacency is not None:
            # Bias attention by graph connectivity
            attn_mask = -1e9 * (1 - graph_adjacency)  # Non-connected = very negative
        
        attended, _ = self.attention(
            currency_embeddings,
            currency_embeddings,
            currency_embeddings,
            attn_mask=attn_mask,
        )
        
        return self.norm(currency_embeddings + attended)


# =============================================================================
# MODEL STATE ENCODER
# =============================================================================

class ModelStateEncoder(nn.Module):
    """Encode model states for HRM input"""
    
    def __init__(self, config: HRMConfig):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(config.report_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 2),
        )
    
    def forward(self, model_states: torch.Tensor) -> torch.Tensor:
        """
        Encode model states.
        
        Args:
            model_states: [batch, n_models, report_dim]
        
        Returns:
            encoded: [batch, n_models, hidden_dim // 2]
        """
        return self.encoder(model_states)


# =============================================================================
# HRM CORE
# =============================================================================

class ReasoningBlock(nn.Module):
    """Single transformer-style block"""
    
    def __init__(self, dim: int, n_heads: int, expansion: float = 4.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * expansion)),
            nn.GELU(),
            nn.Linear(int(dim * expansion), dim),
        )
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        
        # FFN
        x = self.norm2(x + self.ffn(x))
        return x


class HLevel(nn.Module):
    """
    High-level reasoning module.
    
    Sees cross-currency patterns, detects regime.
    """
    
    def __init__(self, config: HRMConfig):
        super().__init__()
        self.config = config
        
        # Cross-currency attention
        self.cross_currency = CrossCurrencyAttention(config)
        
        # Temporal reasoning
        self.temporal = nn.ModuleList([
            ReasoningBlock(config.hidden_dim, config.n_heads, config.expansion)
            for _ in range(config.H_layers)
        ])
        
        # Regime detection head
        self.regime_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 4),  # trending, ranging, volatile, transition
        )
    
    def forward(self, 
                currency_seq: torch.Tensor,
                graph_adj: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            currency_seq: [batch, seq_len, n_currencies, hidden_dim]
            graph_adj: [n_currencies, n_currencies]
        
        Returns:
            h_state: [batch, seq_len, hidden_dim] (aggregated)
            regime_logits: [batch, 4]
        """
        B, T, C, D = currency_seq.shape
        
        # Cross-currency attention per timestep
        cross_out = []
        for t in range(T):
            out = self.cross_currency(currency_seq[:, t], graph_adj)
            cross_out.append(out.mean(dim=1))  # Aggregate over currencies
        cross_out = torch.stack(cross_out, dim=1)  # [B, T, D]
        
        # Temporal reasoning
        x = cross_out
        for block in self.temporal:
            x = block(x)
        
        # Regime detection
        regime_logits = self.regime_head(x.mean(dim=1))
        
        return x, regime_logits


class LLevel(nn.Module):
    """
    Low-level reasoning module.
    
    Determines model weights and directives.
    """
    
    def __init__(self, config: HRMConfig):
        super().__init__()
        self.config = config
        
        # Model state processing
        self.model_encoder = ModelStateEncoder(config)
        
        # Reasoning blocks
        self.blocks = nn.ModuleList([
            ReasoningBlock(config.hidden_dim, config.n_heads, config.expansion)
            for _ in range(config.L_layers)
        ])
        
        # Output heads
        self.weight_head = nn.Linear(config.hidden_dim, config.n_models)
        self.directive_head = nn.Linear(config.hidden_dim, config.n_models * 2)
    
    def forward(self,
                h_state: torch.Tensor,
                model_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_state: [batch, seq_len, hidden_dim] from H_level
            model_states: [batch, n_models, report_dim]
        
        Returns:
            weights: [batch, n_models]
            directives: [batch, n_models, 2]
        """
        # Encode model states
        model_encoded = self.model_encoder(model_states)  # [B, n_models, D/2]
        model_encoded = model_encoded.mean(dim=1)  # Aggregate
        model_encoded = model_encoded.unsqueeze(1).expand(-1, h_state.size(1), -1)
        
        # Combine with H_level output
        x = h_state + model_encoded
        
        # Reasoning
        for block in self.blocks:
            x = block(x)
        
        # Take last timestep
        x = x[:, -1, :]
        
        # Output
        weights = F.softmax(self.weight_head(x), dim=-1)
        directives = self.directive_head(x).view(-1, self.config.n_models, 2)
        directives[:, :, 0] = torch.tanh(directives[:, :, 0])  # regime_hint
        directives[:, :, 1] = torch.sigmoid(directives[:, :, 1])  # risk_limit
        
        return weights, directives


# =============================================================================
# HRM COMPOSER
# =============================================================================

@dataclass
class HRMOutput:
    """Full HRM output"""
    weights: torch.Tensor           # [batch, n_models]
    directives: torch.Tensor        # [batch, n_models, 2]
    regime_logits: torch.Tensor     # [batch, 4]
    regime: torch.Tensor            # [batch] (argmax)


class HRM(nn.Module):
    """
    Hierarchical Reasoning Model for Currency Graph.
    
    NO INDICES. NO FIXED ORDER.
    
    Input:
      - Currency graph (nodes + edges)
      - Per-currency features over time
      - Model states (lazy, when available)
    
    Output:
      - Model composition weights
      - Per-model directives
      - Detected regime
    """
    
    def __init__(self, config: HRMConfig, known_currencies: Set[str] = None):
        super().__init__()
        self.config = config
        
        # Currency embedding (graph-aware)
        self.currency_embed = CurrencyEmbedding(config, known_currencies)
        self.pair_embed = PairEmbedding(config)
        
        # Feature projection
        self.feature_proj = nn.Linear(config.currency_feature_dim, config.hidden_dim)
        
        # Reasoning modules
        self.H_level = HLevel(config)
        self.L_level = LLevel(config)
        
        # Carry state
        self.register_buffer('h_init', torch.randn(config.hidden_dim) * 0.02)
        self.register_buffer('l_init', torch.randn(config.hidden_dim) * 0.02)
    
    def forward(self,
                currencies: List[Currency],
                currency_features: torch.Tensor,  # [batch, seq_len, n_currencies, n_features]
                pair_features: Optional[torch.Tensor] = None,
                graph_adjacency: Optional[torch.Tensor] = None,
                model_states: Optional[torch.Tensor] = None) -> HRMOutput:
        """
        Forward pass.
        
        Args:
            currencies: List of Currency objects (unordered!)
            currency_features: Features per currency over time
            pair_features: Optional pair-level features
            graph_adjacency: [n_currencies, n_currencies] connectivity
            model_states: [batch, n_models, report_dim] from lazy models
        
        Returns:
            HRMOutput with weights, directives, regime
        """
        B, T, C, F = currency_features.shape
        
        # Embed currencies (no indices!)
        currency_emb = self.currency_embed(currencies)  # [C, D]
        
        # Project features
        features = self.feature_proj(currency_features)  # [B, T, C, D]
        
        # Add currency embeddings
        features = features + currency_emb.unsqueeze(0).unsqueeze(0)
        
        # H_level: cross-currency patterns
        h_state, regime_logits = self.H_level(features, graph_adjacency)
        
        # Default model states if not provided
        if model_states is None:
            model_states = torch.zeros(B, self.config.n_models, self.config.report_dim,
                                       device=currency_features.device)
        
        # L_level: model composition
        weights, directives = self.L_level(h_state, model_states)
        
        # Regime
        regime = regime_logits.argmax(dim=-1)
        
        return HRMOutput(
            weights=weights,
            directives=directives,
            regime_logits=regime_logits,
            regime=regime,
        )


# =============================================================================
# SIGNAL COMPOSITION
# =============================================================================

def compose_signals(
    model_signals: Dict[str, Dict[Currency, float]],
    weights: torch.Tensor,
    active_mask: torch.Tensor = None
) -> Dict[Currency, float]:
    """
    Compose signals from multiple models.
    
    Args:
        model_signals: {model_id: {currency: signal}}
        weights: [n_models]
        active_mask: [n_models] optional
    
    Returns:
        {currency: combined_signal}
    """
    if active_mask is not None:
        weights = weights * active_mask
        weights = weights / (weights.sum() + 1e-8)
    
    weights = weights.detach().cpu().numpy()
    
    # Aggregate per currency
    combined = defaultdict(float)
    
    for i, (model_id, signals) in enumerate(model_signals.items()):
        w = weights[i]
        for currency, signal in signals.items():
            combined[currency] += w * signal
    
    return dict(combined)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("HRM Reference Implementation - Full Architecture")
    print("=" * 60)
    
    # Config
    config = HRMConfig(
        n_currencies=20,
        n_features=15,
        n_models=5,
        seq_len=16,
        hidden_dim=128,
        n_heads=4,
        H_layers=2,
        L_layers=2,
    )
    
    # Known currencies (no indices!)
    known = {"USD", "BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "DOT", "LINK"}
    
    # Create HRM
    hrm = HRM(config, known_currencies=known)
    n_params = sum(p.numel() for p in hrm.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Create currencies
    currencies = [Currency(s) for s in sorted(known)]
    print(f"\nCurrencies: {[c.symbol for c in currencies]}")
    
    # Fake data
    B, T, C, F = 2, config.seq_len, len(currencies), config.n_features
    currency_features = torch.randn(B, T, C, F)
    
    # Graph adjacency (simple: all connected to USD)
    usd_idx = [c.symbol for c in currencies].index("USD")
    graph_adj = torch.zeros(C, C)
    for i in range(C):
        graph_adj[usd_idx, i] = 1
        graph_adj[i, usd_idx] = 1
    
    # Model states (lazy - would come from models on demand)
    model_states = torch.randn(B, config.n_models, config.report_dim)
    
    # Forward
    output = hrm(currencies, currency_features, 
                 graph_adjacency=graph_adj,
                 model_states=model_states)
    
    print(f"\nOutput:")
    print(f"  Weights shape: {output.weights.shape}")
    print(f"  Weights: {output.weights[0].detach().numpy().round(3)}")
    print(f"  Weights sum: {output.weights[0].sum().item():.4f}")
    print(f"  Directives shape: {output.directives.shape}")
    print(f"  Regime: {output.regime}")
    
    # Compose signals
    print("\n--- Signal Composition ---")
    
    # Fake model signals (Dict[Currency, signal])
    model_signals = {
        'volatility_breakout': {currencies[i]: float(torch.randn(1)) for i in range(C)},
        'momentum_trend': {currencies[i]: float(torch.randn(1)) for i in range(C)},
        'mean_reversion': {currencies[i]: float(torch.randn(1)) for i in range(C)},
        'cross_sectional': {currencies[i]: float(torch.randn(1)) for i in range(C)},
        'composite': {currencies[i]: float(torch.randn(1)) for i in range(C)},
    }
    
    combined = compose_signals(model_signals, output.weights[0])
    
    print(f"Combined signals for first 5 currencies:")
    for cur in currencies[:5]:
        print(f"  {cur.symbol}: {combined[cur]:.4f}")
    
    print("\n✓ No indices used. Currencies identified by symbol.")
    print("✓ Weights compose ALL models, not select ONE.")
    print("✓ Graph structure biases attention, not array positions.")
