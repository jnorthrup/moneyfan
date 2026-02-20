"""
HRM Configuration for 128 Trade Pairs

Underfit for fail-fast testing, scalable to production.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class HRMConfig:
    """HRM configuration"""
    # Asset dimensions
    n_assets: int = 128          # 128 trade pairs (USD at index 0)
    n_features: int = 15         # Per-asset features (10 market + 5 time)
    n_models: int = 5            # Number of swarm models (quant + mapreduce + agentic)
    seq_len: int = 32            # Lookback (underfit: 32, prod: 128)
    report_dim: int = 5          # Model reports: energy, entropy, confidence, perf, active
    
    # Model dimensions - underfit
    hidden_dim: int = 128
    n_heads: int = 4
    expansion: float = 4.0
    
    # HRM structure - underfit
    H_cycles: int = 2            # High-level iterations
    L_cycles: int = 2            # Low-level iterations
    H_layers: int = 2            # Transformer blocks in H_level
    L_layers: int = 2            # Transformer blocks in L_level
    
    # Training
    lr: float = 1e-4
    weight_decay: float = 0.1
    batch_size: int = 32
    epochs: int = 1000
    
    # Derived dimensions
    @property
    def input_dim(self) -> int:
        # Market state + model reports (full duplex)
        # Option 1: Flatten all assets
        # Option 2: Cross-asset attention (we use this)
        return self.n_features + self.n_models * self.report_dim
    
    @property
    def output_dim(self) -> int:
        # Weights + directives per model
        return self.n_models + self.n_models * 2


@dataclass
class SwarmConfig:
    """Swarm model configuration"""
    n_assets: int = 128
    n_models: int = 5
    lookback: int = 20
    
    # Model types
    model_types: List[str] = field(default_factory=lambda: [
        "volatility_breakout",   # Proven winner (quant)
        "momentum_trend",        # Quant model
        "mean_reversion",        # Quant model
        "mapreduce_sector",      # Aggregate by sector
        "agentic_composite",     # Self-composing agent
    ])
    
    # Model weights (initial)
    initial_weights: Optional[List[float]] = None


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Data
    db_path: str = "hrm/data/coinbase.duckdb"
    train_start: str = "2024-01-01"
    train_end: str = "2024-12-31"
    val_start: str = "2025-01-01"
    val_end: str = "2025-01-31"
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    epochs: int = 1000
    warmup_steps: int = 100
    gradient_clip: float = 1.0
    
    # Checkpointing
    checkpoint_dir: str = "hrm/checkpoints"
    checkpoint_interval: int = 100
    
    # A/B testing
    run_ab_test: bool = True
    ab_test_interval: int = 10  # Every N epochs


# Underfit configs for fail-fast testing
UNDERFIT_HRM = HRMConfig(
    n_assets=128,
    n_features=15,
    n_models=5,
    seq_len=16,           # Very short for testing
    hidden_dim=64,        # Small
    H_layers=1,
    L_layers=1,
    batch_size=4,
    epochs=100,
)

UNDERFIT_SWARM = SwarmConfig(
    n_assets=128,
    n_models=5,
    lookback=10,
)

UNDERFIT_TRAINING = TrainingConfig(
    batch_size=4,
    epochs=100,
    checkpoint_interval=10,
)


# Production configs
PRODUCTION_HRM = HRMConfig(
    n_assets=128,
    n_features=15,
    n_models=5,
    seq_len=128,          # Longer lookback
    hidden_dim=512,       # Larger model
    n_heads=8,
    H_layers=4,
    L_layers=4,
    batch_size=64,
    epochs=10000,
)

PRODUCTION_SWARM = SwarmConfig(
    n_assets=128,
    n_models=5,
    lookback=50,
)

PRODUCTION_TRAINING = TrainingConfig(
    batch_size=64,
    epochs=10000,
    checkpoint_interval=500,
)


if __name__ == "__main__":
    print("HRM Configuration")
    print("=" * 50)
    
    config = UNDERFIT_HRM
    print(f"Assets: {config.n_assets}")
    print(f"Features per asset: {config.n_features}")
    print(f"Models in swarm: {config.n_models}")
    print(f"Sequence length: {config.seq_len}")
    print(f"Hidden dim: {config.hidden_dim}")
    print(f"Input dim: {config.input_dim}")
    print(f"Output dim: {config.output_dim}")
    
    # Estimate parameters
    # Rough estimate: input_proj + output_proj + H_level + L_level
    params = (
        config.input_dim * config.hidden_dim +  # input_proj
        config.hidden_dim * config.output_dim +  # output_proj
        2 * config.H_layers * (  # H_level
            config.hidden_dim * config.hidden_dim * 4 +  # attention
            config.hidden_dim * config.hidden_dim * 8     # ffn
        ) +
        2 * config.L_layers * (  # L_level
            config.hidden_dim * config.hidden_dim * 4 +
            config.hidden_dim * config.hidden_dim * 8
        )
    )
    print(f"Estimated params: {params:,}")
