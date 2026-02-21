"""
HRM — Hierarchical Reasoning Model Trading System
==================================================

Architecture:
┌──────────────────────────────────────────────────────────────────┐
│                    HRM (Meta-Allocator)                          │
│  - Reads regime_state + tactical_state from all codec experts    │
│  - Allocates notional across experts via allocation_confidence    │
│  - macro_regime_layer guides, tactical_execution_layer executes  │
└──────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│              INSTRUMENTS (Lazy Pandas Services)                   │
│  - tick_data      : Lazy[DataFrame]  — OHLCV per pair            │
│  - indicator_vecs : Lazy[DataFrame]  — computed features         │
│  - codec_scores   : Lazy[DataFrame]  — 24-expert signal panel    │
└──────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│            CODEC EXPERT PANEL (24 SOTA Experts)                  │
│  - volatility_breakout (PROVEN $37K)                             │
│  - momentum_trend                                                │
│  - mean_reversion                                                │
│  - cross_sectional (mapreduce)                                   │
│  - composite (agentic)                                           │
│  - [BACKLOG] sentiment_analysis                                  │
└──────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│            KERNELS (Numba-accelerated)                           │
│  - Rolling: mean, std, zscore, max, min, quantile                │
│  - Signal: volatility_breakout, momentum, reversion              │
│  - Cross-sectional: rank, zscore, neutralize                     │
└──────────────────────────────────────────────────────────────────┘

Extent definition:
    extent = T + n  (bar window T bars + prediction horizon n bars)
    The HRM sees T bars of context and predicts/acts on n bars forward.

Usage:
    from hrm.hierarchical_codec_mlx import HRMConfig, MLXHierarchicalCodec, MLXBasketTrainer
    config = HRMConfig(n_codec_outputs=24, ob_depth_frames=20)
    model  = MLXHierarchicalCodec(config)
"""

__version__ = "0.3.0"

# Minimal init — allows submodules to import cleanly without crashing
# on missing legacy components (e.g. instruments.py).

__all__ = []
