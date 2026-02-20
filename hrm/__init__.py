"""
HRM Trading System

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                      HRM (Controller)                        │
│  - Reads all model states                                    │
│  - Chooses ONE model to act                                  │
│  - Lazy evaluation: only computes what's selected            │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              INSTRUMENTS (Lazy Pandas Services)              │
│  - market_data: Lazy[DataFrame]                              │
│  - features: Lazy[DataFrame]                                 │
│  - signals: Lazy[DataFrame]                                  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    MODELS (Quant Tools)                      │
│  - volatility_breakout (PROVEN $37K)                         │
│  - momentum_trend                                            │
│  - mean_reversion                                            │
│  - cross_sectional (mapreduce)                               │
│  - composite (agentic)                                       │
│  - [BACKLOG] sentiment_analysis                              │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    KERNELS (Numba-accelerated)               │
│  - Rolling: mean, std, zscore, max, min, quantile            │
│  - Signal: volatility_breakout, momentum, reversion          │
│  - Cross-sectional: rank, zscore, neutralize                 │
└─────────────────────────────────────────────────────────────┘

Usage:
    # Create instruments
    from hrm.instruments import LazyInstrument, InstrumentRegistry
    
    registry = InstrumentRegistry()
    registry.register('market_data', LazyInstrument('market_data', load_market))
    
    # Create models
    from hrm.models import create_all_models, read_states
    models = create_all_models()
    
    # HRM reads states and chooses
    states = read_states(models)
    # ... HRM decision logic ...
    
    # Activate chosen model
    signals = models[chosen_id].compute_signals(registry)
"""

__version__ = "0.2.0"

# Core components
from .instruments import (
    LazyInstrument,
    InstrumentRegistry,
    QuantModel,
    registry,
)
try:
    from .kernels import (
        rolling_mean_kernel,
        rolling_std_kernel,
        rolling_zscore_kernel,
        volatility_breakout_kernel,
        momentum_trend_kernel,
        mean_reversion_kernel,
        cross_sectional_rank,
        RollingMean,
        RollingStd,
        RollingZScore,
    )
except ImportError:
    # Fallback or silent failure if kernels/numba not available
    # This allows other modules like signal_hrm to be imported
    rolling_mean_kernel = None
    rolling_std_kernel = None
    rolling_zscore_kernel = None
    volatility_breakout_kernel = None
    momentum_trend_kernel = None
    mean_reversion_kernel = None
    cross_sectional_rank = None
    RollingMean = None
    RollingStd = None
    RollingZScore = None
from .models import (
    ModelSpec,
    SPECS,
    create_model,
    create_all_models,
    read_states,
)
from .features import (
    add_time_features,
    add_price_features,
    compute_all_features,
    compute_all_signals,
)
from .spreadsheet import Spreadsheet
from .training_framework import (
    CANONICAL_PIPELINE,
    DEFAULT_COMPETITOR_MODELS_24,
    CandidateMetrics,
    FractalVerdict,
    FailFastResult,
    FailFastThresholds,
    HRMTrainingFramework,
    ObjectiveWeights,
    TrainingScrews,
    BenchmarkVerdict,
    build_fractal_screws_grid,
    build_coinpair_graph,
    build_data_failfast_metrics,
    compute_liquidity_flow_capture,
    compute_weighted_objective,
    evaluate_failfast,
    judge_fractal_winner,
    sample_stochastic_expanse,
)
from .convergence import (
    ModelSignalConvergenceTracker,
    convergence_from_snapshot,
    rolling_convergence,
)

# Config
from .config.assets import TRADE_PAIRS, COINBASE_PAIRS, SECTORS

__all__ = [
    # Instruments
    'LazyInstrument',
    'InstrumentRegistry', 
    'QuantModel',
    'registry',
    # Kernels
    'rolling_mean_kernel',
    'rolling_std_kernel',
    'rolling_zscore_kernel',
    'volatility_breakout_kernel',
    'momentum_trend_kernel',
    'mean_reversion_kernel',
    'cross_sectional_rank',
    'RollingMean',
    'RollingStd',
    'RollingZScore',
    # Models
    'ModelSpec',
    'SPECS',
    'create_model',
    'create_all_models',
    'read_states',
    # Features
    'add_time_features',
    'add_price_features',
    'compute_all_features',
    'compute_all_signals',
    # Spreadsheet
    'Spreadsheet',
    # Training framework
    'CANONICAL_PIPELINE',
    'DEFAULT_COMPETITOR_MODELS_24',
    'CandidateMetrics',
    'FractalVerdict',
    'FailFastResult',
    'FailFastThresholds',
    'HRMTrainingFramework',
    'ObjectiveWeights',
    'TrainingScrews',
    'BenchmarkVerdict',
    'build_fractal_screws_grid',
    'build_coinpair_graph',
    'build_data_failfast_metrics',
    'compute_liquidity_flow_capture',
    'compute_weighted_objective',
    'evaluate_failfast',
    'judge_fractal_winner',
    'sample_stochastic_expanse',
    # Convergence
    'ModelSignalConvergenceTracker',
    'convergence_from_snapshot',
    'rolling_convergence',
    # Config
    'TRADE_PAIRS',
    'COINBASE_PAIRS',
    'SECTORS',
]
