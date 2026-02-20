"""
HRM Composition - How HRM combines models

NOT: "pick one model"
BUT: "compose models optimally"

Options:
1. Softmax ensemble: weight all models
2. Sparse selection: zero out bad ones  
3. Regime-conditional: different weights per regime
4. Learned composition: HRM is the composer
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class CompositionMode(Enum):
    ENSEMBLE = "ensemble"           # Weighted average of ALL models
    SPARSE = "sparse"               # Zero out poorly performing models
    REGIME_CONDITIONAL = "regime"   # Different weights per market regime
    LEARNED = "learned"             # HRM learns composition function


@dataclass
class CompositionOutput:
    """HRM's composition decision"""
    weights: np.ndarray              # [n_models] weight distribution
    active_mask: np.ndarray          # [n_models] which models to use
    regime: Optional[str]            # Detected regime (if applicable)
    method: str                      # Which composition method was used
    
    def apply(self, model_signals: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply composition to model signals"""
        # Stack signals
        signal_matrix = np.stack(list(model_signals.values()))  # [n_models, n_assets]
        
        # Apply mask and weights
        active_weights = self.weights * self.active_mask
        active_weights = active_weights / (active_weights.sum() + 1e-8)  # Renormalize
        
        # Weighted combination
        combined = np.sum(signal_matrix * active_weights[:, np.newaxis], axis=0)
        
        return combined


# =============================================================================
# COMPOSITION STRATEGIES
# =============================================================================

def softmax_composition(model_states: pd.DataFrame) -> CompositionOutput:
    """
    Simple softmax over model energies.
    
    All models included, weights proportional to recent performance.
    """
    # Use energy as the score
    energies = model_states['energy'].values
    n = len(energies)
    
    # Softmax
    exp_e = np.exp(energies - energies.max())
    weights = exp_e / exp_e.sum()
    
    return CompositionOutput(
        weights=weights,
        active_mask=np.ones(n),
        regime=None,
        method='ensemble'
    )


def sparse_composition(model_states: pd.DataFrame, 
                        min_energy: float = 0.1) -> CompositionOutput:
    """
    Zero out models with low energy.
    
    Only include models performing above threshold.
    """
    energies = model_states['energy'].values
    n = len(energies)
    
    # Mask: include only if energy > threshold
    mask = (energies > min_energy).astype(float)
    
    # Weights from softmax of energies (masked models get zero)
    exp_e = np.exp(energies - energies.max())
    weights = exp_e / (exp_e.sum() + 1e-8)
    
    return CompositionOutput(
        weights=weights,
        active_mask=mask,
        regime=None,
        method='sparse'
    )


def regime_conditional_composition(model_states: pd.DataFrame,
                                    regime: str) -> CompositionOutput:
    """
    Different weights per market regime.
    
    Regimes:
    - trending: momentum models favored
    - ranging: volatility_breakout, mean_reversion favored
    - volatile: reduce position, favor stable models
    - transition: equal weight
    """
    n = len(model_states)
    model_ids = model_states['model_id'].values if 'model_id' in model_states.columns else [f"model_{i}" for i in range(n)]
    
    # Define regime-specific weights
    REGIME_WEIGHTS = {
        'trending': {
            'volatility_breakout': 0.2,
            'momentum_trend': 0.6,
            'mean_reversion': 0.0,
            'cross_sectional': 0.1,
            'composite': 0.1,
        },
        'ranging': {
            'volatility_breakout': 0.5,
            'momentum_trend': 0.1,
            'mean_reversion': 0.3,
            'cross_sectional': 0.0,
            'composite': 0.1,
        },
        'volatile': {
            'volatility_breakout': 0.3,
            'momentum_trend': 0.2,
            'mean_reversion': 0.3,
            'cross_sectional': 0.1,
            'composite': 0.1,
        },
        'transition': {
            'volatility_breakout': 0.2,
            'momentum_trend': 0.2,
            'mean_reversion': 0.2,
            'cross_sectional': 0.2,
            'composite': 0.2,
        },
    }
    
    # Get weights for this regime
    regime_weights = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS['transition'])
    
    # Build weight array
    weights = np.array([
        regime_weights.get(mid, 0.2) for mid in model_ids
    ])
    
    # Normalize
    weights = weights / weights.sum()
    
    return CompositionOutput(
        weights=weights,
        active_mask=np.ones(n),
        regime=regime,
        method='regime_conditional'
    )


def learned_composition(model_states: pd.DataFrame,
                        market_features: np.ndarray,
                        hrm_weights: np.ndarray = None) -> CompositionOutput:
    """
    HRM learns the composition function.
    
    Input: model_states + market_features
    Output: weights
    
    This is where the neural HRM would be used.
    For now, simple linear combination.
    """
    n_models = len(model_states)
    
    # Flatten states for input
    state_features = np.array([
        model_states['energy'].values,
        model_states['confidence'].values,
        model_states['last_return'].values,
    ]).flatten()
    
    # Combine with market features
    full_input = np.concatenate([state_features, market_features[:n_models]])
    
    # Simple linear projection (this would be neural net in full HRM)
    if hrm_weights is None:
        hrm_weights = np.ones(len(full_input)) / len(full_input)
    
    scores = full_input * hrm_weights
    scores = scores[:n_models]
    
    # Softmax
    exp_s = np.exp(scores - scores.max())
    weights = exp_s / exp_s.sum()
    
    return CompositionOutput(
        weights=weights,
        active_mask=np.ones(n_models),
        regime=None,
        method='learned'
    )


# =============================================================================
# HRM COMPOSER
# =============================================================================

class HRMComposer:
    """
    HRM that composes models, not just selects one.
    
    Can use any composition strategy.
    """
    
    def __init__(self, 
                 mode: CompositionMode = CompositionMode.ENSEMBLE,
                 min_energy: float = 0.1):
        self.mode = mode
        self.min_energy = min_energy
        self.current_regime = 'transition'
        self.hrm_weights = None  # Learned weights
    
    def compose(self, 
                model_states: pd.DataFrame,
                market_features: np.ndarray = None,
                regime: str = None) -> CompositionOutput:
        """
        Compose model outputs based on current mode.
        """
        if self.mode == CompositionMode.ENSEMBLE:
            return softmax_composition(model_states)
        
        elif self.mode == CompositionMode.SPARSE:
            return sparse_composition(model_states, self.min_energy)
        
        elif self.mode == CompositionMode.REGIME_CONDITIONAL:
            return regime_conditional_composition(
                model_states, 
                regime or self.current_regime
            )
        
        elif self.mode == CompositionMode.LEARNED:
            return learned_composition(
                model_states,
                market_features or np.zeros(5),
                self.hrm_weights
            )
        
        # Default to ensemble
        return softmax_composition(model_states)
    
    def update_regime(self, regime: str):
        """Update detected regime"""
        self.current_regime = regime
    
    def set_learned_weights(self, weights: np.ndarray):
        """Set learned HRM weights"""
        self.hrm_weights = weights


# =============================================================================
# DETECT REGIME
# =============================================================================

def detect_regime(market_data: pd.DataFrame, 
                  lookback: int = 100) -> str:
    """
    Detect market regime from recent data.
    
    Returns: 'trending', 'ranging', 'volatile', or 'transition'
    """
    if len(market_data) < lookback:
        return 'transition'
    
    recent = market_data.tail(lookback)
    
    # Features for regime detection
    returns = recent['close'].pct_change().dropna()
    
    # Trending: autocorrelation > 0 (momentum exists)
    autocorr = returns.autocorr(lag=1) if len(returns) > 1 else 0
    
    # Volatility: std of returns
    volatility = returns.std()
    
    # Range: high-low spread relative to close
    range_ratio = (recent['high'] - recent['low']) / recent['close']
    avg_range = range_ratio.mean()
    
    # ADX-like: directional movement
    up_moves = recent['high'].diff().clip(lower=0)
    down_moves = -recent['low'].diff().clip(lower=0)
    directional = (up_moves.sum() - down_moves.sum()) / (up_moves.sum() + down_moves.sum() + 1e-8)
    
    # Classify regime
    if autocorr > 0.1 and abs(directional) > 0.3:
        return 'trending'
    elif autocorr < -0.1 and avg_range < 0.02:
        return 'ranging'
    elif volatility > 0.02 or avg_range > 0.05:
        return 'volatile'
    else:
        return 'transition'


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("HRM Composition Strategies")
    print("=" * 50)
    
    # Fake model states
    states = pd.DataFrame({
        'model_id': ['volatility_breakout', 'momentum_trend', 'mean_reversion', 'cross_sectional', 'composite'],
        'energy': [0.8, 0.3, 0.5, 0.2, 0.4],
        'confidence': [0.7, 0.4, 0.6, 0.3, 0.5],
        'last_return': [0.05, 0.02, -0.01, 0.01, 0.03],
        'status': ['active'] * 5,
    })
    
    print("\nModel States:")
    print(states)
    
    print("\n--- Ensemble (softmax all) ---")
    comp = softmax_composition(states)
    print(f"Weights: {comp.weights.round(3)}")
    print(f"Active: {comp.active_mask}")
    
    print("\n--- Sparse (threshold=0.3) ---")
    comp = sparse_composition(states, min_energy=0.3)
    print(f"Weights: {comp.weights.round(3)}")
    print(f"Active: {comp.active_mask}")
    
    print("\n--- Regime-Conditional (ranging) ---")
    comp = regime_conditional_composition(states, regime='ranging')
    print(f"Weights: {comp.weights.round(3)}")
    print(f"Regime: {comp.regime}")
    
    print("\n--- Full Composer ---")
    composer = HRMComposer(mode=CompositionMode.REGIME_CONDITIONAL)
    composer.update_regime('trending')
    comp = composer.compose(states)
    print(f"Weights for trending: {comp.weights.round(3)}")
    
    composer.update_regime('ranging')
    comp = composer.compose(states)
    print(f"Weights for ranging: {comp.weights.round(3)}")
    
    print("\n--- Apply to signals ---")
    # Fake signals from each model
    model_signals = {
        'volatility_breakout': np.array([0.5, 0.3, -0.2, 0.1]),
        'momentum_trend': np.array([0.2, 0.4, 0.1, -0.1]),
        'mean_reversion': np.array([-0.3, -0.1, 0.4, 0.2]),
        'cross_sectional': np.array([0.1, 0.2, 0.3, 0.0]),
        'composite': np.array([0.0, 0.1, 0.0, 0.1]),
    }
    
    comp = regime_conditional_composition(states, 'ranging')
    combined = comp.apply(model_signals)
    print(f"Combined signal: {combined.round(3)}")
