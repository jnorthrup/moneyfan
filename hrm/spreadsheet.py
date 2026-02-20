"""
The Spreadsheet Approach

No complex swarm. Just simple weighted model combinations.
HRM learns the weights. Models are stateless pure functions.

proven: volatility × breakout = $37,308
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# MODEL DEFINITIONS (Pure Functions - No State)
# =============================================================================

def volatility_breakout(features: np.ndarray) -> np.ndarray:
    """
    PROVEN WINNER: volatility × breakout
    
    Signal = volatility_zscore × breakout_direction
    
    features: [n_assets, n_features]
    returns: [n_assets] signals
    """
    n_assets = features.shape[0]
    signals = np.zeros(n_assets)
    
    for i in range(n_assets):
        # Features: [open, high, low, close, volume, returns, volatility, momentum, rsi, ma_ratio, ...time]
        high = features[i, 1]
        low = features[i, 2]
        close = features[i, 3]
        volatility = features[i, 6]
        
        # Volatility signal (0-1)
        vol_signal = np.clip(volatility, 0, 1)
        
        # Breakout direction
        price_position = (close - low) / (high - low + 1e-8)
        breakout = 2 * price_position - 1
        
        signals[i] = vol_signal * breakout
    
    return signals


def momentum_trend(features: np.ndarray) -> np.ndarray:
    """momentum × trend"""
    n_assets = features.shape[0]
    signals = np.zeros(n_assets)
    
    for i in range(n_assets):
        momentum = features[i, 7]
        ma_ratio = features[i, 9]
        
        trend = np.sign(ma_ratio - 1.0)
        strength = np.clip(np.abs(momentum), 0, 1)
        
        signals[i] = strength * trend
    
    return signals


def mean_reversion(features: np.ndarray) -> np.ndarray:
    """mean reversion (buy dips, sell rips)"""
    n_assets = features.shape[0]
    signals = np.zeros(n_assets)
    
    for i in range(n_assets):
        ma_ratio = features[i, 9]
        rsi = features[i, 8]
        
        deviation = ma_ratio - 1.0
        rsi_signal = (50 - rsi) / 50
        
        if np.abs(deviation) > 0.02:
            signals[i] = -np.sign(deviation)
        
        signals[i] = 0.5 * signals[i] + 0.5 * rsi_signal
    
    return signals


def sector_mapreduce(features: np.ndarray, sector_indices: Dict[str, List[int]]) -> np.ndarray:
    """
    Aggregate signals by sector, then reduce.
    
    This IS the mapreduce - nothing fancy.
    """
    n_assets = features.shape[0]
    
    # Base signals from momentum
    base = momentum_trend(features)
    
    # Sector-level aggregation
    sector_signals = {}
    for sector_name, indices in sector_indices.items():
        sector_values = base[indices]
        sector_signals[sector_name] = np.mean(sector_values)
    
    # Reduce: assign sector signal to each asset
    signals = np.zeros(n_assets)
    for i in range(n_assets):
        for sector_name, indices in sector_indices.items():
            if i in indices:
                signals[i] = sector_signals[sector_name]
                break
    
    return signals


def composite_winner(features: np.ndarray) -> np.ndarray:
    """
    The proven composite: volatility × breakout
    Already computed, just returns it.
    """
    return volatility_breakout(features)


# =============================================================================
# THE SPREADSHEET
# =============================================================================

@dataclass
class SpreadsheetRow:
    """One row in the spreadsheet"""
    model_name: str
    model_func: callable
    weight: float = 0.2


class Spreadsheet:
    """
    Simple weighted combination of models.
    
    No stateful agents.
    No full duplex.
    Just a spreadsheet.
    """
    
    def __init__(self, sector_indices: Dict[str, List[int]] = None):
        self.models = [
            SpreadsheetRow("volatility_breakout", volatility_breakout, 0.4),  # Heavily weight proven winner
            SpreadsheetRow("momentum_trend", momentum_trend, 0.2),
            SpreadsheetRow("mean_reversion", mean_reversion, 0.2),
            SpreadsheetRow("sector_mapreduce", 
                          lambda f: sector_mapreduce(f, sector_indices or {}), 0.1),
            SpreadsheetRow("composite_winner", composite_winner, 0.1),
        ]
        self.sector_indices = sector_indices or {}
    
    def compute_signals(self, 
                        features: np.ndarray,
                        weights: np.ndarray = None) -> np.ndarray:
        """
        Compute weighted signals.
        
        features: [n_assets, n_features]
        weights: [n_models] or None for defaults
        
        returns: [n_assets] combined signals
        """
        if weights is not None:
            for i, w in enumerate(weights):
                if i < len(self.models):
                    self.models[i].weight = w
        
        # Weighted sum
        signals = np.zeros(features.shape[0])
        for model in self.models:
            model_signals = model.model_func(features)
            signals += model.weight * model_signals
        
        return signals
    
    def get_weights(self) -> np.ndarray:
        """Current model weights"""
        return np.array([m.weight for m in self.models])
    
    def set_weights(self, weights: np.ndarray):
        """Set weights from HRM output"""
        weights = weights / weights.sum()  # Normalize
        for i, w in enumerate(weights):
            if i < len(self.models):
                self.models[i].weight = w


# =============================================================================
# HRM INTERFACE
# =============================================================================

def hrm_input_from_features(features: np.ndarray) -> np.ndarray:
    """
    Convert features to HRM input.
    
    HRM sees: regime features (cross-asset aggregates)
    - mean_volatility
    - mean_momentum
    - mean_returns
    - cross_sectional_std
    - time factors
    """
    # Cross-asset aggregates
    mean_vol = np.mean(features[:, 6])      # volatility
    mean_mom = np.mean(features[:, 7])      # momentum
    mean_ret = np.mean(features[:, 5])      # returns
    std_close = np.std(features[:, 3])      # cross-sectional std
    
    # Time factors (same for all assets)
    time_factors = features[0, 10:15]
    
    # Combine
    return np.concatenate([
        [mean_vol, mean_mom, mean_ret, std_close],
        time_factors,
        # Previous model weights (for continuity)
        np.zeros(5),  # Will be filled by caller
    ])


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("The Spreadsheet Approach")
    print("=" * 50)
    
    # Fake features [128 assets, 15 features]
    np.random.seed(42)
    features = np.random.randn(128, 15) * 0.1 + 0.5
    
    # Simple sector indices
    sector_indices = {
        "majors": list(range(1, 11)),
        "defi": list(range(20, 30)),
        "layer1": list(range(1, 20)),
    }
    
    # Create spreadsheet
    spreadsheet = Spreadsheet(sector_indices)
    
    # Compute signals with default weights
    signals = spreadsheet.compute_signals(features)
    
    print(f"Features shape: {features.shape}")
    print(f"Signals shape: {signals.shape}")
    print(f"Default weights: {spreadsheet.get_weights()}")
    print(f"Signal range: [{signals.min():.3f}, {signals.max():.3f}]")
    
    # Set custom weights (e.g., from HRM)
    custom_weights = np.array([0.6, 0.15, 0.1, 0.1, 0.05])
    spreadsheet.set_weights(custom_weights)
    signals2 = spreadsheet.compute_signals(features)
    
    print(f"\nWith custom weights {custom_weights}:")
    print(f"Signal range: [{signals2.min():.3f}, {signals2.max():.3f}]")
    
    # Show per-model contributions
    print("\nPer-model contributions:")
    for model in spreadsheet.models:
        model_signals = model.model_func(features)
        print(f"  {model.name}: weight={model.weight:.2f}, "
              f"mean_signal={np.mean(model_signals):.4f}")
