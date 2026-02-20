"""
Computational Kernels

High-performance computation kernels for:
- Rolling calculations
- Signal generation
- Cross-sectional operations

Used by instruments and models.
"""

import numpy as np
import pandas as pd
try:
    from numba import jit, prange
except ImportError:
    # Dummy jit decorator if numba is missing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    # Dummy prange
    prange = range
from typing import Tuple, Optional
from functools import wraps


# =============================================================================
# ROLLING KERNELS (Numba-accelerated)
# =============================================================================

@jit(nopython=True, cache=True)
def rolling_mean_kernel(values: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling mean"""
    n = len(values)
    result = np.full(n, np.nan)
    
    if n < window:
        return result
    
    # Initial sum
    s = 0.0
    for i in range(window):
        s += values[i]
    result[window - 1] = s / window
    
    # Rolling
    for i in range(window, n):
        s += values[i] - values[i - window]
        result[i] = s / window
    
    return result


@jit(nopython=True, cache=True)
def rolling_std_kernel(values: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling standard deviation"""
    n = len(values)
    result = np.full(n, np.nan)
    
    if n < window:
        return result
    
    # Use Welford's algorithm for numerical stability
    for i in range(window - 1, n):
        mean = 0.0
        m2 = 0.0
        for j in range(i - window + 1, i + 1):
            delta = values[j] - mean
            mean += delta / (j - i + window)
            delta2 = values[j] - mean
            m2 += delta * delta2
        
        result[i] = np.sqrt(m2 / window)
    
    return result


@jit(nopython=True, cache=True)
def rolling_max_kernel(values: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling max"""
    n = len(values)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        result[i] = np.max(values[i - window + 1:i + 1])
    
    return result


@jit(nopython=True, cache=True)
def rolling_min_kernel(values: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling min"""
    n = len(values)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        result[i] = np.min(values[i - window + 1:i + 1])
    
    return result


@jit(nopython=True, cache=True)
def rolling_zscore_kernel(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling z-score (value - mean) / std"""
    mean = rolling_mean_kernel(values, window)
    std = rolling_std_kernel(values, window)
    return (values - mean) / (std + 1e-8)


@jit(nopython=True, cache=True)
def rolling_quantile_kernel(values: np.ndarray, window: int, q: float) -> np.ndarray:
    """Rolling quantile"""
    n = len(values)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        window_vals = np.sort(values[i - window + 1:i + 1])
        idx = int(q * (window - 1))
        result[i] = window_vals[idx]
    
    return result


# =============================================================================
# SIGNAL KERNELS
# =============================================================================

@jit(nopython=True, cache=True)
def volatility_breakout_kernel(
    open_p: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int = 20
) -> np.ndarray:
    """
    Volatility × Breakout signal kernel
    
    Returns signals array
    """
    n = len(close)
    signals = np.zeros(n)
    
    for i in range(window, n):
        # Volatility: normalized high-low range
        vol = (high[i] - low[i]) / (open_p[i] + 1e-8)
        vol_signal = min(max(vol, 0.0), 1.0)
        
        # Breakout: price position in range
        price_pos = (close[i] - low[i]) / (high[i] - low[i] + 1e-8)
        breakout = 2.0 * price_pos - 1.0
        
        signals[i] = vol_signal * breakout
    
    return signals


@jit(nopython=True, cache=True)
def momentum_trend_kernel(
    close: np.ndarray,
    window: int = 20
) -> np.ndarray:
    """Momentum × Trend signal kernel"""
    n = len(close)
    signals = np.zeros(n)
    
    # Rolling mean
    ma = rolling_mean_kernel(close, window)
    
    for i in range(window, n):
        # Trend: price vs MA
        trend = np.sign(close[i] / ma[i] - 1.0)
        
        # Momentum: recent returns
        ret = (close[i] - close[i - window]) / (close[i - window] + 1e-8)
        strength = min(max(abs(ret), 0.0), 1.0)
        
        signals[i] = strength * trend
    
    return signals


@jit(nopython=True, cache=True)
def mean_reversion_kernel(
    close: np.ndarray,
    window: int = 20
) -> np.ndarray:
    """Mean reversion signal kernel"""
    n = len(close)
    signals = np.zeros(n)
    
    ma = rolling_mean_kernel(close, window)
    std = rolling_std_kernel(close, window)
    
    for i in range(window, n):
        # Z-score from mean
        zscore = (close[i] - ma[i]) / (std[i] + 1e-8)
        
        # Reversion signal (opposite of deviation)
        signals[i] = -np.sign(zscore) * min(abs(zscore) / 2.0, 1.0)
    
    return signals


# =============================================================================
# CROSS-SECTIONAL KERNELS
# =============================================================================

@jit(nopython=True, parallel=True, cache=True)
def cross_sectional_rank(values: np.ndarray) -> np.ndarray:
    """
    Cross-sectional rank (0 to 1)
    
    Input: [n_assets]
    Output: [n_assets] ranks
    """
    n = len(values)
    ranks = np.zeros(n)
    
    # Simple ranking
    sorted_idx = np.argsort(values)
    for rank, idx in enumerate(sorted_idx):
        ranks[idx] = rank / (n - 1)
    
    return ranks


@jit(nopython=True, parallel=True, cache=True)
def cross_sectional_zscore(values: np.ndarray) -> np.ndarray:
    """
    Cross-sectional z-score
    
    Input: [n_assets]
    Output: [n_assets] z-scores
    """
    mean = np.mean(values)
    std = np.std(values)
    return (values - mean) / (std + 1e-8)


@jit(nopython=True, parallel=True, cache=True)
def sector_neutralize(signals: np.ndarray, sector_ids: np.ndarray) -> np.ndarray:
    """
    Neutralize signals within sectors
    
    Input:
        signals: [n_assets] signal values
        sector_ids: [n_assets] sector identifiers (0, 1, 2, ...)
    Output:
        [n_assets] sector-neutral signals
    """
    n = len(signals)
    result = np.zeros(n)
    
    n_sectors = int(np.max(sector_ids)) + 1
    
    for s in range(n_sectors):
        mask = sector_ids == s
        sector_signals = signals[mask]
        if len(sector_signals) > 0:
            sector_mean = np.mean(sector_signals)
            result[mask] = sector_signals - sector_mean
    
    return result


# =============================================================================
# PANDAS WRAPPERS
# =============================================================================

def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Pandas wrapper for rolling mean kernel"""
    values = series.values.astype(np.float64)
    result = rolling_mean_kernel(values, window)
    return pd.Series(result, index=series.index)


def rolling_std(series: pd.Series, window: int) -> pd.Series:
    """Pandas wrapper for rolling std kernel"""
    values = series.values.astype(np.float64)
    result = rolling_std_kernel(values, window)
    return pd.Series(result, index=series.index)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Pandas wrapper for rolling z-score kernel"""
    values = series.values.astype(np.float64)
    result = rolling_zscore_kernel(values, window)
    return pd.Series(result, index=series.index)


# =============================================================================
# ROLLING THINGS - CONVENIENCE CLASSES
# =============================================================================

class RollingThing:
    """
    Base class for rolling calculations.
    
    Example:
        rm = RollingMean(window=20)
        result = rm(series)
    """
    def __init__(self, window: int):
        self.window = window
    
    def __call__(self, series: pd.Series) -> pd.Series:
        raise NotImplementedError


class RollingMean(RollingThing):
    def __call__(self, series: pd.Series) -> pd.Series:
        return rolling_mean(series, self.window)


class RollingStd(RollingThing):
    def __call__(self, series: pd.Series) -> pd.Series:
        return rolling_std(series, self.window)


class RollingZScore(RollingThing):
    def __call__(self, series: pd.Series) -> pd.Series:
        return rolling_zscore(series, self.window)


class RollingMax(RollingThing):
    def __call__(self, series: pd.Series) -> pd.Series:
        values = series.values.astype(np.float64)
        return pd.Series(rolling_max_kernel(values, self.window), index=series.index)


class RollingMin(RollingThing):
    def __call__(self, series: pd.Series) -> pd.Series:
        values = series.values.astype(np.float64)
        return pd.Series(rolling_min_kernel(values, self.window), index=series.index)


class RollingQuantile(RollingThing):
    def __init__(self, window: int, q: float):
        super().__init__(window)
        self.q = q
    
    def __call__(self, series: pd.Series) -> pd.Series:
        values = series.values.astype(np.float64)
        return pd.Series(rolling_quantile_kernel(values, self.window, self.q), index=series.index)


# =============================================================================
# KERNEL REGISTRY
# =============================================================================

KERNELS = {
    'rolling_mean': rolling_mean_kernel,
    'rolling_std': rolling_std_kernel,
    'rolling_max': rolling_max_kernel,
    'rolling_min': rolling_min_kernel,
    'rolling_zscore': rolling_zscore_kernel,
    'rolling_quantile': rolling_quantile_kernel,
    'volatility_breakout': volatility_breakout_kernel,
    'momentum_trend': momentum_trend_kernel,
    'mean_reversion': mean_reversion_kernel,
    'cross_sectional_rank': cross_sectional_rank,
    'cross_sectional_zscore': cross_sectional_zscore,
    'sector_neutralize': sector_neutralize,
}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Computational Kernels")
    print("=" * 50)
    
    # Test data
    np.random.seed(42)
    n = 1000
    close = 100 + np.random.randn(n).cumsum()
    open_p = close + np.random.randn(n) * 0.5
    high = np.maximum(open_p, close) + np.abs(np.random.randn(n)) * 0.3
    low = np.minimum(open_p, close) - np.abs(np.random.randn(n)) * 0.3
    
    # Test rolling kernels
    import time
    
    print("\nTesting rolling_mean:")
    start = time.time()
    ma = rolling_mean_kernel(close, 20)
    print(f"  Time: {(time.time() - start)*1000:.2f}ms")
    print(f"  Result shape: {ma.shape}")
    print(f"  Last 5 values: {ma[-5:]}")
    
    print("\nTesting volatility_breakout:")
    start = time.time()
    signals = volatility_breakout_kernel(open_p, high, low, close, 20)
    print(f"  Time: {(time.time() - start)*1000:.2f}ms")
    print(f"  Signal range: [{signals.min():.3f}, {signals.max():.3f}]")
    
    print("\nTesting cross_sectional_rank:")
    values = np.random.randn(128)  # 128 assets
    ranks = cross_sectional_rank(values)
    print(f"  Ranks: {ranks[:5]}")
    
    print("\nRollingThings:")
    rm = RollingMean(20)
    rz = RollingZScore(20)
    series = pd.Series(close)
    print(f"  RollingMean: {rm(series).iloc[-5:].values}")
    print(f"  RollingZScore: {rz(series).iloc[-5:].values}")
