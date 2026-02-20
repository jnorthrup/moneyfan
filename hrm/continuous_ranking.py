
#!/usr/bin/env python3
"""
Continuous Stochastic Ranking (Nexus Simulation Mode)
=====================================================

Runs the Nexus in a continuous simulation loop to rank the 16 canonical algorithms.
Utilizes the BinanceAdapterPipeline for stochastic market data.

Usage:
    python3 hrm/continuous_ranking.py [--loop]
"""

import sys
import os
import argparse
import signal
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from hrm.binance_adapter import BinanceAdapterPipeline
from hrm.nexus import (
    Nexus, 
    PandasMemoryAdapter, 
    CanonicalSignalGenerator, 
    RankingBrain, 
    SimulationBackend
)
from hrm.kernels import (
    volatility_breakout_kernel,
    momentum_trend_kernel,
    mean_reversion_kernel,
    rolling_mean_kernel,
    rolling_std_kernel,
    rolling_max_kernel,
    rolling_min_kernel
)

# -----------------------------------------------------------------------------
# Legacy Signal Compute (Kept for Kernel Access by Nexus)
# -----------------------------------------------------------------------------

def compute_signals_16(df: pd.DataFrame) -> dict:
    """
    Compute the 16 canonical signals for a given dataframe.
    Returns dict: {signal_name: np.array}
    """
    close = df['close'].values.astype(np.float64)
    open_p = df['open'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    # volume = df['volume'].values.astype(np.float64)
    
    signals = {}
    
    # --- TREND (4) ---
    # 1. MACD Crossover (Proxy)
    ma12 = pd.Series(close).rolling(12).mean().values
    ma26 = pd.Series(close).rolling(26).mean().values
    macd = ma12 - ma26
    signal_line = pd.Series(macd).rolling(9).mean().values
    signals['macd_crossover'] = np.sign(macd - signal_line)

    # 2. SOTA Momentum
    signals['sota_momentum'] = momentum_trend_kernel(close, 20)
    
    # 3. Momentum Trend
    signals['momentum_trend'] = momentum_trend_kernel(close, 20) # Alias
    
    # 4. Mom Trend Additive
    signals['mom_trend_additive'] = signals['momentum_trend'] + np.sign(close - ma26)

    # --- MEAN REVERSION (4) ---
    # 5. RSI Mean Reversion
    signals['rsi_mean_reversion'] = mean_reversion_kernel(close, 14)
    
    # 6. Bollinger Reversion
    ma20 = rolling_mean_kernel(close, 20)
    std20 = rolling_std_kernel(close, 20)
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    boll = np.zeros_like(close)
    mask_high = close > upper
    mask_low = close < lower
    boll[mask_high] = -1.0
    boll[mask_low] = 1.0
    signals['bollinger_reversion'] = boll

    # 7. Grid Reversion
    ranges = rolling_max_kernel(high, 50) - rolling_min_kernel(low, 50)
    grid_pos = (close - rolling_min_kernel(low, 50)) / (ranges + 1e-8)
    signals['grid_reversion'] = 1.0 - 2.0 * grid_pos

    # 8. HRM Mean Reversion
    signals['hrm_mean_reversion'] = mean_reversion_kernel(close, 30)

    # --- VOLATILITY (3) ---
    # 9. Volatility Breakout
    signals['volatility_breakout'] = volatility_breakout_kernel(open_p, high, low, close, 20)
    
    # 10. Vol x Breakout Proven
    signals['vol_x_breakout_proven'] = signals['volatility_breakout'] 

    # 11. Momentum x Vol
    vol = (high - low) / (open_p + 1e-8)
    mom = np.zeros_like(close)
    mom[20:] = (close[20:] - close[:-20]) / (close[:-20] + 1e-8)
    signals['momentum_x_vol'] = mom * vol

    # --- STAT ARB (2) ---
    # 12. Bent Penny (Proxy)
    params = rolling_std_kernel(close, 100)
    z = (close - rolling_mean_kernel(close, 100)) / (params + 1e-8)
    signals['bent_penny'] = -np.sign(z) * (np.abs(z) > 2.0).astype(float)

    # 13. Pairs Spread (Proxy)
    signals['pairs_spread'] = mean_reversion_kernel(close, 10)

    # --- SYSTEMATIC (1) ---
    # 14. DCA Baseline
    signals['dca_baseline'] = np.ones_like(close)

    # --- ML / COMPOSITE (2) ---
    # 15. Technical ML (Proxy)
    signals['technical_ml'] = np.sign(np.sin(close)) 

    # 16. RSI x Trend
    signals['rsi_x_trend'] = signals['rsi_mean_reversion'] * signals['momentum_trend']
    
    return signals

def backtest_signal_vectorized(signal: np.ndarray, prices: np.ndarray) -> tuple:
    """
    Vectorized backtest of a single signal array.
    Returns (total_return, sharpe_ratio).
    """
    pos = np.roll(signal, 1)
    pos[0] = 0.0
    pos = np.nan_to_num(pos)
    ret = np.diff(prices) / (prices[:-1] + 1e-8)
    ret = np.insert(ret, 0, 0.0)
    strat_ret = pos * ret
    
    cum_log_ret = np.sum(np.log1p(strat_ret))
    final_equity_mult = np.exp(cum_log_ret)
    
    mean_ret = np.mean(strat_ret)
    std_ret = np.std(strat_ret)
    
    sharpe = 0.0
    if std_ret > 1e-9:
        sharpe = (mean_ret / std_ret) * np.sqrt(252 * 1440)
        
    return final_equity_mult, sharpe

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    args = parser.parse_args()
    
    print("Initializing Nexus (Tier 3 System)...")
    
    # 1. Wiring the Nexus
    pipeline = BinanceAdapterPipeline()
    memory = PandasMemoryAdapter(pipeline.arrow_store)
    senses = CanonicalSignalGenerator()
    brain = RankingBrain()
    executor = SimulationBackend()
    router = None # Not used in simple ranking
    
    nexus = Nexus(
        memory=memory,
        senses=senses,
        brain=brain,
        executor=executor,
        router=router
    )

    if args.loop:
        nexus.run_simulation_loop(pipeline)
    else:
        # One pass
        print("Running single pass...")
        bags = pipeline.sample_training_bag(n_samples=5)
        for bag in bags:
            nexus.pulse(bag)
        nexus.print_leaderboard()

if __name__ == "__main__":
    main()
