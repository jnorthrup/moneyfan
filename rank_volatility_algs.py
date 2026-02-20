
import pandas as pd
import numpy as np
import time
import os
import sys
sys.path.append(os.getcwd())
from datetime import datetime
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from hrm.binance_adapter import BinanceAdapterPipeline
from hrm.kernels import (
    volatility_breakout_kernel,
    momentum_trend_kernel,
    mean_reversion_kernel,
    rolling_mean_kernel,
    rolling_std_kernel,
    rolling_max_kernel,
    rolling_min_kernel
)

def compute_signals(df):
    """Compute 16 canonical signals from hrm.signal_hrm"""
    close = df['close'].values
    open_p = df['open'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    
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
    signals['momentum_trend'] = momentum_trend_kernel(close, 20) # Redundant in kernel but listed
    
    # 4. Mom Trend Additive (Composite proxy)
    signals['mom_trend_additive'] = signals['momentum_trend'] + np.sign(close - ma26)

    # --- MEAN REVERSION (4) ---
    # 5. RSI Mean Reversion
    signals['rsi_mean_reversion'] = mean_reversion_kernel(close, 14) # Approx
    
    # 6. Bollinger Reversion
    ma20 = rolling_mean_kernel(close, 20)
    std20 = rolling_std_kernel(close, 20)
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    boll = np.zeros_like(close)
    boll[close > upper] = -1.0
    boll[close < lower] = 1.0
    signals['bollinger_reversion'] = boll

    # 7. Grid Reversion (Proxy)
    # Simple grid based on recent high/low
    ranges = rolling_max_kernel(high, 50) - rolling_min_kernel(low, 50)
    grid_pos = (close - rolling_min_kernel(low, 50)) / (ranges + 1e-8)
    signals['grid_reversion'] = 1.0 - 2.0 * grid_pos

    # 8. HRM Mean Reversion
    signals['hrm_mean_reversion'] = mean_reversion_kernel(close, 30)

    # --- VOLATILITY (3) ---
    # 9. Volatility Breakout (Proven Winner)
    signals['volatility_breakout'] = volatility_breakout_kernel(open_p, high, low, close, 20)
    
    # 10. Vol x Breakout Proven (Alias/Variant)
    signals['vol_x_breakout_proven'] = signals['volatility_breakout'] 

    # 11. Momentum x Vol
    vol = (high - low) / (open_p + 1e-8)
    mom = np.zeros_like(close)
    mom[20:] = (close[20:] - close[:-20]) / (close[:-20] + 1e-8)
    signals['momentum_x_vol'] = mom * vol

    # --- STAT ARB (2) ---
    # 12. Bent Penny (Proxy: tail reversion)
    params = rolling_std_kernel(close, 100)
    z = (close - rolling_mean_kernel(close, 100)) / (params + 1e-8)
    signals['bent_penny'] = -np.sign(z) * (np.abs(z) > 2.0).astype(float)

    # 13. Pairs Spread (Single asset proxy: correlation with BTC)
    # We don't have BTC here easily, use internal mean reversion as placeholder
    signals['pairs_spread'] = mean_reversion_kernel(close, 10)

    # --- SYSTEMATIC (1) ---
    # 14. DCA Baseline
    signals['dca_baseline'] = np.ones_like(close)

    # --- ML / COMPOSITE (2) ---
    # 15. Technical ML (Proxy)
    signals['technical_ml'] = np.sign(np.random.randn(len(close))) # Dummy implementation 

    # 16. RSI x Trend
    signals['rsi_x_trend'] = signals['rsi_mean_reversion'] * signals['momentum_trend']
    
    return signals

def backtest_signal(signal, prices):
    """Simple vectorized backtest"""
    # Lag signal by 1 
    pos = np.roll(signal, 1)
    pos[0] = 0
    pos = np.nan_to_num(pos) # Handle NaNs
    
    # Daily returns
    ret = np.diff(prices) / (prices[:-1] + 1e-8)
    ret = np.insert(ret, 0, 0)
    
    # Strategy returns
    strat_ret = pos * ret
    
    # Metrics
    cum_ret = np.sum(strat_ret)
    mean_ret = np.mean(strat_ret)
    std_ret = np.std(strat_ret)
    sharpe = (mean_ret / (std_ret + 1e-8)) * np.sqrt(252*1440) 
    
    return cum_ret, sharpe

def rank_algs():
    print("Initializing BinanceAdapterPipeline...")
    pipeline = BinanceAdapterPipeline()
    
    assets = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'DOGE-USDT', 'AVAX-USDT'] 
    
    results = []
    
    print("\\nfetching and processing (16 Canonical Signals)...")
    
    for asset in assets:
        print(f"Processing {asset}...")
        engine = pipeline.get_backtrace_engine()
        
        # Try to load last 60 days
        end_date = datetime.utcnow()
        start_date = end_date - pd.Timedelta(days=60)
        
        try:
             # Force fetch if needed
             if hasattr(engine, 'fetcher') and engine.fetcher:
                 print(f"  Ensuring data is fetched for {asset}...")
                 engine.fetcher.fetch_window(asset, start_date, end_date)

             # Just load, it will trigger lazy fetch if configured
             df = engine.store.load(asset, start_date, end_date)
             
             if df.empty or len(df) < 1000:
                 print(f"  Not enough data for {asset}, skipping...")
                 continue
                 
             signals = compute_signals(df)
             
             for name, sig in signals.items():
                 # Filter NaNs
                 valid_idx = ~np.isnan(sig)
                 
                 if np.sum(valid_idx) < 100:
                     continue
                     
                 # Backtest
                 cum_ret, sharpe = backtest_signal(sig, df['close'].values)
                 
                 results.append({
                     'Asset': asset,
                     'Algorithm': name,
                     'TotalReturn': cum_ret,
                     'Sharpe': sharpe,
                 })
                 
        except Exception as e:
            print(f"Error processing {asset}: {e}")
            
    # Aggregate results
    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("No results generated.")
        return

    # Group by Algorithm
    grouped = df_res.groupby('Algorithm').agg({
        'TotalReturn': 'mean',
        'Sharpe': 'mean',
    }).sort_values('TotalReturn', ascending=False)
    
    print("\n" + "="*80)
    print("THE RANKING: 16 CANONICAL TRADE ALGS (Average across Volatile Assets)")
    print("="*80)
    print(grouped)
    print("="*80)

if __name__ == "__main__":
    rank_algs()
