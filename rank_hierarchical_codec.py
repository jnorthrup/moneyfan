#!/usr/bin/env python3
"""
rank_hierarchical_codec.py

Rank the hierarchical codec against baseline strategies using Binance Arrow data.

Usage:
    python rank_hierarchical_codec.py --arrow hrm/data/arrow/
"""

import argparse
import time
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from hrm.hierarchical_codec import HierarchicalCodec, HierarchicalCodecConfig
from hrm.duck_store import DuckStore
from hrm.pipeline import compute_features, compute_all_signals

N_SIGNALS = 24
STARTING_CAPITAL = 100.0


def sharpe(pnl: pd.Series) -> float:
    return pnl.mean() / pnl.std() if pnl.std() else 0.0


def sortino(pnl: pd.Series) -> float:
    downside = pnl[pnl < 0]
    downside_std = downside.std() if len(downside) > 0 else 0
    return pnl.mean() / downside_std if downside_std else 0.0


def max_drawdown(pnl: pd.Series) -> float:
    cummax = pnl.cummax()
    drawdown = pnl - cummax
    return drawdown.min()


def profit_factor(pnl: pd.Series) -> float:
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = abs(pnl[pnl < 0].sum())
    return gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0


def kd_ratio(dirs: np.ndarray, moves: np.ndarray, conf: np.ndarray, thresh: float = 0.3) -> float:
    mask = conf > thresh
    if mask.sum() == 0:
        return 0.0
    kills = (dirs == np.sign(moves))[mask].sum()
    deaths = (dirs != np.sign(moves))[mask].sum()
    return kills / deaths if deaths else float(kills)


def simulate_trades(predictions: np.ndarray, confidence: np.ndarray, returns: np.ndarray,
                    starting_capital: float = STARTING_CAPITAL) -> dict:
    """Simulate trading with $100 starting capital."""
    pos = predictions * confidence
    pos = np.clip(pos, -1.0, 1.0)
    
    step_pnl_pct = pos * returns
    
    pnl_series = pd.Series(step_pnl_pct).cumsum()
    pnl_dollars = pnl_series * starting_capital
    
    return {
        "total_pnl_pct": pnl_series.iloc[-1] if len(pnl_series) > 0 else 0,
        "total_pnl_dollars": pnl_dollars.iloc[-1] if len(pnl_dollars) > 0 else 0,
        "sharpe": sharpe(pd.Series(step_pnl_pct)),
        "sortino": sortino(pd.Series(step_pnl_pct)),
        "max_drawdown": max_drawdown(pnl_series),
        "profit_factor": profit_factor(pd.Series(step_pnl_pct)),
        "kd": kd_ratio(np.sign(predictions), returns, confidence),
    }


def compute_signal_tensor(df: pd.DataFrame, signal_df: pd.DataFrame) -> torch.Tensor:
    """Convert signal DataFrame to tensor."""
    signal_names = [
        "macd_crossover", "sota_momentum", "momentum_trend", "sector_rotation",
        "rsi_mean_reversion", "bollinger_reversion", "grid_reversion", "hrm_mean_reversion",
        "harvest_rebalance", "kilo_rebalance",
        "volatility_breakout", "bollinger_vol_regime", "vol_inverse_sizing",
        "bent_penny", "pairs_spread",
        "dca_baseline", "weekly_cadence",
        "technical_ml", "grid_x_trend", "rsi_x_trend",
        "momentum_x_vol", "vol_x_breakout_proven", "mom_trend_additive", "rsi_trend_additive",
    ]
    
    if signal_df is None or len(signal_df) == 0:
        return torch.zeros(len(df), N_SIGNALS * 2, dtype=torch.float32)
    
    T = len(df)
    tensor = np.zeros((T, N_SIGNALS * 2), dtype=np.float32)
    
    ts_to_idx = {ts: i for i, ts in enumerate(df.index)}
    
    for i, sig_name in enumerate(signal_names):
        mask = signal_df['model'] == sig_name
        if mask.any():
            sig_rows = signal_df[mask]
            for _, row in sig_rows.iterrows():
                t_idx = ts_to_idx.get(row['timestamp'])
                if t_idx is not None:
                    tensor[t_idx, i] = row['signal']
                    tensor[t_idx, N_SIGNALS + i] = row['confidence']
    
    return torch.from_numpy(tensor)


def load_stochastic_bag(arrow_dir: str, n_pairs: int = 30, min_rows: int = 1000):
    """Load stochastic bag of pairs."""
    store = DuckStore(arrow_dir=arrow_dir)
    
    all_files = list(Path(arrow_dir).glob("*.feather"))
    if len(all_files) == 0:
        return []
    
    sample_files = random.sample(all_files, min(n_pairs, len(all_files)))
    
    bag = []
    for feather_file in sample_files:
        symbol = feather_file.stem.replace("_", "-")
        try:
            df = store.load(symbol)
            if len(df) >= min_rows:
                df['symbol'] = symbol
                bag.append((symbol, df))
        except Exception:
            continue
    
    return bag


def backtest_codec(codec: HierarchicalCodec, bag: list, seq_len: int = 32) -> dict:
    """Backtest hierarchical codec on stochastic bag."""
    all_predictions = []
    all_confidence = []
    all_returns = []
    
    codec.eval()
    
    with torch.no_grad():
        for symbol, df in bag:
            if len(df) < seq_len + 150:
                continue
            
            window = df.iloc[100:].copy()
            window['symbol'] = symbol
            
            try:
                features_df = compute_features(window)
                signal_df = compute_all_signals(window, features_df)
            except Exception:
                continue
            
            if signal_df is None or len(signal_df) == 0:
                continue
            
            signals = compute_signal_tensor(window, signal_df)
            
            if signals.shape[0] < seq_len + 1:
                continue
            
            returns = window['close'].pct_change().fillna(0).values
            
            for i in range(0, len(signals) - seq_len - 1, seq_len):
                batch = signals[i:i + seq_len + 1].unsqueeze(0)
                
                output, _ = codec(batch, mode="trade")
                pred_return = output[0, 0].item()
                conf = output[0, 1].item()
                
                actual_return = returns[i + seq_len] if i + seq_len < len(returns) else 0
                
                all_predictions.append(pred_return)
                all_confidence.append(conf)
                all_returns.append(actual_return)
    
    if len(all_predictions) == 0:
        return {"total_pnl_pct": 0, "sharpe": 0, "kd": 0}
    
    predictions = np.array(all_predictions)
    confidence = np.array(all_confidence)
    returns = np.array(all_returns)
    
    return simulate_trades(predictions, confidence, returns)


def backtest_signal(signal_name: str, bag: list) -> dict:
    """Backtest a single signal as baseline."""
    all_predictions = []
    all_confidence = []
    all_returns = []
    
    for symbol, df in bag:
        if len(df) < 150:
            continue
        
        window = df.iloc[100:].copy()
        window['symbol'] = symbol
        
        try:
            features_df = compute_features(window)
            signal_df = compute_all_signals(window, features_df)
        except Exception:
            continue
        
        if signal_df is None or len(signal_df) == 0:
            continue
        
        mask = signal_df['model'] == signal_name
        if not mask.any():
            continue
        
        returns = window['close'].pct_change().fillna(0).values
        
        ts_to_idx = {ts: i for i, ts in enumerate(window.index)}
        
        for _, row in signal_df[mask].iterrows():
            t_idx = ts_to_idx.get(row['timestamp'])
            if t_idx is not None and t_idx < len(returns):
                all_predictions.append(row['signal'])
                all_confidence.append(row['confidence'])
                all_returns.append(returns[t_idx])
    
    if len(all_predictions) == 0:
        return {"total_pnl_pct": 0, "sharpe": 0, "kd": 0}
    
    predictions = np.array(all_predictions)
    confidence = np.array(all_confidence)
    returns = np.array(all_returns)
    
    return simulate_trades(predictions, confidence, returns)


def print_rankings(rankings: list, sort_by: str = "sharpe"):
    """Print rankings as markdown table."""
    rankings.sort(key=lambda x: x.get(sort_by, 0), reverse=True)
    
    print(f"\n## Rankings (sorted by {sort_by})")
    print(f"\n| Rank | Agent | PnL % | Sharpe | Sortino | MaxDD | PF | KD |")
    print("|------|-------|-------|--------|---------|-------|-----|-----|")
    
    for i, r in enumerate(rankings[:25]):
        print(f"| {i+1:2d} | {r['agent']:<24} | {r['total_pnl_pct']*100:>+6.2f}% | "
              f"{r['sharpe']:>+6.2f} | {r['sortino']:>+7.2f} | {r['max_drawdown']*100:>6.1f}% | "
              f"{r['profit_factor']:>4.2f} | {r['kd']:>4.1f} |")


def main():
    parser = argparse.ArgumentParser(description="Rank hierarchical codec vs baseline signals")
    parser.add_argument("--arrow", type=str, default="hrm/data/arrow", help="Arrow directory")
    parser.add_argument("--checkpoint", type=str, default="hrm/checkpoints", help="Checkpoint directory")
    parser.add_argument("--n-pairs", type=int, default=10, help="Number of pairs for stochastic bag")
    parser.add_argument("--sort-by", type=str, default="sharpe", help="Sort by metric")
    args = parser.parse_args()
    
    print("=" * 70)
    print("  HIERARCHICAL CODEC RANKING")
    print("=" * 70)
    
    print("\nLoading stochastic bag...")
    bag = load_stochastic_bag(args.arrow, n_pairs=args.n_pairs)
    print(f"Loaded {len(bag)} pairs")
    
    if len(bag) == 0:
        print("ERROR: No data loaded!")
        return
    
    checkpoint_dir = Path(args.checkpoint)
    codec_files = sorted(checkpoint_dir.glob("hierarchical_codec_*.pt"), reverse=True)
    
    codec = None
    if codec_files:
        print(f"\nLoading codec from {codec_files[0]}")
        checkpoint = torch.load(codec_files[0], map_location='cpu', weights_only=False)
        config = checkpoint['config']
        codec = HierarchicalCodec(config)
        codec.load_state_dict(checkpoint['model_state'])
        codec.eval()
    else:
        print("\nNo checkpoint found, using random initialized codec")
        config = HierarchicalCodecConfig(n_signals=N_SIGNALS)
        codec = HierarchicalCodec(config)
    
    print("\nRunning backtests...")
    rankings = []
    
    print("  Backtesting hierarchical codec...")
    codec_result = backtest_codec(codec, bag)
    rankings.append({"agent": "hierarchical_codec", **codec_result})
    print(f"    PnL: {codec_result['total_pnl_pct']*100:+.2f}%, Sharpe: {codec_result['sharpe']:+.2f}")
    
    signal_names = [
        "macd_crossover", "sota_momentum", "momentum_trend", "sector_rotation",
        "rsi_mean_reversion", "bollinger_reversion", "grid_reversion", "hrm_mean_reversion",
        "harvest_rebalance", "kilo_rebalance",
        "volatility_breakout", "bollinger_vol_regime", "vol_inverse_sizing",
        "bent_penny", "pairs_spread",
        "dca_baseline", "weekly_cadence",
        "technical_ml",
    ]
    
    for sig_name in signal_names:
        result = backtest_signal(sig_name, bag)
        rankings.append({"agent": sig_name, **result})
    
    print_rankings(rankings, sort_by=args.sort_by)
    
    print("\n" + "=" * 70)
    print("  RANKING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
