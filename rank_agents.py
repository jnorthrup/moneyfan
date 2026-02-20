#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rank_agents.py

Loads Binance Arrow data, computes instrument-metric features,
runs the 24 codecs and HRM meta-allocator, and prints a markdown table
ranking every agent (including HRM) by PnL, Alpha, Sharpe, and KD-ratio.

Usage:
    python rank_agents.py --arrow hrm/data/arrow/BTC_USDT.feather
"""

import argparse, time
from pathlib import Path

import numpy as np, pandas as pd, pyarrow.parquet as pq
import torch

# New codec architecture
from hrm.codecs import CodecCollection, HRM, CodecConfig, HRMConfig, Codec

# Starting capital for backtesting
STARTING_CAPITAL = 100.0  # $100

# ----------------------------------------------------------------------
def load_arrow(path: Path) -> pd.DataFrame:
    import os
    
    path_str = str(path)
    if path_str.endswith('.feather'):
        df = pd.read_feather(path)
    elif os.path.isdir(path_str):
        # Load multiple assets for stochastic bag
        import glob
        import random
        files = glob.glob(os.path.join(path_str, "*.feather"))
        if not files:
            raise FileNotFoundError(f"No .feather files found in {path}")
        
        # Sample from multiple assets
        sample_files = random.sample(files, min(10, len(files)))
        df_list = []
        
        for f in sample_files:
            try:
                df = pd.read_feather(f)
                if len(df) > 1000:
                    # Add symbol column
                    symbol = os.path.basename(f).replace('.feather', '').replace('_', '-')
                    df['symbol'] = symbol
                    df_list.append(df)
            except:
                continue
        
        if not df_list:
            raise ValueError("Could not load any valid Arrow files")
        
        df = pd.concat(df_list, ignore_index=True)
    else:
        raise ValueError(f"Path must be .feather file or directory: {path}")
    
    for col in ["open","high","low","close","volume","volatility","momentum"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Add required features for signal computation
    df = add_required_features(df)
    return df

# ----------------------------------------------------------------------
# Metric helpers (ML trading precedents)
# ----------------------------------------------------------------------
def returns_series(df: pd.DataFrame) -> pd.Series:
    """Percent return of the close price."""
    return df["close"].pct_change().fillna(0.0)

def sharpe(pnl: pd.Series) -> float:
    """Sharpe ratio: returns_mean / returns_std"""
    return pnl.mean() / pnl.std() if pnl.std() else 0.0

def sortino(pnl: pd.Series) -> float:
    """Sortino ratio: returns_mean / downside_std (ML precedent)"""
    downside = pnl[pnl < 0]
    downside_std = downside.std() if len(downside) > 0 else 0
    return pnl.mean() / downside_std if downside_std else 0.0

def max_drawdown(pnl: pd.Series) -> float:
    """Maximum drawdown: peak to trough (ML precedent)"""
    cummax = pnl.cummax()
    drawdown = pnl - cummax
    return drawdown.min()

def profit_factor(pnl: pd.Series) -> float:
    """Profit factor: gross_profit / gross_loss (ML precedent)"""
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = abs(pnl[pnl < 0].sum())
    return gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

def recovery_factor(pnl: pd.Series) -> float:
    """Recovery factor: net_profit / max_drawdown (ML precedent)"""
    net_profit = pnl.iloc[-1] if len(pnl) > 0 else 0
    dd = abs(max_drawdown(pnl))
    return net_profit / dd if dd > 0 else 0.0

def kd_ratio(dirs: np.ndarray, moves: np.ndarray, conf: np.ndarray, thresh: float = 0.5) -> float:
    """Kill/Death ratio: correct predictions / incorrect predictions"""
    mask = conf > thresh
    kills = (dirs == np.sign(moves))[mask].sum()
    deaths = (dirs != np.sign(moves))[mask].sum()
    return kills / deaths if deaths else float(kills)

def deviation(current: float, baseline: float) -> float:
    """Deviation from baseline (Kotlin precedent)"""
    return (current - baseline) / baseline if baseline != 0 else 0.0

def alpha(pnl: pd.Series, bh: pd.Series) -> float:
    """Alpha: excess return vs buy-and-hold"""
    return pnl.cumsum().iloc[-1] - bh.cumsum().iloc[-1]

# ----------------------------------------------------------------------
# Simple trade simulation (same for models and HRM)
# $100 starting capital
# ----------------------------------------------------------------------
def simulate(trade_dir: np.ndarray, confidence: np.ndarray, returns: np.ndarray,
            conf_thresh: float = 0.5, max_pos: float = 1.0, starting_capital: float = STARTING_CAPITAL) -> dict:
    """
    Simulate trading with $100 starting capital.
    Returns all standard ML trading metrics.
    """
    pos = np.where(confidence > conf_thresh, trade_dir * confidence, 0.0)
    pos = np.clip(pos, -max_pos, max_pos)
    
    # PnL in percentage terms
    step_pnl_pct = pos * returns
    
    # PnL in dollar terms (starting with $100)
    step_pnl_dollars = step_pnl_pct * starting_capital
    
    pnl_series_pct = pd.Series(step_pnl_pct).cumsum()
    pnl_series_dollars = pd.Series(step_pnl_dollars).cumsum()
    
    return {
        "pnl_series": pnl_series_pct,
        "pnl_series_dollars": pnl_series_dollars,
        "total_pnl": pnl_series_pct.iloc[-1],  # Percentage
        "total_pnl_dollars": pnl_series_dollars.iloc[-1],  # Dollars
        "sharpe": sharpe(pnl_series_pct),
        "sortino": sortino(pnl_series_pct),
        "max_drawdown": max_drawdown(pnl_series_pct),
        "profit_factor": profit_factor(pd.Series(step_pnl_pct)),
        "recovery_factor": recovery_factor(pnl_series_pct),
        "kd": kd_ratio(trade_dir, returns, confidence, conf_thresh),
    }

# --------------------------------------------------------------
def add_required_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features required by the model signals"""
    # This is a placeholder - in real implementation, compute all needed features
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute instrument-metric features for codecs"""
    from hrm.features import compute_all_features
    
    # Ensure we have required columns
    if 'asset' not in df.columns:
        df['asset'] = 'BTC-USD'  # Default asset
    
    # Compute all features
    features = compute_all_features(df)
    
    return features


def load_models(checkpoint_dir: str):
    """Load trained codecs and HRM"""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None, None
    
    # Find latest HRM checkpoint
    hrm_files = sorted(checkpoint_path.glob("hrm_*.pt"), reverse=True)
    if not hrm_files:
        return None, None
    
    # Load HRM
    hrm_file = hrm_files[0]
    hrm_checkpoint = torch.load(hrm_file, map_location='cpu')
    hrm_config = hrm_checkpoint['config']
    hrm = HRM(hrm_config)
    hrm.load_state_dict(hrm_checkpoint['model_state_dict'])
    hrm.eval()
    
    # Load all codecs
    codec_files = list(checkpoint_path.glob("codec_*.pt"))
    codecs = {}
    for codec_file in codec_files:
        codec_checkpoint = torch.load(codec_file, map_location='cpu')
        agent_name = codec_checkpoint['agent_name']
        codec = Codec(agent_name, codec_checkpoint['config'])
        codec.load_state_dict(codec_checkpoint['model_state_dict'])
        codec.eval()
        codecs[agent_name] = codec
    
    return hrm, codecs


def evaluate_codec(codec, features: pd.DataFrame, agent_name: str) -> dict:
    """Evaluate a single codec model"""
    # Get instrument-metric inputs
    # For simplicity, use the last row's features
    feature_cols = [c for c in features.columns if c not in ['asset', 'timestamp', 'time']]
    
    if len(feature_cols) < 15:
        # Pad with zeros if not enough features
        input_vec = np.zeros(15)
        for i, col in enumerate(feature_cols[:15]):
            input_vec[i] = features.iloc[-1][col]
    else:
        input_vec = features.iloc[-1][feature_cols[:15]].values
    
    # Get codec output
    with torch.no_grad():
        input_tensor = torch.tensor([input_vec], dtype=torch.float32)
        output = codec(input_tensor)  # [1, 3]
        
        confidence = output[0, 0].item()
        direction = output[0, 1].item()
        regime_fit = output[0, 2].item()
    
    # Simulate trade with this signal
    # For now, use direction as trade signal
    trade_dir = direction
    conf = confidence
    
    # Get returns for simulation
    if 'close' in features.columns:
        returns = features['close'].pct_change().fillna(0).values
    else:
        returns = np.zeros(len(features))
    
    # Simulate
    metrics = simulate(
        np.array([trade_dir]), 
        np.array([conf]), 
        returns[:1],  # Just first return for single prediction
        conf_thresh=0.5,
        max_pos=1.0
    )
    
    return {
        "agent": agent_name,
        "total_pnl": metrics["total_pnl"],
        "total_pnl_dollars": metrics["total_pnl_dollars"],
        "sharpe": metrics["sharpe"],
        "sortino": metrics["sortino"],
        "profit_factor": metrics["profit_factor"],
        "max_drawdown": metrics["max_drawdown"],
        "recovery_factor": metrics["recovery_factor"],
        "kd": metrics["kd"],
        "alpha": metrics["total_pnl"],  # Simplified
    }


def evaluate_hrm(hrm, features: pd.DataFrame) -> dict:
    """Evaluate HRM (emulates all 24 codecs)"""
    # Get instrument-metric inputs
    feature_cols = [c for c in features.columns if c not in ['asset', 'timestamp', 'time']]
    
    if len(feature_cols) < 15:
        input_vec = np.zeros(15)
        for i, col in enumerate(feature_cols[:15]):
            input_vec[i] = features.iloc[-1][col]
    else:
        input_vec = features.iloc[-1][feature_cols[:15]].values
    
    # Get HRM predictions (all 24 codec outputs)
    with torch.no_grad():
        input_tensor = torch.tensor([input_vec], dtype=torch.float32)
        output = hrm.forward(input_tensor)  # [1, 24, 3]
        
        # Use the first codec's output as HRM's signal
        # In real implementation, would combine all 24
        confidence = output[0, 0, 0].item()
        direction = output[0, 0, 1].item()
    
    # Simulate trade
    if 'close' in features.columns:
        returns = features['close'].pct_change().fillna(0).values
    else:
        returns = np.zeros(len(features))
    
    metrics = simulate(
        np.array([direction]), 
        np.array([confidence]), 
        returns[:1],
        conf_thresh=0.5,
        max_pos=1.0
    )
    
    return {
        "agent": "HRM_Meta",
        "total_pnl": metrics["total_pnl"],
        "total_pnl_dollars": metrics["total_pnl_dollars"],
        "sharpe": metrics["sharpe"],
        "sortino": metrics["sortino"],
        "profit_factor": metrics["profit_factor"],
        "max_drawdown": metrics["max_drawdown"],
        "recovery_factor": metrics["recovery_factor"],
        "kd": metrics["kd"],
        "alpha": metrics["total_pnl"],
    }


def print_rankings(rankings: list, sort_by: str):
    """Print rankings as markdown table"""
    print("\n" + "=" * 80)
    print(f"  RANKINGS (sorted by {sort_by})")
    print("=" * 80)
    
    # Print header
    print("| Rank | Agent | Sharpe | Sortino | PF | MaxDD | KD | PnL | Alpha |")
    print("|------|-------|-------:|--------:|---:|------:|---:|----:|------:|")
    
    # Print rows
    for i, r in enumerate(rankings[:25]):  # Show top 25
        print(f"| {i+1:4d} | {r['agent'][:20]:20s} | {r['sharpe']:6.2f} | {r['sortino']:7.2f} | {r['profit_factor']:3.1f} | {r['max_drawdown']:5.2f} | {r['kd']:3.2f} | {r['total_pnl']:4.2f} | {r['alpha']:5.2f} |")


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------
if __name__ == "__main__":
    main()
