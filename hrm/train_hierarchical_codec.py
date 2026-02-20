"""
Train Hierarchical Codec with Pandas Harness

Two-phase training:
1. Pre-train: predict next signals (self-supervised)
2. Fine-tune: predict returns (supervised)

Data flow:
    DuckDB/Arrow → pandas → signals (24x2) → HierarchicalCodec
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
import random

from hrm.hierarchical_codec import HierarchicalCodec, HierarchicalCodecConfig, CodecTrainer
from hrm.duck_store import DuckStore
from hrm.pipeline import compute_all_signals, compute_features, Regime


N_SIGNALS = 24


def load_stochastic_bag(
    arrow_dir: str, 
    n_pairs: int = 30,
    min_rows: int = 200
) -> List[Tuple[str, pd.DataFrame]]:
    """
    Load stochastic bag of pairs from Arrow/DuckDB.
    
    GOALS.md:
    - 30 pairs + USD in stochastic bag
    - Up to 75% missing data allowed
    """
    store = DuckStore(arrow_dir=arrow_dir)
    
    all_files = list(Path(arrow_dir).glob("*.feather"))
    if len(all_files) == 0:
        print(f"No .feather files in {arrow_dir}")
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
                print(f"  Loaded {symbol}: {len(df)} rows")
        except Exception as e:
            print(f"  Failed {symbol}: {e}")
    
    return bag


def compute_signal_tensor(df: pd.DataFrame, signal_df: pd.DataFrame) -> torch.Tensor:
    """
    Convert signal DataFrame to tensor for codec.
    
    Output shape: [T, N_SIGNALS * 2]
    - First N_SIGNALS: signal values [-1, 1]
    - Next N_SIGNALS: confidence values [0, 1]
    """
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
    
    if isinstance(signal_df, list):
        signal_df = pd.DataFrame(signal_df)
    
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


def compute_returns(df: pd.DataFrame, horizon: int = 1) -> torch.Tensor:
    """Compute forward returns as target for fine-tuning."""
    close = df['close'].values
    returns = np.zeros(len(close), dtype=np.float32)
    returns[:-horizon] = (close[horizon:] - close[:-horizon]) / (close[:-horizon] + 1e-8)
    return torch.from_numpy(returns)


def create_training_batches(
    bag: List[Tuple[str, pd.DataFrame]],
    seq_len: int = 32,
    n_batches: int = 100,
    pretrain: bool = True
) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """
    Create training batches from stochastic bag.
    
    For pre-training: (signals, None)
    For fine-tuning: (signals, returns)
    
    Note: Pipeline requires >= 60 rows for rolling calculations,
    so we use a larger window and then trim to seq_len.
    """
    batches = []
    min_history = 100
    
    for _ in range(n_batches):
        symbol, df = random.choice(bag)
        
        if len(df) < seq_len + min_history + 10:
            continue
        
        max_start = len(df) - seq_len - min_history
        if max_start <= min_history:
            continue
        
        start = random.randint(min_history, max_start)
        window = df.iloc[start:start + seq_len + min_history].copy()
        
        if 'symbol' not in window.columns:
            window['symbol'] = symbol
        
        try:
            features_df = compute_features(window)
            signal_df = compute_all_signals(window, features_df)
        except Exception as e:
            continue
        
        if signal_df is None or len(signal_df) == 0:
            continue
        
        signals = compute_signal_tensor(window, signal_df)
        
        if signals.shape[0] < seq_len + 1:
            continue
        
        signals = signals[:seq_len + 1]
        
        signals = signals.unsqueeze(0)
        
        if pretrain:
            batches.append((signals, None))
        else:
            returns = compute_returns(window, horizon=1)
            target_return = returns[-seq_len-1:-seq_len].unsqueeze(0)
            batches.append((signals, target_return))
    
    return batches


def train_codec(
    arrow_dir: str = "hrm/data/arrow",
    checkpoint_dir: str = "hrm/checkpoints",
    n_epochs_pretrain: int = 10,
    n_epochs_finetune: int = 5,
    n_batches_per_epoch: int = 100,
    seq_len: int = 32
):
    """
    Train hierarchical codec with pandas harness.
    """
    print("=" * 60)
    print("  HIERARCHICAL CODEC TRAINING")
    print("=" * 60)
    
    config = HierarchicalCodecConfig(n_signals=N_SIGNALS)
    trainer = CodecTrainer(config)
    
    print(f"\nModel: {sum(p.numel() for p in trainer.model.parameters()):,} parameters")
    print(f"Config: H_cycles={config.H_cycles}, L_cycles={config.L_cycles}")
    print(f"        H_layers={config.H_layers}, L_layers={config.L_layers}")
    print(f"        Sparkline: {config.sparkline_frames} frames, horizon={config.sparkline_horizon}")
    
    print("\n" + "-" * 60)
    print("Loading stochastic bag from Arrow...")
    bag = load_stochastic_bag(arrow_dir, n_pairs=30)
    
    if len(bag) == 0:
        print("ERROR: No data loaded!")
        return
    
    print(f"\nLoaded {len(bag)} pairs")
    
    print("\n" + "=" * 60)
    print("  PHASE 1: PRE-TRAINING (predict next signals)")
    print("=" * 60)
    
    for epoch in range(n_epochs_pretrain):
        batches = create_training_batches(bag, seq_len=seq_len, n_batches=n_batches_per_epoch, pretrain=True)
        
        if len(batches) == 0:
            print(f"Epoch {epoch}: No valid batches")
            continue
        
        total_loss = 0.0
        for signals, _ in batches:
            loss = trainer.pretrain_step(signals)
            total_loss += loss
        
        avg_loss = total_loss / len(batches)
        print(f"Epoch {epoch:2d}/{n_epochs_pretrain}: Pre-train loss = {avg_loss:.4f}")
    
    print("\n" + "=" * 60)
    print("  PHASE 2: FINE-TUNING (predict returns)")
    print("=" * 60)
    
    for epoch in range(n_epochs_finetune):
        batches = create_training_batches(bag, seq_len=seq_len, n_batches=n_batches_per_epoch, pretrain=False)
        
        if len(batches) == 0:
            print(f"Epoch {epoch}: No valid batches")
            continue
        
        total_loss = 0.0
        total_return = 0.0
        total_conf = 0.0
        
        for signals, returns in batches:
            loss, pred_ret, conf = trainer.finetune_step(signals, returns)
            total_loss += loss
            total_return += pred_ret.abs().mean().item()
            total_conf += conf.mean().item()
        
        avg_loss = total_loss / len(batches)
        avg_ret = total_return / len(batches)
        avg_conf = total_conf / len(batches)
        print(f"Epoch {epoch:2d}/{n_epochs_finetune}: Trade loss = {avg_loss:.4f} | "
              f"Avg |return| = {avg_ret:.4f} | Avg conf = {avg_conf:.2f}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(checkpoint_dir, f"hierarchical_codec_{timestamp}.pt")
    trainer.save(checkpoint_path)
    print(f"\nSaved checkpoint: {checkpoint_path}")
    
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Pre-training: {n_epochs_pretrain} epochs")
    print(f"  Fine-tuning:  {n_epochs_finetune} epochs")
    print(f"  Total params: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print("=" * 60)


if __name__ == "__main__":
    train_codec()
