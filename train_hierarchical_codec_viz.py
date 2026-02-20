#!/usr/bin/env python3
"""
train_hierarchical_codec_viz.py

Train hierarchical codec with live visualization of:
- Training loss over time
- Convergence metrics
- Signal prediction accuracy
- Model progress

Usage:
    python train_hierarchical_codec_viz.py
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
import time

from hrm.hierarchical_codec import HierarchicalCodec, HierarchicalCodecConfig, CodecTrainer
from hrm.duck_store import DuckStore
from hrm.pipeline import compute_all_signals, compute_features, Regime

N_SIGNALS = 24


class TrainingVisualizer:
    """Live training progress visualization."""
    
    def __init__(self):
        self.history = {
            'epoch': [],
            'pretrain_loss': [],
            'finetune_loss': [],
            'signal_mse': [],
            'direction_acc': [],
            'confidence_mean': [],
            'convergence': [],
            'sparkline_entropy': [],
        }
        self.start_time = time.time()
    
    def update(self, epoch: int, pretrain_loss: float = None, finetune_loss: float = None,
               signal_mse: float = None, direction_acc: float = None, 
               confidence_mean: float = None, convergence: float = None,
               sparkline_entropy: float = None):
        """Update metrics."""
        self.history['epoch'].append(epoch)
        self.history['pretrain_loss'].append(pretrain_loss or 0)
        self.history['finetune_loss'].append(finetune_loss or 0)
        self.history['signal_mse'].append(signal_mse or 0)
        self.history['direction_acc'].append(direction_acc or 0)
        self.history['confidence_mean'].append(confidence_mean or 0)
        self.history['convergence'].append(convergence or 0)
        self.history['sparkline_entropy'].append(sparkline_entropy or 0)
    
    def print_progress(self, phase: str, epoch: int, total: int, metrics: dict):
        """Print progress bar with metrics."""
        elapsed = time.time() - self.start_time
        
        bar_len = 30
        filled = int(bar_len * epoch / total)
        bar = '█' * filled + '░' * (bar_len - filled)
        
        print(f"\r{phase} [{bar}] {epoch}/{total} | "
              f"Loss: {metrics.get('loss', 0):.4f} | "
              f"Acc: {metrics.get('acc', 0):.1%} | "
              f"Conf: {metrics.get('conf', 0):.2f} | "
              f"Time: {elapsed:.0f}s", end='')
    
    def print_summary(self):
        """Print training summary."""
        print("\n\n" + "=" * 70)
        print("  TRAINING SUMMARY")
        print("=" * 70)
        
        if len(self.history['epoch']) > 0:
            print(f"\n  Pre-train loss: {self.history['pretrain_loss'][0]:.4f} → {self.history['pretrain_loss'][-1]:.4f}")
            print(f"  Direction accuracy: {self.history['direction_acc'][-1]:.1%}")
            print(f"  Mean confidence: {self.history['confidence_mean'][-1]:.2f}")
            print(f"  Convergence: {self.history['convergence'][-1]:.2f}")
        
        total_time = time.time() - self.start_time
        print(f"\n  Total training time: {total_time:.1f}s")
        print("=" * 70)
    
    def print_live_chart(self, metric: str = 'pretrain_loss'):
        """Print ASCII chart of metric over time."""
        values = self.history.get(metric, [])
        if len(values) < 2:
            return
        
        print(f"\n\n{metric} over time:")
        
        h = 8
        w = 50
        chart = [[' ' for _ in range(w)] for _ in range(h)]
        
        v_min = min(values)
        v_max = max(values)
        v_range = v_max - v_min if v_max != v_min else 1
        
        for i, v in enumerate(values):
            if i >= w:
                break
            y = int((v - v_min) / v_range * (h - 1))
            y = h - 1 - y
            chart[y][i] = '●'
            if i > 0:
                prev_v = values[i - 1]
                prev_y = int((prev_v - v_min) / v_range * (h - 1))
                prev_y = h - 1 - prev_y
                for yy in range(min(y, prev_y), max(y, prev_y) + 1):
                    if chart[yy][i] == ' ':
                        chart[yy][i] = '│'
        
        for row in chart:
            print('  ' + ''.join(row))
        
        print(f"  {v_min:.4f}" + ' ' * (w - 16) + f"{v_max:.4f}")


def load_stochastic_bag(arrow_dir: str, n_pairs: int = 30, min_rows: int = 500):
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
                print(f"  ✓ {symbol}: {len(df)} rows")
        except Exception as e:
            pass
    
    return bag


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


def create_batch(bag: list, seq_len: int = 32, min_history: int = 100):
    """Create a single training batch."""
    symbol, df = random.choice(bag)
    
    if len(df) < seq_len + min_history + 10:
        return None, None
    
    max_start = len(df) - seq_len - min_history
    if max_start <= min_history:
        return None, None
    
    start = random.randint(min_history, max_start)
    window = df.iloc[start:start + seq_len + min_history].copy()
    
    if 'symbol' not in window.columns:
        window['symbol'] = symbol
    
    try:
        features_df = compute_features(window)
        signal_df = compute_all_signals(window, features_df)
    except Exception:
        return None, None
    
    if signal_df is None or len(signal_df) == 0:
        return None, None
    
    signals = compute_signal_tensor(window, signal_df)
    
    if signals.shape[0] < seq_len + 1:
        return None, None
    
    signals = signals[:seq_len + 1].unsqueeze(0)
    
    returns = window['close'].pct_change().fillna(0).values
    target_return = torch.tensor([returns[-seq_len-1]], dtype=torch.float32) if len(returns) > seq_len else torch.tensor([0.0])
    
    return signals, target_return


def compute_metrics(codec: HierarchicalCodec, bag: list, n_samples: int = 20):
    """Compute validation metrics."""
    codec.eval()
    
    total_signal_mse = 0.0
    total_direction_correct = 0
    total_direction_total = 0
    total_confidence = 0.0
    total_convergence = 0.0
    n_valid = 0
    
    with torch.no_grad():
        for _ in range(n_samples):
            signals, returns = create_batch(bag)
            if signals is None:
                continue
            
            B, T, F = signals.shape
            
            output, memory = codec(signals[:, :-1, :], mode="pretrain")
            
            target = signals[:, 1:, :]
            signal_mse = torch.nn.functional.mse_loss(output, target).item()
            total_signal_mse += signal_mse
            
            output, memory = codec(signals[:, :-1, :], mode="trade")
            pred_return = output[0, 0].item()
            conf = output[0, 1].item()
            
            actual_return = returns.item() if returns is not None else 0
            
            if conf > 0.3:
                total_direction_total += 1
                if np.sign(pred_return) == np.sign(actual_return):
                    total_direction_correct += 1
            
            total_confidence += conf
            
            if isinstance(memory, tuple) and len(memory) > 0:
                sparkline = memory[0]
                if sparkline is not None:
                    entropy = -torch.sum(sparkline * torch.log(sparkline + 1e-8)).item()
                    total_convergence += min(entropy / 100, 1.0)
            
            n_valid += 1
    
    codec.train()
    
    if n_valid == 0:
        return {'signal_mse': 0, 'direction_acc': 0, 'confidence': 0, 'convergence': 0}
    
    return {
        'signal_mse': total_signal_mse / n_valid,
        'direction_acc': total_direction_correct / max(total_direction_total, 1),
        'confidence': total_confidence / n_valid,
        'convergence': total_convergence / n_valid,
    }


def train_with_visualization(
    arrow_dir: str = "hrm/data/arrow",
    checkpoint_dir: str = "hrm/checkpoints",
    n_epochs_pretrain: int = 20,
    n_epochs_finetune: int = 10,
    n_batches_per_epoch: int = 50,
    seq_len: int = 32,
    n_pairs: int = 15
):
    """Train with live visualization."""
    
    print("=" * 70)
    print("  HIERARCHICAL CODEC TRAINING WITH VISUALIZATION")
    print("=" * 70)
    
    config = HierarchicalCodecConfig(n_signals=N_SIGNALS)
    codec = HierarchicalCodec(config)
    optimizer = torch.optim.AdamW(codec.parameters(), lr=1e-4)
    
    viz = TrainingVisualizer()
    
    print(f"\nModel: {sum(p.numel() for p in codec.parameters()):,} parameters")
    print(f"Config: H_cycles={config.H_cycles}, L_cycles={config.L_cycles}, "
          f"H_layers={config.H_layers}, L_layers={config.L_layers}")
    print(f"        Sparkline: {config.sparkline_frames} frames, horizon={config.sparkline_horizon}")
    
    print(f"\n{'─' * 70}")
    print("Loading stochastic bag...")
    bag = load_stochastic_bag(arrow_dir, n_pairs=n_pairs)
    
    if len(bag) == 0:
        print("ERROR: No data loaded!")
        return
    
    print(f"\nLoaded {len(bag)} pairs")
    
    print(f"\n{'=' * 70}")
    print("  PHASE 1: PRE-TRAINING (predict next signals)")
    print("=" * 70)
    
    for epoch in range(n_epochs_pretrain):
        epoch_loss = 0.0
        n_valid = 0
        
        for batch_idx in range(n_batches_per_epoch):
            signals, _ = create_batch(bag, seq_len)
            if signals is None:
                continue
            
            optimizer.zero_grad()
            loss, _ = codec.pretrain_loss(signals)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_valid += 1
            
            viz.print_progress("Pre-train", batch_idx + 1, n_batches_per_epoch, 
                              {'loss': epoch_loss / n_valid})
        
        if n_valid > 0:
            metrics = compute_metrics(codec, bag)
            viz.update(
                epoch=epoch,
                pretrain_loss=epoch_loss / n_valid,
                signal_mse=metrics['signal_mse'],
                direction_acc=metrics['direction_acc'],
                confidence_mean=metrics['confidence'],
                convergence=metrics['convergence']
            )
            
            print(f"\n  Epoch {epoch+1:2d}/{n_epochs_pretrain}: "
                  f"Loss={epoch_loss/n_valid:.4f} | "
                  f"Signal MSE={metrics['signal_mse']:.4f} | "
                  f"Dir Acc={metrics['direction_acc']:.1%} | "
                  f"Conf={metrics['confidence']:.2f}")
    
    viz.print_live_chart('pretrain_loss')
    
    print(f"\n{'=' * 70}")
    print("  PHASE 2: FINE-TUNING (predict returns)")
    print("=" * 70)
    
    for epoch in range(n_epochs_finetune):
        epoch_loss = 0.0
        n_valid = 0
        total_return = 0.0
        total_conf = 0.0
        
        for batch_idx in range(n_batches_per_epoch):
            signals, returns = create_batch(bag, seq_len)
            if signals is None or returns is None:
                continue
            
            optimizer.zero_grad()
            loss, _, pred_ret, conf = codec.trade_loss(signals, returns)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            total_return += pred_ret.abs().mean().item()
            total_conf += conf.mean().item()
            n_valid += 1
            
            viz.print_progress("Fine-tune", batch_idx + 1, n_batches_per_epoch,
                              {'loss': epoch_loss / n_valid, 
                               'acc': total_conf / n_valid,
                               'conf': total_conf / n_valid})
        
        if n_valid > 0:
            metrics = compute_metrics(codec, bag)
            viz.update(
                epoch=n_epochs_pretrain + epoch,
                finetune_loss=epoch_loss / n_valid,
                direction_acc=metrics['direction_acc'],
                confidence_mean=metrics['confidence'],
                convergence=metrics['convergence']
            )
            
            print(f"\n  Epoch {epoch+1:2d}/{n_epochs_finetune}: "
                  f"Loss={epoch_loss/n_valid:.4f} | "
                  f"Avg |Ret|={total_return/n_valid:.4f} | "
                  f"Conf={total_conf/n_valid:.2f} | "
                  f"Dir Acc={metrics['direction_acc']:.1%}")
    
    viz.print_live_chart('direction_acc')
    viz.print_live_chart('confidence_mean')
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(checkpoint_dir, f"hierarchical_codec_{timestamp}.pt")
    torch.save({
        'config': config,
        'model_state': codec.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'history': viz.history,
    }, checkpoint_path)
    print(f"\nSaved: {checkpoint_path}")
    
    viz.print_summary()


if __name__ == "__main__":
    train_with_visualization()
