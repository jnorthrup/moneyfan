#!/usr/bin/env python3
"""
Quick ranking of hierarchical codec vs baseline signals.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import random

from hrm.hierarchical_codec import HierarchicalCodec
from hrm.pipeline import compute_features, compute_all_signals

N_SIGNALS = 24

def sharpe(pnl):
    return pnl.mean() / pnl.std() if len(pnl) > 1 and pnl.std() > 0 else 0.0

def sortino(pnl):
    downside = pnl[pnl < 0]
    return pnl.mean() / downside.std() if len(downside) > 0 and downside.std() > 0 else 0.0

def compute_signal_tensor(df, signal_df):
    signal_names = [
        'macd_crossover', 'sota_momentum', 'momentum_trend', 'sector_rotation',
        'rsi_mean_reversion', 'bollinger_reversion', 'grid_reversion', 'hrm_mean_reversion',
        'harvest_rebalance', 'kilo_rebalance',
        'volatility_breakout', 'bollinger_vol_regime', 'vol_inverse_sizing',
        'bent_penny', 'pairs_spread',
        'dca_baseline', 'weekly_cadence',
        'technical_ml', 'grid_x_trend', 'rsi_x_trend',
        'momentum_x_vol', 'vol_x_breakout_proven', 'mom_trend_additive', 'rsi_trend_additive',
    ]
    if signal_df is None or len(signal_df) == 0:
        return torch.zeros(len(df), N_SIGNALS * 2)
    T = len(df)
    tensor = np.zeros((T, N_SIGNALS * 2), dtype=np.float32)
    ts_to_idx = {ts: i for i, ts in enumerate(df.index)}
    for i, sig_name in enumerate(signal_names):
        mask = signal_df['model'] == sig_name
        if mask.any():
            for _, row in signal_df[mask].iterrows():
                t_idx = ts_to_idx.get(row['timestamp'])
                if t_idx is not None:
                    tensor[t_idx, i] = row['signal']
                    tensor[t_idx, N_SIGNALS + i] = row['confidence']
    return torch.from_numpy(tensor)

STARTING_CAPITAL = 100.0  # $100 baseline

def main():
    print("=" * 70)
    print("  HIERARCHICAL CODEC RANKING")
    print("=" * 70)
    
    # Load codec
    checkpoint_files = sorted(Path('hrm/checkpoints').glob('hierarchical_codec_*.pt'), reverse=True)
    if not checkpoint_files:
        print("No checkpoint found!")
        return
    
    checkpoint = torch.load(checkpoint_files[0], map_location='cpu', weights_only=False)
    codec = HierarchicalCodec(checkpoint['config'])
    codec.load_state_dict(checkpoint['model_state'])
    codec.eval()
    print(f"\nLoaded: {checkpoint_files[0].name}")
    
    # Load data
    arrow_dir = Path('hrm/data/arrow')
    all_files = list(arrow_dir.glob("*.feather"))
    sample_files = random.sample(all_files, min(5, len(all_files)))
    
    all_results = []
    
    for feather_file in sample_files:
        symbol = feather_file.stem.replace("_", "-")
        
        # Load small window
        df = pd.read_feather(feather_file)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
        elif 'timestamp' in df.columns:
            df['time'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
            df = df.set_index('time')
        
        df = df.sort_index()
        if len(df) < 200:
            continue
        
        # 200 rows max per symbol
        start_idx = random.randint(50, max(50, len(df) - 250))
        window = df.iloc[start_idx:start_idx + 200].copy()
        window['symbol'] = symbol
        
        print(f"\n{symbol}:")
        
        try:
            features_df = compute_features(window)
            signal_df = compute_all_signals(window, features_df)
            
            if signal_df is None or len(signal_df) == 0:
                print("  No signals")
                continue
            
            signals = compute_signal_tensor(window, signal_df)
            returns = window['close'].pct_change().fillna(0).values
            
            # === Codec ===
            seq_len = 16
            all_preds, all_rets, all_confs = [], [], []
            
            with torch.no_grad():
                for i in range(0, min(len(signals) - seq_len - 1, 30), 3):
                    batch = signals[i:i+seq_len+1].unsqueeze(0)
                    output, _ = codec(batch, mode='trade')
                    pred_ret = output[0, 0].item()
                    conf = output[0, 1].item()
                    actual_ret = returns[i+seq_len] if i+seq_len < len(returns) else 0
                    all_preds.append(pred_ret)
                    all_rets.append(actual_ret)
                    all_confs.append(conf)
            
            if len(all_preds) > 0:
                preds = np.array(all_preds)
                rets = np.array(all_rets)
                confs = np.array(all_confs)
                
                pos = np.sign(preds) * confs
                pnl_pct = pos * rets  # fractional returns
                pnl_dollars = pnl_pct * STARTING_CAPITAL  # $100 starting capital
                
                result = {
                    'symbol': symbol,
                    'agent': 'hierarchical_codec',
                    'pnl_pct': np.sum(pnl_pct) * 100,  # as percentage
                    'pnl_dollars': np.sum(pnl_dollars),  # dollar figure
                    'sharpe': sharpe(pnl_pct),
                    'dir_acc': np.mean(np.sign(preds) == np.sign(rets)),
                }
                all_results.append(result)
                print(f"  codec: PnL=${result['pnl_dollars']:+.2f} ({result['pnl_pct']:+.2f}%), Sharpe={result['sharpe']:+.2f}")
            
            # === Baseline signals ===
            signal_names = ['macd_crossover', 'bollinger_reversion', 'volatility_breakout', 'rsi_mean_reversion']
            
            for sig_name in signal_names:
                mask = signal_df['model'] == sig_name
                if not mask.any():
                    continue
                
                # Get signal values aligned with window
                ts_to_idx = {ts: i for i, ts in enumerate(window.index)}
                sig_vals = []
                sig_confs = []
                actual_rets = []
                
                for _, row in signal_df[mask].iterrows():
                    t_idx = ts_to_idx.get(row['timestamp'])
                    if t_idx is not None and t_idx < len(returns):
                        sig_vals.append(row['signal'])
                        sig_confs.append(row['confidence'])
                        actual_rets.append(returns[t_idx])
                
                if len(sig_vals) < 10:
                    continue
                
                sig_vals = np.array(sig_vals)
                sig_confs = np.array(sig_confs)
                actual_rets = np.array(actual_rets)
                
                pos = sig_vals * sig_confs
                pnl_pct = pos * actual_rets  # fractional returns
                pnl_dollars = pnl_pct * STARTING_CAPITAL  # $100 starting capital
                
                result = {
                    'symbol': symbol,
                    'agent': sig_name,
                    'pnl_pct': np.sum(pnl_pct) * 100,  # as percentage
                    'pnl_dollars': np.sum(pnl_dollars),  # dollar figure
                    'sharpe': sharpe(pnl_pct),
                    'dir_acc': np.mean(np.sign(sig_vals) == np.sign(actual_rets)),
                }
                all_results.append(result)
                print(f"  {sig_name}: PnL=${result['pnl_dollars']:+.2f} ({result['pnl_pct']:+.2f}%), Sharpe={result['sharpe']:+.2f}")
                
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Print summary
    print("\n" + "=" * 70)
    print("  SUMMARY (sorted by Sharpe)")
    print("=" * 70)
    
    all_results.sort(key=lambda x: x['sharpe'], reverse=True)
    
    print(f"\n| {'Rank':>4} | {'Agent':<22} | {'PnL $':>8} | {'PnL %':>7} | {'Sharpe':>7} | {'DirAcc':>6} |")
    print("|------|------------------------|---------|--------|---------|--------|")
    
    for i, r in enumerate(all_results[:15]):
        print(f"| {i+1:4d} | {r['agent']:<22} | ${r['pnl_dollars']:>+6.2f} | {r['pnl_pct']:>+5.2f}% | {r['sharpe']:>+7.2f} | {r['dir_acc']:>5.0%} |")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
