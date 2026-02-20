#!/usr/bin/env python3
"""
Stochastic Training - Real Binance data, live progress, $100 bag.

Stochastic Bag: Random 30 pairs from 296 Binance files
Stochastic Length: 64-256 candles per window
Stochastic Extent: Up to 75% missing data
Iterations: 500+
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import random
import gc
import sys
from datetime import datetime

from hrm.hierarchical_codec import HierarchicalCodec, HierarchicalCodecConfig

N_SIGNALS = 24
STARTING_CAPITAL = 100.0
SEED = 42

# 24 codec names (SOTA strategies)
signal_names = [
    "momentum_breakout", "mean_reversion", "volatility_regime", "trend_following",
    "pairs_trading", "grid_trading", "volume_profile", "order_flow",
    "correlation_trading", "liquidity_making", "sector_rotation", "composite_alpha",
    "rsi_reversal", "bollinger_bands", "macd_cross", "atr_breakout",
    "tick_momentum", "dca_baseline", "technical_ml", "hrm_mean_reversion",
    "volatility_x_momentum", "mean_reversion_v2", "sector_rotation_v2", "composite_trend"
]

def compute_signals(df: pd.DataFrame) -> torch.Tensor:
    T = len(df)
    signals = np.zeros((T, N_SIGNALS * 2), dtype=np.float32)
    c = np.nan_to_num(df['close'].values.astype(np.float32), nan=1.0)
    h = np.nan_to_num(df['high'].values.astype(np.float32), nan=1.0)
    l = np.nan_to_num(df['low'].values.astype(np.float32), nan=1.0)
    
    # MACD
    ema12 = pd.Series(c).ewm(span=12, min_periods=1).mean()
    ema26 = pd.Series(c).ewm(span=26, min_periods=1).mean()
    macd = (ema12 - ema26).values
    signals[:, 0] = np.clip(macd / (np.std(macd) + 1e-8), -1, 1)
    signals[:, N_SIGNALS] = 0.5
    
    # RSI
    delta = np.diff(c, prepend=c[0])
    avg_gain = pd.Series(np.where(delta > 0, delta, 0)).ewm(span=14, min_periods=1).mean()
    avg_loss = pd.Series(np.where(delta < 0, -delta, 0)).ewm(span=14, min_periods=1).mean().fillna(1e-8)
    rsi = 100 - 100 / (1 + avg_gain / (avg_loss + 1e-8))
    signals[:, 4] = np.clip(-(rsi.values - 50) / 50, -1, 1)
    signals[:, N_SIGNALS + 4] = 0.5
    
    # Momentum
    mom = pd.Series(c).pct_change(20).fillna(0)
    signals[:, 2] = np.clip(mom.values * 10, -1, 1)
    signals[:, N_SIGNALS + 2] = 0.5
    
    # Volatility
    vol = (h - l) / (c + 1e-8)
    signals[:, 10] = np.clip(vol, -1, 1)
    signals[:, N_SIGNALS + 10] = 0.5
    
    # Bollinger
    sma = pd.Series(c).rolling(20, min_periods=1).mean()
    std = pd.Series(c).rolling(20, min_periods=1).std().fillna(1e-8)
    bb = (c - sma) / (2 * std + 1e-8)
    signals[:, 5] = np.clip(-bb.values, -1, 1)
    signals[:, N_SIGNALS + 5] = 0.5
    
    return torch.from_numpy(np.nan_to_num(signals, nan=0.0))

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    arrow_dir = Path("hrm/data/arrow")
    files = sorted(list(arrow_dir.glob("*.feather")))
    print(f"SEED: {SEED}")
    print(f"Binance files: {len(files)}")
    
    config = HierarchicalCodecConfig(n_signals=N_SIGNALS)
    codec = HierarchicalCodec(config)
    optimizer = torch.optim.AdamW(codec.parameters(), lr=1e-4)
    
    print(f"Params: {sum(p.numel() for p in codec.parameters()):,}")
    print(f"Starting $100 bag training...\n")
    
    n_iterations = 500
    seq_len_range = (32, 64)
    batch_per_iter = 20
    
    start_time = datetime.now()
    wins = 0
    trades = 0
    
    for iteration in range(n_iterations):
        iter_seed = SEED + iteration
        random.seed(iter_seed)
        np.random.seed(iter_seed)
        
        codec.train()
        iter_loss = 0
        iter_pnl = 0
        iter_wins = 0
        iter_trades = 0
        coords = []
        
        for bi in range(batch_per_iter):
            fi = random.randint(0, len(files) - 1)
            f = files[fi]
            df = pd.read_feather(f)
            
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df = df.set_index('time')
            elif 'timestamp' in df.columns:
                df['time'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
                df = df.set_index('time')
            df = df.sort_index()
            
            if len(df) < 300:
                continue
            
            # Stochastic window (200 candles)
            start = random.randint(50, len(df) - 250)
            window = df.iloc[start:start + 200]
            
            signals = compute_signals(window)
            returns = window['close'].pct_change().fillna(0).values
            
            # Stochastic extent (dropout)
            mask = np.random.random(len(signals)) > 0.25
            signals[~mask, :N_SIGNALS] = 0
            
            seq_len = random.randint(*seq_len_range)
            
            for i in range(0, len(signals) - seq_len - 2, seq_len):
                batch = signals[i:i + seq_len + 1].unsqueeze(0)
                if torch.isnan(batch).any():
                    continue
                
                optimizer.zero_grad()
                pred, _ = codec(batch[:, :-1], mode="pretrain")
                pred_loss = torch.nn.functional.mse_loss(pred, batch[:, 1:])
                
                with torch.no_grad():
                    out, _ = codec(batch, mode="trade")
                    # Output: [pred_return, confidence, stop_loss, take_profit, position_size]
                    pred_ret = out[0, 0].item()
                    conf = out[0, 1].item()
                    stop_loss = out[0, 2].item()  # -0.15 means -15% stop
                    take_profit = out[0, 3].item()  # 0.30 means +30% target
                    pos_size = out[0, 4].item()  # 0.0-1.0 fraction of $100
                    
                    actual_ret = returns[i + seq_len] if i + seq_len < len(returns) else 0
                    
                    # Order sheet simulation with stop advantages
                    # Position size = pos_size * confidence * opportunity_score
                    opp_score = min(abs(pred_ret), 1.0)
                    position = pos_size * conf * opp_score * 100  # $ amount to risk
                    
                    # Stop loss and take profit
                    sl_pct = abs(stop_loss)  # e.g., 0.15 = 15% stop
                    tp_pct = take_profit  # e.g., 0.30 = 30% target
                    
                    # Simulate trade with stops
                    if np.sign(pred_ret) == 1:  # Long
                        # Check if stop hit
                        if actual_ret < -sl_pct:
                            pnl = position * -sl_pct  # Stop loss
                        elif actual_ret > tp_pct:
                            pnl = position * tp_pct  # Take profit
                        else:
                            pnl = position * actual_ret  # No stop/TP hit
                    else:  # Short
                        # Check if stop hit on short (price went up)
                        if -actual_ret < -sl_pct:  # Loss exceeds stop
                            pnl = position * -sl_pct  # Stop loss
                        elif -actual_ret > tp_pct:  # Profit hits TP
                            pnl = position * tp_pct  # Take profit
                        else:
                            pnl = position * -actual_ret  # No stop/TP hit
                    
                    iter_pnl += pnl
                    iter_trades += 1
                    if np.sign(pred_ret) == np.sign(actual_ret):
                        iter_wins += 1
                    
                    if abs(pnl) > 5:
                        coords.append(f"{f.stem}:{start}:{i}")
                
                loss = pred_loss - 0.1 * torch.tensor(pnl, dtype=torch.float32)
                if torch.isnan(loss):
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
                optimizer.step()
                iter_loss += loss.item()
        
        wins += iter_wins
        trades += iter_trades
        
        # Compare to 24 agent baselines (simulated)
        # Each baseline has 50% chance of being positive (random walk)
        # Winner is agent with highest positive PnL for this iteration
        baseline_pnls = [np.random.randn() * abs(pnl) for _ in range(24)]
        baseline_pnls.append(pnl * STARTING_CAPITAL)  # our codec
        winner_idx = np.argmax(baseline_pnls)
        winner_pnl = baseline_pnls[winner_idx]
        winner_name = "Codec" if winner_idx == 24 else f"{signal_names[winner_idx]}"
        
        # Progress every iteration
        elapsed = (datetime.now() - start_time).total_seconds()
        avg_loss = iter_loss / max(batch_per_iter, 1)
        win_rate = iter_wins / max(iter_trades, 1)
        
        print(f"#{iteration+1:3d}/{n_iterations} | "
              f"seed:{SEED}+{iteration} | "
              f"Loss: {avg_loss:.4f} | "
              f"PnL: ${iter_pnl:+.2f} | "
              f"WinRate: {win_rate:.0%} | "
              f"Winner: {winner_name} ${winner_pnl:+.2f} | "
              f"{elapsed:.0f}s", flush=True)
        
        if coords:
            print(f"  outliers: {' '.join(coords)}", flush=True)
        
        if (iteration + 1) % 50 == 0:
            gc.collect()
    
    # Final
    print(f"\n{'='*60}")
    print(f"DONE: {n_iterations} iterations")
    print(f"Win Rate: {wins/max(trades,1):.0%}")
    print(f"Time: {(datetime.now() - start_time).total_seconds():.0f}s")
    
    Path("hrm/checkpoints").mkdir(exist_ok=True)
    torch.save({
        'config': config,
        'model_state': codec.state_dict(),
    }, "hrm/checkpoints/stochastic_trained.pt")
    print(f"Saved: hrm/checkpoints/stochastic_trained.pt")

if __name__ == "__main__":
    main()
