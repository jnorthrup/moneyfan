import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from hrm.coinbase_pipeline import CoinbasePipeline
from hrm.orchestrator_bridge import OrchestratorBridge
from hrm.signal_hrm import SignalHRM, SignalHRMConfig, portfolio_loss

class StrategyCodecTrainer:
    """
    Trains HRM to act as a 'Codec Predictor' for tradebot strategies.
    Maps: Tradebot Signals (T) + Sparkline (T) -> Realized Profit (T+k)
    """
    
    def __init__(self, target_window: int = 5, optimizer: str = "muon"):
        self.pipeline = CoinbasePipeline()
        self.bridge = OrchestratorBridge()
        self.target_window = target_window # k-bars lookahead
        
        # Configuration
        self.cfg = SignalHRMConfig(seq_len=16, hidden_dim=64)
        self.model = SignalHRM(self.cfg)
        
        # Select SOTA optimizer
        if HAS_MLX:
            if optimizer == "lion":
                self.optimizer = optim.Lion(learning_rate=5e-4)
                print(f"[StrategyCodecTrainer] Using LION optimizer")
            else:
                self.optimizer = optim.Muon(learning_rate=5e-4)
                print(f"[StrategyCodecTrainer] Using MUON optimizer")
        else:
            self.optimizer = None
        
        # Stats
        self.history = []

    def run_epoch(self, n_bags: int = 10):
        print(f"Codec Training Epoch Started | Target Window: {self.target_window} bars")
        
        for i in range(n_bags):
            bags = self.pipeline.sample_stratified_bag(n_samples=1)
            if not bags: continue
            
            df = bags[0]
            symbols = df['symbol'].unique()
            
            for sym in symbols:
                # 1. 'Before' State: Market Tensor at time T
                
                # Slicing and indikators
                sym_df = df[df['symbol'] == sym]
                if len(sym_df) < self.cfg.seq_len + self.target_window + 5:
                    continue

                # Full signal computation for the bag
                res = self.bridge.orchestrator.run(sym_df)
                
                # Recalculate sparkline for the full series
                close = sym_df['close']
                
                # 2. 'After' State: Target Realization
                realized_ret = close.pct_change(self.target_window).shift(-self.target_window)
                
                # 3. Training Loop over valid indices
                valid_indices = realized_ret.dropna().index
                if len(valid_indices) < self.cfg.seq_len + 2: 
                    print(f"Skipping {sym}: too few valid indices ({len(valid_indices)})")
                    continue
                
                # For efficiency, we pick a few random windows from this bag
                match_count = 0
                for _ in range(5):
                    # Ensure we have at least 36 bars for indicators-room
                    idx_start = self.cfg.seq_len + 25
                    if idx_start >= len(valid_indices):
                         # Fallback to the latest possible
                         idx = len(valid_indices) - 1
                    else:
                         idx = np.random.randint(idx_start, len(valid_indices))
                    
                    t_end = valid_indices[idx]
                    
                    # Target realization for this window
                    target_val = realized_ret.loc[t_end]
                    
                    sub_df = sym_df.loc[:t_end]
                    x = self.bridge.compute_tensor(sym, df_input=sub_df, seq_len=16) 
                    
                    if x is not None and HAS_MLX:
                        match_count += 1
                        # Forward pass to get current prediction
                        weights, alpha, convergence, _mem = self.model(mx.array(x))
                        pred_alpha = float(alpha)
                        
                        # Convert target to tensor
                        y = mx.array([target_val])
                        
                        def loss_fn(model, x, y):
                            weights, alpha, convergence, _mem = model(x)
                            return portfolio_loss(weights, alpha, convergence, y)
                            
                        loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)
                        loss, grads = loss_and_grad_fn(self.model, mx.array(x), y)
                        self.optimizer.update(self.model, grads)
                        mx.eval(self.model.parameters(), self.optimizer.state)
                        
                        self.history.append({
                            'symbol': sym,
                            'loss': float(loss),
                            'predicted_alpha': pred_alpha,
                            'realized': target_val
                        })
                
                if match_count > 0:
                    print(f"      Matched {match_count} windows for {sym}")
                else:
                    # Debug: Why did it fail?
                    print(f"      No windows matched for {sym}. Valid indices: {len(valid_indices)}, idx_start: {idx_start}")

            print(f"   Progress: Bag {i+1}/{n_bags} processed.")

    def report(self):
        if not self.history: 
            print("\nNO SAMPLES COLLECTED for Before/After analysis.")
            return
        df_hist = pd.DataFrame(self.history)
        
        # Calculate Correlation
        corr = df_hist['predicted_alpha'].corr(df_hist['realized'])
        
        print("\n" + "="*70)
        print("  CODEC TRAINING ANALYSIS: 'BEFORE' VS 'AFTER'")
        print("="*70)
        print(f"Total Samples: {len(df_hist)}")
        print(f"Avg Loss:      {df_hist['loss'].mean():.6f}")
        print(f"Correlation (Alpha vs Realization): {corr:.4f}")
        
        print("\nSAMPLE PARITY (Before -> After):")
        print(f"{'Symbol':<12} {'Pred Alpha':>12} {'Realized Ret':>12}")
        for idx, row in df_hist.tail(10).iterrows():
            print(f"{row['symbol']:<12} {row['predicted_alpha']:12.4f} {row['realized']:12.4f}")
        
        print("="*70)

if __name__ == "__main__":
    trainer = StrategyCodecTrainer()
    trainer.run_epoch(n_bags=5)
    trainer.report()
