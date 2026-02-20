import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from hrm.signal_hrm import SignalHRM, SignalHRMConfig, portfolio_loss, N_SIGNALS
from hrm.orchestrator_bridge import OrchestratorBridge
from hrm.coinbase_pipeline import CoinbasePipeline

class SteadyTrainer:
    """
    Steady HRM Trainer:
    Trains Hierarchical Reasoning Model on Stratified bags (Winners, Losers, Counter).
    Optimized for MLX (Apple Silicon) and Arrow storage.
    """
    def __init__(self, model_cfg: SignalHRMConfig = None):
        self.cfg = model_cfg or SignalHRMConfig()
        self.model = SignalHRM(self.cfg)
        self.optimizer = optim.Adam(learning_rate=1e-4) if HAS_MLX else None
        self.bridge = OrchestratorBridge()
        self.pipeline = CoinbasePipeline()
        self.checkpoint_path = "hrm/checkpoints/steady_hrm.npz"
        os.makedirs("hrm/checkpoints", exist_ok=True)

    def train_step(self, x: np.ndarray, returns: np.ndarray):
        """Single training update using MLX"""
        if not HAS_MLX:
            return 0.0, 0.5 # Dummy loss, convergence
            
        x_mx = mx.array(x)
        y_mx = mx.array(returns)

        def loss_fn(model, x, y):
            weights, alpha, convergence, _mem = model(x)
            # We minimize negative PnL/convergence-gated return
            return portfolio_loss(weights, alpha, convergence, y)

        loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)
        loss, grads = loss_and_grad_fn(self.model, x_mx, y_mx)
        
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state)
        
        # Get metrics for logging
        _, _, convergence, _mem = self.model(x_mx)
        return float(loss), float(mx.mean(convergence))

    def run_training(self, n_rounds: int = 100):
        print(f"Starting Steady HRM Training (MLX: {HAS_MLX})")
        print(f"Stratified Bag Composition: 10 Winners, 10 Losers, 10 Countercoins")
        
        for r in range(n_rounds):
            print(f"\n--- Round {r+1}/{n_rounds} ---")
            
            # 1. Sample Stratified Bag
            # Note: We need bags with concurrent timestamps to compute returns correctly
            # Simplified for now: pull a bag and iterate through it
            bags = self.pipeline.sample_stratified_bag(n_samples=1, lookback_days=7)
            if not bags:
                print("Wait: No data in bag.")
                continue
            
            df_bag = bags[0]
            symbols = df_bag['symbol'].unique()
            
            # 2. Process Tensors
            # For each symbol, we need a batch of tensors at different timestamps
            # OR we treat symbols as the 'batch' if aligned.
            
            batch_x = []
            batch_y = []
            
            for sym in symbols:
                # Compute tensor (includes context features dim=40)
                tensor = self.bridge.compute_tensor(sym, seq_len=self.cfg.seq_len)
                if tensor is None:
                    continue
                
                # Get target return (next 5m close change)
                # In real training we'd use a window, here we take the last known return
                sym_data = df_bag[df_bag['symbol'] == sym]
                if len(sym_data) > 1:
                    last_ret = (sym_data['close'].iloc[-1] / sym_data['close'].iloc[-2]) - 1.0
                    batch_x.append(tensor[0])
                    batch_y.append(last_ret)
            
            if not batch_x:
                print("No valid tensors in bag.")
                continue

            # 3. Optimize
            # X: [B, seq_len, 40]
            X = np.stack(batch_x).astype(np.float32)
            Y = np.array(batch_y).astype(np.float32)
            
            loss, conv = self.train_step(X, Y)
            print(f"Loss: {loss:.6f} | Convergence: {conv:.4f}")
            
            # 4. Checkpoint
            if (r + 1) % 10 == 0:
                self.save_model()

    def save_model(self):
        if HAS_MLX:
            mx.savez(self.checkpoint_path, **dict(self.model.parameters()))
            print(f"Saved checkpoint to {self.checkpoint_path}")

    def load_model(self):
        if HAS_MLX and os.path.exists(self.checkpoint_path):
            self.model.update(mx.load(self.checkpoint_path))
            print(f"Loaded checkpoint from {self.checkpoint_path}")

if __name__ == "__main__":
    trainer = SteadyTrainer()
    # If resuming
    # trainer.load_model()
    trainer.run_training(n_rounds=50)
