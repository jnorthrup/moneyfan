import os
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from datetime import datetime
import sys
sys.path.append(os.getcwd())

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from hrm.signal_hrm import SignalHRM, SignalHRMConfig, portfolio_loss
from hrm.orchestrator_bridge import OrchestratorBridge
from hrm.coinbase_pipeline import CoinbasePipeline

class RankedTrader:
    """Individual HRM trader with its own weight state and optimizer"""
    def __init__(self, trader_id: int, cfg: SignalHRMConfig, mode: str = "B"):
        self.trader_id = trader_id
        self.cfg = cfg
        self.mode = mode # "A" (CPU/NumPy) or "B" (MLX)
        self.model = SignalHRM(cfg, force_cpu=(mode == "A"))
        self.optimizer = optim.Adam(learning_rate=1e-4) if (HAS_MLX and mode == "B") else None
        
        # Performance Tracking
        self.cumulative_reward = 0.0
        self.balance = 10000.0
        self.position = 0.0 # in units
        
        # Persistent memory (z_H, z_L for MLX; EMA vector for CPU)
        self.memory = None
        
        # Limits & Triggers State
        self.active_order = None # {type: 'LIMIT', side: 'BUY', price: 100, size: 1.0}
        self.stop_loss = None
        self.take_profit = None
        
    def update(self, x: mx.array | np.ndarray, y_return: mx.array | np.ndarray, current_price: float):
        """Training-at-test-time update step + Heuristic execution"""
        
        # 1. Test-time adjustment (If Model B)
        if self.mode == "B" and HAS_MLX:
            x_mx = x if isinstance(x, mx.array) else mx.array(x)
            ret_mx = y_return if isinstance(y_return, mx.array) else mx.array(y_return)
            mem = self.memory  # pass prior memory into TTL
            
            def loss_fn(model, x, y):
                weights, alpha, convergence, _mem = model(x, memory=mem)
                return portfolio_loss(weights, alpha, convergence, y)
                
            loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)
            loss, grads = loss_and_grad_fn(self.model, x_mx, ret_mx)
            self.optimizer.update(self.model, grads)
            mx.eval(self.model.parameters(), self.optimizer.state)
        
        # 2. Extract Alpha & Convergence (with memory carry-forward)
        weights, alpha, conv, self.memory = self.model(x, memory=self.memory)
        alpha_val = float(alpha) if not hasattr(alpha, "item") else alpha.item()
        conv_val = float(conv) if not hasattr(conv, "item") else conv.item()
        
        # 3. Decision Heuristics
        # Reward: alpha × return (ungated — differentiates agents even pre-convergence)
        reward = alpha_val * float(y_return)
        self.cumulative_reward += reward
        
        # Trade when alpha signal is nonzero (low convergence gate for early learning)
        if conv_val > 0.1:
            if alpha_val > 0.05 and self.position == 0:
                # Buy: limit at 0.5% below current
                self.active_order = {'type': 'LIMIT', 'side': 'BUY', 'price': current_price * 0.995, 'size': self.balance / current_price}
            elif alpha_val < -0.05 and self.position > 0:
                # Sell: market sell
                self.balance += self.position * current_price * 0.995  # 0.5% fee proxy
                self.position = 0
                self.active_order = None

        return alpha_val, conv_val


    def process_candle(self, low: float, high: float, close: float):
        """Check if limits or triggers were hit"""
        if self.active_order:
            order = self.active_order
            if order['side'] == 'BUY' and low <= order['price']:
                # Filled
                self.position = order['size']
                self.balance -= order['size'] * order['price']
                self.active_order = None
                self.stop_loss = order['price'] * 0.95 # 5% stop
        
        # Check Stop Loss
        if self.position > 0 and self.stop_loss and low <= self.stop_loss:
            self.balance += self.position * self.stop_loss
            self.position = 0
            self.stop_loss = None

from hrm.binance_adapter import BinanceAdapterPipeline

class RankingHarness:
    """Tournament for 25 ranked traders comparing CPU (A) vs MLX (B)"""
    def __init__(self, n_traders: int = 25, seq_len: int = 32):
        self.n_traders = n_traders
        self.cfg = SignalHRMConfig(seq_len=seq_len, hidden_dim=32) 
        
        # Split traders between Model A (CPU) and Model B (MLX)
        self.traders = []
        for i in range(n_traders):
            mode = "A" if i < 12 else "B" # Roughly half and half
            self.traders.append(RankedTrader(i, self.cfg, mode=mode))
            
        self.bridge = OrchestratorBridge()
        self.pipeline = BinanceAdapterPipeline()
        
    def run_tournament(self, n_rounds: int = 10):
        print(f"Tournament Start: {self.n_traders} Traders | Seq Len: {self.cfg.seq_len}")
        
        for r in range(n_rounds):
            # Sample stratified bag (identical for all traders this round)
            # BinanceAdapterPipeline uses sample_training_bag or we can use the engine directly
            # sample_stratified_bag is a stub in BinanceAdapter.
            # We should use sample_training_bag for now
            bags = self.pipeline.sample_training_bag(n_samples=1)
            if not bags: continue
            
            df = bags[0]
            symbols = df['symbol'].unique()
            
            # Shared Data: Compute tensors once per symbol
            symbol_tensors = {}
            symbol_returns = {}
            
            for sym in symbols:
                tensor = self.bridge.compute_tensor(sym, seq_len=self.cfg.seq_len)
                if tensor is not None:
                    # In this tournament, each trader sees the same window
                    symbol_tensors[sym] = mx.array(tensor)
                    
                    # Target return
                    sym_data = df[df['symbol'] == sym]
                    if len(sym_data) > 1:
                        symbol_returns[sym] = mx.array([(sym_data['close'].iloc[-1] / sym_data['close'].iloc[-2]) - 1.0])

            if not symbol_tensors: 
                print("Wait: No symbol tensors computed (history too short or signal error).")
                continue

            print(f"Update: Processed {len(symbol_tensors)} symbols for {self.n_traders} traders.")
            for sym in symbol_tensors:
                if sym in symbol_returns:
                    # Get price info for heuristics
                    sym_data = df[df['symbol'] == sym]
                    low = sym_data['low'].iloc[-1]
                    high = sym_data['high'].iloc[-1]
                    close = sym_data['close'].iloc[-1]
                    
                    for trader in self.traders:
                        trader.process_candle(low, high, close)
                        trader.update(symbol_tensors[sym], symbol_returns[sym], close)

            self.report_rankings()

    def report_rankings(self):
        # Rank by current cumulative reward
        sorted_traders = sorted(self.traders, key=lambda t: t.cumulative_reward, reverse=True)
        
        print("\n" + "="*70)
        print(f"TRADER RANKINGS (By Reward-From-Alpha) - A=CPU, B=MLX")
        print("="*70)
        for i, t in enumerate(sorted_traders[:10]): # Top 10
            print(f"{i+1:2d}. Trader {t.trader_id:2d} ({t.mode}) | Reward: {t.cumulative_reward:+.6f} | Wealth: ${t.balance + t.position*100:,.2f}")
        print("...")
        print("="*70)

if __name__ == "__main__":
    harness = RankingHarness(n_traders=25)
    harness.run_tournament(n_rounds=1)
