
"""
The Nexus
=========

Central Nervous System of the HRM Architecture.
Coordinates Memory, Senses, Cognition, and Action.
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import defaultdict

from hrm.interfaces import (
    INexus,
    IMemoryStore,
    ISignalGenerator,
    ICognitiveModel,
    IRoutingEngine,
    IExecutionBackend
)

# Implementation of Interfaces for the Continuous Ranking Loop

class PandasMemoryAdapter(IMemoryStore):
    """Adapts a simple DataFrame store or ArrowStore to IMemoryStore."""
    def __init__(self, arrow_store):
        self.store = arrow_store
    
    def load(self, symbol: str, start: Optional[datetime] = None, end: Optional[datetime] = None) -> pd.DataFrame:
        return self.store.load(symbol, start, end)

    def list_symbols(self) -> List[str]:
        # Using a fixed list for now as per ranking script
        return ["BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK", "UNI", "ADA"]

class CanonicalSignalGenerator(ISignalGenerator):
    """
    Implements the 16 Canonical Signals.
    Refactored from continuous_ranking.py to satisfy ISignalGenerator.
    """
    def __init__(self):
        from hrm.continuous_ranking import compute_signals_16
        self._compute_func = compute_signals_16
        self._signal_names = [
            "macd_crossover", "sota_momentum", "momentum_trend", "mom_trend_additive",
            "rsi_mean_reversion", "bollinger_reversion", "grid_reversion", "hrm_mean_reversion",
            "volatility_breakout", "vol_x_breakout_proven", "momentum_x_vol",
            "bent_penny", "pairs_spread", "dca_baseline", "technical_ml", "rsi_x_trend"
        ]

    @property
    def signal_names(self) -> List[str]:
        return self._signal_names + ["neural_hrm"]

    def compute_signals(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        if df.empty: return {}
        # 1. Base Signals
        signals = self._compute_func(df)
        
        # 2. Neural HRM Proxy (Vectorized)
        # We mimic the CPU-mode SignalHRM logic:
        # - Pool (mean) of last 32 bars
        # - Random weights (deterministic)
        # - Softmax -> Combined Alpha
        try:
            # Stack [T, 16]
            base_keys = [k for k in self._signal_names if k in signals]
            if base_keys:
                stacked = np.stack([signals[k] for k in base_keys], axis=1) # (T, 16)
                
                # Confidence proxy (abs)
                confs = np.abs(stacked)
                
                # Interleave for "Input" [Signal, Conf, Signal, Conf...]
                # Shape (T, 32)
                T, N = stacked.shape
                x = np.zeros((T, N * 2))
                x[:, 0::2] = stacked
                x[:, 1::2] = confs
                
                # Rolling Mean (Seq Len 32)
                # We use pandas for speed on the rolling window
                pooled = pd.DataFrame(x).rolling(32, min_periods=1).mean().values
                
                # Weights (Linear Layer)
                # Deterministic random weights to simulate untrained net
                rng = np.random.default_rng(42)
                # We want to weight the *Features* (32) to get *Signal Weights* (16)?
                # SignalHRM CPU logic is weird, let's just do a direct projection:
                # Features (32) -> Logits (16)
                W = rng.standard_normal((N * 2, N)) 
                logits = pooled @ W # (T, 16)
                
                # Softmax
                # exp(x) / sum(exp(x))
                # Stability fix: subtract max
                logits_safe = logits - np.max(logits, axis=1, keepdims=True)
                exp_logits = np.exp(logits_safe)
                weights = exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + 1e-9)
                
                # Alpha = sum(weights * signals)
                alpha = (weights * stacked).sum(axis=1)
                
                signals["neural_hrm"] = alpha
        except Exception as e:
            # Fallback if something explodes (e.g. shape mismatch)
            # print(f"Neural HRM failed: {e}")
            pass
            
        return signals

class RankingBrain(ICognitiveModel):
    """
    A 'Meta-Brain' that doesn't just predict, but evaluates 
    all signals against future outcomes (Hindsight Training).
    """
    def predict(self, tensor: np.ndarray, context: Optional[np.ndarray] = None) -> np.ndarray:
        # In ranking mode, we don't predict. We judge.
        return np.zeros(1)

class SimulationBackend(IExecutionBackend):
    """No-op backend for ranking."""
    def execute_order(self, symbol: str, side: str, amount: float, price: Optional[float] = None) -> Dict[str, Any]:
        return {"status": "simulated"}

    def get_balance(self, asset: str) -> float:
        return 10000.0

@dataclass
class NexusContext:
    """State of the Nexus for a single pulse."""
    timestamp: datetime
    active_symbols: List[str]
    market_data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    signals: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    # Ranking specific
    ranking_stats: Dict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: {'count': 0, 'sum_ret': 0.0, 'sum_sharpe': 0.0}))

class Nexus(INexus):
    def __init__(
        self,
        memory: IMemoryStore,
        senses: ISignalGenerator,
        brain: ICognitiveModel,
        router: IRoutingEngine,
        executor: IExecutionBackend,
    ):
        self.memory = memory
        self.senses = senses
        self.brain = brain
        self.router = router
        self.executor = executor
        
        self.logger = logging.getLogger("Nexus")
        self.logger.setLevel(logging.INFO)
        # Ranking stats storage
        self.stats = defaultdict(lambda: {'count': 0, 'sum_ret': 0.0, 'sum_sharpe': 0.0})
        self.total_bags = 0

    def pulse(self, bag_df: pd.DataFrame = None) -> NexusContext:
        """
        Execute one atomic cycle.
        In simulation mode, 'pulse' processes one historical bag.
        """
        ctx = NexusContext(
            timestamp=datetime.utcnow(),
            active_symbols=[] # Driven by bag
        )
        
        if bag_df is None:
            return ctx

        # 1. Perception
        # Handle single vs multi-symbol bags
        if 'symbol' in bag_df.columns:
            # If multiple symbols exist, split them
            # Check if all one symbol or multiple
            unique_syms = bag_df['symbol'].unique()
            print(f"📊 Bag Content ({len(unique_syms)}/64): {', '.join(sorted(unique_syms))}")

            for sym in unique_syms:
                sub_df = bag_df[bag_df['symbol'] == sym].copy()
                ctx.active_symbols.append(sym)
                ctx.market_data[sym] = sub_df
                
                # Senses: Compute signals
                try:
                    signals = self.senses.compute_signals(sub_df)
                    ctx.signals[sym] = signals

                    # 2. Cognition (Ranking Logic)
                    # Calculate returns for the bag
                    prices = sub_df['close'].values
                    
                    # Vectorized backtest (inline for performance in this loop)
                    from hrm.continuous_ranking import backtest_signal_vectorized
                    
                    for name, sig in signals.items():
                        # returns (final_equity_mult, sharpe)
                        mult, sharpe = backtest_signal_vectorized(sig, prices)
                        final_balance = 100.0 * mult
                        
                        self.stats[name]['count'] += 1
                        self.stats[name]['sum_ret'] += final_balance # Storing Balance Sum now
                        self.stats[name]['sum_sharpe'] += sharpe
                        
                except Exception as e:
                    self.logger.error(f"Pulse failed for {sym}: {e}")
            
            self.total_bags += 1

            
        return ctx

    def print_leaderboard(self):
        print("\n" + "="*80)
        print(f" NEXUS LEADERBOARD (Bags: {self.total_bags}) | {datetime.now().strftime('%H:%M:%S')}")
        print("="*80)
        print(f"{'Algorithm':<25} | {'Sharpe':>8} | {'Avg Bal ($)':>12} | {'Count':>5}")
        print("-" * 60)
        
        ranked = []
        for alg, s in self.stats.items():
            if s['count'] > 0:
                avg_sharpe = s['sum_sharpe'] / s['count']
                avg_bal = s['sum_ret'] / s['count']
                ranked.append((alg, avg_sharpe, avg_bal, s['count']))
        
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        for alg, sh, bal, cnt in ranked:
            print(f"{alg:<25} | {sh:8.2f} | ${bal:11.2f} | {cnt:5d}")
        print("="*80 + "\n")

    def run_simulation_loop(self, pipeline):
        """Dedicated loop for continuous ranking simulation."""
        print("Starting Nexus Simulation Loop...")
        try:
            while True:
                # 0. Environment Step (Stochastic Sampling)
                bags = pipeline.sample_training_bag(n_samples=1, min_len=1000)
                
                if not bags:
                    time.sleep(1)
                    continue

                for bag in bags:
                    # 1. Pulse
                    self.pulse(bag)
                    # Print every bag for immediate feedback
                    self.print_leaderboard()

                # if self.total_bags % 10 == 0:
                #    self.print_leaderboard()
                    
        except KeyboardInterrupt:
            print("\nNexus Simulation Stopped.")
            self.print_leaderboard()

