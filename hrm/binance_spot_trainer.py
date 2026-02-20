#!/usr/bin/env python3
"""
Binance Spot Trainer - Proxies Binance historical data for Coinbase model training

Architecture:
  Binance Spot Data (1m candles) → ArrowStore → Signal Orchestrator → HRM Codec Training

Key Features:
1. Spot-only pair filtering (no futures, leveraged tokens, stablecoin-stablecoin)
2. Binance→Coinbase symbol mapping for cross-exchange transfer learning
3. Hierarchical signal learning: HRM learns which signal groups are most predictive
4. 64-pair synchronized megabags with 2-month windows

Signal Hierarchy (6 layers):
  L0: Raw OHLCV features
  L1: Trend signals (MACD, momentum, trend)
  L2: Mean Reversion signals (RSI, Bollinger, Grid)
  L3: Volatility signals (breakout, vol_x_momentum)
  L4: Stat Arb signals (pairs spread, bent penny)
  L5: Meta signals (composite, ML ensemble)

HRM learns weights over these layers to identify which regimes are predictive.
"""

import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from hrm.arrow_store import ArrowStore
    from hrm.signal_hrm import SignalHRM, SignalHRMConfig, SIGNAL_16, SIGNAL_REGIMES, N_SIGNALS, portfolio_loss
    from hrm.orchestrator_bridge import OrchestratorBridge
except ImportError:
    from arrow_store import ArrowStore
    from signal_hrm import SignalHRM, SignalHRMConfig, SIGNAL_16, SIGNAL_REGIMES, N_SIGNALS, portfolio_loss
    from orchestrator_bridge import OrchestratorBridge

BINANCE_SPOT_PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "SOLUSDT",
    "DOTUSDT", "MATICUSDT", "LTCUSDT", "AVAXUSDT", "LINKUSDT", "ATOMUSDT", "UNIUSDT",
    "ETCUSDT", "XLMUSDT", "ALGOUSDT", "VETUSDT", "FILUSDT", "ICPUSDT", "THETAUSDT",
    "AAVEUSDT", "NEARUSDT", "AXSUSDT", "FTMUSDT", "SANDUSDT", "MANAUSDT", "GALAUSDT",
    "ENJUSDT", "COMPUSDT", "LRCUSDT", "1INCHUSDT", "SUSHIUSDT", "YFIUSDT", "SNXUSDT",
    "MKRUSDT", "BATUSDT", "CRVUSDT", "ZILUSDT", "DASHUSDT", "ZECUSDT", "XMRUSDT",
    "EOSUSDT", "XTZUSDT", "KAVAUSDT", "RUNEUSDT", "CAKEUSDT", "DYDXUSDT", "APEUSDT",
    "ARBUSDT", "OPUSDT", "INJUSDT", "HBARUSDT", "QNTUSDT", "ENSUSDT", "LDOUSDT",
    "APTUSDT", "BLURUSDT", "CFXUSDT", "GMXUSDT", "AGIXUSDT", "RNDRUSDT", "WOOUSDT",
]

COINBASE_EQUIV = {
    "BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD", "SOLUSDT": "SOL-USD",
    "ADAUSDT": "ADA-USD", "DOGEUSDT": "DOGE-USD", "XRPUSDT": "XRP-USD",
    "DOTUSDT": "DOT-USD", "MATICUSDT": "MATIC-USD", "LTCUSDT": "LTC-USD",
    "AVAXUSDT": "AVAX-USD", "LINKUSDT": "LINK-USD", "ATOMUSDT": "ATOM-USD",
    "UNIUSDT": "UNI-USD", "ETCUSDT": "ETC-USD", "XLMUSDT": "XLM-USD",
    "ALGOUSDT": "ALGO-USD", "FILUSDT": "FIL-USD", "AAVEUSDT": "AAVE-USD",
    "NEARUSDT": "NEAR-USD", "APEUSDT": "APE-USD", "ARBUSDT": "ARB-USD",
    "OPUSDT": "OP-USD", "INJUSDT": "INJ-USD", "HBARUSDT": "HBAR-USD",
}

SIGNAL_LAYERS = {
    "L0_raw": [],
    "L1_trend": ["macd_crossover", "sota_momentum", "momentum_trend", "mom_trend_additive"],
    "L2_mean_reversion": ["rsi_mean_reversion", "bollinger_reversion", "grid_reversion", "hrm_mean_reversion"],
    "L3_volatility": ["volatility_breakout", "vol_x_breakout_proven", "momentum_x_vol"],
    "L4_stat_arb": ["bent_penny", "pairs_spread"],
    "L5_meta": ["dca_baseline", "technical_ml", "rsi_x_trend"],
}

LAYER_BY_SIGNAL = {}
for layer, signals in SIGNAL_LAYERS.items():
    for s in signals:
        LAYER_BY_SIGNAL[s] = layer


@dataclass
class BinanceSpotConfig:
    n_pairs: int = 64
    lookback_days: int = 60
    seq_len: int = 32
    target_window: int = 5
    min_data_rows: int = 50000
    hidden_dim: int = 64
    learning_rate: float = 5e-4
    batch_size: int = 16
    n_epochs: int = 10
    optimizer: str = "muon"  # "muon" or "lion"


@dataclass
class HierarchicalWeights:
    layer_weights: Dict[str, float] = field(default_factory=lambda: {k: 1.0/6 for k in SIGNAL_LAYERS.keys()})
    signal_weights: np.ndarray = field(default_factory=lambda: np.ones(N_SIGNALS) / N_SIGNALS)
    confidence_threshold: float = 0.25
    
    def get_signal_weight(self, signal_name: str) -> float:
        layer = LAYER_BY_SIGNAL.get(signal_name, "L5_meta")
        layer_w = self.layer_weights.get(layer, 1.0/6)
        signal_idx = SIGNAL_16.index(signal_name) if signal_name in SIGNAL_16 else 0
        signal_w = self.signal_weights[signal_idx] if signal_idx < len(self.signal_weights) else 1.0/N_SIGNALS
        return layer_w * signal_w


def binance_to_coinbase(symbol: str) -> str:
    """Map Binance pair to Coinbase format."""
    if symbol in COINBASE_EQUIV:
        return COINBASE_EQUIV[symbol]
    if symbol.endswith("USDT"):
        base = symbol[:-4]
        return f"{base}-USD"
    return symbol


def is_valid_spot_pair(symbol: str) -> bool:
    """Filter for real spot pairs only."""
    if not symbol.endswith("USDT"):
        return False
    if "UP" in symbol or "DOWN" in symbol:
        return False
    if "BULL" in symbol or "BEAR" in symbol:
        return False
    base = symbol[:-4]
    if base in ["BUSD", "USDC", "DAI", "TUSD", "USDP"]:
        return False
    return True


class BinanceSpotDataLoader:
    """Loads synchronized Binance spot data from ArrowStore."""
    
    def __init__(self, arrow_dir: str = "hrm/data/arrow"):
        self.store = ArrowStore(arrow_dir)
        self.bridge = OrchestratorBridge(arrow_dir)
        
    def get_available_pairs(self) -> List[str]:
        """Get list of pairs with data in ArrowStore."""
        pairs = []
        if os.path.exists(self.store.base_dir):
            for f in os.listdir(self.store.base_dir):
                if f.endswith(".feather"):
                    # File format: ADA_USDT.feather -> ADAUSDT
                    symbol = f.replace(".feather", "").replace("_", "")
                    if is_valid_spot_pair(symbol):
                        pairs.append(symbol)
        return pairs
    
    def load_pair(self, symbol: str, start: datetime = None, end: datetime = None,
                  min_rows: int = 1000) -> Optional[pd.DataFrame]:
        """Load data for a single pair."""
        # Try both formats: ADAUSDT and ADA_USDT
        df = self.store.load(symbol, start, end)
        if df.empty:
            # Try with underscore
            alt_symbol = symbol[:-4] + "_" + symbol[-4:]
            df = self.store.load(alt_symbol, start, end)
        
        if df.empty or len(df) < min_rows:
            return None
        df = df.copy()
        df['symbol'] = symbol
        df['coinbase_symbol'] = binance_to_coinbase(symbol)
        return df
    
    def sample_megabag(self, n_pairs: int = 64, lookback_days: int = 60,
                       min_rows_per_pair: int = 5000) -> Tuple[pd.DataFrame, List[str]]:
        """Sample synchronized megabag across multiple pairs."""
        available = self.get_available_pairs()
        selected = available[:n_pairs]
        
        end = datetime.utcnow()
        start = end - timedelta(days=lookback_days)
        
        frames = []
        loaded_symbols = []
        
        for sym in selected:
            df = self.load_pair(sym, start, end, min_rows=min_rows_per_pair)
            if df is not None:
                frames.append(df)
                loaded_symbols.append(sym)
        
        if not frames:
            return pd.DataFrame(), []
        
        megabag = pd.concat(frames, ignore_index=True)
        return megabag, loaded_symbols


class HierarchicalSignalRanker:
    """Ranks signal layers by predictive power."""
    
    def __init__(self):
        self.layer_scores = {k: [] for k in SIGNAL_LAYERS.keys()}
        self.signal_scores = {s: [] for s in SIGNAL_16}
        
    def add_sample(self, signals: Dict[str, float], realized_return: float):
        """Add a sample for scoring."""
        for signal_name, signal_val in signals.items():
            if signal_name not in SIGNAL_16:
                continue
            prediction = np.sign(signal_val)
            if prediction != 0:
                accuracy = 1.0 if prediction * realized_return > 0 else 0.0
                self.signal_scores[signal_name].append(accuracy)
                layer = LAYER_BY_SIGNAL.get(signal_name, "L5_meta")
                self.layer_scores[layer].append(accuracy)
    
    def get_layer_weights(self) -> Dict[str, float]:
        """Get normalized layer weights."""
        weights = {}
        total = 0
        for layer, scores in self.layer_scores.items():
            if scores:
                w = np.mean(scores)
                weights[layer] = w
                total += w
            else:
                weights[layer] = 1.0 / 6
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        return weights
    
    def get_signal_weights(self) -> np.ndarray:
        """Get normalized signal weights."""
        weights = np.zeros(N_SIGNALS)
        for i, sig in enumerate(SIGNAL_16):
            scores = self.signal_scores.get(sig, [])
            weights[i] = np.mean(scores) if scores else 1.0 / N_SIGNALS
        weights = weights / weights.sum()
        return weights
    
    def get_hierarchical_weights(self) -> HierarchicalWeights:
        """Get combined hierarchical weights."""
        return HierarchicalWeights(
            layer_weights=self.get_layer_weights(),
            signal_weights=self.get_signal_weights(),
        )


class BinanceSpotCodecTrainer:
    """
    Trains HRM codec on Binance spot data for Coinbase deployment.
    
    Learns:
    1. Which signal layers (Trend, Reversion, Volatility, etc.) are predictive
    2. Which individual signals within layers are predictive
    3. How to combine signals hierarchically for maximum alpha
    """
    
    def __init__(self, config: BinanceSpotConfig = None):
        self.config = config or BinanceSpotConfig()
        self.loader = BinanceSpotDataLoader()
        self.ranker = HierarchicalSignalRanker()
        
        self.cfg = SignalHRMConfig(
            seq_len=self.config.seq_len,
            hidden_dim=self.config.hidden_dim,
        )
        self.model = SignalHRM(self.cfg)
        
        if HAS_MLX:
            # Select SOTA optimizer based on config
            if self.config.optimizer == "lion":
                # LION: Evolved Sign Momentum (Chen et al., 2023)
                # - Only 2 momentum terms (vs Adam's 4)
                # - Better memory efficiency
                # - Sign-based updates
                self.optimizer = optim.Lion(
                    learning_rate=self.config.learning_rate,
                    betas=(0.9, 0.99),
                )
                print(f"[BinanceSpotTrainer] Using LION optimizer (lr={self.config.learning_rate})")
            else:  # default to MUON
                # MUON: Momentum-based Orthogonalization Update
                # - Orthogonalizes gradients for better convergence
                # - More stable than Adam for large models
                # - Particularly effective for attention-based architectures
                self.optimizer = optim.Muon(
                    learning_rate=self.config.learning_rate,
                    betas=(0.95, 0.999),
                )
                print(f"[BinanceSpotTrainer] Using MUON optimizer (lr={self.config.learning_rate})")
        else:
            self.optimizer = None
        
        self.history = []
        self.layer_history = {k: [] for k in SIGNAL_LAYERS.keys()}
        
    def compute_signals_for_window(self, df: pd.DataFrame, symbol: str) -> Optional[Dict[str, float]]:
        """Compute all 16 signals for a data window."""
        try:
            tensor = self.loader.bridge.compute_tensor(symbol, df_input=df)
            if tensor is None:
                return None
            signals = {}
            for i, sig_name in enumerate(SIGNAL_16):
                signals[sig_name] = float(tensor[0, -1, i * 2])
            return signals
        except Exception:
            return None
    
    def train_epoch(self, n_bags: int = 5, verbose: bool = True):
        """Train for one epoch on Binance spot data."""
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Binance Spot Codec Training")
            print(f"  Target: {self.config.target_window}-bar forward return")
            print(f"{'='*60}")
        
        for bag_idx in range(n_bags):
            if verbose:
                print(f"\n[Bag {bag_idx+1}/{n_bags}] Sampling megabag...")
            
            megabag, symbols = self.loader.sample_megabag(
                n_pairs=self.config.n_pairs,
                lookback_days=self.config.lookback_days,
            )
            
            if megabag.empty or not symbols:
                if verbose:
                    print("  No data available, skipping...")
                continue
            
            if verbose:
                print(f"  Loaded {len(symbols)} pairs, {len(megabag)} rows")
            
            for sym in symbols[:10]:
                sym_df = megabag[megabag['symbol'] == sym].copy()
                if len(sym_df) < self.config.seq_len + self.config.target_window + 20:
                    continue
                
                # Handle time column - could be 'time' or index
                if 'time' in sym_df.columns:
                    sym_df = sym_df.sort_values('time')
                elif sym_df.index.name is not None:
                    sym_df = sym_df.sort_index()
                
                close = sym_df['close']
                realized_ret = close.pct_change(self.config.target_window).shift(-self.config.target_window)
                
                valid_idx = realized_ret.dropna().index
                if len(valid_idx) < self.config.seq_len + 2:
                    continue
                
                n_windows = min(5, len(valid_idx) - self.config.seq_len)
                matches = 0
                
                for _ in range(n_windows):
                    idx = np.random.randint(self.config.seq_len + 20, len(valid_idx) - 1)
                    t_end = valid_idx[idx]
                    target_val = realized_ret.loc[t_end]
                    
                    sub_df = sym_df.loc[:t_end]
                    signals = self.compute_signals_for_window(sub_df, sym)
                    
                    if signals is None:
                        continue
                    
                    self.ranker.add_sample(signals, target_val)
                    matches += 1
                    
                    if HAS_MLX:
                        tensor = self.loader.bridge.compute_tensor(sym, df_input=sub_df)
                        if tensor is not None:
                            x = mx.array(tensor)
                            y = mx.array([target_val])
                            
                            def loss_fn(model, x, y):
                                weights, alpha, convergence, _mem = model(x)
                                return portfolio_loss(weights, alpha, convergence, y)
                            
                            loss_and_grad = nn.value_and_grad(self.model, loss_fn)
                            loss, grads = loss_and_grad(self.model, x, y)
                            self.optimizer.update(self.model, grads)
                            mx.eval(self.model.parameters(), self.optimizer.state)
                            
                            self.history.append({
                                'symbol': sym,
                                'coinbase_symbol': binance_to_coinbase(sym),
                                'loss': float(loss),
                                'realized': target_val,
                                'timestamp': datetime.utcnow().isoformat(),
                            })
                
                if verbose and matches > 0:
                    print(f"    {sym} → {binance_to_coinbase(sym)}: {matches} windows")
        
        self._update_layer_scores()
        
        if verbose:
            self.report()
    
    def _update_layer_scores(self):
        """Update layer performance tracking."""
        for layer in SIGNAL_LAYERS.keys():
            signal_names = SIGNAL_LAYERS[layer]
            for sig in signal_names:
                scores = self.ranker.signal_scores.get(sig, [])
                if scores:
                    self.layer_history[layer].append(np.mean(scores))
    
    def report(self):
        """Print training report with hierarchical weights."""
        print(f"\n{'='*70}")
        print(f"  HIERARCHICAL SIGNAL ANALYSIS")
        print(f"{'='*70}")
        
        hw = self.ranker.get_hierarchical_weights()
        
        print(f"\n  Layer Weights (predictive power):")
        layer_order = ["L1_trend", "L2_mean_reversion", "L3_volatility", "L4_stat_arb", "L5_meta"]
        for layer in layer_order:
            w = hw.layer_weights.get(layer, 0)
            bar = "█" * int(w * 40)
            print(f"    {layer:<20} {w:>6.1%} {bar}")
        
        print(f"\n  Top 5 Signals by Layer:")
        signal_weights = [(s, hw.signal_weights[SIGNAL_16.index(s)]) for s in SIGNAL_16]
        signal_weights.sort(key=lambda x: -x[1])
        for sig, w in signal_weights[:5]:
            layer = LAYER_BY_SIGNAL.get(sig, "?")
            print(f"    {sig:<24} {w:>6.1%} ({layer})")
        
        if self.history:
            df = pd.DataFrame(self.history)
            corr = df['loss'].corr(df['realized'].abs()) if len(df) > 10 else 0
            print(f"\n  Training Stats:")
            print(f"    Samples: {len(df)}")
            print(f"    Avg Loss: {df['loss'].mean():.6f}")
            print(f"    Loss-Realized Corr: {corr:.4f}")
        
        print(f"{'='*70}")
    
    def get_transferable_weights(self) -> Dict:
        """Get weights for transfer to Coinbase models."""
        hw = self.ranker.get_hierarchical_weights()
        return {
            'layer_weights': hw.layer_weights,
            'signal_weights': hw.signal_weights.tolist(),
            'signal_names': SIGNAL_16,
            'mapping': {s: binance_to_coinbase(s) for s in BINANCE_SPOT_PAIRS[:30]},
        }
    
    def save_checkpoint(self, path: str):
        """Save training state."""
        import json
        state = {
            'config': self.config.__dict__,
            'hierarchical_weights': self.get_transferable_weights(),
            'history': self.history[-1000:],
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        print(f"Saved checkpoint to {path}")


def main():
    print("=" * 60)
    print("  Binance Spot → Coinbase Transfer Learning")
    print("  HRM Hierarchical Signal Training")
    print("=" * 60)
    
    config = BinanceSpotConfig(
        n_pairs=32,
        lookback_days=30,
        seq_len=32,
        target_window=5,
    )
    
    trainer = BinanceSpotCodecTrainer(config)
    
    # Multiple epochs for convergence
    n_epochs = 10
    convergence_log = []
    
    for epoch in range(1, n_epochs + 1):
        print(f"\n{'#'*70}")
        print(f"#  EPOCH {epoch}/{n_epochs}")
        print(f"{'#'*70}")
        
        trainer.train_epoch(n_bags=3, verbose=True)
        
        # Track convergence
        hw = trainer.ranker.get_hierarchical_weights()
        convergence_log.append({
            'epoch': epoch,
            'layer_weights': hw.layer_weights.copy(),
            'signal_weights': hw.signal_weights.copy(),
            'samples': len(trainer.history),
            'avg_loss': np.mean([h['loss'] for h in trainer.history[-30:]]) if trainer.history else 0,
        })
        
        # Save checkpoint after each epoch
        trainer.save_checkpoint(f"hrm/data/binance_spot_weights_epoch{epoch}.json")
        
        print(f"\nEpoch {epoch} completed.")
        
        if epoch < n_epochs:
            print("\nContinuing to next epoch...")
            time.sleep(1)
    
    # Final weights
    weights = trainer.get_transferable_weights()
    print(f"\n{'='*70}")
    print("  FINAL CONVERGENCE REPORT")
    print(f"{'='*70}")
    
    # Print convergence table
    print(f"\n{'Epoch':>5} | {'Samples':>7} | {'Loss':>8} | L1_trend | L2_rev | L3_vol | L4_arb | L5_meta")
    print("-" * 80)
    for log in convergence_log:
        lw = log['layer_weights']
        print(f"{log['epoch']:>5} | {log['samples']:>7} | {log['avg_loss']:>8.5f} | {lw.get('L1_trend',0):>7.1%} | {lw.get('L2_mean_reversion',0):>5.1%} | {lw.get('L3_volatility',0):>5.1%} | {lw.get('L4_stat_arb',0):>5.1%} | {lw.get('L5_meta',0):>5.1%}")
    
    print(f"\nFinal transferable weights for {len(weights['signal_names'])} signals")
    print(f"Coinbase mapping for {len(weights['mapping'])} pairs")
    
    trainer.save_checkpoint("hrm/data/binance_spot_weights_final.json")
    
    # Plot convergence if matplotlib available
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Layer weights over epochs
        epochs = [log['epoch'] for log in convergence_log]
        layer_names = ['L1_trend', 'L2_mean_reversion', 'L3_volatility', 'L4_stat_arb', 'L5_meta']
        for layer in layer_names:
            vals = [log['layer_weights'].get(layer, 0) for log in convergence_log]
            axes[0,0].plot(epochs, vals, marker='o', label=layer)
        axes[0,0].set_title("Layer Weight Convergence")
        axes[0,0].set_xlabel("Epoch")
        axes[0,0].set_ylabel("Weight")
        axes[0,0].legend(loc='best', fontsize=8)
        axes[0,0].grid(True, alpha=0.3)
        
        # Top signal weights over epochs
        hw_final = trainer.ranker.get_hierarchical_weights()
        signal_weights = [(s, hw_final.signal_weights[SIGNAL_16.index(s)]) for s in SIGNAL_16]
        signal_weights.sort(key=lambda x: -x[1])
        top_signals = [s[0] for s in signal_weights[:5]]
        
        for sig in top_signals:
            idx = SIGNAL_16.index(sig)
            vals = [log['signal_weights'][idx] for log in convergence_log]
            axes[0,1].plot(epochs, vals, marker='o', label=sig[:15])
        axes[0,1].set_title("Top 5 Signal Weight Convergence")
        axes[0,1].set_xlabel("Epoch")
        axes[0,1].set_ylabel("Weight")
        axes[0,1].legend(loc='best', fontsize=8)
        axes[0,1].grid(True, alpha=0.3)
        
        # Loss convergence
        losses = [log['avg_loss'] for log in convergence_log]
        axes[1,0].plot(epochs, losses, marker='o', color='red', linewidth=2)
        axes[1,0].set_title("Loss Convergence")
        axes[1,0].set_xlabel("Epoch")
        axes[1,0].set_ylabel("Avg Loss")
        axes[1,0].grid(True, alpha=0.3)
        
        # Sample count
        samples = [log['samples'] for log in convergence_log]
        axes[1,1].plot(epochs, samples, marker='o', color='green', linewidth=2)
        axes[1,1].set_title("Training Samples Accumulated")
        axes[1,1].set_xlabel("Epoch")
        axes[1,1].set_ylabel("Total Samples")
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("hrm/data/convergence_plot.png", dpi=150)
        print("\nConvergence plot saved to hrm/data/convergence_plot.png")
        
    except ImportError as e:
        print(f"\nMatplotlib not available ({e}). Skipping convergence plot.")


if __name__ == "__main__":
    main()
