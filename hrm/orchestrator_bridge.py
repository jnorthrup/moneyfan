"""
Orchestrator Bridge
===================

Connects the `signal_orchestrator.py` (The Senses) to `hrm/signal_hrm.py` (The Brain).
Adapts the lazy, pandas-based signals into the [batch, seq_len, 32] tensor required by HRM.
"""

import sys
import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# Ensure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal_orchestrator import (
    Orchestrator,
    GridService,
    MomentumService,
    RSIService,
    TrendService,
    VolatilityService,
    VolumeService,
    DataLoader
)
try:
    from hrm.signal_hrm import SIGNAL_16, N_SIGNALS
    from hrm.duck_store import DuckStore
except ImportError:
    from signal_hrm import SIGNAL_16, N_SIGNALS
    from duck_store import DuckStore

class OrchestratorBridge:
    def __init__(self, duck_dir: str = 'hrm/data/market.duckdb'):
        self.store = DuckStore(duck_dir)
        self.orchestrator = Orchestrator(max_workers=4)
        self._register_services()
        self._setup_compositions()

    def _register_services(self):
        """Register all available signal services"""
        self.orchestrator.register_service(GridService())
        self.orchestrator.register_service(MomentumService())
        self.orchestrator.register_service(RSIService())
        self.orchestrator.register_service(TrendService())
        self.orchestrator.register_service(VolatilityService())
        self.orchestrator.register_service(VolumeService())

    def _setup_compositions(self):
        """Setup synthetic signals to fill the 16 HRM slots"""
        pass

    def get_signal_map(self, signals: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Map available orchestrator signals to the 16 canonical HRM signals.
        Returns a dictionary keyed by SIGNAL_16 names.
        """
        # Base signals
        grid = signals.get('grid', pd.Series(0, index=list(signals.values())[0].index))
        mom = signals.get('momentum', grid * 0)
        rsi = signals.get('rsi', grid * 0)
        trend = signals.get('trend', grid * 0)
        vol = signals.get('volatility', grid * 0)
        volume = signals.get('volume', grid * 0)

        # MAPPING LOGIC
        # We construct the 16 signals from our available primitives
        
        m = {}
        
        # TREND (4)
        m["macd_crossover"]     = trend              # Proxy
        m["sota_momentum"]      = mom                # Direct
        m["momentum_trend"]     = (mom + trend) / 2  # Composition
        m["mom_trend_additive"] = mom + trend        # Composition

        # MEAN REVERSION (4)
        m["rsi_mean_reversion"]  = rsi               # Direct
        m["bollinger_reversion"] = grid              # Proxy (grid is mean rev)
        m["grid_reversion"]      = grid              # Direct
        m["hrm_mean_reversion"]  = (rsi + grid) / 2  # Composition

        # VOLATILITY (3)
        m["volatility_breakout"]   = vol             # Direct
        m["vol_x_breakout_proven"] = vol * mom       # Composition
        m["momentum_x_vol"]        = mom * vol       # Composition

        # STAT ARB (2) - Lacking distinct data, use proxies
        m["bent_penny"]   = grid * 0.5               # Weak proxy
        m["pairs_spread"] = grid * 0.5               # Weak proxy

        # SYSTEMATIC (1)
        m["dca_baseline"] = trend * 0.2              # Weak trend bias

        # ML (1)
        m["technical_ml"] = (mom + rsi + trend) / 3  # Ensemble proxy

        # COMPOSITE (1)
        m["rsi_x_trend"] = rsi * trend               # Composition

        return m

        return m

    def get_context_vector(self, symbol: str) -> np.ndarray:
        """
        Generate "Agent Footprint" / Context features.
        Returns a 1D array of floats.
        
        Features (Total 8 dims):
        - [0-3] Market Regime (One-hot: Bull, Bear, Crab, Volatile) - Placeholder
        - [4-7] Sector identity (One-hot: L1, DeFi, Meme, Other) - Placeholder
        """
        # Feature sizes
        N_REGIME = 4
        N_SECTOR = 4
        
        ctx = np.zeros(N_REGIME + N_SECTOR, dtype=np.float32)
        
        # 1. Regime (Heuristic placeholder - could come from Orchestrator global state)
        # Default to 'Volatile' [0, 0, 0, 1] for crypto
        ctx[3] = 1.0 
        
        # 2. Sector (Simple mapping)
        if symbol in ["BTC", "ETH", "SOL", "ADA", "AVAX"]:
            ctx[4] = 1.0 # L1
        elif symbol in ["UNI", "AAVE", "LINK"]:
            ctx[5] = 1.0 # DeFi
        elif symbol in ["DOGE", "SHIB"]:
            ctx[6] = 1.0 # Meme
        else:
            ctx[7] = 1.0 # Other
            
        return ctx

    def compute_tensor(self, symbol: str, lookback: int = 100, seq_len: int = 32, 
                       include_context: bool = True, df_input: pd.DataFrame = None) -> Optional[np.ndarray]:
        """
        Compute input tensor for HRM.
        
        Args:
            include_context: If True, appends context/footprint features to the channel dim.
                             Output shape: [1, seq_len, 32 + context_dim + 1]
            df_input: Optional DataFrame to use instead of loading from store.
        """
        # 1. Load Data
        df = df_input if df_input is not None else self.store.load(symbol)
        if df.empty or len(df) < seq_len + 10: 
            return None

        # 1.1 Compute Relative Sparkline Coefficient
        # Sparkline = EMA(Price) slope normalized by ATR
        close = df['close']
        ema_fast = close.ewm(span=5).mean()
        ema_slow = close.ewm(span=20).mean()
        atr = (df['high'] - df['low']).rolling(20).mean()
        
        # sparkline_coef: normalized local velocity
        spark_coef = (ema_fast - ema_slow) / (atr + 1e-8)
        spark_coef = np.clip(spark_coef.values[-seq_len:], -3.0, 3.0) / 3.0 # Soft normalize

        # 2. Run Orchestrator
        # Synchronous run for simplicity in this bridge
        results = self.orchestrator.run(df)
        signals = results['signals']

        # 3. Map to HRM canonicals
        mapped_signals = self.get_signal_map(signals)

        # 4. Construct Tensor
        # Shape: [seq_len, N_SIGNALS * 2]
        # Interleaved: [sig, conf, sig, conf...]
        
        # Helper to get numpy array last seq_len
        def get_vals(s: pd.Series):
            v = s.values[-seq_len:]
            if len(v) < seq_len:
                # Pad with zeros
                pad = np.zeros(seq_len - len(v))
                v = np.concatenate([pad, v])
            return v.astype(np.float32)

        # 4. Process Signals
        tensor = np.zeros((seq_len, N_SIGNALS * 2), dtype=np.float32)
        
        # 4.1 Collect Tradebot Context (New)
        # Include current composition values as context features
        comp_features = []
        for name in ['grid_momentum', 'rsi_trend', 'all_multiply']:
            c_sig = results.get('compositions', {}).get(name, pd.Series(0, index=df.index))
            comp_features.append(np.clip(c_sig.iloc[-1], -2.0, 2.0) / 2.0)
        
        ctx_vec = self.get_context_vector(symbol)
        # Combine static context with dynamic tradebot context
        ctx_vec = np.concatenate([ctx_vec, np.array(comp_features, dtype=np.float32)])
        
        # Mapping signals to the 16 slots
        for i, name in enumerate(SIGNAL_16):
            sig_series = mapped_signals.get(name, pd.Series(0))
            
            # Interleaved: [sig, conf, sig, conf...]
            
            # Extract signal values
            raw_sig = get_vals(sig_series)
            
            # 4.1 Hyperbolic Lensing (Model B preference)
            # Expand the signal near boundaries using tanh
            lensed_sig = np.tanh(raw_sig * 1.5) # x_lensed = tanh(k*x)
            
            # Calculate confidence (heuristic: magnitude of signal)
            # Clip signal to [-1, 1] just in case
            sig_val = np.clip(lensed_sig, -1.0, 1.0)
            conf_val = np.abs(sig_val)

            # Fill tensor
            tensor[:, i * 2]     = sig_val
            tensor[:, i * 2 + 1] = conf_val

        # 5. Append Context (Footprint) if requested
        if include_context:
             # Agent Footprint (Context)
            # Replicate ctx count as a persistent feature channel [seq_len, 8+3]
            ctx_broadcast = np.tile(ctx_vec, (seq_len, 1))
            
            # Concatenate along feature dim
            tensor = np.concatenate([tensor, ctx_broadcast], axis=1)

        # 6. Append Sparkline Coefficient (New)
        # Add as a dedicated channel at the end [seq_len, 1]
        spark_broadcast = spark_coef[:, np.newaxis]
        tensor = np.concatenate([tensor, spark_broadcast], axis=1)

        # Add batch dimension
        return tensor[np.newaxis, :, :]

    def warmup(self):
        """Warm up the orchestrator (e.g. pre-load data)"""
        pass

if __name__ == "__main__":
    # Smoke test
    bridge = OrchestratorBridge()
    print("Bridge initialized.")
    
    # Try to load some data (assuming BTC exists in the DB, if not this might fail gracefully)
    # We construct a fake DF if DB is empty for test purposes
    try:
        tensor = bridge.compute_tensor("BTC")
        if tensor is not None:
            print(f"Computed tensor shape: {tensor.shape}")
        else:
            print("No data found for BTC, skipping tensor check.")
    except Exception as e:
        print(f"Error during smoke test: {e}")
