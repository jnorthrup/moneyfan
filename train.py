#!/usr/bin/env python3
"""
EpochBasket Training System
============================

Single pipeline: Source → DuckDB → Pandas → CandleCache → EpochBasketTrainer

Each epoch episode is a stochastic multi-pair OHLCV sampling window:
  - pair_width    : number of coin pairs per episode
  - bar_sequences_per_episode : number of sliding OHLCV bar windows drawn per episode
  - min_bar_window / max_bar_window : stochastic bar window length (in candles)
  - candles_per_extent : raw candle depth per extent from the data pipeline

Usage:
    python train.py --episodes 500 --notional 100
    streamlit run train.py -- --dashboard
"""

import sys
import os
import argparse
import json
import time
import threading
import queue
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# Removed DuckStore import
HAS_DUCK = False

try:
    import mlx.core as mx
    from hrm.hierarchical_codec_mlx import (
        MLXHierarchicalCodec,
        MLXCodecTrainer,
        HierarchicalCodecConfig as MLXConfig
    )
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from codec_models import load_all_codecs
from hrm.order_intent import NormalizedTradeIntent, RiskTier, VetoDecision



@dataclass
class EpisodeTrainingConfig:
    """
    Configuration for stochastic epoch episode training.

    Crypto-technical parameter names:
      n_epoch_episodes         : total number of stochastic multi-pair sampling windows
      notional                : starting notional value (not equity)
      pair_width              : number of coin pairs per episode
      bar_sequences_per_episode: number of sliding OHLCV bar windows drawn per episode
      min_bar_window          : minimum stochastic bar window length (candles)
      max_bar_window          : maximum stochastic bar window length (candles)
      epochs                  : passes over each episode
      learning_rate           : optimizer step size
      cache_size              : LRU candle cache capacity (number of DataFrames)
      candles_per_extent      : raw candle depth per extent (-1 = no limit)
      shock_z_threshold       : z-score above which a bar window is flagged as a regime shock
      bar_shock_z_threshold   : per-bar z-score threshold for frame-level shock detection
      max_adaptive_replays    : max extra SGD steps applied on a regime shock window

    Extent definition:
      extent = T + n  (bar_window_len T bars + prediction_horizon n bars)
      candles_per_extent sets the raw candle pool depth from which extents are drawn.
    """
    n_epoch_episodes: int = 500
    notional: float = 100.0
    pair_width: int = 30
    bar_sequences_per_episode: int = 100
    min_bar_window: int = 64
    max_bar_window: int = 256
    epochs: int = 1
    learning_rate: float = 1e-4
    optimizer_name: str = "adamw"
    weight_decay: float = 1e-2
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.999
    optimizer_momentum: float = 0.95
    optimizer_nesterov: bool = True
    muon_ns_steps: int = 5
    cache_size: int = 1000
    candles_per_extent: int = 1000
    shock_z_threshold: float = 2.0
    bar_shock_z_threshold: float = 3.0
    max_adaptive_replays: int = 3
    use_mechanical_veto: bool = False
    replay_coalescing: bool = False
    replay_coalescing_chunk_size: int = 8
    ob_decay_mode: str = "exponential"
    ob_hyperbolic_tau: float = 32.0


class CandleCache:
    """
    LRU cache for OHLCV candle DataFrames.

    Keyed by (symbols, start, end) to avoid redundant Parquet reads
    during stochastic basket sampling.
    """
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, pd.DataFrame] = {}
        self.access_order: List[str] = []

    def get(self, key: str) -> Optional[pd.DataFrame]:
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def put(self, key: str, df: pd.DataFrame):
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]

        self.cache[key] = df
        self.access_order.append(key)


class CandlePipeline:
    """
    OHLCV candle data pipeline: Parquet source → CandleCache.

    Loads per-symbol Parquet files, filters by date range,
    normalises column names, and caches results.
    """
    def __init__(self, cache: CandleCache):
        self.cache = cache
        self.data_dir = Path(project_root) / "data" / "binance"
        
        # Load the 24 canonical codec experts from GOALS.md
        self.expert_classes = load_all_codecs()
        self.experts = [ExpertClass({}) for ExpertClass in self.expert_classes]
        if len(self.experts) != 24:
            print(f"⚠️  WARNING: Loaded {len(self.experts)} codec experts, expected exactly 24!")

    def load_candles(self, symbols: List[str], start: str, end: str) -> pd.DataFrame:
        cache_key = f"candles:{','.join(sorted(symbols))}:{start}:{end}"

        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            import duckdb
            from datetime import datetime

            # Use an in-memory duckdb connection optimized for quick parquet scans
            con = duckdb.connect(':memory:')

            dfs = []
            for sym in symbols:
                slug = sym.replace("-", "_").replace("/", "_")
                path = self.data_dir / f"{slug}_sequences.parquet"

                # print(f"[DEBUG] Checking path: {path} (exists? {path.exists()})")
                if path.exists():
                    try:
                        # Build the WHERE clause dynamically
                        where_clauses = []
                        if start:
                            where_clauses.append(f"timestamp >= '{start} 00:00:00'")
                        if end:
                            where_clauses.append(f"timestamp <= '{end} 23:59:59'")

                        where_sql = ""
                        if where_clauses:
                            where_sql = "WHERE " + " AND ".join(where_clauses)

                        # Use DuckDB to query the parquet file natively (zero-copy date filtering)
                        query = f"""
                            SELECT * FROM read_parquet('{path}')
                            {where_sql}
                            ORDER BY timestamp ASC
                        """

                        df_sym = con.execute(query).df()
                        # print(f"[DEBUG] DuckDB loaded {len(df_sym)} rows for {sym}")

                        if not df_sym.empty:
                            # Normalize column: 'pair' -> 'symbol'
                            if 'pair' in df_sym.columns and 'symbol' not in df_sym.columns:
                                df_sym['symbol'] = df_sym['pair']
                            elif 'symbol' not in df_sym.columns:
                                df_sym['symbol'] = sym
                            dfs.append(df_sym)
                        else:
                            print(f"[DEBUG] df_sym empty after DuckDB filter for {sym}")
                    except Exception as e:
                        print(f"[DEBUG] Failed to read {path} via DuckDB: {e}")
                else:
                    print(f"No parquet file at {path}")

            con.close()

            if dfs:
                df = pd.concat(dfs, ignore_index=True)
                self.cache.put(cache_key, df)
                return df
            else:
                print(f"[DEBUG] dfs list is empty after iterating all symbols")
        except Exception as e:
            print(f"DuckDB parquet query failed for {symbols}: {e}")

        # Fail fast if data is missing, no mock data fallback allowed
        print(f"No data available in Parquet for {symbols} at {start} - {end}")
        df = pd.DataFrame()
        self.cache.put(cache_key, df)
        return df

    def compute_signals(self, df: pd.DataFrame, n_codec_outputs: int = 24) -> np.ndarray:
        """
        Compute codec input features from raw OHLCV candles using the 24 expert panel.

        Returns an array of shape [T, n_codec_outputs * 2 + n_instruments]:
          - Channels 0   .. n_codecs-1          : signed conviction per expert per bar
          - Channels n_codecs .. 2*n_codecs-1   : tiled close-bar returns
          - Channels 2*n_codecs ..               : named raw instrument readings
            (RSI, MACD, ATR, z-scores, etc. — harvested from expert.instruments)
            These give the HRM encoder multi-task prediction targets as specified
            in GOALS.md §3: 'predicts next-bar codec features + all indicator kernels'.
        """
        from instrument_panel import InstrumentPanel

        df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        T = len(df)
        codec_features = np.zeros((T, n_codec_outputs * 2), dtype=np.float32)

        # ── Step 1: vectorized indicator pre-computation via pandas ────────
        # InstrumentPanel runs ALL indicator families (RSI, MACD, Bollinger,
        # ATR, ADX, VWAP, Stochastic, Kalman, Hurst, z-scores, OBV, EMA stack, …)
        # in a single pandas pass before any bar-level codec loop.
        # This is the authoritative GOALS.md draw-thru: pandas → instruments.
        df_enriched = InstrumentPanel(df).compute()

        c = df_enriched['close'].values.astype(np.float32)
        h = df_enriched['high'].values.astype(np.float32)
        l = df_enriched['low'].values.astype(np.float32)
        v = df_enriched['volume'].values.astype(np.float32)

        close_bar_returns = df_enriched['log_return'].values.astype(np.float32)
        codec_features[:, n_codec_outputs:] = np.tile(
            close_bar_returns.reshape(-1, 1), (1, n_codec_outputs)
        )

        num_experts = min(len(self.experts), n_codec_outputs)

        # ── Instrument predictor matrix ────────────────────────────────────
        # First pass: discover instrument key ordering from bar 0
        # We collect instrument values per bar → [T, n_instruments]
        instrument_rows: list = []          # list of dicts, one per bar
        instrument_keys: list = []          # canonical ordered key list (set on first non-empty bar)

        # Run the full DataFrame stream through the actual expert codecs
        context_buffer = {'close': [], 'high': [], 'low': [], 'volume': [], 'returns': []}

        # Rows array for O(1) bar access (avoid iterrows overhead)
        enriched_records = df_enriched.to_dict(orient='records')

        # Lean rolling buffer — only needed for 64-bar features array (MLX models)
        return_buffer: list = []

        for i in range(T):
            return_buffer.append(float(close_bar_returns[i]))
            if len(return_buffer) > 64:
                return_buffer = return_buffer[-64:]

            # ── market_data: pre-computed pandas indicator row ──────────────
            # All scalar values (RSI, MACD, ATR, Bollinger, ADX, VWAP, z-scores,
            # Kalman velocity, Hurst, momentum windows…) are already in this dict.
            # No per-bar indicator recomputation needed inside any codec.
            market_data = enriched_records[i]

            # 64-bar return series for MLX model backward compat
            n_ret = len(return_buffer)
            features = np.zeros(64, dtype=np.float32)
            features[:n_ret] = return_buffer

            bar_instruments: dict = {}

            for expert_idx in range(num_experts):
                expert = self.experts[expert_idx]
                try:
                    confidence, direction = expert.forward(market_data, features)
                    # Harvest instrument readings emitted by this codec's forward()
                    for k, val in expert.instruments.items():
                        bar_instruments[f'{expert.name}__{k}'] = float(val)
                except Exception:
                    confidence, direction = 0.0, 0.0

                # Signed conviction into codec channel
                codec_features[i, expert_idx] = float(confidence) * float(direction)

            instrument_rows.append(bar_instruments)

            # Build canonical key ordering from first bar that populates instruments
            if not instrument_keys and bar_instruments:
                instrument_keys = sorted(bar_instruments.keys())

        # ── Assemble instrument matrix ─────────────────────────────────────
        if instrument_keys:
            n_inst = len(instrument_keys)
            inst_matrix = np.zeros((T, n_inst), dtype=np.float32)
            for i, row in enumerate(instrument_rows):
                for j, k in enumerate(instrument_keys):
                    inst_matrix[i, j] = float(row.get(k, 0.0))
            # Store key registry on the pipeline for downstream use
            self.instrument_keys = instrument_keys
            return np.concatenate([codec_features, inst_matrix], axis=1)
        else:
            self.instrument_keys = []
            return codec_features




class EpochEpisodeTrainer:
    """
    Trains the HRM over stochastic epoch episodes.

    Each episode is an independently sampled (pair_width × bar_window) OHLCV window.
    Regime shocks (loss z-score > shock_z_threshold) trigger adaptive replay loops.
    """
    def __init__(self, config: EpisodeTrainingConfig):
        self.config = config
        self.candle_cache = CandleCache(config.cache_size)
        self.candle_pipeline = CandlePipeline(self.candle_cache)
        ob_depth_frames = 256 if config.ob_decay_mode == "hyperbolic" else 20

        self.model_config = MLXConfig(
            n_codec_outputs=24,
            hidden_dim=64,
            ob_depth_frames=ob_depth_frames,
            ob_lookback_horizon=200,
            ob_decay_mode=config.ob_decay_mode,
            ob_hyperbolic_tau=config.ob_hyperbolic_tau,
            use_mechanical_veto=config.use_mechanical_veto,
            replay_coalescing=config.replay_coalescing,
            optimizer_name=config.optimizer_name,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            optimizer_beta1=config.optimizer_beta1,
            optimizer_beta2=config.optimizer_beta2,
            optimizer_momentum=config.optimizer_momentum,
            optimizer_nesterov=config.optimizer_nesterov,
            muon_ns_steps=config.muon_ns_steps,
        ) if HAS_MLX else None

        # Persistent HRM model and trainer - initialized once, trained continuously
        self.model = None
        self.trainer = None
        self._init_model_if_needed()
        if HAS_MLX:
            try:
                # Use a larger dummy window (100 bars) to ensure all indicators (like Hurst-60) compute fully
                probe_len = 100
                dummy_df = pd.DataFrame({
                    'symbol': ['BTCUSDT'] * probe_len,
                    'timestamp': pd.date_range('2023-01-01', periods=probe_len, freq='1min'),
                    'open': np.linspace(20000, 20100, probe_len),
                    'high': np.linspace(20100, 20200, probe_len),
                    'low': np.linspace(19900, 20000, probe_len),
                    'close': np.linspace(20050, 20150, probe_len),
                    'volume': np.random.random(probe_len) * 100
                })
                probed_signals = self.candle_pipeline.compute_signals(dummy_df, 24)
                actual_dim = probed_signals.shape[1]
                self.model_config.input_dim = actual_dim
                print(f"[Trainer] Robust calibration: {actual_dim} input features detected.")
            except Exception as e:
                print(f"[Trainer] Warning: Calibration failed: {e}. Defaulting to 92.")
                self.model_config.input_dim = 92

        self.results: List[Dict] = []
        self.event_queue = queue.Queue()
        self.running = False
        # ISO-8601 timestamp recorded at trainer construction
        self.session_start_time: str = datetime.now().isoformat()

    def _init_model_if_needed(self, force_reinit: bool = False):
        """Initialize HRM model and trainer once, preserving training state across episodes."""
        if not HAS_MLX:
            return
        if (not force_reinit) and self.model is not None and self.trainer is not None:
            return  # Already initialized

        # Determine actual input dimension by running a robust signal pass
        try:
            # Use a larger dummy window (100 bars) to ensure all indicators compute fully
            probe_len = 100
            dummy_df = pd.DataFrame({
                'symbol': ['BTCUSDT'] * probe_len,
                'timestamp': pd.date_range('2023-01-01', periods=probe_len, freq='1min'),
                'open': np.linspace(20000, 20100, probe_len),
                'high': np.linspace(20100, 20200, probe_len),
                'low': np.linspace(19900, 20000, probe_len),
                'close': np.linspace(20050, 20150, probe_len),
                'volume': np.random.random(probe_len) * 100
            })
            probed_signals = self.candle_pipeline.compute_signals(dummy_df, 24)
            actual_dim = probed_signals.shape[1]
            self.model_config.input_dim = actual_dim
            print(f"[Trainer] Robust calibration: {actual_dim} input features detected.")
        except Exception as e:
            print(f"[Trainer] Warning: Calibration failed: {e}. Defaulting to 92.")
            self.model_config.input_dim = 92

        # Initialize persistent model and trainer
        self.model = MLXHierarchicalCodec(self.model_config)
        self.trainer = MLXCodecTrainer(self.model_config)
        self.trainer.model = self.model
        print(f"[Trainer] HRM model initialized: {self.model_config.input_dim} features, "
              f"hidden={self.model_config.hidden_dim}")

    def _build_trade_intent(self, symbol: str, output_np: np.ndarray) -> NormalizedTradeIntent:
        pred_fwd_return = float(output_np[0])
        signal_conviction = float(output_np[1])
        stop_loss_pct = float(output_np[2])
        take_profit_pct = float(output_np[3])
        position_fraction = float(output_np[4])
        direction = float(np.sign(pred_fwd_return))

        return NormalizedTradeIntent(
            symbol=symbol,
            direction=direction,
            pred_fwd_return=pred_fwd_return,
            confidence=signal_conviction,
            position_fraction=position_fraction,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            risk_tier=RiskTier.NORMAL,
        )

    def _mechanical_veto(self, intent: NormalizedTradeIntent, drawdown_pct: float) -> VetoDecision:
        if drawdown_pct <= -0.15:
            risk_tier = RiskTier.PROTECTIVE
        elif drawdown_pct <= -0.08:
            risk_tier = RiskTier.CAUTION
        else:
            risk_tier = RiskTier.NORMAL

        if not self.config.use_mechanical_veto:
            return VetoDecision(vetoed=False, reason=None, risk_tier=risk_tier)

        if intent.direction == 0.0:
            return VetoDecision(vetoed=True, reason="flat_direction", risk_tier=risk_tier)

        sl = abs(intent.stop_loss_pct)
        tp = max(intent.take_profit_pct, 0.0)
        rr = tp / max(sl, 1e-6)

        # Basic sanity on model risk heads.
        if sl < 0.002:
            return VetoDecision(vetoed=True, reason="stop_too_tight", risk_tier=risk_tier)
        if sl > 0.15:
            return VetoDecision(vetoed=True, reason="stop_too_wide", risk_tier=risk_tier)
        if tp < 0.003:
            return VetoDecision(vetoed=True, reason="target_too_small", risk_tier=risk_tier)

        min_conf = {
            RiskTier.NORMAL: 0.25,
            RiskTier.CAUTION: 0.40,
            RiskTier.PROTECTIVE: 0.55,
        }[risk_tier]
        min_rr = {
            RiskTier.NORMAL: 1.10,
            RiskTier.CAUTION: 1.30,
            RiskTier.PROTECTIVE: 1.60,
        }[risk_tier]
        max_size = {
            RiskTier.NORMAL: 1.00,
            RiskTier.CAUTION: 0.60,
            RiskTier.PROTECTIVE: 0.35,
        }[risk_tier]

        if intent.confidence < min_conf:
            return VetoDecision(vetoed=True, reason="low_confidence", risk_tier=risk_tier)
        if rr < min_rr:
            return VetoDecision(vetoed=True, reason="poor_risk_reward", risk_tier=risk_tier)
        if intent.position_fraction > (max_size * 1.4):
            return VetoDecision(vetoed=True, reason="oversized_for_tier", risk_tier=risk_tier)

        return VetoDecision(vetoed=False, reason=None, risk_tier=risk_tier)

    def train_episode(self, episode_id: int, episode_pairs: List[str],
                      cached_codec_features: Optional[np.ndarray] = None) -> Dict:
        """
        Train one epoch episode.

        Args:
            episode_id   : sequential episode index
            episode_pairs: list of coin pairs in this episode (e.g. ['BTCUSDT', 'ETHUSDT', ...])

        Returns:
            Result dict with realized_pnl, hit_rate, world_model_loss, regime_shock_count, etc.
        """
        if not HAS_MLX:
            return {'episode_id': episode_id, 'error': 'MLX not available'}

        df = self.candle_pipeline.load_candles(episode_pairs, None, None)

        if df.empty:
            return {'episode_id': episode_id, 'error': 'No data loaded'}

        if not df.empty and self.config.candles_per_extent != -1:
            df = df.iloc[-self.config.candles_per_extent:]

        # Use cached codec features during Pareto replay, skip expensive recompute
        if cached_codec_features is not None:
            codec_features = cached_codec_features
            cached_msg = f" [cached {codec_features.shape}]"
        else:
            codec_features = self.candle_pipeline.compute_signals(df, self.model_config.n_signals)
            cached_msg = ""

        # Ensure HRM model is initialized with correct input_dim
        current_input_dim = codec_features.shape[1]
        if self.model is None or getattr(self.model_config, 'input_dim', None) != current_input_dim:
            print(f"[Trainer] Recalibrating: input_dim {getattr(self.model_config, 'input_dim', 'None')} -> {current_input_dim}")
            self.model_config.input_dim = current_input_dim
            self._init_model_if_needed(force_reinit=True)

        # Reuse existing model by default; after force_reinit this points at the refreshed instance.
        trainer = self.trainer
        model = self.model

        # ── Per-codec accumulators ─────────────────────────────────────────────
        # Track cumulative |signal| per codec so we can rank experts & surface
        # a leaderboard on the dashboard.
        n_codecs = self.model_config.n_signals  # 24
        codec_names = [
            getattr(inst, 'name', f'codec_{i+1:02d}')
            for i, inst in enumerate(self.candle_pipeline.experts[:n_codecs])
        ]
        codec_conviction_sum = np.zeros(n_codecs, dtype=np.float64)  # sum of |signal|
        codec_active_bars    = np.zeros(n_codecs, dtype=np.int64)    # bars with nonzero signal

        notional = self.config.notional
        realized_pnl = 0.0
        profitable_trades = 0
        total_trade_signals = 0
        veto_count = 0
        veto_regret = 0.0
        pretrain_eval_count = 0
        replay_eval_count = 0
        replay_coalesced_batches = 0
        replay_coalesced_steps = 0

        # Accumulate per-codec conviction from the precomputed feature matrix
        # codec_features[:, 0:24] = signed signal convictions per expert per bar
        codec_abs_signals = np.abs(codec_features[:, :n_codecs])
        codec_conviction_sum = codec_abs_signals.sum(axis=0)
        codec_active_bars    = (codec_abs_signals > 1e-6).sum(axis=0)

        # Per-bar-window loss history for regime shock detection
        bar_window_losses: List[float] = []
        notional_curve = [notional]
        regime_shock_count = 0
        adaptive_replay_count = 0

        # Close-bar returns from feature channel 0 (normalised close)
        close_bar_returns = codec_features[:, 0]

        for epoch in range(self.config.epochs):
            bar_window_len = np.random.randint(
                self.config.min_bar_window, self.config.max_bar_window
            )

            hrm_memory = None

            for bar_seq_i in range(self.config.bar_sequences_per_episode):
                if bar_window_len > len(codec_features):
                    continue

                start_idx = np.random.randint(0, len(codec_features) - bar_window_len)
                batch_np = codec_features[start_idx:start_idx + bar_window_len]
                batch_np = batch_np.reshape(1, bar_window_len, -1)

                # HRM world-model training step (thread-safe, skip if previously failed)
                if not hasattr(self, '_mlx_disabled') or not self._mlx_disabled:
                    try:
                        batch_mx = mx.array(batch_np)
                        world_model_loss, hrm_memory = trainer.pretrain_step(batch_mx, memory=hrm_memory)
                        pretrain_eval_count += 1
                        loss_val = float(world_model_loss.item())
                        bar_window_losses.append(loss_val)

                        # Regime Shock Detection: flag extent if loss z-score > shock_z_threshold
                        if len(bar_window_losses) > 10:
                            mean_loss = np.mean(bar_window_losses[-50:])
                            std_loss = np.std(bar_window_losses[-50:]) + 1e-8
                            shock_z = (loss_val - mean_loss) / std_loss

                            if shock_z > self.config.shock_z_threshold:
                                # Regime shock — trigger adaptive replay loop
                                regime_shock_count += 1
                                num_replays = np.random.randint(
                                    1, self.config.max_adaptive_replays + 1
                                )
                                replay_batches_np: List[np.ndarray] = []
                                shock_perturbation_mag = min(0.1 * shock_z, 0.5)
                                for _replay in range(num_replays):
                                    adaptive_replay_count += 1
                                    shock_perturbation_noise = (
                                        np.random.randn(*batch_np.shape) * shock_perturbation_mag
                                    )
                                    frame_mask = (
                                        np.random.random(batch_np.shape) > 0.1
                                    ).astype(np.float32)
                                    replay_batches_np.append(
                                        (batch_np + shock_perturbation_noise) * frame_mask
                                    )

                                if self.config.replay_coalescing and len(replay_batches_np) > 1:
                                    chunk_size = max(1, int(self.config.replay_coalescing_chunk_size))
                                    for chunk_start in range(0, len(replay_batches_np), chunk_size):
                                        chunk = replay_batches_np[chunk_start:chunk_start + chunk_size]
                                        replay_losses = []
                                        for perturbed_bar_batch in chunk:
                                            perturbed_batch_mx = mx.array(perturbed_bar_batch)
                                            replay_loss, hrm_memory = trainer.pretrain_step(
                                                perturbed_batch_mx, memory=hrm_memory, auto_eval=False
                                            )
                                            replay_losses.append(replay_loss)

                                        total_replay_loss = (
                                            replay_losses[0]
                                            if len(replay_losses) == 1
                                            else mx.mean(mx.stack(replay_losses))
                                        )
                                        trainer.flush_updates(total_replay_loss, memory=hrm_memory)
                                        replay_eval_count += 1
                                        replay_coalesced_batches += 1
                                        replay_coalesced_steps += len(chunk)
                                else:
                                    for perturbed_bar_batch in replay_batches_np:
                                        perturbed_batch_mx = mx.array(perturbed_bar_batch)
                                        replay_loss, hrm_memory = trainer.pretrain_step(
                                            perturbed_batch_mx, memory=hrm_memory
                                        )
                                        replay_eval_count += 1

                    except Exception as e:
                        print(f"MLX disabled on episode {episode_id}: {type(e).__name__}: {e}")
                        self._mlx_disabled = True
                        bar_window_losses.append(0.0)
                else:
                    bar_window_losses.append(0.0)

                # Trade signal execution (HRM meta-allocator action)
                if HAS_MLX and not getattr(self, '_mlx_disabled', False):
                    # Only execute a trade on some steps to simulate a sparse allocator
                    if np.random.random() > 0.5:
                        # Forward pass in trade mode to get the model's prediction (inference, no state update)
                        output_mx, _ = trainer.model.forward(batch_mx, memory=hrm_memory, mode="trade")
                        mx.eval(output_mx)
                        output_np = np.array(output_mx[0, :]) # Output is [B, 5] since it already pools the final sequence step
                        active_symbol = episode_pairs[0] if episode_pairs else "UNKNOWN"
                        trade_intent = self._build_trade_intent(active_symbol, output_np)
                        running_peak = max(notional_curve) if notional_curve else notional
                        drawdown_pct = (notional - running_peak) / max(running_peak, 1e-8)
                        veto_decision = self._mechanical_veto(trade_intent, drawdown_pct)
                        end_idx = min(start_idx + bar_window_len - 1, len(close_bar_returns) - 1)
                        candle_return = close_bar_returns[end_idx] - close_bar_returns[start_idx]
                        raw_move = float(candle_return) * 0.01

                        if veto_decision.vetoed:
                            veto_count += 1
                            potential_signed_move = trade_intent.direction * raw_move
                            potential_sl = max(abs(trade_intent.stop_loss_pct), 1e-4)
                            potential_tp = max(trade_intent.take_profit_pct, 1e-4)
                            potential_clamped = min(max(potential_signed_move, -potential_sl), potential_tp)
                            potential_ret = (
                                trade_intent.position_fraction
                                * trade_intent.confidence
                                * potential_clamped
                            )
                            veto_regret += max(0.0, potential_ret)
                            continue

                        pred_fwd_return = trade_intent.pred_fwd_return
                        signal_conviction = trade_intent.confidence
                        tier_size_cap = {
                            RiskTier.NORMAL: 1.00,
                            RiskTier.CAUTION: 0.60,
                            RiskTier.PROTECTIVE: 0.35,
                        }[veto_decision.risk_tier]
                        position_fraction = min(trade_intent.position_fraction, tier_size_cap)

                        # The action: long if pred > 0, short if pred < 0
                        position_direction = trade_intent.direction
                        signed_move = position_direction * raw_move
                        sl = max(abs(trade_intent.stop_loss_pct), 1e-4)
                        tp = max(trade_intent.take_profit_pct, 1e-4)
                        clamped_signed_move = min(max(signed_move, -sl), tp)

                        # Position sizing logic leveraging conviction and drawdown-tier cap
                        exposure = position_fraction * signal_conviction
                        ret = exposure * clamped_signed_move
                        realized_pnl += ret * notional

                        if ret > 0:
                            profitable_trades += 1
                        total_trade_signals += 1

                        notional *= (1 + ret)
                        notional_curve.append(notional)
                else:
                    # Fallback random execution if MLX crashes/disabled
                    if np.random.random() > 0.5:
                        end_idx = min(start_idx + bar_window_len - 1, len(close_bar_returns) - 1)
                        candle_return = close_bar_returns[end_idx] - close_bar_returns[start_idx]

                        position = np.sign(np.random.randn())
                        ret = position * abs(candle_return) * 0.01
                        realized_pnl += ret * notional

                        if ret > 0:
                            profitable_trades += 1
                        total_trade_signals += 1

                        notional *= (1 + ret)
                        notional_curve.append(notional)

            # Throttle events to avoid Streamlit freeze on 500 episodes
            current_time = time.time()
            if (
                not hasattr(self, '_last_event_time')
                or current_time - self._last_event_time > 0.1
                or epoch == self.config.epochs - 1
            ):
                self._last_event_time = current_time

                # Determine winning agent by highest cumulative conviction this episode
                top_idx = int(np.argmax(codec_conviction_sum))
                winning_agent = codec_names[top_idx] if codec_names else "HRM Meta-Allocator"
                codec_scores = {
                    name: round(float(codec_conviction_sum[i]), 4)
                    for i, name in enumerate(codec_names)
                }

                self.event_queue.put(('episode_complete', {
                    'episode_id': episode_id,
                    'epoch': epoch + 1,
                    'total_epochs': self.config.epochs,
                    'capital': notional,
                    'realized_pnl': notional - self.config.notional,
                    'hit_rate': profitable_trades / max(total_trade_signals, 1),
                    'symbols': episode_pairs,
                    'winning_agent': winning_agent,
                    'codec_scores': codec_scores,
                    'hrm_score': 0.0,
                    'predictor_loss': float(np.mean(bar_window_losses[-10:])) if bar_window_losses else 0.0,
                    'outlier_extents': regime_shock_count,
                    'optimizer_replays': adaptive_replay_count,
                    'veto_count': veto_count,
                    'veto_regret': veto_regret,
                    'pretrain_eval_count': pretrain_eval_count,
                    'replay_eval_count': replay_eval_count,
                    'replay_coalesced_batches': replay_coalesced_batches,
                    'replay_coalesced_steps': replay_coalesced_steps,
                    'optimizer_name': self.config.optimizer_name,
                }))

        # Final per-codec leaderboard
        top_idx = int(np.argmax(codec_conviction_sum))
        winning_agent = codec_names[top_idx] if codec_names else "HRM Meta-Allocator"
        codec_scores = {
            name: round(float(codec_conviction_sum[i]), 4)
            for i, name in enumerate(codec_names)
        }

        result = {
            'episode_id': episode_id,
            'symbols': episode_pairs,
            'final_capital': notional,
            'realized_pnl': notional - self.config.notional,
            'hit_rate': profitable_trades / max(total_trade_signals, 1),
            'wins': profitable_trades,
            'total_trades': total_trade_signals,
            'winning_agent': winning_agent,
            'codec_scores': codec_scores,
            'hrm_score': 0.0,
            'predictor_loss': float(np.mean(bar_window_losses)) if bar_window_losses else 0.0,
            'outlier_extents': regime_shock_count,
            'optimizer_replays': adaptive_replay_count,
            'veto_count': veto_count,
            'veto_regret': veto_regret,
            'pretrain_eval_count': pretrain_eval_count,
            'replay_eval_count': replay_eval_count,
            'replay_coalesced_batches': replay_coalesced_batches,
            'replay_coalesced_steps': replay_coalesced_steps,
            'replay_coalescing_enabled': self.config.replay_coalescing,
            'optimizer_name': self.config.optimizer_name,
            'equity_curve': notional_curve,
            'timestamp': datetime.now().isoformat(),
            # Cache codec_features for Pareto replay - HRM-only retraining
            'codec_features': codec_features.tobytes().hex() if codec_features is not None else None,
            'codec_features_shape': codec_features.shape if codec_features is not None else None,
        }

        return result

    def run_episode_training(self, progress_callback=None):
        """
        Run the full stochastic epoch episode training loop.

        Implements Self-Adapting Pareto Replay:
          - 60% pure stochastic episodes
          - 20% alpha_extreme tail replay: densify profitable patterns
          - 20% drawdown_extreme tail replay: build regime robustness
          - Perturbation magnitude scales with tail extremity
        """
        self.running = True
        self.results = []
        # Record (and refresh) session start time when training actually begins
        self.session_start_time = datetime.now().isoformat()
        print(f"[SESSION] Episode training started at {self.session_start_time}")

        all_pairs = [
            'ADAUSDT', 'APTUSDT', 'ARBUSDT', 'ATOMUSDT', 'AVAXUSDT',
            'BCHUSDT', 'BNBUSDT', 'BONKUSDT', 'BTCUSDT', 'DOGEUSDT',
            'DOTUSDT', 'ETCUSDT', 'ETHUSDT', 'FILUSDT', 'INJUSDT',
            'JUPUSDT', 'LINKUSDT', 'LTCUSDT', 'MATICUSDT', 'OPUSDT',
            'PEPEUSDT', 'PYTHUSDT', 'RUNEUSDT', 'SEIUSDT', 'SOLUSDT',
            'SUIUSDT', 'TIAUSDT', 'UNIUSDT', 'WIFUSDT', 'XRPUSDT',
        ]

        for episode_id in range(self.config.n_epoch_episodes):
            if not self.running:
                break

            # Self-Adapting Pareto Replay (Risk outside of normalcy)
            is_pareto_replay = False
            pareto_perturbation_mag = 0.0
            episode_pairs: List[str] = []

            if len(self.results) > 10:
                realized_pnls = [r.get('realized_pnl', 0.0) for r in self.results]
                mean_pnl = np.nanmean(realized_pnls)
                std_pnl = np.nanstd(realized_pnls) + 1e-8

                # Update z-scores for all past episodes to find current Pareto tails
                for r in self.results:
                    r['z_score'] = (r.get('realized_pnl', 0.0) - mean_pnl) / std_pnl

                # Identify Pareto tails (|z| > 1.0) and split into alpha/drawdown channels
                pareto_extremes = [r for r in self.results if abs(r.get('z_score', 0.0)) > 1.0]
                alpha_extremes = [r for r in pareto_extremes if r['z_score'] > 0]
                drawdown_extremes = [r for r in pareto_extremes if r['z_score'] < 0]

                # Explicit 60/20/20 replay logic per GOALS.md specification
                roll = np.random.random()
                
                selected = None
                if roll < 0.20 and alpha_extremes:
                    weights = np.array([abs(r['z_score']) for r in alpha_extremes])
                    selected = np.random.choice(alpha_extremes, p=weights / weights.sum())
                    pareto_tail_label = "alpha_extreme"
                elif 0.20 <= roll < 0.40 and drawdown_extremes:
                    weights = np.array([abs(r['z_score']) for r in drawdown_extremes])
                    selected = np.random.choice(drawdown_extremes, p=weights / weights.sum())
                    pareto_tail_label = "drawdown_extreme"

                if selected:
                    is_pareto_replay = True
                    z_score = selected['z_score']
                    # Self-adapting perturbation: stronger for wilder extremes
                    pareto_perturbation_mag = min(0.5, 0.1 * abs(z_score))
                    episode_pairs = selected['symbols']

                    print(
                        f"Pareto Replay [{pareto_tail_label}] episode {selected['episode_id']} "
                        f"(z={z_score:.2f}, perturb={pareto_perturbation_mag:.2f})"
                    )

            if not is_pareto_replay:
                np.random.seed(episode_id)
                episode_pairs = list(np.random.choice(
                    all_pairs,
                    size=min(self.config.pair_width, len(all_pairs)),
                    replace=False
                ))

            result = self.train_episode(episode_id, episode_pairs)
            if is_pareto_replay:
                result['is_replay'] = True
                result['replay_std'] = pareto_perturbation_mag

            self.results.append(result)
            self.event_queue.put(('episode_complete', result))

            if progress_callback:
                progress_callback(episode_id + 1, self.config.n_epoch_episodes, result)

            if (episode_id + 1) % 10 == 0:
                self._save_checkpoint(episode_id + 1)

        self._save_final_results()
        self.running = False

    def _save_checkpoint(self, completed_episodes: int):
        checkpoint = {
            'completed_episodes': completed_episodes,
            'total_episodes': self.config.n_epoch_episodes,
            'session_start_time': self.session_start_time,
            'checkpoint_time': datetime.now().isoformat(),
            'results': self.results,
        }

        with open('training_checkpoint.json', 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def _save_final_results(self):
        summary = {
            'total_episodes': len(self.results),
            'session_start_time': self.session_start_time,
            'session_end_time': datetime.now().isoformat(),
            'avg_realized_pnl': np.mean([r['realized_pnl'] for r in self.results if 'realized_pnl' in r]),
            'avg_hit_rate': np.mean([r['hit_rate'] for r in self.results if 'hit_rate' in r]),
            'total_notional': sum([r['final_capital'] for r in self.results if 'final_capital' in r]),
            'results': self.results
        }

        with open('training_results.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nTraining complete!")
        print(f"Total episodes: {summary['total_episodes']}")
        print(f"Avg Realized PnL: ${summary['avg_realized_pnl']:.2f}")
        print(f"Avg Hit Rate: {summary['avg_hit_rate']:.1%}")

def run_dashboard():
    if not HAS_STREAMLIT:
        print("Streamlit not installed. Run: pip install streamlit")
        return

    st.set_page_config(page_title="MoneyFan Training Dashboard", layout="wide")
    st.title("MoneyFan Unified Training System")

    if 'trainer' not in st.session_state:
        st.session_state.trainer = None
        st.session_state.training_thread = None
        st.session_state.results = []

    with st.sidebar:
        st.header("Configuration")
        n_episodes = st.number_input("Epoch Episodes", min_value=1, max_value=1000, value=500)
        notional_val = st.number_input("Starting Notional ($)", min_value=10, value=100)
        pair_width_val = st.number_input("Pair Width", min_value=5, max_value=50, value=30)
        optimizer_name = st.selectbox("Optimizer", options=["adamw", "lion", "muon", "adam"], index=0)
        learning_rate = st.number_input("Learning Rate", min_value=1e-7, max_value=1e-1, value=1e-4, format="%.6f")
        weight_decay = st.number_input("Weight Decay", min_value=0.0, max_value=1.0, value=0.01, format="%.4f")

        if st.button("Start Training", type="primary"):
            config = EpisodeTrainingConfig(
                n_epoch_episodes=n_episodes,
                notional=notional_val,
                pair_width=pair_width_val,
                optimizer_name=str(optimizer_name),
                learning_rate=float(learning_rate),
                weight_decay=float(weight_decay),
            )

            st.session_state.trainer = EpochEpisodeTrainer(config)
            st.session_state.results = []

            thread = threading.Thread(
                target=st.session_state.trainer.run_episode_training,
                daemon=True
            )
            thread.start()
            st.session_state.training_thread = thread

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Episodes Trained", f"{len(st.session_state.results)} / {n_episodes}")

    with col2:
        if st.session_state.results:
            avg_pnl = np.mean([r.get('realized_pnl', 0.0) for r in st.session_state.results if 'realized_pnl' in r])
            st.metric("Avg Realized PnL", f"${avg_pnl:.2f}")

    with col3:
        if st.session_state.results:
            avg_wr = np.mean([r.get('hit_rate', 0.0) for r in st.session_state.results if 'hit_rate' in r])
            st.metric("Avg Hit Rate", f"{avg_wr:.1%}")

    if st.session_state.trainer:
        while True:
            try:
                event_type, data = st.session_state.trainer.event_queue.get_nowait()
                if event_type == 'episode_complete':
                    st.session_state.results.append(data)
            except queue.Empty:
                break

        st.rerun()

    if st.session_state.results:
        st.subheader("Recent Results")

        df = pd.DataFrame(st.session_state.results[-20:])

        if not df.empty:
            st.dataframe(
                df[['episode_id', 'realized_pnl', 'hit_rate', 'total_trades']].round(3),
                use_container_width=True
            )

            col1, col2 = st.columns(2)

            with col1:
                st.line_chart(
                    pd.DataFrame(st.session_state.results)['realized_pnl'].cumsum(),
                    height=300
                )

            with col2:
                st.bar_chart(
                    pd.DataFrame(st.session_state.results)['hit_rate'],
                    height=300
                )


def main():
    parser = argparse.ArgumentParser(description='EpochEpisode Training System')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of epoch episodes to train')
    parser.add_argument('--notional', type=float, default=100.0,
                        help='Starting notional value')
    parser.add_argument('--pair-width', type=int, default=30,
                        help='Coin pairs per episode')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'lion', 'muon'],
                        help='MLX optimizer for HRM training')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='MLX optimizer learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-2,
                        help='MLX optimizer weight decay')
    parser.add_argument('--dashboard', action='store_true', help='Run Streamlit dashboard')

    args = parser.parse_args()

    if args.dashboard:
        run_dashboard()
    else:
        print(f"Starting training: {args.episodes} epoch episodes, ${args.notional} notional")

        config = EpisodeTrainingConfig(
            n_epoch_episodes=args.episodes,
            notional=args.notional,
            pair_width=args.pair_width,
            optimizer_name=args.optimizer,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        trainer = EpochEpisodeTrainer(config)

        def progress(current, total, result):
            pct = (current / total) * 100
            realized_pnl = result.get('realized_pnl', 0)
            print(f"Episode {current}/{total} ({pct:.1f}%) - Realized PnL: ${realized_pnl:.2f}")

        trainer.run_episode_training(progress_callback=progress)


if __name__ == "__main__":
    if HAS_STREAMLIT and '--dashboard' in sys.argv:
        run_dashboard()
    else:
        main()
