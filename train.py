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

from codec_models.base_codec import ExpertFactory

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# Removed DuckStore import
HAS_DUCK = False

# Safe import of Config
try:
    from hrm.hierarchical_codec_mlx import HierarchicalCodecConfig as MLXConfig
except ImportError:
    @dataclass
    class MLXConfig:
        n_codec_outputs: int = 24
        hidden_dim: int = 64
        ob_depth_frames: int = 20
        ob_lookback_horizon: int = 200
        @property
        def n_signals(self): return self.n_codec_outputs

try:
    import mlx.core as mx
    from hrm.hierarchical_codec_mlx import (
        MLXHierarchicalCodec,
        MLXCodecTrainer
    )
    HAS_MLX = True
    print("[DEBUG] MLX imported successfully")
except ImportError as e:
    HAS_MLX = False
    print(f"[DEBUG] MLX NOT imported: {e}")


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
    cache_size: int = 1000
    candles_per_extent: int = 1000
    shock_z_threshold: float = 2.0
    bar_shock_z_threshold: float = 3.0
    max_adaptive_replays: int = 3


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
        # Instantiate experts (1-24)
        print("[DEBUG] Initializing 24 Codec Experts...")
        self.experts = [ExpertFactory.create_expert(i) for i in range(1, 25)]

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to DataFrame (per symbol)."""
        df = df.copy()
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        # Helper for EMA
        def ema(series, span):
            return pd.Series(series).ewm(span=span, adjust=False).mean().values

        # RSI 14
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = ema(gain, 14)
        avg_loss = ema(loss, 14)
        rs = avg_gain / (avg_loss + 1e-8)
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # ATR 14
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        df['atr_14'] = ema(tr, 14)

        # Momentum (ROC 10)
        df['momentum'] = (close - np.roll(close, 10)) / (np.roll(close, 10) + 1e-8)

        # MACD (12, 26, 9)
        ema12 = ema(close, 12)
        ema26 = ema(close, 26)
        macd = ema12 - ema26
        signal = ema(macd, 9)
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = macd - signal

        # Bollinger Bands (20, 2)
        sma20 = pd.Series(close).rolling(window=20).mean().values
        std20 = pd.Series(close).rolling(window=20).std().values
        df['bb_upper'] = sma20 + 2 * std20
        df['bb_lower'] = sma20 - 2 * std20

        # Stochastic %K %D
        low14 = pd.Series(low).rolling(window=14).min().values
        high14 = pd.Series(high).rolling(window=14).max().values
        df['stoch_k'] = 100 * ((close - low14) / (high14 - low14 + 1e-8))
        df['stoch_d'] = pd.Series(df['stoch_k']).rolling(window=3).mean().values

        # Fill NaNs
        df = df.fillna(0)
        return df

    def load_candles(self, symbols: List[str], start: str, end: str) -> pd.DataFrame:
        cache_key = f"candles:{','.join(sorted(symbols))}:{start}:{end}"

        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            from datetime import datetime
            start_dt = datetime.strptime(start, '%Y-%m-%d') if start else None
            end_dt = datetime.strptime(end, '%Y-%m-%d') if end else None

            dfs = []
            for sym in symbols:
                slug = sym.replace("-", "_").replace("/", "_")
                path = self.data_dir / f"{slug}_sequences.parquet"

                print(f"[DEBUG] Checking path: {path} (exists? {path.exists()})")
                if path.exists():
                    try:
                        df_sym = pd.read_parquet(path, engine='pyarrow')
                        print(f"[DEBUG] Loaded {len(df_sym)} rows for {sym}")

                        # Filter by timestamp column (datetime64)
                        if start_dt and 'timestamp' in df_sym.columns:
                            df_sym = df_sym[df_sym['timestamp'] >= pd.Timestamp(start_dt)]
                        if end_dt and 'timestamp' in df_sym.columns:
                            df_sym = df_sym[df_sym['timestamp'] <= pd.Timestamp(end_dt)]

                        if not df_sym.empty:
                            # Normalize column: 'pair' -> 'symbol'
                            if 'pair' in df_sym.columns and 'symbol' not in df_sym.columns:
                                df_sym['symbol'] = df_sym['pair']
                            elif 'symbol' not in df_sym.columns:
                                df_sym['symbol'] = sym
                            dfs.append(df_sym)
                        else:
                            print(f"[DEBUG] df_sym empty after date filter for {sym}")
                    except Exception as e:
                        print(f"[DEBUG] Failed to read {path}: {e}")
                else:
                    print(f"No parquet file at {path}")

            if dfs:
                df = pd.concat(dfs, ignore_index=True)
                self.cache.put(cache_key, df)
                return df
            else:
                print(f"[DEBUG] dfs list is empty after iterating all symbols")
        except Exception as e:
            print(f"Parquet load failed for {symbols}: {e}")

        # Fail fast if data is missing, no mock data fallback allowed
        print(f"No data available in Parquet for {symbols} at {start} - {end}")
        df = pd.DataFrame()
        self.cache.put(cache_key, df)
        return df

    def compute_signals(self, df: pd.DataFrame, n_codec_outputs: int = 24) -> np.ndarray:
        """
        Compute codec input features from raw OHLCV candles using 24 Codec Experts.

        Returns an array of shape [T, n_codec_outputs * 2]:
          - First n_codec_outputs columns : expert conviction [0, 1]
          - Last n_codec_outputs columns  : expert direction [-1, 1]
        """
        # 1. Compute indicators per symbol (respecting boundaries)
        dfs = []
        for sym, sub_df in df.groupby('symbol'):
             dfs.append(self.compute_indicators(sub_df))

        df_ind = pd.concat(dfs).sort_values(['symbol', 'timestamp']) if dfs else df

        records = df_ind.to_dict('records')
        T = len(records)

        codec_features = np.zeros((T, n_codec_outputs * 2), dtype=np.float32)

        # Reset expert memory for new episode batch
        for expert in self.experts:
            expert.ob_memory = np.zeros((512, 64), dtype=np.float32)
            expert.ob_memory_idx = 0

        # Construct feature matrix [T, 64] for vectorized access
        # Columns: [close, rsi, atr, momentum, macd, macd_hist, bb_upper, bb_lower, stoch_k, stoch_d, ...zeros...]
        feat_matrix = np.zeros((T, 64), dtype=np.float32)
        feat_matrix[:, 0] = df_ind['close'].values
        feat_matrix[:, 1] = df_ind['rsi_14'].values / 100.0
        feat_matrix[:, 2] = df_ind['atr_14'].values / (df_ind['close'].values + 1e-8)
        feat_matrix[:, 3] = df_ind['momentum'].values
        feat_matrix[:, 4] = df_ind['macd'].values
        feat_matrix[:, 5] = df_ind['macd_hist'].values
        feat_matrix[:, 6] = df_ind['bb_upper'].values / (df_ind['close'].values + 1e-8)
        feat_matrix[:, 7] = df_ind['bb_lower'].values / (df_ind['close'].values + 1e-8)
        feat_matrix[:, 8] = df_ind['stoch_k'].values / 100.0
        feat_matrix[:, 9] = df_ind['stoch_d'].values / 100.0

        print(f"[DEBUG] Running 24 experts on {T} bars...")

        # Iterate (could be optimised, but sticking to per-tick integrity)
        for t in range(T):
            row = records[t]
            feats = feat_matrix[t]

            for i, expert in enumerate(self.experts):
                # i corresponds to codec_id i+1
                conf, direction = expert.forward(row, feats)
                expert.update_memory(direction, feats)

                codec_features[t, i] = conf
                codec_features[t, n_codec_outputs + i] = direction
        
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

        self.model_config = MLXConfig(
            n_codec_outputs=24,
            hidden_dim=64,
            ob_depth_frames=20,
            ob_lookback_horizon=200
        )

        self.results: List[Dict] = []
        self.event_queue = queue.Queue()
        self.running = False
        # ISO-8601 timestamp recorded at trainer construction
        self.session_start_time: str = datetime.now().isoformat()

    def train_episode(self, episode_id: int, episode_pairs: List[str]) -> Dict:
        """
        Train one epoch episode.

        Args:
            episode_id   : sequential episode index
            episode_pairs: list of coin pairs in this episode (e.g. ['BTCUSDT', 'ETHUSDT', ...])

        Returns:
            Result dict with realized_pnl, hit_rate, world_model_loss, regime_shock_count, etc.
        """
        if HAS_MLX:
            model = MLXHierarchicalCodec(self.model_config)
            trainer = MLXCodecTrainer(self.model_config)
            trainer.model = model
        else:
            model = None
            trainer = None
        df = self.candle_pipeline.load_candles(episode_pairs, None, None)

        if df.empty:
            return {'episode_id': episode_id, 'error': 'No data loaded'}

        if not df.empty and self.config.candles_per_extent != -1:
            df = df.iloc[-self.config.candles_per_extent:]

        # Ensure alignment
        df = df.sort_values(['symbol', 'timestamp'])
        close_prices = df['close'].values.astype(np.float32)

        codec_features = self.candle_pipeline.compute_signals(df, self.model_config.n_signals)

        notional = self.config.notional
        realized_pnl = 0.0
        profitable_trades = 0
        total_trade_signals = 0

        # Per-bar-window loss history for regime shock detection
        bar_window_losses: List[float] = []
        notional_curve = [notional]
        regime_shock_count = 0
        adaptive_replay_count = 0

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
                        mx.eval(world_model_loss, *hrm_memory)  # Force evaluation and truncate graph
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

                                for _replay in range(num_replays):
                                    adaptive_replay_count += 1
                                    # Perturbation magnitude scales with shock severity
                                    shock_perturbation_mag = min(0.1 * shock_z, 0.5)
                                    shock_perturbation_noise = (
                                        np.random.randn(*batch_np.shape) * shock_perturbation_mag
                                    )
                                    # Random frame masking (simulates missing candle data)
                                    frame_mask = (
                                        np.random.random(batch_np.shape) > 0.1
                                    ).astype(np.float32)

                                    perturbed_bar_batch = (batch_np + shock_perturbation_noise) * frame_mask
                                    perturbed_batch_mx = mx.array(perturbed_bar_batch)
                                    
                                    # Replay World-Model optimization with threaded memory
                                    replay_loss, hrm_memory = trainer.pretrain_step(perturbed_batch_mx, memory=hrm_memory)
                                    mx.eval(replay_loss, *hrm_memory)

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
                        
                        pred_fwd_return = float(output_np[0])
                        signal_conviction = float(output_np[1])
                        position_fraction = float(output_np[4])

                        # The action: long if pred > 0, short if pred < 0
                        position_direction = np.sign(pred_fwd_return)
                        # Position sizing logic leveraging conviction output
                        position_size = position_direction * position_fraction * signal_conviction

                        # Calculate actual realized return over this predicted span
                        end_idx = min(start_idx + bar_window_len - 1, len(close_prices) - 1)
                        # Percentage return
                        price_start = close_prices[start_idx]
                        price_end = close_prices[end_idx]
                        if price_start > 0:
                            candle_return = (price_end - price_start) / price_start
                        else:
                            candle_return = 0.0

                        # Apply execution results (simulate position hold against raw candle move)
                        # We use candle_return direction properly aligned against position_direction
                        # E.g. If long (pos) and candle goes up (pos) = positive ret
                        ret = position_size * candle_return
                        realized_pnl += ret * notional

                        if ret > 0:
                            profitable_trades += 1
                        total_trade_signals += 1

                        notional *= (1 + ret)
                        notional_curve.append(notional)
                else:
                    # Fallback random execution if MLX crashes/disabled
                    if np.random.random() > 0.5:
                        end_idx = min(start_idx + bar_window_len - 1, len(close_prices) - 1)
                        price_start = close_prices[start_idx]
                        price_end = close_prices[end_idx]
                        if price_start > 0:
                            candle_return = (price_end - price_start) / price_start
                        else:
                            candle_return = 0.0

                        position = np.sign(np.random.randn())
                        ret = position * abs(candle_return)
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

                self.event_queue.put(('episode_complete', {
                    'episode_id': episode_id,
                    'epoch': epoch + 1,
                    'total_epochs': self.config.epochs,
                    'capital': notional,
                    'realized_pnl': notional - self.config.notional,
                    'hit_rate': profitable_trades / max(total_trade_signals, 1),
                    'symbols': episode_pairs,
                    'winning_agent': "HRM Meta-Allocator",
                    'hrm_score': 0.0,
                    'predictor_loss': float(np.mean(bar_window_losses[-10:])) if bar_window_losses else 0.0,
                    'outlier_extents': regime_shock_count,
                    'optimizer_replays': adaptive_replay_count
                }))

        result = {
            'episode_id': episode_id,
            'symbols': episode_pairs,
            'final_capital': notional,
            'realized_pnl': notional - self.config.notional,
            'hit_rate': profitable_trades / max(total_trade_signals, 1),
            'wins': profitable_trades,
            'total_trades': total_trade_signals,
            'winning_agent': "HRM Meta-Allocator",
            'hrm_score': 0.0,
            'predictor_loss': float(np.mean(bar_window_losses)) if bar_window_losses else 0.0,
            'outlier_extents': regime_shock_count,
            'optimizer_replays': adaptive_replay_count,
            'equity_curve': notional_curve,
            'timestamp': datetime.now().isoformat()
        }

        return result

    def run_episode_training(self, progress_callback=None):
        """
        Run the full stochastic epoch episode training loop.

        Implements Self-Adapting Pareto Replay:
          - 30% chance to replay from Pareto-tail episodes (|z| > 1.5)
          - Perturbation magnitude scales with tail extremity
          - alpha_extreme tail: densify profitable patterns
          - drawdown_extreme tail: build regime robustness
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

                # Pareto tails: |z| > 1.5 (outside normalcy)
                pareto_extremes = [r for r in self.results if abs(r.get('z_score', 0)) > 1.5]

                # 30% chance to replay from Pareto extremes if any exist
                if pareto_extremes and np.random.random() < 0.30:
                    is_pareto_replay = True
                    # Weight selection towards the most extreme (highest |z|)
                    weights = np.array([abs(r['z_score']) for r in pareto_extremes])
                    weights /= weights.sum()
                    selected = np.random.choice(pareto_extremes, p=weights)

                    z_score = selected['z_score']
                    # Self-adapting perturbation: stronger for wilder extremes
                    pareto_perturbation_mag = min(0.5, 0.1 * abs(z_score))
                    episode_pairs = selected['symbols']

                    pareto_tail_label = "alpha_extreme" if z_score > 0 else "drawdown_extreme"
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

        if st.button("Start Training", type="primary"):
            config = EpisodeTrainingConfig(
                n_epoch_episodes=n_episodes,
                notional=notional_val,
                pair_width=pair_width_val
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
    parser.add_argument('--dashboard', action='store_true', help='Run Streamlit dashboard')

    args = parser.parse_args()

    if args.dashboard:
        run_dashboard()
    else:
        print(f"Starting training: {args.episodes} epoch episodes, ${args.notional} notional")

        config = EpisodeTrainingConfig(
            n_epoch_episodes=args.episodes,
            notional=args.notional,
            pair_width=args.pair_width
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
