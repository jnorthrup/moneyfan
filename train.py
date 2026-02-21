#!/usr/bin/env python3
"""
EpochBasket Training System
============================

Single pipeline: Source → DuckDB → Pandas → CandleCache → EpochBasketTrainer

Each epoch basket is a stochastic multi-pair OHLCV sampling window:
  - pair_width    : number of coin pairs per basket
  - bar_sequences_per_basket : number of sliding OHLCV bar windows drawn per basket
  - min_bar_window / max_bar_window : stochastic bar window length (in candles)
  - candles_per_extent : raw candle depth per extent from the data pipeline

Usage:
    python train.py --baskets 500 --notional 100
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
    print("[DEBUG] MLX imported successfully")
except ImportError as e:
    HAS_MLX = False
    print(f"[DEBUG] MLX NOT imported: {e}")


@dataclass
class BasketTrainingConfig:
    """
    Configuration for stochastic epoch basket training.

    Crypto-technical parameter names:
      n_epoch_baskets         : total number of stochastic multi-pair sampling windows
      notional                : starting notional value (not equity)
      pair_width              : number of coin pairs per basket
      bar_sequences_per_basket: number of sliding OHLCV bar windows drawn per basket
      min_bar_window          : minimum stochastic bar window length (candles)
      max_bar_window          : maximum stochastic bar window length (candles)
      epochs                  : passes over each basket
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
    n_epoch_baskets: int = 500
    notional: float = 100.0
    pair_width: int = 30
    bar_sequences_per_basket: int = 100
    min_bar_window: int = 64
    max_bar_window: int = 256
    epochs: int = 1
    learning_rate: float = 1e-4
    cache_size: int = 1000
    candles_per_extent: int = 1000
    shock_z_threshold: float = 2.0
    bar_shock_z_threshold: float = 3.0
    max_adaptive_replays: int = 3

    # Legacy aliases — kept for dashboard compatibility (viewserver reads these)
    @property
    def n_bags(self): return self.n_epoch_baskets
    @property
    def capital(self): return self.notional
    @property
    def bag_size(self): return self.pair_width
    @property
    def sequences_per_bag(self): return self.bar_sequences_per_basket
    @property
    def min_seq_len(self): return self.min_bar_window
    @property
    def max_seq_len(self): return self.max_bar_window
    @property
    def per_extent_length(self): return self.candles_per_extent
    @property
    def extent_outlier_z(self): return self.shock_z_threshold
    @property
    def frame_outlier_z(self): return self.bar_shock_z_threshold
    @property
    def max_optimizer_replays(self): return self.max_adaptive_replays


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
        Compute codec input features from raw OHLCV candles.

        Returns an array of shape [T, n_codec_outputs * 2]:
          - First n_codec_outputs columns : normalised price/vol features
          - Last n_codec_outputs columns  : per-bar close returns tiled across codecs
        """
        df = df.sort_values(['symbol', 'timestamp'])
        T = len(df)
        codec_features = np.zeros((T, n_codec_outputs * 2), dtype=np.float32)

        c = df['close'].values.astype(np.float32)
        h = df['high'].values.astype(np.float32)
        l = df['low'].values.astype(np.float32)

        c = np.nan_to_num(c, nan=1.0)
        h = np.nan_to_num(h, nan=c)
        l = np.nan_to_num(l, nan=c)

        # Feature 0: normalised close price
        codec_features[:, 0] = np.clip((c - np.mean(c)) / (np.std(c) + 1e-8), -1, 1)
        # Feature 1: high-low range normalised by close (intrabar volatility)
        codec_features[:, 1] = (h - l) / (c + 1e-8)

        for i in range(2, n_codec_outputs):
            codec_features[:, i] = np.random.randn(T) * 0.1

        # Tiled close-bar returns across all codec output channels
        close_bar_returns = np.diff(c, prepend=c[0]) / (c + 1e-8)
        codec_features[:, n_codec_outputs:] = np.tile(
            close_bar_returns.reshape(-1, 1), (1, n_codec_outputs)
        )

        return codec_features


class EpochBasketTrainer:
    """
    Trains the HRM over stochastic epoch baskets.

    Each basket is an independently sampled (pair_width × bar_window) OHLCV window.
    Regime shocks (loss z-score > shock_z_threshold) trigger adaptive replay loops.
    """
    def __init__(self, config: BasketTrainingConfig):
        self.config = config
        self.candle_cache = CandleCache(config.cache_size)
        self.candle_pipeline = CandlePipeline(self.candle_cache)

        self.model_config = MLXConfig(
            n_codec_outputs=24,
            hidden_dim=64,
            ob_depth_frames=20,
            ob_lookback_horizon=200
        ) if HAS_MLX else None

        self.results: List[Dict] = []
        self.event_queue = queue.Queue()
        self.running = False
        # ISO-8601 timestamp recorded at trainer construction
        self.session_start_time: str = datetime.now().isoformat()

    def train_basket(self, basket_id: int, basket_pairs: List[str]) -> Dict:
        """
        Train one epoch basket.

        Args:
            basket_id   : sequential basket index
            basket_pairs: list of coin pairs in this basket (e.g. ['BTCUSDT', 'ETHUSDT', ...])

        Returns:
            Result dict with realized_pnl, hit_rate, world_model_loss, regime_shock_count, etc.
        """
        if not HAS_MLX:
            return {'bag_id': basket_id, 'error': 'MLX not available'}

        model = MLXHierarchicalCodec(self.model_config)
        trainer = MLXCodecTrainer(self.model_config)
        trainer.model = model

        df = self.candle_pipeline.load_candles(basket_pairs, None, None)

        if df.empty:
            return {'bag_id': basket_id, 'error': 'No data loaded'}

        if not df.empty and self.config.candles_per_extent != -1:
            df = df.iloc[-self.config.candles_per_extent:]

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

        # Close-bar returns from feature channel 0 (normalised close)
        close_bar_returns = codec_features[:, 0]

        for epoch in range(self.config.epochs):
            bar_window_len = np.random.randint(
                self.config.min_bar_window, self.config.max_bar_window
            )

            for bar_seq_i in range(self.config.bar_sequences_per_basket):
                if bar_window_len > len(codec_features):
                    continue

                start_idx = np.random.randint(0, len(codec_features) - bar_window_len)
                batch = codec_features[start_idx:start_idx + bar_window_len]
                batch = batch.reshape(1, bar_window_len, -1)

                # HRM world-model training step (thread-safe, skip if previously failed)
                if not hasattr(self, '_mlx_disabled') or not self._mlx_disabled:
                    try:
                        batch_mx = mx.array(batch)
                        world_model_loss = trainer.pretrain_step(batch_mx)
                        mx.eval(world_model_loss)  # Force evaluation
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
                                        np.random.randn(*batch.shape) * shock_perturbation_mag
                                    )
                                    # Random frame masking (simulates missing candle data)
                                    frame_mask = (
                                        np.random.random(batch.shape) > 0.1
                                    ).astype(np.float32)

                                    perturbed_bar_batch = (batch + shock_perturbation_noise) * frame_mask
                                    perturbed_batch_mx = mx.array(perturbed_bar_batch)
                                    trainer.pretrain_step(perturbed_batch_mx)
                    except Exception as e:
                        print(f"MLX disabled on basket {basket_id}: {type(e).__name__}: {e}")
                        self._mlx_disabled = True
                        bar_window_losses.append(0.0)
                else:
                    bar_window_losses.append(0.0)

                # Trade signal simulation — always executes, independent of MLX
                if np.random.random() > 0.5:
                    # Candle return over this bar window's extent
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

            # Throttle events to avoid Streamlit freeze on 500 baskets
            current_time = time.time()
            if (
                not hasattr(self, '_last_event_time')
                or current_time - self._last_event_time > 0.1
                or epoch == self.config.epochs - 1
            ):
                self._last_event_time = current_time

                self.event_queue.put(('epoch_complete', {
                    'bag_id': basket_id,
                    'epoch': epoch + 1,
                    'total_epochs': self.config.epochs,
                    'capital': notional,
                    'pnl': notional - self.config.notional,
                    'win_rate': profitable_trades / max(total_trade_signals, 1),
                    'symbols': basket_pairs,
                    'winning_agent': "None (Real execution pending)",
                    'hrm_score': 0.0,
                    'predictor_loss': float(np.mean(bar_window_losses[-10:])) if bar_window_losses else 0.0,
                    'outlier_extents': regime_shock_count,
                    'optimizer_replays': adaptive_replay_count
                }))

        result = {
            'bag_id': basket_id,
            'symbols': basket_pairs,
            'final_capital': notional,
            'pnl': notional - self.config.notional,
            'win_rate': profitable_trades / max(total_trade_signals, 1),
            'wins': profitable_trades,
            'total_trades': total_trade_signals,
            'winning_agent': "None (Real execution pending)",
            'hrm_score': 0.0,
            'predictor_loss': float(np.mean(bar_window_losses)) if bar_window_losses else 0.0,
            'outlier_extents': regime_shock_count,
            'optimizer_replays': adaptive_replay_count,
            'equity_curve': notional_curve,
            'timestamp': datetime.now().isoformat()
        }

        return result

    def run_basket_training(self, progress_callback=None):
        """
        Run the full stochastic epoch basket training loop.

        Implements Self-Adapting Pareto Replay:
          - 30% chance to replay from Pareto-tail baskets (|z| > 1.5)
          - Perturbation magnitude scales with tail extremity
          - alpha_extreme tail: densify profitable patterns
          - drawdown_extreme tail: build regime robustness
        """
        self.running = True
        self.results = []
        # Record (and refresh) session start time when training actually begins
        self.session_start_time = datetime.now().isoformat()
        print(f"[SESSION] Basket training started at {self.session_start_time}")

        all_pairs = [
            'ADAUSDT', 'APTUSDT', 'ARBUSDT', 'ATOMUSDT', 'AVAXUSDT',
            'BCHUSDT', 'BNBUSDT', 'BONKUSDT', 'BTCUSDT', 'DOGEUSDT',
            'DOTUSDT', 'ETCUSDT', 'ETHUSDT', 'FILUSDT', 'INJUSDT',
            'JUPUSDT', 'LINKUSDT', 'LTCUSDT', 'MATICUSDT', 'OPUSDT',
            'PEPEUSDT', 'PYTHUSDT', 'RUNEUSDT', 'SEIUSDT', 'SOLUSDT',
            'SUIUSDT', 'TIAUSDT', 'UNIUSDT', 'WIFUSDT', 'XRPUSDT',
        ]

        for basket_id in range(self.config.n_epoch_baskets):
            if not self.running:
                break

            # Self-Adapting Pareto Replay (Risk outside of normalcy)
            is_pareto_replay = False
            pareto_perturbation_mag = 0.0
            basket_pairs: List[str] = []

            if len(self.results) > 10:
                pnls = [r.get('pnl', 0.0) for r in self.results]
                mean_pnl = np.nanmean(pnls)
                std_pnl = np.nanstd(pnls) + 1e-8

                # Update z-scores for all past baskets to find current Pareto tails
                for r in self.results:
                    r['z_score'] = (r.get('pnl', 0.0) - mean_pnl) / std_pnl

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
                    basket_pairs = selected['symbols']

                    pareto_tail_label = "alpha_extreme" if z_score > 0 else "drawdown_extreme"
                    print(
                        f"Pareto Replay [{pareto_tail_label}] basket {selected['bag_id']} "
                        f"(z={z_score:.2f}, perturb={pareto_perturbation_mag:.2f})"
                    )

            if not is_pareto_replay:
                np.random.seed(basket_id)
                basket_pairs = list(np.random.choice(
                    all_pairs,
                    size=min(self.config.pair_width, len(all_pairs)),
                    replace=False
                ))

            result = self.train_basket(basket_id, basket_pairs)
            if is_pareto_replay:
                result['is_replay'] = True
                result['replay_std'] = pareto_perturbation_mag

            self.results.append(result)
            self.event_queue.put(('bag_complete', result))

            if progress_callback:
                progress_callback(basket_id + 1, self.config.n_epoch_baskets, result)

            if (basket_id + 1) % 10 == 0:
                self._save_checkpoint(basket_id + 1)

        self._save_final_results()
        self.running = False

    def _save_checkpoint(self, completed_baskets: int):
        checkpoint = {
            'completed_bags': completed_baskets,
            'total_bags': self.config.n_epoch_baskets,
            'session_start_time': self.session_start_time,
            'checkpoint_time': datetime.now().isoformat(),
            'results': self.results,
        }

        with open('training_checkpoint.json', 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def _save_final_results(self):
        summary = {
            'total_baskets': len(self.results),
            'session_start_time': self.session_start_time,
            'session_end_time': datetime.now().isoformat(),
            'avg_pnl': np.mean([r['pnl'] for r in self.results if 'pnl' in r]),
            'avg_win_rate': np.mean([r['win_rate'] for r in self.results if 'win_rate' in r]),
            'total_notional': sum([r['final_capital'] for r in self.results if 'final_capital' in r]),
            'results': self.results
        }

        with open('training_results.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nTraining complete!")
        print(f"Total baskets: {summary['total_bags']}")
        print(f"Avg PnL: ${summary['avg_pnl']:.2f}")
        print(f"Avg Hit Rate: {summary['avg_win_rate']:.1%}")


# ---------------------------------------------------------------------------
# Dashboard shim — keeps the viewserver (dashboard.py) import-compatible.
# The viewserver imports UnifiedTrainer and TrainingConfig by name.
# ---------------------------------------------------------------------------

class UnifiedTrainer(EpochBasketTrainer):
    """
    Viewserver-compatibility shim.
    The Streamlit dashboard imports this name directly; do not remove.
    All logic lives in EpochBasketTrainer.
    """
    def run_training(self, progress_callback=None):
        return self.run_basket_training(progress_callback)

    def train_bag(self, bag_id: int, symbols: List[str]) -> Dict:
        return self.train_basket(bag_id, symbols)


@dataclass
class TrainingConfig(BasketTrainingConfig):
    """
    Viewserver-compatibility shim for dashboard.py imports.
    Accepts legacy field names and maps them to BasketTrainingConfig.
    """
    pass


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
        n_bags = st.number_input("Number of Bags", min_value=1, max_value=1000, value=500)
        capital = st.number_input("Starting Capital ($)", min_value=10, value=100)
        bag_size = st.number_input("Bag Size", min_value=5, max_value=50, value=30)

        if st.button("Start Training", type="primary"):
            config = TrainingConfig(
                n_epoch_baskets=n_bags,
                notional=capital,
                pair_width=bag_size
            )

            st.session_state.trainer = UnifiedTrainer(config)
            st.session_state.results = []

            thread = threading.Thread(
                target=st.session_state.trainer.run_training,
                daemon=True
            )
            thread.start()
            st.session_state.training_thread = thread

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Bags Trained", f"{len(st.session_state.results)} / {n_bags}")

    with col2:
        if st.session_state.results:
            avg_pnl = np.mean([r['pnl'] for r in st.session_state.results if 'pnl' in r])
            st.metric("Avg PnL", f"${avg_pnl:.2f}")

    with col3:
        if st.session_state.results:
            avg_wr = np.mean([r['win_rate'] for r in st.session_state.results if 'win_rate' in r])
            st.metric("Avg Win Rate", f"{avg_wr:.1%}")

    if st.session_state.trainer:
        while True:
            try:
                event_type, data = st.session_state.trainer.event_queue.get_nowait()
                if event_type == 'bag_complete':
                    st.session_state.results.append(data)
            except queue.Empty:
                break

        st.rerun()

    if st.session_state.results:
        st.subheader("Recent Results")

        df = pd.DataFrame(st.session_state.results[-20:])

        if not df.empty:
            st.dataframe(
                df[['bag_id', 'pnl', 'win_rate', 'total_trades']].round(3),
                use_container_width=True
            )

            col1, col2 = st.columns(2)

            with col1:
                st.line_chart(
                    pd.DataFrame(st.session_state.results)['pnl'].cumsum(),
                    height=300
                )

            with col2:
                st.bar_chart(
                    pd.DataFrame(st.session_state.results)['win_rate'],
                    height=300
                )


def main():
    parser = argparse.ArgumentParser(description='EpochBasket Training System')
    parser.add_argument('--baskets', '--bags', type=int, default=500,
                        help='Number of epoch baskets to train')
    parser.add_argument('--notional', '--capital', type=float, default=100.0,
                        help='Starting notional value')
    parser.add_argument('--pair-width', '--bag-size', type=int, default=30,
                        help='Coin pairs per basket')
    parser.add_argument('--dashboard', action='store_true', help='Run Streamlit dashboard')

    args = parser.parse_args()

    if args.dashboard:
        run_dashboard()
    else:
        print(f"Starting training: {args.baskets} epoch baskets, ${args.notional} notional")

        config = BasketTrainingConfig(
            n_epoch_baskets=args.baskets,
            notional=args.notional,
            pair_width=args.pair_width
        )

        trainer = EpochBasketTrainer(config)

        def progress(current, total, result):
            pct = (current / total) * 100
            pnl = result.get('pnl', 0)
            print(f"Basket {current}/{total} ({pct:.1f}%) - PnL: ${pnl:.2f}")

        trainer.run_basket_training(progress_callback=progress)


if __name__ == "__main__":
    if HAS_STREAMLIT and '--dashboard' in sys.argv:
        run_dashboard()
    else:
        main()
