#!/usr/bin/env python3
"""
EpochBasket Training System
============================

Single pipeline: Source → DuckDB → Pandas → CandleCache → EpochBasketTrainer

Each epoch episode is a stochastic multi-pair OHLCV sampling window:
  - pair_width              : number of coin pairs per episode
  - n_epoch_episodes        : total number of stochastic multi-pair sampling windows
  - bar_sequences_per_episode : number of sliding OHLCV bar windows drawn per episode
  - min_bar_window          : minimum stochastic bar window length (candles)
  - max_bar_window          : maximum stochastic bar window length (candles)
  - candles_per_extent      : raw candle depth per extent from the data pipeline

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
from dataclasses import dataclass, field, asdict
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
        HierarchicalCodecConfig as MLXConfig,
        enable_ane_optimization
    )
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    # Define a no-op stub when MLX is not available
    def enable_ane_optimization():
        print("⚠️  MLX not available - ANE/GPU optimization disabled")

from codec_models import load_all_codecs
from hrm.order_intent import NormalizedTradeIntent, RiskTier, VetoDecision

import signal
import threading


DEFAULT_TRAINING_PAIRS: List[str] = [
    'ADAUSDT', 'APTUSDT', 'ARBUSDT', 'ATOMUSDT', 'AVAXUSDT',
    'BCHUSDT', 'BNBUSDT', 'BONKUSDT', 'BTCUSDT', 'DOGEUSDT',
    'DOTUSDT', 'ETCUSDT', 'ETHUSDT', 'FILUSDT', 'INJUSDT',
    'JUPUSDT', 'LINKUSDT', 'LTCUSDT', 'MATICUSDT', 'OPUSDT',
    'PEPEUSDT', 'PYTHUSDT', 'RUNEUSDT', 'SEIUSDT', 'SOLUSDT',
    'SUIUSDT', 'TIAUSDT', 'UNIUSDT', 'WIFUSDT', 'XRPUSDT',
]



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

    Data split parameters:
      train_split             : fraction of symbols/episodes for training (0.0-1.0)
      val_split               : fraction for validation
      test_split              : fraction for testing (derived: 1.0 - train - val)
      split_mode              : 'symbols' (split by symbols) or 'time' (split by timestamp)
      time_split_fraction     : fraction of time period for train (used when split_mode='time')

    Randomness parameters:
      use_true_randomness     : use system entropy instead of episode_id seeding
      random_seed             : optional fixed seed for reproducibility (None = system entropy)

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
    trade_update_prob: float = 0.10
    trade_update_min_abs_return: float = 0.0
    # Trade-step scheduling controls for increasing signal density
    trade_step_schedule_mode: str = "probabilistic"  # "probabilistic", "deterministic", "density_gated"
    trade_step_min_density: float = 0.0  # Minimum sample density threshold for density_gated mode
    trade_step_schedule_interval: int = 0  # Interval for deterministic mode (0 = every step)
    energy_update_prob: float = 0.0
    energy_update_min_abs_return: float = 0.0
    pretrain_only: bool = False
    min_extent_days: int = 0
    max_extent_days: int = 0
    min_extent_rows: int = 256
    strict_calendar_extent: bool = False
    candle_source: str = "auto"
    duckdb_corpus_path: str = ""
    pair_universe_file: str = ""
    codec_outputs: int = 24
    energy_discount_gamma: float = 0.99
    energy_roundtrip_cost_bps: float = 16.0
    energy_churn_penalty: float = 0.0
    energy_target_clip: float = 0.25
    objective_world_model_weight: float = 1.0
    objective_trade_head_weight: float = 1.0
    objective_energy_routing_weight: float = 0.0
    objective_cost_turnover_weight: float = 0.0
    objective_regime_weight_scale: float = 1.0
    weights_path: str = ""
    hidden_dim: int = 64
    regime_layers: int = 2
    tactical_layers: int = 2
    attention_heads: int = 4
    # Timer-based stochastic training controls
    timer_based: bool = False
    min_interval_seconds: int = 30
    max_interval_seconds: int = 86400
    min_pair_width: int = 3
    max_pair_width: int = 45
    max_training_seconds: int = 0
    # Data split controls (standard ML practice)
    split_mode: str = "symbols"  # "symbols" or "time"
    train_split: float = 0.70  # 70% for training
    val_split: float = 0.15    # 15% for validation
    test_split: float = 0.15   # 15% for testing (calculated: 1.0 - train - val)
    time_split_fraction: float = 0.70  # fraction of time period for train (when split_mode='time')
    # Randomness controls (standard ML practice)
    use_true_randomness: bool = True  # Use system entropy instead of episode_id seeding
    random_seed: Optional[int] = None  # Optional fixed seed for reproducibility (None = system entropy)
    reseed_pairs_by_episode: bool = False  # DEPRECATED: now controlled by use_true_randomness


OBJECTIVE_TELEMETRY_VERSION = 1


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def objective_weight_config_from_config(config: Optional[EpisodeTrainingConfig]) -> Dict[str, Any]:
    """
    Extract and normalize profit-oriented objective weight controls for auditing.
    """
    return {
        "world_model_weight": _safe_float(getattr(config, "objective_world_model_weight", 1.0), 1.0),
        "trade_head_weight": _safe_float(getattr(config, "objective_trade_head_weight", 1.0), 1.0),
        "energy_routing_weight": _safe_float(getattr(config, "objective_energy_routing_weight", 0.0), 0.0),
        "cost_turnover_weight": _safe_float(getattr(config, "objective_cost_turnover_weight", 0.0), 0.0),
        "regime_weight_scale": _safe_float(getattr(config, "objective_regime_weight_scale", 1.0), 1.0),
        "trade_step_schedule_mode": str(getattr(config, "trade_step_schedule_mode", "probabilistic")),
        "trade_step_min_density": _safe_float(getattr(config, "trade_step_min_density", 0.0), 0.0),
        "trade_step_schedule_interval": _safe_int(getattr(config, "trade_step_schedule_interval", 0), 0),
    }


def build_episode_objective_telemetry(
    episode_metrics: Dict[str, Any],
    config: Optional[EpisodeTrainingConfig] = None,
) -> Dict[str, Any]:
    """
    Build an auditable decomposition of the training objective from episode metrics.

    Notes:
      - `world_model_term` and `trade_head_term` are direct telemetry from current
        autograd losses already emitted by training.
      - `cost_turnover_term` and `regime_weighting_term` are proxy terms for now.
        They provide visibility for profit-oriented control design before the full
        composite objective is fused into the MLX loss graph.
    """
    world_model_term = _safe_float(episode_metrics.get("predictor_loss"), 0.0)
    trade_head_term = _safe_float(
        episode_metrics.get("trade_train_loss_mean", episode_metrics.get("trade_train_loss_last", 0.0)),
        0.0,
    )
    energy_routing_term = _safe_float(
        episode_metrics.get("energy_train_loss_mean", episode_metrics.get("energy_train_loss_last", 0.0)),
        0.0,
    )
    total_trades = max(_safe_int(episode_metrics.get("total_trades"), 0), 0)
    trade_train_eval_count = max(_safe_int(episode_metrics.get("trade_train_eval_count"), 0), 0)
    pretrain_eval_count = max(_safe_int(episode_metrics.get("pretrain_eval_count"), 0), 0)
    outlier_extents = max(_safe_int(episode_metrics.get("outlier_extents"), 0), 0)
    optimizer_replays = max(_safe_int(episode_metrics.get("optimizer_replays"), 0), 0)

    fallback_denom = 1
    if config is not None:
        fallback_denom = max(int(config.bar_sequences_per_episode) * max(int(config.epochs), 1), 1)
    turnover_denom = max(trade_train_eval_count + pretrain_eval_count, fallback_denom, 1)
    cost_turnover_term = float(total_trades) / float(turnover_denom)

    # Proxy multiplier for regime-aware weighting pressure (shock/replay density).
    regime_events = outlier_extents + optimizer_replays
    regime_event_denom = max(pretrain_eval_count, fallback_denom, 1)
    regime_weighting_term = 1.0 + (float(regime_events) / float(regime_event_denom))

    additive_proxy = world_model_term + trade_head_term + energy_routing_term + cost_turnover_term
    regime_adjusted_proxy = additive_proxy * regime_weighting_term
    objective_weights = objective_weight_config_from_config(config)
    weighted_additive_proxy = (
        objective_weights["world_model_weight"] * world_model_term
        + objective_weights["trade_head_weight"] * trade_head_term
        + objective_weights["energy_routing_weight"] * energy_routing_term
        + objective_weights["cost_turnover_weight"] * cost_turnover_term
    )
    weighted_regime_multiplier = 1.0 + (
        (regime_weighting_term - 1.0) * objective_weights["regime_weight_scale"]
    )
    weighted_regime_adjusted_proxy = weighted_additive_proxy * weighted_regime_multiplier

    return {
        "version": OBJECTIVE_TELEMETRY_VERSION,
        "objective_weight_config": objective_weights,
        "components": {
            "world_model_term": {
                "value": world_model_term,
                "kind": "loss",
                "is_proxy": False,
                "source_fields": ["predictor_loss"],
            },
            "trade_head_term": {
                "value": trade_head_term,
                "kind": "loss",
                "is_proxy": False,
                "source_fields": ["trade_train_loss_mean", "trade_train_loss_last"],
            },
            "energy_routing_term": {
                "value": energy_routing_term,
                "kind": "loss",
                "is_proxy": False,
                "source_fields": ["energy_train_loss_mean", "energy_train_loss_last"],
            },
            "cost_turnover_term": {
                "value": cost_turnover_term,
                "kind": "penalty_proxy",
                "is_proxy": True,
                "source_fields": ["total_trades", "trade_train_eval_count", "pretrain_eval_count"],
            },
            "regime_weighting_term": {
                "value": regime_weighting_term,
                "kind": "multiplier_proxy",
                "is_proxy": True,
                "source_fields": ["outlier_extents", "optimizer_replays", "pretrain_eval_count"],
            },
        },
        "counts": {
            "total_trades": total_trades,
            "trade_train_eval_count": trade_train_eval_count,
            "pretrain_eval_count": pretrain_eval_count,
            "outlier_extents": outlier_extents,
            "optimizer_replays": optimizer_replays,
        },
        "weighted_components": {
            "world_model_term": float(objective_weights["world_model_weight"] * world_model_term),
            "trade_head_term": float(objective_weights["trade_head_weight"] * trade_head_term),
            "energy_routing_term": float(objective_weights["energy_routing_weight"] * energy_routing_term),
            "cost_turnover_term": float(objective_weights["cost_turnover_weight"] * cost_turnover_term),
            "regime_weighting_term": float(weighted_regime_multiplier),
        },
        "composite_proxy_unweighted": float(additive_proxy),
        "composite_proxy_regime_adjusted": float(regime_adjusted_proxy),
        "composite_proxy_weighted": float(weighted_additive_proxy),
        "composite_proxy_weighted_regime_adjusted": float(weighted_regime_adjusted_proxy),
        "notes": {
            "cost_turnover_term": "Proxy until explicit differentiable transaction-cost / turnover term is fused into MLX trade loss.",
            "regime_weighting_term": "Proxy multiplier until regime-weighted autograd objective is explicitly implemented.",
        },
    }


def attach_episode_objective_telemetry(
    episode_metrics: Dict[str, Any],
    config: Optional[EpisodeTrainingConfig] = None,
) -> Dict[str, Any]:
    enriched = dict(episode_metrics)
    enriched["objective_telemetry"] = build_episode_objective_telemetry(enriched, config=config)
    return enriched


def should_run_trade_step(
    bar_seq_i: int,
    raw_move: float,
    config: EpisodeTrainingConfig,
    sample_density: float = 1.0,
) -> bool:
    """
    Determine if trade-step should run based on scheduling mode.
    
    Args:
        bar_seq_i: Current bar sequence index
        raw_move: Raw window return magnitude
        config: Training configuration
        sample_density: Current sample density (for density_gated mode)
    
    Returns:
        bool: True if trade-step should run
    """
    mode = getattr(config, 'trade_step_schedule_mode', 'probabilistic')
    
    # First check the minimum return threshold (gating)
    min_abs_return = getattr(config, 'trade_update_min_abs_return', 0.0)
    if abs(raw_move) < float(min_abs_return):
        return False
    
    if mode == "deterministic":
        interval = getattr(config, 'trade_step_schedule_interval', 0)
        if interval <= 0:
            return True  # Every step
        return (bar_seq_i % interval) == 0
    
    elif mode == "density_gated":
        min_density = getattr(config, 'trade_step_min_density', 0.0)
        if sample_density < float(min_density):
            return False
        # Also apply probability in density_gated mode
        prob = getattr(config, 'trade_update_prob', 0.10)
        return np.random.random() < float(prob)
    
    else:  # "probabilistic" (default)
        prob = getattr(config, 'trade_update_prob', 0.10)
        return np.random.random() < float(prob)


def summarize_training_objective_telemetry(
    episode_results: List[Dict[str, Any]],
    config: Optional[EpisodeTrainingConfig] = None,
) -> Dict[str, Any]:
    """
    Aggregate per-episode objective telemetry into a training-session summary.
    """
    normalized = [attach_episode_objective_telemetry(r, config=config) for r in (episode_results or [])]
    component_names = [
        "world_model_term",
        "trade_head_term",
        "energy_routing_term",
        "cost_turnover_term",
        "regime_weighting_term",
    ]
    components: Dict[str, Dict[str, float]] = {}
    for name in component_names:
        values = [
            _safe_float(r.get("objective_telemetry", {}).get("components", {}).get(name, {}).get("value"), 0.0)
            for r in normalized
        ]
        if values:
            components[name] = {
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        else:
            components[name] = {"mean": 0.0, "min": 0.0, "max": 0.0}

    composite_unweighted = [
        _safe_float(r.get("objective_telemetry", {}).get("composite_proxy_unweighted"), 0.0)
        for r in normalized
    ]
    composite_regime_adjusted = [
        _safe_float(r.get("objective_telemetry", {}).get("composite_proxy_regime_adjusted"), 0.0)
        for r in normalized
    ]
    return {
        "version": OBJECTIVE_TELEMETRY_VERSION,
        "episode_count": int(len(normalized)),
        "objective_weight_config": objective_weight_config_from_config(config),
        "components": components,
        "composite_proxy_unweighted_mean": float(np.mean(composite_unweighted)) if composite_unweighted else 0.0,
        "composite_proxy_regime_adjusted_mean": (
            float(np.mean(composite_regime_adjusted)) if composite_regime_adjusted else 0.0
        ),
        "scope": "training_episode_results",
        "notes": "world_model/trade_head are direct losses; cost_turnover/regime terms are proxies pending full autograd fusion.",
    }


def sample_stochastic_calendar_extent_df(
    df: pd.DataFrame,
    min_days: int,
    max_days: int,
    min_rows: int = 256,
    strict_min_days: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Sample a stochastic calendar extent from a candle dataframe.

    This is the missing stochastic-weeks/months training distribution control:
    it slices by timestamp span (days) instead of only truncating the most recent N rows.
    """
    meta: Dict[str, Any] = {
        "mode": "disabled",
        "applied": False,
        "span_days_requested": 0,
        "span_days_actual": 0.0,
        "extent_start": None,
        "extent_end": None,
        "rows_before": int(len(df)),
        "rows_after": int(len(df)),
        "available_span_days_total": 0.0,
        "span_days_target_met": False,
        "fallback_reason": None,
    }
    if df is None or df.empty:
        meta["fallback_reason"] = "empty_df"
        return df, meta
    if "timestamp" not in df.columns:
        meta["fallback_reason"] = "missing_timestamp"
        return df, meta

    min_days_i = max(0, int(min_days))
    max_days_i = max(0, int(max_days))
    if max_days_i <= 0:
        meta["fallback_reason"] = "disabled"
        return df, meta
    if min_days_i <= 0:
        min_days_i = 1
    if max_days_i < min_days_i:
        max_days_i = min_days_i

    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    valid_mask = ts.notna()
    if not bool(valid_mask.any()):
        meta["fallback_reason"] = "no_valid_timestamps"
        return df, meta

    df_valid = df.loc[valid_mask].copy()
    ts_valid = ts.loc[valid_mask]
    if df_valid.empty:
        meta["fallback_reason"] = "valid_timestamp_rows_empty"
        return df, meta

    earliest_all = ts_valid.min()
    latest_all = ts_valid.max()
    available_span_days_total = 0.0
    if pd.notna(earliest_all) and pd.notna(latest_all):
        available_span_days_total = float((latest_all - earliest_all).total_seconds() / 86400.0)
    meta["available_span_days_total"] = available_span_days_total
    if available_span_days_total + 1e-9 < float(min_days_i):
        meta["mode"] = "calendar_days"
        meta["span_days_requested"] = int(min_days_i)
        meta["fallback_reason"] = f"available_span_below_min_days:{available_span_days_total:.3f}<{min_days_i}"
        if strict_min_days:
            return df, meta

    span_days_requested = int(np.random.randint(min_days_i, max_days_i + 1))
    meta["mode"] = "calendar_days"
    meta["span_days_requested"] = span_days_requested

    # Prefer endpoints with at least `min_days` history behind them when possible.
    earliest = ts_valid.min()
    eligible_end = ts_valid >= (earliest + pd.Timedelta(days=min_days_i))
    if bool(eligible_end.any()):
        end_candidates = ts_valid[eligible_end]
    else:
        end_candidates = ts_valid
        meta["fallback_reason"] = "insufficient_history_for_min_days"

    end_idx = int(np.random.randint(0, len(end_candidates)))
    end_ts = end_candidates.iloc[end_idx]
    start_ts = end_ts - pd.Timedelta(days=span_days_requested)

    sampled_mask = (ts_valid >= start_ts) & (ts_valid <= end_ts)
    sampled_df = df_valid.loc[sampled_mask].copy()
    if sampled_df.empty:
        meta["fallback_reason"] = "sample_empty"
        return df, meta

    min_rows_i = max(1, int(min_rows))
    if len(sampled_df) < min_rows_i:
        meta["fallback_reason"] = f"sample_rows_below_min:{len(sampled_df)}<{min_rows_i}"
        return df, meta

    sampled_ts = pd.to_datetime(sampled_df["timestamp"], errors="coerce", utc=True)
    actual_start = sampled_ts.min()
    actual_end = sampled_ts.max()
    actual_span_days = 0.0
    if pd.notna(actual_start) and pd.notna(actual_end):
        actual_span_days = float((actual_end - actual_start).total_seconds() / 86400.0)
    target_met = bool(actual_span_days + 0.5 >= float(span_days_requested))

    meta.update(
        {
            "applied": True,
            "extent_start": str(actual_start.isoformat()) if pd.notna(actual_start) else None,
            "extent_end": str(actual_end.isoformat()) if pd.notna(actual_end) else None,
            "span_days_actual": actual_span_days,
            "span_days_target_met": target_met,
            "rows_after": int(len(sampled_df)),
            "fallback_reason": None if meta.get("fallback_reason") == "disabled" else meta.get("fallback_reason"),
        }
    )
    return sampled_df, meta


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
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.candle_source = "auto"
        self.duckdb_corpus_path = str((self.data_dir / "hrm_data.duckdb").resolve())
        self.last_candle_source_used: str = "unknown"
        self.last_candle_source_detail: Optional[str] = None
        self.instrument_keys: List[str] = []
        self.context_feature_keys: List[str] = []
        self.last_feature_symbols: List[str] = []
        self.last_feature_timestamps: List[str] = []
        self.last_symbol_ranges: List[Dict[str, int]] = []
        
        # Load the 24 canonical codec experts from GOALS.md
        self.expert_classes = load_all_codecs()
        self.experts = [ExpertClass({}) for ExpertClass in self.expert_classes]
        if len(self.experts) != 24:
            print(f"⚠️  WARNING: Loaded {len(self.experts)} codec experts, expected exactly 24!")

    def configure_source(self, candle_source: str = "auto", duckdb_corpus_path: Optional[str] = None) -> None:
        self.candle_source = str(candle_source or "auto")
        if duckdb_corpus_path:
            self.duckdb_corpus_path = str(Path(duckdb_corpus_path).expanduser().resolve())

    def _duckdb_symbol_table_name(self, symbol: str) -> str:
        norm = str(symbol or "").upper().replace("-", "").replace("/", "")
        for suffix in ("USDT", "USD"):
            if norm.endswith(suffix) and len(norm) > len(suffix):
                norm = norm[: -len(suffix)]
                break
        return norm.lower()

    def _load_candles_duckdb_symbol_tables(self, con: Any, symbols: List[str], start: str, end: str) -> Tuple[List[pd.DataFrame], Optional[str]]:
        dfs: List[pd.DataFrame] = []
        where_clauses: List[str] = []
        if start:
            where_clauses.append(f"timestamp >= '{start} 00:00:00'")
        if end:
            where_clauses.append(f"timestamp <= '{end} 23:59:59'")
        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        try:
            existing_tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        except Exception:
            existing_tables = set()

        for sym in symbols:
            table = self._duckdb_symbol_table_name(sym)
            if table not in existing_tables:
                continue
            try:
                q = f"SELECT * FROM {table} {where_sql} ORDER BY timestamp ASC"
                df_sym = con.execute(q).df()
                if df_sym.empty:
                    continue
                if 'symbol' not in df_sym.columns:
                    df_sym['symbol'] = sym
                dfs.append(df_sym)
            except Exception:
                continue
        return dfs, "duckdb_symbol_tables"

    def _load_candles_duckdb_sequences_import(self, con: Any, symbols: List[str], start: str, end: str) -> Tuple[List[pd.DataFrame], Optional[str]]:
        dfs: List[pd.DataFrame] = []
        if not symbols:
            return dfs, "duckdb_sequences_import"

        symbol_sql = ", ".join([f"'{str(s)}'" for s in symbols])
        where_clauses = [f"symbol IN ({symbol_sql})"]
        if start:
            where_clauses.append(f"timestamp >= '{start} 00:00:00'")
        if end:
            where_clauses.append(f"timestamp <= '{end} 23:59:59'")
        q = (
            "SELECT * FROM binance_sequences_import "
            f"WHERE {' AND '.join(where_clauses)} "
            "ORDER BY symbol ASC, timestamp ASC"
        )
        try:
            df = con.execute(q).df()
            if not df.empty:
                dfs.append(df)
        except Exception:
            pass
        return dfs, "duckdb_sequences_import"

    def _load_candles_parquet_sequences(self, symbols: List[str], start: str, end: str) -> Tuple[List[pd.DataFrame], Optional[str]]:
        import duckdb
        dfs: List[pd.DataFrame] = []
        con = duckdb.connect(':memory:')
        try:
            for sym in symbols:
                slug = sym.replace("-", "_").replace("/", "_")
                path = self.data_dir / f"{slug}_sequences.parquet"
                if not path.exists():
                    continue
                where_clauses = []
                if start:
                    where_clauses.append(f"timestamp >= '{start} 00:00:00'")
                if end:
                    where_clauses.append(f"timestamp <= '{end} 23:59:59'")
                where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
                query = f"""
                    SELECT * FROM read_parquet('{path}')
                    {where_sql}
                    ORDER BY timestamp ASC
                """
                df_sym = con.execute(query).df()
                if df_sym.empty:
                    continue
                if 'pair' in df_sym.columns and 'symbol' not in df_sym.columns:
                    df_sym['symbol'] = df_sym['pair']
                elif 'symbol' not in df_sym.columns:
                    df_sym['symbol'] = sym
                dfs.append(df_sym)
        finally:
            con.close()
        return dfs, "parquet_sequences"

    def load_candles(self, symbols: List[str], start: str, end: str) -> pd.DataFrame:
        source_tag = str(getattr(self, "candle_source", "auto") or "auto")
        db_tag = str(getattr(self, "duckdb_corpus_path", "") or "")
        cache_key = f"candles:{source_tag}:{db_tag}:{','.join(sorted(symbols))}:{start}:{end}"

        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            import duckdb

            self.last_candle_source_used = "none"
            self.last_candle_source_detail = None

            requested_source = str(getattr(self, "candle_source", "auto") or "auto")
            source_order = {
                "auto": ["duckdb_symbol_tables", "duckdb_sequences_import", "parquet_sequences"],
                "duckdb_symbol_tables": ["duckdb_symbol_tables"],
                "duckdb_sequences_import": ["duckdb_sequences_import"],
                "parquet_sequences": ["parquet_sequences"],
            }.get(requested_source, [requested_source])

            dfs: List[pd.DataFrame] = []
            source_used: Optional[str] = None
            detail_parts: List[str] = []

            for candidate in source_order:
                if candidate.startswith("duckdb_"):
                    db_path = Path(str(getattr(self, "duckdb_corpus_path", "") or "")).expanduser()
                    if not db_path.exists():
                        detail_parts.append(f"{candidate}:missing_db")
                        continue
                    try:
                        con = duckdb.connect(str(db_path), read_only=True)
                        try:
                            if candidate == "duckdb_symbol_tables":
                                dfs, source_used = self._load_candles_duckdb_symbol_tables(con, symbols, start, end)
                            elif candidate == "duckdb_sequences_import":
                                dfs, source_used = self._load_candles_duckdb_sequences_import(con, symbols, start, end)
                            else:
                                dfs, source_used = ([], None)
                        finally:
                            con.close()
                    except Exception as e:
                        detail_parts.append(f"{candidate}:error:{type(e).__name__}")
                        dfs = []
                        source_used = None
                    if dfs:
                        break
                    detail_parts.append(f"{candidate}:empty")
                    continue

                if candidate == "parquet_sequences":
                    try:
                        dfs, source_used = self._load_candles_parquet_sequences(symbols, start, end)
                    except Exception as e:
                        detail_parts.append(f"{candidate}:error:{type(e).__name__}")
                        dfs = []
                        source_used = None
                    if dfs:
                        break
                    detail_parts.append(f"{candidate}:empty")
                    continue

            if dfs:
                df = pd.concat(dfs, ignore_index=True)
                if 'timestamp' in df.columns:
                    try:
                        df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
                    except Exception:
                        df = df.sort_values(['timestamp']).reset_index(drop=True)
                self.last_candle_source_used = str(source_used or requested_source)
                self.last_candle_source_detail = ",".join(detail_parts) if detail_parts else None
                self.cache.put(cache_key, df)
                return df
            else:
                self.last_candle_source_used = "none"
                self.last_candle_source_detail = ",".join(detail_parts) if detail_parts else None
                print(
                    f"[DEBUG] No candles loaded for {symbols} source={requested_source} "
                    f"details={self.last_candle_source_detail or 'none'}"
                )
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

        Returns an array of shape [T, n_codec_outputs * 2 + n_context + n_instruments]:
          - Channels 0   .. n_codecs-1          : signed conviction per expert per bar
          - Channels n_codecs .. 2*n_codecs-1   : tiled close-bar returns
          - Next n_context channels              : symbol/source boundary metadata
          - Remaining channels                   : named raw instrument readings
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
        row_symbols = df_enriched['symbol'].astype(str).tolist() if 'symbol' in df_enriched.columns else ["UNKNOWN"] * len(df_enriched)
        self.last_feature_symbols = row_symbols
        self.last_feature_timestamps = (
            df_enriched['timestamp'].astype(str).tolist()
            if 'timestamp' in df_enriched.columns else []
        )

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
        # We collect instrument values per bar → [T, n_instruments].
        # Important: key discovery must be UNION-based across the full stream,
        # otherwise late-populating indicators (e.g. longer lookbacks) cause
        # feature-width drift between calibration and live inference.
        instrument_rows: list = []          # list of dicts, one per bar
        instrument_key_set = set(self.instrument_keys)  # persistent schema warm-start
        symbol_change_flag = np.zeros((T, 1), dtype=np.float32)

        symbol_id_map = {
            symbol: idx for idx, symbol in enumerate(sorted(set(row_symbols)))
        } if row_symbols else {}
        symbol_counts = {
            symbol: row_symbols.count(symbol) for symbol in symbol_id_map.keys()
        }
        symbol_id_norm = np.zeros((T, 1), dtype=np.float32)
        symbol_pos_norm = np.zeros((T, 1), dtype=np.float32)

        source_col = next(
            (col for col in ('exchange', 'source', 'venue') if col in df_enriched.columns),
            None,
        )
        timeframe_col = next(
            (col for col in ('timeframe', 'interval', 'resolution') if col in df_enriched.columns),
            None,
        )
        source_values = (
            df_enriched[source_col].astype(str).tolist() if source_col else ['unknown'] * T
        )
        timeframe_values = (
            df_enriched[timeframe_col].astype(str).tolist() if timeframe_col else ['unknown'] * T
        )
        source_id_map = {v: i for i, v in enumerate(sorted(set(source_values)))}
        timeframe_id_map = {v: i for i, v in enumerate(sorted(set(timeframe_values)))}
        source_id_norm = np.zeros((T, 1), dtype=np.float32)
        timeframe_id_norm = np.zeros((T, 1), dtype=np.float32)

        # Run the full DataFrame stream through the actual expert codecs
        # Rows array for O(1) bar access (avoid iterrows overhead)
        enriched_records = df_enriched.to_dict(orient='records')

        # Lean rolling buffer — only needed for 64-bar features array (MLX models)
        return_buffer: list = []
        prev_symbol: Optional[str] = None
        pos_in_symbol = -1
        symbol_ranges: List[Dict[str, int]] = []
        range_start_idx = 0

        for i in range(T):
            current_symbol = row_symbols[i] if i < len(row_symbols) else "UNKNOWN"
            is_symbol_boundary = (i == 0) or (current_symbol != prev_symbol)
            if is_symbol_boundary:
                return_buffer = []
                pos_in_symbol = 0
                symbol_change_flag[i, 0] = 1.0
                for expert in self.experts[:num_experts]:
                    reset_runtime_state = getattr(expert, "reset_runtime_state", None)
                    if callable(reset_runtime_state):
                        reset_runtime_state()
                if i > 0 and prev_symbol is not None:
                    symbol_ranges.append({
                        'symbol': str(prev_symbol),
                        'start': int(range_start_idx),
                        'end': int(i),
                        'length': int(i - range_start_idx),
                    })
                range_start_idx = i
            else:
                pos_in_symbol += 1

            return_buffer.append(float(close_bar_returns[i]))
            if len(return_buffer) > 64:
                return_buffer = return_buffer[-64:]

            symbol_idx = symbol_id_map.get(current_symbol, 0)
            symbol_den = max(len(symbol_id_map) - 1, 1)
            symbol_id_norm[i, 0] = float(symbol_idx) / float(symbol_den)

            symbol_count = symbol_counts.get(current_symbol, 1)
            symbol_pos_norm[i, 0] = float(pos_in_symbol) / float(max(symbol_count - 1, 1))

            source_idx = source_id_map.get(source_values[i], 0)
            source_den = max(len(source_id_map) - 1, 1)
            source_id_norm[i, 0] = float(source_idx) / float(source_den)

            timeframe_idx = timeframe_id_map.get(timeframe_values[i], 0)
            timeframe_den = max(len(timeframe_id_map) - 1, 1)
            timeframe_id_norm[i, 0] = float(timeframe_idx) / float(timeframe_den)

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

            if bar_instruments:
                instrument_key_set.update(bar_instruments.keys())

            prev_symbol = current_symbol

        if T > 0:
            last_symbol = row_symbols[-1]
            symbol_ranges.append({
                'symbol': str(last_symbol),
                'start': int(range_start_idx),
                'end': int(T),
                'length': int(T - range_start_idx),
            })

        self.last_symbol_ranges = symbol_ranges
        self.context_feature_keys = [
            'ctx_symbol_change',
            'ctx_symbol_id_norm',
            'ctx_symbol_pos_norm',
            'ctx_source_id_norm',
            'ctx_timeframe_id_norm',
        ]
        context_matrix = np.concatenate(
            [
                symbol_change_flag,
                symbol_id_norm,
                symbol_pos_norm,
                source_id_norm,
                timeframe_id_norm,
            ],
            axis=1,
        ).astype(np.float32)

        # ── Assemble instrument matrix ─────────────────────────────────────
        instrument_keys = sorted(instrument_key_set)
        if instrument_keys:
            n_inst = len(instrument_keys)
            inst_matrix = np.zeros((T, n_inst), dtype=np.float32)
            for i, row in enumerate(instrument_rows):
                for j, k in enumerate(instrument_keys):
                    inst_matrix[i, j] = float(row.get(k, 0.0))
            # Store key registry on the pipeline for downstream use
            self.instrument_keys = instrument_keys
            return np.concatenate([codec_features, context_matrix, inst_matrix], axis=1)
        else:
            self.instrument_keys = []
            return np.concatenate([codec_features, context_matrix], axis=1)




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
        self.candle_pipeline.configure_source(
            candle_source=str(getattr(config, "candle_source", "auto") or "auto"),
            duckdb_corpus_path=str(getattr(config, "duckdb_corpus_path", "") or "") or None,
        )
        ob_depth_frames = 256 if config.ob_decay_mode == "hyperbolic" else 20

        # Check for model config override
        model_config_override = None
        override_path = os.environ.get('MODEL_CONFIG_OVERRIDE', '')
        if override_path and Path(override_path).exists():
            try:
                with open(override_path, 'r') as f:
                    model_config_override = json.load(f)
                print(f"[Trainer] Loaded model config override from {override_path}")
            except Exception as e:
                print(f"[Trainer] Error loading model config override: {e}")
        
        # Use command-line args or config or override
        hidden_dim = (
            getattr(config, 'hidden_dim', 64) if not model_config_override 
            else model_config_override.get('hidden_dim', 64)
        )
        regime_layers = (
            getattr(config, 'regime_layers', 2) if not model_config_override
            else model_config_override.get('regime_attn_layers', 2)
        )
        tactical_layers = (
            getattr(config, 'tactical_layers', 2) if not model_config_override
            else model_config_override.get('tactical_attn_layers', 2)
        )
        attention_heads = (
            getattr(config, 'attention_heads', 4) if not model_config_override
            else model_config_override.get('n_heads', 4)
        )
        codec_outputs = (
            max(int(getattr(config, 'codec_outputs', 24)), 1)
            if not model_config_override
            else max(int(model_config_override.get('n_codec_outputs', getattr(config, 'codec_outputs', 24))), 1)
        )
        
        self.model_config = MLXConfig(
            n_codec_outputs=codec_outputs,
            hidden_dim=hidden_dim,
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
            energy_discount_gamma=config.energy_discount_gamma,
            energy_roundtrip_cost_bps=config.energy_roundtrip_cost_bps,
            energy_churn_penalty=config.energy_churn_penalty,
            energy_target_clip=config.energy_target_clip,
            regime_attn_layers=regime_layers,
            tactical_attn_layers=tactical_layers,
            n_heads=attention_heads,
            cost_turnover_weight=config.objective_cost_turnover_weight,
            world_model_weight=config.objective_world_model_weight,
            trade_head_weight=config.objective_trade_head_weight,
        ) if HAS_MLX else None
        
        if HAS_MLX:
            print(f"[Trainer] Model config: hidden_dim={hidden_dim}, "
                  f"regime_layers={regime_layers}, tactical_layers={tactical_layers}, "
                  f"heads={attention_heads}")
            # Enable ANE/GPU optimization for better performance on Apple Silicon
            enable_ane_optimization()

        # Persistent HRM model and trainer - initialized once, trained continuously
        self.model = None
        self.trainer = None
        self._init_model_if_needed()

        # Load existing weights if provided
        if hasattr(config, 'weights_path') and config.weights_path:
            self._try_load_weights(config.weights_path)

        self.results: List[Dict] = []
        self.event_queue = queue.Queue()
        self.running = False
        self.session_start_time: str = datetime.now().isoformat()
        # Track recently Pareto-replayed episodes to avoid repeated noise injection
        self._pareto_replay_cooldown: set = set()  # episode_ids in cooldown
        self._pareto_cooldown_window: int = 100  # episodes to skip after replay
        # Signal handling for graceful checkpointing
        self._shutdown_requested = False
        self._last_checkpoint_time: float = 0.0
        self._checkpoint_interval: int = 300  # 5 minutes in seconds
        self._periodic_checkpoint_thread: Optional[threading.Thread] = None

        # Data split state (standard ML practice)
        self._train_symbols: List[str] = []
        self._val_symbols: List[str] = []
        self._test_symbols: List[str] = []
        self._train_time_range: Optional[Tuple[str, str]] = None  # (start, end)
        self._val_time_range: Optional[Tuple[str, str]] = None
        self._test_time_range: Optional[Tuple[str, str]] = None
        self._current_split: str = "train"  # train, val, or test

        # Initialize randomness based on config
        self._init_randomness()

        # Resolve pair universe once per run (file/connectome/DuckDB fallback).
        self._all_pairs_source: str = "default"
        self._all_pairs_universe: List[str] = self._resolve_pair_universe()

        # Initialize data splits
        self._init_data_splits(all_symbols=self._all_pairs_universe)

    def _init_model_if_needed(self, force_reinit: bool = False, known_input_dim: Optional[int] = None):
        """Initialize HRM model and trainer once, preserving training state across episodes."""
        if not HAS_MLX:
            return
        if (not force_reinit) and self.model is not None and self.trainer is not None:
            return  # Already initialized

        # Determine actual input dimension by running a robust signal pass
        if known_input_dim is not None:
            self.model_config.input_dim = int(known_input_dim)
            print(f"[Trainer] Reinit using known input_dim={int(known_input_dim)}.")
        else:
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
                codec_outputs = max(int(getattr(self.model_config, "n_signals", 24)), 1)
                probed_signals = self.candle_pipeline.compute_signals(dummy_df, codec_outputs)
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

    def _try_load_weights(self, weights_path: str):
        """Try to load existing weights from the specified path."""
        if not HAS_MLX:
            return
        if not weights_path:
            return
        try:
            from pathlib import Path
            weights_file = Path(weights_path)
            if not weights_file.exists():
                print(f"⚠️  Weights file not found: {weights_file}")
                return
            # Guard: verify bar_feature_proj shape matches current input_dim before loading.
            # Mismatched weights cause the cryptic MLX [addmm] ValueError that disables GPU training.
            import mlx.core as _mx
            candidate = dict(_mx.load(str(weights_file)))
            proj_key = next((k for k in candidate if 'bar_feature_proj' in k and 'weight' in k), None)
            if proj_key is not None:
                stored_in_dim = candidate[proj_key].shape[-1]
                expected_in_dim = int(self.model_config.input_dim)
                if stored_in_dim != expected_in_dim:
                    print(
                        f"⚠️  Skipping weight load: bar_feature_proj input_dim mismatch "
                        f"(stored={stored_in_dim}, expected={expected_in_dim}). "
                        "Reinitializing with fresh weights."
                    )
                    return
            self.model.load_weights(str(weights_file))
            print(f"✅ Loaded existing weights from {weights_file}")
        except Exception as e:
            print(f"⚠️  Failed to load weights from {weights_path}: {e}")

    def _init_randomness(self):
        """Initialize random number generator based on configuration."""
        if self.config.random_seed is not None:
            # Use specified seed for reproducibility
            np.random.seed(self.config.random_seed)
            print(f"[RANDOMNESS] Seeded with fixed seed: {self.config.random_seed}")
        elif self.config.use_true_randomness:
            # Use system entropy (default behavior when seed is None)
            print(f"[RANDOMNESS] Using system entropy (true randomness)")
        else:
            # Legacy behavior: seed per episode based on episode_id
            print(f"[RANDOMNESS] Using episode_id-based seeding (DEPRECATED - deterministic)")
            print(f"            Consider using --use-true-randomness for stochastic sampling")

    @staticmethod
    def _normalize_symbol(raw: str) -> str:
        sym = str(raw or "").strip().upper().replace("-", "").replace("/", "")
        return sym

    def _load_pair_universe_file(self, path: Path) -> List[str]:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except Exception:
            return []
        if not text:
            return []

        symbols: List[str] = []
        try:
            doc = json.loads(text)
            if isinstance(doc, list):
                symbols = [self._normalize_symbol(x) for x in doc]
            elif isinstance(doc, dict):
                for key in ("mapped_symbols", "symbols", "training_pairs", "pairs"):
                    value = doc.get(key)
                    if isinstance(value, list):
                        symbols.extend(self._normalize_symbol(x) for x in value)
        except Exception:
            # Plain text fallback: one symbol per line (supports connectome symbol files).
            symbols = [
                self._normalize_symbol(line.split("#", 1)[0])
                for line in text.splitlines()
            ]

        deduped: List[str] = []
        seen = set()
        for sym in symbols:
            if not sym:
                continue
            if sym in seen:
                continue
            seen.add(sym)
            deduped.append(sym)
        return deduped

    def _discover_pairs_from_duckdb(self, db_path: Path, timeframe: str = "5m") -> List[str]:
        try:
            import duckdb
        except Exception:
            return []
        if not db_path.exists():
            return []

        con = None
        try:
            con = duckdb.connect(str(db_path), read_only=True)
            tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}

            if "ohlcv" in tables:
                rows = con.execute(
                    "SELECT DISTINCT pair FROM ohlcv WHERE timeframe = ? ORDER BY pair ASC",
                    [timeframe],
                ).fetchall()
                pairs = [self._normalize_symbol(row[0]) for row in rows if row and row[0]]
                if pairs:
                    return pairs

            if "binance_sequences_import" in tables:
                rows = con.execute(
                    "SELECT DISTINCT symbol FROM binance_sequences_import ORDER BY symbol ASC"
                ).fetchall()
                pairs = [self._normalize_symbol(row[0]) for row in rows if row and row[0]]
                if pairs:
                    return pairs

            if "binance_klines" in tables:
                rows = con.execute(
                    "SELECT DISTINCT symbol FROM binance_klines WHERE timeframe = ? ORDER BY symbol ASC",
                    [timeframe],
                ).fetchall()
                pairs = [self._normalize_symbol(row[0]) for row in rows if row and row[0]]
                if pairs:
                    return pairs
        except Exception as e:
            print(f"[PAIR_UNIVERSE] DuckDB discovery failed: {e}")
        finally:
            if con is not None:
                try:
                    con.close()
                except Exception:
                    pass
        return []

    def _resolve_pair_universe(self) -> List[str]:
        required_anchors = ("BTCUSDT", "ETHUSDT", "SOLUSDT")

        def with_anchors(symbols: List[str]) -> List[str]:
            out = [self._normalize_symbol(s) for s in symbols if self._normalize_symbol(s)]
            seen = set(out)
            for anchor in required_anchors:
                if anchor not in seen:
                    out.append(anchor)
                    seen.add(anchor)
            return out

        configured = str(getattr(self.config, "pair_universe_file", "") or "").strip()
        if configured:
            path = Path(configured).expanduser()
            if path.exists():
                symbols = self._load_pair_universe_file(path)
                if symbols:
                    self._all_pairs_source = f"pair_universe_file:{path}"
                    symbols = with_anchors(symbols)
                    print(f"[PAIR_UNIVERSE] Loaded {len(symbols)} symbols from {path} (anchors=BTC,ETH,SOL)")
                    return symbols
                print(f"[PAIR_UNIVERSE] No symbols found in {path}, falling back")
            else:
                print(f"[PAIR_UNIVERSE] File missing: {path}, falling back")

        db_path_raw = str(getattr(self.config, "duckdb_corpus_path", "") or "").strip()
        if db_path_raw:
            db_path = Path(db_path_raw).expanduser()
            symbols = self._discover_pairs_from_duckdb(db_path, timeframe="5m")
            if symbols:
                self._all_pairs_source = f"duckdb:{db_path}"
                symbols = with_anchors(symbols)
                print(f"[PAIR_UNIVERSE] Discovered {len(symbols)} symbols from DuckDB {db_path} (anchors=BTC,ETH,SOL)")
                return symbols

        self._all_pairs_source = "default"
        return with_anchors(list(DEFAULT_TRAINING_PAIRS))

    def _init_data_splits(self, all_symbols: Optional[List[str]] = None):
        """
        Initialize train/val/test splits based on configuration.

        Two modes:
        1. 'symbols': Split symbols into train/val/test sets
        2. 'time': Split time periods into train/val/test sets
        """
        if all_symbols is None:
            all_symbols = list(DEFAULT_TRAINING_PAIRS)
        all_symbols = [self._normalize_symbol(s) for s in all_symbols if self._normalize_symbol(s)]
        if not all_symbols:
            all_symbols = list(DEFAULT_TRAINING_PAIRS)

        if self.config.split_mode == "symbols":
            # Split by symbols (standard ML practice)
            self._split_by_symbols(all_symbols)
        elif self.config.split_mode == "time":
            # Split by time period (standard ML practice for time series)
            self._split_by_time(all_symbols)
        else:
            raise ValueError(f"Unknown split_mode: {self.config.split_mode}")

    def _split_by_symbols(self, all_symbols: List[str]):
        """Split symbols into train/val/test sets using stratified sampling."""
        # Normalize splits
        train_ratio = float(self.config.train_split)
        val_ratio = float(self.config.val_split)
        test_ratio = float(self.config.test_split)

        # Ensure ratios sum to 1.0
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.01:
            # Auto-normalize
            train_ratio /= total
            val_ratio /= total
            test_ratio = 1.0 - train_ratio - val_ratio

        # Shuffle symbols for random split
        shuffled = all_symbols.copy()
        np.random.shuffle(shuffled)

        # Calculate split indices
        n_total = len(shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        # Assign symbols to splits
        self._train_symbols = shuffled[:n_train]
        self._val_symbols = shuffled[n_train:n_train + n_val]
        self._test_symbols = shuffled[n_train + n_val:]

        print(f"\n[DATA_SPLIT] Split mode: symbols")
        print(f"[DATA_SPLIT] Total symbols: {n_total}")
        print(f"[DATA_SPLIT] Train symbols ({len(self._train_symbols)}): {', '.join(self._train_symbols)}")
        print(f"[DATA_SPLIT] Val symbols ({len(self._val_symbols)}): {', '.join(self._val_symbols)}")
        print(f"[DATA_SPLIT] Test symbols ({len(self._test_symbols)}): {', '.join(self._test_symbols)}")

    def _split_by_time(self, all_symbols: List[str]):
        """
        Split by time period for time series validation.

        This requires loading data first to determine time range.
        The split will be done per-episode based on timestamp.
        """
        train_fraction = float(self.config.time_split_fraction)
        if train_fraction <= 0 or train_fraction >= 1.0:
            raise ValueError(f"time_split_fraction must be between 0 and 1, got {train_fraction}")

        val_fraction = (1.0 - train_fraction) / 2.0
        test_fraction = (1.0 - train_fraction) / 2.0

        self._train_symbols = all_symbols
        self._val_symbols = all_symbols  # Same symbols, different time range
        self._test_symbols = all_symbols

        print(f"\n[DATA_SPLIT] Split mode: time")
        print(f"[DATA_SPLIT] Train fraction: {train_fraction:.2%}")
        print(f"[DATA_SPLIT] Val fraction: {val_fraction:.2%}")
        print(f"[DATA_SPLIT] Test fraction: {test_fraction:.2%}")
        print(f"[DATA_SPLIT] Time ranges will be determined when loading data")

    def _get_symbols_for_split(self, split: str) -> List[str]:
        """Get symbol list for specified split (train, val, or test)."""
        if split == "train":
            return self._train_symbols
        elif split == "val":
            return self._val_symbols
        elif split == "test":
            return self._test_symbols
        else:
            raise ValueError(f"Unknown split: {split}")

    def _get_sample_pairs_for_episode(self, episode_id: int, all_pairs: List[str]) -> Tuple[List[str], str]:
        """
        Get sample pairs for an episode based on data split.

        Returns:
            tuple: (episode_pairs, split_name)
        """
        # Determine which split this episode belongs to
        if self.config.split_mode == "symbols":
            # Assign episodes to splits in round-robin fashion for balanced training
            split_cycle = episode_id % 3
            if split_cycle == 0:
                split_name = "train"
                available_pairs = self._train_symbols
            elif split_cycle == 1:
                split_name = "val"
                available_pairs = self._val_symbols
            else:
                split_name = "test"
                available_pairs = self._test_symbols
        else:
            # Time-based split: always train (time split happens at data loading)
            split_name = "train"
            available_pairs = all_pairs

        # Select pairs from the appropriate split
        pair_width = min(self.config.pair_width, len(available_pairs))
        if pair_width == 0:
            raise ValueError(f"No symbols available for split '{split_name}'")

        # Use true randomness or episode_id-based seeding
        if self.config.use_true_randomness:
            # True randomness - no seeding (system entropy)
            episode_pairs = list(np.random.choice(
                available_pairs,
                size=pair_width,
                replace=False
            ))
        else:
            # Legacy deterministic behavior
            np.random.seed(episode_id)
            episode_pairs = list(np.random.choice(
                available_pairs,
                size=pair_width,
                replace=False
            ))

        return episode_pairs, split_name

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful checkpointing on Ctrl-C."""
        def signal_handler(signum, frame):
            print(f"\n\n{'='*60}")
            print(f"🚨 Received signal {signum} (Ctrl-C) - graceful shutdown initiated")
            print(f"   Saving checkpoint before exiting...")
            print(f"{'='*60}\n")
            self._shutdown_requested = True
            # Don't raise KeyboardInterrupt - let the training loop handle it
            signal.signal(signal.SIGINT, signal.SIG_DFL)

        signal.signal(signal.SIGINT, signal_handler)

    def _start_periodic_checkpoint(self):
        """Start a background thread for periodic checkpointing every 5 minutes."""
        def checkpoint_worker():
            while self.running and not self._shutdown_requested:
                time.sleep(60)  # Check every minute
                if not self.running or self._shutdown_requested:
                    break
                elapsed = time.time() - self._last_checkpoint_time
                if elapsed >= self._checkpoint_interval:
                    print(f"\n[CHECKPOINT] Auto-save triggered after {elapsed:.0f}s...")
                    if self.results:
                        self._save_checkpoint(len(self.results))
                        self._last_checkpoint_time = time.time()

        self._periodic_checkpoint_thread = threading.Thread(target=checkpoint_worker, daemon=True)
        self._periodic_checkpoint_thread.start()

    def _save_checkpoint(self, completed_episodes: int, force_suffix: str = ""):
        """Save checkpoint with optional timestamp suffix for atomic saves."""
        serialized_results = [attach_episode_objective_telemetry(r, self.config) for r in self.results]
        self.results = serialized_results
        objective_weight_config = objective_weight_config_from_config(self.config)

        # Save HRM artifacts
        checkpoint_artifacts = self._save_hrm_artifacts(
            Path("hrm/checkpoints"),
            f"hrm_latest{force_suffix}",
        )

        # Build checkpoint metadata
        checkpoint = {
            'completed_episodes': completed_episodes,
            'total_episodes': self.config.n_epoch_episodes,
            'session_start_time': self.session_start_time,
            'checkpoint_time': datetime.now().isoformat(),
            'results': serialized_results,
            'objective_telemetry': summarize_training_objective_telemetry(serialized_results, self.config),
            'objective_weight_config': objective_weight_config,
            'hrm_artifacts': checkpoint_artifacts,
            'model_config': {
                'hidden_dim': getattr(self.model_config, 'hidden_dim', None) if self.model_config else None,
                'regime_attn_layers': getattr(self.model_config, 'regime_attn_layers', None) if self.model_config else None,
                'tactical_attn_layers': getattr(self.model_config, 'tactical_attn_layers', None) if self.model_config else None,
                'n_heads': getattr(self.model_config, 'n_heads', None) if self.model_config else None,
                'input_dim': getattr(self.model_config, 'input_dim', None) if self.model_config else None,
            },
            'data_split_config': {
                'split_mode': self.config.split_mode,
                'train_symbols': self._train_symbols,
                'val_symbols': self._val_symbols,
                'test_symbols': self._test_symbols,
                'train_split': self.config.train_split,
                'val_split': self.config.val_split,
                'test_split': self.config.test_split,
            },
            'randomness_config': {
                'use_true_randomness': self.config.use_true_randomness,
                'random_seed': self.config.random_seed,
            },
        }

        # Save checkpoint JSON (with temp file for atomic write)
        checkpoint_path = f'training_checkpoint{force_suffix}.json'
        temp_path = checkpoint_path + '.tmp'
        try:
            with open(temp_path, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            os.replace(temp_path, checkpoint_path)
            print(f"[CHECKPOINT] Saved to {checkpoint_path} (ep={completed_episodes})")
        except Exception as e:
            print(f"[CHECKPOINT] Failed to save {checkpoint_path}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

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
        debug_batch = os.environ.get("MONEYFAN_DEBUG_BATCH", "").strip().lower() in {"1", "true", "yes", "on"}
        mlx_fail_error: Optional[str] = None

        def _dbg(msg: str) -> None:
            if debug_batch:
                print(msg)

        df = self.candle_pipeline.load_candles(episode_pairs, None, None)
        extent_meta: Dict[str, Any] = {
            "mode": "none",
            "applied": False,
            "span_days_requested": 0,
            "span_days_actual": 0.0,
            "available_span_days_total": 0.0,
            "span_days_target_met": False,
            "extent_start": None,
            "extent_end": None,
            "rows_before": int(len(df)) if isinstance(df, pd.DataFrame) else 0,
            "rows_after": int(len(df)) if isinstance(df, pd.DataFrame) else 0,
            "fallback_reason": None,
        }

        if df.empty:
            return {'episode_id': episode_id, 'error': 'No data loaded'}

        # Calendar-span stochastic sampling (weeks/months) takes precedence over tail truncation.
        if (
            not df.empty
            and cached_codec_features is None
            and int(getattr(self.config, "max_extent_days", 0) or 0) > 0
        ):
            df, extent_meta = sample_stochastic_calendar_extent_df(
                df,
                int(getattr(self.config, "min_extent_days", 0) or 0),
                int(getattr(self.config, "max_extent_days", 0) or 0),
                min_rows=int(getattr(self.config, "min_extent_rows", 256) or 256),
                strict_min_days=bool(getattr(self.config, "strict_calendar_extent", False)),
            )
            if (
                bool(getattr(self.config, "strict_calendar_extent", False))
                and not bool(extent_meta.get("applied", False))
                and str(extent_meta.get("fallback_reason", "")).startswith("available_span_below_min_days:")
            ):
                return {
                    'episode_id': episode_id,
                    'error': (
                        "Calendar extent requirement unmet by corpus: "
                        f"{extent_meta.get('fallback_reason')}"
                    ),
                    'symbols': episode_pairs,
                    'extent_sampling_mode': extent_meta.get("mode"),
                    'extent_sampling_applied': bool(extent_meta.get("applied", False)),
                    'extent_span_days_requested': int(extent_meta.get("span_days_requested", 0) or 0),
                    'extent_span_days_actual': float(extent_meta.get("span_days_actual", 0.0) or 0.0),
                    'extent_available_span_days_total': float(extent_meta.get("available_span_days_total", 0.0) or 0.0),
                    'extent_span_days_target_met': bool(extent_meta.get("span_days_target_met", False)),
                    'extent_fallback_reason': extent_meta.get("fallback_reason"),
                    'candle_source_used': getattr(self.candle_pipeline, "last_candle_source_used", None),
                    'candle_source_detail': getattr(self.candle_pipeline, "last_candle_source_detail", None),
                }
        elif not df.empty and self.config.candles_per_extent != -1:
            before_rows = int(len(df))
            df = df.iloc[-self.config.candles_per_extent:]
            extent_meta = {
                "mode": "tail_rows",
                "applied": True,
                "span_days_requested": 0,
                "span_days_actual": 0.0,
                "available_span_days_total": 0.0,
                "span_days_target_met": True,
                "extent_start": None,
                "extent_end": None,
                "rows_before": before_rows,
                "rows_after": int(len(df)),
                "fallback_reason": None,
            }

        # Use cached codec features during Pareto replay, skip expensive recompute
        if cached_codec_features is not None:
            codec_features = cached_codec_features
            cached_msg = f" [cached {codec_features.shape}]"
            extent_meta = {
                "mode": "cached_codec_features",
                "applied": True,
                "span_days_requested": 0,
                "span_days_actual": 0.0,
                "available_span_days_total": 0.0,
                "span_days_target_met": True,
                "extent_start": None,
                "extent_end": None,
                "rows_before": 0,
                "rows_after": int(codec_features.shape[0]) if hasattr(codec_features, "shape") else 0,
                "fallback_reason": None,
            }
        else:
            codec_features = self.candle_pipeline.compute_signals(df, self.model_config.n_signals)
            cached_msg = ""

        # Ensure HRM model is initialized with correct input_dim
        current_input_dim = codec_features.shape[1]
        if self.model is None or getattr(self.model_config, 'input_dim', None) != current_input_dim:
            print(f"[Trainer] Recalibrating: input_dim {getattr(self.model_config, 'input_dim', 'None')} -> {current_input_dim}")
            self._init_model_if_needed(force_reinit=True, known_input_dim=current_input_dim)

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
        trade_train_eval_count = 0
        trade_train_losses: List[float] = []
        energy_train_eval_count = 0
        energy_train_losses: List[float] = []

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

        # Close-bar log returns are tiled in channels [n_codecs .. 2*n_codecs-1].
        close_return_channel = min(n_codecs, max(codec_features.shape[1] - 1, 0))
        close_bar_returns = codec_features[:, close_return_channel]
        feature_row_symbols = list(getattr(self.candle_pipeline, 'last_feature_symbols', []))
        symbol_ranges = list(getattr(self.candle_pipeline, 'last_symbol_ranges', []))

        for epoch in range(self.config.epochs):
            min_bar = max(int(self.config.min_bar_window), 1)
            max_bar = max(int(self.config.max_bar_window), min_bar)
            if max_bar <= min_bar:
                bar_window_len = min_bar
            else:
                # randint upper-bound is exclusive; include max_bar by adding 1.
                bar_window_len = int(np.random.randint(min_bar, max_bar + 1))

            hrm_memory = None

            for bar_seq_i in range(self.config.bar_sequences_per_episode):
                _dbg(f"[DEBUG] bar_seq_i: {bar_seq_i}, hrm_memory: {type(hrm_memory)}")
                if mlx_fail_error is not None:
                    break
                if bar_window_len > len(codec_features):
                    continue

                selected_range = None
                eligible_ranges = [
                    r for r in symbol_ranges
                    if int(r.get('length', 0)) >= int(bar_window_len)
                ]
                if eligible_ranges:
                    selected_range = eligible_ranges[np.random.randint(0, len(eligible_ranges))]
                    range_start = int(selected_range['start'])
                    range_end = int(selected_range['end'])
                    max_start_exclusive = range_end - bar_window_len + 1
                    if max_start_exclusive <= range_start:
                        start_idx = range_start
                    else:
                        start_idx = np.random.randint(range_start, max_start_exclusive)
                else:
                    start_idx = np.random.randint(0, len(codec_features) - bar_window_len + 1)

                batch_np = codec_features[start_idx:start_idx + bar_window_len]
                batch_np = batch_np.reshape(1, bar_window_len, -1)
                end_idx = min(start_idx + bar_window_len - 1, len(close_bar_returns) - 1)
                window_log_return = float(np.nansum(close_bar_returns[start_idx:end_idx + 1]))
                raw_move = window_log_return
                active_symbol = (
                    str(feature_row_symbols[end_idx])
                    if end_idx < len(feature_row_symbols)
                    else (str(selected_range.get('symbol')) if selected_range else (episode_pairs[0] if episode_pairs else "UNKNOWN"))
                )

                # HRM world-model training step (thread-safe, skip if previously failed)
                if not hasattr(self, '_mlx_disabled') or not self._mlx_disabled:
                    try:
                        # DEBUG: Check what batch_np is before creating MX array
                        _dbg(f"[DEBUG] batch_np type: {type(batch_np)}")
                        _dbg(f"[DEBUG] batch_np shape: {batch_np.shape if hasattr(batch_np, 'shape') else 'NO SHAPE'}")
                        if isinstance(batch_np, dict):
                            _dbg(f"[DEBUG] batch_np is a dict with keys: {batch_np.keys()}")
                        elif hasattr(batch_np, 'dtype'):
                            _dbg(f"[DEBUG] batch_np dtype: {batch_np.dtype}")

                        _dbg("[DEBUG] About to call mx.array(batch_np)...")
                        batch_mx = mx.array(batch_np)
                        _dbg(f"[DEBUG] batch_mx created successfully, type: {type(batch_mx)}")

                        # Apply coalescing to regular training if enabled

                        # Apply coalescing to regular training if enabled
                        if self.config.replay_coalescing and self.config.trade_update_prob == 0.0 and self.config.energy_update_prob == 0.0:
                            # Pretraining-only mode with coalescing enabled
                            # Skip memory persistence to reduce overhead and simplify autograd graph
                            world_model_loss, _ = trainer.pretrain_step(
                                batch_mx,
                                memory=None,  # No memory needed for pretraining-only
                                auto_eval=False,
                                clip_gradients=True,
                                max_gradient_norm=1.0
                            )
                            _dbg("[DEBUG] coalescing branch: pretrain_step returned")
                            trainer.flush_updates(world_model_loss, memory=None)
                            pretrain_eval_count += 1
                        else:
                            # Standard training with optional gradient clipping
                            use_memory = hrm_memory if (
                                self.config.trade_update_prob > 0.0 or
                                self.config.energy_update_prob > 0.0
                            ) else None

                            world_model_loss, hrm_memory = trainer.pretrain_step(
                                batch_mx,
                                memory=use_memory,
                                clip_gradients=True,
                                max_gradient_norm=1.0
                            )
                            _dbg("[DEBUG] standard branch: pretrain_step returned")
                            pretrain_eval_count += 1

                        # Extract loss value for logging
                        _dbg("[DEBUG] About to call world_model_loss.item()...")
                        loss_val = float(world_model_loss.item())
                        _dbg(f"[DEBUG] loss_val extracted: {loss_val}")
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
                                shock_perturbation_mag = min(0.08 * shock_z, 0.20)  # Capped at 20% (was 50%)
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
                                    print(f"[DEBUG] Taking REPLAY_COALESCING branch")
                                    chunk_size = max(1, int(self.config.replay_coalescing_chunk_size))
                                    chunk_size = max(1, int(self.config.replay_coalescing_chunk_size))
                                    for chunk_start in range(0, len(replay_batches_np), chunk_size):
                                        chunk = replay_batches_np[chunk_start:chunk_start + chunk_size]
                                        replay_losses = []
                                        for perturbed_bar_batch in chunk:
                                            perturbed_batch_mx = mx.array(perturbed_bar_batch)
                                            replay_loss, hrm_memory = trainer.pretrain_step(
                                                perturbed_batch_mx,
                                                memory=hrm_memory,
                                                auto_eval=False,
                                                scale=float(getattr(self.config, 'objective_regime_weight_scale', 1.0))
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
                                        _dbg(f"[DEBUG] perturbed_bar_batch type: {type(perturbed_bar_batch)}")
                                        if isinstance(perturbed_bar_batch, dict):
                                            _dbg(f"[DEBUG] perturbed_bar_batch is a dict with keys: {perturbed_bar_batch.keys()}")
                                        perturbed_batch_mx = mx.array(perturbed_bar_batch)
                                        replay_loss, hrm_memory = trainer.pretrain_step(
                                            perturbed_batch_mx,
                                            memory=hrm_memory,
                                            scale=float(getattr(self.config, 'objective_regime_weight_scale', 1.0))
                                        )
                                        replay_eval_count += 1

                    except Exception as e:
                        mlx_fail_error = (
                            f"MLX disabled on episode {episode_id}: {type(e).__name__}: {e}"
                        )
                        print(mlx_fail_error)
                        self._mlx_disabled = True
                        bar_window_losses.append(0.0)
                        if bool(getattr(self.config, "pretrain_only", False)):
                            break
                else:
                    bar_window_losses.append(0.0)
                    if bool(getattr(self.config, "pretrain_only", False)):
                        mlx_fail_error = (
                            f"MLX disabled on episode {episode_id}; "
                            "pretrain-only requires working MLX pretrain_step"
                        )
                        break

                # Compute signal density for this window (fraction of non-zero signals across active codecs)
                n_signals = max(int(getattr(self.model_config, "n_signals", 24)), 1)
                window_signals = batch_np[0, :, :n_signals]
                total_possible_signals = window_signals.size
                nonzero_count = np.count_nonzero(window_signals)
                sample_density = float(nonzero_count) / float(total_possible_signals) if total_possible_signals > 0 else 0.0

                # Back-burner "simmering" alpha updates: low-rate trade-head training
                # so the trade heads learn realized-return alignment without dominating
                # the world-model objective.
                if (
                    HAS_MLX
                    and not getattr(self, '_mlx_disabled', False)
                    and should_run_trade_step(bar_seq_i, raw_move, self.config, sample_density)
                ):
                    try:
                        realized_returns_mx = mx.array(np.array([raw_move], dtype=np.float32))
                        alpha_loss, hrm_memory = trainer.trade_step(
                            batch_mx,
                            realized_returns_mx,
                            memory=hrm_memory,
                            auto_eval=not self.config.replay_coalescing,  # Support coalescing
                            clip_gradients=True,
                            max_gradient_norm=1.0,
                        )
                        trade_train_eval_count += 1
                        trade_train_losses.append(float(alpha_loss.item()))
                    except Exception as e:
                        print(f"Trade-step skipped on episode {episode_id}: {type(e).__name__}: {e}")

                # Energy-routing autograd updates (training-only proxy):
                # teach the existing trade outputs to synthesize a discounted net-alpha score.
                if (
                    HAS_MLX
                    and not getattr(self, '_mlx_disabled', False)
                    and float(self.config.energy_update_prob) > 0.0
                    and abs(raw_move) >= float(self.config.energy_update_min_abs_return)
                    and np.random.random() < float(self.config.energy_update_prob)
                ):
                    try:
                        realized_returns_mx = mx.array(np.array([raw_move], dtype=np.float32))
                        energy_loss, hrm_memory = trainer.energy_step(
                            batch_mx,
                            realized_returns_mx,
                            memory=hrm_memory,
                            auto_eval=not self.config.replay_coalescing,  # Support coalescing
                            clip_gradients=True,
                            max_gradient_norm=1.0,
                        )
                        energy_train_eval_count += 1
                        energy_train_losses.append(float(energy_loss.item()))
                    except Exception as e:
                        print(f"Energy-step skipped on episode {episode_id}: {type(e).__name__}: {e}")

                # Trade signal execution (HRM meta-allocator action)
                if HAS_MLX and not getattr(self, '_mlx_disabled', False):
                    # Only execute a trade on some steps to simulate a sparse allocator
                    if np.random.random() > 0.5:
                        # Forward pass in trade mode to get the model's prediction (inference, no state update)
                        output_mx, _ = trainer.model.forward(batch_mx, memory=hrm_memory, mode="trade")
                        mx.eval(output_mx)
                        output_np = np.array(output_mx[0, :]) # Output is [B, 5] since it already pools the final sequence step
                        trade_intent = self._build_trade_intent(active_symbol, output_np)
                        running_peak = max(notional_curve) if notional_curve else notional
                        drawdown_pct = (notional - running_peak) / max(running_peak, 1e-8)
                        veto_decision = self._mechanical_veto(trade_intent, drawdown_pct)

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
                        candle_return = raw_move

                        position = np.sign(np.random.randn())
                        ret = position * abs(candle_return)
                        realized_pnl += ret * notional

                        if ret > 0:
                            profitable_trades += 1
                        total_trade_signals += 1

                        notional *= (1 + ret)
                        notional_curve.append(notional)

            if mlx_fail_error is not None:
                break

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

                episode_event = {
                    'episode_id': episode_id,
                    'epoch': epoch + 1,
                    'total_epochs': self.config.epochs,
                    'capital': notional,
                    'realized_pnl': notional - self.config.notional,
                    'hit_rate': profitable_trades / max(total_trade_signals, 1),
                    'total_trades': total_trade_signals,
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
                    'trade_train_eval_count': trade_train_eval_count,
                    'trade_train_loss_mean': float(np.mean(trade_train_losses)) if trade_train_losses else 0.0,
                    'trade_train_loss_last': float(trade_train_losses[-1]) if trade_train_losses else 0.0,
                    'energy_train_eval_count': energy_train_eval_count,
                    'energy_train_loss_mean': float(np.mean(energy_train_losses)) if energy_train_losses else 0.0,
                    'energy_train_loss_last': float(energy_train_losses[-1]) if energy_train_losses else 0.0,
                    'extent_sampling_mode': extent_meta.get("mode"),
                    'extent_sampling_applied': bool(extent_meta.get("applied", False)),
                    'extent_span_days_requested': int(extent_meta.get("span_days_requested", 0) or 0),
                    'extent_span_days_actual': float(extent_meta.get("span_days_actual", 0.0) or 0.0),
                    'extent_available_span_days_total': float(extent_meta.get("available_span_days_total", 0.0) or 0.0),
                    'extent_span_days_target_met': bool(extent_meta.get("span_days_target_met", False)),
                    'extent_start': extent_meta.get("extent_start"),
                    'extent_end': extent_meta.get("extent_end"),
                    'extent_rows_before': int(extent_meta.get("rows_before", 0) or 0),
                    'extent_rows_after': int(extent_meta.get("rows_after", 0) or 0),
                    'extent_fallback_reason': extent_meta.get("fallback_reason"),
                    'candle_source_used': getattr(self.candle_pipeline, "last_candle_source_used", None),
                    'candle_source_detail': getattr(self.candle_pipeline, "last_candle_source_detail", None),
                }
                self.event_queue.put(('episode_complete', attach_episode_objective_telemetry(episode_event, self.config)))

            # Flush any coalesced optimizer updates at end of epoch
            if HAS_MLX and self.config.replay_coalescing and trainer is not None and bar_window_losses:
                try:
                    last_loss = mx.array(bar_window_losses[-1])
                    trainer.flush_updates(last_loss, memory=hrm_memory)
                except Exception as e:
                    print(f"[Warning] Failed to flush updates at epoch end: {e}")

        if mlx_fail_error is not None:
            return {
                'episode_id': episode_id,
                'error': mlx_fail_error,
                'symbols': episode_pairs,
                'candle_source_used': getattr(self.candle_pipeline, "last_candle_source_used", None),
                'candle_source_detail': getattr(self.candle_pipeline, "last_candle_source_detail", None),
                'extent_sampling_mode': extent_meta.get("mode"),
                'extent_sampling_applied': bool(extent_meta.get("applied", False)),
                'extent_rows_before': int(extent_meta.get("rows_before", 0) or 0),
                'extent_rows_after': int(extent_meta.get("rows_after", 0) or 0),
            }

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
            'trade_train_eval_count': trade_train_eval_count,
            'trade_train_loss_mean': float(np.mean(trade_train_losses)) if trade_train_losses else 0.0,
            'trade_train_loss_last': float(trade_train_losses[-1]) if trade_train_losses else 0.0,
            'energy_train_eval_count': energy_train_eval_count,
            'energy_train_loss_mean': float(np.mean(energy_train_losses)) if energy_train_losses else 0.0,
            'energy_train_loss_last': float(energy_train_losses[-1]) if energy_train_losses else 0.0,
            'extent_sampling_mode': extent_meta.get("mode"),
            'extent_sampling_applied': bool(extent_meta.get("applied", False)),
            'extent_span_days_requested': int(extent_meta.get("span_days_requested", 0) or 0),
            'extent_span_days_actual': float(extent_meta.get("span_days_actual", 0.0) or 0.0),
            'extent_available_span_days_total': float(extent_meta.get("available_span_days_total", 0.0) or 0.0),
            'extent_span_days_target_met': bool(extent_meta.get("span_days_target_met", False)),
            'extent_start': extent_meta.get("extent_start"),
            'extent_end': extent_meta.get("extent_end"),
            'extent_rows_before': int(extent_meta.get("rows_before", 0) or 0),
            'extent_rows_after': int(extent_meta.get("rows_after", 0) or 0),
            'extent_fallback_reason': extent_meta.get("fallback_reason"),
            'candle_source_used': getattr(self.candle_pipeline, "last_candle_source_used", None),
            'candle_source_detail': getattr(self.candle_pipeline, "last_candle_source_detail", None),
            'equity_curve': notional_curve,
            'timestamp': datetime.now().isoformat(),
        }
        return attach_episode_objective_telemetry(result, self.config)

    def run_episode_training(self, progress_callback=None, resume_from_checkpoint: Optional[Dict] = None):
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

        # Resume from checkpoint if provided
        start_episode = 0
        if resume_from_checkpoint:
            try:
                results = resume_from_checkpoint.get('results', [])
                if results:
                    self.results = results
                    start_episode = len(results)
                    print(f"[RESUME] Restored {start_episode} results from checkpoint")
                    # Restore last checkpoint time for periodic checkpointing
                    checkpoint_time_str = resume_from_checkpoint.get('checkpoint_time')
                    if checkpoint_time_str:
                        try:
                            checkpoint_time = datetime.fromisoformat(checkpoint_time_str)
                            self._last_checkpoint_time = checkpoint_time.timestamp()
                        except Exception:
                            self._last_checkpoint_time = time.time()
            except Exception as e:
                print(f"[RESUME] Failed to restore results from checkpoint: {e}")

        # Record (and refresh) session start time when training actually begins
        self.session_start_time = datetime.now().isoformat()
        if resume_from_checkpoint:
            original_start = resume_from_checkpoint.get('session_start_time')
            if original_start:
                self.session_start_time = original_start
                print(f"[RESUME] Preserving original session start time: {original_start}")

        print(f"[SESSION] Episode training started at {self.session_start_time}")
        print(f"[OBJECTIVE] {objective_weight_config_from_config(self.config)}")

        # Set up signal handlers for graceful checkpointing
        self._setup_signal_handlers()

        # Start periodic checkpoint thread
        self._start_periodic_checkpoint()

        # Timer-based training tracking
        timer_start = time.time()
        episode_id = start_episode
        max_training_secs = getattr(self.config, 'max_training_seconds', 0) or 0

        all_pairs = list(self._all_pairs_universe) if self._all_pairs_universe else list(DEFAULT_TRAINING_PAIRS)
        if not all_pairs:
            all_pairs = list(DEFAULT_TRAINING_PAIRS)
        print(f"[PAIR_UNIVERSE] active_count={len(all_pairs)} source={self._all_pairs_source}")
        # Rebuild splits in case universe changed between trainer init and runtime.
        self._init_data_splits(all_symbols=all_pairs)

        if start_episode > 0:
            print(f"[RESUME] Starting from episode {start_episode} (skipping first {start_episode} episodes)")

        while self.running:
            # Check max training time limit
            elapsed = time.time() - timer_start
            if max_training_secs > 0 and elapsed >= max_training_secs:
                print(f"[TIMER] Reached max training time: {max_training_secs}s elapsed ({elapsed:.1f}s)")
                break

            # Check n_epoch_episodes limit if not in timer-based mode
            if not getattr(self.config, 'timer_based', False):
                if episode_id >= self.config.n_epoch_episodes:
                    break
            else:
                # In timer-based mode, report progress differently
                if episode_id > 0 and episode_id % 100 == 0:
                    print(f"[TIMER] Episode {episode_id} after {elapsed/60:.1f} minutes")
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
                # Exclude episodes in cooldown to prevent repeated noise injection
                pareto_extremes = [
                    r for r in self.results
                    if abs(r.get('z_score', 0.0)) > 1.0
                    and r.get('episode_id') not in self._pareto_replay_cooldown
                ]
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
                    # Self-adapting perturbation: stronger for wilder extremes (capped at 15%)
                    pareto_perturbation_mag = min(0.15, 0.05 * abs(z_score))
                    episode_pairs = selected['symbols']
                    # Add to cooldown to avoid repeated noise injection
                    self._pareto_replay_cooldown.add(selected['episode_id'])
                    # Clean up old cooldown entries (> 2x window ago)
                    self._pareto_replay_cooldown = {
                        eid for eid in self._pareto_replay_cooldown
                        if eid > episode_id - 2 * self._pareto_cooldown_window
                    }

                    print(
                        f"Pareto Replay [{pareto_tail_label}] episode {selected['episode_id']} "
                        f"(z={z_score:.2f}, perturb={pareto_perturbation_mag:.2f})"
                    )

            if not is_pareto_replay:
                # Use new data split logic with true stochastic sampling
                episode_pairs, split_name = self._get_sample_pairs_for_episode(episode_id, all_pairs)
                # Track which split this episode belongs to
                self._current_split = split_name

            result = self.train_episode(episode_id, episode_pairs)
            # Add split information to result
            if isinstance(result, dict):
                result['data_split'] = self._current_split

            if (
                bool(getattr(self.config, "strict_calendar_extent", False))
                and isinstance(result, dict)
                and str(result.get('error', '')).startswith("Calendar extent requirement unmet by corpus:")
            ):
                print(f"[TRAIN_ABORT] {result['error']}")
                self.results.append(result)
                self.event_queue.put(('episode_complete', result))
                if progress_callback:
                    progress_callback(episode_id + 1, self.config.n_epoch_episodes, result)
                self.running = False
                break
            if is_pareto_replay:
                result['is_replay'] = True
                result['replay_std'] = pareto_perturbation_mag

            self.results.append(result)
            self.event_queue.put(('episode_complete', result))

            if progress_callback:
                progress_callback(episode_id + 1, self.config.n_epoch_episodes, result)

            # Check for periodic checkpoint
            if self._last_checkpoint_time == 0:
                self._last_checkpoint_time = time.time()

            # Check for graceful shutdown request
            if self._shutdown_requested:
                print(f"\n{'='*60}")
                print(f"🛑 Graceful shutdown requested - saving checkpoint...")
                self._save_checkpoint(episode_id + 1, force_suffix="_interrupt")
                print(f"   Saved checkpoint at episode {episode_id + 1}")
                print(f"{'='*60}\n")
                break

            if (episode_id + 1) % 10 == 0:
                self._save_checkpoint(episode_id + 1)

        self._save_final_results()
        self.running = False
        print(f"\n[SESSION] Training completed. Total episodes: {len(self.results)}")

    def _save_hrm_artifacts(self, out_dir: Path, stem: str) -> Dict[str, Any]:
        """
        Persist the deployable HRM runtime (MLX weights + model config).

        `training_results.json` tracks episode metrics only; these artifacts are what
        the paper/live runner needs to reproduce the HRM trade heads.
        """
        if not HAS_MLX or self.model is None or self.model_config is None:
            return {
                'saved': False,
                'reason': 'mlx_or_model_unavailable',
                'objective_weight_config': objective_weight_config_from_config(self.config),
            }

        out_dir.mkdir(parents=True, exist_ok=True)
        weights_path = out_dir / f"{stem}_weights.npz"
        config_path = out_dir / f"{stem}_model_config.json"
        schema_path = out_dir / f"{stem}_feature_schema.json"
        objective_path = out_dir / f"{stem}_objective_config.json"

        try:
            self.model.save_weights(str(weights_path))
            with open(config_path, 'w') as f:
                json.dump(asdict(self.model_config), f, indent=2)
            with open(schema_path, 'w') as f:
                json.dump(
                    {
                        'instrument_keys': list(getattr(self.candle_pipeline, 'instrument_keys', []) or []),
                        'context_feature_keys': list(getattr(self.candle_pipeline, 'context_feature_keys', []) or []),
                    },
                    f,
                    indent=2,
                )
            with open(objective_path, 'w') as f:
                json.dump(objective_weight_config_from_config(self.config), f, indent=2)
            return {
                'saved': True,
                'weights_path': str(weights_path.resolve()),
                'config_path': str(config_path.resolve()),
                'feature_schema_path': str(schema_path.resolve()),
                'objective_config_path': str(objective_path.resolve()),
                'objective_weight_config': objective_weight_config_from_config(self.config),
            }
        except Exception as e:
            print(f"[Trainer] Failed to save HRM artifacts to {out_dir}: {e}")
            return {
                'saved': False,
                'reason': str(e),
                'weights_path': str(weights_path),
                'config_path': str(config_path),
                'feature_schema_path': str(schema_path),
                'objective_config_path': str(objective_path),
                'objective_weight_config': objective_weight_config_from_config(self.config),
            }

    def _save_final_results(self):
        serialized_results = [attach_episode_objective_telemetry(r, self.config) for r in self.results]
        self.results = serialized_results
        objective_weight_config = objective_weight_config_from_config(self.config)
        realized_pnls = [r['realized_pnl'] for r in serialized_results if 'realized_pnl' in r]
        hit_rates = [r['hit_rate'] for r in serialized_results if 'hit_rate' in r]
        final_capitals = [r['final_capital'] for r in serialized_results if 'final_capital' in r]
        trained_artifacts = self._save_hrm_artifacts(
            Path("models/trained"),
            "hrm_latest",
        )
        summary = {
            'total_episodes': len(serialized_results),
            'session_start_time': self.session_start_time,
            'session_end_time': datetime.now().isoformat(),
            'avg_realized_pnl': float(np.mean(realized_pnls)) if realized_pnls else 0.0,
            'avg_hit_rate': float(np.mean(hit_rates)) if hit_rates else 0.0,
            'total_notional': float(sum(final_capitals)) if final_capitals else 0.0,
            'results': serialized_results,
            'objective_telemetry': summarize_training_objective_telemetry(serialized_results, self.config),
            'objective_weight_config': objective_weight_config,
            'hrm_artifacts': trained_artifacts,
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
    parser.add_argument('--bar-sequences-per-episode', type=int, default=100,
                        help='Number of stochastic bar windows sampled per episode')
    parser.add_argument('--min-bar-window', type=int, default=64,
                        help='Minimum sampled bar window length (candles)')
    parser.add_argument('--max-bar-window', type=int, default=256,
                        help='Maximum sampled bar window length (candles)')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'lion', 'muon'],
                        help='MLX optimizer for HRM training')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='MLX optimizer learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-2,
                        help='MLX optimizer weight decay')
    parser.add_argument('--trade-update-prob', type=float, default=0.10,
                        help='Probability of a low-rate trade-head (alpha) update per sampled bar window')
    parser.add_argument('--trade-update-min-abs-return', type=float, default=0.0,
                        help='Skip trade-head updates when realized window return magnitude is below this threshold')
    parser.add_argument('--energy-update-prob', type=float, default=0.0,
                        help='Probability of a low-rate energy-routing proxy update per sampled bar window')
    parser.add_argument('--energy-update-min-abs-return', type=float, default=0.0,
                        help='Skip energy-routing updates when realized window return magnitude is below this threshold')
    parser.add_argument('--energy-discount-gamma', type=float, default=0.99,
                        help='Discount factor for energy-routing target proxy (training-only)')
    parser.add_argument('--energy-roundtrip-cost-bps', type=float, default=16.0,
                        help='Roundtrip cost (bps) used in energy-routing target proxy')
    parser.add_argument('--energy-churn-penalty', type=float, default=0.0,
                        help='Additional size-proportional churn penalty in energy-routing target proxy')
    parser.add_argument('--energy-target-clip', type=float, default=0.25,
                        help='Clip realized return target magnitude for energy-routing proxy loss')
    parser.add_argument('--pretrain-only', action='store_true',
                        help='Disable trade-head and energy-routing updates to focus on world-model pretraining')
    parser.add_argument('--fully-stochastic-pair-sampling', action='store_true',
                        help='DEPRECATED: Use --use-true-randomness instead')
    parser.add_argument('--min-extent-days', type=int, default=0,
                        help='Minimum stochastic calendar extent in days (0 disables calendar-span sampling)')
    parser.add_argument('--max-extent-days', type=int, default=0,
                        help='Maximum stochastic calendar extent in days (0 disables calendar-span sampling)')
    parser.add_argument('--candles-per-extent', type=int, default=1000,
                        help='Raw candle depth per extent used by the world-model context window')
    parser.add_argument('--ob-decay-mode', type=str, default='exponential',
                        choices=['exponential', 'hyperbolic'],
                        help='Temporal order-book decay mode for horizon compression')
    parser.add_argument('--ob-hyperbolic-tau', type=float, default=32.0,
                        help='Hyperbolic tau constant when --ob-decay-mode=hyperbolic')
    # Data split arguments (standard ML practice)
    parser.add_argument('--split-mode', type=str, default='symbols',
                        choices=['symbols', 'time'],
                        help='How to split data: "symbols" (split by symbol) or "time" (split by time period)')
    parser.add_argument('--train-split', type=float, default=0.70,
                        help='Fraction of symbols/time for training (when split_mode=symbols)')
    parser.add_argument('--val-split', type=float, default=0.15,
                        help='Fraction of symbols/time for validation')
    parser.add_argument('--test-split', type=float, default=0.15,
                        help='Fraction of symbols/time for testing (auto-calculated if 0)')
    parser.add_argument('--time-split-fraction', type=float, default=0.70,
                        help='Fraction of time period for train (when split_mode=time)')
    # Randomness arguments (standard ML practice)
    parser.add_argument('--use-true-randomness', action='store_true', default=True,
                        help='Use system entropy for true randomness (recommended)')
    parser.add_argument('--no-true-randomness', action='store_true',
                        help='Use episode_id-based seeding (DEPRECATED, deterministic)')
    parser.add_argument('--random-seed', type=int, default=None,
                        help='Fixed random seed for reproducibility (None = system entropy)')
    parser.add_argument('--min-extent-rows', type=int, default=256,
                        help='Minimum rows required after calendar-span sampling; falls back if too small')
    parser.add_argument('--strict-calendar-extent', action='store_true',
                        help='Abort training if requested min calendar span cannot be satisfied by loaded candle history')
    parser.add_argument('--candle-source', type=str, default='auto',
                        choices=['auto', 'parquet_sequences', 'duckdb_sequences_import', 'duckdb_symbol_tables'],
                        help='Candle corpus source for training data loads')
    parser.add_argument('--duckdb-corpus-path', type=str, default='',
                        help='Optional DuckDB corpus path (default: data/binance/hrm_data.duckdb)')
    parser.add_argument('--pair-universe-file', type=str, default='',
                        help='Optional symbol universe file (JSON list/object or newline-delimited symbols)')
    parser.add_argument('--codec-outputs', type=int, default=24,
                        help='Active codec outputs for training (e.g. 4 for convergence smoke, 24 for full panel)')
    parser.add_argument('--objective-world-model-weight', type=float, default=1.0,
                        help='Audit/control weight for the world-model objective term')
    parser.add_argument('--objective-trade-head-weight', type=float, default=1.0,
                        help='Audit/control weight for the trade-head (alpha) objective term')
    parser.add_argument('--objective-energy-routing-weight', type=float, default=0.0,
                        help='Audit/control weight for the energy-routing objective term')
    parser.add_argument('--objective-cost-turnover-weight', type=float, default=0.0,
                        help='Audit/control weight for the turnover/cost penalty proxy term')
    parser.add_argument('--objective-regime-weight-scale', type=float, default=1.0,
                        help='Audit/control scale for the regime-weighting proxy multiplier')
    parser.add_argument('--weights-path', type=str, default='',
                        help='Path to existing weights .npz file to resume training from')
    parser.add_argument('--resume-checkpoint', action='store_true',
                        help='Resume from training_checkpoint.json if it exists and model config matches')
    parser.add_argument('--checkpoint-file', type=str, default='training_checkpoint.json',
                        help='Checkpoint file to resume from (default: training_checkpoint.json)')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden dimension size for the HRM model')
    parser.add_argument('--regime-layers', type=int, default=2,
                        help='Number of regime attention layers')
    parser.add_argument('--tactical-layers', type=int, default=2,
                        help='Number of tactical attention layers')
    parser.add_argument('--attention-heads', type=int, default=4,
                        help='Number of attention heads')
    # Timer-based stochastic training arguments
    parser.add_argument('--timer-based', action='store_true',
                        help='Enable timer-driven episode scheduling (overrides --episodes)')
    parser.add_argument('--min-interval-seconds', type=int, default=30,
                        help='Minimum seconds between timer-based episodes')
    parser.add_argument('--max-interval-seconds', type=int, default=86400,
                        help='Maximum seconds between timer-based episodes (1 day)')
    parser.add_argument('--max-training-seconds', type=int, default=0,
                        help='Maximum total training time in seconds (0 = unlimited)')
    parser.add_argument('--min-pair-width', type=int, default=3,
                        help='Minimum number of pairs per stochastic episode')
    parser.add_argument('--max-pair-width', type=int, default=45,
                        help='Maximum number of pairs per stochastic episode')
    # Trade-step scheduling arguments
    parser.add_argument('--trade-step-schedule-mode', type=str, default='probabilistic',
                        choices=['probabilistic', 'deterministic', 'density_gated'],
                        help='Scheduling mode for trade-head updates')
    parser.add_argument('--trade-step-min-density', type=float, default=0.0,
                        help='Minimum sample density for density_gated mode (0.0 to 1.0)')
    parser.add_argument('--trade-step-schedule-interval', type=int, default=0,
                        help='Step interval for deterministic mode (0 = every step)')
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
            bar_sequences_per_episode=int(args.bar_sequences_per_episode),
            min_bar_window=int(args.min_bar_window),
            max_bar_window=int(args.max_bar_window),
            optimizer_name=args.optimizer,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            trade_update_prob=(0.0 if args.pretrain_only else float(args.trade_update_prob)),
            trade_update_min_abs_return=float(args.trade_update_min_abs_return),
            trade_step_schedule_mode=str(args.trade_step_schedule_mode),
            trade_step_min_density=float(args.trade_step_min_density),
            trade_step_schedule_interval=int(args.trade_step_schedule_interval),
            energy_update_prob=(0.0 if args.pretrain_only else float(args.energy_update_prob)),
            energy_update_min_abs_return=float(args.energy_update_min_abs_return),
            pretrain_only=bool(args.pretrain_only),
            # New randomness controls (standard ML practice)
            use_true_randomness=bool(args.use_true_randomness) if not args.no_true_randomness else False,
            random_seed=args.random_seed if args.random_seed is not None else None,
            reseed_pairs_by_episode=False,  # DEPRECATED - now controlled by use_true_randomness
            # Data split controls
            split_mode=str(args.split_mode),
            train_split=float(args.train_split),
            val_split=float(args.val_split),
            test_split=float(args.test_split) if args.test_split > 0 else 1.0 - float(args.train_split) - float(args.val_split),  # Auto-calculate if 0
            time_split_fraction=float(args.time_split_fraction),
            min_extent_days=int(args.min_extent_days),
            max_extent_days=int(args.max_extent_days),
            candles_per_extent=int(args.candles_per_extent),
            ob_decay_mode=str(args.ob_decay_mode),
            ob_hyperbolic_tau=float(args.ob_hyperbolic_tau),
            min_extent_rows=int(args.min_extent_rows),
            strict_calendar_extent=bool(args.strict_calendar_extent),
            candle_source=str(args.candle_source),
            duckdb_corpus_path=str(args.duckdb_corpus_path or ""),
            pair_universe_file=str(args.pair_universe_file or ""),
            codec_outputs=max(int(args.codec_outputs), 1),
            energy_discount_gamma=float(args.energy_discount_gamma),
            energy_roundtrip_cost_bps=float(args.energy_roundtrip_cost_bps),
            energy_churn_penalty=float(args.energy_churn_penalty),
            energy_target_clip=float(args.energy_target_clip),
            objective_world_model_weight=float(args.objective_world_model_weight),
            objective_trade_head_weight=float(args.objective_trade_head_weight),
            objective_energy_routing_weight=float(args.objective_energy_routing_weight),
            objective_cost_turnover_weight=float(args.objective_cost_turnover_weight),
            objective_regime_weight_scale=float(args.objective_regime_weight_scale),
            weights_path=str(args.weights_path or ""),
            hidden_dim=int(args.hidden_dim),
            regime_layers=int(args.regime_layers),
            tactical_layers=int(args.tactical_layers),
            attention_heads=int(args.attention_heads),
        )
        print(f"Objective weight controls: {objective_weight_config_from_config(config)}")
        randomness_mode = "true_random" if config.use_true_randomness else ("fixed_seed" if config.random_seed is not None else "deterministic")
        print(
            f"Training mode: {'PRETRAIN_ONLY' if config.pretrain_only else 'JOINT'} | "
            f"randomness={randomness_mode} | "
            f"codec_outputs={int(config.codec_outputs)} | "
            f"split_mode={config.split_mode} | "
            f"train_split={config.train_split:.2f} val_split={config.val_split:.2f} test_split={config.test_split:.2f} | "
            f"candle_source={config.candle_source} | "
            f"pair_universe_file={config.pair_universe_file or 'auto'} | "
            f"calendar_extent_days={config.min_extent_days}-{config.max_extent_days if config.max_extent_days else 0} | "
            f"calendar_extent_strict={'on' if config.strict_calendar_extent else 'off'} | "
            f"trade_update_prob={config.trade_update_prob:.3f} | energy_update_prob={config.energy_update_prob:.3f}"
        )

        # Attempt to resume from checkpoint if requested
        checkpoint_resume = None
        if args.resume_checkpoint:
            checkpoint_file = Path(args.checkpoint_file)
            if checkpoint_file.exists():
                print(f"\n[RESUME] Checking checkpoint file: {checkpoint_file}")
                try:
                    with open(checkpoint_file, 'r') as f:
                        checkpoint = json.load(f)
                    # Verify model config compatibility
                    saved_config = checkpoint.get('model_config', {})
                    if saved_config:
                        config_matches = True
                        mismatches = []
                        if saved_config.get('hidden_dim') != config.hidden_dim:
                            config_matches = False
                            mismatches.append(f"hidden_dim: {saved_config.get('hidden_dim')} != {config.hidden_dim}")
                        if saved_config.get('regime_attn_layers') != config.regime_layers:
                            config_matches = False
                            mismatches.append(f"regime_attn_layers: {saved_config.get('regime_attn_layers')} != {config.regime_layers}")
                        if saved_config.get('tactical_attn_layers') != config.tactical_layers:
                            config_matches = False
                            mismatches.append(f"tactical_attn_layers: {saved_config.get('tactical_attn_layers')} != {config.tactical_layers}")
                        if saved_config.get('n_heads') != config.attention_heads:
                            config_matches = False
                            mismatches.append(f"n_heads: {saved_config.get('n_heads')} != {config.attention_heads}")
                        if saved_config.get('input_dim') and saved_config.get('input_dim') != 92:
                            # input_dim is auto-detected, so only warn if it's set and different
                            print(f"[RESUME] Note: saved input_dim={saved_config.get('input_dim')} (will be auto-detected)")

                        if config_matches:
                            completed = checkpoint.get('completed_episodes', 0)
                            print(f"[RESUME] ✅ Config matches! Resuming from episode {completed}")
                            print(f"[RESUME]    Session started: {checkpoint.get('session_start_time')}")
                            print(f"[RESUME]    Checkpoint time: {checkpoint.get('checkpoint_time')}")
                            print(f"[RESUME]    Results count: {len(checkpoint.get('results', []))}")
                            checkpoint_resume = checkpoint
                        else:
                            print(f"[RESUME] ❌ Config mismatch - cannot resume")
                            for mismatch in mismatches:
                                print(f"         {mismatch}")
                    else:
                        print(f"[RESUME] ❌ No model config in checkpoint - cannot verify compatibility")
                except Exception as e:
                    print(f"[RESUME] ❌ Failed to load checkpoint: {e}")
            else:
                print(f"[RESUME] ❌ Checkpoint file not found: {checkpoint_file}")

        trainer = EpochEpisodeTrainer(config)

        predictor_loss_hist: List[float] = []
        trade_loss_hist: List[float] = []
        energy_loss_hist: List[float] = []

        def _rolling_mean(vals: List[float], window: int) -> float:
            if not vals:
                return 0.0
            return float(np.mean(vals[-max(1, int(window)) :]))

        def _rolling_median(vals: List[float], window: int) -> float:
            if not vals:
                return 0.0
            return float(np.median(vals[-max(1, int(window)) :]))

        def _rolling_percentile(vals: List[float], window: int, q: float) -> float:
            if not vals:
                return 0.0
            return float(np.percentile(vals[-max(1, int(window)) :], q))

        def progress(current, total, result):
            pct = (current / total) * 100
            realized_pnl = result.get('realized_pnl', 0)
            predictor_loss = float(result.get('predictor_loss', 0.0) or 0.0)
            trade_loss = float(result.get('trade_train_loss_mean', result.get('trade_train_loss_last', 0.0)) or 0.0)
            energy_loss = float(result.get('energy_train_loss_mean', result.get('energy_train_loss_last', 0.0)) or 0.0)
            predictor_loss_hist.append(predictor_loss)
            trade_loss_hist.append(trade_loss)
            energy_loss_hist.append(energy_loss)

            pred_ma5 = _rolling_mean(predictor_loss_hist, 5)
            pred_ma20 = _rolling_mean(predictor_loss_hist, 20)
            pred_med20 = _rolling_median(predictor_loss_hist, 20)
            pred_p90_20 = _rolling_percentile(predictor_loss_hist, 20, 90.0)
            trade_ma5 = _rolling_mean(trade_loss_hist, 5)
            trade_ma20 = _rolling_mean(trade_loss_hist, 20)
            energy_ma5 = _rolling_mean(energy_loss_hist, 5)
            energy_ma20 = _rolling_mean(energy_loss_hist, 20)

            trade_eval_count = int(result.get('trade_train_eval_count', 0) or 0)
            energy_eval_count = int(result.get('energy_train_eval_count', 0) or 0)
            pretrain_eval_count = int(result.get('pretrain_eval_count', 0) or 0)
            hit_rate = float(result.get('hit_rate', 0.0) or 0.0)
            extent_days = float(result.get('extent_span_days_actual', 0.0) or 0.0)
            extent_mode = str(result.get('extent_sampling_mode', 'none') or 'none')
            extent_req_days = int(result.get('extent_span_days_requested', 0) or 0)
            extent_avail_days = float(result.get('extent_available_span_days_total', 0.0) or 0.0)
            extent_target_met = bool(result.get('extent_span_days_target_met', False))
            extent_err = str(result.get('extent_fallback_reason', '') or '')
            candle_source_used = str(result.get('candle_source_used', '') or '')
            extent_suffix = ""
            if extent_req_days > 0:
                extent_suffix = f"(req={extent_req_days}d"
                if extent_avail_days > 0.0:
                    extent_suffix += f",avail={extent_avail_days:.1f}d"
                if not extent_target_met:
                    extent_suffix += ",target_miss"
                extent_suffix += ")"
            if extent_err:
                extent_suffix += f"[{extent_err}]"

            # Get data split info
            data_split = result.get('data_split', 'unknown')

            print(
                f"Episode {current}/{total} ({pct:.1f}%) "
                f"[{data_split.upper()}] "  # Show data split (TRAIN/VAL/TEST)
                f"pnl=${realized_pnl:.2f} hit={hit_rate:.1%} "
                f"src={candle_source_used or 'n/a'} extent={extent_mode}:{extent_days:.1f}d{extent_suffix} "
                f"pred_loss={predictor_loss:.4f} (ma5={pred_ma5:.4f}, ma20={pred_ma20:.4f}, med20={pred_med20:.4f}, p90_20={pred_p90_20:.4f}) "
                f"trade_loss={trade_loss:.4f} (ma5={trade_ma5:.4f}, ma20={trade_ma20:.4f}, n={trade_eval_count}) "
                f"energy_loss={energy_loss:.4f} (ma5={energy_ma5:.4f}, ma20={energy_ma20:.4f}, n={energy_eval_count}) "
                f"pretrain_steps={pretrain_eval_count}"
            )

        trainer.run_episode_training(progress_callback=progress, resume_from_checkpoint=checkpoint_resume)


if __name__ == "__main__":
    if HAS_STREAMLIT and '--dashboard' in sys.argv:
        run_dashboard()
    else:
        main()
