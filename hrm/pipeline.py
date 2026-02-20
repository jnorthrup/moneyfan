"""
HRM All-Weather Pipeline
========================

25 signals → confidence-weighted regime buckets → HRM regime weights → adaptive alpha

Architecture:
    DuckDB/Arrow candles
        → candles_df  [T × {symbol, OHLCV}]
        → features_df [T × (n_assets × 15 features)]
        → signals_df  [T × 25 models × {signal, confidence, signal_type}]
        → HRM inference → regime_weights [6 regime buckets]
        → alpha_df    [T × symbol × {alpha, regime, top_model, confidence}]
        → decisions_df → report_df

Regime Buckets (all-weather: never zero weight):
    TREND          – MACD, Momentum×Trend, SOTA, Sector Rotation
    MEAN_REVERSION – RSI, Bollinger, Grid, Mean Reversion, Pairs, Author/Kilo
    VOLATILITY     – Bollinger (vol), Volatility Breakout, Vol Signal
    STAT_ARB       – Pairs Trading (JS + Python), Bent Penny
    SYSTEMATIC     – DCA, Author's Original rebalance
    ML             – Technical ML, HRM itself

Bear market: MEAN_REVERSION + SYSTEMATIC weight rises → weather the drawdown
Bull swing:  TREND + VOLATILITY weight rises → pump priority to breakout pair

M3 Pro: PyTorch MPS backend, float32 everywhere, vectorized pandas.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Device setup (MPS → CUDA → CPU)
# ---------------------------------------------------------------------------
try:
    import torch
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    DTYPE = torch.float32  # MPS requires float32
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    DEVICE = None
    DTYPE = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_FEATURES = 15          # per asset
SEQ_LEN = 128            # HRM lookback window (hours)
REPORT_DIR = Path(__file__).parent / "data" / "reports"
DB_PATH = Path(__file__).parent / "data" / "coinbase.duckdb"

# Regime bucket names
class Regime:
    TREND          = "trend"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY     = "volatility"
    STAT_ARB       = "stat_arb"
    SYSTEMATIC     = "systematic"
    ML             = "ml"
    ALL = [TREND, MEAN_REVERSION, VOLATILITY, STAT_ARB, SYSTEMATIC, ML]


# ---------------------------------------------------------------------------
# Signal record
# ---------------------------------------------------------------------------
@dataclass
class SignalRecord:
    """One model's output at one timestep for one symbol."""
    name: str
    regime: str          # Regime bucket
    signal: float        # [-1, 1] short → long
    confidence: float    # [0, 1] certainty
    inputs_used: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 1: Ingest — DuckDB/Arrow → candles_df
# ---------------------------------------------------------------------------
def load_candles(db_path: Path = None,
                 symbols: Optional[List[str]] = None,
                 limit_hours: Optional[int] = None) -> pd.DataFrame:
    """
    Load OHLCV candles from DuckDB/Arrow.

    Returns:
        DataFrame with columns [symbol, open, high, low, close, volume]
        indexed by UTC DatetimeIndex.
    """
    try:
        from duck_store import DuckStore
    except ImportError:
        from hrm.duck_store import DuckStore
    
    store = DuckStore(str(db_path) if db_path else "hrm/data/coinbase.duckdb")
    
    if symbols is None:
        symbols = store.get_symbols()
    
    dfs = []
    for symbol in symbols:
        df = store.load(symbol)
        if len(df) > 0:
            df = df.copy()
            df['symbol'] = symbol
            dfs.append(df)
    
    if not dfs:
        raise FileNotFoundError(f"No data found in DuckDB/Arrow store")
    
    combined = pd.concat(dfs)
    
    if 'time' not in combined.columns and combined.index.name == 'time':
        combined = combined.reset_index()
    
    if 'time' in combined.columns:
        combined['time'] = pd.to_datetime(combined['time'], utc=True)
        combined = combined.set_index('time').sort_index()
    
    combined[["open", "high", "low", "close", "volume"]] = \
        combined[["open", "high", "low", "close", "volume"]].astype(np.float32)
    
    if limit_hours:
        combined = combined.tail(limit_hours)
    
    return combined


# ---------------------------------------------------------------------------
# Stage 2: Features — candles_df → features_df
# ---------------------------------------------------------------------------
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-8)
    return (100 - 100 / (1 + rs)) / 100.0  # normalised to [0, 1]


def compute_features(candles: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 15 features per (timestamp, symbol) pair.

    Input:  candles_df — long format [symbol, open, high, low, close, volume]
    Output: features_df — long format [symbol, f0..f14]

    Features per asset:
        0  returns          close.pct_change()
        1  vol_20           rolling(20) std of returns
        2  momentum_20      close.pct_change(20)
        3  rsi_14           RSI(14) normalised 0-1
        4  ma_ratio_50      close / SMA(50)
        5  ma_ratio_20      close / SMA(20)
        6  bb_pos           Bollinger Band position (0=lower, 1=upper)
        7  vwap_dev         deviation from rolling VWAP (20 bar)
        8  vol_rank_100     vol_20 percentile rank over 100 bars
        9  volume_norm      volume / SMA(volume, 20)
        10 volatility_hl    (high-low)/open — raw candle volatility
        11 breakout         2*(close-low)/(high-low)-1 — intrabar position
        12 hour_sin         sin(2π*hour/24) — time of day
        13 hour_cos         cos(2π*hour/24)
        14 dow_sin          sin(2π*dayofweek/7) — day of week
    """
    records = []

    for symbol, grp in candles.groupby("symbol", sort=False):
        g = grp.sort_index()
        c = g["close"].astype(np.float32)
        h = g["high"].astype(np.float32)
        l = g["low"].astype(np.float32)
        o = g["open"].astype(np.float32)
        v = g["volume"].astype(np.float32)

        ret = c.pct_change().astype(np.float32)
        vol20 = ret.rolling(20).std().astype(np.float32)
        sma20 = c.rolling(20).mean()
        sma50 = c.rolling(50).mean()
        bb_std = c.rolling(20).std()
        bb_upper = sma20 + 2 * bb_std
        bb_lower = sma20 - 2 * bb_std

        vwap = (c * v).rolling(20).sum() / (v.rolling(20).sum() + 1e-8)

        feat = pd.DataFrame(index=g.index)
        feat["symbol"]       = symbol
        feat["f0_returns"]   = ret
        feat["f1_vol20"]     = vol20
        feat["f2_mom20"]     = c.pct_change(20).astype(np.float32)
        feat["f3_rsi14"]     = _rsi(c, 14)
        feat["f4_ma50"]      = (c / (sma50 + 1e-8) - 1).astype(np.float32)
        feat["f5_ma20"]      = (c / (sma20 + 1e-8) - 1).astype(np.float32)
        feat["f6_bb_pos"]    = ((c - bb_lower) / (bb_upper - bb_lower + 1e-8)).clip(0, 1).astype(np.float32)
        feat["f7_vwap_dev"]  = ((c - vwap) / (vwap + 1e-8)).astype(np.float32)
        feat["f8_vol_rank"]  = vol20.rolling(100).rank(pct=True).astype(np.float32)
        feat["f9_vol_norm"]  = (v / (v.rolling(20).mean() + 1e-8)).astype(np.float32)
        feat["f10_hl_vol"]   = ((h - l) / (o + 1e-8)).astype(np.float32)
        feat["f11_breakout"] = (2 * (c - l) / (h - l + 1e-8) - 1).astype(np.float32)

        hour = g.index.hour.astype(np.float32)
        dow  = g.index.dayofweek.astype(np.float32)
        feat["f12_hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
        feat["f13_hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
        feat["f14_dow_sin"]  = np.sin(2 * np.pi * dow / 7).astype(np.float32)

        records.append(feat)

    return pd.concat(records).sort_index()


# ---------------------------------------------------------------------------
# Stage 3: Signal Catalogue — all 25 models
# ---------------------------------------------------------------------------

def _norm_conf(series: pd.Series) -> pd.Series:
    """Normalize absolute value to [0, 1] using rolling 99th percentile."""
    abs_s = series.abs()
    p99 = abs_s.rolling(200, min_periods=20).quantile(0.99).replace(0, np.nan).ffill()
    return (abs_s / p99).clip(0, 1).fillna(0).astype(np.float32)


def compute_all_signals(candles: pd.DataFrame,
                        features: pd.DataFrame) -> pd.DataFrame:
    """
    Run all 25 models over the full time × symbol space.

    Returns signals_df with columns:
        [timestamp, symbol, model, regime, signal, confidence]

    signal     ∈ [-1, 1]  (short → long)
    confidence ∈ [0,  1]  (certainty / magnitude)
    """
    rows = []

    for symbol, cgrp in candles.groupby("symbol", sort=False):
        cgrp = cgrp.sort_index()
        fgrp = features[features["symbol"] == symbol].sort_index()
        if len(cgrp) < 60:
            continue  # not enough history

        c  = cgrp["close"].astype(np.float32)
        h  = cgrp["high"].astype(np.float32)
        l_ = cgrp["low"].astype(np.float32)
        o  = cgrp["open"].astype(np.float32)
        v  = cgrp["volume"].astype(np.float32)
        ret = c.pct_change().fillna(0).astype(np.float32)
        idx = cgrp.index

        # ── TREND ──────────────────────────────────────────────────────────
        # 1. MACD Crossover
        ema12 = c.ewm(span=12).mean()
        ema26 = c.ewm(span=26).mean()
        macd  = ema12 - ema26
        sig9  = macd.ewm(span=9).mean()
        hist  = macd - sig9
        rows += _emit(idx, symbol, "macd_crossover", Regime.TREND,
                      np.tanh(hist / (hist.std() + 1e-8)), _norm_conf(hist))

        # 2. SOTA Momentum-Trailing
        mom20 = c.pct_change(20).fillna(0)
        atr14 = (h - l_).rolling(14).mean()
        sota_sig = np.tanh(mom20 / (atr20 := atr14 / (c + 1e-8) + 1e-8))
        rows += _emit(idx, symbol, "sota_momentum", Regime.TREND,
                      sota_sig, _norm_conf(mom20))

        # 3. Momentum×Trend (hrm/instruments.py)
        sma20 = c.rolling(20).mean()
        sma50 = c.rolling(50).mean()
        ma_ratio = (c / (sma50 + 1e-8)).astype(np.float32)
        trend_dir = np.sign(ma_ratio - 1)
        strength  = mom20.abs().clip(0, 1)
        rows += _emit(idx, symbol, "momentum_trend", Regime.TREND,
                      (strength * trend_dir).astype(np.float32), _norm_conf(strength))

        # 4. Sector Rotation (cross-sectional, per-symbol score vs universe mean)
        roll10 = ret.rolling(10).mean().fillna(0)
        rows += _emit(idx, symbol, "sector_rotation", Regime.TREND,
                      np.tanh(roll10 * 20), _norm_conf(roll10))

        # ── MEAN REVERSION ─────────────────────────────────────────────────
        # 5. RSI Mean Reversion
        rsi = fgrp.get("f3_rsi14", _rsi(c)) if "f3_rsi14" in fgrp.columns else _rsi(c)
        rsi_sig = -(rsi - 0.5) / 0.5  # +1 when oversold, -1 when overbought
        rows += _emit(idx, symbol, "rsi_mean_reversion", Regime.MEAN_REVERSION,
                      rsi_sig, _norm_conf(rsi_sig))

        # 6. Bollinger Band Mean Reversion
        bb_pos = fgrp["f6_bb_pos"] if "f6_bb_pos" in fgrp.columns else \
                 ((c - c.rolling(20).mean()) / (2 * c.rolling(20).std() + 1e-8)).clip(-1, 1)
        bb_sig = -np.tanh(bb_pos * 2)  # revert: buy at lower band, sell at upper
        rows += _emit(idx, symbol, "bollinger_reversion", Regime.MEAN_REVERSION,
                      bb_sig if isinstance(bb_sig, pd.Series) else pd.Series(bb_sig, index=idx),
                      _norm_conf(pd.Series(np.abs(bb_pos) if isinstance(bb_pos, np.ndarray)
                                           else bb_pos.abs(), index=idx)))

        # 7. Grid Signal (z-score reversion)
        zscore = ((c - sma20) / (c.rolling(20).std() + 1e-8)).clip(-3, 3)
        rows += _emit(idx, symbol, "grid_reversion", Regime.MEAN_REVERSION,
                      -np.tanh(zscore / 2), _norm_conf(zscore.abs()))

        # 8. HRM Mean Reversion (instruments.py)
        ma_ratio20 = (c / (sma20 + 1e-8) - 1).astype(np.float32)
        rsi_contrib = (0.5 - rsi).astype(np.float32)
        hrm_mr = (0.5 * (-np.sign(ma_ratio20)) + 0.5 * rsi_contrib * 2).astype(np.float32)
        rows += _emit(idx, symbol, "hrm_mean_reversion", Regime.MEAN_REVERSION,
                      hrm_mr, _norm_conf(hrm_mr))

        # 9. Author's Original (harvest-rebalance proxy: deviation from equal weight)
        price_norm = (c / (c.rolling(30).mean() + 1e-8) - 1).astype(np.float32)
        harvest_sig = np.where(price_norm > 0.03, -0.7,
                      np.where(price_norm < -0.04, 0.7, 0.0)).astype(np.float32)
        rows += _emit(idx, symbol, "harvest_rebalance", Regime.MEAN_REVERSION,
                      pd.Series(harvest_sig, index=idx),
                      pd.Series(np.abs(harvest_sig), index=idx))

        # 10. Kilo's Suggestion (asset-tuned thresholds, tighter)
        kilo_sig = np.where(price_norm > 0.02, -0.6,
                   np.where(price_norm < -0.02, 0.6, 0.0)).astype(np.float32)
        rows += _emit(idx, symbol, "kilo_rebalance", Regime.MEAN_REVERSION,
                      pd.Series(kilo_sig, index=idx),
                      pd.Series(np.abs(kilo_sig) * 0.8, index=idx))  # lower conf (wider params)

        # ── VOLATILITY ─────────────────────────────────────────────────────
        # 11. Volatility Breakout (PROVEN $37K — hrm/instruments.py)
        hl_vol   = ((h - l_) / (o + 1e-8)).clip(0, 1).astype(np.float32)
        intrabar = (2 * (c - l_) / (h - l_ + 1e-8) - 1).astype(np.float32)
        vb_sig   = (hl_vol * intrabar).astype(np.float32)
        rows += _emit(idx, symbol, "volatility_breakout", Regime.VOLATILITY,
                      vb_sig, _norm_conf(vb_sig))

        # 12. Bollinger Volatility (band width as position sizer)
        bb_width = (c.rolling(20).std() * 4 / (sma20 + 1e-8)).astype(np.float32)
        vol_rank = bb_width.rolling(100).rank(pct=True).fillna(0.5).astype(np.float32)
        # High vol → contrarian (revert); Low vol → trend (breakout)
        vol_regime_sig = np.where(vol_rank > 0.7, -bb_pos if isinstance(bb_pos, pd.Series)
                         else pd.Series(-bb_pos, index=idx),
                         trend_dir * strength).astype(np.float32)
        rows += _emit(idx, symbol, "bollinger_vol_regime", Regime.VOLATILITY,
                      pd.Series(vol_regime_sig, index=idx), vol_rank)

        # 13. Volatility Signal (inverse-vol sizing from signal_orchestrator.py)
        vol20 = ret.rolling(20).std().fillna(0)
        vol_rank_sig = (1 - vol_rank).astype(np.float32)  # low vol → high confidence
        rows += _emit(idx, symbol, "vol_inverse_sizing", Regime.VOLATILITY,
                      pd.Series(np.zeros(len(idx), dtype=np.float32), index=idx),  # pure sizer
                      vol_rank_sig)

        # ── STAT ARB ──────────────────────────────────────────────────────
        # 14. Bent Penny (Sharpe×Momentum edge detection)
        roll30_ret   = ret.rolling(30 * 24).mean().fillna(0)
        roll30_std   = ret.rolling(30 * 24).std().fillna(1)
        sharpe30     = (roll30_ret / (roll30_std + 1e-8)).clip(-3, 3)
        bend         = (sharpe30 * 0.5 + np.tanh(mom20 * 10) * 0.5).astype(np.float32)
        rows += _emit(idx, symbol, "bent_penny", Regime.STAT_ARB,
                      bend, _norm_conf(bend))

        # 15. Pairs Trading proxy (z-score vs rolling BTC correlation)
        # Each symbol vs its own 30-day mean — full cross-pair needs multi-symbol pass
        pairs_zscore = ((c - c.rolling(30 * 24).mean()) /
                        (c.rolling(30 * 24).std() + 1e-8)).clip(-3, 3).astype(np.float32)
        rows += _emit(idx, symbol, "pairs_spread", Regime.STAT_ARB,
                      -np.tanh(pairs_zscore / 1.5), _norm_conf(pairs_zscore.abs()))

        # ── SYSTEMATIC ─────────────────────────────────────────────────────
        # 16. Dollar Cost Averaging (always-on baseline long bias, low confidence)
        dca_sig = pd.Series(np.full(len(idx), 0.3, dtype=np.float32), index=idx)
        dca_conf = pd.Series(np.full(len(idx), 0.2, dtype=np.float32), index=idx)
        rows += _emit(idx, symbol, "dca_baseline", Regime.SYSTEMATIC, dca_sig, dca_conf)

        # 17. Author rebalance cadence (7-day cycle bias)
        cycle7 = pd.Series(
            np.sin(2 * np.pi * np.arange(len(idx)) / (7 * 24)).astype(np.float32),
            index=idx)
        rows += _emit(idx, symbol, "weekly_cadence", Regime.SYSTEMATIC, cycle7 * 0.2,
                      pd.Series(np.full(len(idx), 0.15, dtype=np.float32), index=idx))

        # ── ML ─────────────────────────────────────────────────────────────
        # 18. Technical ML (11-feature linear model, weights learned via online ridge)
        feat_cols = [col for col in fgrp.columns if col.startswith("f") and col != "symbol"]
        if len(feat_cols) >= 8 and len(fgrp) > 0:
            fmat = fgrp[feat_cols].fillna(0).astype(np.float32)
            # Random projection weights (placeholder — replace with trained weights)
            rng = np.random.default_rng(42)
            w   = rng.standard_normal(len(feat_cols)).astype(np.float32)
            raw = fmat.values @ w
            ml_sig = pd.Series(np.tanh(raw), index=fgrp.index).reindex(idx).fillna(0)
            ml_conf = _norm_conf(ml_sig.abs())
        else:
            ml_sig = pd.Series(np.zeros(len(idx), dtype=np.float32), index=idx)
            ml_conf = pd.Series(np.zeros(len(idx), dtype=np.float32), index=idx)
        rows += _emit(idx, symbol, "technical_ml", Regime.ML, ml_sig, ml_conf)

        # 19-25: Composition signals from signal_synergies.json
        # Grid × Trend (multiply)
        grid_s  = -np.tanh(zscore / 2)
        trend_s = np.tanh((sma20 - sma50) / (sma50 + 1e-8) * 20)
        gt_mult = (grid_s * trend_s).astype(np.float32)
        rows += _emit(idx, symbol, "grid_x_trend", Regime.MEAN_REVERSION,
                      gt_mult, _norm_conf(gt_mult))

        # RSI × Trend (multiply)
        rt_mult = (rsi_sig * trend_s).astype(np.float32) \
            if isinstance(rsi_sig, pd.Series) else \
            pd.Series((rsi_sig * trend_s).values.astype(np.float32), index=idx)
        rows += _emit(idx, symbol, "rsi_x_trend", Regime.MEAN_REVERSION,
                      rt_mult, _norm_conf(rt_mult))

        # Momentum × Volatility (multiply)
        mv_mult = (np.tanh(mom20 * 10) * hl_vol).astype(np.float32)
        rows += _emit(idx, symbol, "momentum_x_vol", Regime.VOLATILITY,
                      mv_mult, _norm_conf(mv_mult))

        # Volatility × Breakout (proven multiply — Sharpe 4.6)
        vxb = (hl_vol * intrabar).astype(np.float32)  # same as volatility_breakout
        rows += _emit(idx, symbol, "vol_x_breakout_proven", Regime.VOLATILITY,
                      vxb, _norm_conf(vxb) * 1.0)  # full conf — proven model

        # Momentum × Trend (additive 60/40)
        mt_add = (0.6 * np.tanh(mom20 * 10) + 0.4 * trend_s).astype(np.float32)
        rows += _emit(idx, symbol, "mom_trend_additive", Regime.TREND,
                      mt_add, _norm_conf(mt_add))

        # RSI + Trend (additive 70/30)
        rsi_arr = rsi_sig.values if isinstance(rsi_sig, pd.Series) else rsi_sig
        rt_add = (0.7 * rsi_arr + 0.3 * trend_s.values).astype(np.float32)
        rows += _emit(idx, symbol, "rsi_trend_additive", Regime.MEAN_REVERSION,
                      pd.Series(rt_add, index=idx), _norm_conf(pd.Series(rt_add, index=idx)))

        # MACD + Momentum (dual confirmation)
        hist_norm = np.tanh(hist / (hist.std() + 1e-8))
        mc_dual = (0.5 * hist_norm + 0.5 * np.tanh(mom20 * 10)).astype(np.float32)
        rows += _emit(idx, symbol, "macd_momentum_dual", Regime.TREND,
                      mc_dual, _norm_conf(mc_dual))

    if not rows:
        return pd.DataFrame(columns=["symbol", "model", "regime", "signal", "confidence"])

    return pd.DataFrame(rows).sort_values(["timestamp", "symbol", "model"])


def _emit(idx: pd.DatetimeIndex, symbol: str, model: str, regime: str,
          signal: pd.Series, confidence: pd.Series) -> List[dict]:
    """Convert parallel Series to list of dicts for concat."""
    if not isinstance(signal, pd.Series):
        signal = pd.Series(signal, index=idx)
    if not isinstance(confidence, pd.Series):
        confidence = pd.Series(confidence, index=idx)
    signal = signal.reindex(idx).fillna(0).clip(-1, 1).astype(np.float32)
    confidence = confidence.reindex(idx).fillna(0).clip(0, 1).astype(np.float32)
    return [
        {"timestamp": t, "symbol": symbol, "model": model,
         "regime": regime, "signal": float(s), "confidence": float(cf)}
        for t, s, cf in zip(idx, signal, confidence)
    ]


# ---------------------------------------------------------------------------
# Stage 4: Regime aggregation → alpha
# ---------------------------------------------------------------------------

def compute_regime_alphas(signals_df: pd.DataFrame,
                          regime_weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Aggregate per-model signals into per-regime alpha using confidence weighting.

    If regime_weights is None, uses equal weighting (1/6 each — all-weather baseline).
    HRM inference provides these weights when a checkpoint is available.

    Returns alpha_df:
        [timestamp, symbol, regime, regime_alpha, regime_confidence,
         top_model, top_signal, top_confidence]
    """
    if regime_weights is None:
        regime_weights = {r: 1.0 / len(Regime.ALL) for r in Regime.ALL}

    # Normalize weights
    total = sum(regime_weights.values()) + 1e-8
    regime_weights = {k: v / total for k, v in regime_weights.items()}

    records = []
    grouped = signals_df.groupby(["timestamp", "symbol", "regime"])

    for (ts, sym, regime), grp in grouped:
        conf_sum = grp["confidence"].sum()
        if conf_sum < 1e-8:
            regime_alpha = 0.0
            regime_conf  = 0.0
        else:
            # Confidence-weighted average signal
            regime_alpha = float((grp["signal"] * grp["confidence"]).sum() / conf_sum)
            regime_conf  = float(grp["confidence"].mean())

        top_row = grp.loc[grp["confidence"].idxmax()]
        records.append({
            "timestamp":        ts,
            "symbol":           sym,
            "regime":           regime,
            "regime_alpha":     regime_alpha,
            "regime_confidence": regime_conf,
            "regime_weight":    regime_weights.get(regime, 0.0),
            "top_model":        top_row["model"],
            "top_signal":       float(top_row["signal"]),
            "top_confidence":   float(top_row["confidence"]),
        })

    return pd.DataFrame(records)


def compute_final_alpha(regime_alphas: pd.DataFrame) -> pd.DataFrame:
    """
    Combine regime alphas → single alpha per (timestamp, symbol).

    alpha = Σ regime_weight × regime_alpha × regime_confidence

    Returns decisions_df:
        [timestamp, symbol, alpha, dominant_regime, dominant_model,
         bull_score, bear_score, regime_breakdown (dict)]
    """
    records = []
    for (ts, sym), grp in regime_alphas.groupby(["timestamp", "symbol"]):
        weighted = grp["regime_weight"] * grp["regime_alpha"] * (1 + grp["regime_confidence"])
        alpha = float(weighted.sum() / (grp["regime_weight"].sum() + 1e-8))

        dom = grp.loc[(grp["regime_weight"] * grp["regime_confidence"]).idxmax()]

        # Bull/bear decomposition
        bull_regimes = [Regime.TREND, Regime.VOLATILITY, Regime.ML]
        bear_regimes = [Regime.MEAN_REVERSION, Regime.STAT_ARB]
        bull_score = float(grp[grp["regime"].isin(bull_regimes)]["regime_alpha"].mean())
        bear_score = float(grp[grp["regime"].isin(bear_regimes)]["regime_alpha"].mean())

        records.append({
            "timestamp":       ts,
            "symbol":          sym,
            "alpha":           np.clip(alpha, -1, 1),
            "dominant_regime": dom["regime"],
            "dominant_model":  dom["top_model"],
            "bull_score":      bull_score,
            "bear_score":      bear_score,
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df["alpha"] = df["alpha"].astype(np.float32)
        df["bull_score"] = df["bull_score"].astype(np.float32)
        df["bear_score"] = df["bear_score"].astype(np.float32)
    return df


# ---------------------------------------------------------------------------
# Stage 5: HRM regime weight inference (PyTorch MPS)
# ---------------------------------------------------------------------------

def hrm_regime_weights(features_df: pd.DataFrame,
                        checkpoint_path: Optional[Path] = None,
                        symbols: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Run HRM neural net to produce regime weights.

    Without a checkpoint → returns softmax uniform weights (equal all-weather).
    With a checkpoint → loads HRM and infers regime weights from feature tensor.

    The 6 regime buckets are HRM's output classes.
    """
    if not HAS_TORCH:
        return {r: 1.0 / len(Regime.ALL) for r in Regime.ALL}

    if checkpoint_path is None or not checkpoint_path.exists():
        # Uniform baseline — all-weather equal weight
        return {r: 1.0 / len(Regime.ALL) for r in Regime.ALL}

    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model = checkpoint.get("model")
        if model is None:
            return {r: 1.0 / len(Regime.ALL) for r in Regime.ALL}

        model = model.to(DEVICE).eval()

        # Assemble feature tensor: [1, SEQ_LEN, n_assets × N_FEATURES]
        feat_cols = [c for c in features_df.columns if c.startswith("f") and c != "symbol"]
        pivot = features_df.pivot_table(index=features_df.index,
                                         columns="symbol",
                                         values=feat_cols)
        pivot = pivot.fillna(0).astype(np.float32)
        tail = pivot.iloc[-SEQ_LEN:].values  # [seq_len, n_assets * n_features]

        x = torch.tensor(tail, dtype=DTYPE, device=DEVICE).unsqueeze(0)  # [1, T, D]

        with torch.no_grad():
            logits = model(x)  # expect [1, 6] — one logit per regime
            weights = torch.softmax(logits.squeeze(0), dim=-1).cpu().numpy()

        return dict(zip(Regime.ALL, weights.tolist()))

    except Exception as e:
        print(f"[HRM] Checkpoint inference failed ({e}), using uniform weights.")
        return {r: 1.0 / len(Regime.ALL) for r in Regime.ALL}


# ---------------------------------------------------------------------------
# Stage 6: Report
# ---------------------------------------------------------------------------

def compute_report(decisions: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    """
    Build report_df with per-symbol P&L, Sharpe, model attribution.

    Decisions → alpha signal → simulated position → P&L
    """
    records = []

    for symbol, dgrp in decisions.groupby("symbol"):
        cgrp = candles[candles["symbol"] == symbol].sort_index()
        dgrp = dgrp.set_index("timestamp").sort_index()

        # Align
        aligned = dgrp.join(cgrp[["close"]], how="inner")
        if len(aligned) < 2:
            continue

        fwd_ret = aligned["close"].pct_change().shift(-1).fillna(0).astype(np.float32)
        pnl     = (aligned["alpha"] * fwd_ret).astype(np.float32)
        cum_pnl = pnl.cumsum()
        roll7   = pnl.rolling(7 * 24).mean() / (pnl.rolling(7 * 24).std() + 1e-8)

        model_counts = dgrp["dominant_model"].value_counts()
        top_model = model_counts.index[0] if len(model_counts) else "unknown"

        records.append({
            "symbol":        symbol,
            "total_pnl":     float(cum_pnl.iloc[-1]) if len(cum_pnl) else 0.0,
            "sharpe_7d":     float(roll7.iloc[-1]) if len(roll7) else 0.0,
            "top_model":     top_model,
            "bull_exposure": float(aligned["bull_score"].mean()),
            "bear_exposure": float(aligned["bear_score"].mean()),
            "dominant_regime": dgrp["dominant_regime"].mode().iloc[0]
                               if len(dgrp) else "unknown",
            "n_bars":        len(aligned),
        })

    df = pd.DataFrame(records).sort_values("sharpe_7d", ascending=False)
    return df


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_pipeline(db_path: Path = DB_PATH,
                 symbols: Optional[List[str]] = None,
                 checkpoint_path: Optional[Path] = None,
                 limit_hours: Optional[int] = None,
                 verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: candles → signals → regime alpha → report.

    Returns: (decisions_df, report_df, signals_df)

    Bear market mode: MEAN_REVERSION + STAT_ARB + SYSTEMATIC dominate.
    Bull swing mode:  TREND + VOLATILITY dominate → top breakout pair surfaces.
    HRM regime weights are adaptive — checkpoint required for non-uniform.
    """
    def log(msg: str):
        if verbose:
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(f"[{ts}] {msg}")

    log(f"Loading candles from {db_path} (device: {DEVICE})")
    candles = load_candles(db_path, symbols=symbols, limit_hours=limit_hours)
    log(f"  {len(candles):,} rows × {candles['symbol'].nunique()} symbols")

    log("Computing features (15 per asset)...")
    features = compute_features(candles)
    log(f"  {len(features):,} feature rows")

    log("Running 25 signal models...")
    signals = compute_all_signals(candles, features)
    log(f"  {len(signals):,} signal records, {signals['model'].nunique()} models")

    log("Inferring HRM regime weights...")
    regime_weights = hrm_regime_weights(features, checkpoint_path=checkpoint_path,
                                        symbols=symbols)
    log(f"  Regime weights: { {k: f'{v:.3f}' for k, v in regime_weights.items()} }")

    log("Aggregating regime alphas...")
    regime_alphas = compute_regime_alphas(signals, regime_weights)

    log("Computing final alpha per symbol...")
    decisions_all = compute_final_alpha(regime_alphas)
    # Latest bar per symbol for display
    decisions = (decisions_all.sort_values("timestamp")
                               .groupby("symbol", as_index=False)
                               .last())

    log("Building report...")
    report = compute_report(decisions_all, candles)

    if verbose:
        print("\n── Report (top 10 by Sharpe 7d) ──────────────────────────────")
        print(report.head(10).to_string(index=False))
        print()
        bull = decisions.groupby("symbol")["bull_score"].mean().sort_values(ascending=False)
        print(f"── Top bull-biased pairs (swing long candidates) ──────────────")
        print(bull.head(5).to_string())
        print()
        bear_hedge = decisions.groupby("symbol")["bear_score"].mean().sort_values(ascending=False)
        print(f"── Top bear-biased pairs (mean-reversion / hedge candidates) ──")
        print(bear_hedge.head(5).to_string())

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    out_path = REPORT_DIR / f"pipeline_{today}.parquet"
    decisions.to_parquet(out_path, index=False)
    log(f"Decisions saved → {out_path}")

    return decisions, report, signals


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="HRM All-Weather Pipeline")
    ap.add_argument("--db",         default=str(DB_PATH), help="SQLite DB path")
    ap.add_argument("--symbols",    nargs="*",            help="Symbol filter (e.g. BTC-USD ETH-USD)")
    ap.add_argument("--checkpoint", default=None,         help="HRM checkpoint path (.pt)")
    ap.add_argument("--limit",      type=int, default=None, help="Limit to N most recent hours")
    ap.add_argument("--quiet",      action="store_true",  help="Suppress progress output")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint) if args.checkpoint else None
    run_pipeline(
        db_path=Path(args.db),
        symbols=args.symbols,
        checkpoint_path=ckpt,
        limit_hours=args.limit,
        verbose=not args.quiet,
    )
