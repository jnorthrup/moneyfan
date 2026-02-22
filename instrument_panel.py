"""
instrument_panel.py — Lazy Vectorized Indicator Panel
========================================================

Each indicator family is a @cached_property — computed only on first access,
cached for subsequent reads.  `compute()` accepts an optional list of families
so callers can request only what they need (or pass None to compute all).

Design: pandas → instruments (lazy) → {codecs} → HRM IO  (GOALS.md draw-thru)
"""

import numpy as np
import pandas as pd
from functools import cached_property
from typing import Optional, List


# ── Public lazy accessor ────────────────────────────────────────────────────────

class InstrumentPanel:
    """
    Lazy indicator panel over a pandas OHLCV DataFrame.

    Each indicator family is a @cached_property — the computation runs once
    on first access and is cached, never re-run.  Families not accessed by
    any codec are never computed.

    Usage::

        panel = InstrumentPanel(df)
        df_enriched = panel.compute()          # compute all families
        df_enriched = panel.compute(['rsi', 'macd', 'atr'])  # only what you need
    """

    def __init__(self, df: pd.DataFrame):
        # Store a sorted, clean copy once — all property access reads from it
        self._df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True).copy()
        if 'log_return' not in self._df.columns:
            self._df['log_return'] = np.log(
                self._df['close'] / self._df['close'].shift(1)
            ).fillna(0.0)
        if 'volume' not in self._df.columns:
            self._df['volume'] = 1.0

    # ── Lazy indicator families ─────────────────────────────────────────────────

    @cached_property
    def returns_momentum(self) -> pd.DataFrame:
        """Log/pct returns + momentum at 3/5/10/20/30/60 bars."""
        df = self._df
        out = pd.DataFrame(index=df.index)
        out['pct_return']  = df['close'].pct_change().fillna(0.0)
        out['momentum_3']  = df['close'].pct_change(3).fillna(0.0)
        out['momentum_5']  = df['close'].pct_change(5).fillna(0.0)
        out['momentum_10'] = df['close'].pct_change(10).fillna(0.0)
        out['momentum_20'] = df['close'].pct_change(20).fillna(0.0)
        out['momentum_30'] = df['close'].pct_change(30).fillna(0.0)
        out['momentum_60'] = df['close'].pct_change(60).fillna(0.0)
        return out

    @cached_property
    def ema_macd(self) -> pd.DataFrame:
        """EMA stack 5/10/12/20/26/50 + MACD line/signal/histogram."""
        c = self._df['close']
        out = pd.DataFrame(index=self._df.index)
        for span in [5, 10, 12, 20, 26, 50]:
            out[f'ema_{span}'] = c.ewm(span=span, adjust=False).mean()
        out['ema_ratio_5_20']  = out['ema_5']  / (out['ema_20']  + 1e-8)
        out['ema_ratio_12_26'] = out['ema_12'] / (out['ema_26']  + 1e-8)
        out['macd_line']   = out['ema_12'] - out['ema_26']
        out['macd_signal'] = out['macd_line'].ewm(span=9, adjust=False).mean()
        out['macd_hist']   = out['macd_line'] - out['macd_signal']
        return out

    @cached_property
    def rsi(self) -> pd.DataFrame:
        """RSI(14) with Wilder's smoothing."""
        delta = self._df['close'].diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rs    = gain / (loss + 1e-8)
        out = pd.DataFrame(index=self._df.index)
        out['rsi_14'] = (100 - 100 / (1 + rs)).fillna(50.0)
        return out

    @cached_property
    def bollinger(self) -> pd.DataFrame:
        """Bollinger Bands(20,2): upper/lower/mid/%B/width/squeeze flag."""
        c   = self._df['close']
        sma = c.rolling(20, min_periods=1).mean()
        std = c.rolling(20, min_periods=1).std().fillna(1e-8)
        out = pd.DataFrame(index=self._df.index)
        out['bb_upper']   = sma + 2 * std
        out['bb_lower']   = sma - 2 * std
        out['bb_mid']     = sma
        out['bb_pct_b']   = (c - out['bb_lower']) / (out['bb_upper'] - out['bb_lower'] + 1e-8)
        out['bb_width']   = (out['bb_upper'] - out['bb_lower']) / (sma + 1e-8)
        out['bb_squeeze'] = (out['bb_width'] < out['bb_width'].rolling(20, min_periods=1).mean()).astype(float)
        return out

    @cached_property
    def atr(self) -> pd.DataFrame:
        """True Range + ATR(14) + normalised ATR."""
        df  = self._df
        pc  = df['close'].shift(1)
        tr  = pd.concat([
            df['high'] - df['low'],
            (df['high'] - pc).abs(),
            (df['low']  - pc).abs(),
        ], axis=1).max(axis=1)
        out = pd.DataFrame(index=df.index)
        out['tr']      = tr
        out['atr_14']  = tr.ewm(alpha=1/14, adjust=False).mean().fillna(tr)
        out['atr_norm'] = out['atr_14'] / (df['close'] + 1e-8)
        return out

    @cached_property
    def stochastic(self) -> pd.DataFrame:
        """Stochastic %K(14) and %D(3)."""
        df = self._df
        lo = df['low'].rolling(14, min_periods=1).min()
        hi = df['high'].rolling(14, min_periods=1).max()
        out = pd.DataFrame(index=df.index)
        out['stoch_k'] = 100 * (df['close'] - lo) / (hi - lo + 1e-8)
        out['stoch_d'] = out['stoch_k'].rolling(3, min_periods=1).mean()
        return out

    @cached_property
    def adx(self) -> pd.DataFrame:
        """ADX(14) + DI+/DI- directional movement system."""
        df   = self._df
        ph   = df['high'].shift(1)
        pl   = df['low'].shift(1)
        pc   = df['close'].shift(1)
        period = 14

        dm_p = np.where((df['high'] - ph) > (pl - df['low']),
                        np.maximum(df['high'] - ph, 0.0), 0.0)
        dm_m = np.where((pl - df['low']) > (df['high'] - ph),
                        np.maximum(pl - df['low'], 0.0), 0.0)
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - pc).abs(),
            (df['low']  - pc).abs(),
        ], axis=1).max(axis=1)

        atr_s  = tr.ewm(alpha=1/period, adjust=False).mean()
        dmp_s  = pd.Series(dm_p, index=df.index).ewm(alpha=1/period, adjust=False).mean()
        dmm_s  = pd.Series(dm_m, index=df.index).ewm(alpha=1/period, adjust=False).mean()
        di_p   = 100 * dmp_s / (atr_s + 1e-8)
        di_m   = 100 * dmm_s / (atr_s + 1e-8)
        dx     = 100 * (di_p - di_m).abs() / (di_p + di_m + 1e-8)

        out = pd.DataFrame(index=df.index)
        out['adx']      = dx.ewm(alpha=1/period, adjust=False).mean().fillna(0.0)
        out['plus_di']  = di_p.fillna(0.0)
        out['minus_di'] = di_m.fillna(0.0)
        return out

    @cached_property
    def vwap(self) -> pd.DataFrame:
        """Rolling VWAP(20) and deviation from VWAP."""
        df  = self._df
        tp  = (df['high'] + df['low'] + df['close']) / 3.0
        vol = df['volume'].replace(0, 1.0)
        out = pd.DataFrame(index=df.index)
        out['vwap']     = (tp * vol).rolling(20, min_periods=1).sum() / vol.rolling(20, min_periods=1).sum()
        out['vwap_dev'] = (df['close'] - out['vwap']) / (out['vwap'] + 1e-8)
        return out

    @cached_property
    def zscore(self) -> pd.DataFrame:
        """Rolling z-scores at 10/20/60 bars."""
        c = self._df['close']
        out = pd.DataFrame(index=self._df.index)
        for w in [10, 20, 60]:
            mu  = c.rolling(w, min_periods=2).mean()
            std = c.rolling(w, min_periods=2).std().fillna(1.0)
            out[f'zscore_{w}'] = ((c - mu) / (std + 1e-8)).fillna(0.0)
        return out

    @cached_property
    def volatility(self) -> pd.DataFrame:
        """Realised volatility at 5/10/20/60 bars + ratio."""
        lr = self._df['log_return']
        out = pd.DataFrame(index=self._df.index)
        for w in [5, 10, 20, 60]:
            out[f'vol_{w}'] = lr.rolling(w, min_periods=2).std().fillna(0.0)
        out['vol_ratio_20_60'] = out['vol_20'] / (out['vol_60'] + 1e-8)
        return out

    @cached_property
    def donchian(self) -> pd.DataFrame:
        """Donchian channel (20) and position within channel."""
        df = self._df
        ch_max = df['high'].rolling(20, min_periods=1).max()
        ch_min = df['low'].rolling(20, min_periods=1).min()
        out = pd.DataFrame(index=df.index)
        out['donchian_upper'] = ch_max
        out['donchian_lower'] = ch_min
        out['donchian_pos']   = (df['close'] - ch_min) / (ch_max - ch_min + 1e-8)
        return out

    @cached_property
    def volume_flow(self) -> pd.DataFrame:
        """OBV proxy, volume ratio, bar delta (order-flow proxy), CVD(20)."""
        df  = self._df
        vol = df['volume'].replace(0, 1.0)
        ret = self._df['log_return']
        sign_ret = np.sign(ret)
        out = pd.DataFrame(index=df.index)
        out['vol_sma_20']  = vol.rolling(20, min_periods=1).mean()
        out['vol_ratio']   = vol / (out['vol_sma_20'] + 1e-8)
        out['obv_proxy']   = (sign_ret * vol).cumsum()
        out['obv_signal']  = out['obv_proxy'].ewm(span=10, adjust=False).mean()
        out['obv_hist']    = out['obv_proxy'] - out['obv_signal']
        pc  = df['close'].shift(1).fillna(df['close'])
        rng = (df['high'] - df['low']).clip(lower=1e-8)
        out['bar_delta']   = (df['close'] - pc) / rng
        out['cvd_20']      = out['bar_delta'].rolling(20, min_periods=1).sum()
        return out

    @cached_property
    def spread_proxy(self) -> pd.DataFrame:
        """EMA-channel spread + grid band position (pairs-trading & grid proxies)."""
        c = self._df['close']
        out = pd.DataFrame(index=self._df.index)
        ef = c.ewm(span=10, adjust=False).mean()
        es = c.ewm(span=30, adjust=False).mean()
        mid    = (ef + es) / 2.0
        ch     = (ef - es).abs() + 1e-8
        out['ema_spread']      = ef - es
        out['ema_channel_pos'] = (c - mid) / ch
        out['ema_spread_z']    = out['ema_spread'].rolling(30, min_periods=2).apply(
            lambda x: (x[-1] - x.mean()) / (x.std() + 1e-8), raw=True
        ).fillna(0.0)
        sma20 = c.rolling(20, min_periods=1).mean()
        std20 = c.rolling(20, min_periods=1).std().fillna(1.0)
        out['grid_band_pos']   = (c - sma20) / (2.0 * std20 + 1e-8)
        return out

    @cached_property
    def autocorr(self) -> pd.DataFrame:
        """Lag-1 rolling autocorrelation (20-bar window)."""
        lr = self._df['log_return']
        out = pd.DataFrame(index=self._df.index)
        out['autocorr_lag1'] = lr.rolling(20, min_periods=4).apply(
            lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 3 else 0.0,
            raw=False
        ).fillna(0.0)
        return out

    @cached_property
    def percentile_rank(self) -> pd.DataFrame:
        """Rolling percentile rank of close within its own 20/60-bar distribution."""
        c = self._df['close']
        out = pd.DataFrame(index=self._df.index)
        for w, col in [(20, 'pct_rank_20'), (60, 'pct_rank_60')]:
            out[col] = c.rolling(w, min_periods=2).apply(
                lambda x: float(np.mean(x < x[-1])), raw=True
            ).fillna(0.5)
        return out

    @cached_property
    def kalman(self) -> pd.DataFrame:
        """Scalar Kalman filter: smoothed price and velocity column."""
        closes = self._df['close'].values.astype(np.float64)
        n = len(closes)
        price_est = np.empty(n)
        velocity  = np.empty(n)
        x = np.zeros(2)
        P = np.eye(2)
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Q = np.eye(2) * 0.01
        R = np.array([[0.1]])
        for i in range(n):
            xp   = F @ x
            Pp   = F @ P @ F.T + Q
            y    = closes[i] - float(H @ xp)
            S    = float((H @ Pp @ H.T)[0, 0]) + R[0, 0]
            K    = (Pp @ H.T / S).flatten()
            x    = xp + K * y
            P    = (np.eye(2) - np.outer(K, H)) @ Pp
            price_est[i] = x[0]
            velocity[i]  = x[1]
        out = pd.DataFrame(index=self._df.index)
        out['kalman_price']    = price_est
        out['kalman_velocity'] = velocity
        return out

    @cached_property
    def hurst(self) -> pd.DataFrame:
        """Rolling Hurst exponent via R/S analysis (60-bar window, lazy — most expensive)."""
        ret = self._df['log_return'].values
        n   = len(ret)
        window = 60
        h_vals = np.full(n, 0.5)
        for i in range(window, n):
            seg  = ret[i - window:i]
            mn   = seg.mean()
            dev  = np.cumsum(seg - mn)
            R    = dev.max() - dev.min()
            S    = seg.std() + 1e-10
            rs   = R / S
            h_vals[i] = float(np.log(rs) / np.log(window)) if rs > 0 else 0.5
        out = pd.DataFrame(index=self._df.index)
        out['hurst_exponent'] = h_vals
        return out

    # ── Ordered family registry ─────────────────────────────────────────────────

    # All families in dependency order. Callers can pass a subset to compute().
    FAMILIES = [
        'returns_momentum', 'ema_macd', 'rsi', 'bollinger', 'atr',
        'stochastic', 'adx', 'vwap', 'zscore', 'volatility',
        'donchian', 'volume_flow', 'spread_proxy', 'autocorr',
        'percentile_rank', 'kalman', 'hurst',
    ]

    def compute(self, families: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Merge computed indicator families into self._df and return the enriched DataFrame.

        Args:
            families: list of family names to compute (default: all FAMILIES).
                      Pass a subset for faster turnaround when only a few are needed.
        """
        targets = families if families is not None else self.FAMILIES

        result = self._df.copy()
        for name in targets:
            cols = getattr(self, name)   # triggers @cached_property on first call
            for col in cols.columns:
                result[col] = cols[col].values

        # Clean up any remaining NaN / Inf
        result = result.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        self._df = result
        return result

    @property
    def indicator_columns(self) -> list:
        """All indicator column names currently merged into _df."""
        base = {'symbol', 'timestamp', 'open', 'close', 'high', 'low', 'volume', 'log_return'}
        return [c for c in self._df.columns if c not in base]
