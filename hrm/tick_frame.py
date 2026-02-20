"""
Tick Frame
==========

Seekable, DataFrame-backed tick iterator.
All 25 agents observe the same tick on each step.
Backed by pre-computed signals from SignalCache.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List


class TickFrame:
    """Seekable 5-year tick iterator backed by pandas DataFrames.
    
    Holds OHLCV + pre-computed signals per symbol. Supports:
    - Sequential stepping (for backtest)
    - Random seek (for analysis / resume)
    - Windowed lookback (for HRM tensor construction)
    """
    
    def __init__(self, candles: Dict[str, pd.DataFrame],
                 signals: Dict[str, pd.DataFrame]):
        """
        Args:
            candles: {symbol: DataFrame[time, open, high, low, close, volume]}
            signals: {symbol: DataFrame[time, grid, momentum, rsi, ...]}
        """
        self.symbols = sorted(candles.keys())
        
        # Build unified time index (union of all symbols)
        all_times = set()
        for sym in self.symbols:
            if sym in candles and len(candles[sym]) > 0:
                all_times.update(candles[sym].index.tolist())
        self.time_index = pd.DatetimeIndex(sorted(all_times))
        
        # Store per-symbol data, reindexed to unified time (forward-fill)
        self._candles: Dict[str, pd.DataFrame] = {}
        self._signals: Dict[str, pd.DataFrame] = {}
        
        for sym in self.symbols:
            if sym in candles and len(candles[sym]) > 0:
                c = candles[sym].reindex(self.time_index, method='ffill')
                self._candles[sym] = c
            if sym in signals and len(signals[sym]) > 0:
                s = signals[sym].reindex(self.time_index, method='ffill')
                self._signals[sym] = s
        
        # Pointer
        self._pos = 0
        self._len = len(self.time_index)
    
    @property
    def current_time(self) -> pd.Timestamp:
        """Current tick timestamp."""
        if self._pos < self._len:
            return self.time_index[self._pos]
        return self.time_index[-1]
    
    @property 
    def current_pos(self) -> int:
        return self._pos
    
    def step(self) -> Optional[Dict[str, pd.Series]]:
        """Advance 1 tick. Returns {symbol: Series(ohlcv + signals)} or None if done."""
        if self._pos >= self._len:
            return None
        
        tick = {}
        t = self.time_index[self._pos]
        for sym in self.symbols:
            if sym in self._candles:
                row = self._candles[sym].iloc[self._pos].copy()
                if sym in self._signals:
                    sig_row = self._signals[sym].iloc[self._pos]
                    row = pd.concat([row, sig_row])
                tick[sym] = row
        
        self._pos += 1
        return tick
    
    def seek(self, timestamp: pd.Timestamp):
        """Random access — jump to a specific time."""
        self._pos = int(self.time_index.searchsorted(timestamp))
    
    def window(self, symbol: str, n: int) -> Optional[pd.DataFrame]:
        """Last n bars of candle+signal data for a symbol."""
        if symbol not in self._candles:
            return None
        start = max(0, self._pos - n)
        end = self._pos
        c = self._candles[symbol].iloc[start:end]
        if symbol in self._signals:
            s = self._signals[symbol].iloc[start:end]
            return pd.concat([c, s], axis=1)
        return c
    
    def remaining(self) -> int:
        """Ticks remaining."""
        return self._len - self._pos
    
    def total(self) -> int:
        return self._len
    
    def progress(self) -> float:
        """0.0 to 1.0"""
        return self._pos / max(self._len, 1)
    
    def reset(self):
        self._pos = 0
