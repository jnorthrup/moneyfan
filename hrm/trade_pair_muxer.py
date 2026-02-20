"""
TradePairMuxer - Per-symbol event emitter for stochastic bag training.

Borrowed intent from Kotlin acapulco.old:
- TradePairEventMuxer: each asset has its own muxer
- intra_events: shared flow of (data, metadata) tuples
- Assets without data don't emit
- decorateView: lazy candle functions that all agents see as inputs

Pandas replaces Cursor system for data handling.
"""

from dataclasses import dataclass, field
from typing import Iterator, Optional, Callable
from functools import cached_property, lru_cache
import numpy as np
import pandas as pd
import random


@dataclass
class TradePairMuxer:
    """
    Per-symbol event emitter - like Kotlin TradePairEventMuxer.
    Only emits when data exists for this symbol.
    
    USD is special - it's the base currency with $100 holdings.
    """
    symbol: str
    df: Optional[pd.DataFrame]
    _current_idx: int = 0
    
    def __post_init__(self):
        if self.symbol == "USD":
            # USD is the base currency, no OHLCV data needed
            # It's represented as position 1.0, value $100
            if self.df is not None and len(self.df) > 0:
                # Keep the dataframe for compatibility
                pass
            else:
                # Create dummy USD dataframe
                self.df = pd.DataFrame({
                    'open': [1.0],
                    'high': [1.0],
                    'low': [1.0],
                    'close': [1.0],
                    'volume': [0.0]
                }, index=pd.to_datetime(['2000-01-01']))
    
    # --- Lazy Candle Functions (computed on demand) ---
    # These are the features that all 24 algorithms will see as inputs
    
    @cached_property
    def ohlcv(self) -> pd.DataFrame:
        """Basic OHLCV columns - like Kotlin cursor columns"""
        if self.symbol == "USD":
            # USD doesn't have meaningful OHLCV
            return pd.DataFrame({
                'open': [1.0],
                'high': [1.0],
                'low': [1.0],
                'close': [1.0],
                'volume': [0.0]
            })
        return self.df[['open', 'high', 'low', 'close', 'volume']].copy()
    
    @cached_property
    def returns(self) -> pd.Series:
        """Percent returns"""
        return self.df['close'].pct_change().fillna(0.0)
    
    @cached_property
    def volatility(self) -> pd.Series:
        """Normalized high-low range"""
        return (self.df['high'] - self.df['low']) / self.df['open']
    
    @cached_property
    def momentum(self) -> pd.Series:
        """Intraday momentum"""
        return (self.df['close'] / self.df['open']) - 1
    
    @cached_property
    def rsi(self) -> pd.Series:
        """RSI (14-period, normalized to 0-1)"""
        delta = self.df['close'].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return (rsi / 100.0).fillna(0.5)
    
    @cached_property
    def breakout(self) -> pd.Series:
        """Breakout indicator (z-score from 20-period mean)"""
        mean = self.df['close'].rolling(20).mean()
        std = self.df['close'].rolling(20).std()
        return ((self.df['close'] - mean) / (std + 1e-8)).fillna(0.0)
    
    @cached_property
    def trend(self) -> pd.Series:
        """Trend indicator (50/200 MA ratio)"""
        ma50 = self.df['close'].rolling(50).mean()
        ma200 = self.df['close'].rolling(200).mean()
        return ((ma50 / ma200) - 1).fillna(0.0)
    
    @cached_property
    def zscore(self) -> pd.Series:
        """Z-score for mean reversion"""
        mean = self.df['close'].rolling(20).mean()
        std = self.df['close'].rolling(20).std()
        return ((self.df['close'] - mean) / (std + 1e-8)).fillna(0.0)
    
    @cached_property
    def macd(self) -> pd.DataFrame:
        """MACD indicator"""
        fast = self.df['close'].ewm(span=12).mean()
        slow = self.df['close'].ewm(span=26).mean()
        macd_line = fast - slow
        signal = macd_line.ewm(span=9).mean()
        return pd.DataFrame({
            'macd': macd_line.fillna(0.0),
            'signal': signal.fillna(0.0),
            'histogram': (macd_line - signal).fillna(0.0)
        })
    
    @cached_property
    def bollinger(self) -> pd.DataFrame:
        """Bollinger Bands"""
        mid = self.df['close'].rolling(20).mean()
        std = self.df['close'].rolling(20).std()
        return pd.DataFrame({
            'mid': mid.fillna(0.0),
            'upper': (mid + 2 * std).fillna(0.0),
            'lower': (mid - 2 * std).fillna(0.0),
            'std': std.fillna(0.0),
            'position': ((self.df['close'] - mid) / (std + 1e-8)).fillna(0.0)
        })
    
    @cached_property
    def volume_ratio(self) -> pd.Series:
        """Volume ratio vs 20-period average"""
        avg = self.df['volume'].rolling(20).mean()
        return (self.df['volume'] / (avg + 1e-8)).fillna(1.0)
    
    @cached_property
    def atr(self) -> pd.Series:
        """Average True Range ratio"""
        high_low = self.df['high'] - self.df['low']
        high_close = abs(self.df['high'] - self.df['close'].shift(1))
        low_close = abs(self.df['low'] - self.df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        return (tr / (atr + 1e-8)).fillna(1.0)
    
    @cached_property
    def price_position(self) -> pd.Series:
        """Price position (0-1) between 20-period min and max"""
        low = self.df['close'].rolling(20).min()
        high = self.df['close'].rolling(20).max()
        return ((self.df['close'] - low) / (high - low + 1e-8)).fillna(0.5)
    
    @cached_property
    def all_features(self) -> pd.DataFrame:
        """
        All lazy candle features combined.
        This is what the 24 algorithms see as inputs.
        Like Kotlin decorateView() output.
        
        SCALE PRESERVATION:
        - Raw OHLCV preserved at actual scale
        - Ratios computed but NOT normalized (preserve fidelity)
        - Previous candle values included for scale reference
        """
        # Previous candle values (for scale reference)
        prev_close = self.df['close'].shift(1)
        prev_volume = self.df['volume'].shift(1)
        prev_high = self.df['high'].shift(1)
        prev_low = self.df['low'].shift(1)
        
        return pd.DataFrame({
            # Raw OHLCV - preserved at actual scale
            'open': self.df['open'],
            'high': self.df['high'],
            'low': self.df['low'],
            'close': self.df['close'],
            'volume': self.df['volume'],
            
            # Previous candle (scale reference)
            'prev_close': prev_close.fillna(self.df['close']),
            'prev_volume': prev_volume.fillna(self.df['volume']),
            'prev_high': prev_high.fillna(self.df['high']),
            'prev_low': prev_low.fillna(self.df['low']),
            
            # Returns - preserves direction and magnitude
            'returns': self.returns,
            
            # Range - actual price range, not normalized
            'range': (self.df['high'] - self.df['low']),
            'body': (self.df['close'] - self.df['open']),
            
            # Ratios (preserve scale)
            'volatility': self.volatility,
            'momentum': self.momentum,
            'volume_ratio': self.volume_ratio,
            'atr_ratio': self.atr,
            
            # Position indicators (0-1 bounded)
            'rsi': self.rsi,
            'price_position': self.price_position,
            
            # Deviation indicators (z-score scale)
            'breakout': self.breakout,
            'zscore': self.zscore,
            'trend': self.trend,
            'bb_position': self.bollinger['position'],
            
            # MACD (actual values, not normalized)
            'macd': self.macd['macd'],
            'macd_signal': self.macd['signal'],
        }).fillna(0.0)
    
    @cached_property
    def scaled_features(self) -> pd.DataFrame:
        """
        Scaled features for model input.
        Uses previous candle as scale reference (minmax preservation).
        
        Each feature is scaled relative to previous candle's range,
        preserving the relationship between candles.
        """
        df = self.all_features.copy()
        
        # Scale OHLCV relative to previous close
        ref_close = df['prev_close'].replace(0, 1)
        for col in ['open', 'high', 'low', 'close', 'prev_close', 'prev_high', 'prev_low']:
            df[f'{col}_scaled'] = df[col] / ref_close
        
        # Scale volume relative to previous volume
        ref_vol = df['prev_volume'].replace(0, 1)
        df['volume_scaled'] = df['volume'] / ref_vol
        df['prev_volume_scaled'] = df['prev_volume'] / ref_vol
        
        # Keep other features as-is (already relative/normalized)
        return df.fillna(0.0)
    
    # --- Event Emission Methods ---
    
    def has_data(self, start: int, end: int) -> bool:
        """Check if data exists in range [start, end)"""
        return end <= len(self.df)
    
    def emit_window(self, start: int, end: int) -> Optional[tuple[pd.DataFrame, int]]:
        """
        Emit (window_data, intra_count) tuple if data exists.
        Returns None if no data in range.
        """
        if not self.has_data(start, end):
            return None
        window = self.all_features.iloc[start:end]
        intra = 0  # Could track incomplete candles
        return window, intra
    
    def emit_event(self, idx: int, window_size: int) -> Optional[tuple[pd.DataFrame, int]]:
        """Emit event at index with lookback window"""
        if idx < window_size or idx >= len(self.df):
            return None
        window = self.all_features.iloc[idx-window_size:idx]
        intra = 0
        return window, intra
    
    def __len__(self) -> int:
        if self.symbol == "USD":
            # USD always "has data" for trading purposes
            return 1000000  # Effectively infinite
        return len(self.df)
    
    def clear_cache(self):
        """Clear all cached properties to free memory"""
        for attr in ['ohlcv', 'returns', 'volatility', 'momentum', 'rsi', 
                     'breakout', 'trend', 'zscore', 'macd', 'bollinger',
                     'volume_ratio', 'atr', 'price_position', 'all_features']:
            if attr in self.__dict__:
                del self.__dict__[attr]


def stochastic_bag(
    muxers: list[TradePairMuxer],
    n_select: tuple[int, int] = (5, 30),
    seed: Optional[int] = None
) -> list[TradePairMuxer]:
    """
    Select random subset of muxers that have data.
    Like Kotlin: assets without data don't participate.
    """
    if seed is not None:
        random.seed(seed)
    
    count = random.randint(*n_select)
    available = [m for m in muxers if len(m) > 0]
    count = min(count, len(available))
    return random.sample(available, count)


def stochastic_extent(
    df: pd.DataFrame,
    extent_range: tuple[int, int] = (32, 256),
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Random time window from data.
    Like Kotlin horizon() - variable lookback.
    """
    if seed is not None:
        random.seed(seed)
    
    extent = random.randint(*extent_range)
    extent = min(extent, len(df))
    start = random.randint(0, max(0, len(df) - extent))
    return df.iloc[start:start+extent]


def horizon_compress(
    df: pd.DataFrame,
    horizon_width: int = 64,
    columns: Optional[list[str]] = None
) -> np.ndarray:
    """
    Compress variable-length data to fixed horizon.
    Like Kotlin pancake() - flatten to fixed size.
    
    Uses even sampling across the data (compression).
    """
    if columns is not None:
        df = df[columns]
    
    if len(df) == 0:
        return np.zeros(horizon_width * len(df.columns))
    
    if len(df) < horizon_width:
        # Pad with zeros if too short (front-pad like Kotlin reverse)
        n_pad = horizon_width - len(df)
        padded = pd.DataFrame(0, index=range(n_pad), columns=df.columns)
        df = pd.concat([padded, df], ignore_index=True)
    
    # Sample evenly across the data (compression)
    indices = np.linspace(0, len(df)-1, horizon_width, dtype=int)
    compressed = df.iloc[indices].values
    
    # Pancake: flatten to 1D
    return compressed.flatten()


def assemble_row(
    muxer: TradePairMuxer,
    time_window: int,
    horizon_width: int,
    columns: Optional[list[str]] = None
) -> Optional[np.ndarray]:
    """
    Assemble a single training row from muxer data.
    Like Kotlin MuxIo.assembleRow.
    """
    result = muxer.emit_window(0, time_window)
    if result is None:
        return None
    
    window, intra = result
    
    # Compress to horizon
    compressed = horizon_compress(window, horizon_width, columns)
    
    # Add metadata (like Kotlin: depth, intra)
    depth = len(window)
    metadata = np.array([depth, intra], dtype=np.float32)
    
    return np.concatenate([metadata, compressed])


async def generate_stochastic_batch(
    muxers: list[TradePairMuxer],
    bag_range: tuple[int, int] = (5, 30),
    extent_range: tuple[int, int] = (32, 256),
    horizon_width: int = 64,
    n_instruments: int = 30,
    columns: Optional[list[str]] = None,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate one stochastic training batch.
    
    Returns:
        batch: [n_instruments, feature_dim] array
        mask: [n_instruments] array (1 = real data, 0 = padding)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # 1. Stochastic extent
    extent = random.randint(*extent_range)
    
    # 2. Stochastic bag: random subset of assets
    bag = stochastic_bag(muxers, bag_range, seed=None)
    
    # 3. Assemble rows
    rows = []
    for muxer in bag:
        row = assemble_row(muxer, extent, horizon_width, columns)
        if row is not None:
            rows.append(row)
    
    if len(rows) == 0:
        # No data available, return zeros
        feature_dim = 2 + horizon_width * 5  # metadata + OHLCV
        return np.zeros((n_instruments, feature_dim)), np.zeros(n_instruments)
    
    # 4. Stack into batch
    batch = np.stack(rows, dtype=np.float32)
    
    # 5. Pad to fixed instrument count
    mask = np.ones(len(rows), dtype=np.float32)
    
    if len(rows) < n_instruments:
        padding = np.zeros((n_instruments - len(rows), rows[0].shape[0]), dtype=np.float32)
        batch = np.concatenate([batch, padding], axis=0)
        mask = np.concatenate([mask, np.zeros(n_instruments - len(rows))], axis=0)
    elif len(rows) > n_instruments:
        batch = batch[:n_instruments]
        mask = mask[:n_instruments]
    
    return batch, mask


class MuxerRegistry:
    """
    Registry of TradePairMuxers.
    Like Kotlin Streamer.eventMuxers map.
    """
    def __init__(self):
        self.muxers: dict[str, TradePairMuxer] = {}
    
    def register(self, symbol: str, df: pd.DataFrame) -> TradePairMuxer:
        """Register a muxer for a symbol"""
        if symbol == "USD":
            # USD doesn't need a dataframe
            muxer = TradePairMuxer(symbol, df=None)
        else:
            muxer = TradePairMuxer(symbol, df)
        self.muxers[symbol] = muxer
        return muxer
    
    def unregister(self, symbol: str):
        """Remove a muxer"""
        if symbol in self.muxers:
            del self.muxers[symbol]
    
    def get(self, symbol: str) -> Optional[TradePairMuxer]:
        """Get a muxer by symbol"""
        return self.muxers.get(symbol)
    
    def get_all(self) -> list[TradePairMuxer]:
        """Get all muxers"""
        return list(self.muxers.values())
    
    def get_active(self, min_length: int = 0) -> list[TradePairMuxer]:
        """Get muxers with data"""
        return [m for m in self.muxers.values() if len(m) >= min_length]
    
    def __len__(self) -> int:
        return len(self.muxers)
    
    def __contains__(self, symbol: str) -> bool:
        return symbol in self.muxers
