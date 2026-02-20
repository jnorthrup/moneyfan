"""
Feature Engineering - Pandas-first, Spark-optional

All features computed as pandas DataFrames.
Can be converted to Spark DF for scale.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class FeatureConfig:
    """Feature configuration"""
    n_assets: int = 128
    lookback: int = 20
    
    # Feature columns (per asset)
    price_cols: tuple = ('open', 'high', 'low', 'close', 'volume')
    derived_cols: tuple = ('returns', 'volatility', 'momentum', 'rsi', 'ma_ratio')
    time_cols: tuple = ('hour', 'day_of_week', 'session', 'month', 'month_end')


# =============================================================================
# TIME FEATURES (Pandas)
# =============================================================================

def add_time_features(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    Add time features to DataFrame.
    
    Input: df with timestamp column
    Output: df with hour, day_of_week, session, month, month_end columns
    """
    df = df.copy()
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='s')
    
    dt = df[timestamp_col].dt
    
    df['hour'] = dt.hour / 23.0
    df['day_of_week'] = dt.dayofweek / 6.0
    df['month'] = (dt.month - 1) / 11.0
    df['month_end'] = (dt.day >= 28).astype(float)
    
    # Market session (UTC)
    # Weekend = 0, Asian = 1, European = 2, US = 3
    def get_session(row):
        if row['day_of_week'] >= 5/6:  # Weekend
            return 0.0
        elif row['hour'] < 8/23:  # Asian
            return 1/3
        elif row['hour'] < 16/23:  # European
            return 2/3
        else:  # US
            return 1.0
    
    df['session'] = df.apply(get_session, axis=1)
    
    return df


# =============================================================================
# PRICE FEATURES (Pandas - grouped by asset)
# =============================================================================

def add_price_features(group: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Add derived price features for one asset.
    
    Input: group with open, high, low, close, volume
    Output: group with returns, volatility, momentum, rsi, ma_ratio
    """
    group = group.copy()
    
    # Returns
    group['returns'] = group['close'].pct_change()
    
    # Volatility (normalized high-low range)
    group['volatility'] = (group['high'] - group['low']) / group['open']
    
    # Momentum (intraday)
    group['momentum'] = (group['close'] / group['open']) - 1
    
    # RSI (simplified 14-period)
    delta = group['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    group['rsi'] = 100 - (100 / (1 + rs))
    group['rsi'] = group['rsi'] / 100.0  # Normalize to 0-1
    
    # MA ratio (close / 20-period MA)
    ma = group['close'].rolling(lookback).mean()
    group['ma_ratio'] = group['close'] / ma
    
    # Volume normalized
    vol_ma = group['volume'].rolling(lookback).mean()
    group['volume_norm'] = group['volume'] / vol_ma
    
    return group


def compute_all_features(df: pd.DataFrame, 
                         asset_col: str = 'asset',
                         timestamp_col: str = 'timestamp',
                         lookback: int = 20) -> pd.DataFrame:
    """
    Compute all features for all assets.
    
    Input: df with [asset, timestamp, open, high, low, close, volume]
    Output: df with all features
    """
    # Add time features (applies to all rows)
    df = add_time_features(df, timestamp_col)
    
    # Group by asset and add price features
    df = df.groupby(asset_col, group_keys=False).apply(
        lambda g: add_price_features(g, lookback)
    )
    
    return df


# =============================================================================
# SPARK COMPATIBILITY
# =============================================================================

def pandas_to_spark(pdf: pd.DataFrame, spark_session=None):
    """Convert pandas DF to Spark DF (if available)"""
    try:
        from pyspark.sql import SparkSession
        if spark_session is None:
            spark = SparkSession.builder.getOrCreate()
        else:
            spark = spark_session
        return spark.createDataFrame(pdf)
    except ImportError:
        print("PySpark not available, returning pandas DataFrame")
        return pdf


def spark_to_pandas(sdf) -> pd.DataFrame:
    """Convert Spark DF to pandas DF"""
    return sdf.toPandas()


# =============================================================================
# MODEL SIGNALS (Pandas Vectorized)
# =============================================================================

def signal_volatility_breakout(df: pd.DataFrame) -> pd.Series:
    """
    PROVEN WINNER: volatility × breakout
    
    Vectorized pandas computation.
    """
    vol_signal = df['volatility'].clip(0, 1)
    price_position = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    breakout = 2 * price_position - 1
    return vol_signal * breakout


def signal_momentum_trend(df: pd.DataFrame) -> pd.Series:
    """momentum × trend"""
    trend = np.sign(df['ma_ratio'] - 1)
    strength = df['momentum'].abs().clip(0, 1)
    return strength * trend


def signal_mean_reversion(df: pd.DataFrame) -> pd.Series:
    """mean reversion"""
    deviation = df['ma_ratio'] - 1
    rsi_signal = (50 - df['rsi'] * 100) / 50  # rsi was normalized 0-1
    signal = np.where(deviation.abs() > 0.02, -np.sign(deviation), 0)
    return 0.5 * signal + 0.5 * rsi_signal


def compute_all_signals(df: pd.DataFrame, 
                        weights: Dict[str, float] = None) -> pd.DataFrame:
    """
    Compute all model signals and weighted combination.
    
    Input: df with features
    Output: df with signal columns
    """
    df = df.copy()
    
    # Default weights (favor proven winner)
    if weights is None:
        weights = {
            'volatility_breakout': 0.5,
            'momentum_trend': 0.2,
            'mean_reversion': 0.3,
        }
    
    # Compute individual signals
    df['signal_volatility_breakout'] = signal_volatility_breakout(df)
    df['signal_momentum_trend'] = signal_momentum_trend(df)
    df['signal_mean_reversion'] = signal_mean_reversion(df)
    
    # Weighted combination
    df['signal_combined'] = (
        weights['volatility_breakout'] * df['signal_volatility_breakout'] +
        weights['momentum_trend'] * df['signal_momentum_trend'] +
        weights['mean_reversion'] * df['signal_mean_reversion']
    )
    
    return df


# =============================================================================
# NOTEBOOK UTILITIES
# =============================================================================

def load_sample_data(db_path: str = 'hrm/data/coinbase.duckdb',
                     n_assets: int = 10,
                     n_rows: int = 1000) -> pd.DataFrame:
    """
    Load sample data for notebook exploration using DuckStore.
    """
    try:
        from duck_store import DuckStore
    except ImportError:
        from hrm.duck_store import DuckStore
    
    store = DuckStore(db_path)
    symbols = store.get_symbols()[:n_assets]
    
    dfs = []
    for symbol in symbols:
        df = store.load(symbol)
        if len(df) > 0:
            df = df.tail(n_rows).copy()
            df['asset'] = symbol
            dfs.append(df)
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame(columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'asset'
        ])


def quick_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Quick summary for notebook display"""
    return df.describe().T[['count', 'mean', 'std', 'min', 'max']]


def plot_signals(df: pd.DataFrame, asset: str = None):
    """Plot signals for notebook (returns matplotlib figure)"""
    import matplotlib.pyplot as plt
    
    if asset:
        df = df[df['asset'] == asset]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # Price
    axes[0].plot(df['close'].values)
    axes[0].set_title('Close Price')
    
    # Individual signals
    axes[1].plot(df['signal_volatility_breakout'].values, label='vol_breakout')
    axes[1].plot(df['signal_momentum_trend'].values, label='mom_trend')
    axes[1].plot(df['signal_mean_reversion'].values, label='mean_rev')
    axes[1].legend()
    axes[1].set_title('Individual Signals')
    
    # Combined signal
    axes[2].plot(df['signal_combined'].values, color='black')
    axes[2].axhline(0, color='gray', linestyle='--')
    axes[2].set_title('Combined Signal')
    
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("HRM Feature Engineering (Pandas-first)")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n = 1000
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='5min'),
        'asset': 'BTC-USD',
        'open': 40000 + np.random.randn(n).cumsum() * 100,
        'high': 40100 + np.random.randn(n).cumsum() * 100,
        'low': 39900 + np.random.randn(n).cumsum() * 100,
        'close': 40050 + np.random.randn(n).cumsum() * 100,
        'volume': np.random.randn(n).abs() * 1000,
    })
    
    # Compute features
    df = add_time_features(df)
    df = add_price_features(df)
    
    print(f"Features computed: {df.columns.tolist()}")
    print(f"\n{quick_summary(df)}")
    
    # Compute signals
    df = compute_all_signals(df)
    
    print(f"\nSignal columns: {[c for c in df.columns if 'signal' in c]}")
    print(f"\nSignal summary:\n{quick_summary(df[[c for c in df.columns if 'signal' in c]])}")
