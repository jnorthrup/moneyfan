"""
Test script to verify 48-column schema implementation
"""

import sys
import os
sys.path.insert(0, '/Users/jim/work/moneyfan')

import pandas as pd
import numpy as np
from datetime import datetime

def test_48_column_schema():
    """Test if the schema has 48 columns"""
    
    # Create a test DataFrame with basic OHLCV
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(100) + 100,
        'high': np.random.randn(100) + 100 + 1,
        'low': np.random.randn(100) + 100 - 1,
        'close': np.random.randn(100) + 100,
        'volume': np.random.randn(100) * 1000,
        'quote_volume': np.random.randn(100) * 100000,
        'trades': np.random.randint(10, 100, 100),
        'taker_buy_base': np.random.randn(100) * 500,
        'taker_buy_quote': np.random.randn(100) * 50000,
    })
    
    df.set_index('timestamp', inplace=True)
    
    # Now apply the technical indicator calculation from public_binance_loader
    # This is copied from the _calculate_technical_indicators method
    
    # Calculate SMA and EMA indicators
    windows = [5, 15, 60]
    for window in windows:
        df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
    
    # RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD (12, 26, 9)
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands (20, 2)
    bb_window = 20
    bb_std = df['close'].rolling(window=bb_window).std()
    bb_mid = df['close'].rolling(window=bb_window).mean()
    df['bb_mid'] = bb_mid
    df['bb_upper'] = bb_mid + (2 * bb_std)
    df['bb_lower'] = bb_mid - (2 * bb_std)
    
    # ATR (14)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(window=14).mean()
    
    # ADX (14)
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr_smooth = tr.rolling(window=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / tr_smooth)
    minus_di = (-minus_dm).rolling(window=14).mean() / tr_smooth
    
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['adx_14'] = dx.rolling(window=14).mean()
    
    # Synthetic orderbook features (emulated)
    df['ob_imbalance'] = (df['taker_buy_base'] - df['volume'] * 0.5) / df['volume']
    df['bid_price'] = df['close'] * 0.9995  # Simulated 0.05% spread
    df['ask_price'] = df['close'] * 1.0005  # Simulated 0.05% spread
    df['bid_size'] = df['volume'] * np.random.uniform(0.8, 1.2, len(df))
    df['ask_size'] = df['volume'] * np.random.uniform(0.8, 1.2, len(df))
    df['depth_5_bid'] = df['bid_size'] * 5
    df['depth_5_ask'] = df['ask_size'] * 5
    df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2
    df['spread_pct'] = (df['ask_price'] - df['bid_price']) / df['mid_price']
    
    # VWAP
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Returns (multiple horizons)
    df['returns_1m'] = df['close'].pct_change(1)
    df['returns_5m'] = df['close'].pct_change(5)
    df['returns_15m'] = df['close'].pct_change(15)
    df['returns_1h'] = df['close'].pct_change(60)
    
    # Volatility
    df['vol_5m'] = df['close'].rolling(window=5).std()
    
    # Regime label (simplified - based on slope)
    df['regime_label'] = 1  # Default to flat
    sma_15 = df['sma_15']
    sma_15_diff = sma_15.diff()
    df.loc[sma_15_diff > 0.001, 'regime_label'] = 2  # Up
    df.loc[sma_15_diff < -0.001, 'regime_label'] = 0  # Down
    
    # Stochastic compass (simplified GBM drift)
    returns = df['close'].pct_change().dropna()
    if len(returns) > 0:
        mu = returns.mean() * 1440  # Annualized
        sigma = returns.std() * np.sqrt(1440)
        df['stochastic_compass'] = mu / (sigma + 1e-8)
    else:
        df['stochastic_compass'] = 0.0
    
    # Horizon tags
    df['horizon_tag'] = '05m'  # Default
    
    # Predictor confidences (placeholders)
    df['predictor_conf_5m'] = 0.5
    df['predictor_conf_15m'] = 0.5
    df['predictor_conf_1h'] = 0.5
    
    # HRM reward (placeholder)
    df['hrm_reward'] = 0.0
    
    # Veto flag (placeholder)
    df['veto_flag'] = False
    
    # Position size (placeholder)
    df['position_size_usd'] = 0.0
    
    # Equity curve (placeholder)
    df['equity_curve'] = 0.0
    
    # Print results
    print("48-COLUMN SCHEMA VERIFICATION")
    print("="*80)
    print(f"Total columns: {len(df.columns)}")
    print(f"DataFrame shape: {df.shape}")
    print()
    print("Columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
    
    # Expected columns (from the specification)
    expected_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote',
        'sma_5', 'sma_15', 'sma_60', 'ema_5', 'ema_15', 'ema_60',
        'rsi_14', 'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_lower', 'bb_mid', 'atr_14', 'adx_14',
        'ob_imbalance', 'bid_price', 'ask_price', 'bid_size', 'ask_size',
        'depth_5_bid', 'depth_5_ask', 'mid_price', 'spread_pct', 'vwap',
        'returns_1m', 'returns_5m', 'returns_15m', 'returns_1h',
        'vol_5m',
        'regime_label', 'stochastic_compass', 'horizon_tag',
        'predictor_conf_5m', 'predictor_conf_15m', 'predictor_conf_1h',
        'hrm_reward', 'veto_flag', 'position_size_usd', 'equity_curve',
    ]
    
    print()
    print("MISSING COLUMNS:")
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        for col in missing:
            print(f"  ❌ {col}")
        print(f"\nTotal missing: {len(missing)}")
    else:
        print("  ✅ All expected columns present")
    
    print()
    print("EXTRA COLUMNS:")
    extra = [col for col in df.columns if col not in expected_columns and col not in ['timestamp', 'time']]
    if extra:
        for col in extra:
            print(f"  ⚠️  {col}")
    else:
        print("  ✅ No extra columns")
    
    # Count by category
    print()
    print("COLUMN COUNT BY CATEGORY:")
    basic = ['open', 'high', 'low', 'close', 'volume']
    binance = ['quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']
    technical = ['sma_5', 'sma_15', 'sma_60', 'ema_5', 'ema_15', 'ema_60',
                'rsi_14', 'macd', 'macd_signal', 'macd_hist',
                'bb_upper', 'bb_lower', 'bb_mid', 'atr_14', 'adx_14']
    synthetic_ob = ['ob_imbalance', 'bid_price', 'ask_price', 'bid_size', 'ask_size',
                   'depth_5_bid', 'depth_5_ask', 'mid_price', 'spread_pct', 'vwap']
    returns = ['returns_1m', 'returns_5m', 'returns_15m', 'returns_1h']
    volatility = ['vol_5m']
    regime = ['regime_label', 'stochastic_compass', 'horizon_tag']
    predictor = ['predictor_conf_5m', 'predictor_conf_15m', 'predictor_conf_1h']
    hrm = ['hrm_reward', 'veto_flag', 'position_size_usd', 'equity_curve']
    
    categories = {
        'Basic OHLCV': basic,
        'Binance-specific': binance,
        'Technical indicators': technical,
        'Synthetic orderbook': synthetic_ob,
        'Returns': returns,
        'Volatility': volatility,
        'Regime & labels': regime,
        'Predictor confidences': predictor,
        'HRM-specific': hrm,
    }
    
    for cat_name, cat_cols in categories.items():
        present = [col for col in cat_cols if col in df.columns]
        print(f"  {cat_name}: {len(present)}/{len(cat_cols)}")
    
    return df

if __name__ == "__main__":
    df = test_48_column_schema()