"""
Test backfilling a specific Coinbase file
"""

import sys
import os
sys.path.insert(0, '/Users/jim/work/moneyfan')

import pandas as pd
import numpy as np
from pathlib import Path
from coinbase_backfill_agent import CoinbaseBackfillAgent, CoinbaseBackfillConfig

def test_specific_file():
    """Test backfilling a specific file"""
    
    # Look for BTC_USD.feather
    btc_file = Path("hrm/data/arrow/BTC_USD.feather")
    
    if not btc_file.exists():
        print(f"File not found: {btc_file}")
        return
    
    print(f"Found file: {btc_file}")
    print(f"File size: {btc_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Load the file
    df = pd.read_feather(btc_file)
    print(f"Loaded DataFrame with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Index type: {type(df.index)}")
    print(f"Index name: {df.index.name}")
    
    # Check if it has timestamp column
    if 'timestamp' in df.columns:
        print(f"Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        # Convert to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)
    elif 'time' in df.columns:
        print(f"Time range: {df['time'].min()} to {df['time'].max()}")
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
    
    # Now test with agent
    config = CoinbaseBackfillConfig()
    agent = CoinbaseBackfillAgent(config)
    
    # Transform to 48 columns
    df_48 = agent.load_48_column_schema(df, 'BTC-USD', '1m')
    
    print(f"\nAfter 48-column transformation:")
    print(f"Shape: {df_48.shape}")
    print(f"Columns: {len(df_48.columns)}")
    
    # Show some sample data
    print(f"\nSample data (first 5 rows):")
    print(df_48.head()[['open', 'high', 'low', 'close', 'volume', 'timeframe']])
    
    # Compute hash
    data_hash = agent.compute_data_hash(df_48)
    print(f"\nData hash: {data_hash[:32]}...")
    
    # Test DuckDB insertion
    print(f"\nTesting DuckDB insertion...")
    
    # Insert first 10 rows
    rows_inserted = 0
    for timestamp, row in df_48.head(10).iterrows():
        try:
            values = (
                'BTC-USD',
                timestamp,
                '1m',
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume']),
                float(row.get('quote_volume', 0.0)),
                float(row.get('trades', 0.0)),
                float(row.get('taker_buy_base', 0.0)),
                float(row.get('taker_buy_quote', 0.0)),
                float(row.get('sma_5', 0.0)),
                float(row.get('sma_15', 0.0)),
                float(row.get('sma_60', 0.0)),
                float(row.get('ema_5', 0.0)),
                float(row.get('ema_15', 0.0)),
                float(row.get('ema_60', 0.0)),
                float(row.get('rsi_14', 0.0)),
                float(row.get('macd', 0.0)),
                float(row.get('macd_signal', 0.0)),
                float(row.get('macd_hist', 0.0)),
                float(row.get('bb_upper', 0.0)),
                float(row.get('bb_lower', 0.0)),
                float(row.get('bb_mid', 0.0)),
                float(row.get('atr_14', 0.0)),
                float(row.get('adx_14', 0.0)),
                float(row.get('ob_imbalance', 0.0)),
                float(row.get('bid_price', 0.0)),
                float(row.get('ask_price', 0.0)),
                float(row.get('bid_size', 0.0)),
                float(row.get('ask_size', 0.0)),
                float(row.get('depth_5_bid', 0.0)),
                float(row.get('depth_5_ask', 0.0)),
                float(row.get('mid_price', 0.0)),
                float(row.get('spread_pct', 0.0)),
                float(row.get('vwap', 0.0)),
                float(row.get('returns_1m', 0.0)),
                float(row.get('returns_5m', 0.0)),
                float(row.get('returns_15m', 0.0)),
                float(row.get('returns_1h', 0.0)),
                float(row.get('vol_5m', 0.0)),
                float(row.get('regime_label', 1.0)),
                float(row.get('stochastic_compass', 0.0)),
                str(row.get('horizon_tag', '1m')),
                float(row.get('predictor_conf_5m', 0.5)),
                float(row.get('predictor_conf_15m', 0.5)),
                float(row.get('predictor_conf_1h', 0.5)),
                float(row.get('hrm_reward', 0.0)),
                bool(row.get('veto_flag', False)),
                float(row.get('position_size_usd', 0.0)),
                float(row.get('equity_curve', 0.0)),
                str(btc_file),
                config.import_timestamp,
                data_hash
            )
            
            agent.duck_store.conn.execute("""
                INSERT OR REPLACE INTO coinbase_source 
                (symbol, timestamp, timeframe, open, high, low, close, volume,
                 quote_volume, trades, taker_buy_base, taker_buy_quote,
                 sma_5, sma_15, sma_60, ema_5, ema_15, ema_60, rsi_14,
                 macd, macd_signal, macd_hist, bb_upper, bb_lower, bb_mid,
                 atr_14, adx_14, ob_imbalance, bid_price, ask_price,
                 bid_size, ask_size, depth_5_bid, depth_5_ask, mid_price,
                 spread_pct, vwap, returns_1m, returns_5m, returns_15m,
                 returns_1h, vol_5m, regime_label, stochastic_compass,
                 horizon_tag, predictor_conf_5m, predictor_conf_15m,
                 predictor_conf_1h, hrm_reward, veto_flag, position_size_usd,
                 equity_curve, source_file, import_timestamp, data_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, values)
            rows_inserted += 1
        except Exception as e:
            print(f"  Error inserting row: {e}")
            break
    
    print(f"Inserted {rows_inserted} rows into DuckDB")
    
    # Verify insertion
    conn = agent.duck_store.conn
    result = conn.execute("SELECT COUNT(*) as count FROM coinbase_source WHERE symbol = 'BTC-USD'").fetchone()
    print(f"Total BTC-USD rows in database: {result[0]}")
    
    # Show sample data
    result = conn.execute("SELECT * FROM coinbase_source WHERE symbol = 'BTC-USD' LIMIT 3").fetchall()
    print(f"Sample rows from database: {result}")

if __name__ == "__main__":
    test_specific_file()