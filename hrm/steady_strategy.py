
import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta

try:
    from hrm.duck_store import DuckStore
except ImportError:
    try:
        from hrm.duck_store import DuckStore
    except:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from duck_store import DuckStore

def load_data(symbol, start_date=None, db_path="hrm/data/coinbase.duckdb"):
    """Steady data loader using DuckStore (DuckDB + Arrow)"""
    store = DuckStore(db_path)
    df = store.load(symbol, start=start_date)
    return df

class VectorizedHarvestStrategy:
    """
    Implements the 'Steady' volatility harvesting logic using Pandas.
    Logic:
      - Mantain a 'Baseline' value for the asset.
      - If (CurrentValue - Baseline) / Baseline > Threshold:
           Trigger Harvest (Sell Surplus).
           Update Baseline (optional growth).
      - If Value < Baseline:
           Do nothing (or buyback if enabled).
    """
    def __init__(self, initial_capital=10000.0, harvest_trigger=0.05, harvest_period=12):
        self.initial_capital = initial_capital
        self.trigger = harvest_trigger
        self.period = harvest_period # roughly equivalent to 'wait cycles'
        
    def run(self, df):
        # "Steady" Loop: Simple, robust, explicit state.
        # Vectorized is harder for path-dependent equity.
        baseline = self.initial_capital
        units = self.initial_capital / df['close'].iloc[0]
        cash = 0.0
        
        last_action_idx = -999
        
        # Pre-convert to numpy for max speed
        close_arr = df['close'].values
        ts_arr = df.index
        
        equity = []
        trades = []
        
        for i, price in enumerate(close_arr):
            val = units * price
            curr_equity = val + cash
            equity.append(curr_equity)
            
            dev = (val - baseline) / baseline
            
            if dev > self.trigger and (i - last_action_idx) > self.period:
                # Harvest
                surplus = val - baseline
                units_to_sell = surplus / price
                units -= units_to_sell
                cash += surplus
                trades.append({'time': ts_arr[i], 'type': 'HARVEST', 'val': surplus})
                last_action_idx = i
                
                # In steady mode, baseline ideally grows or we capture cash.
                # Here we capture cash.
                
        return pd.DataFrame({'close': close_arr, 'equity': equity}, index=df.index), trades

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SOL-USD")
    parser.add_argument("--db", default="hrm/data/coinbase.duckdb")
    args = parser.parse_args()
    
    print(f"Loading {args.symbol} from {args.db}...")
    df = load_data(args.symbol, db_path=args.db)
    
    if df.empty:
        print("No data found.")
        return

    print(f"Loaded {len(df)} rows.")
    
    strat = VectorizedHarvestStrategy(harvest_trigger=0.10, harvest_period=24) # 10% gain, 2 hour cooldown (5m * 24)
    res_df, trades = strat.run(df)
    
    total_harvested = sum(t['val'] for t in trades)
    final_equity = res_df['equity'].iloc[-1]
    hodl_equity = (10000.0 / df['close'].iloc[0]) * df['close'].iloc[-1]
    
    print("-" * 60)
    print(f"STEADY STRATEGY REPORT: {args.symbol}")
    print("-" * 60)
    print(f"Total Trades:     {len(trades)}")
    print(f"Total Harvested:  ${total_harvested:,.2f}")
    print(f"Final Equity:     ${final_equity:,.2f} (Portfolio Value)")
    print(f"HODL Equity:      ${hodl_equity:,.2f}")
    print(f"Performance:      {final_equity / hodl_equity:.2%} vs HODL")
    print("-" * 60)

if __name__ == "__main__":
    main()
