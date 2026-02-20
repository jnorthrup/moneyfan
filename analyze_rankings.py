
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.getcwd())

from hrm.coinbase_pipeline import CoinbasePipeline
from signal_orchestrator import (
    Orchestrator, GridService, MomentumService, RSIService, 
    TrendService, VolatilityService, VolumeService, CompositionModel
)

def analyze_rankings(days=365*5):
    print("="*80)
    print(f"  HISTORICAL BACKTEST ANALYSIS ({days//365} YEARS)")
    print("="*80)
    
    pipeline = CoinbasePipeline()
    
    pairs = [
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'AVAX-USD',
        'DOGE-USD', 'DOT-USD', 'MATIC-USD', 'LINK-USD', 'LTC-USD',
        'SHIB-USD', 'UNI-USD', 'XLM-USD', 'ALGO-USD', 'BCH-USD',
        'NEAR-USD', 'ATOM-USD', 'FIL-USD', 'HBAR-USD', 'APT-USD',
        'LDO-USD', 'VET-USD', 'QNT-USD', 'MKR-USD', 'AAVE-USD',
        'FTM-USD', 'SAND-USD', 'MANA-USD', 'XTZ-USD', 'EOS-USD',
        'ETH-BTC', 'SOL-ETH', 'SOL-BTC', 'AVAX-BTC', 'MATIC-BTC'
    ]
    
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    data_map = {}
    
    print(f"Loading data for {len(pairs)} pairs from DB...")
    
    def fetch_pair(pair):
        # print(f"   Checking {pair}...", flush=True) 
        try:
            df = pipeline.history.load_range(pair, start_date, end_date)
            if not df.empty and len(df) > 1000:
                df = df[~df.index.duplicated(keep='first')]
                return (pair, df)
        except Exception as e:
            return None
        return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_pair, p): p for p in pairs}
        for future in as_completed(futures):
            res = future.result()
            if res:
                data_map[res[0]] = res[1]
                print(f"   Loaded {res[0]} ({len(res[1])} rows)")

    print(f"\nLoaded valid data for {len(data_map)} pairs.")
    
    if not data_map:
        print("No data found! Check hrm/data/arrow/ or hrm/data/coinbase.duckdb")
        return

    # Setup Orchestrator
    orchestrator = Orchestrator(max_workers=8) 
    orchestrator.register_service(GridService())
    orchestrator.register_service(MomentumService())
    orchestrator.register_service(RSIService())
    orchestrator.register_service(TrendService())
    orchestrator.register_service(VolatilityService())
    orchestrator.register_service(VolumeService())
    
    orchestrator.register_composition(
        CompositionModel('composite_alpha')
            .add_signal('momentum', 1.0)
            .add_signal('rsi', 1.0, op='multiply')
            .add_signal('trend', 1.0, op='multiply')
            .add_signal('volatility', 1.0, op='multiply') 
    )
    
    print("\nRunning Signal Pipeline...")
    results = []
    
    for pair, df in data_map.items():
        try:
            res = orchestrator.run(df)
            # signal = res['compositions']['composite_alpha'] # Assuming dict return
            # Using orchestrator might return results asynchronously? 
            # signal_orchestrator.py typically returns a dict with 'signals' and 'compositions'
            
            if 'compositions' in res and 'composite_alpha' in res['compositions']:
                signal = res['compositions']['composite_alpha']
            else:
                # Fallback if structure differs
                print(f"Warn: No signal for {pair}")
                continue

            returns = df['close'].pct_change().shift(-1).fillna(0)
            
            # Align signal to returns
            # signal is typically Series sharing index with df
            common_idx = signal.index.intersection(returns.index)
            signal = signal.loc[common_idx]
            returns = returns.loc[common_idx]
            
            strat_ret = signal * returns
            
            cum_ret = np.prod(1 + strat_ret) - 1
            years = len(df) / (24*12*365) # Effective years
            ann_ret = (1 + cum_ret) ** (1/max(years, 0.1)) - 1
            
            vol = strat_ret.std() * np.sqrt(365*24*12) 
            sharpe = (ann_ret - 0.02) / (vol + 1e-8)
            
            results.append({
                'Symbol': pair,
                'Sharpe': sharpe,
                'CAGR': ann_ret,
                'Total Return': cum_ret,
                'Vol': vol,
                'Rows': len(df)
            })
            print(f"  Processed {pair}: Sharpe={sharpe:.2f}")
        except Exception as e:
            print(f"Error processing {pair}: {e}")
            
    # Output
    results.sort(key=lambda x: x['Sharpe'], reverse=True)
    print("\n" + "="*85)
    print(f"  FINAL RANKINGS (Based on available history)")
    print("="*85)
    print(f"{'Rank':<5} {'Symbol':<15} {'Sharpe':>10} {'CAGR':>10} {'Total Ret':>12} {'Vol':>10} {'Data Pts':>10}")
    print("-" * 85)
    
    for i, r in enumerate(results[:25], 1): 
        print(f"{i:<5} {r['Symbol']:<15} {r['Sharpe']:>10.2f} {r['CAGR']:>10.2%} {r['Total Return']:>12.2%} {r['Vol']:>10.2%} {r['Rows']:>10}")
    print("="*85)

if __name__ == "__main__":
    analyze_rankings()
