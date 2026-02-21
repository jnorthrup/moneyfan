#!/usr/bin/env python3
"""
Binance Vision ZIP to Parquet Ingestor
======================================

Downloads monthly/daily zipped CSV archives directly from data.binance.vision 
into memory, parses them with Pandas, and upserts into our native PyArrow Parquet 
store. Completely bypasses the need for an intermediate `mpdata` file tree.

Usage:
    python hrm/data/fetch_vision.py --symbol BTCUSDT --timeframe 1m --start_year 2021
"""

import os
import io
import zipfile
import argparse
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

# Binance Vision Base URLs
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"

# The header order matched from your fetchklines.sh `echo` command
BINANCE_CSV_HEADERS = [
    'Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time',
    'Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume',
    'Taker_buy_quote_asset_volume', 'Ignore'
]

COL_MAP = {
    'Open_time': 'timestamp',
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume',
    'Number_of_trades': 'trades',
    'Quote_asset_volume': 'quote_volume',
    'Taker_buy_base_asset_volume': 'taker_buy_base',
    'Taker_buy_quote_asset_volume': 'taker_buy_quote'
}

def download_and_ingest(symbol: str, timeframe: str, start_year: int, end_year: int, data_dir: str):
    """
    Downloads monthly ZIP archives from Binance Vision and streams them directly into Parquet.
    """
    import tempfile
    import shutil
    import platform
    import sys
    
    out_path = Path(data_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    slug = symbol.replace("-", "_").replace("/", "_")
    
    parquet_target = out_path / f"{slug}.parquet"
    if timeframe != "5m":
        parquet_target = out_path / f"{slug}_{timeframe}.parquet"

    print(f"Targeting Parquet Datastore: {parquet_target}")
    
    session = requests.Session()
    session.headers.update({'User-Agent': 'moneyfan-vision-loader/1.0'})

    current_year = datetime.now().year
    end_year = min(end_year, current_year)
    
    # Use macOS private cache if applicable, otherwise POSIX tmp
    if platform.system() == "Darwin":
        cache_base = Path.home() / "Library" / "Caches" / "moneyfan_vision"
        cache_base.mkdir(parents=True, exist_ok=True)
        tmpdir_context = tempfile.TemporaryDirectory(dir=cache_base)
    else:
        tmpdir_context = tempfile.TemporaryDirectory(prefix="moneyfan_vision_")

    with tmpdir_context as tmpdir:
        tmp_path = Path(tmpdir)
        
        # If we already have a dataset, copy it to tmp for isolated operations
        tmp_target = tmp_path / parquet_target.name
        if parquet_target.exists():
            shutil.copy2(parquet_target, tmp_target)
            
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                month_str = f"{month:02d}"
                filename = f"{symbol}-{timeframe}-{year}-{month_str}.zip"
                url = f"{BASE_URL}/{symbol}/{timeframe}/{filename}"
            
            # Skip future months
            if year == current_year and month > datetime.now().month:
                break
                
            print(f"Fetching {filename}...")
            
            try:
                response = session.get(url, timeout=30)
                
                if response.status_code == 404:
                    print(f"  [404] {filename} not found (might not exist yet)")
                    continue
                    
                response.raise_for_status()
                
                # Unzip in memory
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    csv_filename = z.namelist()[0]
                    with z.open(csv_filename) as f:
                        # Binance CSVs are often headerless. We pass the headers explicitly.
                        df = pd.read_csv(f, names=BINANCE_CSV_HEADERS)
                        
                        # Apply mapping
                        df = df.rename(columns=COL_MAP)
                        
                        # Set Time Index
                        df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df = df.set_index('time')
                        
                        # Drop Ignore / Close_time to save space and match the trainer
                        if 'Ignore' in df.columns:
                            df = df.drop(columns=['Ignore'])
                        if 'Close_time' in df.columns:
                            df = df.drop(columns=['Close_time'])
                        
                        # Upsert lazily in the tmp cache atomic sandbox
                        if tmp_target.exists():
                            try:
                                existing_df = pd.read_parquet(tmp_target)
                                merged = pd.concat([existing_df, df])
                                merged = merged[~merged.index.duplicated(keep='last')].sort_index()
                                merged.to_parquet(tmp_target, engine='pyarrow')
                            except Exception as e:
                                print(f"  [Error] Failed to merge Parquet block: {e}")
                                df.to_parquet(tmp_target, engine='pyarrow')
                        else:
                            df.to_parquet(tmp_target, engine='pyarrow')
                            
                        print(f"  [Success] Merged {len(df)} rows.")
                        
            except requests.exceptions.RequestException as e:
                print(f"  [Error] Failed to download {filename}: {e}")
            except zipfile.BadZipFile:
                print(f"  [Error] Bad zip file from {url}")
            except Exception as e:
                print(f"  [Error] Unexpected error processing {filename}: {e}")
                
        # Atomic commit from POSIX tmp to the requested target
        if tmp_target.exists():
            shutil.copy2(tmp_target, parquet_target)
            print(f"Atomically committed {parquet_target}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Directly pipe Binance Vision ZIP CSVs into Parquet.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading Symbol (e.g. BTCUSDT)")
    parser.add_argument("--timeframe", default="1m", help="Candle timeframe (e.g. 1m, 5m, 1h)")
    parser.add_argument("--start_year", type=int, default=2021, help="Start year to fetch")
    parser.add_argument("--end_year", type=int, default=2030, help="End year to fetch")
    parser.add_argument("--dest", default="hrm/data/parquet", help="Dest dir for Parquet files")
    
    args = parser.parse_args()
    download_and_ingest(args.symbol, args.timeframe, args.start_year, args.end_year, args.dest)
    print("\nVision to Parquet Ingestion Complete!")
