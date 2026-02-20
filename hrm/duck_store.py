"""
DuckDB Store - Zero-copy Arrow integration, SQL interface.

DuckDB is a better Arrow wrapper than ArrowStore because:
1. Native Arrow support (zero-copy reads)
2. SQL interface (SQLite compatible)
3. Can query multiple Arrow files as single table
4. Faster analytical queries
5. Drop-in replacement for SQLite
"""

import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import duckdb
import pandas as pd
import pyarrow.feather as feather


class DuckStore:
    """
    DuckDB-backed storage with native Arrow integration.
    
    Features:
    - Zero-copy reads from Arrow files
    - SQL interface (SQLite compatible)
    - Can query multiple Arrow files as a single table
    - Fast analytical queries
    """
    
    def __init__(self, db_path: str = "hrm/data/market.duckdb", arrow_dir: str = "hrm/data/arrow"):
        self.db_path = db_path
        self.arrow_dir = arrow_dir
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self._init_db()
        
    def _init_db(self):
        self.conn = duckdb.connect(self.db_path)
        
    def _slugify(self, symbol: str) -> str:
        return symbol.replace("-", "_").replace("/", "_")
    
    def _get_path(self, symbol: str) -> str:
        return os.path.join(self.arrow_dir, f"{self._slugify(symbol)}.feather")
    
    def upsert(self, symbol: str, df: pd.DataFrame):
        """Upsert data into Arrow file"""
        if df.empty:
            return
        
        path = self._get_path(symbol)
        
        # Ensure consistent schema
        df = df.copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].astype('float64')
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].astype('int64')
        
        if os.path.exists(path):
            try:
                existing_df = pd.read_feather(path)
                if 'time' in existing_df.columns:
                    existing_df['time'] = pd.to_datetime(existing_df['time'])
                    existing_df.set_index('time', inplace=True)
                elif 'timestamp' in existing_df.columns:
                    existing_df['time'] = pd.to_datetime(existing_df['timestamp'], unit='s')
                    existing_df.set_index('time', inplace=True)
                
                # Merge and deduplicate
                merged = pd.concat([existing_df, df])
                merged = merged[~merged.index.duplicated(keep='last')]
                merged.reset_index(drop=False).to_feather(path)
            except:
                df.reset_index(drop=False).to_feather(path)
        else:
            df.reset_index(drop=False).to_feather(path)
    
    def load(self, symbol: str, start: datetime = None, end: datetime = None) -> pd.DataFrame:
        """
        Load data for a symbol using DuckDB's Arrow integration (zero-copy)
        """
        path = self._get_path(symbol)
        
        if not os.path.exists(path):
            return pd.DataFrame()
        
        try:
            # DuckDB can read Arrow files directly with zero-copy
            if start and end:
                start_ts = int(start.timestamp())
                end_ts = int(end.timestamp())
                query = f"""
                    SELECT * FROM read_parquet('{path}')
                    WHERE timestamp >= {start_ts} AND timestamp <= {end_ts}
                    ORDER BY timestamp
                """
            else:
                query = f"SELECT * FROM read_parquet('{path}') ORDER BY timestamp"
            
            df = self.conn.execute(query).fetchdf()
            
            if len(df) > 0:
                df['time'] = pd.to_datetime(df['timestamp'], unit='s')
                df = df.set_index('time')
            
            return df
        except Exception as e:
            # Fallback to pandas
            try:
                df = pd.read_feather(path)
                if 'timestamp' in df.columns:
                    df['time'] = pd.to_datetime(df['timestamp'], unit='s')
                    df = df.set_index('time')
                return df
            except:
                return pd.DataFrame()
    
    def load_all_symbols(self) -> Dict[str, pd.DataFrame]:
        """Load all symbols from Arrow directory"""
        results = {}
        
        for feather_file in Path(self.arrow_dir).glob("*.feather"):
            symbol = feather_file.stem.replace("_", "-")
            try:
                df = pd.read_feather(feather_file)
                if len(df) > 0:
                    if 'timestamp' in df.columns:
                        df['time'] = pd.to_datetime(df['timestamp'], unit='s')
                        df = df.set_index('time')
                    results[symbol] = df
            except:
                pass
        
        return results
    
    def execute(self, query: str, params: list = None):
        """Execute SQL query using DuckDB's Arrow integration"""
        if params:
            return self.conn.execute(query, params)
        return self.conn.execute(query)
    
    def query_arrow_files(self, columns: str = "*", where: str = None) -> pd.DataFrame:
        """
        Query multiple Arrow files as a single table using DuckDB
        """
        if not os.path.exists(self.arrow_dir):
            return pd.DataFrame()
        
        files = list(Path(self.arrow_dir).glob("*.feather"))
        if not files:
            return pd.DataFrame()
        
        # Build query for all files
        file_list = ", ".join([f"'{f}'" for f in files])
        query = f"SELECT {columns} FROM read_parquet([{file_list}])"
        
        if where:
            query += f" WHERE {where}"
        
        return self.conn.execute(query).fetchdf()
