"""
Provenance Import - Import Binance and Coinbase Data with Full Provenance Tracking
===================================================================================

Imports trade pair data from both exchanges with complete provenance:
- Binance source: Binance spot data (CSV/Arrow files)
- Coinbase source: Coinbase historical data
- Full provenance tracking with timestamps, sources, and metadata
"""

import os
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib

try:
    from hrm.duck_store import DuckStore
    HAS_DUCK_STORE = True
except ImportError:
    HAS_DUCK_STORE = False
    print("[ProvenanceImport] DuckStore not available")

@dataclass
class ProvenanceConfig:
    """Configuration for provenance import"""
    # Exchange sources
    binance_sources: List[str] = field(default_factory=lambda: [
        "hrm/data/binance_spot",  # Existing Binance spot data
        "hrm/data/binance_arrow",  # Arrow files from previous system
        "data/binance_spot",  # Alternative location
    ])
    
    coinbase_sources: List[str] = field(default_factory=lambda: [
        "hrm/data/coinbase",  # Coinbase historical data
        "hrm/data/coinbase_arrow",  # Arrow files from previous system
        "data/coinbase",  # Alternative location
    ])
    
    # DuckDB database
    duck_db_path: str = "hrm/data/market.duckdb"
    
    # Import settings
    import_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    data_retention_days: int = 365  # Keep data for 1 year
    
    # Binance pairs to import (basic trade pairs)
    binance_pairs: List[str] = field(default_factory=lambda: [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
        "LINKUSDT", "UNIUSDT", "ATOMUSDT", "LTCUSDT", "BCHUSDT",
        "ETCUSDT", "FILUSDT", "APTUSDT", "OPUSDT", "ARBUSDT",
    ])
    
    # Coinbase pairs to import
    coinbase_pairs: List[str] = field(default_factory=lambda: [
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
        "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "MATIC-USD",
        "LINK-USD", "UNI-USD", "ATOM-USD", "LTC-USD", "BCH-USD",
        "ETC-USD", "FIL-USD", "APT-USD", "OP-USD", "ARB-USD",
    ])

class ProvenanceImport:
    """
    Import Binance and Coinbase data with full provenance tracking
    """
    
    def __init__(self, config: ProvenanceConfig):
        self.config = config
        self.provenance_log = []
        
        if HAS_DUCK_STORE:
            self.duck_store = DuckStore(config.duck_db_path)
            self._init_provenance_tables()
        else:
            self.duck_store = None
        
        print(f"[ProvenanceImport] Initialized with DuckDB: {config.duck_db_path}")
    
    def _init_provenance_tables(self):
        """Initialize provenance tables in DuckDB"""
        if not HAS_DUCK_STORE:
            return
        
        # Provenance metadata table
        self.duck_store.conn.execute("""
            CREATE TABLE IF NOT EXISTS provenance_metadata (
                id INTEGER PRIMARY KEY,
                source_exchange TEXT,
                source_file TEXT,
                import_timestamp TIMESTAMP,
                data_timestamp_start TIMESTAMP,
                data_timestamp_end TIMESTAMP,
                row_count INTEGER,
                data_hash TEXT,
                config JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Binance source table
        self.duck_store.conn.execute("""
            CREATE TABLE IF NOT EXISTS binance_source (
                symbol TEXT,
                timestamp TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                source_file TEXT,
                import_timestamp TIMESTAMP,
                data_hash TEXT,
                PRIMARY KEY (symbol, timestamp)
            )
        """)
        
        # Coinbase source table
        self.duck_store.conn.execute("""
            CREATE TABLE IF NOT EXISTS coinbase_source (
                symbol TEXT,
                timestamp TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                source_file TEXT,
                import_timestamp TIMESTAMP,
                data_hash TEXT,
                PRIMARY KEY (symbol, timestamp)
            )
        """)
        
        # Unified market data view
        self.duck_store.conn.execute("""
            CREATE VIEW IF NOT EXISTS market_data AS
            SELECT 
                symbol,
                timestamp,
                open,
                high,
                low,
                close,
                volume,
                'binance' as source_exchange,
                source_file,
                import_timestamp
            FROM binance_source
            UNION ALL
            SELECT 
                symbol,
                timestamp,
                open,
                high,
                low,
                close,
                volume,
                'coinbase' as source_exchange,
                source_file,
                import_timestamp
            FROM coinbase_source
        """)
        
        # Create indexes for provenance queries
        self.duck_store.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_binance_symbol ON binance_source(symbol)
        """)
        self.duck_store.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_binance_timestamp ON binance_source(timestamp)
        """)
        self.duck_store.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_coinbase_symbol ON coinbase_source(symbol)
        """)
        self.duck_store.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_coinbase_timestamp ON coinbase_source(timestamp)
        """)
        
        print("[ProvenanceImport] Initialized provenance tables in DuckDB")
    
    def compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of dataframe for provenance tracking"""
        # Sort for consistent hashing
        df_sorted = df.sort_index()
        # Convert to string and hash
        data_str = df_sorted.to_csv(index=False)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def import_binance_data(self) -> Dict[str, Any]:
        """Import Binance data from arrow files with provenance"""
        print(f"\n{'='*80}")
        print("IMPORTING BINANCE DATA WITH PROVENANCE")
        print(f"{'='*80}")
        
        if not HAS_DUCK_STORE:
            return {"error": "DuckStore not available"}
        
        binance_results = {
            'total_pairs': 0,
            'total_rows': 0,
            'failed_pairs': [],
            'sources_found': []
        }
        
        # Try different source locations
        for source_path in self.config.binance_sources:
            source_path = Path(source_path)
            
            if not source_path.exists():
                print(f"  ℹ️  Source not found: {source_path}")
                continue
            
            binance_results['sources_found'].append(str(source_path))
            print(f"\n  Checking source: {source_path}")
            
            # Find feather files
            feather_files = list(source_path.glob("*.feather"))
            print(f"  Found {len(feather_files)} feather files")
            
            for feather_file in feather_files:
                # Extract symbol from filename
                symbol_raw = feather_file.stem
                # Convert to Binance format (BTCUSDT)
                symbol = symbol_raw.replace("_", "").replace("-", "").upper()
                
                # Check if this is a valid Binance pair
                if symbol not in self.config.binance_pairs:
                    continue
                
                print(f"\n    Processing {symbol}...")
                
                try:
                    # Load feather file
                    df = pd.read_feather(feather_file)
                    
                    if df.empty:
                        print(f"      ⚠️  Empty data")
                        binance_results['failed_pairs'].append(f"{symbol} (empty)")
                        continue
                    
                    # Ensure proper schema
                    df = df.copy()
                    
                    # Check and convert timestamp
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                        df.set_index('timestamp', inplace=True)
                    elif 'time' in df.columns:
                        df['time'] = pd.to_datetime(df['time'])
                        df.set_index('time', inplace=True)
                    
                    # Ensure required columns exist
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    for col in required_cols:
                        if col not in df.columns:
                            df[col] = 0.0
                    
                    # Ensure proper types
                    for col in required_cols:
                        df[col] = df[col].astype('float64')
                    
                    # Compute data hash
                    data_hash = self.compute_data_hash(df[required_cols])
                    
                    # Insert into DuckDB
                    rows_inserted = 0
                    for timestamp, row in df.iterrows():
                        try:
                            self.duck_store.conn.execute("""
                                INSERT OR REPLACE INTO binance_source 
                                (symbol, timestamp, open, high, low, close, volume, 
                                 source_file, import_timestamp, data_hash)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                symbol,
                                timestamp,
                                float(row['open']),
                                float(row['high']),
                                float(row['low']),
                                float(row['close']),
                                float(row['volume']),
                                str(feather_file),
                                self.config.import_timestamp,
                                data_hash
                            ))
                            rows_inserted += 1
                        except Exception as e:
                            print(f"      Warning: Failed to insert row {timestamp}: {e}")
                            break
                    
                    # Log provenance
                    provenance_entry = {
                        'source_exchange': 'binance',
                        'source_file': str(feather_file),
                        'import_timestamp': self.config.import_timestamp,
                        'data_timestamp_start': df.index.min().isoformat() if len(df) > 0 else None,
                        'data_timestamp_end': df.index.max().isoformat() if len(df) > 0 else None,
                        'row_count': rows_inserted,
                        'data_hash': data_hash,
                        'symbol': symbol,
                        'config': self.config.__dict__
                    }
                    
                    self.provenance_log.append(provenance_entry)
                    
                    # Insert provenance metadata
                    self.duck_store.conn.execute("""
                        INSERT INTO provenance_metadata 
                        (source_exchange, source_file, import_timestamp, 
                         data_timestamp_start, data_timestamp_end, row_count, data_hash, config)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        provenance_entry['source_exchange'],
                        provenance_entry['source_file'],
                        provenance_entry['import_timestamp'],
                        provenance_entry['data_timestamp_start'],
                        provenance_entry['data_timestamp_end'],
                        provenance_entry['row_count'],
                        provenance_entry['data_hash'],
                        json.dumps(provenance_entry['config'])
                    ))
                    
                    binance_results['total_pairs'] += 1
                    binance_results['total_rows'] += rows_inserted
                    
                    print(f"      ✓ Imported {rows_inserted} rows")
                    
                except Exception as e:
                    print(f"      ✗ Failed: {e}")
                    binance_results['failed_pairs'].append(f"{symbol} ({e})")
        
        # Save provenance log
        self.save_provenance_log()
        
        return binance_results
    
    def import_coinbase_data(self) -> Dict[str, Any]:
        """Import Coinbase data with provenance"""
        print(f"\n{'='*80}")
        print("IMPORTING COINBASE DATA WITH PROVENANCE")
        print(f"{'='*80}")
        
        if not HAS_DUCK_STORE:
            return {"error": "DuckStore not available"}
        
        coinbase_results = {
            'total_pairs': 0,
            'total_rows': 0,
            'failed_pairs': [],
            'sources_found': []
        }
        
        # Try different source locations
        for source_path in self.config.coinbase_sources:
            source_path = Path(source_path)
            
            if not source_path.exists():
                print(f"  ℹ️  Source not found: {source_path}")
                continue
            
            coinbase_results['sources_found'].append(str(source_path))
            print(f"\n  Checking source: {source_path}")
            
            # Find feather files
            feather_files = list(source_path.glob("*.feather"))
            print(f"  Found {len(feather_files)} feather files")
            
            for feather_file in feather_files:
                # Extract symbol from filename
                symbol_raw = feather_file.stem
                # Convert to Coinbase format (BTC-USD)
                symbol = symbol_raw.replace("_", "-")
                
                # Check if this is a valid Coinbase pair
                if symbol not in self.config.coinbase_pairs:
                    continue
                
                print(f"\n    Processing {symbol}...")
                
                try:
                    # Load feather file
                    df = pd.read_feather(feather_file)
                    
                    if df.empty:
                        print(f"      ⚠️  Empty data")
                        coinbase_results['failed_pairs'].append(f"{symbol} (empty)")
                        continue
                    
                    # Ensure proper schema
                    df = df.copy()
                    
                    # Check and convert timestamp
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                        df.set_index('timestamp', inplace=True)
                    elif 'time' in df.columns:
                        df['time'] = pd.to_datetime(df['time'])
                        df.set_index('time', inplace=True)
                    
                    # Ensure required columns exist
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    for col in required_cols:
                        if col not in df.columns:
                            df[col] = 0.0
                    
                    # Ensure proper types
                    for col in required_cols:
                        df[col] = df[col].astype('float64')
                    
                    # Compute data hash
                    data_hash = self.compute_data_hash(df[required_cols])
                    
                    # Insert into DuckDB
                    rows_inserted = 0
                    for timestamp, row in df.iterrows():
                        try:
                            self.duck_store.conn.execute("""
                                INSERT OR REPLACE INTO coinbase_source 
                                (symbol, timestamp, open, high, low, close, volume, 
                                 source_file, import_timestamp, data_hash)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                symbol,
                                timestamp,
                                float(row['open']),
                                float(row['high']),
                                float(row['low']),
                                float(row['close']),
                                float(row['volume']),
                                str(feather_file),
                                self.config.import_timestamp,
                                data_hash
                            ))
                            rows_inserted += 1
                        except Exception as e:
                            print(f"      Warning: Failed to insert row {timestamp}: {e}")
                            break
                    
                    # Log provenance
                    provenance_entry = {
                        'source_exchange': 'coinbase',
                        'source_file': str(feather_file),
                        'import_timestamp': self.config.import_timestamp,
                        'data_timestamp_start': df.index.min().isoformat() if len(df) > 0 else None,
                        'data_timestamp_end': df.index.max().isoformat() if len(df) > 0 else None,
                        'row_count': rows_inserted,
                        'data_hash': data_hash,
                        'symbol': symbol,
                        'config': self.config.__dict__
                    }
                    
                    self.provenance_log.append(provenance_entry)
                    
                    # Insert provenance metadata
                    self.duck_store.conn.execute("""
                        INSERT INTO provenance_metadata 
                        (source_exchange, source_file, import_timestamp, 
                         data_timestamp_start, data_timestamp_end, row_count, data_hash, config)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        provenance_entry['source_exchange'],
                        provenance_entry['source_file'],
                        provenance_entry['import_timestamp'],
                        provenance_entry['data_timestamp_start'],
                        provenance_entry['data_timestamp_end'],
                        provenance_entry['row_count'],
                        provenance_entry['data_hash'],
                        json.dumps(provenance_entry['config'])
                    ))
                    
                    coinbase_results['total_pairs'] += 1
                    coinbase_results['total_rows'] += rows_inserted
                    
                    print(f"      ✓ Imported {rows_inserted} rows")
                    
                except Exception as e:
                    print(f"      ✗ Failed: {e}")
                    coinbase_results['failed_pairs'].append(f"{symbol} ({e})")
        
        # Save provenance log
        self.save_provenance_log()
        
        return coinbase_results
    
    def save_provenance_log(self):
        """Save provenance log to DuckDB and file"""
        if not self.provenance_log:
            return
        
        # Save to file
        log_file = Path("hrm/data/provenance_log.json")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_file, 'w') as f:
            json.dump(self.provenance_log, f, indent=2, default=str)
        
        print(f"\n  Provenance log saved to: {log_file}")
    
    def query_provenance(self, symbol: Optional[str] = None, 
                        source_exchange: Optional[str] = None) -> pd.DataFrame:
        """Query provenance metadata"""
        if not HAS_DUCK_STORE:
            return pd.DataFrame()
        
        query = """
            SELECT * FROM provenance_metadata
            WHERE 1=1
        """
        
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if source_exchange:
            query += " AND source_exchange = ?"
            params.append(source_exchange)
        
        query += " ORDER BY import_timestamp DESC"
        
        return self.duck_store.conn.execute(query, params).fetchdf()
    
    def generate_report(self, binance_results: Dict[str, Any], 
                       coinbase_results: Dict[str, Any]) -> str:
        """Generate import report"""
        report = []
        report.append("="*80)
        report.append("PROVENANCE IMPORT REPORT")
        report.append("="*80)
        report.append("")
        
        report.append("BINANCE IMPORT:")
        report.append(f"  Total pairs imported: {binance_results.get('total_pairs', 0)}")
        report.append(f"  Total rows imported: {binance_results.get('total_rows', 0)}")
        report.append(f"  Sources found: {len(binance_results.get('sources_found', []))}")
        if binance_results.get('failed_pairs'):
            report.append(f"  Failed pairs: {len(binance_results['failed_pairs'])}")
        report.append("")
        
        report.append("COINBASE IMPORT:")
        report.append(f"  Total pairs imported: {coinbase_results.get('total_pairs', 0)}")
        report.append(f"  Total rows imported: {coinbase_results.get('total_rows', 0)}")
        report.append(f"  Sources found: {len(coinbase_results.get('sources_found', []))}")
        if coinbase_results.get('failed_pairs'):
            report.append(f"  Failed pairs: {len(coinbase_results['failed_pairs'])}")
        report.append("")
        
        report.append("PROVENANCE TRACKING:")
        report.append(f"  Import timestamp: {self.config.import_timestamp}")
        report.append(f"  Total entries logged: {len(self.provenance_log)}")
        report.append(f"  DuckDB database: {self.config.duck_db_path}")
        report.append("")
        
        report.append("DATA STRUCTURE:")
        report.append("  binance_source - Binance spot data with provenance")
        report.append("  coinbase_source - Coinbase historical data with provenance")
        report.append("  provenance_metadata - Complete provenance tracking")
        report.append("  market_data - Unified view of all data")
        report.append("")
        
        report.append("QUERIES:")
        report.append("  SELECT * FROM provenance_metadata WHERE source_exchange = 'binance'")
        report.append("  SELECT * FROM provenance_metadata WHERE symbol = 'BTC-USD'")
        report.append("  SELECT * FROM market_data WHERE symbol = 'BTC-USD' ORDER BY timestamp")
        report.append("  SELECT * FROM binance_source WHERE timestamp >= '2024-01-01'")
        
        return "\n".join(report)

# Example usage
async def main():
    config = ProvenanceConfig(
        binance_sources=[
            "hrm/data/binance_spot",
            "hrm/data/binance_arrow",
        ],
        coinbase_sources=[
            "hrm/data/coinbase",
            "hrm/data/coinbase_arrow",
        ],
        duck_db_path="hrm/data/market.duckdb"
    )
    
    importer = ProvenanceImport(config)
    
    print(f"\n{'='*80}")
    print("PROVENANCE IMPORT - BINANCE AND COINBASE DATA")
    print(f"{'='*80}\n")
    
    # Import Binance data
    binance_results = importer.import_binance_data()
    
    # Import Coinbase data
    coinbase_results = importer.import_coinbase_data()
    
    # Generate report
    report = importer.generate_report(binance_results, coinbase_results)
    print(f"\n{report}")
    
    # Query provenance
    print(f"\n{'='*80}")
    print("PROVENANCE QUERY EXAMPLES")
    print(f"{'='*80}\n")
    
    provenance_df = importer.query_provenance()
    if not provenance_df.empty:
        print(f"Total provenance entries: {len(provenance_df)}")
        print("\nFirst 5 entries:")
        print(provenance_df.head()[['source_exchange', 'symbol', 'row_count', 'data_timestamp_start', 'data_timestamp_end']])
    
    print(f"\n{'='*80}")
    print("✅ IMPORT COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())