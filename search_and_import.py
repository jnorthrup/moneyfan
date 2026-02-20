"""
Search and Import - Find existing arrow files and import them
==============================================================

Searches for existing arrow files from previous systems and imports them
with proper provenance tracking.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    from hrm.duck_store import DuckStore
    HAS_DUCK_STORE = True
except ImportError:
    HAS_DUCK_STORE = False

class ArrowFileFinder:
    """Find and import arrow files from various locations"""
    
    def __init__(self, duck_db_path: str = "hrm/data/market.duckdb"):
        self.duck_db_path = duck_db_path
        self.found_files = {
            'binance': [],
            'coinbase': [],
            'other': []
        }
        
        if HAS_DUCK_STORE:
            self.duck_store = DuckStore(duck_db_path)
            self._init_source_tables()
        else:
            self.duck_store = None
    
    def _init_source_tables(self):
        """Initialize source tables in DuckDB"""
        if not HAS_DUCK_STORE:
            return
        
        # Create source tracking table
        self.duck_store.conn.execute("""
            CREATE TABLE IF NOT EXISTS source_files (
                id INTEGER PRIMARY KEY,
                exchange TEXT,
                file_path TEXT,
                file_size INTEGER,
                import_timestamp TIMESTAMP,
                status TEXT,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        print(f"[ArrowFileFinder] Initialized source tracking table")
    
    def search_arrow_files(self, search_paths: List[str]) -> Dict[str, Any]:
        """Search for arrow files in given paths"""
        print(f"\n{'='*80}")
        print("SEARCHING FOR ARROW FILES")
        print(f"{'='*80}\n")
        
        for search_path in search_paths:
            path = Path(search_path)
            if not path.exists():
                print(f"  ⚠️  Path not found: {path}")
                continue
            
            print(f"  Searching: {path}")
            
            # Search for feather files (arrow format)
            feather_files = list(path.glob("**/*.feather"))
            print(f"    Found {len(feather_files)} feather files")
            
            for feather_file in feather_files:
                # Determine exchange based on path or filename
                file_path_str = str(feather_file)
                
                if 'binance' in file_path_str.lower():
                    self.found_files['binance'].append(feather_file)
                    print(f"      📁 Binance: {feather_file.name}")
                elif 'coinbase' in file_path_str.lower():
                    self.found_files['coinbase'].append(feather_file)
                    print(f"      📁 Coinbase: {feather_file.name}")
                else:
                    self.found_files['other'].append(feather_file)
                    print(f"      📁 Other: {feather_file.name}")
        
        # Search for numpy memmap files too
        for search_path in search_paths:
            path = Path(search_path)
            if not path.exists():
                continue
            
            # Look for vector_store directories
            vector_dirs = list(path.glob("**/vector_store"))
            for vector_dir in vector_dirs:
                print(f"\n  Found vector_store directory: {vector_dir}")
                
                # Look for vectors.dat
                vectors_dat = vector_dir / "vectors.dat"
                if vectors_dat.exists():
                    print(f"    Found vectors.dat: {vectors_dat}")
                    self.found_files['other'].append(vectors_dat)
        
        return self.found_files
    
    def import_binance_file(self, file_path: Path, exchange: str = "binance") -> Dict[str, Any]:
        """Import a single Binance feather file with provenance"""
        if not HAS_DUCK_STORE:
            return {"error": "DuckStore not available"}
        
        try:
            print(f"\n    Importing: {file_path.name}")
            
            # Extract symbol
            symbol_raw = file_path.stem
            symbol = symbol_raw.replace("_", "").replace("-", "").upper()
            
            # Load data
            df = pd.read_feather(file_path)
            
            if df.empty:
                return {"error": "Empty file", "symbol": symbol}
            
            # Ensure proper schema
            df = df.copy()
            
            # Convert timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
            elif 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
            
            # Insert into binance_source
            rows_inserted = 0
            for timestamp, row in df.iterrows():
                try:
                    self.duck_store.conn.execute("""
                        INSERT OR REPLACE INTO binance_source 
                        (symbol, timestamp, open, high, low, close, volume, 
                         source_file, import_timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        timestamp,
                        float(row.get('open', 0.0)),
                        float(row.get('high', 0.0)),
                        float(row.get('low', 0.0)),
                        float(row.get('close', 0.0)),
                        float(row.get('volume', 0.0)),
                        str(file_path),
                        pd.Timestamp.now().isoformat()
                    ))
                    rows_inserted += 1
                except Exception as e:
                    break
            
            # Log to source_files
            self.duck_store.conn.execute("""
                INSERT INTO source_files (exchange, file_path, file_size, import_timestamp, status)
                VALUES (?, ?, ?, ?, ?)
            """, (
                exchange,
                str(file_path),
                file_path.stat().st_size,
                pd.Timestamp.now().isoformat(),
                'imported'
            ))
            
            return {
                "symbol": symbol,
                "rows_imported": rows_inserted,
                "file_path": str(file_path),
                "status": "success"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "file_path": str(file_path),
                "status": "failed"
            }
    
    def import_coinbase_file(self, file_path: Path, exchange: str = "coinbase") -> Dict[str, Any]:
        """Import a single Coinbase feather file with provenance"""
        if not HAS_DUCK_STORE:
            return {"error": "DuckStore not available"}
        
        try:
            print(f"\n    Importing: {file_path.name}")
            
            # Extract symbol
            symbol_raw = file_path.stem
            symbol = symbol_raw.replace("_", "-")
            
            # Load data
            df = pd.read_feather(file_path)
            
            if df.empty:
                return {"error": "Empty file", "symbol": symbol}
            
            # Ensure proper schema
            df = df.copy()
            
            # Convert timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
            elif 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
            
            # Insert into coinbase_source
            rows_inserted = 0
            for timestamp, row in df.iterrows():
                try:
                    self.duck_store.conn.execute("""
                        INSERT OR REPLACE INTO coinbase_source 
                        (symbol, timestamp, open, high, low, close, volume, 
                         source_file, import_timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        timestamp,
                        float(row.get('open', 0.0)),
                        float(row.get('high', 0.0)),
                        float(row.get('low', 0.0)),
                        float(row.get('close', 0.0)),
                        float(row.get('volume', 0.0)),
                        str(file_path),
                        pd.Timestamp.now().isoformat()
                    ))
                    rows_inserted += 1
                except Exception as e:
                    break
            
            # Log to source_files
            self.duck_store.conn.execute("""
                INSERT INTO source_files (exchange, file_path, file_size, import_timestamp, status)
                VALUES (?, ?, ?, ?, ?)
            """, (
                exchange,
                str(file_path),
                file_path.stat().st_size,
                pd.Timestamp.now().isoformat(),
                'imported'
            ))
            
            return {
                "symbol": symbol,
                "rows_imported": rows_inserted,
                "file_path": str(file_path),
                "status": "success"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "file_path": str(file_path),
                "status": "failed"
            }
    
    def import_all_found(self) -> Dict[str, Any]:
        """Import all found files"""
        if not HAS_DUCK_STORE:
            return {"error": "DuckStore not available"}
        
        print(f"\n{'='*80}")
        print("IMPORTING ALL FOUND FILES")
        print(f"{'='*80}\n")
        
        results = {
            'binance': {'imported': 0, 'failed': 0, 'details': []},
            'coinbase': {'imported': 0, 'failed': 0, 'details': []},
            'other': {'imported': 0, 'failed': 0, 'details': []}
        }
        
        # Import Binance files
        if self.found_files['binance']:
            print(f"\nImporting {len(self.found_files['binance'])} Binance files...")
            for file_path in self.found_files['binance']:
                result = self.import_binance_file(file_path, 'binance')
                results['binance']['details'].append(result)
                
                if 'error' in result:
                    results['binance']['failed'] += 1
                    print(f"      ✗ Failed: {file_path.name} - {result['error']}")
                else:
                    results['binance']['imported'] += 1
                    print(f"      ✓ Imported: {result['symbol']} - {result['rows_imported']} rows")
        
        # Import Coinbase files
        if self.found_files['coinbase']:
            print(f"\nImporting {len(self.found_files['coinbase'])} Coinbase files...")
            for file_path in self.found_files['coinbase']:
                result = self.import_coinbase_file(file_path, 'coinbase')
                results['coinbase']['details'].append(result)
                
                if 'error' in result:
                    results['coinbase']['failed'] += 1
                    print(f"      ✗ Failed: {file_path.name} - {result['error']}")
                else:
                    results['coinbase']['imported'] += 1
                    print(f"      ✓ Imported: {result['symbol']} - {result['rows_imported']} rows")
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate import report"""
        report = []
        report.append("="*80)
        report.append("ARROW FILE IMPORT REPORT")
        report.append("="*80)
        report.append("")
        
        report.append("FILES FOUND:")
        for exchange in ['binance', 'coinbase', 'other']:
            count = len(self.found_files.get(exchange, []))
            if count > 0:
                report.append(f"  {exchange.upper()}: {count} files")
        report.append("")
        
        if 'binance' in results:
            report.append("BINANCE IMPORT:")
            report.append(f"  Imported: {results['binance']['imported']} files")
            report.append(f"  Failed: {results['binance']['failed']} files")
            report.append("")
        
        if 'coinbase' in results:
            report.append("COINBASE IMPORT:")
            report.append(f"  Imported: {results['coinbase']['imported']} files")
            report.append(f"  Failed: {results['coinbase']['failed']} files")
            report.append("")
        
        report.append("DUCKDB DATABASE:")
        report.append(f"  {self.duck_db_path}")
        report.append("")
        
        report.append("QUERIES:")
        report.append("  SELECT * FROM source_files WHERE exchange = 'binance'")
        report.append("  SELECT * FROM binance_source LIMIT 10")
        report.append("  SELECT * FROM coinbase_source LIMIT 10")
        
        return "\n".join(report)

# Example usage
async def main():
    # Search paths
    search_paths = [
        "hrm/data/binance_spot",
        "hrm/data/binance_arrow",
        "hrm/data/coinbase",
        "hrm/data/coinbase_arrow",
        "data/binance_spot",
        "data/coinbase",
        "hrm/data/arrow",  # Legacy ArrowStore location
    ]
    
    finder = ArrowFileFinder(duck_db_path="hrm/data/market.duckdb")
    
    # Search for files
    found_files = finder.search_arrow_files(search_paths)
    
    # Import found files
    results = finder.import_all_found()
    
    # Generate report
    report = finder.generate_report(results)
    print(f"\n{report}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())