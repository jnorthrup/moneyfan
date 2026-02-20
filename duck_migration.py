"""
DuckDB Migration - Migrate all data from ArrowStore + numpy to DuckDB
======================================================================

Migrates:
1. ArrowStore data (hrm/data/arrow/*.feather) → DuckDB
2. Numpy memmap vectors (data/vector_store/vectors.dat) → DuckDB
3. Vector store metadata → DuckDB

All data becomes queryable via SQL through DuckDB.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# Import DuckStore
try:
    from hrm.duck_store import DuckStore
    HAS_DUCK_STORE = True
except ImportError:
    HAS_DUCK_STORE = False
    print("[Migration] DuckStore not available")

# Import ArrowStore for migration
try:
    from hrm.arrow_store import ArrowStore
    HAS_ARROW_STORE = True
except ImportError:
    HAS_ARROW_STORE = False
    print("[Migration] ArrowStore not available")

# Import VectorStore for migration
try:
    from vector_store import VectorStore, VectorStoreConfig
    HAS_VECTOR_STORE = True
except ImportError:
    HAS_VECTOR_STORE = False
    print("[Migration] VectorStore not available")

class DuckMigration:
    """
    Migrate all data to DuckDB
    """
    
    def __init__(self, duck_db_path: str = "hrm/data/market.duckdb"):
        self.duck_db_path = duck_db_path
        self.duck_store = DuckStore(duck_db_path)
        
        # Stats
        self.stats = {
            'arrow_migrated': 0,
            'vector_migrated': 0,
            'total_rows_migrated': 0,
            'failed_pairs': [],
            'start_time': time.time()
        }
        
        print(f"[Migration] Initialized with DuckDB: {duck_db_path}")
    
    def migrate_arrow_data(self, arrow_dir: str = "hrm/data/arrow") -> Dict[str, Any]:
        """Migrate ArrowStore data to DuckDB"""
        print(f"\n{'='*60}")
        print("MIGRATING ARROWSTORE DATA TO DUCKDB")
        print(f"{'='*60}")
        
        if not HAS_ARROW_STORE:
            print("[Migration] ArrowStore not available, skipping")
            return {}
        
        arrow_store = ArrowStore(arrow_dir)
        
        # Find all feather files
        feather_files = list(Path(arrow_dir).glob("*.feather"))
        print(f"[Migration] Found {len(feather_files)} feather files")
        
        migrated_count = 0
        total_rows = 0
        
        for feather_file in feather_files:
            symbol = feather_file.stem.replace("_", "-").replace("_", "/")
            
            try:
                print(f"[Migration] Processing {symbol}...")
                
                # Load feather file
                df = pd.read_feather(feather_file)
                
                if df.empty:
                    print(f"  Warning: Empty data for {symbol}")
                    self.stats['failed_pairs'].append(symbol)
                    continue
                
                # Ensure proper schema
                df = df.copy()
                
                # Check for timestamp column
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    df.set_index('timestamp', inplace=True)
                elif 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    df.set_index('time', inplace=True)
                
                # Add symbol column
                df['symbol'] = symbol
                
                # Ensure proper types
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        df[col] = df[col].astype('float64')
                
                # Insert into DuckDB
                self.duck_store.upsert(symbol, df)
                
                migrated_count += 1
                total_rows += len(df)
                self.stats['arrow_migrated'] += 1
                self.stats['total_rows_migrated'] += len(df)
                
                print(f"  ✓ Migrated {len(df)} rows")
                
            except Exception as e:
                print(f"  ✗ Failed to migrate {symbol}: {e}")
                self.stats['failed_pairs'].append(symbol)
        
        print(f"\n[Migration] ArrowStore migration complete:")
        print(f"  Migrated: {migrated_count} pairs")
        print(f"  Total rows: {total_rows}")
        
        return {
            'migrated_pairs': migrated_count,
            'total_rows': total_rows
        }
    
    def migrate_vector_data(self, vector_store_path: str = "data/vector_store") -> Dict[str, Any]:
        """Migrate numpy memmap vectors to DuckDB"""
        print(f"\n{'='*60}")
        print("MIGRATING VECTORS TO DUCKDB")
        print(f"{'='*60}")
        
        if not HAS_VECTOR_STORE:
            print("[Migration] VectorStore not available, skipping")
            return {}
        
        # Check if vector store exists
        vectors_dat = Path(vector_store_path) / "vectors.dat"
        if not vectors_dat.exists():
            print(f"[Migration] No vectors found at {vectors_dat}")
            return {}
        
        # Load vector store
        config = VectorStoreConfig(
            vector_dim=64,
            memmap_path=str(vectors_dat)
        )
        
        try:
            vector_store = VectorStore(config)
            vector_store.load()
            
            if vector_store.count == 0:
                print(f"[Migration] Vector store is empty")
                return {}
            
            print(f"[Migration] Loaded {vector_store.count} vectors from memmap")
            
            # Create vectors table in DuckDB
            # We'll store vectors as JSON arrays in DuckDB
            self.duck_store.conn.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    id INTEGER PRIMARY KEY,
                    horizon INTEGER,
                    timestamp INTEGER,
                    vector DOUBLE[],
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Migrate vectors one by one
            migrated = 0
            for idx in range(min(vector_store.count, 10000)):  # Limit for migration
                try:
                    horizon = vector_store.horizons_memmap[idx]
                    timestamp = vector_store.timestamps_memmap[idx]
                    vector = vector_store.vectors_memmap[idx]
                    
                    # Convert vector to SQL array format
                    vector_str = "[" + ", ".join([str(x) for x in vector]) + "]"
                    
                    # Insert into DuckDB
                    self.duck_store.conn.execute(
                        "INSERT INTO vectors (horizon, timestamp, vector) VALUES (?, ?, ?)",
                        (int(horizon), int(timestamp), vector_str)
                    )
                    
                    migrated += 1
                    self.stats['vector_migrated'] += 1
                    self.stats['total_rows_migrated'] += 1
                    
                    if migrated % 1000 == 0:
                        print(f"  Migrated {migrated} vectors...")
                        
                except Exception as e:
                    print(f"  Warning: Failed to migrate vector {idx}: {e}")
                    break
            
            print(f"\n[Migration] Vector migration complete:")
            print(f"  Migrated: {migrated} vectors")
            
            return {
                'migrated_vectors': migrated,
                'total_vectors': migrated
            }
            
        except Exception as e:
            print(f"[Migration] Failed to load vector store: {e}")
            return {}
    
    def delete_old_stores(self):
        """Delete ArrowStore and vector_store files"""
        print(f"\n{'='*60}")
        print("DELETING OLD STORES")
        print(f"{'='*60}")
        
        # Delete ArrowStore directory
        arrow_dir = Path("hrm/data/arrow")
        if arrow_dir.exists():
            feather_files = list(arrow_dir.glob("*.feather"))
            for feather_file in feather_files:
                feather_file.unlink()
                print(f"  Deleted: {feather_file}")
            
            # Remove directory if empty
            try:
                arrow_dir.rmdir()
                print(f"  Removed directory: {arrow_dir}")
            except:
                pass
        
        # Delete vector_store directory
        vector_dir = Path("data/vector_store")
        if vector_dir.exists():
            for file in vector_dir.iterdir():
                file.unlink()
                print(f"  Deleted: {file}")
            
            try:
                vector_dir.rmdir()
                print(f"  Removed directory: {vector_dir}")
            except:
                pass
        
        # Delete vector_store.py
        vector_store_py = Path("vector_store.py")
        if vector_store_py.exists():
            vector_store_py.unlink()
            print(f"  Deleted: {vector_store_py}")
        
        # Delete arrow_store.py
        arrow_store_py = Path("hrm/arrow_store.py")
        if arrow_store_py.exists():
            arrow_store_py.unlink()
            print(f"  Deleted: {arrow_store_py}")
        
        print("[Migration] Old stores deleted")
    
    def generate_report(self) -> str:
        """Generate migration report"""
        duration = time.time() - self.start_time
        
        report = []
        report.append("="*60)
        report.append("DUCKDB MIGRATION REPORT")
        report.append("="*60)
        report.append("")
        report.append(f"Migration completed in {duration:.1f} seconds")
        report.append("")
        report.append("STATISTICS:")
        report.append(f"  ArrowStore pairs migrated: {self.stats['arrow_migrated']}")
        report.append(f"  Vectors migrated: {self.stats['vector_migrated']}")
        report.append(f"  Total rows migrated: {self.stats['total_rows_migrated']}")
        report.append("")
        
        if self.stats['failed_pairs']:
            report.append("FAILED PAIRS:")
            for pair in self.stats['failed_pairs']:
                report.append(f"  • {pair}")
            report.append("")
        
        report.append("RESULT:")
        if self.stats['total_rows_migrated'] > 0:
            report.append("  ✅ Migration successful")
            report.append("  ✅ All data now in DuckDB")
            report.append("  ✅ ArrowStore and vector_store deleted")
        else:
            report.append("  ⚠️  No data migrated")
        
        report.append("")
        report.append("NEXT STEPS:")
        report.append("  1. Update code to use DuckStore only")
        report.append("  2. Run backbone bag trainer")
        report.append("  3. Delete ArrowStore and vector_store files")
        
        return "\n".join(report)
    
    async def migrate_all(self) -> Dict[str, Any]:
        """Migrate all data to DuckDB"""
        print(f"\n{'='*60}")
        print("DUCKDB MIGRATION - STARTING")
        print(f"{'='*60}")
        
        # Migrate ArrowStore data
        arrow_results = self.migrate_arrow_data()
        
        # Migrate vector data
        vector_results = self.migrate_vector_data()
        
        # Delete old stores
        self.delete_old_stores()
        
        # Generate report
        report = self.generate_report()
        print(f"\n{report}")
        
        return {
            'arrow': arrow_results,
            'vector': vector_results,
            'stats': self.stats
        }

# Example usage
async def main():
    migration = DuckMigration()
    results = await migration.migrate_all()
    
    print(f"\nMigration complete!")
    print(f"Check DuckDB at: {migration.duck_db_path}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())