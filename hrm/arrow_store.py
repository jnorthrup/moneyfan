import os
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
from datetime import datetime

class ArrowStore:
    """
    Idempotent Arrow (Feather) storage for candle data.
    Provides fast, zero-copy, memory-mapped access.
    """
    def __init__(self, base_dir: str = "hrm/data/arrow"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _get_path(self, symbol: str) -> str:
        # Sanitize symbol for filename (e.g. BTC-USD -> BTC_USD)
        slug = symbol.replace("-", "_").replace("/", "_")
        return os.path.join(self.base_dir, f"{slug}.feather")

    def upsert(self, symbol: str, df: pd.DataFrame):
        """
        Merge new data into the existing Arrow file.
        Ensures idempotency by deduplicating on index (timestamp).
        """
        if df.empty:
            return

        path = self._get_path(symbol)
        
        # Ensure consistent schema/types
        df = df.copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].astype('float64')
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].astype('int64')

        if os.path.exists(path):
            try:
                # Read existing data (memory mapped)
                existing_df = pd.read_feather(path)
                
                # Restore index if possible
                if 'time' in existing_df.columns:
                    existing_df['time'] = pd.to_datetime(existing_df['time'])
                    existing_df.set_index('time', inplace=True)
                elif 'timestamp' in existing_df.columns:
                    existing_df['time'] = pd.to_datetime(existing_df['timestamp'], unit='s')
                    existing_df.set_index('time', inplace=True)
                
                # Combine and deduplicate
                # If the input DF has a DatetimeIndex but the stored one doesn't (or vice versa)
                # handle that. We prefer storing with a simple RangeIndex and a 'timestamp' column
                # OR a DatetimeIndex. Let's stick to DatetimeIndex for Pandas compatibility.
                
                combined = pd.concat([existing_df, df])
                # Deduplicate by index (time) or 'timestamp' column
                if 'timestamp' in combined.columns:
                    combined = combined.drop_duplicates(subset=['timestamp'])
                else:
                    # Deduplicate by index
                    combined = combined[~combined.index.duplicated(keep='last')]
                
                combined = combined.sort_index()
                df = combined
            except Exception as e:
                print(f"Warning: Could not read existing arrow file {path}: {e}")
                # Fallback to just writing the new data if file is corrupt

        # Write to Feather (uncompressed for maximum mmap speed, or lz4 for disk efficiency)
        # The user wants "steady", so uncompressed + mmap is best for strategy speed.
        df.reset_index().to_feather(path, compression='uncompressed')

    def get_bounds(self, symbol: str):
        """
        Get the (min_time, max_time) of the stored data for a symbol.
        Returns (None, None) if no data exists.
        """
        path = self._get_path(symbol)
        if not os.path.exists(path):
            return None, None
        try:
            # Memory map to read metadata/columns without full load
            # Using read_table implies we might read more than needed if strict, 
            # but memory mapping makes it efficient to access just the column chunks.
            t = feather.read_table(path, memory_map=True)
            if t.num_rows == 0:
                return None, None
            
            # Identify time column
            time_col = None
            if 'time' in t.column_names:
                time_col = t['time']
            elif 'timestamp' in t.column_names:
                time_col = t['timestamp']
            elif 'index' in t.column_names:
                time_col = t['index']
            
            if time_col is None:
                return None, None
                
            min_val = time_col[0].as_py()
            max_val = time_col[t.num_rows - 1].as_py()
            
            # Normalize to datetime
            if isinstance(min_val, (int, float)):
                 min_val = datetime.fromtimestamp(min_val)
            if isinstance(max_val, (int, float)):
                 max_val = datetime.fromtimestamp(max_val)
                 
            return min_val, max_val
        except Exception as e:
            print(f"Error reading bounds for {symbol}: {e}")
            return None, None

    def load(self, symbol: str, start: datetime = None, end: datetime = None) -> pd.DataFrame:
        """
        Load data as a Pandas DataFrame using memory mapping.
        """
        path = self._get_path(symbol)
        if not os.path.exists(path):
            return pd.DataFrame()

        try:
            # use pyarrow directly for memory mapping to be safe
            table = feather.read_table(path, memory_map=True)
            df = table.to_pandas()
            
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
            elif 'timestamp' in df.columns:
                df['time'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('time', inplace=True)
            
            if start:
                df = df[df.index >= start]
            if end:
                df = df[df.index <= end]
            
            return df
        except Exception as e:
            print(f"Error loading {symbol} from arrow: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    # Smoke test
    store = ArrowStore("hrm/data/test_arrow")
    test_df = pd.DataFrame({
        'timestamp': [1000, 1001],
        'close': [100.0, 101.0]
    }, index=pd.to_datetime([1000, 1001], unit='s'))
    test_df.index.name = 'time'
    
    print("Saving test data...")
    store.upsert("BTC-USD", test_df)
    
    print("Loading test data...")
    loaded = store.load("BTC-USD")
    print(loaded)
