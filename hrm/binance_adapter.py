#!/usr/bin/env python3
"""
Binance‑Adapter – combines Coinbase’s symbol‑selection / bag logic with Binance
historical candle data.

- Symbol selection (including popularity‑based stratified bags) is delegated to
  the existing ``CoinbasePipeline`` class, which knows how to rank symbols,
  filter by holdings, etc.
- For each selected symbol we read the pre‑downloaded Binance CSV (produced
  by ``fetchklines.sh``) and slice it to the same time window that the Coinbase
  bag uses.
- The resulting list of pandas DataFrames matches the format expected by the
  ``ContinuousTrainer`` (i.e. each DataFrame contains a ``symbol`` column and a
  datetime index).
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict

# Re‑use the Coinbase pipeline for the *selection* logic (popularity, holdings, etc.)
try:
    from .coinbase_pipeline import CoinbasePipeline
except ImportError:
    from coinbase_pipeline import CoinbasePipeline

# Arrow store is used elsewhere in the project for persistence – we keep the
# same location so that any downstream code that expects data in ``hrm/data/arrow``
# continues to work.
try:
    from .arrow_store import ArrowStore
except ImportError:
    try:
        from arrow_store import ArrowStore
    except ImportError:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from hrm.arrow_store import ArrowStore

try:
    from .backtest_curves import BacktestCurve
except ImportError:
    try:
        from backtest_curves import BacktestCurve
    except ImportError:
        pass # Optional import or handled elsewhere

# ----------------------------------------------------------------------
# Helper: load a Binance CSV for a given symbol.
# ----------------------------------------------------------------------
import subprocess
import calendar
from datetime import datetime, timedelta

def _load_binance_dataframe(store: ArrowStore, symbol: str, start: datetime = None, end: datetime = None) -> pd.DataFrame:
    """Load data from ArrowStore for a given symbol."""
    return store.load(symbol, start, end)

class SmartBinanceFetcher:
    """
    Intelligent fetcher that checks local ArrowStore coverage and only downloads
    missing monthly segments from Binance Vision.
    """
    def __init__(self, arrow_store: ArrowStore, dl_dir: str = "/tmp/smart_dl"):
        self.store = arrow_store
        self.dl_dir = dl_dir
        os.makedirs(dl_dir, exist_ok=True)
        
    def _get_months(self, start: datetime, end: datetime) -> List[tuple]:
        """Return list of (year, month) tuples between start and end."""
        months = []
        current = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_limit = end
        now = datetime.utcnow()
        
        while current <= end_limit:
            if current >= now:
                break
            months.append((current.year, current.month))
            # Next month
            days = calendar.monthrange(current.year, current.month)[1]
            current += timedelta(days=days)
        return months

    def fetch_window(self, symbol: str, start: datetime, end: datetime):
        """Ensure data exists for the specific window. Delegates to batch."""
        self.fetch_batch([(symbol, start, end)])

    def fetch_batch(self, requests: List[tuple]):
        """
        Fetch multiple windows in a single batched aria2c call.
        requests: List of (symbol, start, end)
        """
        all_urls = []
        
        # 1. Expand requests into URLs
        for symbol, start, end in requests:
            # Map symbol
            if symbol.endswith("-USD"):
                base_sym = symbol.replace("-USD", "USDT")
            else:
                base_sym = symbol.replace("-", "")
            
            years_months = self._get_months(start, end)
            for y, m in years_months:
                 zip_name = f"{base_sym}-1m-{y}-{m:02d}.zip"
                 # Only add if we don't have the zip AND we haven't successfully ingested it (hard to check ingest without arrow query)
                 # For speed, check zip existence
                 if not os.path.exists(os.path.join(self.dl_dir, zip_name)):
                     url = f"https://data.binance.vision/data/spot/monthly/klines/{base_sym}/1m/{zip_name}"
                     all_urls.append(url)

        if not all_urls:
            return

        all_urls = list(set(all_urls))
        
        # 2. Write Batch File
        input_file = os.path.join(self.dl_dir, "batch_input.txt")
        with open(input_file, "w") as f:
            for url in all_urls:
                f.write(f"{url}\n")
        
        # 3. Aria2c One-Shot
        # -j64/x16 for maximum parallelism
        # User requested "-c" explicitly
        cmd = [
            "aria2c", "-i", input_file, "-d", self.dl_dir,
            "-j", "64", "-x", "16", "-s", "16", "-k", "1M",
            "-c", "--allow-overwrite=true", "--quiet=false",
            "--max-connection-per-server=16", "--min-split-size=1M"
        ]
        
        print(f"🚀 ONE BATCH: aria2c fetching {len(all_urls)} files...")
        subprocess.run(cmd, check=False)
        
        # 4. Ingest All (iterate requests again)
        count = 0
        for symbol, start, end in requests:
             if symbol.endswith("-USD"):
                base_sym = symbol.replace("-USD", "USDT")
             else:
                base_sym = symbol.replace("-", "")
                
             years_months = self._get_months(start, end)
             for y, m in years_months:
                 zip_name = f"{base_sym}-1m-{y}-{m:02d}.zip"
                 zip_path = os.path.join(self.dl_dir, zip_name)
                 
                 if os.path.exists(zip_path):
                     try:
                         self._ingest_zip(zip_path, symbol)
                         count += 1
                     except Exception as e:
                         print(f"Ingest error {zip_name}: {e}")
        
        if count > 0:
            print(f"📦 Batch Ingest: Processed {count} months.")

    def _ingest_zip(self, zip_path: str, symbol: str):
        """Unzip and ingest a single file into ArrowStore."""
        import subprocess
        
        # unzip
        subprocess.run(["unzip", "-o", "-q", zip_path, "-d", self.dl_dir], check=False)
        
        # The csv name inside is typically matching the zip name
        base_name = os.path.basename(zip_path).replace(".zip", ".csv")
        csv_path = os.path.join(self.dl_dir, base_name)
        
        if not os.path.exists(csv_path):
            return

        try:
            # Read CSV
            # Columns: Open time, Open, High, Low, Close, Volume, Close time, Quote asset volume, Number of trades, Taker buy base asset volume, Taker buy quote asset volume, Ignore
            df = pd.read_csv(csv_path, header=None, 
                             names=['open_time','open','high','low','close','volume','close_time','qav','num_trades','taker_base','taker_quote','ignore'])
            
            df = df[['open_time','open','high','low','close','volume']]
            df.rename(columns={'open_time': 'timestamp'}, inplace=True)
            
            # Type handling and cleanup
            first_ts = df['timestamp'].iloc[0]
            # Fix object types if any (header rows etc)
            if isinstance(first_ts, str) or df['timestamp'].dtype == object:
                 df = df[pd.to_numeric(df['timestamp'], errors='coerce').notnull()]
                 df['timestamp'] = df['timestamp'].astype(float)
                 first_ts = df['timestamp'].iloc[0]

            # Unit detection
            if first_ts > 1e14: # Microseconds
                df['timestamp'] = df['timestamp'] / 1000000.0
            elif first_ts > 1e11: # Milliseconds
                df['timestamp'] = df['timestamp'] / 1000.0
            
            df['timestamp'] = df['timestamp'].astype('int64')
            df.set_index(pd.to_datetime(df['timestamp'], unit='s'), inplace=True)
            df.index.name = 'time'
            
            # Upsert
            self.store.upsert(symbol, df)
            
        finally:
            # Cleanup CSV
             if os.path.exists(csv_path):
                 os.remove(csv_path)
    
    def ensure_data(self, symbol: str, start_year: int = 2020):
        """Legacy init."""
        self.fetch_window(symbol, datetime(start_year, 1, 1), datetime.utcnow())


# ... existing ArrowBacktraceEngine ...

# ------------------------------------------------------------------
# ArrowBacktraceEngine definition (retained for context/completeness if needed, 
# but usually we just want to replace SmartBinanceFetcher)
# ------------------------------------------------------------------
# We will skip replacing ArrowBacktraceEngine if it wasn't broken, 
# but we need to ensure the file is valid. 
# The Replace block below Targets SmartBinanceFetcher..End of file to be safe?
# No, let's just replace the broken chunk properly.


class ArrowBacktraceEngine:
    """
    Backtrace engine using ArrowStore instead of SQLite.
    """
    def __init__(self, arrow_store: ArrowStore, fetcher=None):
        self.store = arrow_store
        self.fetcher = fetcher
        self.db_path = arrow_store.base_dir
        
    def load_segment(self, symbol: str, start: datetime, end: datetime, granularity: int = 60) -> pd.DataFrame:
        df = self.store.load(symbol, start, end)
        if df.empty and self.fetcher:
             try:
                 self.fetcher.fetch_window(symbol, start, end)
                 df = self.store.load(symbol, start, end)
             except Exception: pass
        if df.empty: return df
        return self._compute_features(df)
    
    def sample_synchronized_batch(self, symbols: List[str], n_samples: int = 1000, lookback: int = 50) -> List[tuple]:
        # Minimal impl for backtrace compatibility
        return []

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Standard feature set
        df = df.copy()
        df['returns'] = df['close'].pct_change()
        return df.dropna()


class BinanceAdapterPipeline:
    """A thin wrapper that mimics ``CoinbasePipeline`` but pulls data from Binance."""
    def __init__(self):
        self.coinbase = CoinbasePipeline()
        self.arrow_store = ArrowStore("hrm/data/arrow")
        self.fetcher = SmartBinanceFetcher(self.arrow_store)

    def get_backtrace_engine(self):
        return ArrowBacktraceEngine(self.arrow_store, fetcher=self.fetcher)

    def get_active_instruments(self, top_n: int = 64) -> List[str]:
        """Get spot-only pairs for Binance training."""
        try:
            from hrm.binance_spot_trainer import BINANCE_SPOT_PAIRS, is_valid_spot_pair
        except ImportError:
            # Fallback for direct execution
            BINANCE_SPOT_PAIRS = [
                "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
                "SOLUSDT", "DOTUSDT", "MATICUSDT", "LTCUSDT", "AVAXUSDT", "LINKUSDT",
                "ATOMUSDT", "UNIUSDT", "ETCUSDT", "XLMUSDT", "ALGOUSDT", "VETUSDT",
                "FILUSDT", "ICPUSDT", "THETAUSDT", "AAVEUSDT", "NEARUSDT", "AXSUSDT",
                "FTMUSDT", "SANDUSDT", "MANAUSDT", "GALAUSDT", "ENJUSDT", "COMPUSDT"
            ]
            def is_valid_spot_pair(symbol: str) -> bool:
                if not symbol.endswith("USDT"): return False
                if "UP" in symbol or "DOWN" in symbol: return False
                if "BULL" in symbol or "BEAR" in symbol: return False
                base = symbol[:-4]
                if base in ["BUSD", "USDC", "DAI", "TUSD", "USDP"]: return False
                return True
        
        available = []
        arrow_dir = "hrm/data/arrow"
        if os.path.exists(arrow_dir):
            for f in os.listdir(arrow_dir):
                if f.endswith(".feather"):
                    sym = f.replace(".feather", "").replace("_", "")
                    if is_valid_spot_pair(sym):
                        available.append(sym)
        
        if not available:
            # Use currency graph to select top_n pairs by depth
            try:
                from hrm.currency_graph import build_coinbase_graph_depth
                graph = build_coinbase_graph_depth()
                top_by_depth = graph.get_top_30_pairs_by_depth()
                
                # Map currency symbols to Binance format
                available = []
                for currency in top_by_depth:
                    if currency != "USD":  # Skip USD as base
                        # Add USDT pairs
                        available.append(currency + "USDT")
                        if len(available) >= top_n:
                            break
                
                # If we don't have enough, use fallback
                if len(available) < top_n:
                    available.extend(BINANCE_SPOT_PAIRS[:top_n - len(available)])
                    
            except Exception as e:
                print(f"Warning: Could not use currency graph: {e}")
                available = BINANCE_SPOT_PAIRS[:top_n]
        
        return available[:top_n]

    def update_holdings(self, holdings: Dict[str, float]):
        self.coinbase.update_holdings(holdings)

    def initialize(self, *_, **__) -> bool:
        print("[BinanceAdapter] initialize() – ArrowStore backed.")
        self.ensure_data_for_active_instruments()
        return True
        
    def ensure_data_for_active_instruments(self):
        pass # Lazy now

    # ------------------------------------------------------------------
    # Training bag – random stochastic bag using Arrow data.
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Training bag – forced wide synchronized bag
    # ------------------------------------------------------------------
    def sample_training_bag(self, n_samples: int = 5, min_len: int = 128) -> List[pd.DataFrame]:
        """
        Create ``n_samples`` "Megabags".
        
        Definition of a Bag (User Mandate):
          - 60 Days (2 months) of 1-minute data
          - Across ALL active instruments (Top 64)
          - Synchronized in time (same start date for all pairs in a bag)
          
        This method pre-calculates the requirements for ALL bags and executes
        a SINGLE massive aria2c batch download (typically 100+ files per call).
        """
        symbols = self.get_active_instruments() # 64 wide
        bags: List[pd.DataFrame] = []
        
        # Mandate: 2 months minimum depth
        stochastic_duration = timedelta(days=60)
        
        # Valid Universe: 2020-01-01 to Now
        global_start = datetime(2020, 1, 1)
        now = datetime.utcnow()
        max_start = now - stochastic_duration
        
        total_seconds = (max_start - global_start).total_seconds()
        
        # ---------------------------------------------------
        # 1. GENERATE SPECS (Synchronized Time Windows)
        # ---------------------------------------------------
        # storage for spec: (start_ts, end_ts) per bag index
        bag_time_windows = [] 
        
        print(f"🎲 Designing {n_samples} synchronized megabags (2-months x {len(symbols)} pairs)...")

        for i in range(n_samples):
            if total_seconds <= 0:
                start_ts = global_start
            else:
                rand_sec = np.random.randint(0, int(total_seconds))
                start_ts = global_start + timedelta(seconds=rand_sec)
            
            end_ts = start_ts + stochastic_duration
            bag_time_windows.append((start_ts, end_ts))
            print(f"  Bag {i+1}: {start_ts.date()} to {end_ts.date()}")

        # ---------------------------------------------------
        # 2. PRE-FLIGHT CHECK: Collect ALL Missing Data URLs
        # ---------------------------------------------------
        # We need data for (Symbol S) within (Window W) for all S in Symbols, for all W in Windows.
        # This is n_samples * n_symbols requests.
        
        all_requests = []
        
        for start_ts, end_ts in bag_time_windows:
            for sym in symbols:
                all_requests.append((sym, start_ts, end_ts))
                
        # EXECUTE MASSIVE BATCH DOWNLOAD
        if all_requests:
            print(f"🚀 PRE-FLIGHT: Resolving data for {len(all_requests)} segments (64-wide x 2-months)...")
            # We trust fetch_batch to dedupe URLs (multiple bags might overlap months)
            self.fetcher.fetch_batch(all_requests)

        # ---------------------------------------------------
        # 3. ASSEMBLY PASS: Load Data
        # ---------------------------------------------------
        print("💾 Assembling megabags from storage...")
        
        for i, (start_ts, end_ts) in enumerate(bag_time_windows):
            frames = []
            
            for sym in symbols:
                 try:
                     df = self.arrow_store.load(sym, start=start_ts, end=end_ts)
                     if not df.empty and len(df) >= min_len:
                         df = df.copy()
                         df["symbol"] = sym
                         frames.append(df)
                 except: pass
            
            if frames:
                 # A Megabag is just all these frames concatenated.
                 # Nexus will group by symbol.
                 bag = pd.concat(frames).reset_index()
                 # Tag the bag with metadata if useful?
                 # bag.attrs['window'] = (start_ts, end_ts)
                 bags.append(bag)
                 print(f"  ✅ Bag {i+1}/{n_samples} assembled ({len(frames)} pairs coverage)")
            else:
                 print(f"  ⚠️ Bag {i+1}/{n_samples} empty (no data found)")

        print(f"👜 Binance‑Adapter: Ready with {len(bags)} megabags.")
        return bags


    def sample_stratified_bag(self, n_samples: int = 1, lookback_days: int = 30) -> List[pd.DataFrame]:
        return []

    # Realtime placeholder
    def start_realtime(self, *_, **__):
        pass

    @property
    def holdings(self) -> Dict[str, float]:
        return self.coinbase.holdings

    def get_codec_trainer(self):
        """Get a codec trainer configured for Binance spot data."""
        from hrm.binance_spot_trainer import BinanceSpotCodecTrainer, BinanceSpotConfig
        config = BinanceSpotConfig(
            n_pairs=64,
            lookback_days=60,
            seq_len=32,
            target_window=5,
        )
        return BinanceSpotCodecTrainer(config)

    def train_and_export(self, n_epochs: int = 5, output_path: str = "hrm/data/binance_spot_weights.json"):
        """Train hierarchical signal model and export weights for Coinbase deployment."""
        trainer = self.get_codec_trainer()
        trainer.train_epoch(n_bags=n_epochs, verbose=True)
        trainer.save_checkpoint(output_path)
        return trainer.get_transferable_weights()

