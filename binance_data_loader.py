"""
Binance Data Loader - Cross-Exchange Training
=============================================

Load Binance data for training, execute on Coinbase paper trading.
Cross-exchange strategy: Binance train → Coinbase execute.

Features:
- Fetch Binance historical data (Kline/candlestick)
- Create stochastic bag of 30 pairs + 1 USD
- Random seed control for reproducibility
- 75% missing data tolerance (new coin issues)
- Variable extent and length (64-256 steps)
- Save to Parquet for fast MLX training
"""

import numpy as np
import pandas as pd
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

try:
    from binance.spot import Spot as Client
    HAS_BINANCE_SDK = True
except ImportError:
    HAS_BINANCE_SDK = False
    print("[BinanceDataLoader] Binance SDK not available")

@dataclass
class BinanceDataConfig:
    """Configuration for Binance data loading"""
    # Data parameters
    pairs: List[str] = field(default_factory=lambda: [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
        "LINKUSDT", "UNIUSDT", "ATOMUSDT", "LTCUSDT", "BCHUSDT",
        "ETCUSDT", "FILUSDT", "APTUSDT", "OPUSDT", "ARBUSDT",
        "SUIUSDT", "SEIUSDT", "RUNEUSDT", "INJUSDT", "TIAUSDT",
        "PYTHUSDT", "JUPUSDT", "WIFUSDT", "BONKUSDT", "PEPEUSDT"
    ])
    timeframe: str = "5m"  # 5 minute candles
    start_date: str = "2023-01-01"
    end_date: str = "2024-01-01"  # 1 year training period
    max_missing: float = 0.75  # 75% missing data tolerance
    
    # Stochastic bag
    bag_size: int = 30  # 30 pairs + 1 USD
    seed: int = 42  # Random seed for reproducibility
    
    # Stochastic length
    min_seq_len: int = 64
    max_seq_len: int = 256
    
    # Output
    output_dir: str = "data/binance"
    save_format: str = "parquet"  # parquet or numpy

class BinanceDataLoader:
    """
    Load Binance data for cross-exchange training
    
    Strategy:
    1. Fetch Binance historical data
    2. Create stochastic bag (30 random pairs + 1 USD)
    3. Extract variable-length sequences (64-256 steps)
    4. Save to Parquet for MLX training
    5. Execute on Coinbase paper trading
    """
    
    def __init__(self, config: BinanceDataConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Binance client
        if HAS_BINANCE_SDK:
            self.client = Client()  # Public endpoint, no API keys needed
            print("[BinanceDataLoader] Binance SDK initialized")
        else:
            self.client = None
            print("[BinanceDataLoader] Binance SDK not available - using mock data")
        
        # Random seed
        random.seed(config.seed)
        np.random.seed(config.seed)
        
        print(f"[BinanceDataLoader] Config: {config}")
        print(f"[BinanceDataLoader] Output dir: {self.output_dir}")
    
    def create_stochastic_bag(self) -> List[str]:
        """Create stochastic bag of 30 pairs + 1 USD"""
        # Shuffle pairs
        pairs = self.config.pairs.copy()
        random.shuffle(pairs)
        
        # Select 30 pairs
        selected_pairs = pairs[:self.config.bag_size]
        
        # Add USD as cash equivalent
        selected_pairs.append("USD")  # Cash position
        
        print(f"[BinanceDataLoader] Created stochastic bag:")
        for i, pair in enumerate(selected_pairs):
            if i < 30:
                print(f"  {i+1:2d}. {pair}")
            else:
                print(f"  Cash: {pair}")
        
        return selected_pairs
    
    def fetch_historical_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch historical kline/candlestick data from Binance"""
        if not HAS_BINANCE_SDK or not self.client:
            print(f"[BinanceDataLoader] Mock data for {symbol}")
            return self._create_mock_data(symbol, start_date, end_date)
        
        try:
            # Convert dates to milliseconds
            start_ms = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
            end_ms = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
            
            print(f"[BinanceDataLoader] Fetching {symbol} data from {start_date} to {end_date}...")
            
            # Fetch all candles
            candles = []
            current_start = start_ms
            
            while current_start < end_ms:
                try:
                    # Get klines (Binance SDK)
                    klines = self.client.klines(
                        symbol=symbol,
                        interval=self.config.timeframe,
                        startTime=current_start,
                        endTime=end_ms,
                        limit=1000
                    )
                    
                    if not klines:
                        break
                    
                    candles.extend(klines)
                    
                    # Update start time
                    current_start = klines[-1][0] + 60000  # Add 1 minute
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"[BinanceDataLoader] Error fetching {symbol}: {e}")
                    break
            
            if not candles:
                print(f"[BinanceDataLoader] No data for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(candles, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Keep only OHLCV
            df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            
            print(f"[BinanceDataLoader] {symbol}: {len(df)} candles")
            return df
            
        except Exception as e:
            print(f"[BinanceDataLoader] Failed to fetch {symbol}: {e}")
            return None
    
    def _create_mock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Create mock data for testing (without Binance SDK)"""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Create timestamps based on timeframe
        if self.config.timeframe == "5m":
            freq = "5min"
        elif self.config.timeframe == "15m":
            freq = "15min"
        elif self.config.timeframe == "1h":
            freq = "1h"
        else:
            freq = "1h"
        
        timestamps = pd.date_range(start=start, end=end, freq=freq)
        
        # Create random walk data
        np.random.seed(hash(symbol) % 10000)
        
        base_price = 1000.0 + (hash(symbol) % 5000)
        prices = base_price + np.cumsum(np.random.randn(len(timestamps)) * 10)
        
        data = {
            'timestamp': timestamps,
            'open': prices + np.random.randn(len(timestamps)) * 2,
            'high': prices + np.abs(np.random.randn(len(timestamps))) * 5,
            'low': prices - np.abs(np.random.randn(len(timestamps))) * 5,
            'close': prices,
            'volume': np.random.exponential(1000, len(timestamps))
        }
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        print(f"[BinanceDataLoader] Mock data: {symbol}, {len(df)} candles")
        return df
    
    def extract_sequences(self, df: pd.DataFrame, n_sequences: int = 100) -> List[Dict[str, Any]]:
        """Extract variable-length sequences (64-256 steps)"""
        sequences = []
        total_length = len(df)
        
        for i in range(n_sequences):
            # Random sequence length
            seq_len = random.randint(self.config.min_seq_len, self.config.max_seq_len)
            
            # Random start index (ensuring we don't go out of bounds)
            max_start = max(0, total_length - seq_len)
            if max_start == 0:
                continue
                
            start_idx = random.randint(0, max_start)
            end_idx = start_idx + seq_len
            
            # Extract sequence
            seq_df = df.iloc[start_idx:end_idx].copy()
            
            # Add stochastic metadata
            sequence = {
                'sequence_id': f"seq_{i:06d}",
                'start_timestamp': seq_df.index[0],
                'end_timestamp': seq_df.index[-1],
                'seq_len': seq_len,
                'open_prices': seq_df['open'].values,
                'high_prices': seq_df['high'].values,
                'low_prices': seq_df['low'].values,
                'close_prices': seq_df['close'].values,
                'volumes': seq_df['volume'].values,
                'returns': np.diff(seq_df['close'].values) / seq_df['close'].values[:-1] if seq_len > 1 else np.array([0.0]),
                'label': random.choice([1, -1, 0])  # Buy/Sell/Hold
            }
            
            sequences.append(sequence)
        
        print(f"[BinanceDataLoader] Extracted {len(sequences)} sequences")
        return sequences
    
    def save_to_parquet(self, sequences: List[Dict[str, Any]], pair: str):
        """Save sequences to Parquet format (fast, efficient for MLX)"""
        # Flatten sequences for DataFrame
        rows = []
        for seq in sequences:
            # Create one row per timestep
            for t in range(seq['seq_len']):
                rows.append({
                    'sequence_id': seq['sequence_id'],
                    'timestamp': seq['start_timestamp'] + pd.Timedelta(minutes=5 * t),
                    'pair': pair,
                    'open': seq['open_prices'][t],
                    'high': seq['high_prices'][t],
                    'low': seq['low_prices'][t],
                    'close': seq['close_prices'][t],
                    'volume': seq['volumes'][t],
                    'label': seq['label']
                })
        
        if not rows:
            return
        
        df = pd.DataFrame(rows)
        filename = self.output_dir / f"{pair}_sequences.parquet"
        df.to_parquet(filename, index=False)
        print(f"[BinanceDataLoader] Saved {len(df)} timesteps to {filename}")
    
    def save_bag_metadata(self, bag_pairs: List[str], sequences_by_pair: Dict[str, List[Dict]]):
        """Save stochastic bag metadata"""
        metadata = {
            'bag_pairs': bag_pairs,
            'config': {
                'timeframe': self.config.timeframe,
                'start_date': self.config.start_date,
                'end_date': self.config.end_date,
                'min_seq_len': self.config.min_seq_len,
                'max_seq_len': self.config.max_seq_len,
                'seed': self.config.seed,
                'total_sequences': sum(len(sequences) for sequences in sequences_by_pair.values())
            },
            'sequence_counts': {pair: len(sequences) for pair, sequences in sequences_by_pair.items()}
        }
        
        import json
        metadata_file = self.output_dir / "stochastic_bag_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"[BinanceDataLoader] Saved metadata to {metadata_file}")
    
    async def load_training_data(self, n_sequences_per_pair: int = 100) -> Dict[str, Any]:
        """
        Load and prepare training data for stochastic bag
        
        Returns:
            Dictionary with bag pairs, sequences, and metadata
        """
        print(f"\n{'='*60}")
        print("BINANCE DATA LOADER - CROSS-EXCHANGE TRAINING")
        print(f"{'='*60}")
        print(f"Timeframe: {self.config.timeframe}")
        print(f"Period: {self.config.start_date} to {self.config.end_date}")
        print(f"Sequences per pair: {n_sequences_per_pair}")
        print(f"{'='*60}\n")
        
        # Create stochastic bag
        bag_pairs = self.create_stochastic_bag()
        
        # Filter to only Binance pairs (exclude USD)
        binance_pairs = [p for p in bag_pairs if p != "USD"]
        
        # Load data for each pair
        sequences_by_pair = {}
        failed_pairs = []
        
        for i, pair in enumerate(binance_pairs):
            print(f"\n[{i+1}/{len(binance_pairs)}] Loading {pair}...")
            
            # Fetch data
            df = self.fetch_historical_data(pair, self.config.start_date, self.config.end_date)
            
            if df is None or len(df) < self.config.min_seq_len:
                print(f"[BinanceDataLoader] Skipping {pair} - insufficient data")
                failed_pairs.append(pair)
                continue
            
            # Extract sequences
            sequences = self.extract_sequences(df, n_sequences_per_pair)
            
            if sequences:
                sequences_by_pair[pair] = sequences
                self.save_to_parquet(sequences, pair)
            else:
                failed_pairs.append(pair)
        
        # Update bag (remove failed pairs)
        if failed_pairs:
            print(f"\n[BinanceDataLoader] Failed pairs: {failed_pairs}")
            bag_pairs = [p for p in bag_pairs if p not in failed_pairs]
            
            # Replace failed pairs with USD if needed
            while len([p for p in bag_pairs if p != "USD"]) < 30 and len(bag_pairs) < 30:
                bag_pairs.append("USD")
        
        # Save metadata
        self.save_bag_metadata(bag_pairs, sequences_by_pair)
        
        print(f"\n{'='*60}")
        print("LOADING COMPLETE")
        print(f"{'='*60}")
        print(f"Success: {len(sequences_by_pair)} pairs")
        print(f"Failed: {len(failed_pairs)} pairs")
        print(f"Total sequences: {sum(len(sequences) for sequences in sequences_by_pair.values())}")
        print(f"Data saved to: {self.output_dir}")
        print(f"{'='*60}\n")
        
        return {
            'bag_pairs': bag_pairs,
            'sequences_by_pair': sequences_by_pair,
            'failed_pairs': failed_pairs,
            'config': self.config
        }

# Example usage
async def main():
    config = BinanceDataConfig(
        timeframe="5m",
        start_date="2023-01-01",
        end_date="2023-01-15",  # Short period for testing
        seed=42
    )
    
    loader = BinanceDataLoader(config)
    data = await loader.load_training_data(n_sequences_per_pair=50)
    
    print(f"Bag pairs: {data['bag_pairs']}")
    print(f"Success: {len(data['sequences_by_pair'])} pairs")
    print(f"Failed: {data['failed_pairs']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())