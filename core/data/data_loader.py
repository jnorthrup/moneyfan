"""
Data loading module - handles data ingestion and preprocessing.

Pure logic, no framework dependencies.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


@dataclass
class DataConfig:
    """Data loading configuration"""
    db_path: str = "core/data/coinbase.duckdb"
    symbol_list: List[str] = field(default_factory=lambda: ["BTC-USD", "ETH-USD", "SOL-USD"])
    time_range: Tuple[str, str] = ("2024-01-01", "2025-01-01")
    resample_frequency: str = "1H"  # 1 hour candles
    n_features: int = 15
    seq_len: int = 32
    cache_size: int = 1000


@dataclass
class CandleData:
    """Candle data structure"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str


@dataclass
class FeatureSet:
    """Feature set structure"""
    timestamp: datetime
    symbol: str
    features: np.ndarray
    metadata: Dict[str, float]


class DataLoader:
    """
    Load and preprocess data for training and inference.
    
    Responsibilities:
    - Data ingestion from various sources
    - Feature engineering
    - Data validation and cleaning
    - Cache management
    - Batch preparation
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.cache: Dict[str, List[FeatureSet]] = {}
        self.raw_data_cache: Dict[str, pd.DataFrame] = {}
        
    def load_candles(self, 
                    symbols: Optional[List[str]] = None,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load candle data.
        
        Args:
            symbols: List of symbols to load
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Candle DataFrame
        """
        if symbols is None:
            symbols = self.config.symbol_list
            
        if start_date is None:
            start_date = self.config.time_range[0]
            
        if end_date is None:
            end_date = self.config.time_range[1]
        
        # In a real implementation, this would load from database
        # For now, generate synthetic data
        return self._generate_synthetic_candles(symbols, start_date, end_date)
    
    def _generate_synthetic_candles(self, 
                                   symbols: List[str],
                                   start_date: str,
                                   end_date: str) -> pd.DataFrame:
        """Generate synthetic candle data for testing"""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Convert frequency to pandas format
        freq = self.config.resample_frequency
        if freq == "1H":
            freq = "h"
        elif freq == "4H":
            freq = "4h"
        elif freq == "1D":
            freq = "D"
        else:
            freq = "h"  # Default to hourly
        
        date_range = pd.date_range(start=start, end=end, freq=freq)
        
        data = []
        for symbol in symbols:
            base_price = 100.0 if "BTC" in symbol else 50.0
            for timestamp in date_range:
                # Generate synthetic price
                noise = np.random.normal(0, 0.02)  # 2% volatility
                price = base_price * (1 + noise)
                
                candle = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'open': price * (1 + np.random.uniform(-0.01, 0.01)),
                    'high': price * (1 + np.random.uniform(0, 0.02)),
                    'low': price * (1 + np.random.uniform(-0.02, 0)),
                    'close': price,
                    'volume': np.random.uniform(1000, 10000)
                }
                data.append(candle)
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def compute_features(self, candles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features from candle data.
        
        Args:
            candles_df: Candle DataFrame
            
        Returns:
            Features DataFrame
        """
        if len(candles_df) == 0:
            return pd.DataFrame()
        
        features_list = []
        
        # Group by symbol
        for symbol, group in candles_df.groupby('symbol'):
            group = group.sort_index()
            
            # Basic features
            features = group[['open', 'high', 'low', 'close', 'volume']].copy()
            
            # Technical indicators
            features['returns'] = features['close'].pct_change()
            features['log_returns'] = np.log(features['close'] / features['close'].shift(1))
            features['volatility'] = features['returns'].rolling(20).std()
            features['ma_20'] = features['close'].rolling(20).mean()
            features['ma_50'] = features['close'].rolling(50).mean()
            features['rsi'] = self._compute_rsi(features['close'])
            features['bollinger_upper'] = features['ma_20'] + 2 * features['volatility']
            features['bollinger_lower'] = features['ma_20'] - 2 * features['volatility']
            features['bollinger_width'] = (features['bollinger_upper'] - features['bollinger_lower']) / features['ma_20']
            
            # Time-based features
            features['hour'] = features.index.hour
            features['day_of_week'] = features.index.dayofweek
            features['month'] = features.index.month
            
            # Normalize features
            features_normalized = self._normalize_features(features)
            
            # Add symbol info
            features_normalized['symbol'] = symbol
            
            features_list.append(features_normalized)
        
        if not features_list:
            return pd.DataFrame()
        
        # Combine all symbols
        all_features = pd.concat(features_list, ignore_index=False)
        
        return all_features
    
    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Normalize features to [0, 1] range"""
        features_normalized = features.copy()
        
        # Select numeric columns
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col == 'symbol':
                continue
                
            col_data = features[col]
            # Skip if all NaN or all same
            if col_data.isna().all() or col_data.nunique() == 1:
                features_normalized[col] = 0.0
                continue
            
            # Remove outliers using IQR
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            clipped = col_data.clip(lower_bound, upper_bound)
            
            # Normalize to [0, 1]
            min_val = clipped.min()
            max_val = clipped.max()
            
            if max_val > min_val:
                normalized = (clipped - min_val) / (max_val - min_val)
            else:
                normalized = clipped * 0.0
                
            features_normalized[col] = normalized
        
        return features_normalized
    
    def prepare_training_batch(self,
                              features_df: pd.DataFrame,
                              batch_size: int = 32,
                              seq_len: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training batch.
        
        Args:
            features_df: Features DataFrame
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            (inputs, targets, masks)
        """
        if seq_len is None:
            seq_len = self.config.seq_len
        
        if len(features_df) < seq_len + batch_size:
            raise ValueError(f"Not enough data: {len(features_df)} < {seq_len + batch_size}")
        
        # Get unique timestamps
        timestamps = sorted(features_df.index.unique())
        
        # Select random start indices
        max_start = len(timestamps) - seq_len - 1
        starts = np.random.randint(0, max_start, size=batch_size)
        
        inputs_list = []
        targets_list = []
        masks_list = []
        
        for start in starts:
            ts_window = timestamps[start: start + seq_len]
            ts_next = timestamps[start + seq_len]
            
            # Get features for window
            window_features = features_df.loc[features_df.index.isin(ts_window)]
            
            # Get target (next timestamp)
            target_features = features_df.loc[features_df.index == ts_next]
            
            if len(window_features) == 0 or len(target_features) == 0:
                continue
            
            # Extract numeric features
            numeric_cols = window_features.select_dtypes(include=[np.number]).columns
            if 'symbol' in numeric_cols:
                numeric_cols = numeric_cols.drop('symbol')
            
            # Create input array
            input_array = window_features[numeric_cols].values
            
            # Pad if needed
            if input_array.shape[0] < seq_len:
                pad_size = seq_len - input_array.shape[0]
                input_array = np.pad(input_array, ((0, pad_size), (0, 0)), mode='constant')
            
            # Get target
            if 'symbol' in target_features.columns:
                target_numeric = target_features[numeric_cols].values[0]
            else:
                target_numeric = target_features[numeric_cols].iloc[0].values
            
            inputs_list.append(input_array)
            targets_list.append(target_numeric)
            
            # Create mask (1 for valid, 0 for padded)
            mask = np.ones(seq_len)
            if input_array.shape[0] < seq_len:
                mask[-(seq_len - input_array.shape[0]):] = 0
            masks_list.append(mask)
        
        if len(inputs_list) == 0:
            # Return empty arrays
            input_dim = features_df.select_dtypes(include=[np.number]).shape[1]
            return (np.zeros((batch_size, seq_len, input_dim)),
                    np.zeros((batch_size, input_dim)),
                    np.zeros((batch_size, seq_len)))
        
        # Stack arrays
        inputs = np.stack(inputs_list, axis=0)
        targets = np.stack(targets_list, axis=0)
        masks = np.stack(masks_list, axis=0)
        
        return inputs, targets, masks
    
    def prepare_inference_batch(self,
                               features_df: pd.DataFrame,
                               symbol: str = None,
                               recent: int = None) -> np.ndarray:
        """
        Prepare inference batch.
        
        Args:
            features_df: Features DataFrame
            symbol: Specific symbol
            recent: Number of recent timesteps
            
        Returns:
            Input array for inference
        """
        if symbol is not None:
            features_df = features_df[features_df['symbol'] == symbol]
        
        if recent is None:
            recent = self.config.seq_len
        
        if len(features_df) < recent:
            # Pad if needed
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            if 'symbol' in numeric_cols:
                numeric_cols = numeric_cols.drop('symbol')
            
            input_array = features_df[numeric_cols].values
            pad_size = recent - len(input_array)
            input_array = np.pad(input_array, ((0, pad_size), (0, 0)), mode='constant')
            return input_array
        
        # Take most recent
        features_df = features_df.sort_index().tail(recent)
        
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        if 'symbol' in numeric_cols:
            numeric_cols = numeric_cols.drop('symbol')
        
        return features_df[numeric_cols].values
    
    def get_batch_iterator(self,
                          features_df: pd.DataFrame,
                          batch_size: int = 32,
                          seq_len: int = None,
                          shuffle: bool = True):
        """Create batch iterator"""
        if seq_len is None:
            seq_len = self.config.seq_len
        
        timestamps = sorted(features_df.index.unique())
        
        if shuffle:
            np.random.shuffle(timestamps)
        
        # Create batches
        n_batches = (len(timestamps) - seq_len - 1) // batch_size
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            batch_timestamps = timestamps[start_idx:end_idx]
            
            # Process each timestamp in batch
            batch_inputs = []
            batch_targets = []
            
            for ts in batch_timestamps:
                ts_idx = timestamps.index(ts)
                window = timestamps[ts_idx:ts_idx + seq_len]
                next_ts = timestamps[ts_idx + seq_len]
                
                # Get features
                window_features = features_df.loc[features_df.index.isin(window)]
                target_features = features_df.loc[features_df.index == next_ts]
                
                if len(window_features) == 0 or len(target_features) == 0:
                    continue
                
                # Extract numeric columns
                numeric_cols = features_df.select_dtypes(include=[np.number]).columns
                if 'symbol' in numeric_cols:
                    numeric_cols = numeric_cols.drop('symbol')
                
                # Create input
                input_array = window_features[numeric_cols].values
                if input_array.shape[0] < seq_len:
                    pad_size = seq_len - input_array.shape[0]
                    input_array = np.pad(input_array, ((0, pad_size), (0, 0)), mode='constant')
                
                # Get target
                if 'symbol' in target_features.columns:
                    target_numeric = target_features[numeric_cols].values[0]
                else:
                    target_numeric = target_features[numeric_cols].iloc[0].values
                
                batch_inputs.append(input_array)
                batch_targets.append(target_numeric)
            
            if len(batch_inputs) == 0:
                continue
            
            # Stack
            inputs = np.stack(batch_inputs, axis=0)
            targets = np.stack(batch_targets, axis=0)
            
            yield inputs, targets


# Factory functions
def create_data_loader(config: DataConfig = None) -> DataLoader:
    """Factory function to create data loader"""
    if config is None:
        config = DataConfig()
    return DataLoader(config)