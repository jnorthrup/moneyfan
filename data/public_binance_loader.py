"""
Public Binance Loader - Load public Binance klines data (no login required)
==========================================================================

Loads public Binance klines data using only public endpoints:
- /api/v3/klines for historical candle data
- No API keys required (public endpoints only)
- Chunked downloading with rate limiting
- Emulates faster Binance feed via synthetic augmentation

This is the "emulated faster Binance-style feed" referenced in GOALS.md
and the training path for live agents.
"""

import requests
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import hashlib
from dataclasses import dataclass, field

# Public Binance endpoints (no authentication required)
BINANCE_API_BASE = "https://api.binance.com"
BINANCE_PUBLIC_KLINES = "/api/v3/klines"

# Rate limiting (public endpoints)
REQUEST_DELAY = 0.1  # 100ms between requests
MAX_RETRIES = 3
TIMEOUT = 30

@dataclass
class PublicBinanceConfig:
    """Configuration for public Binance data loading"""
    # Trading pairs to load (basic trade pairs)
    symbols: List[str] = field(default_factory=lambda: [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
        "LINKUSDT", "UNIUSDT", "ATOMUSDT", "LTCUSDT", "BCHUSDT",
        "ETCUSDT", "FILUSDT", "APTUSDT", "OPUSDT", "ARBUSDT",
    ])
    
    # Timeframes
    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h"])
    
    # Date range
    start_date: datetime = field(default_factory=lambda: datetime(2020, 1, 1))
    end_date: datetime = field(default_factory=lambda: datetime.now())
    
    # Download settings
    chunk_size: int = 1000  # Number of candles per request
    max_candles: int = 500000  # Max candles per symbol
    skip_existing: bool = True
    
    # Data directory
    data_dir: str = "hrm/data/public_binance"
    
    # Synthetic augmentation settings (to emulate "faster" feed)
    enable_synthetic: bool = True
    synthetic_granularity: str = "1m"  # Target granularity for synthetic data
    bag_size: int = 100  # Size for stochastic bagging


class PublicBinanceLoader:
    """
    Load public Binance klines data with synthetic augmentation
    """
    
    def __init__(self, config: PublicBinanceConfig = None):
        self.config = config or PublicBinanceConfig()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'moneyfan-public-loader/1.0',
            'Accept': 'application/json',
        })
        
        # Ensure data directory exists
        Path(self.config.data_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"[PublicBinanceLoader] Initialized with {len(self.config.symbols)} symbols")
        print(f"[PublicBinanceLoader] Data directory: {self.config.data_dir}")
        print(f"[PublicBinanceLoader] Date range: {self.config.start_date.date()} to {self.config.end_date.date()}")
    
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict]:
        """
        Make HTTP request to Binance public API with rate limiting and retries
        """
        for attempt in range(MAX_RETRIES):
            try:
                time.sleep(REQUEST_DELAY)  # Rate limiting
                response = self.session.get(
                    f"{BINANCE_API_BASE}{endpoint}",
                    params=params,
                    timeout=TIMEOUT
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if "451" in str(e):
                    print(f"  Binance geo-blocked (451) - using synthetic data")
                    return None
                print(f"  Request failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt == MAX_RETRIES - 1:
                    print(f"  Failed after {MAX_RETRIES} attempts")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
        return None
    
    def _fetch_klines(self, symbol: str, interval: str, start_time: int, end_time: int) -> pd.DataFrame:
        """
        Fetch klines from Binance public API
        """
        print(f"    Fetching {symbol} {interval} from {start_time} to {end_time}")
        
        all_klines = []
        current_start = start_time
        
        while current_start < end_time:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_start,
                'endTime': end_time,
                'limit': self.config.chunk_size
            }
            
            klines_data = self._make_request(BINANCE_PUBLIC_KLINES, params)
            
            if not klines_data:
                print(f"    No data returned for {symbol} {interval}")
                break
            
            if not klines_data:
                break
            
            all_klines.extend(klines_data)
            
            # Update start time for next chunk
            if len(klines_data) < self.config.chunk_size:
                break  # Reached the end
            
            # Get last timestamp and add 1ms to get next chunk
            last_timestamp = klines_data[-1][0]
            current_start = last_timestamp + 1
            
            # Progress update
            if len(all_klines) % 1000 == 0:
                print(f"    Fetched {len(all_klines)} candles so far...")
            
            # Safety limit
            if len(all_klines) >= self.config.max_candles:
                print(f"    Reached max candles limit: {self.config.max_candles}")
                break
        
        if not all_klines:
            return pd.DataFrame()
        
        # Convert to DataFrame
        columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ]
        
        df = pd.DataFrame(all_klines, columns=columns)
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades',
                       'taker_buy_base', 'taker_buy_quote']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove unused columns
        df = df.drop(['close_time', 'ignore'], axis=1)
        
        return df
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the 48-column schema
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Basic OHLCV is already present
        # Calculate SMA and EMA indicators
        windows = [5, 15, 60]
        for window in windows:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        
        # RSI (14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD (12, 26, 9)
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands (20, 2)
        bb_window = 20
        bb_std = df['close'].rolling(window=bb_window).std()
        bb_mid = df['close'].rolling(window=bb_window).mean()
        df['bb_mid'] = bb_mid
        df['bb_upper'] = bb_mid + (2 * bb_std)
        df['bb_lower'] = bb_mid - (2 * bb_std)
        
        # ATR (14)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(window=14).mean()
        
        # ADX (14)
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr_smooth = tr.rolling(window=14).mean()
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / tr_smooth)
        minus_di = (-minus_dm).rolling(window=14).mean() / tr_smooth
        
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        df['adx_14'] = dx.rolling(window=14).mean()
        
        # Synthetic orderbook features (emulated)
        # These are derived from OHLCV to emulate faster Binance feed
        df['ob_imbalance'] = (df['taker_buy_base'] - df['volume'] * 0.5) / df['volume']
        df['bid_price'] = df['close'] * 0.9995  # Simulated 0.05% spread
        df['ask_price'] = df['close'] * 1.0005  # Simulated 0.05% spread
        df['bid_size'] = df['volume'] * np.random.uniform(0.8, 1.2, len(df))  # Simulated size
        df['ask_size'] = df['volume'] * np.random.uniform(0.8, 1.2, len(df))  # Simulated size
        df['depth_5_bid'] = df['bid_size'] * 5  # Simulated depth
        df['depth_5_ask'] = df['ask_size'] * 5  # Simulated depth
        df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2
        df['spread_pct'] = (df['ask_price'] - df['bid_price']) / df['mid_price']
        
        # VWAP
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Returns (multiple horizons)
        df['returns_1m'] = df['close'].pct_change(1)
        df['returns_5m'] = df['close'].pct_change(5)
        df['returns_15m'] = df['close'].pct_change(15)
        df['returns_1h'] = df['close'].pct_change(60)
        
        # Volatility
        df['vol_5m'] = df['close'].rolling(window=5).std()
        
        # Regime label (simplified - based on slope)
        df['regime_label'] = 1  # Default to flat
        sma_15 = df['sma_15']
        sma_15_diff = sma_15.diff()
        df.loc[sma_15_diff > 0.001, 'regime_label'] = 2  # Up
        df.loc[sma_15_diff < -0.001, 'regime_label'] = 0  # Down
        
        # Stochastic compass (simplified GBM drift)
        returns = df['close'].pct_change().dropna()
        mu = returns.mean() * 1440  # Annualized
        sigma = returns.std() * np.sqrt(1440)
        df['stochastic_compass'] = mu / (sigma + 1e-8)
        
        # Horizon tags
        df['horizon_tag'] = '05m'  # Default
        
        # Predictor confidences (placeholders)
        df['predictor_conf_5m'] = 0.5
        df['predictor_conf_15m'] = 0.5
        df['predictor_conf_1h'] = 0.5
        
        # HRM reward (placeholder)
        df['hrm_reward'] = 0.0
        
        # Veto flag (placeholder)
        df['veto_flag'] = False
        
        # Position size (placeholder)
        df['position_size_usd'] = 0.0
        
        # Equity curve (placeholder)
        df['equity_curve'] = 0.0
        
        return df
    
    def _synthetic_augmentation(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Apply synthetic augmentation to emulate faster feed
        Uses stochastic bagging similar to binance_stochastic_bag_trainer
        """
        if not self.config.enable_synthetic or df.empty:
            return df
        
        print(f"    Applying synthetic augmentation...")
        
        # Ensure we have required columns for bagging
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"    Missing required columns for synthetic augmentation: {missing_cols}")
            return df
        
        # Create synthetic high-frequency data through stochastic bagging
        # This emulates the "faster Binance feed" by creating synthetic candles
        synthetic_data = []
        
        for i in range(len(df) - 1):
            current_row = df.iloc[i]
            next_row = df.iloc[i + 1]
            
            # Extract OHLCV
            open_price = current_row['open']
            high_price = current_row['high']
            low_price = current_row['low']
            close_price = current_row['close']
            volume = current_row['volume']
            
            # Create synthetic intermediate candles (bagging)
            bag_size = self.config.bag_size
            
            for j in range(bag_size):
                # Stochastic variation within the candle range
                time_fraction = j / bag_size
                
                # Price variation (random within candle range)
                price_variation = np.random.uniform(0, 1)
                synthetic_close = low_price + price_variation * (high_price - low_price)
                
                # Volume variation
                volume_variation = np.random.uniform(0.8, 1.2)
                synthetic_volume = volume * volume_variation / bag_size
                
                # Create synthetic timestamp (interpolated)
                if i < len(df) - 1:
                    time_diff = (df.index[i + 1] - df.index[i]).total_seconds()
                    synthetic_time = df.index[i] + timedelta(seconds=time_diff * (j / bag_size))
                else:
                    synthetic_time = df.index[i]
                
                synthetic_row = {
                    'timestamp': synthetic_time,
                    'open': synthetic_close,  # Use close as open for synthetic
                    'high': max(synthetic_close, high_price * 0.999),
                    'low': min(synthetic_close, low_price * 1.001),
                    'close': synthetic_close,
                    'volume': synthetic_volume,
                    'quote_volume': synthetic_volume * synthetic_close,
                    'trades': int(current_row['trades'] / bag_size) if 'trades' in current_row else 1,
                    'taker_buy_base': synthetic_volume * 0.5,
                    'taker_buy_quote': synthetic_volume * synthetic_close * 0.5,
                }
                
                synthetic_data.append(synthetic_row)
        
        if not synthetic_data:
            return df
        
        # Convert synthetic data to DataFrame
        synthetic_df = pd.DataFrame(synthetic_data)
        synthetic_df.set_index('timestamp', inplace=True)
        
        # Calculate technical indicators for synthetic data
        synthetic_df = self._calculate_technical_indicators(synthetic_df)
        
        # Combine with original data
        combined_df = pd.concat([df, synthetic_df]).sort_index()
        
        print(f"    Added {len(synthetic_data)} synthetic candles")
        
        return combined_df
    
    def load_symbol(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Load data for a single symbol and timeframe
        """
        print(f"\nLoading {symbol} {timeframe}...")
        
        # Check if data already exists
        filename = f"{symbol}_{timeframe}.feather"
        filepath = Path(self.config.data_dir) / filename
        
        if self.config.skip_existing and filepath.exists():
            print(f"    File exists: {filename}")
            try:
                df = pd.read_feather(filepath)
                print(f"    Loaded existing data: {len(df)} rows")
                return df
            except Exception as e:
                print(f"    Failed to load existing file: {e}")
        
        # Calculate timestamps
        start_ts = int(self.config.start_date.timestamp() * 1000)
        end_ts = int(self.config.end_date.timestamp() * 1000)
        
        # Fetch data from Binance
        df = self._fetch_klines(symbol, timeframe, start_ts, end_ts)
        
        if df.empty:
            print(f"    Binance unavailable - generating synthetic data")
            df = self._generate_synthetic_data(symbol, timeframe)
        
        if df.empty:
            print(f"    No data for {symbol} {timeframe}")
            return pd.DataFrame()
        
        print(f"    Loaded {len(df)} candles")
        
        # Calculate technical indicators
        df = self._calculate_technical_indicators(df)
        
        # Apply synthetic augmentation for 1m timeframe (emulating faster feed)
        if timeframe == '1m' and self.config.enable_synthetic:
            df = self._synthetic_augmentation(df, symbol)
        
        # Save to feather
        try:
            df.reset_index().to_feather(filepath)
            print(f"    Saved to {filename}: {len(df)} rows")
        except Exception as e:
            print(f"    Failed to save: {e}")
        
        return df
    
    def _generate_synthetic_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Generate realistic synthetic data for training when Binance is unavailable
        """
        np.random.seed(hash(symbol) % 2**31)
        
        # Determine frequency
        freq_map = {'1m': '1min', '5m': '5min', '15m': '15min', '1h': '1h'}
        freq = freq_map.get(timeframe, '5min')
        
        # Generate timestamps
        timestamps = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq=freq
        )
        
        n = len(timestamps)
        if n == 0:
            return pd.DataFrame()
        
        # Base price varies by symbol
        base_prices = {
            'BTCUSDT': 45000, 'ETHUSDT': 2500, 'SOLUSDT': 100,
            'BNBUSDT': 300, 'XRPUSDT': 0.5, 'ADAUSDT': 0.5,
            'DOGEUSDT': 0.08, 'AVAXUSDT': 35, 'DOTUSDT': 7,
            'MATICUSDT': 0.9, 'LINKUSDT': 15, 'UNIUSDT': 6,
        }
        base_price = base_prices.get(symbol, 100.0)
        
        # Generate GBM price path
        dt = 1/525600  # Annualized
        mu = 0.1  # 10% annual drift
        sigma = 0.8  # 80% annual volatility
        
        returns = np.random.randn(n) * sigma * np.sqrt(dt) + mu * dt
        log_returns = np.log(1 + returns)
        prices = base_price * np.exp(np.cumsum(log_returns))
        
        # Generate OHLCV
        high_mult = 1 + np.abs(np.random.randn(n)) * 0.01
        low_mult = 1 - np.abs(np.random.randn(n)) * 0.01
        
        df = pd.DataFrame({
            'open': prices * (1 + np.random.randn(n) * 0.002),
            'high': prices * high_mult,
            'low': prices * low_mult,
            'close': prices,
            'volume': np.random.exponential(1000, n) * (prices / base_price),
            'quote_volume': np.random.exponential(1000000, n),
            'trades': np.random.randint(100, 10000, n),
            'taker_buy_base': np.random.exponential(500, n),
            'taker_buy_quote': np.random.exponential(500000, n),
        }, index=timestamps)
        
        print(f"    Generated {n} synthetic candles for {symbol}")
        return df
    
    def load_all_symbols(self) -> Dict[str, pd.DataFrame]:
        """
        Load data for all configured symbols and timeframes
        """
        results = {}
        
        print(f"\n{'='*80}")
        print("LOADING PUBLIC BINANCE DATA")
        print(f"{'='*80}")
        
        for symbol in self.config.symbols:
            for timeframe in self.config.timeframes:
                df = self.load_symbol(symbol, timeframe)
                if not df.empty:
                    key = f"{symbol}_{timeframe}"
                    results[key] = df
        
        print(f"\n{'='*80}")
        print(f"LOAD COMPLETE: {len(results)} datasets loaded")
        print(f"{'='*80}")
        
        return results
    
    def generate_dataset_report(self, results: Dict[str, pd.DataFrame]) -> str:
        """Generate report of loaded datasets"""
        report = []
        report.append("="*80)
        report.append("PUBLIC BINANCE DATA REPORT")
        report.append("="*80)
        report.append("")
        
        for key, df in results.items():
            if not df.empty:
                report.append(f"{key}:")
                report.append(f"  Rows: {len(df)}")
                report.append(f"  Date range: {df.index.min()} to {df.index.max()}")
                report.append(f"  Columns: {len(df.columns)}")
                report.append("")
        
        return "\njoin(report)"


async def main():
    """Example usage of PublicBinanceLoader"""
    config = PublicBinanceConfig(
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        timeframes=["1m", "5m", "15m"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime.now(),
        enable_synthetic=True
    )
    
    loader = PublicBinanceLoader(config)
    results = loader.load_all_symbols()
    
    # Generate report
    report = loader.generate_dataset_report(results)
    print(f"\n{report}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())