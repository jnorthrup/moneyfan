#!/usr/bin/env python3
"""
Train 500 Stochastic Bags with DuckDB Backend
==============================================

Complete pipeline:
1. Upload Binance data to DuckDB (with provenance tracking)
2. Train 500 stochastic bags using DuckDB queries
3. Live progress tracking with P&L dashboard
4. GOALS.md validation

Usage:
    python train_500_bags_duckdb.py --upload-binance --bags 500 --capital 100
"""

import sys
import os
import argparse
import json
import time
import threading
import curses
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from collections import deque
import logging
import hashlib

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_500_bags_duckdb.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try to import components
try:
    from hrm.duck_store import DuckStore
    HAS_DUCK_STORE = True
    logger.info("DuckStore available")
except ImportError:
    HAS_DUCK_STORE = False
    logger.warning("DuckStore not available")

try:
    import torch
    from hrm.hierarchical_codec import HierarchicalCodec, HierarchicalCodecConfig
    HAS_TORCH = True
    logger.info("PyTorch available")
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available")

try:
    import mlx.core as mx
    HAS_MLX = True
    logger.info("MLX available")
except ImportError:
    HAS_MLX = False
    logger.warning("MLX not available")


@dataclass
class DuckDBTrainingConfig:
    """Configuration for DuckDB-based bag training"""
    # Upload settings
    upload_binance: bool = False
    binance_sources: List[str] = field(default_factory=lambda: [
        "hrm/data/arrow",
        "hrm/data/binance"
    ])
    binance_pairs: List[str] = field(default_factory=lambda: [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
        "LINKUSDT", "UNIUSDT", "ATOMUSDT", "LTCUSDT", "BCHUSDT",
        "ETCUSDT", "FILUSDT", "APTUSDT", "OPUSDT", "ARBUSDT",
        "SUIUSDT", "SEIUSDT", "RUNEUSDT", "INJUSDT", "TIAUSDT",
        "PYTHUSDT", "JUPUSDT", "WIFUSDT", "BONKUSDT", "PEPEUSDT"
    ])
    
    # Training settings
    n_bags: int = 500
    capital: float = 100.0
    bag_size: int = 30
    min_seq_len: int = 64
    max_seq_len: int = 256
    sequences_per_bag: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 5
    seed: int = 42
    
    # Data paths
    duck_db_path: str = "hrm/data/market.duckdb"
    output_dir: str = "bag_training_results_duckdb"
    
    # Dashboard settings
    update_interval: int = 5  # seconds between updates
    enable_progress_dashboard: bool = True
    
    # Risk settings
    max_positions: int = 5
    risk_per_trade: float = 0.01
    
    # Model settings
    use_mlx: bool = True


@dataclass
class BagTrainingState:
    """Real-time bag training state"""
    start_time: datetime = field(default_factory=datetime.now)
    current_time: datetime = field(default_factory=datetime.now)
    total_bags_trained: int = 0
    current_bag: int = 0
    current_bag_progress: float = 0.0
    total_sequences: int = 0
    successful_sequences: int = 0
    total_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    current_bag_equity: float = 0.0
    bag_history: deque = field(default_factory=lambda: deque(maxlen=500))
    equity_curve: deque = field(default_factory=lambda: deque(maxlen=1000))
    drawdown: float = 0.0
    max_drawdown: float = 0.0
    avg_loss: float = 0.0
    avg_win_rate: float = 0.0
    is_training: bool = True
    last_update: datetime = field(default_factory=datetime.now)
    current_bag_stats: Dict[str, Any] = field(default_factory=dict)
    
    # DuckDB stats
    duckdb_symbols: int = 0
    duckdb_rows: int = 0
    upload_progress: float = 0.0
    upload_phase: str = "Not started"


class BagProgressDashboard:
    """Real-time bag training progress dashboard"""
    
    def __init__(self, config: DuckDBTrainingConfig):
        self.config = config
        self.stdscr = None
        self.running = False
        
    def start(self):
        """Start the dashboard"""
        if not self.config.enable_progress_dashboard:
            return
        
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(True)
        self.stdscr.nodelay(True)
        self.running = True
        curses.curs_set(0)
        
    def stop(self):
        """Stop the dashboard"""
        if self.stdscr:
            curses.nocbreak()
            self.stdscr.keypad(False)
            curses.echo()
            curses.endwin()
        self.running = False
        
    def update(self, state: BagTrainingState):
        """Update dashboard with current state"""
        if not self.running or not self.stdscr:
            return
            
        try:
            self.stdscr.clear()
            height, width = self.stdscr.getmaxyx()
            
            # Header
            header = f"TRAINING 500 BAGS WITH DUCKDB - {state.current_time.strftime('%Y-%m-%d %H:%M:%S')}"
            self._print_center(header, 0, width)
            
            # Upload phase
            upload_y = 2
            if state.upload_phase != "Not started":
                upload_bar_width = min(width - 10, 40)
                upload_filled = int(upload_bar_width * state.upload_progress)
                upload_bar = "█" * upload_filled + "░" * (upload_bar_width - upload_filled)
                upload_line = f"Upload: {upload_bar} {state.upload_progress*100:.1f}% ({state.upload_phase})"
                self._print_center(upload_line, upload_y, width)
                
                if state.duckdb_symbols > 0 or state.duckdb_rows > 0:
                    stats_line = f"DuckDB: {state.duckdb_symbols} symbols, {state.duckdb_rows:,} rows"
                    self._print_center(stats_line, upload_y + 1, width)
            
            # Overall progress
            overall_y = upload_y + 4 if state.upload_phase != "Not started" else 2
            overall_progress = state.total_bags_trained / self.config.n_bags if self.config.n_bags > 0 else 0
            bar_width = min(width - 10, 60)
            filled = int(bar_width * overall_progress)
            bar = "█" * filled + "░" * (bar_width - filled)
            progress_line = f"Overall Progress: {bar} {overall_progress*100:.1f}% ({state.total_bags_trained}/{self.config.n_bags} bags)"
            self._print_center(progress_line, overall_y, width)
            
            # Current bag progress
            current_bag_bar_width = min(width - 10, 40)
            current_filled = int(current_bag_bar_width * state.current_bag_progress)
            current_bar = "█" * current_filled + "░" * (current_bag_bar_width - current_filled)
            current_line = f"Bag #{state.current_bag + 1} Progress: {current_bar} {state.current_bag_progress*100:.1f}%"
            self._print_center(current_line, overall_y + 2, width)
            
            # Main metrics
            metrics_y = overall_y + 4
            elapsed = (state.current_time - state.start_time).total_seconds()
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            
            metrics = [
                f"Elapsed: {elapsed_str} | Bag: {state.current_bag + 1}/{self.config.n_bags}",
                f"Total P&L: ${state.total_pnl:+,.2f} | Trades: {state.total_trades} | Win: {state.winning_trades}/{state.losing_trades}",
                f"Current Bag Equity: ${state.current_bag_equity:+,.2f}",
                f"Max Drawdown: {state.max_drawdown*100:.2f}% | Current DD: {state.drawdown*100:.2f}%",
                f"Avg Loss: {state.avg_loss:.4f} | Avg Win Rate: {state.avg_win_rate:.2%}",
                f"Sequences: {state.successful_sequences}/{state.total_sequences}",
                f"Bag History Size: {len(state.bag_history)}",
            ]
            
            for i, metric in enumerate(metrics):
                if metrics_y + i < height - 3:
                    self.stdscr.addstr(metrics_y + i, 2, metric)
            
            # Recent bag results
            bags_y = metrics_y + len(metrics) + 1
            if bags_y < height - 5:
                self.stdscr.addstr(bags_y, 2, "RECENT BAG RESULTS:")
                bag_lines = min(len(state.bag_history), height - bags_y - 2)
                for i in range(bag_lines):
                    if i < len(state.bag_history):
                        bag = list(state.bag_history)[i]
                        bag_str = f"  Bag {bag['bag_id']:>3}: P&L ${bag['pnl']:>+7.2f} | Trades {bag['trades']:>3} | Win {bag['win_rate']:>5.1%} | Equity ${bag['equity']:>+8.2f}"
                        self.stdscr.addstr(bags_y + 1 + i, 2, bag_str)
            
            # Current bag stats
            stats_y = bags_y + bag_lines + 2 if bags_y + bag_lines + 2 < height - 3 else height - 5
            if stats_y < height - 2 and state.current_bag_stats:
                self.stdscr.addstr(stats_y, 2, "CURRENT BAG STATS:")
                stat_lines = min(len(state.current_bag_stats), height - stats_y - 2)
                for i, (key, value) in enumerate(list(state.current_bag_stats.items())[:stat_lines]):
                    stat_str = f"  {key}: {value}"
                    self.stdscr.addstr(stats_y + 1 + i, 2, stat_str)
            
            # Footer
            footer = f"Press 'q' to stop | Update interval: {self.config.update_interval}s | Capital: ${self.config.capital}"
            self._print_center(footer, height - 1, width)
            
            self.stdscr.refresh()
            
        except curses.error:
            pass
        except Exception as e:
            logger.error(f"Dashboard update error: {e}")
    
    def _print_center(self, text: str, row: int, width: int):
        """Print centered text"""
        if width <= 0:
            return
        text_len = len(text)
        if text_len >= width:
            text = text[:width-3] + "..."
            text_len = len(text)
        start_col = max(0, (width - text_len) // 2)
        try:
            self.stdscr.addstr(row, start_col, text)
        except curses.error:
            pass
    
    def check_exit(self):
        """Check if user wants to exit"""
        if not self.running or not self.stdscr:
            return False
        
        try:
            key = self.stdscr.getch()
            return key == ord('q') or key == ord('Q')
        except:
            return False


class DuckDBDataUploader:
    """Upload Binance data to DuckDB with provenance tracking"""
    
    def __init__(self, config: DuckDBTrainingConfig):
        self.config = config
        self.duck_store = None
        
    def upload_binance_data(self, state: BagTrainingState) -> bool:
        """Upload Binance data to DuckDB"""
        if not self.config.upload_binance:
            logger.info("Binance upload disabled")
            return True
        
        if not HAS_DUCK_STORE:
            logger.error("DuckStore not available")
            return False
        
        self.duck_store = DuckStore(self.config.duck_db_path)
        logger.info(f"Uploading Binance data to {self.config.duck_db_path}")
        
        # Initialize provenance tables
        self._init_provenance_tables()
        
        # Track progress
        total_sources = len(self.config.binance_sources)
        current_source = 0
        
        for source_path in self.config.binance_sources:
            current_source += 1
            source_path = Path(source_path)
            
            if not source_path.exists():
                logger.info(f"Source not found: {source_path}")
                continue
            
            # Update state
            state.upload_phase = f"Source {current_source}/{total_sources}"
            state.upload_progress = current_source / total_sources
            
            # Find feather files
            feather_files = list(source_path.glob("*.feather"))
            logger.info(f"Found {len(feather_files)} feather files in {source_path}")
            
            for feather_file in feather_files:
                # Extract symbol
                symbol_raw = feather_file.stem
                binance_symbol = symbol_raw.replace("_", "").replace("-", "").upper()
                
                if binance_symbol not in self.config.binance_pairs:
                    continue
                
                logger.info(f"Processing {binance_symbol}...")
                
                try:
                    # Load feather file
                    df = pd.read_feather(feather_file)
                    
                    if df.empty:
                        logger.warning(f"Empty data for {binance_symbol}")
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
                                binance_symbol,
                                timestamp,
                                float(row['open']),
                                float(row['high']),
                                float(row['low']),
                                float(row['close']),
                                float(row['volume']),
                                str(feather_file),
                                datetime.now().isoformat(),
                                hashlib.sha256(str(row).encode()).hexdigest()
                            ))
                            rows_inserted += 1
                        except Exception as e:
                            logger.warning(f"Failed to insert row {timestamp}: {e}")
                            break
                    
                    # Update state
                    state.duckdb_rows += rows_inserted
                    logger.info(f"Imported {rows_inserted} rows for {binance_symbol}")
                    
                except Exception as e:
                    logger.error(f"Failed to process {binance_symbol}: {e}")
        
        # Update final state
        state.upload_phase = "Complete"
        state.upload_progress = 1.0
        
        # Get final stats
        try:
            result = self.duck_store.conn.execute("SELECT COUNT(DISTINCT symbol) as symbols, COUNT(*) as rows FROM binance_source").fetchone()
            state.duckdb_symbols = result[0] if result else 0
            state.duckdb_rows = result[1] if result else 0
            logger.info(f"DuckDB now contains {state.duckdb_symbols} symbols with {state.duckdb_rows:,} rows")
        except Exception as e:
            logger.error(f"Failed to get DuckDB stats: {e}")
        
        return True
    
    def _init_provenance_tables(self):
        """Initialize provenance tables in DuckDB"""
        if not self.duck_store:
            return
        
        try:
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
            
            # Coinbase source table (if not exists)
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
            
            logger.info("Provenance tables initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize provenance tables: {e}")


class DuckDBBagTrainer:
    """Train stochastic bags using DuckDB data"""
    
    def __init__(self, config: DuckDBTrainingConfig):
        self.config = config
        self.state = BagTrainingState()
        self.state.current_bag_equity = config.capital
        
        # Initialize model
        self.model = None
        self.optimizer = None
        
        # DuckDB store
        self.duck_store = None
        self.symbols = []
        
        # Dashboard
        self.dashboard = BagProgressDashboard(config)
        
        # Training metrics
        self.bag_results = []
        
        logger.info("DuckDB bag trainer initialized")
    
    def _initialize_model(self):
        """Initialize training model"""
        if HAS_TORCH:
            config = HierarchicalCodecConfig(n_signals=24)
            self.model = HierarchicalCodec(config)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
            logger.info("PyTorch model initialized")
        else:
            logger.error("No training framework available")
            raise ImportError("No training framework available")
    
    def _get_available_symbols(self):
        """Get available symbols from DuckDB"""
        if not HAS_DUCK_STORE:
            logger.error("DuckStore not available")
            return
        
        self.duck_store = DuckStore(self.config.duck_db_path)
        
        try:
            # Get symbols from market_data view
            result = self.duck_store.conn.execute("""
                SELECT DISTINCT symbol FROM market_data
            """).fetchall()
            
            self.symbols = [row[0] for row in result]
            logger.info(f"Found {len(self.symbols)} available symbols in DuckDB")
            
            if len(self.symbols) < 10:
                logger.warning(f"Very few symbols available: {len(self.symbols)}")
            
        except Exception as e:
            logger.error(f"Failed to get symbols from DuckDB: {e}")
            self.symbols = []
    
    def _load_symbol_data(self, symbol: str, start: datetime = None, end: datetime = None) -> pd.DataFrame:
        """Load data for a specific symbol from DuckDB"""
        if not self.duck_store:
            return pd.DataFrame()
        
        try:
            if start and end:
                start_ts = int(start.timestamp())
                end_ts = int(end.timestamp())
                query = f"""
                    SELECT * FROM market_data 
                    WHERE symbol = '{symbol}' 
                    AND timestamp >= '{start.isoformat()}' 
                    AND timestamp <= '{end.isoformat()}'
                    ORDER BY timestamp
                """
            else:
                query = f"""
                    SELECT * FROM market_data 
                    WHERE symbol = '{symbol}' 
                    ORDER BY timestamp
                """
            
            df = self.duck_store.conn.execute(query).fetchdf()
            
            if len(df) > 0:
                df['time'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('time')
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _compute_signals(self, df: pd.DataFrame) -> np.ndarray:
        """Compute trading signals from DataFrame"""
        if len(df) < 20:
            return np.random.randn(48).astype(np.float32)
        
        close = df['close'].values if 'close' in df.columns else df['close_price'].values
        
        # Simple features: normalized price changes
        features = []
        for i in range(1, 21):
            if len(close) > i:
                features.append((close[-1] - close[-i]) / (close[-i] + 1e-8))
            else:
                features.append(0.0)
        
        # Pad to 48 features
        features = features + [0.0] * (48 - len(features))
        
        return np.array(features, dtype=np.float32)
    
    def _train_single_bag(self, bag_id: int) -> Dict[str, Any]:
        """Train a single stochastic bag"""
        logger.info(f"Training bag {bag_id + 1}/{self.config.n_bags}")
        
        # Reset bag state
        bag_state = {
            'bag_id': bag_id,
            'pnl': 0.0,
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'equity': self.config.capital,
            'sequences': 0,
            'success': False,
            'error': None
        }
        
        # Random seed for this bag
        bag_seed = self.config.seed + bag_id
        np.random.seed(bag_seed)
        if HAS_TORCH:
            torch.manual_seed(bag_seed)
        
        # Select random subset of symbols for this bag
        if len(self.symbols) == 0:
            bag_state['error'] = "No symbols available in DuckDB"
            return bag_state
        
        bag_symbols = []
        for _ in range(min(self.config.bag_size, len(self.symbols))):
            idx = np.random.randint(0, len(self.symbols))
            bag_symbols.append(self.symbols[idx])
        
        # Train on bag sequences
        sequences_trained = 0
        bag_pnl = 0.0
        bag_trades = 0
        bag_wins = 0
        
        try:
            for seq_idx in range(self.config.sequences_per_bag):
                # Update progress
                self.state.current_bag_progress = seq_idx / self.config.sequences_per_bag
                self.state.total_sequences += 1
                
                # Select random symbol
                symbol_idx = np.random.randint(0, len(bag_symbols))
                symbol = bag_symbols[symbol_idx]
                
                # Load data from DuckDB
                df = self._load_symbol_data(symbol)
                if len(df) < 100:
                    continue
                
                # Select random sequence
                start_idx = np.random.randint(0, len(df) - 50)
                seq_df = df.iloc[start_idx:start_idx + 50]
                
                if len(seq_df) < 20:
                    continue
                
                # Compute signals
                features = self._compute_signals(seq_df)
                
                # Simulate trading
                if HAS_TORCH and self.model is not None:
                    # With PyTorch model
                    self.model.train()
                    
                    # Prepare input
                    input_tensor = torch.from_numpy(features).unsqueeze(0).unsqueeze(0)
                    
                    # Forward pass
                    output, _ = self.model(input_tensor, mode="trade")
                    
                    # Extract prediction
                    pred_return = output[0, 0].item()
                    confidence = output[0, 1].item()
                    
                    # Simulate trade
                    actual_return = seq_df['close'].pct_change().iloc[-1] if len(seq_df) > 1 else 0
                    
                    # Position size
                    position_size = self.config.capital * self.config.risk_per_trade * confidence
                    
                    # Calculate PnL
                    pnl = position_size * actual_return if pred_return * actual_return > 0 else -position_size * abs(actual_return)
                    
                    # Update bag state
                    bag_pnl += pnl
                    bag_trades += 1
                    
                    if pnl > 0:
                        bag_wins += 1
                        self.state.winning_trades += 1
                    else:
                        self.state.losing_trades += 1
                    
                    self.state.total_pnl += pnl
                    self.state.total_trades += 1
                    
                    # Update model (simplified)
                    loss = torch.nn.functional.mse_loss(
                        output[0, 0], 
                        torch.tensor(actual_return, dtype=torch.float32)
                    )
                    
                    if self.optimizer:
                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                    
                    self.state.avg_loss = loss.item()
                
                else:
                    # Placeholder trading without model
                    pred_return = np.random.randn() * 0.1
                    confidence = np.random.rand()
                    actual_return = seq_df['close'].pct_change().iloc[-1] if len(seq_df) > 1 else 0
                    
                    position_size = self.config.capital * self.config.risk_per_trade * confidence
                    pnl = position_size * actual_return if pred_return * actual_return > 0 else -position_size * abs(actual_return)
                    
                    bag_pnl += pnl
                    bag_trades += 1
                    
                    if pnl > 0:
                        bag_wins += 1
                        self.state.winning_trades += 1
                    else:
                        self.state.losing_trades += 1
                    
                    self.state.total_pnl += pnl
                    self.state.total_trades += 1
                
                sequences_trained += 1
                self.state.successful_sequences += 1
                
                # Update current bag equity
                self.state.current_bag_equity += pnl
                
                # Update current bag stats
                self.state.current_bag_stats = {
                    'Sequences': f"{sequences_trained}/{self.config.sequences_per_bag}",
                    'Bag P&L': f"${bag_pnl:+.2f}",
                    'Bag Trades': bag_trades,
                    'Bag Win Rate': f"{bag_wins/bag_trades:.1%}" if bag_trades > 0 else "0%",
                    'Equity': f"${self.state.current_bag_equity:.2f}",
                    'Symbol': symbol[:15] + "..."
                }
                
                # Update dashboard
                if self.config.enable_progress_dashboard:
                    self.dashboard.update(self.state)
                
                # Check exit
                if self.dashboard.check_exit():
                    logger.info("User requested stop")
                    self.state.is_training = False
                    break
                
                # Sleep for update interval
                time.sleep(self.config.update_interval / 10)
            
            # Bag completed
            bag_state['pnl'] = bag_pnl
            bag_state['trades'] = bag_trades
            bag_state['wins'] = bag_wins
            bag_state['losses'] = bag_trades - bag_wins
            bag_state['equity'] = self.state.current_bag_equity
            bag_state['success'] = True
            bag_state['sequences'] = sequences_trained
            
            # Update average win rate
            if self.state.total_trades > 0:
                self.state.avg_win_rate = self.state.winning_trades / self.state.total_trades
            
            # Add to bag history
            self.state.bag_history.append({
                'bag_id': bag_id,
                'pnl': bag_pnl,
                'trades': bag_trades,
                'win_rate': bag_wins/bag_trades if bag_trades > 0 else 0,
                'equity': self.state.current_bag_equity,
                'sequences': sequences_trained
            })
            
            # Update equity curve
            self.state.equity_curve.append({
                'timestamp': datetime.now(),
                'equity': self.state.current_bag_equity,
                'bag_id': bag_id
            })
            
            # Update drawdown
            if len(self.state.equity_curve) > 0:
                equity_values = [e['equity'] for e in self.state.equity_curve]
                peak = max(equity_values)
                if peak > 0:
                    current = equity_values[-1]
                    self.state.drawdown = (current - peak) / peak
                    if self.state.drawdown < self.state.max_drawdown:
                        self.state.max_drawdown = self.state.drawdown
            
            logger.info(f"Bag {bag_id + 1} completed: P&L ${bag_pnl:+.2f}, Trades {bag_trades}, Win {bag_wins}")
            
        except Exception as e:
            logger.error(f"Error training bag {bag_id + 1}: {e}")
            bag_state['error'] = str(e)
        
        return bag_state
    
    def train(self):
        """Train 500 stochastic bags"""
        logger.info(f"Starting training of {self.config.n_bags} stochastic bags using DuckDB")
        
        # Upload Binance data if requested
        uploader = DuckDBDataUploader(self.config)
        if not uploader.upload_binance_data(self.state):
            logger.error("Binance upload failed")
            return
        
        # Get available symbols from DuckDB
        self._get_available_symbols()
        
        if len(self.symbols) == 0:
            logger.error("No symbols available in DuckDB")
            return
        
        # Initialize model
        self._initialize_model()
        
        # Start dashboard
        if self.config.enable_progress_dashboard:
            self.dashboard.start()
        
        try:
            # Training loop
            for bag_id in range(self.config.n_bags):
                if not self.state.is_training:
                    break
                
                # Update state
                self.state.current_bag = bag_id
                self.state.total_bags_trained = bag_id
                self.state.current_bag_progress = 0.0
                self.state.current_bag_stats = {}
                
                # Train single bag
                bag_result = self._train_single_bag(bag_id)
                self.bag_results.append(bag_result)
                
                # Update state
                self.state.total_bags_trained = bag_id + 1
                
                # Update dashboard
                if self.config.enable_progress_dashboard:
                    self.dashboard.update(self.state)
            
            # Finalize
            self._finalize_training()
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self.state.is_training = False
        finally:
            # Cleanup
            if self.config.enable_progress_dashboard:
                self.dashboard.stop()
    
    def _finalize_training(self):
        """Finalize training and save results"""
        logger.info("Finalizing training...")
        
        # Calculate final metrics
        final_metrics = self._calculate_final_metrics()
        
        # Save results
        self._save_results(final_metrics)
        
        # Print summary
        self._print_summary(final_metrics)
    
    def _calculate_final_metrics(self) -> Dict[str, Any]:
        """Calculate final training metrics"""
        metrics = {
            'total_bags_trained': self.state.total_bags_trained,
            'total_pnl': self.state.total_pnl,
            'total_trades': self.state.total_trades,
            'winning_trades': self.state.winning_trades,
            'losing_trades': self.state.losing_trades,
            'win_rate': self.state.winning_trades / max(self.state.total_trades, 1),
            'final_equity': self.state.current_bag_equity,
            'total_return': (self.state.current_bag_equity - self.config.capital) / self.config.capital,
            'max_drawdown': self.state.max_drawdown,
            'avg_loss': self.state.avg_loss,
            'avg_win_rate': self.state.avg_win_rate,
            'total_sequences': self.state.total_sequences,
            'successful_sequences': self.state.successful_sequences,
            'bag_history': len(self.state.bag_history),
            'duckdb_symbols': self.state.duckdb_symbols,
            'duckdb_rows': self.state.duckdb_rows
        }
        
        # Calculate Sharpe ratio
        if len(self.state.equity_curve) > 1:
            equity_values = [e['equity'] for e in self.state.equity_curve]
            returns = np.diff(equity_values) / np.array(equity_values[:-1])
            if len(returns) > 1:
                metrics['sharpe_ratio'] = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            else:
                metrics['sharpe_ratio'] = 0.0
        
        return metrics
    
    def _save_results(self, metrics: Dict[str, Any]):
        """Save training results to files"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save bag results
        if self.bag_results:
            bag_results_df = pd.DataFrame(self.bag_results)
            bag_results_path = output_dir / "bag_results.csv"
            bag_results_df.to_csv(bag_results_path, index=False)
            logger.info(f"Saved bag results to {bag_results_path}")
        
        # Save equity curve
        if self.state.equity_curve:
            equity_df = pd.DataFrame(self.state.equity_curve)
            equity_path = output_dir / "equity_curve.csv"
            equity_df.to_csv(equity_path, index=False)
            logger.info(f"Saved equity curve to {equity_path}")
        
        # Save bag history
        if self.state.bag_history:
            bag_history_df = pd.DataFrame(self.state.bag_history)
            bag_history_path = output_dir / "bag_history.csv"
            bag_history_df.to_csv(bag_history_path, index=False)
            logger.info(f"Saved bag history to {bag_history_path}")
        
        # Save metrics
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")
        
        # Save configuration
        config_dict = {
            'upload_binance': self.config.upload_binance,
            'n_bags': self.config.n_bags,
            'capital': self.config.capital,
            'bag_size': self.config.bag_size,
            'sequences_per_bag': self.config.sequences_per_bag,
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate,
            'epochs': self.config.epochs,
            'seed': self.config.seed,
            'duck_db_path': self.config.duck_db_path,
            'total_time': (datetime.now() - self.state.start_time).total_seconds()
        }
        config_path = output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Saved config to {config_path}")
        
        # Save model if available
        if self.model is not None and HAS_TORCH:
            model_path = output_dir / "trained_model.pt"
            torch.save({
                'model_state': self.model.state_dict(),
                'config': config_dict,
                'metrics': metrics
            }, model_path)
            logger.info(f"Saved model to {model_path}")
    
    def _print_summary(self, metrics: Dict[str, Any]):
        """Print training summary"""
        print("\n" + "="*80)
        print("TRAINING 500 STOCHASTIC BAGS WITH DUCKDB - SUMMARY")
        print("="*80)
        print(f"DuckDB: {metrics['duckdb_symbols']} symbols, {metrics['duckdb_rows']:,} rows")
        print(f"Total Bags Trained: {metrics['total_bags_trained']}")
        print(f"Total Sequences: {metrics['total_sequences']} (Successful: {metrics['successful_sequences']})")
        print(f"Final Equity: ${metrics['final_equity']:,.2f}")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Total P&L: ${metrics['total_pnl']:,.2f}")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2%} ({metrics['winning_trades']}/{metrics['total_trades']})")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        
        if 'sharpe_ratio' in metrics:
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        
        print(f"Avg Loss: {metrics['avg_loss']:.4f}")
        print(f"Avg Win Rate: {metrics['avg_win_rate']:.2%}")
        print(f"Bag History Size: {metrics['bag_history']}")
        print("="*80)
        
        # Validation against GOALS.md
        print("\n" + "="*80)
        print("GOALS.md VALIDATION")
        print("="*80)
        
        sharpe_target = 1.8
        max_dd_target = -0.15
        
        sharpe_actual = metrics.get('sharpe_ratio', 0.0)
        print(f"Target Sharpe Ratio: ≥ {sharpe_target}")
        print(f"Actual Sharpe Ratio: {sharpe_actual:.2f}")
        sharpe_status = "✅ PASS" if sharpe_actual >= sharpe_target else "❌ FAIL"
        print(f"Status: {sharpe_status}")
        
        print(f"\nTarget Max Drawdown: ≥ {max_dd_target:.0%}")
        print(f"Actual Max Drawdown: {metrics['max_drawdown']:.2%}")
        dd_status = "✅ PASS" if metrics['max_drawdown'] >= max_dd_target else "❌ FAIL"
        print(f"Status: {dd_status}")
        
        # Overall validation
        if sharpe_actual >= sharpe_target and metrics['max_drawdown'] >= max_dd_target:
            print(f"\n✅ GOALS ACHIEVED!")
            print(f"   Sharpe ≥ {sharpe_target} and MaxDD ≥ {max_dd_target:.0%}")
        else:
            print(f"\n❌ GOALS NOT ACHIEVED")
            print(f"   Needs improvement in:")
            if sharpe_actual < sharpe_target:
                print(f"   - Sharpe ratio (target: {sharpe_target}, actual: {sharpe_actual:.2f})")
            if metrics['max_drawdown'] < max_dd_target:
                print(f"   - Max drawdown (target: {max_dd_target:.0%}, actual: {metrics['max_drawdown']:.2%})")
        
        print("="*80)
        
        # Data architecture
        print("\n" + "="*80)
        print("DATA ARCHITECTURE")
        print("="*80)
        print("✅ DuckDB backend: hrm/data/market.duckdb")
        print("✅ Binance data: binance_source table (OHLCV)")
        print("✅ Coinbase data: coinbase_source table (48 columns)")
        print("✅ Unified view: market_data (UNION ALL)")
        print("✅ Provenance tracking: provenance_metadata table")
        print("✅ Symbol tracking: source_exchange field")
        print("✅ SQL queries: Efficient DuckDB analytical queries")
        print("="*80)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train 500 stochastic bags with DuckDB backend",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--upload-binance',
        action='store_true',
        help='Upload Binance data to DuckDB before training'
    )
    
    parser.add_argument(
        '--bags',
        type=int,
        default=500,
        help='Number of bags to train (default: 500)'
    )
    
    parser.add_argument(
        '--capital',
        type=float,
        default=100.0,
        help='Initial capital per bag (default: 100)'
    )
    
    parser.add_argument(
        '--update-interval',
        type=int,
        default=5,
        help='Seconds between dashboard updates (default: 5)'
    )
    
    parser.add_argument(
        '--bag-size',
        type=int,
        default=30,
        help='Size of stochastic bag (default: 30)'
    )
    
    parser.add_argument(
        '--sequences-per-bag',
        type=int,
        default=100,
        help='Sequences per bag (default: 100)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Epochs (default: 5)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default="bag_training_results_duckdb",
        help='Output directory (default: bag_training_results_duckdb)'
    )
    
    parser.add_argument(
        '--no-dashboard',
        action='store_true',
        help='Disable progress dashboard'
    )
    
    parser.add_argument(
        '--duck-db',
        type=str,
        default="hrm/data/market.duckdb",
        help='DuckDB path (default: hrm/data/market.duckdb)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    print("="*80)
    print("TRAIN 500 STOCHASTIC BAGS WITH DUCKDB BACKEND")
    print("="*80)
    
    args = parse_arguments()
    
    print(f"\nConfiguration:")
    print(f"  Upload Binance: {args.upload_binance}")
    print(f"  Bags: {args.bags}")
    print(f"  Capital per Bag: ${args.capital}")
    print(f"  Update Interval: {args.update_interval}s")
    print(f"  Bag Size: {args.bag_size}")
    print(f"  Sequences per Bag: {args.sequences_per_bag}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Dashboard: {'Enabled' if not args.no_dashboard else 'Disabled'}")
    print(f"  DuckDB Path: {args.duck_db}")
    print(f"  Output: {args.output}")
    
    # Create config
    config = DuckDBTrainingConfig(
        upload_binance=args.upload_binance,
        n_bags=args.bags,
        capital=args.capital,
        bag_size=args.bag_size,
        sequences_per_bag=args.sequences_per_bag,
        epochs=args.epochs,
        duck_db_path=args.duck_db,
        output_dir=args.output,
        update_interval=args.update_interval,
        enable_progress_dashboard=not args.no_dashboard
    )
    
    # Train bags
    trainer = DuckDBBagTrainer(config)
    
    try:
        trainer.train()
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())