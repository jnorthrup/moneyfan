#!/usr/bin/env python3
"""
Bag Trading Marathon - Long-Running Trading Session
====================================================

Continuous trading session with real-time progress monitoring:
- Live PnL tracking
- Position management
- Risk controls
- Progress dashboard
- No timeout (configurable interval)
- Real-time metrics and visualization

Usage:
    python bag_trading_marathon.py --hours 24 --capital 1000 --update-interval 5
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
from contextlib import contextmanager

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bag_trading_marathon.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import core components
try:
    from paper_trading import (
        PaperTradingEngine, PaperTradingConfig, 
        StochasticBag, CodecAgent, HRMMetaAllocator,
        Trade, PerformanceMetrics
    )
    HAS_PAPER_TRADING = True
except ImportError:
    HAS_PAPER_TRADING = False
    logger.warning("Paper trading components not available")


@dataclass
class MarathonConfig:
    """Marathon trading configuration"""
    trading_mode: str = "marathon"
    symbol_list: List[str] = field(default_factory=list)
    time_interval: str = "1H"
    initial_capital: float = 100.0
    risk_per_trade: float = 0.01
    hours: int = 24
    n_codecs: int = 24
    bag_size: int = 30
    use_mlx: bool = True
    output_dir: str = "marathon_results"
    save_equity_curve: bool = True
    update_interval: int = 5  # seconds between updates
    log_level: str = "INFO"
    max_positions: int = 5  # Maximum concurrent positions
    max_daily_drawdown: float = 0.05  # 5% daily drawdown limit
    stop_loss_hard: float = 0.02  # 2% hard stop loss
    enable_progress_dashboard: bool = True


@dataclass
class MarathonState:
    """Real-time marathon state"""
    start_time: datetime = field(default_factory=datetime.now)
    current_time: datetime = field(default_factory=datetime.now)
    equity: float = 0.0
    initial_equity: float = 0.0
    total_pnl: float = 0.0
    trade_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    current_positions: Dict[str, Dict] = field(default_factory=dict)
    recent_trades: deque = field(default_factory=lambda: deque(maxlen=20))
    equity_curve: deque = field(default_factory=lambda: deque(maxlen=1000))
    drawdown: float = 0.0
    max_drawdown: float = 0.0
    current_bag: List[str] = field(default_factory=list)
    regime_confidence: float = 0.0
    veto_count: int = 0
    total_signals: int = 0
    skipped_signals: int = 0
    is_running: bool = True
    last_update: datetime = field(default_factory=datetime.now)


class ProgressDashboard:
    """Real-time progress dashboard using curses"""
    
    def __init__(self, config: MarathonConfig):
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
        
        # Hide cursor
        curses.curs_set(0)
        
    def stop(self):
        """Stop the dashboard"""
        if self.stdscr:
            curses.nocbreak()
            self.stdscr.keypad(False)
            curses.echo()
            curses.endwin()
        self.running = False
        
    def update(self, state: MarathonState):
        """Update dashboard with current state"""
        if not self.running or not self.stdscr:
            return
            
        try:
            self.stdscr.clear()
            height, width = self.stdscr.getmaxyx()
            
            # Header
            header = f"BAG TRADING MARATHON - {state.current_time.strftime('%Y-%m-%d %H:%M:%S')}"
            self._print_center(header, 0, width)
            
            # Progress bar
            elapsed = (state.current_time - state.start_time).total_seconds()
            total_time = self.config.hours * 3600
            progress = min(elapsed / total_time, 1.0)
            bar_width = min(width - 10, 50)
            filled = int(bar_width * progress)
            bar = "█" * filled + "░" * (bar_width - filled)
            progress_line = f"Progress: {bar} {progress*100:.1f}%"
            self._print_center(progress_line, 2, width)
            
            # Main metrics
            metrics_y = 4
            metrics = [
                f"Equity: ${state.equity:,.2f} (${state.equity - state.initial_equity:+,.2f})",
                f"Total P&L: ${state.total_pnl:,.2f}",
                f"Win Rate: {state.winning_trades}/{state.trade_count} ({state.winning_trades/max(state.trade_count,1)*100:.1f}%)",
                f"Max Drawdown: {state.max_drawdown*100:.2f}%",
                f"Current Drawdown: {state.drawdown*100:.2f}%",
                f"Positions: {len(state.current_positions)}/{self.config.max_positions}",
                f"Bag Size: {len(state.current_bag)} symbols",
                f"Regime Confidence: {state.regime_confidence:.2f}",
                f"Vetoes: {state.veto_count}",
                f"Total Signals: {state.total_signals} (Skipped: {state.skipped_signals})"
            ]
            
            for i, metric in enumerate(metrics):
                if metrics_y + i < height - 3:
                    self.stdscr.addstr(metrics_y + i, 2, metric)
            
            # Recent trades
            trades_y = metrics_y + len(metrics) + 1
            if trades_y < height - 5:
                self.stdscr.addstr(trades_y, 2, "RECENT TRADES:")
                trade_lines = min(len(state.recent_trades), height - trades_y - 2)
                for i in range(trade_lines):
                    if i < len(state.recent_trades):
                        trade = list(state.recent_trades)[i]
                        trade_str = f"  {trade['timestamp'].strftime('%H:%M')} {trade['symbol']:>6} {trade['action']:>4} ${trade['price']:>7.2f} P&L: ${trade['pnl']:>7.2f}"
                        self.stdscr.addstr(trades_y + 1 + i, 2, trade_str)
            
            # Current positions
            pos_y = trades_y + trade_lines + 2 if trades_y + trade_lines + 2 < height - 3 else height - 5
            if pos_y < height - 3:
                self.stdscr.addstr(pos_y, 2, "CURRENT POSITIONS:")
                pos_count = min(len(state.current_positions), height - pos_y - 2)
                for i, (symbol, pos) in enumerate(list(state.current_positions.items())[:pos_count]):
                    entry = pos.get('entry_price', 0)
                    current = pos.get('current_price', 0)
                    size = pos.get('size', 0)
                    pnl = (current - entry) * size if entry > 0 else 0
                    pos_str = f"  {symbol:>6}: Size {size:>6.2f} Entry ${entry:>7.2f} Current ${current:>7.2f} P&L ${pnl:>7.2f}"
                    self.stdscr.addstr(pos_y + 1 + i, 2, pos_str)
            
            # Bag info
            bag_y = pos_y + pos_count + 2 if pos_y + pos_count + 2 < height - 3 else height - 3
            if bag_y < height - 2:
                self.stdscr.addstr(bag_y, 2, f"Current Bag: {', '.join(state.current_bag[:8])}{'...' if len(state.current_bag) > 8 else ''}")
            
            # Footer
            footer = f"Press 'q' to stop | Update interval: {self.config.update_interval}s | Capital: ${self.config.initial_capital}"
            self._print_center(footer, height - 1, width)
            
            self.stdscr.refresh()
            
        except curses.error:
            # Ignore curses errors
            pass
        except Exception as e:
            logger.error(f"Dashboard update error: {e}")
    
    def _print_center(self, text: str, row: int, width: int):
        """Print centered text"""
        if width <= 0:
            return
        text_len = len(text)
        if text_len >= width:
            # Truncate if too long
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


class MarathonTradingEngine:
    """Main marathon trading engine"""
    
    def __init__(self, config: MarathonConfig):
        self.config = config
        self.state = MarathonState()
        self.state.initial_equity = config.initial_capital
        self.state.equity = config.initial_capital
        
        # Initialize components
        self._initialize_components()
        
        # Dashboard
        self.dashboard = ProgressDashboard(config)
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        
        logger.info(f"Marathon trading engine initialized with ${config.initial_capital} capital")
    
    def _initialize_components(self):
        """Initialize trading components"""
        if HAS_PAPER_TRADING:
            # Initialize stochastic bag
            arrow_dir = Path("hrm/data/arrow")
            self.bag = StochasticBag(arrow_dir, bag_size=self.config.bag_size)
            
            # Initialize codec agents
            self.codecs = []
            for i in range(self.config.n_codecs):
                codec_config = {}
                codec = CodecAgent(i, codec_config)
                self.codecs.append(codec)
            
            # Initialize HRM meta-allocator
            self.hrm = HRMMetaAllocator(self.config.n_codecs, {})
            
            logger.info("Trading components initialized")
        else:
            logger.error("Paper trading components not available")
            raise ImportError("Paper trading components required")
    
    def run(self):
        """Run marathon trading session"""
        logger.info(f"Starting marathon trading session for {self.config.hours} hours")
        
        # Start dashboard
        if self.config.enable_progress_dashboard:
            self.dashboard.start()
        
        try:
            # Calculate end time
            end_time = self.state.start_time + timedelta(hours=self.config.hours)
            
            # Main trading loop
            while self.state.is_running and self.state.current_time < end_time:
                # Update state time
                self.state.current_time = datetime.now()
                
                # Check exit request from dashboard
                if self.dashboard.check_exit():
                    logger.info("User requested stop via dashboard")
                    self.state.is_running = False
                    break
                
                # Run one trading cycle
                self._run_trading_cycle()
                
                # Update dashboard
                if self.config.enable_progress_dashboard:
                    self.dashboard.update(self.state)
                
                # Sleep for update interval
                time.sleep(self.config.update_interval)
            
            # Finalize
            self._finalize_session()
            
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, stopping marathon")
            self.state.is_running = False
        finally:
            # Cleanup
            if self.config.enable_progress_dashboard:
                self.dashboard.stop()
    
    def _run_trading_cycle(self):
        """Run one trading cycle (hourly)"""
        try:
            # Step 1: Resample stochastic bag (hourly)
            if self.state.current_time.hour != self.state.last_update.hour:
                self.bag.resample()
                self.state.current_bag = self.bag.get_bag()
                logger.info(f"Resampled bag: {self.state.current_bag}")
            
            # Step 2: Get data for current bag
            bag_data = {}
            for symbol in self.state.current_bag:
                if symbol == "USD":
                    continue
                
                df = self.bag.get_data_for_pair(symbol,
                                               start_date=self.state.current_time - timedelta(hours=24),
                                               end_date=self.state.current_time)
                
                if df is not None and len(df) > 0:
                    bag_data[symbol] = df
            
            # Step 3: Process each symbol
            for symbol, df in bag_data.items():
                if len(df) < 20:  # Need minimum data
                    continue
                
                # Get latest data
                latest_row = df.iloc[-1]
                
                # Create market state
                market_state = {
                    'timestamp': self.state.current_time,
                    'symbol': symbol,
                    'price': latest_row.get('close', latest_row.get('close_price', 100.0)),
                    'volume': latest_row.get('volume', 0.0),
                    'high': latest_row.get('high', latest_row.get('high_price', 100.0)),
                    'low': latest_row.get('low', latest_row.get('low_price', 100.0)),
                    'open': latest_row.get('open', latest_row.get('open_price', 100.0)),
                }
                
                # Generate features
                features = self._extract_features(df)
                
                # Generate signals from each codec
                codec_signals = []
                for codec in self.codecs:
                    signal, confidence = codec.generate_signal(features)
                    codec_signals.append((signal, confidence))
                
                # HRM decision
                decision = self.hrm.decide(codec_signals, market_state)
                
                # Update state
                self.state.regime_confidence = decision.get('regime_confidence', 0.0)
                self.state.total_signals += 1
                
                if decision.get('vetoed', False):
                    self.state.veto_count += 1
                    continue
                
                # Check risk controls
                if not self._check_risk_controls(symbol, decision, market_state):
                    self.state.skipped_signals += 1
                    continue
                
                # Execute trade if needed
                if decision['action'] != 'hold':
                    self._execute_trade(symbol, decision, market_state)
                
                # Update equity for this symbol
                self._update_equity(symbol, market_state['price'])
            
            # Update current time
            self.state.last_update = self.state.current_time
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features from DataFrame"""
        # Simplified feature extraction
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
    
    def _check_risk_controls(self, symbol: str, decision: Dict, market_state: Dict) -> bool:
        """Check risk controls before trading"""
        # Check max positions
        if len(self.state.current_positions) >= self.config.max_positions:
            if decision['action'] == 'buy':
                return False
        
        # Check daily drawdown
        if self.state.drawdown <= -self.config.max_daily_drawdown:
            logger.warning(f"Daily drawdown limit reached: {self.state.drawdown*100:.1f}%")
            return False
        
        # Check hard stop loss
        if symbol in self.state.current_positions:
            pos = self.state.current_positions[symbol]
            entry_price = pos.get('entry_price', 0)
            current_price = market_state['price']
            
            if entry_price > 0:
                loss_pct = (current_price - entry_price) / entry_price
                if loss_pct <= -self.config.stop_loss_hard:
                    logger.warning(f"Hard stop loss triggered for {symbol}: {loss_pct*100:.1f}%")
                    # Force close position
                    self._close_position(symbol, market_state['price'])
                    return False
        
        # Check regime confidence
        if self.state.regime_confidence < 0.75:
            return False
        
        return True
    
    def _execute_trade(self, symbol: str, decision: Dict, market_state: Dict):
        """Execute a trade"""
        price = market_state['price']
        
        # Calculate position size
        if decision['action'] == 'buy':
            # Calculate position size based on risk
            risk_per_trade = self.config.risk_per_trade * self.state.equity
            position_size = risk_per_trade / price
            action = 'buy'
            
        elif decision['action'] == 'sell':
            # Close existing position
            if symbol in self.state.current_positions:
                position_size = abs(self.state.current_positions[symbol].get('size', 0.0))
            else:
                # Sell short if no position (for simplicity)
                position_size = self.config.risk_per_trade * self.state.equity / price
            action = 'sell'
        
        else:
            return
        
        if position_size <= 0:
            return
        
        # Calculate PnL
        pnl = 0.0
        if action == 'sell' and symbol in self.state.current_positions:
            pos = self.state.current_positions[symbol]
            entry_price = pos.get('entry_price', 0)
            size = pos.get('size', 0)
            pnl = (price - entry_price) * size
        
        # Update equity
        if action == 'buy':
            # Add to equity
            pass
        elif action == 'sell':
            # Update total PnL
            self.state.total_pnl += pnl
            
            # Update win/loss
            if pnl > 0:
                self.state.winning_trades += 1
            else:
                self.state.losing_trades += 1
        
        # Create trade record
        trade_record = {
            'timestamp': self.state.current_time,
            'symbol': symbol,
            'action': action,
            'size': position_size,
            'price': price,
            'pnl': pnl,
            'confidence': decision['confidence']
        }
        
        # Update state
        self.state.trade_count += 1
        self.state.recent_trades.append(trade_record)
        
        # Update positions
        if action == 'buy':
            self.state.current_positions[symbol] = {
                'entry_price': price,
                'current_price': price,
                'size': position_size,
                'timestamp': self.state.current_time
            }
        elif action == 'sell':
            if symbol in self.state.current_positions:
                del self.state.current_positions[symbol]
        
        logger.info(f"Trade: {action} {symbol} at ${price:.2f}, size: {position_size:.4f}, P&L: ${pnl:.2f}")
    
    def _close_position(self, symbol: str, current_price: float):
        """Force close a position"""
        if symbol not in self.state.current_positions:
            return
        
        pos = self.state.current_positions[symbol]
        entry_price = pos.get('entry_price', 0)
        size = pos.get('size', 0)
        
        if entry_price > 0 and size > 0:
            pnl = (current_price - entry_price) * size
            self.state.total_pnl += pnl
            self.state.trade_count += 1
            
            if pnl > 0:
                self.state.winning_trades += 1
            else:
                self.state.losing_trades += 1
            
            trade_record = {
                'timestamp': self.state.current_time,
                'symbol': symbol,
                'action': 'close',
                'size': size,
                'price': current_price,
                'pnl': pnl,
                'confidence': 0.0
            }
            self.state.recent_trades.append(trade_record)
            
            del self.state.current_positions[symbol]
            
            logger.info(f"Closed position {symbol}: P&L ${pnl:.2f}")
    
    def _update_equity(self, symbol: str, current_price: float):
        """Update equity based on current positions"""
        # Update current price for all positions
        for sym, pos in self.state.current_positions.items():
            if sym == symbol:
                pos['current_price'] = current_price
        
        # Calculate current equity
        equity = self.state.equity
        
        # Add unrealized PnL from positions
        unrealized_pnl = 0.0
        for sym, pos in self.state.current_positions.items():
            entry_price = pos.get('entry_price', 0)
            size = pos.get('size', 0)
            current = pos.get('current_price', 0)
            if entry_price > 0 and size > 0 and current > 0:
                unrealized_pnl += (current - entry_price) * size
        
        equity += unrealized_pnl
        
        # Update drawdown
        if equity > 0:
            self.state.equity = equity
            self.state.equity_curve.append({
                'timestamp': self.state.current_time,
                'equity': equity,
                'unrealized_pnl': unrealized_pnl
            })
            
            # Calculate drawdown
            if len(self.state.equity_curve) > 0:
                peak_equity = max(e['equity'] for e in self.state.equity_curve)
                if peak_equity > 0:
                    self.state.drawdown = (equity - peak_equity) / peak_equity
                    if self.state.drawdown < self.state.max_drawdown:
                        self.state.max_drawdown = self.state.drawdown
    
    def _finalize_session(self):
        """Finalize the trading session"""
        logger.info("Finalizing trading session...")
        
        # Calculate final metrics
        final_metrics = self._calculate_final_metrics()
        
        # Save results
        self._save_results(final_metrics)
        
        # Print summary
        self._print_summary(final_metrics)
    
    def _calculate_final_metrics(self) -> PerformanceMetrics:
        """Calculate final performance metrics"""
        metrics = PerformanceMetrics()
        
        if not self.state.equity_curve:
            return metrics
        
        # Extract equity values
        equity_values = [e['equity'] for e in self.state.equity_curve]
        
        if len(equity_values) < 2:
            return metrics
        
        # Total return
        initial = self.state.initial_equity
        final = equity_values[-1]
        metrics.total_return = (final - initial) / initial
        
        # Annualized return
        elapsed = (self.state.current_time - self.state.start_time).total_seconds()
        if elapsed > 0:
            metrics.annualized_return = ((1 + metrics.total_return) ** (365 * 24 * 3600 / elapsed)) - 1
        
        # Max drawdown
        metrics.max_drawdown = self.state.max_drawdown
        
        # Sharpe ratio (simplified)
        returns = np.diff(equity_values) / np.array(equity_values[:-1])
        if len(returns) > 1:
            metrics.sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24)
        
        # Calmar ratio
        if abs(metrics.max_drawdown) > 1e-8:
            metrics.calmar_ratio = metrics.annualized_return / abs(metrics.max_drawdown)
        
        # Win rate
        metrics.win_rate = self.state.winning_trades / max(self.state.trade_count, 1)
        
        # Total P&L
        metrics.total_pnl = self.state.total_pnl
        
        # Trade count
        metrics.trade_count = self.state.trade_count
        
        # Turnover
        if initial > 0:
            metrics.turnover = abs(metrics.total_pnl) / initial
        
        return metrics
    
    def _save_results(self, metrics: PerformanceMetrics):
        """Save results to files"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save trades
        trades_data = []
        for trade in self.state.recent_trades:
            trades_data.append({
                'timestamp': trade['timestamp'],
                'symbol': trade['symbol'],
                'action': trade['action'],
                'size': trade['size'],
                'price': trade['price'],
                'pnl': trade['pnl'],
                'confidence': trade['confidence']
            })
        
        if trades_data:
            trades_df = pd.DataFrame(trades_data)
            trades_path = output_dir / "trades.csv"
            trades_df.to_csv(trades_path, index=False)
            logger.info(f"Saved trades to {trades_path}")
        
        # Save equity curve
        equity_data = list(self.state.equity_curve)
        if equity_data:
            equity_df = pd.DataFrame(equity_data)
            equity_path = output_dir / "equity_curve.csv"
            equity_df.to_csv(equity_path, index=False)
            logger.info(f"Saved equity curve to {equity_path}")
        
        # Save metrics
        metrics_dict = {
            'total_return': metrics.total_return,
            'sharpe_ratio': metrics.sharpe_ratio,
            'max_drawdown': metrics.max_drawdown,
            'calmar_ratio': metrics.calmar_ratio,
            'win_rate': metrics.win_rate,
            'total_pnl': metrics.total_pnl,
            'trade_count': metrics.trade_count,
            'turnover': metrics.turnover,
            'annualized_return': metrics.annualized_return,
            'session_hours': self.config.hours,
            'initial_equity': self.state.initial_equity,
            'final_equity': self.state.equity,
            'winning_trades': self.state.winning_trades,
            'losing_trades': self.state.losing_trades,
            'veto_count': self.state.veto_count,
            'total_signals': self.state.total_signals,
            'skipped_signals': self.state.skipped_signals
        }
        
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")
    
    def _print_summary(self, metrics: PerformanceMetrics):
        """Print session summary"""
        print("\n" + "="*80)
        print("BAG TRADING MARATHON - SESSION SUMMARY")
        print("="*80)
        print(f"Session Duration: {self.config.hours} hours")
        print(f"Start Time: {self.state.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End Time: {self.state.current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Initial Capital: ${self.state.initial_equity:,.2f}")
        print(f"Final Equity: ${self.state.equity:,.2f}")
        print(f"Total Return: {metrics.total_return:.2%}")
        print(f"Annualized Return: {metrics.annualized_return:.2%}")
        print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
        print(f"Calmar Ratio: {metrics.calmar_ratio:.2f}")
        print(f"Win Rate: {metrics.win_rate:.2%} ({self.state.winning_trades}/{self.state.trade_count})")
        print(f"Total P&L: ${metrics.total_pnl:,.2f}")
        print(f"Trade Count: {metrics.trade_count}")
        print(f"Turnover: {metrics.turnover:.2%}")
        print(f"Veto Count: {self.state.veto_count}")
        print(f"Total Signals: {self.state.total_signals} (Skipped: {self.state.skipped_signals})")
        print("="*80)
        
        # Validation against GOALS.md
        print("\n" + "="*80)
        print("GOALS.md VALIDATION")
        print("="*80)
        
        sharpe_target = 1.8
        max_dd_target = -0.15
        
        print(f"Target Sharpe Ratio: ≥ {sharpe_target}")
        print(f"Actual Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        sharpe_status = "✅ PASS" if metrics.sharpe_ratio >= sharpe_target else "❌ FAIL"
        print(f"Status: {sharpe_status}")
        
        print(f"\nTarget Max Drawdown: ≥ {max_dd_target:.0%}")
        print(f"Actual Max Drawdown: {metrics.max_drawdown:.2%}")
        dd_status = "✅ PASS" if metrics.max_drawdown >= max_dd_target else "❌ FAIL"
        print(f"Status: {dd_status}")
        
        # Overall validation
        if metrics.sharpe_ratio >= sharpe_target and metrics.max_drawdown >= max_dd_target:
            print(f"\n✅ GOALS ACHIEVED!")
            print(f"   Sharpe ≥ {sharpe_target} and MaxDD ≥ {max_dd_target:.0%}")
        else:
            print(f"\n❌ GOALS NOT ACHIEVED")
            print(f"   Needs improvement in:")
            if metrics.sharpe_ratio < sharpe_target:
                print(f"   - Sharpe ratio (target: {sharpe_target}, actual: {metrics.sharpe_ratio:.2f})")
            if metrics.max_drawdown < max_dd_target:
                print(f"   - Max drawdown (target: {max_dd_target:.0%}, actual: {metrics.max_drawdown:.2%})")
        
        print("="*80)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Long bag trading marathon with real-time progress tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--hours',
        type=int,
        default=24,
        help='Number of hours to run the marathon (default: 24)'
    )
    
    parser.add_argument(
        '--capital',
        type=float,
        default=100.0,
        help='Initial capital in USD (default: 100)'
    )
    
    parser.add_argument(
        '--update-interval',
        type=int,
        default=5,
        help='Seconds between updates (default: 5)'
    )
    
    parser.add_argument(
        '--codecs',
        type=int,
        default=24,
        help='Number of codec agents (default: 24)'
    )
    
    parser.add_argument(
        '--bag-size',
        type=int,
        default=30,
        help='Size of stochastic bag (default: 30)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default="marathon_results",
        help='Output directory (default: marathon_results)'
    )
    
    parser.add_argument(
        '--no-dashboard',
        action='store_true',
        help='Disable progress dashboard'
    )
    
    parser.add_argument(
        '--no-mlx',
        action='store_true',
        help='Disable MLX inference'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    print("="*80)
    print("BAG TRADING MARATHON")
    print("Long-Running Trading Session with Live Progress Tracking")
    print("="*80)
    
    args = parse_arguments()
    
    print(f"\nConfiguration:")
    print(f"  Hours: {args.hours}")
    print(f"  Initial Capital: ${args.capital}")
    print(f"  Update Interval: {args.update_interval}s")
    print(f"  Codec Count: {args.codecs}")
    print(f"  Bag Size: {args.bag_size}")
    print(f"  Dashboard: {'Enabled' if not args.no_dashboard else 'Disabled'}")
    print(f"  MLX: {'Enabled' if not args.no_mlx else 'Disabled'}")
    print(f"  Output: {args.output}")
    
    # Create config
    config = MarathonConfig(
        initial_capital=args.capital,
        hours=args.hours,
        n_codecs=args.codecs,
        bag_size=args.bag_size,
        use_mlx=not args.no_mlx,
        output_dir=args.output,
        update_interval=args.update_interval,
        enable_progress_dashboard=not args.no_dashboard
    )
    
    # Run marathon
    engine = MarathonTradingEngine(config)
    
    try:
        engine.run()
        return 0
    except Exception as e:
        logger.error(f"Marathon failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())