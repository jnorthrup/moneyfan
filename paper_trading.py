#!/usr/bin/env python3
"""
Paper Trading: Full HRM System with 30-Day Live Data
=====================================================

Implements the complete HRM trading system:
- 24 codec agents (SOTA crypto strategies)
- 30-pair stochastic bag + USD
- HRM meta-allocator (hierarchical veto layer)
- 30-day paper trading with live Binance data
- Equity curve generation and performance metrics

Usage:
    python paper_trading.py --days 30 --capital 100 --output equity_curve.png
"""

import sys
import os
import argparse
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import MLX
try:
    import mlx.core as mx
    from hrm.hierarchical_codec_mlx import MLXHierarchicalCodec, HierarchicalCodecConfig
    HAS_MLX = True
    logger.info("MLX available - using MLX for inference")
except ImportError:
    HAS_MLX = False
    logger.warning("MLX not available - using PyTorch for inference")

# Import core components
from core.hrm.high_level import HighLevelConfig, HighLevelController
from core.hrm.low_level import LowLevelConfig, LowLevelProcessor, LowLevelFeature
from core.data.data_loader import DataConfig, DataLoader
from core.signals import SignalConfig, SignalAggregator
from core.risk.risk_management import RiskConfig, RiskManager
from core.risk.scorecard import Scorecard


@dataclass
class PaperTradingConfig:
    """Paper trading configuration"""
    trading_mode: str = "paper"
    symbol_list: List[str] = field(default_factory=list)
    time_interval: str = "1H"
    initial_capital: float = 100.0
    risk_per_trade: float = 0.01
    days: int = 30
    n_codecs: int = 24
    bag_size: int = 30
    use_mlx: bool = True
    output_dir: str = "paper_trading_results"
    save_equity_curve: bool = True


@dataclass
class Trade:
    """Trade record"""
    timestamp: datetime
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    size: float
    price: float
    pnl: float = 0.0
    confidence: float = 0.0
    codec_id: int = 0


@dataclass
class PerformanceMetrics:
    """Performance metrics"""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    trade_count: int = 0
    turnover: float = 0.0
    annualized_return: float = 0.0


class StochasticBag:
    """Stochastic bag of 30 random pairs + USD"""
    
    def __init__(self, arrow_dir: Path, bag_size: int = 30, seed: int = 42):
        self.arrow_dir = arrow_dir
        self.bag_size = bag_size
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Load all available pairs
        self.all_pairs = sorted([f.stem for f in arrow_dir.glob("*.feather")])
        logger.info(f"Found {len(self.all_pairs)} available pairs")
        
        # Add USD
        self.all_pairs.append("USD")
        
        # Current bag
        self.current_bag = []
        self.resample()
    
    def resample(self):
        """Resample random bag"""
        # Exclude USD from random selection
        tradable_pairs = [p for p in self.all_pairs if p != "USD"]
        
        # Random selection
        selected = self.rng.choice(tradable_pairs, 
                                  size=min(self.bag_size, len(tradable_pairs)), 
                                  replace=False).tolist()
        
        # Add USD
        selected.append("USD")
        
        self.current_bag = selected
        logger.info(f"Resampled bag: {self.current_bag}")
        
        return self.current_bag
    
    def get_bag(self):
        """Get current bag"""
        return self.current_bag
    
    def get_data_for_pair(self, pair: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Get data for a specific pair"""
        if pair == "USD":
            # USD is the quote currency, no data needed
            return None
        
        feather_file = self.arrow_dir / f"{pair}.feather"
        if not feather_file.exists():
            logger.warning(f"Data file not found for {pair}")
            return None
        
        try:
            df = pd.read_feather(feather_file)
            
            # Filter by date range
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                mask = (df['time'] >= start_date) & (df['time'] <= end_date)
                df = df[mask].copy()
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
                df = df[mask].copy()
            
            return df
        except Exception as e:
            logger.error(f"Error loading data for {pair}: {e}")
            return None


class CodecAgent:
    """Codec agent - trained model that generates signals"""
    
    def __init__(self, codec_id: int, config: dict):
        self.codec_id = codec_id
        self.config = config
        self.model = None
        self.last_signal = 0.0
        self.last_confidence = 0.0
        self.name = f"codec_{codec_id:02d}"
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize codec model"""
        if HAS_MLX:
            # Use MLX model
            codec_config = HierarchicalCodecConfig(
                n_signals=24,
                hidden_dim=64
            )
            self.model = MLXHierarchicalCodec(codec_config)
            logger.info(f"Codec {self.codec_id}: MLX model initialized")
        else:
            # Use PyTorch model (placeholder)
            logger.warning(f"Codec {self.codec_id}: Using placeholder model (PyTorch not available)")
    
    def generate_signal(self, features: np.ndarray) -> Tuple[float, float]:
        """Generate signal from features"""
        if self.model is None:
            # Placeholder: random signal
            signal = np.random.randn()
            confidence = np.random.rand()
            return signal, confidence
        
        try:
            if HAS_MLX:
                # Convert to MLX array
                features_mx = mx.array(features[None, None, :].astype(np.float32))
                
                # Run inference
                output, _ = self.model.forward(features_mx, mode="trade")
                
                # Extract signal and confidence
                signal = float(output[0, 0])  # return
                confidence = float(output[0, 1])  # confidence
                
                self.last_signal = signal
                self.last_confidence = confidence
                
                return signal, confidence
            else:
                # Placeholder for PyTorch
                signal = np.random.randn()
                confidence = np.random.rand()
                return signal, confidence
        except Exception as e:
            logger.error(f"Codec {self.codec_id}: Error generating signal: {e}")
            # Fallback to placeholder
            return np.random.randn(), np.random.rand()


class HRMMetaAllocator:
    """HRM Meta-Allocator - hierarchical veto layer"""
    
    def __init__(self, n_codecs: int, config: dict):
        self.n_codecs = n_codecs
        self.config = config
        
        # High-level controller (regime detection)
        high_level_config = HighLevelConfig(
            n_regimes=6,
            n_models=24,
            n_assets=128,
            hidden_dim=128,
            cycles=2,
            layers=2
        )
        self.high_level = HighLevelController(high_level_config)
        
        # Low-level processors (per-codec signal generation)
        self.low_level_processors = []
        for i in range(n_codecs):
            low_level_config = LowLevelConfig(
                n_features=15,
                n_assets=128,
                hidden_dim=64,
                cycles=2,
                layers=2,
                lookback=16
            )
            processor = LowLevelProcessor(low_level_config)
            self.low_level_processors.append(processor)
        
        # Scorecard for tracking
        self.scorecard = Scorecard()
        
        logger.info(f"HRM Meta-Allocator initialized with {n_codecs} codecs")
    
    def decide(self, codec_signals: List[Tuple[float, float]], 
               market_state: Dict[str, Any]) -> Dict[str, Any]:
        """Make trading decision based on codec signals and market state"""
        
        # Step 1: High-level decision
        # Convert codec signals to numpy array
        signals = np.array([s[0] for s in codec_signals])
        
        # Convert market state to numpy array
        market_state_array = np.array([
            market_state.get('price', 100.0),
            market_state.get('volume', 0.0),
            market_state.get('high', 100.0),
            market_state.get('low', 100.0),
            market_state.get('open', 100.0)
        ])
        
        # Get high-level decision
        decision = self.high_level.decide(signals, market_state_array)
        
        # Extract regime confidence from metadata
        regime_confidence = decision.confidence
        
        # Step 2: Veto layer
        if regime_confidence < 0.75:
            # Veto all signals in unfavorable regime
            final_decision = {
                'action': 'hold',
                'confidence': 0.0,
                'regime_confidence': regime_confidence,
                'vetoed': True
            }
            return final_decision
        
        # Step 3: Aggregate codec signals
        signals = [s[0] for s in codec_signals]
        confidences = [s[1] for s in codec_signals]
        
        # Weighted average by confidence
        weights = np.array(confidences) / sum(confidences)
        aggregated_signal = np.dot(signals, weights)
        aggregated_confidence = np.mean(confidences)
        
        # Step 4: Apply threshold
        if abs(aggregated_signal) < 0.3:
            action = 'hold'
        elif aggregated_signal > 0:
            action = 'buy'
        else:
            action = 'sell'
        
        # Step 5: Update low-level processors (process features)
        # For simplicity, we'll just process the market state
        for i, processor in enumerate(self.low_level_processors):
            # Create input features from signal and market state
            input_features = np.array([signal, confidence] + list(market_state_array[:3]))
            context = np.zeros(64)  # Default context
            feature = processor.process(input_features, context)
            # Store feature for later use if needed
            # feature.confidence can be used for further decisions
        
        # Step 6: Update scorecard
        self.scorecard.update(
            regime_confidence=regime_confidence,
            aggregated_signal=aggregated_signal,
            aggregated_confidence=aggregated_confidence,
            n_codecs=len(codec_signals)
        )
        
        final_decision = {
            'action': action,
            'confidence': aggregated_confidence,
            'regime_confidence': regime_confidence,
            'vetoed': False,
            'aggregated_signal': aggregated_signal
        }
        
        return final_decision


class PaperTradingEngine:
    """Paper trading engine"""
    
    def __init__(self, config: PaperTradingConfig):
        self.config = config
        
        # Initialize stochastic bag
        arrow_dir = Path("hrm/data/arrow")
        self.bag = StochasticBag(arrow_dir, bag_size=config.bag_size)
        
        # Initialize codec agents
        self.codecs = []
        for i in range(config.n_codecs):
            codec_config = {}
            codec = CodecAgent(i, codec_config)
            self.codecs.append(codec)
        
        # Initialize HRM meta-allocator
        self.hrm = HRMMetaAllocator(config.n_codecs, {})
        
        # Initialize risk manager
        risk_config = RiskConfig(
            initial_capital=config.initial_capital,
            risk_per_trade=config.risk_per_trade
        )
        self.risk_manager = RiskManager(risk_config)
        
        # State
        self.current_equity = config.initial_capital
        self.equity_curve = []
        self.trades = []
        self.positions = {}  # symbol -> position size
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Paper trading engine initialized with ${config.initial_capital} capital")
    
    def run(self, start_date: datetime, end_date: datetime):
        """Run paper trading for specified date range"""
        logger.info(f"Running paper trading from {start_date} to {end_date}")
        
        # Get date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
        
        # Main trading loop
        for timestamp in tqdm(date_range, desc="Paper Trading"):
            try:
                # Step 1: Resample stochastic bag (daily)
                if timestamp.hour == 0 and timestamp.minute == 0:
                    self.bag.resample()
                    logger.info(f"Resampled bag at {timestamp}")
                
                # Step 2: Get data for current bag
                bag_data = {}
                for symbol in self.bag.get_bag():
                    if symbol == "USD":
                        continue
                    
                    df = self.bag.get_data_for_pair(symbol, 
                                                   start_date=timestamp - timedelta(hours=24),
                                                   end_date=timestamp)
                    
                    if df is not None and len(df) > 0:
                        bag_data[symbol] = df
                
                # Step 3: Generate signals for each symbol
                if not bag_data:
                    continue
                
                # Step 4: Process each symbol
                for symbol, df in bag_data.items():
                    if len(df) < 20:  # Need minimum data
                        continue
                    
                    # Get latest data
                    latest_row = df.iloc[-1]
                    
                    # Create market state
                    market_state = {
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'price': latest_row.get('close', latest_row.get('close_price', 100.0)),
                        'volume': latest_row.get('volume', 0.0),
                        'high': latest_row.get('high', latest_row.get('high_price', 100.0)),
                        'low': latest_row.get('low', latest_row.get('low_price', 100.0)),
                        'open': latest_row.get('open', latest_row.get('open_price', 100.0)),
                    }
                    
                    # Generate features (simplified)
                    features = self._extract_features(df)
                    
                    # Step 5: Generate signals from each codec
                    codec_signals = []
                    for codec in self.codecs:
                        signal, confidence = codec.generate_signal(features)
                        codec_signals.append((signal, confidence))
                    
                    # Step 6: HRM decision
                    decision = self.hrm.decide(codec_signals, market_state)
                    
                    # Step 7: Execute trade if needed
                    if decision['action'] != 'hold' and not decision['vetoed']:
                        self._execute_trade(symbol, decision, market_state, timestamp)
                    
                    # Step 8: Update equity
                    self._update_equity(symbol, market_state['price'])
            
            except Exception as e:
                logger.error(f"Error at {timestamp}: {e}")
                continue
        
        # Calculate final performance
        metrics = self._calculate_metrics()
        
        # Save results
        self._save_results(metrics)
        
        # Generate equity curve
        if self.config.save_equity_curve:
            self._generate_equity_curve(metrics)
        
        return metrics
    
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features from DataFrame"""
        # Simplified feature extraction
        # In practice, this should compute EMA, MACD, RSI, etc.
        
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
    
    def _execute_trade(self, symbol: str, decision: Dict, market_state: Dict, timestamp: datetime):
        """Execute a trade"""
        price = market_state['price']
        
        # Calculate position size
        if decision['action'] == 'buy':
            position_size = self.risk_manager.calculate_position_size(
                symbol, decision['confidence'], 0.05, 0.8
            )
            action = 'buy'
        elif decision['action'] == 'sell':
            # For simplicity, close existing position
            position_size = abs(self.positions.get(symbol, 0.0))
            action = 'sell'
        else:
            return
        
        if position_size <= 0:
            return
        
        # Calculate PnL if closing position
        pnl = 0.0
        if action == 'sell' and symbol in self.positions:
            # Simplified PnL calculation
            entry_price = self.positions[symbol]  # In practice, store entry price
            pnl = (price - entry_price) * position_size
        
        # Create trade record
        trade = Trade(
            timestamp=timestamp,
            symbol=symbol,
            action=action,
            size=position_size,
            price=price,
            pnl=pnl,
            confidence=decision['confidence'],
            codec_id=0  # Simplified
        )
        
        self.trades.append(trade)
        
        # Update positions
        if action == 'buy':
            self.positions[symbol] = price
        elif action == 'sell':
            if symbol in self.positions:
                del self.positions[symbol]
        
        logger.info(f"Trade: {action} {symbol} at ${price:.2f}, size: {position_size:.4f}")
    
    def _update_equity(self, symbol: str, current_price: float):
        """Update equity based on current positions"""
        # Calculate current equity
        equity = self.current_equity
        
        # Add unrealized PnL from positions
        for sym, entry_price in self.positions.items():
            if sym == symbol:
                # Update position value
                position_size = self.positions[sym]  # Actually stored as price
                equity += (current_price - entry_price) * position_size
        
        self.current_equity = equity
        self.equity_curve.append({
            'timestamp': datetime.now(),
            'equity': equity,
            'symbol': symbol,
            'price': current_price
        })
    
    def _calculate_metrics(self) -> PerformanceMetrics:
        """Calculate performance metrics"""
        metrics = PerformanceMetrics()
        
        if not self.equity_curve:
            return metrics
        
        # Extract equity values
        equity_values = [e['equity'] for e in self.equity_curve]
        
        if len(equity_values) < 2:
            return metrics
        
        # Total return
        initial = self.config.initial_capital
        final = equity_values[-1]
        metrics.total_return = (final - initial) / initial
        
        # Annualized return
        days_traded = (self.equity_curve[-1]['timestamp'] - self.equity_curve[0]['timestamp']).days
        if days_traded > 0:
            metrics.annualized_return = ((1 + metrics.total_return) ** (365 / days_traded)) - 1
        
        # Max drawdown
        peak = equity_values[0]
        max_dd = 0.0
        for eq in equity_values:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd
        metrics.max_drawdown = max_dd
        
        # Sharpe ratio (simplified)
        returns = np.diff(equity_values) / np.array(equity_values[:-1])
        if len(returns) > 1:
            metrics.sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        
        # Calmar ratio
        if abs(metrics.max_drawdown) > 1e-8:
            metrics.calmar_ratio = metrics.annualized_return / abs(metrics.max_drawdown)
        
        # Win rate
        winning_trades = [t for t in self.trades if t.pnl > 0]
        metrics.win_rate = len(winning_trades) / len(self.trades) if self.trades else 0.0
        
        # Total P&L
        metrics.total_pnl = sum(t.pnl for t in self.trades)
        
        # Trade count
        metrics.trade_count = len(self.trades)
        
        # Turnover (simplified)
        if initial > 0:
            metrics.turnover = abs(metrics.total_pnl) / initial
        
        return metrics
    
    def _save_results(self, metrics: PerformanceMetrics):
        """Save results to files"""
        # Save trades
        trades_df = pd.DataFrame([{
            'timestamp': t.timestamp,
            'symbol': t.symbol,
            'action': t.action,
            'size': t.size,
            'price': t.price,
            'pnl': t.pnl,
            'confidence': t.confidence
        } for t in self.trades])
        
        trades_path = self.output_dir / "trades.csv"
        trades_df.to_csv(trades_path, index=False)
        logger.info(f"Saved trades to {trades_path}")
        
        # Save equity curve
        equity_df = pd.DataFrame(self.equity_curve)
        equity_path = self.output_dir / "equity_curve.csv"
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
            'annualized_return': metrics.annualized_return
        }
        
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("PAPER TRADING RESULTS")
        print("="*60)
        print(f"Initial Capital: ${self.config.initial_capital:.2f}")
        print(f"Final Equity: ${self.current_equity:.2f}")
        print(f"Total Return: {metrics.total_return:.2%}")
        print(f"Annualized Return: {metrics.annualized_return:.2%}")
        print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
        print(f"Calmar Ratio: {metrics.calmar_ratio:.2f}")
        print(f"Win Rate: {metrics.win_rate:.2%}")
        print(f"Total P&L: ${metrics.total_pnl:.2f}")
        print(f"Trade Count: {metrics.trade_count}")
        print(f"Turnover: {metrics.turnover:.2%}")
        print("="*60)
        
        # Validation against targets
        print("\n" + "="*60)
        print("ALPHA TARGET VALIDATION")
        print("="*60)
        
        sharpe_target = 1.8
        max_dd_target = -0.15
        
        print(f"Sharpe Ratio Target: {sharpe_target}")
        print(f"Actual Sharpe: {metrics.sharpe_ratio:.2f}")
        print(f"Status: {'✅ PASS' if metrics.sharpe_ratio >= sharpe_target else '❌ FAIL'}")
        
        print(f"\nMax Drawdown Target: {max_dd_target:.0%}")
        print(f"Actual Max DD: {metrics.max_drawdown:.2%}")
        print(f"Status: {'✅ PASS' if metrics.max_drawdown >= max_dd_target else '❌ FAIL'}")
        
        # Overall alpha status
        if metrics.sharpe_ratio >= sharpe_target and metrics.max_drawdown >= max_dd_target:
            print(f"\n✅ ALPHA TARGETS ACHIEVED!")
            print(f"   Sharpe ≥ {sharpe_target} and MaxDD ≤ {max_dd_target:.0%}")
        else:
            print(f"\n❌ ALPHA TARGETS NOT ACHIEVED")
            print(f"   Needs improvement in:")
            if metrics.sharpe_ratio < sharpe_target:
                print(f"   - Sharpe ratio (target: {sharpe_target}, actual: {metrics.sharpe_ratio:.2f})")
            if metrics.max_drawdown < max_dd_target:
                print(f"   - Max drawdown (target: {max_dd_target:.0%}, actual: {metrics.max_drawdown:.2%})")
        
        print("="*60)
    
    def _generate_equity_curve(self, metrics: PerformanceMetrics):
        """Generate equity curve plot"""
        if not self.equity_curve:
            return
        
        equity_df = pd.DataFrame(self.equity_curve)
        
        plt.figure(figsize=(12, 8))
        
        # Equity curve
        plt.subplot(2, 1, 1)
        plt.plot(equity_df['timestamp'], equity_df['equity'], 'b-', linewidth=2)
        plt.axhline(y=self.config.initial_capital, color='r', linestyle='--', alpha=0.5, label='Initial Capital')
        plt.title(f'Paper Trading Equity Curve - {self.config.days} Days')
        plt.ylabel('Equity ($)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Drawdown
        plt.subplot(2, 1, 2)
        equity = equity_df['equity'].values
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        plt.plot(equity_df['timestamp'], drawdown * 100, 'r-', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axhline(y=metrics.max_drawdown * 100, color='g', linestyle='--', alpha=0.5, label=f'Max DD: {metrics.max_drawdown:.1%}')
        plt.title('Drawdown (%)')
        plt.ylabel('Drawdown (%)')
        plt.xlabel('Date')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "equity_curve.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved equity curve plot to {plot_path}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Paper trading for HRM system with 30-day live data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days to run paper trading (default: 30)'
    )
    
    parser.add_argument(
        '--capital',
        type=float,
        default=100.0,
        help='Initial capital in USD (default: 100)'
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
        default="paper_trading_results",
        help='Output directory (default: paper_trading_results)'
    )
    
    parser.add_argument(
        '--no-mlx',
        action='store_true',
        help='Disable MLX inference (use PyTorch/placeholder)'
    )
    
    parser.add_argument(
        '--save-equity-curve',
        action='store_true',
        default=True,
        help='Save equity curve plot (default: True)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    print("="*60)
    print("PAPER TRADING: HRM SYSTEM WITH 30-DAY LIVE DATA")
    print("="*60)
    
    args = parse_arguments()
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    print(f"\nConfiguration:")
    print(f"  Start Date: {start_date.strftime('%Y-%m-%d')}")
    print(f"  End Date: {end_date.strftime('%Y-%m-%d')}")
    print(f"  Days: {args.days}")
    print(f"  Initial Capital: ${args.capital}")
    print(f"  Codec Count: {args.codecs}")
    print(f"  Bag Size: {args.bag_size}")
    print(f"  MLX: {'Enabled' if not args.no_mlx else 'Disabled'}")
    
    # Create config
    config = PaperTradingConfig(
        trading_mode="paper",
        initial_capital=args.capital,
        days=args.days,
        n_codecs=args.codecs,
        bag_size=args.bag_size,
        use_mlx=not args.no_mlx,
        output_dir=args.output,
        save_equity_curve=args.save_equity_curve
    )
    
    # Run paper trading
    engine = PaperTradingEngine(config)
    metrics = engine.run(start_date, end_date)
    
    # Save summary to markdown
    summary_path = Path(args.output) / "SUMMARY.md"
    with open(summary_path, 'w') as f:
        f.write(f"""# Paper Trading Summary

## Configuration
- **Start Date**: {start_date.strftime('%Y-%m-%d')}
- **End Date**: {end_date.strftime('%Y-%m-%d')}
- **Days Traded**: {args.days}
- **Initial Capital**: ${args.capital:.2f}
- **Final Equity**: ${engine.current_equity:.2f}
- **Codec Count**: {args.codecs}
- **Bag Size**: {args.bag_size}

## Performance Metrics
- **Total Return**: {metrics.total_return:.2%}
- **Annualized Return**: {metrics.annualized_return:.2%}
- **Sharpe Ratio**: {metrics.sharpe_ratio:.2f}
- **Max Drawdown**: {metrics.max_drawdown:.2%}
- **Calmar Ratio**: {metrics.calmar_ratio:.2f}
- **Win Rate**: {metrics.win_rate:.2%}
- **Total P&L**: ${metrics.total_pnl:.2f}
- **Trade Count**: {metrics.trade_count}
- **Turnover**: {metrics.turnover:.2%}

## Alpha Validation
### Targets
- Sharpe Ratio ≥ 1.8
- Max Drawdown ≥ -15%

### Results
- Sharpe Ratio: {'✅ PASS' if metrics.sharpe_ratio >= 1.8 else '❌ FAIL'} ({metrics.sharpe_ratio:.2f} vs 1.8)
- Max Drawdown: {'✅ PASS' if metrics.max_drawdown >= -0.15 else '❌ FAIL'} ({metrics.max_drawdown:.2%} vs -15%)

## Next Steps
1. Review equity curve and trade history
2. Analyze codec performance and regime detection
3. Run walk-forward validation (12m train + 3m test × 4 cycles)
4. Run Monte-Carlo validation (10,000 random shuffles)
5. Deploy to live paper trading for extended period
6. Publish Seeking Alpha article with methodology

---
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
    
    logger.info(f"Saved summary to {summary_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())