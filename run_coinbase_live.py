#!/usr/bin/env python3
"""
Coinbase Live Trading System with 24 SOTA Codecs
=================================================

Single entry point for live trading on Coinbase with 24 SOTA codec models.

Architecture:
1. Python/MLX: Run 24 codecs + HRM hierarchy
2. SignalWriter: Write signals to stdout (JSON lines)
3. Execution: Receive signals and execute via exchange API

Usage:
    python run_coinbase_live.py --mode live --capital 500 --risk 0.75%

    # Or for paper trading:
    python run_coinbase_live.py --mode paper --days 30 --capital 500

    # Or for test-time adaptation:
    python run_coinbase_live.py --mode adapt --learning_rate 0.001
"""

import argparse
import json
import sys
import time
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import our modules
from codec_models.base_codec import BaseCodec, CodecFactory, get_all_codecs
from stochastic_bag import StochasticCompass, StochasticBagResampler
from exchange import SignalWriter
from paper_trading import HRMMetaAllocator, PaperTradingConfig, PaperTradingEngine


class CoinbaseLiveTradingSystem:
    """
    Main live trading system for Coinbase
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize live trading system
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.mode = config.get('mode', 'paper')
        self.capital = config.get('capital', 500.0)
        self.risk_per_trade = config.get('risk_per_trade', 0.0075)  # 0.75%
        
        # Initialize components
        self._initialize_codecs()
        self._initialize_hrm()
        self._initialize_compass()
        
        # State
        self.current_bag = []
        self.current_weights = None
        self.performance_history = []
        
        print(f"✅ Coinbase Live Trading System initialized")
        print(f"   Mode: {self.mode}")
        print(f"   Capital: ${self.capital:.2f}")
        print(f"   Risk per trade: {self.risk_per_trade:.3%}")
        print(f"   Codecs: {len(self.codecs)}")
    
    def _initialize_codecs(self):
        """Initialize 24 codecs"""
        print("\n🔄 Initializing 24 SOTA codecs...")
        
        # Get all codecs
        config = {
            'capital': self.capital,
            'risk_per_trade': self.risk_per_trade,
        }
        
        self.codecs = get_all_codecs(config)
        
        print(f"✅ Initialized {len(self.codecs)} codecs")
        for i, codec in enumerate(self.codecs):
            print(f"   {i+1:2d}. {codec.name}")
    
    def _initialize_hrm(self):
        """Initialize HRM meta-allocator"""
        print("\n🔄 Initializing HRM meta-allocator...")
        
        hrm_config = {
            'n_codecs': len(self.codecs),
            'capital': self.capital,
            'risk_per_trade': self.risk_per_trade,
        }
        
        self.hrm = HRMMetaAllocator(len(self.codecs), hrm_config)
        
        print("✅ HRM meta-allocator initialized")
    
    def _initialize_compass(self):
        """Initialize stochastic compass"""
        print("\n🔄 Initializing stochastic compass...")
        
        seed = self.config.get('seed', 42)
        self.compass = StochasticCompass(seed=seed)
        self.resampler = StochasticBagResampler(
            n_codecs=len(self.codecs),
            bag_size=self.config.get('bag_size', 30),
            seed=seed
        )
        
        print("✅ Stochastic compass initialized")
    

    
    def generate_market_data(self, symbol: str, timestamp: datetime) -> Dict[str, Any]:
        """
        Generate market data for a symbol
        
        In production, this would fetch real data from Coinbase API
        For now, we use simulated data
        
        Args:
            symbol: Trading pair
            timestamp: Current timestamp
            
        Returns:
            Market data dictionary
        """
        # Simulate market data (in production, fetch from Coinbase)
        np.random.seed(int(timestamp.timestamp()))
        
        # Base prices (in production, fetch from Coinbase)
        base_prices = {
            'BTC-USD': 70000.0,
            'ETH-USD': 3500.0,
            'SOL-USD': 150.0,
        }
        
        base_price = base_prices.get(symbol, 100.0)
        price = base_price * (1 + np.random.randn() * 0.001)  # 0.1% random movement
        
        # Simulate orderbook imbalance (random uniform)
        lob_imbalance = np.random.uniform(-0.5, 0.5)
        
        # Simulate bid-ask spread
        bid_ask_spread = np.random.uniform(0.0001, 0.005)
        
        # Simulate volume
        volume = np.random.uniform(10000, 1000000)
        
        # Simulate funding rate (for perpetuals)
        funding_rate = np.random.uniform(-0.001, 0.001)
        
        market_data = {
            'symbol': symbol,
            'timestamp': timestamp,
            'price': float(price),
            'volume': float(volume),
            'lob_imbalance': float(lob_imbalance),
            'bid_ask_spread': float(bid_ask_spread),
            'funding_rate': float(funding_rate),
            'high': float(price * 1.002),
            'low': float(price * 0.998),
            'open': float(price),
        }
        
        return market_data
    
    def generate_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        """
        Generate technical indicator features
        
        In production, this would compute real TA indicators
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Feature array [15]
        """
        # Simulate TA features (in production, compute from price history)
        price = market_data['price']
        
        features = np.array([
            market_data['lob_imbalance'],  # 0
            market_data['bid_ask_spread'],  # 1
            np.log10(market_data['volume']),  # 2
            price * 0.99,  # 3: ema_12 (simplified)
            price * 0.98,  # 4: ema_26 (simplified)
            (price * 0.99 - price * 0.98),  # 5: macd
            50.0 + np.random.randn() * 10,  # 6: rsi
            price * 1.01,  # 7: bb_upper
            price * 0.99,  # 8: bb_lower
            price,  # 9: bb_middle
            np.random.randn() * 0.01,  # 10: momentum_20
            np.random.randn() * 0.001,  # 11: volatility
            np.random.randn() * 0.001,  # 12: volume_momentum
            np.random.uniform(0, 1),  # 13: price_position
            market_data['funding_rate'],  # 14: funding_rate
        ], dtype=np.float32)
        
        return features
    
    def run_single_step(self, timestamp: datetime) -> List[Dict[str, Any]]:
        """
        Run a single trading step
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            List of generated signals
        """
        # Get current bag (resample daily)
        if timestamp.hour == 0 and timestamp.minute == 0:
            print(f"\n🔄 Resampling stochastic bag at {timestamp}")
            bag_result = self.resampler.resample_daily()
            self.current_bag = bag_result['bag']
            self.current_weights = bag_result['weights']
            print(f"   Selected {len(self.current_bag)} codecs, avg weight: {bag_result['avg_weight']:.3f}")
        
        if not self.current_bag:
            return []
        
        # Process each symbol in current bag
        symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']  # Simplified - would come from bag
        signals = []
        
        for symbol in symbols[:3]:  # Process up to 3 symbols
            # Get market data
            market_data = self.generate_market_data(symbol, timestamp)
            
            # Generate features
            features = self.generate_features(market_data)
            
            # Generate signals from all codecs in bag
            codec_signals = []
            codec_weights = []
            
            for codec_idx in self.current_bag:
                if codec_idx < len(self.codecs):
                    codec = self.codecs[codec_idx]
                    
                    try:
                        # Generate signal
                        confidence, direction = codec.forward(market_data, features)
                        
                        # Store signal
                        codec_signals.append((direction, confidence))
                        
                        # Store weight if available
                        if self.current_weights is not None and codec_idx < len(self.current_weights):
                            codec_weights.append(self.current_weights[codec_idx])
                        else:
                            codec_weights.append(1.0 / len(self.current_bag))
                    except Exception as e:
                        print(f"⚠️  Codec {codec_idx} failed: {e}")
                        continue
            
            if not codec_signals:
                continue
            
            # HRM decision
            market_state = {
                'price': market_data['price'],
                'volume': market_data['volume'],
                'high': market_data['high'],
                'low': market_data['low'],
                'open': market_data['open'],
                'timestamp': timestamp,
                'symbol': symbol,
            }
            
            hrm_decision = self.hrm.decide(codec_signals, market_state)
            
            # Create signal if not vetoed
            if not hrm_decision.get('vetoed', False):
                signal_strength = hrm_decision.get('aggregated_signal', 0.0)
                confidence = hrm_decision.get('confidence', 0.0)
                
                # Calculate position size
                if abs(signal_strength) > 0.3:  # Threshold
                    # Calculate position size based on risk
                    position_size = self.capital * self.risk_per_trade / abs(signal_strength)
                    
                    # Create signal
                    signal = {
                        'timestamp': timestamp.isoformat(),
                        'symbol': symbol,
                        'signal_strength': float(signal_strength),
                        'confidence': float(confidence),
                        'position_size': float(position_size),
                        'regime': hrm_decision.get('regime', 'neutral'),
                        'stop_loss': float(market_data['price'] * 0.98),  # 2% stop
                        'take_profit': float(market_data['price'] * 1.05),  # 5% target
                        'codec_id': 0,  # HRM decision
                        'weight': float(np.mean(codec_weights)) if codec_weights else 1.0,
                    }
                    
                    signals.append(signal)
                    
                    # Print signal info
                    action = 'BUY' if signal_strength > 0 else 'SELL'
                    print(f"   {action} {symbol}: strength={signal_strength:.3f}, conf={confidence:.3f}")
        
        return signals
    
    def run_live(self, duration_minutes: int = 1440) -> None:
        """
        Run live trading
        
        Args:
            duration_minutes: Duration in minutes (default: 1440 = 24 hours)
        """
        print(f"\n🚀 Starting live trading for {duration_minutes} minutes")
        
        if self.mode == 'paper':
            print("⚠️  Paper trading mode - using simulated data")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        step_count = 0
        
        while time.time() < end_time:
            try:
                timestamp = datetime.now()
                
                # Run single step
                signals = self.run_single_step(timestamp)
                
                # Process signals
                if signals:
                    # Paper trading - just log signals
                    print(f"\n📊 Generated {len(signals)} signal(s) at {timestamp}")
                    for signal in signals:
                        print(f"   {signal['symbol']}: {signal['signal_strength']:.3f}")
                    
                    # Write signals to stdout (for any external process)
                    SignalWriter.write_signals_batch(signals)
                
                # Heartbeat
                if step_count % 10 == 0:
                    SignalWriter.write_heartbeat()
                
                step_count += 1
                
                # Sleep to avoid overwhelming the system
                time.sleep(5)  # 5-second intervals
                
            except KeyboardInterrupt:
                print("\n🛑 Received keyboard interrupt, shutting down...")
                break
            except Exception as e:
                print(f"❌ Error in trading loop: {e}")
                time.sleep(10)  # Wait before retrying
        
        # Cleanup
        self.cleanup()
        
        print(f"\n✅ Live trading completed after {step_count} steps")
    
    def run_paper_trading(self, days: int = 30) -> None:
        """
        Run paper trading for specified days
        
        Args:
            days: Number of days to simulate
        """
        print(f"\n📈 Starting {days}-day paper trading")
        
        # Use PaperTradingEngine from existing module
        config = PaperTradingConfig(
            trading_mode="paper",
            initial_capital=self.capital,
            days=days,
            n_codecs=len(self.codecs),
            bag_size=self.config.get('bag_size', 30),
            use_mlx=True,
            output_dir=f"coinbase_paper_{days}days",
        )
        
        engine = PaperTradingEngine(config)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        metrics = engine.run(start_date, end_date)
        
        # Print results
        print(f"\n✅ Paper trading completed")
        print(f"   Total Return: {metrics.total_return:.2%}")
        print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"   Max Drawdown: {metrics.max_drawdown:.2%}")
        print(f"   Trade Count: {metrics.trade_count}")
        
        # Save detailed results
        results = {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'capital': self.capital,
            'metrics': {
                'total_return': metrics.total_return,
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'trade_count': metrics.trade_count,
            }
        }
        
        results_file = Path(f"coinbase_paper_{days}days/results.json")
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"   Results saved to: {results_file}")
        
        # Alpha validation
        print(f"\n🎯 Alpha Validation:")
        print(f"   Sharpe ≥ 1.8: {'✅ PASS' if metrics.sharpe_ratio >= 1.8 else '❌ FAIL'}")
        print(f"   Max DD ≤ 15%: {'✅ PASS' if metrics.max_drawdown >= -0.15 else '❌ FAIL'}")
    
    def run_test_time_adaptation(self, learning_rate: float = 0.001) -> None:
        """
        Run test-time adaptation mode
        
        Args:
            learning_rate: Learning rate for online updates
        """
        print(f"\n🔄 Starting test-time adaptation mode (LR: {learning_rate})")
        
        if self.mode == 'live':
            print("⚠️  Live adaptation requires Kotlin adapter")
            if self.kotlin_adapter is None:
                print("❌ Kotlin adapter not available")
                return
        
        # Run adaptation loop
        adaptation_batch_size = 100
        adaptation_counter = 0
        
        while True:
            try:
                # Simulate receiving data from Kotlin (would be via stdin)
                # For now, simulate with random data
                timestamp = datetime.now()
                
                # Generate training batch
                batch_data = {
                    'inputs': np.random.randn(adaptation_batch_size, 15).astype(np.float32),
                    'targets': np.random.randn(adaptation_batch_size, 2).astype(np.float32) * 0.5,
                }
                
                # Update all codecs
                updated_count = 0
                for codec in self.codecs:
                    try:
                        codec.test_time_adapter(batch_data, learning_rate=learning_rate)
                        updated_count += 1
                    except Exception as e:
                        print(f"⚠️  Codec adaptation failed: {e}")
                
                adaptation_counter += 1
                
                # Log progress
                if adaptation_counter % 10 == 0:
                    print(f"✅ Adaptation batch {adaptation_counter}: {updated_count}/{len(self.codecs)} codecs updated")
                
                # Check for stdin commands
                import select
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    line = sys.stdin.readline().strip()
                    if line:
                        try:
                            command = json.loads(line)
                            if command.get('type') == 'STOP':
                                print("🛑 Received stop command")
                                break
                        except:
                            pass
                
                time.sleep(1)  # 1-second interval
                
            except KeyboardInterrupt:
                print("\n🛑 Received keyboard interrupt, shutting down...")
                break
            except Exception as e:
                print(f"❌ Error in adaptation loop: {e}")
                time.sleep(5)
        
        print(f"\n✅ Test-time adaptation completed after {adaptation_counter} batches")
    
    def cleanup(self):
        """Cleanup resources"""
        print("\n🔄 Cleaning up resources...")
        
        # Save performance history
        if self.performance_history:
            history_file = Path("performance_history.json")
            with open(history_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
            print(f"   Performance history saved to: {history_file}")
        
        print("✅ Cleanup completed")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Coinbase Live Trading with 24 SOTA Codecs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['live', 'paper', 'adapt'],
        default='paper',
        help='Trading mode: live (execute), paper (simulate), adapt (online learning)'
    )
    
    parser.add_argument(
        '--capital',
        type=float,
        default=500.0,
        help='Initial capital in USD'
    )
    
    parser.add_argument(
        '--risk',
        type=float,
        default=0.0075,
        help='Risk per trade as fraction (0.0075 is 0.75%%)'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days for paper trading'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=1440,
        help='Duration in minutes for live trading'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate for test-time adaptation'
    )
    
    parser.add_argument(
        '--bag_size',
        type=int,
        default=30,
        help='Size of stochastic bag'
    )
    

    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Create config
    config = {
        'mode': args.mode,
        'capital': args.capital,
        'risk_per_trade': args.risk,
        'bag_size': args.bag_size,
        'kotlin_script': args.kotlin_script,
        'seed': args.seed,
    }
    
    # Initialize system
    system = CoinbaseLiveTradingSystem(config)
    
    try:
        # Run based on mode
        if args.mode == 'paper':
            system.run_paper_trading(days=args.days)
        elif args.mode == 'live':
            system.run_live(duration_minutes=args.duration)
        elif args.mode == 'adapt':
            system.run_test_time_adaptation(learning_rate=args.learning_rate)
        else:
            print(f"❌ Unknown mode: {args.mode}")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())