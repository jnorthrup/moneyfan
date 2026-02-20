#!/usr/bin/env python3
"""
Single entry point for live trading with HRM system.

Orchestrates the clean architecture:
1. core/ (pure logic)
2. mlx_adapt/ (MLX-specific inference)
3. strategies/ (concrete trading rules)
4. backtest/ (testing framework)
5. config/ (configuration)
"""
import sys
import os
import argparse
from pathlib import Path
import yaml
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import core modules
from core.hrm.high_level import HighLevelConfig, HighLevelController
from core.hrm.low_level import LowLevelConfig, LowLevelProcessor
from core.data.data_loader import DataConfig, DataLoader
from core.signals import SignalConfig, SignalAggregator
from core.risk.risk_management import RiskConfig, RiskManager

# Import MLX adaptation
try:
    from mlx_adapt import HRMModel, HRMConfig, HRMInference
    from mlx_adapt import enable_ane_optimization, setup_mlx_device
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("⚠️  MLX not available - using pure Python inference")

# Import strategies
from strategies.composite_strategy import CompositeStrategy, CompositeStrategyConfig

# Import backtest
from backtest.backtester import Backtester, BacktestConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LiveConfig:
    """Live trading configuration"""
    trading_mode: str = "paper"  # "live", "paper", "backtest"
    symbol: str = "BTC-USD"
    time_interval: str = "1H"
    initial_capital: float = 10000.0
    risk_per_trade: float = 0.01
    use_mlx: bool = True
    config_path: str = "config/trading_config.yaml"


class LiveTradingOrchestrator:
    """
    Main orchestrator for live trading.
    
    Coordinates all components:
    - Data loading
    - Signal generation
    - Risk management
    - Execution
    """
    
    def __init__(self, config: LiveConfig):
        self.config = config
        self.logger = logger
        
        # Load configuration
        self._load_configuration()
        
        # Initialize components
        self._initialize_components()
        
        # State management
        self.is_running = False
        self.position = 0.0
        
    def _load_configuration(self):
        """Load configuration from YAML"""
        config_path = Path(self.config.config_path)
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            # Update configuration
            self.config_dict = yaml_config
            self.logger.info(f"Loaded configuration from {config_path}")
        else:
            self.config_dict = {}
            self.logger.warning(f"Configuration file not found: {config_path}")
    
    def _initialize_components(self):
        """Initialize all system components"""
        self.logger.info("Initializing system components...")
        
        # 1. Data loader
        data_config = DataConfig(
            symbol_list=[self.config.symbol],
            resample_frequency=self.config.time_interval
        )
        self.data_loader = DataLoader(data_config)
        
        # 2. Signal aggregator
        signal_config = SignalConfig(
            n_models=self.config_dict.get('n_models', 5),
            n_regimes=6
        )
        self.signal_aggregator = SignalAggregator(signal_config)
        
        # 3. Risk manager
        risk_config = RiskConfig(
            risk_per_trade=self.config.risk_per_trade,
            max_position_size=self.config_dict.get('max_position_size', 0.1),
            max_total_exposure=self.config_dict.get('max_total_exposure', 0.8)
        )
        self.risk_manager = RiskManager(risk_config)
        
        # 4. HRM modules
        high_level_config = HighLevelConfig(
            n_regimes=6,
            n_models=signal_config.n_models,
            n_assets=128,
            hidden_dim=64
        )
        self.high_level_controller = HighLevelController(high_level_config)
        
        low_level_config = LowLevelConfig(
            n_features=15,
            n_assets=128,
            hidden_dim=64
        )
        self.low_level_processor = LowLevelProcessor(low_level_config)
        
        # 5. Strategy
        strategy_config = CompositeStrategyConfig(
            trend_weight=0.3,
            mean_reversion_weight=0.3,
            volatility_weight=0.4
        )
        self.strategy = CompositeStrategy(strategy_config)
        
        # 6. Backtester (for paper trading)
        backtest_config = BacktestConfig(
            initial_capital=self.config.initial_capital,
            commission=0.001,
            slippage=0.001
        )
        self.backtester = Backtester(backtest_config)
        
        # 7. MLX inference (if available)
        if self.config.use_mlx and HAS_MLX:
            self.logger.info("Initializing MLX inference...")
            setup_mlx_device()
            
            mlx_config = HRMConfig(
                n_assets=128,
                n_features=15,
                n_models=signal_config.n_models,
                seq_len=16,
                hidden_dim=64
            )
            self.mlx_model = HRMModel(mlx_config)
            self.mlx_inference = HRMInference(self.mlx_model, mlx_config)
            self.use_mlx = True
        else:
            self.logger.warning("MLX not available or disabled - using pure Python")
            self.use_mlx = False
        
        self.logger.info("All components initialized successfully")
    
    def run(self):
        """Main trading loop"""
        self.logger.info(f"Starting live trading in {self.config.trading_mode} mode...")
        self.is_running = True
        
        try:
            while self.is_running:
                # 1. Get latest data
                data = self._get_latest_data()
                if data is None:
                    self.logger.warning("No data available, waiting...")
                    self._sleep(60)  # Wait 1 minute
                    continue
                
                # 2. Process data
                signals, regime_weights = self._process_data(data)
                
                # 3. Generate trading decision
                decision = self._generate_decision(signals, regime_weights, data)
                
                # 4. Execute (or simulate) trade
                self._execute_decision(decision, data)
                
                # 5. Wait for next interval
                self._sleep(self._get_interval_seconds())
                
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, shutting down...")
        except Exception as e:
            self.logger.error(f"Error in trading loop: {e}", exc_info=True)
        finally:
            self.shutdown()
    
    def _get_latest_data(self) -> Optional[Dict[str, Any]]:
        """Get latest data for trading"""
        if self.config.trading_mode == "backtest":
            # For backtest, generate synthetic data
            import numpy as np
            timestamps = [datetime.now()]
            prices = np.array([[100.0]])  # Single symbol
            return {
                'prices': prices,
                'timestamps': timestamps,
                'features': np.random.randn(1, 15)  # [time, features]
            }
        else:
            # In real implementation, this would fetch from exchange
            # For now, generate synthetic data
            import numpy as np
            current_price = 100.0 + np.random.randn() * 5
            timestamps = [datetime.now()]
            prices = np.array([[current_price]])
            features = np.random.randn(1, 15)  # [time, features]
            
            return {
                'prices': prices,
                'timestamps': timestamps,
                'features': features
            }
    
    def _process_data(self, data: Dict[str, Any]) -> tuple:
        """Process data to generate signals"""
        features = data['features']
        
        # Process through low-level module
        low_level_feature, _ = self.low_level_processor.process(features)
        
        # Aggregate signals
        aggregated_signal = self.signal_aggregator.aggregate_from_features(
            features, data['timestamps'][0], self.config.symbol
        )
        
        # Compute regime weights
        regime_weights = self.signal_aggregator.signal_generator.compute_regime_weights(
            [aggregated_signal], 6
        )
        
        return [aggregated_signal], regime_weights
    
    def _generate_decision(self, 
                          signals, 
                          regime_weights, 
                          data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading decision"""
        if not signals:
            return {'action': 'hold', 'size': 0.0, 'reason': 'no_signal'}
        
        # Get current price
        current_price = data['prices'][0, 0]
        
        # Use strategy to compute signal and position size
        signal_strength, confidence, regime = self.strategy.compute_signal(
            data['prices'][0],  # Single price for single symbol
            data['timestamps'][0]
        )
        
        # Compute position size
        position_size = self.strategy.compute_position_size(
            signal_strength, current_price, self.risk_manager.portfolio_value
        )
        
        # Determine action
        if abs(signal_strength) < 0.3:
            action = 'hold'
            reason = 'weak_signal'
        elif signal_strength > 0:
            action = 'buy'
            reason = f'strong_buy_in_{regime}'
        else:
            action = 'sell'
            reason = f'strong_sell_in_{regime}'
        
        return {
            'action': action,
            'size': abs(position_size) if position_size > 0 else 0.0,
            'signal_strength': signal_strength,
            'confidence': confidence,
            'regime': regime,
            'reason': reason,
            'price': current_price,
            'timestamp': data['timestamps'][0]
        }
    
    def _execute_decision(self, decision: Dict[str, Any], data: Dict[str, Any]):
        """Execute (or simulate) trading decision"""
        action = decision['action']
        size = decision['size']
        price = decision['price']
        timestamp = decision['timestamp']
        
        if self.config.trading_mode == "live":
            # Real trading (would connect to exchange API)
            self.logger.info(f"[LIVE] {action.upper()} {size:.4f} units at ${price:.2f}")
            # In real implementation, this would call exchange API
            # self._execute_on_exchange(action, size, price)
            
        elif self.config.trading_mode == "paper":
            # Paper trading
            self.logger.info(f"[PAPER] {action.upper()} {size:.4f} units at ${price:.2f}")
            
            # Update risk manager
            if action == 'buy':
                self.risk_manager.open_position(
                    symbol=self.config.symbol,
                    size=size,
                    price=price,
                    volatility=0.05,  # Simplified
                    signal_direction=1.0,
                    timestamp=timestamp
                )
            elif action == 'sell':
                self.risk_manager.close_position(
                    symbol=self.config.symbol,
                    reason=decision['reason'],
                    timestamp=timestamp
                )
            
        else:  # backtest
            self.logger.info(f"[BACKTEST] {action.upper()} {size:.4f} units at ${price:.2f}")
            
        # Log the decision
        self.logger.info(f"Decision: {decision}")
    
    def _get_interval_seconds(self) -> int:
        """Get interval in seconds based on time_interval"""
        interval_map = {
            '1H': 3600,
            '4H': 14400,
            '1D': 86400
        }
        return interval_map.get(self.config.time_interval, 3600)
    
    def _sleep(self, seconds: int):
        """Sleep for specified seconds"""
        import time
        self.logger.info(f"Sleeping for {seconds} seconds...")
        time.sleep(seconds)
    
    def shutdown(self):
        """Shutdown trading system"""
        self.logger.info("Shutting down trading system...")
        self.is_running = False
        
        # Save final state
        if self.config.trading_mode in ["paper", "live"]:
            metrics = self.risk_manager.compute_portfolio_metrics()
            self.logger.info(f"Final metrics: {metrics}")
        
        self.logger.info("Trading system shutdown complete")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='HRM Live Trading System')
    
    parser.add_argument(
        '--mode', 
        choices=['live', 'paper', 'backtest'],
        default='paper',
        help='Trading mode'
    )
    
    parser.add_argument(
        '--symbol',
        default='BTC-USD',
        help='Trading symbol'
    )
    
    parser.add_argument(
        '--interval',
        choices=['1H', '4H', '1D'],
        default='1H',
        help='Time interval'
    )
    
    parser.add_argument(
        '--capital',
        type=float,
        default=10000.0,
        help='Initial capital'
    )
    
    parser.add_argument(
        '--risk',
        type=float,
        default=0.01,
        help='Risk per trade (0.01 = 1%)'
    )
    
    parser.add_argument(
        '--no-mlx',
        action='store_true',
        help='Disable MLX inference'
    )
    
    parser.add_argument(
        '--config',
        default='config/trading_config.yaml',
        help='Configuration file path'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Create configuration
    config = LiveConfig(
        trading_mode=args.mode,
        symbol=args.symbol,
        time_interval=args.interval,
        initial_capital=args.capital,
        risk_per_trade=args.risk,
        use_mlx=not args.no_mlx,
        config_path=args.config
    )
    
    # Create and run orchestrator
    orchestrator = LiveTradingOrchestrator(config)
    
    try:
        orchestrator.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())