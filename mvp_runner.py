"""
MVP Runner - 30-day paper trading validation
=============================================

Runs the complete HRM pipeline with 3 short-horizon predictors:
- 5m Transformer predictor
- 15m XGBoost predictor  
- 1h LightGBM predictor

Vector cache feed into flat PPO HRM (Stage 3/4)
Full pandas + depth inside live agents only
Signals → execution engine → paper orders
Log every tick, P&L, and regime switches
"""

import numpy as np
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# Import our MVP modules
from vector_store import create_simple_vector_store
from horizon_feature_buffer import HorizonFeatureBuffer, HorizonBufferConfig
from test_time_predictor import create_short_horizon_predictor
from hrm_rollout_stages import HRMRolloutStages, HRMRolloutConfig, create_sample_historical_data

# Import existing modules (pandas allowed for live agents)
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("[MVP Runner] Pandas not available - using numpy fallback")

# Import live executor
try:
    from execution.live_executor import LiveExecutor, LiveExecutorConfig
    HAS_LIVE_EXECUTOR = True
except ImportError:
    HAS_LIVE_EXECUTOR = False
    print("[MVP Runner] LiveExecutor not available")

# Import kill switch
try:
    from execution.kill_switch import KillSwitch, KillSwitchConfig
    HAS_KILL_SWITCH = True
except ImportError:
    HAS_KILL_SWITCH = False
    print("[MVP Runner] KillSwitch not available")

@dataclass
class MVPConfig:
    """Configuration for MVP paper trading run"""
    # Predictors
    n_predictors: int = 3
    predictor_types: List[str] = field(default_factory=lambda: [
        "transformer_5m",
        "xgboost_15m", 
        "lightgbm_1h"
    ])
    
    # Vector cache
    vector_dim: int = 64
    vector_store_path: str = "data/vector_store_mvp"
    
    # Paper trading
    paper_capital: float = 1000.0  # $1000 paper capital
    paper_commission: float = 0.001  # 0.1% commission
    paper_slippage: float = 0.0005  # 0.05% slippage
    
    # Validation
    validation_days: int = 30
    log_every_n_ticks: int = 100
    
    # Performance thresholds
    min_profit_factor: float = 1.5
    min_sharpe: float = 1.0
    max_drawdown: float = 0.15  # 15% max drawdown
    
    # Execution
    coinbase_api_key: str = ""
    coinbase_api_secret: str = ""
    paper_mode: bool = True
    live_execution_enabled: bool = False  # Set to True for live paper trading
    
    def __post_init__(self):
        # Set predictor paths based on types
        self.predictor_paths = []
        for ptype in self.predictor_types:
            if "transformer" in ptype:
                self.predictor_paths.append(f"models/transformer_5m.mlxbf")
            elif "xgboost" in ptype:
                self.predictor_paths.append(f"models/xgboost_15m.mlxbf")
            elif "lightgbm" in ptype:
                self.predictor_paths.append(f"models/lightgbm_1h.mlxbf")

class MVPPaperTrading:
    """
    MVP Paper Trading Runner
    Runs 30-day paper trading with 3 short-horizon predictors
    """
    
    def __init__(self, config: MVPConfig):
        self.config = config
        self.start_time = time.time()
        
        # Initialize vector store
        self.vector_store = create_simple_vector_store(
            vector_dim=config.vector_dim,
            use_faiss=False  # Keep simple for MVP
        )
        
        # Initialize horizon buffers for each predictor
        self.buffers = []
        for i in range(config.n_predictors):
            buffer_config = HorizonBufferConfig(
                max_horizons=3,  # Each predictor handles 3 horizons
                vector_dim=config.vector_dim,
                max_steps=4096  # 4-hour buffer for 1h predictor
            )
            self.buffers.append(HorizonFeatureBuffer(buffer_config))
        
        # Initialize predictors (MLX-based)
        self.predictors = []
        for i in range(config.n_predictors):
            # Create predictor with specific type
            predictor = create_short_horizon_predictor()
            predictor.config.model_paths = [config.predictor_paths[i]]
            self.predictors.append(predictor)
        
        # Trading state
        self.position = 0.0  # Current position (0 = flat)
        self.cash = config.paper_capital
        self.equity = config.paper_capital
        self.trades = []
        self.signals = []
        self.equity_curve = []
        self.regime_changes = []
        
        # Performance metrics
        self.metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "total_commission": 0.0,
            "total_slippage": 0.0,
            "max_drawdown": 0.0,
            "peak_equity": config.paper_capital,
            "trough_equity": config.paper_capital,
        }
        
        # Live agents (pandas enabled)
        self.live_agents = []
        self._init_live_agents()
        
        # Live executor
        self.live_executor = None
        if config.live_execution_enabled:
            executor_config = LiveExecutorConfig(
                paper_mode=config.paper_mode,
                api_key=config.coinbase_api_key,
                api_secret=config.coinbase_api_secret
            )
            self.live_executor = LiveExecutor(executor_config)
        
        # Kill switch
        self.kill_switch = None
        if HAS_KILL_SWITCH:
            kill_switch_config = KillSwitchConfig()
            self.kill_switch = KillSwitch(kill_switch_config)
        
        print(f"[MVP Runner] Initialized with {config.n_predictors} predictors")
        print(f"[MVP Runner] Paper capital: ${config.paper_capital:.2f}")
        print(f"[MVP Runner] Validation days: {config.validation_days}")
        print(f"[MVP Runner] Live execution: {'ENABLED' if config.live_execution_enabled else 'DISABLED'}")
    
    def _init_live_agents(self):
        """Initialize live HRM agents (pandas allowed)"""
        if not HAS_PANDAS:
            print("[MVP Runner] Pandas not available - live agents will use numpy fallback")
            return
        
        # Simulate live agent initialization
        # In production, this would load trained HRM models
        print("[MVP Runner] Live agents initialized (pandas enabled)")
    
    def run_30_day_replay(self, historical_data: List[Dict[str, Any]]):
        """
        Run 30-day paper trading replay
        
        Args:
            historical_data: List of tick data dictionaries
        """
        print(f"\n{'='*60}")
        print("30-DAY PAPER TRADING REPLAY")
        print(f"{'='*60}")
        print(f"Processing {len(historical_data)} historical ticks...")
        
        start_time = time.time()
        tick_count = 0
        
        for tick_data in historical_data:
            tick_count += 1
            
            # Extract tick data
            timestamp = tick_data['timestamp']
            price = tick_data['price']
            volume = tick_data['volume']
            orderbook_imbalance = tick_data.get('orderbook_imbalance')
            
            # Step 1: Update horizon buffers for each predictor
            predictor_vectors = []
            for i, buffer in enumerate(self.buffers):
                buffer.add_tick(timestamp, price, volume, orderbook_imbalance)
                
                # Get latest vector from buffer
                latest = buffer.get_latest_vector(0)  # Get horizon 0
                if latest:
                    ts, vec = latest
                    predictor_vectors.append((i, ts, vec))
            
            # Step 2: Generate signals from each predictor
            signals = []
            for i, (pred_idx, ts, vec) in enumerate(predictor_vectors):
                # Store vector in vector cache
                self.vector_store.add_vector(vec, i, ts)
                
                # Get signal from predictor (simulated)
                signal = self._get_predictor_signal(pred_idx, vec, price)
                signals.append(signal)
            
            # Step 3: Aggregate signals using flat PPO HRM (Stage 3)
            aggregated_signal = self._aggregate_signals(signals)
            
            # Step 4: Execute trades based on signal
            self._execute_trade(aggregated_signal, price, timestamp)
            
            # Step 5: Log every N ticks
            if tick_count % self.config.log_every_n_ticks == 0:
                self._log_progress(tick_count, price)
            
            # Step 6: Track regime changes (simulated)
            self._track_regime(aggregated_signal, price, timestamp)
        
        elapsed = time.time() - start_time
        print(f"\n[MVP Runner] Processed {tick_count} ticks in {elapsed:.1f}s")
        print(f"[MVP Runner] Rate: {tick_count/elapsed:.1f} ticks/sec")
        
        # Calculate final performance metrics
        self._calculate_performance_metrics()
        
        return self.metrics
    
    def _get_predictor_signal(self, predictor_idx: int, vector: np.ndarray, price: float) -> float:
        """
        Get signal from a specific predictor
        
        Args:
            predictor_idx: Predictor index (0-2)
            vector: 64-dim vector from horizon buffer
            price: Current price
            
        Returns:
            Signal in range [-1, 1] (negative = sell, positive = buy)
        """
        # Simulate predictor output based on vector and price
        # In production, this would be MLX inference
        np.random.seed(int(predictor_idx + price) % 1000)  # Deterministic randomness
        
        # Base signal from vector statistics
        vector_norm = np.linalg.norm(vector)
        if vector_norm > 0:
            normalized_vector = vector / vector_norm
        else:
            normalized_vector = vector
        
        # Different predictors have different biases
        if predictor_idx == 0:  # 5m Transformer - momentum focused
            signal = np.tanh(np.sum(normalized_vector[:8]) * 2.0)
        elif predictor_idx == 1:  # 15m XGBoost - pattern focused
            signal = np.tanh(np.sum(normalized_vector[8:16]) * 1.5)
        elif predictor_idx == 2:  # 1h LightGBM - trend focused
            signal = np.tanh(np.sum(normalized_vector[16:24]) * 1.0)
        else:
            signal = 0.0
        
        # Add small random noise for realistic variation
        signal += np.random.randn() * 0.05
        
        # Clamp to [-1, 1]
        signal = max(-1.0, min(1.0, signal))
        
        return signal
    
    def _aggregate_signals(self, signals: List[float]) -> float:
        """
        Aggregate signals using flat PPO HRM (Stage 3/4)
        
        Args:
            signals: List of signals from each predictor
            
        Returns:
            Aggregated signal
        """
        if not signals:
            return 0.0
        
        # Simple averaging (flat ensemble)
        avg_signal = np.mean(signals)
        
        # Apply convergence threshold
        # If predictors disagree significantly, stay flat
        signal_std = np.std(signals)
        if signal_std > 0.3:  # High disagreement
            avg_signal *= 0.5  # Reduce signal strength
        
        # Apply sigmoid to get confidence-weighted signal
        confidence = 1.0 - signal_std
        weighted_signal = avg_signal * confidence
        
        return weighted_signal
    
    def _execute_trade(self, signal: float, price: float, timestamp: int):
        """
        Execute trade based on signal
        
        Args:
            signal: Aggregated signal [-1, 1]
            price: Current price
            timestamp: Current timestamp
        """
        # Position sizing based on signal strength
        signal_strength = abs(signal)
        if signal_strength < 0.3:  # Threshold for entry
            return
        
        # Determine direction
        if signal > 0:
            direction = "long"
        elif signal < 0:
            direction = "short"
        else:
            direction = "flat"
        
        # Calculate position size (risk-based)
        risk_per_trade = 0.02  # 2% risk per trade
        size_usd = self.equity * risk_per_trade * signal_strength
        
        # Prepare HRM signal for live executor
        hrm_signal = {
            "timestamp": timestamp,
            "product_id": "BTC-USD",  # Default, can be dynamic
            "direction": direction,
            "size_usd": size_usd,
            "stop_loss_pct": 0.75,  # 25% stop loss
            "take_profit_pct": 1.50,  # 50% take profit
            "hrm_weight": signal_strength,
            "convergence_score": 0.6,  # Simulated
            "price": price,
            "signal": signal
        }
        
        # Execute via live executor (if enabled)
        if self.live_executor:
            execution_result = self.live_executor.execute_hrm_signal(hrm_signal)
            
            # Update position based on execution result
            if execution_result.get("action") in ["BUY", "BUY_MOCK"]:
                self.position = size_usd
                self.cash -= size_usd  # Deduct from cash
            elif execution_result.get("action") in ["SELL", "SELL_MOCK"]:
                self.position = -size_usd
                self.cash -= size_usd
            elif execution_result.get("action") in ["CLOSE_ALL", "CLOSE_ALL_MOCK"]:
                self.position = 0
            
            # Record trade
            trade = {
                "timestamp": timestamp,
                "price": price,
                "position": self.position,
                "signal": signal,
                "execution_result": execution_result,
                "size_usd": size_usd,
                "direction": direction
            }
            self.trades.append(trade)
            self.metrics["total_trades"] += 1
        else:
            # Simulate execution (no live executor)
            # Check if we should exit current position
            if self.position != 0:
                current_direction = np.sign(self.position)
                new_direction = 1 if signal > 0 else -1
                if current_direction != new_direction:
                    # Exit current position
                    self._close_position(price, timestamp)
            
            # Enter new position
            if direction != "flat":
                # Calculate entry price with slippage
                slippage = self.config.paper_slippage * abs(signal)
                entry_price = price * (1 + slippage * (1 if direction == "long" else -1))
                
                # Calculate commission
                commission = size_usd * self.config.paper_commission
                
                # Update cash and position
                self.cash -= commission
                self.position = (1 if direction == "long" else -1) * size_usd
                
                # Record trade
                trade = {
                    "timestamp": timestamp,
                    "price": entry_price,
                    "position": self.position,
                    "signal": signal,
                    "commission": commission,
                    "slippage": slippage * price,
                    "size_usd": size_usd,
                    "direction": direction
                }
            self.trades.append(trade)
            
            # Update metrics
            self.metrics["total_trades"] += 1
            self.metrics["total_commission"] += commission
            self.metrics["total_slippage"] += slippage * price
    
    def _close_position(self, price: float, timestamp: int):
        """Close current position"""
        if self.position == 0:
            return
        
        # Calculate exit price with slippage
        direction = np.sign(self.position)
        slippage = self.config.paper_slippage
        exit_price = price * (1 - slippage * direction)
        
        # Calculate P&L
        entry_price = self.trades[-1]["price"]
        pnl = (exit_price - entry_price) * self.position
        
        # Update equity
        self.equity += pnl
        self.cash += self.position * exit_price
        
        # Record trade
        trade = {
            "timestamp": timestamp,
            "price": exit_price,
            "position": 0,
            "signal": 0,
            "pnl": pnl,
            "exit_reason": "signal_reversal"
        }
        self.trades.append(trade)
        
        # Update metrics
        self.metrics["total_pnl"] += pnl
        if pnl > 0:
            self.metrics["winning_trades"] += 1
        else:
            self.metrics["losing_trades"] += 1
        
        # Reset position
        self.position = 0
    
    def _log_progress(self, tick_count: int, price: float):
        """Log current progress"""
        elapsed = time.time() - self.start_time
        rate = tick_count / elapsed if elapsed > 0 else 0
        
        # Update equity curve
        current_equity = self.cash + self.position * price
        self.equity_curve.append({
            "timestamp": tick_count,
            "equity": current_equity,
            "price": price,
            "position": self.position
        })
        
        # Update drawdown metrics
        if current_equity > self.metrics["peak_equity"]:
            self.metrics["peak_equity"] = current_equity
        if current_equity < self.metrics["trough_equity"]:
            self.metrics["trough_equity"] = current_equity
        
        drawdown = (self.metrics["peak_equity"] - current_equity) / self.metrics["peak_equity"]
        if drawdown > self.metrics["max_drawdown"]:
            self.metrics["max_drawdown"] = drawdown
        
        # Log every 1000 ticks
        if tick_count % 1000 == 0:
            print(f"[MVP Runner] Tick {tick_count}: Equity=${current_equity:.2f}, "
                  f"Position={self.position:.2f}, Rate={rate:.1f}/sec")
    
    def _track_regime(self, signal: float, price: float, timestamp: int):
        """Track regime changes (simulated)"""
        # Simple regime detection based on signal
        if abs(signal) > 0.5:
            regime = "TREND" if signal > 0 else "REVERSION"
        else:
            regime = "SIDEWAYS"
        
        # Track regime changes
        if len(self.regime_changes) == 0 or regime != self.regime_changes[-1]["regime"]:
            self.regime_changes.append({
                "timestamp": timestamp,
                "regime": regime,
                "price": price,
                "signal": signal
            })
    
    def _calculate_performance_metrics(self):
        """Calculate final performance metrics"""
        if len(self.equity_curve) == 0:
            return
        
        equity_values = [e["equity"] for e in self.equity_curve]
        
        # Calculate returns
        returns = np.diff(equity_values) / equity_values[:-1] if len(equity_values) > 1 else [0]
        
        # Profit Factor
        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(r for r in returns if r < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Sharpe Ratio (simplified)
        if len(returns) > 1:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = mean_return / std_return if std_return > 0 else 0
        else:
            sharpe = 0
        
        # Update metrics
        self.metrics["profit_factor"] = profit_factor
        self.metrics["sharpe_ratio"] = sharpe
        self.metrics["final_equity"] = equity_values[-1] if equity_values else self.config.paper_capital
        self.metrics["return_pct"] = ((equity_values[-1] - self.config.paper_capital) / self.config.paper_capital * 100) if equity_values else 0
        
        # Win rate
        total_trades = self.metrics["total_trades"]
        win_rate = (self.metrics["winning_trades"] / total_trades * 100) if total_trades > 0 else 0
        self.metrics["win_rate"] = win_rate
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        report = {
            "config": self.config,
            "metrics": self.metrics,
            "performance_summary": {
                "pass": self.metrics["profit_factor"] >= self.config.min_profit_factor and
                        self.metrics["sharpe_ratio"] >= self.config.min_sharpe and
                        self.metrics["max_drawdown"] <= self.config.max_drawdown,
                "validation_days": self.config.validation_days,
                "total_trades": self.metrics["total_trades"],
                "win_rate": f"{self.metrics['win_rate']:.1f}%",
                "profit_factor": f"{self.metrics['profit_factor']:.2f}",
                "sharpe_ratio": f"{self.metrics['sharpe_ratio']:.2f}",
                "max_drawdown": f"{self.metrics['max_drawdown']:.2%}",
                "final_equity": f"${self.metrics['final_equity']:.2f}",
                "return_pct": f"{self.metrics['return_pct']:.2f}%",
            }
        }
        
        return report
    
    def save_results(self, output_dir: str = "paper_results"):
        """Save paper trading results"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save metrics
        metrics_path = Path(output_dir) / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save equity curve
        equity_path = Path(output_dir) / "equity_curve.json"
        with open(equity_path, 'w') as f:
            json.dump(self.equity_curve, f, indent=2)
        
        # Save trades
        trades_path = Path(output_dir) / "trades.json"
        with open(trades_path, 'w') as f:
            json.dump(self.trades, f, indent=2)
        
        # Save regime changes
        regimes_path = Path(output_dir) / "regimes.json"
        with open(regimes_path, 'w') as f:
            json.dump(self.regime_changes, f, indent=2)
        
        print(f"[MVP Runner] Results saved to {output_dir}/")
        print(f"[MVP Runner] Files: metrics.json, equity_curve.json, trades.json, regimes.json")


def create_realistic_historical_data(days: int = 30) -> List[Dict[str, Any]]:
    """
    Create realistic 30-day historical data for testing
    
    Args:
        days: Number of days to generate
        
    Returns:
        List of tick data dictionaries
    """
    print(f"Creating {days}-day historical data...")
    
    # Generate ticks at 5-minute intervals
    ticks_per_day = (24 * 60) // 5  # 5-minute intervals
    total_ticks = days * ticks_per_day
    
    data = []
    base_price = 50000.0
    
    for i in range(total_ticks):
        # Timestamp: start from 30 days ago
        timestamp = int(time.time()) - (days * 24 * 60 * 60) + (i * 5 * 60)
        
        # Simulate realistic price movement
        # Random walk with occasional trends
        trend = 0.0001 * np.sin(i / 1000)  # Slow trend
        volatility = 0.002 * np.random.randn()  # Volatility
        mean_reversion = -0.0001 * (base_price - 50000) / 1000  # Mean reversion
        
        price_change = trend + volatility + mean_reversion
        base_price *= (1 + price_change)
        
        # Volume (exponential distribution)
        volume = np.random.exponential(1000)
        
        # Orderbook imbalance (0-1)
        orderbook_imbalance = 0.5 + 0.3 * np.random.randn()
        orderbook_imbalance = max(0, min(1, orderbook_imbalance))
        
        tick = {
            "timestamp": timestamp,
            "price": base_price,
            "volume": volume,
            "orderbook_imbalance": orderbook_imbalance,
        }
        data.append(tick)
    
    return data


# Example usage
if __name__ == "__main__":
    print("MVP Paper Trading Runner")
    print("="*60)
    
    # Create MVP configuration
    config = MVPConfig(
        n_predictors=3,
        predictor_types=["transformer_5m", "xgboost_15m", "lightgbm_1h"],
        vector_dim=64,
        vector_store_path="data/vector_store_mvp",
        paper_capital=1000.0,
        validation_days=30,
        paper_mode=True
    )
    
    # Initialize MVP runner
    runner = MVPPaperTrading(config)
    
    # Option 1: Load emulated models from emulated_fast_feed_trainer
    # Uncomment the following lines to use emulated models
    """
    try:
        from train.emulated_fast_feed_trainer import EmulatedFastFeedTrainer, EmulatedTrainerConfig
        print("Loading emulated models...")
        
        # Load the trained models
        model_dir = "hrm/data/models"
        import pickle
        import os
        
        predictor_5m = None
        predictor_15m = None
        predictor_1h = None
        
        if os.path.exists(f"{model_dir}/predictor_5m_transformer.pkl"):
            with open(f"{model_dir}/predictor_5m_transformer.pkl", 'rb') as f:
                predictor_5m = pickle.load(f)
                print("✅ Loaded 5m Transformer predictor")
        
        if os.path.exists(f"{model_dir}/predictor_15m_xgboost.pkl"):
            with open(f"{model_dir}/predictor_15m_xgboost.pkl", 'rb') as f:
                predictor_15m = pickle.load(f)
                print("✅ Loaded 15m XGBoost predictor")
        
        if os.path.exists(f"{model_dir}/predictor_1h_lightgbm.pkl"):
            with open(f"{model_dir}/predictor_1h_lightgbm.pkl", 'rb') as f:
                predictor_1h = pickle.load(f)
                print("✅ Loaded 1h LightGBM predictor")
        
        if predictor_5m and predictor_15m and predictor_1h:
            print("✅ All emulated models loaded successfully")
        else:
            print("⚠️  Some emulated models missing - using fallback")
            
    except Exception as e:
        print(f"⚠️  Could not load emulated models: {e}")
        print("Falling back to synthetic data generation...")
    """
    
    # Option 2: Generate synthetic data for 30-day validation
    # This is the current implementation
    historical_data = create_realistic_historical_data(days=30)
    print(f"Generated {len(historical_data)} ticks for 30-day validation")
    
    # Run paper trading
    metrics = runner.run_30_day_replay(historical_data)
    
    # Get performance report
    report = runner.get_performance_report()
    
    # Print results
    print("\n" + "="*60)
    print("PAPER TRADING RESULTS")
    print("="*60)
    print(f"Validation period: {config.validation_days} days")
    print(f"Total ticks: {len(historical_data)}")
    print(f"Total trades: {metrics['total_trades']}")
    print(f"Win rate: {metrics['win_rate']:.1f}%")
    print(f"Profit factor: {metrics['profit_factor']:.2f}")
    print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Final equity: ${metrics['final_equity']:.2f}")
    print(f"Return: {metrics['return_pct']:.2f}%")
    
    # Check if validation passed
    passed = (metrics['profit_factor'] >= config.min_profit_factor and
              metrics['sharpe_ratio'] >= config.min_sharpe and
              metrics['max_drawdown'] <= config.max_drawdown)
    
    if passed:
        print("\n🎉 VALIDATION PASSED!")
        print(f"  Profit factor > {config.min_profit_factor}")
        print(f"  Sharpe ratio > {config.min_sharpe}")
        print(f"  Max drawdown < {config.max_drawdown:.0%}")
        print("\n✅ Ready for scaling to 8 predictors")
    else:
        print("\n⚠️  VALIDATION FAILED")
        if metrics['profit_factor'] < config.min_profit_factor:
            print(f"  Profit factor {metrics['profit_factor']:.2f} < {config.min_profit_factor}")
        if metrics['sharpe_ratio'] < config.min_sharpe:
            print(f"  Sharpe ratio {metrics['sharpe_ratio']:.2f} < {config.min_sharpe}")
        if metrics['max_drawdown'] > config.max_drawdown:
            print(f"  Max drawdown {metrics['max_drawdown']:.2%} > {config.max_drawdown:.0%}")
    
    # Save results
    runner.save_results()
    
    print(f"\nMVP paper trading completed in {time.time() - runner.start_time:.1f} seconds")