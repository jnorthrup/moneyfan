"""
Historical Training Session - HRM Learning Progress & Metrics
==============================================================

Shows the prevailing agent and standard measuring metrics of traders,
with 24-hour HRM learning/loss visualization during historical training.
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import seaborn as sns

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import HRM modules
try:
    from hrm_rollout_stages import HRMRolloutStages, HRMRolloutConfig
    HAS_HRM = True
except ImportError:
    HAS_HRM = False
    print("[HistoricalTrainer] HRM not available")

# Import predictor modules
try:
    from test_time_predictor import create_short_horizon_predictor
    HAS_PREDICTORS = True
except ImportError:
    HAS_PREDICTORS = False
    print("[HistoricalTrainer] Predictors not available")

@dataclass
class HistoricalTrainingConfig:
    """Configuration for historical training session"""
    # Time configuration
    start_date: datetime = field(default_factory=lambda: datetime(2024, 1, 1))
    end_date: datetime = field(default_factory=lambda: datetime(2024, 1, 8))
    session_duration_hours: int = 24
    
    # Symbol configuration
    symbol: str = "BTC-USD"
    timeframe: str = "1h"
    
    # Training configuration
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    
    # Visualization settings
    update_frequency_seconds: int = 5  # Update metrics every 5 seconds
    display_metrics: List[str] = field(default_factory=lambda: [
        "loss", "accuracy", "profit_factor", "sharpe_ratio",
        "win_rate", "max_drawdown", "equity", "position_size"
    ])
    
    # Data source
    data_source: str = "duckdb"  # or "arrow", "csv"
    data_path: str = "hrm/data/market.duckdb"
    
    # Output settings
    output_dir: str = "training_sessions"
    save_plots: bool = True
    save_metrics: bool = True


class PrevailingAgentMetrics:
    """Tracks metrics for the prevailing agent (HRM)"""
    
    def __init__(self):
        self.metrics_history = []
        self.current_metrics = {
            "epoch": 0,
            "timestamp": datetime.now(),
            "loss": 0.0,
            "accuracy": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "equity": 1000.0,  # Starting capital
            "position_size": 0.0,
            "trade_count": 0,
            "total_pnl": 0.0,
            "hrm_reward": 0.0,
            "veto_rate": 0.0,
            "regime_confidence": 0.0,
            # Winning agent specific stats
            "winning_trades": 0,
            "losing_trades": 0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "profit_factor_composite": 0.0,
            "expectancy": 0.0,
            "recovery_factor": 0.0,
            "risk_adjusted_return": 0.0,
            "drawdown_depth": 0.0,
            "drawdown_duration": 0,
            "consecutive_wins": 0,
            "consecutive_losses": 0,
            "win_streak": 0,
            "loss_streak": 0,
        }
    
    def update(self, epoch: int, **kwargs):
        """Update metrics for current epoch"""
        self.current_metrics["epoch"] = epoch
        self.current_metrics["timestamp"] = datetime.now()
        
        for key, value in kwargs.items():
            if key in self.current_metrics:
                self.current_metrics[key] = value
        
        # Record history
        self.metrics_history.append(self.current_metrics.copy())
    
    def get_latest(self) -> Dict[str, Any]:
        """Get latest metrics"""
        return self.current_metrics.copy()
    
    def get_history(self) -> pd.DataFrame:
        """Get metrics history as DataFrame"""
        if not self.metrics_history:
            return pd.DataFrame()
        return pd.DataFrame(self.metrics_history)


class TraderMetrics:
    """Standard measuring metrics for traders"""
    
    def __init__(self):
        self.trader_metrics = {}
        
    def _calculate_winning_agent_stats(self, trades: List[Dict]) -> Dict[str, float]:
        """Calculate detailed winning agent statistics"""
        if not trades:
            return self._default_winning_stats()
        
        df = pd.DataFrame(trades)
        
        # Separate winning and losing trades
        winning_trades = df[df['pnl'] > 0]
        losing_trades = df[df['pnl'] < 0]
        
        # Basic counts
        total_trades = len(df)
        num_wins = len(winning_trades)
        num_losses = len(losing_trades)
        
        # Win/Loss values
        win_values = winning_trades['pnl'].values if num_wins > 0 else np.array([0])
        loss_values = losing_trades['pnl'].values if num_losses > 0 else np.array([0])
        
        # Averages
        avg_win = win_values.mean() if num_wins > 0 else 0
        avg_loss = abs(loss_values.mean()) if num_losses > 0 else 0
        
        # Largest win/loss
        largest_win = win_values.max() if num_wins > 0 else 0
        largest_loss = loss_values.min() if num_losses > 0 else 0
        
        # Profit factors
        gross_profit = win_values.sum() if num_wins > 0 else 0
        gross_loss = abs(loss_values.sum()) if num_losses > 0 else 0
        profit_factor_composite = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expectancy
        if total_trades > 0:
            win_rate = num_wins / total_trades
            avg_win_actual = avg_win
            avg_loss_actual = avg_loss
            expectancy = (win_rate * avg_win_actual) - ((1 - win_rate) * avg_loss_actual)
        else:
            expectancy = 0
        
        # Recovery Factor (P&L / Max Drawdown)
        total_pnl = df['pnl'].sum()
        max_dd = self._calculate_max_drawdown(df['pnl'].values)
        recovery_factor = abs(total_pnl / max_dd) if max_dd > 0 else float('inf')
        
        # Risk Adjusted Return (Sharpe-like but simpler)
        if len(df) > 1:
            returns = df['pnl'].values
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            risk_adjusted_return = mean_return / std_return if std_return > 0 else 0
        else:
            risk_adjusted_return = 0
        
        # Drawdown metrics
        drawdown_depth, drawdown_duration = self._calculate_drawdown_metrics(df['pnl'].values)
        
        # Consecutive streaks
        consecutive_wins, consecutive_losses, win_streak, loss_streak = self._calculate_streaks(df['pnl'].values)
        
        return {
            "winning_trades": num_wins,
            "losing_trades": num_losses,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "profit_factor_composite": profit_factor_composite,
            "expectancy": expectancy,
            "recovery_factor": recovery_factor,
            "risk_adjusted_return": risk_adjusted_return,
            "drawdown_depth": drawdown_depth,
            "drawdown_duration": drawdown_duration,
            "consecutive_wins": consecutive_wins,
            "consecutive_losses": consecutive_losses,
            "win_streak": win_streak,
            "loss_streak": loss_streak,
        }
    
    def _calculate_max_drawdown(self, pnl_series: np.ndarray) -> float:
        """Calculate maximum drawdown from P&L series"""
        if len(pnl_series) == 0:
            return 0
        
        # Convert to equity curve
        equity = np.cumsum(pnl_series) + 1000  # Start with 1000
        rolling_max = np.maximum.accumulate(equity)
        drawdown = (equity - rolling_max) / rolling_max
        
        return abs(drawdown.min()) * 100  # Return as percentage
    
    def _calculate_drawdown_metrics(self, pnl_series: np.ndarray) -> Tuple[float, int]:
        """Calculate drawdown depth and duration"""
        if len(pnl_series) == 0:
            return 0, 0
        
        # Convert to equity curve
        equity = np.cumsum(pnl_series) + 1000
        rolling_max = np.maximum.accumulate(equity)
        drawdown = (equity - rolling_max) / rolling_max
        
        # Find deepest drawdown
        deepest_idx = np.argmin(drawdown)
        deepest_value = drawdown[deepest_idx]
        
        # Find duration (how long it took to recover)
        if deepest_idx < len(pnl_series) - 1:
            # Find when equity returns to previous peak
            previous_peak = rolling_max[deepest_idx]
            recovery_idx = deepest_idx
            for i in range(deepest_idx + 1, len(equity)):
                if equity[i] >= previous_peak:
                    recovery_idx = i
                    break
            duration = recovery_idx - deepest_idx
        else:
            duration = 0
        
        return abs(deepest_value) * 100, duration
    
    def _calculate_streaks(self, pnl_series: np.ndarray) -> Tuple[int, int, int, int]:
        """Calculate consecutive wins/losses and streaks"""
        if len(pnl_series) == 0:
            return 0, 0, 0, 0
        
        # Convert to binary (1=win, 0=loss)
        wins = (pnl_series > 0).astype(int)
        
        # Find consecutive streaks
        consecutive_wins = 0
        consecutive_losses = 0
        win_streak = 0
        loss_streak = 0
        
        current_streak = 0
        streak_type = None  # 'win' or 'loss'
        
        for w in wins:
            if w == 1:
                if streak_type == 'win':
                    current_streak += 1
                else:
                    current_streak = 1
                    streak_type = 'win'
                win_streak = max(win_streak, current_streak)
                consecutive_wins = current_streak if streak_type == 'win' else consecutive_wins
            else:
                if streak_type == 'loss':
                    current_streak += 1
                else:
                    current_streak = 1
                    streak_type = 'loss'
                loss_streak = max(loss_streak, current_streak)
                consecutive_losses = current_streak if streak_type == 'loss' else consecutive_losses
        
        return consecutive_wins, consecutive_losses, win_streak, loss_streak
    
    def _default_winning_stats(self) -> Dict[str, float]:
        """Default winning stats when no trades"""
        return {
            "winning_trades": 0,
            "losing_trades": 0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "profit_factor_composite": 0.0,
            "expectancy": 0.0,
            "recovery_factor": 0.0,
            "risk_adjusted_return": 0.0,
            "drawdown_depth": 0.0,
            "drawdown_duration": 0,
            "consecutive_wins": 0,
            "consecutive_losses": 0,
            "win_streak": 0,
            "loss_streak": 0,
        }
    
    def calculate_from_trades(self, trades: List[Dict]) -> Dict[str, float]:
        """Calculate standard metrics from trade list"""
        if not trades:
            return self._default_metrics()
        
        df = pd.DataFrame(trades)
        
        # Basic metrics
        total_trades = len(df)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] < 0])
        
        # Win rate
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Profit/Loss
        total_pnl = df['pnl'].sum()
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(df[df['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
        
        # Profit factor
        gross_profit = df[df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(df[df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Sharpe ratio (simplified)
        returns = df['pnl'].pct_change().dropna()
        if len(returns) > 1:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0.0
        
        # Max drawdown
        equity_curve = df['pnl'].cumsum() + 1000  # Starting equity
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        
        # Average trade metrics
        avg_trade = df['pnl'].mean()
        avg_holding_time = df['holding_time'].mean() if 'holding_time' in df.columns else 0
        
        # Equity curve
        final_equity = 1000 + total_pnl
        
        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_pnl": total_pnl,
            "final_equity": final_equity,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_trade": avg_trade,
            "avg_holding_time": avg_holding_time,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
        }
    
    def _default_metrics(self) -> Dict[str, float]:
        """Default metrics when no trades"""
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "total_pnl": 0.0,
            "final_equity": 1000.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "avg_trade": 0.0,
            "avg_holding_time": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "winning_trades": 0,
            "losing_trades": 0,
        }


class HistoricalTrainingSession:
    """Main historical training session with real-time metrics visualization"""
    
    def __init__(self, config: HistoricalTrainingConfig):
        self.config = config
        self.prevailing_agent = PrevailingAgentMetrics()
        self.trader_metrics = TraderMetrics()
        self.trades = []
        self.current_epoch = 0
        self.start_time = time.time()
        
        # Output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[HistoricalTraining] Session initialized")
        print(f"[HistoricalTraining] Symbol: {self.config.symbol}")
        print(f"[HistoricalTraining] Timeframe: {self.config.timeframe}")
        print(f"[HistoricalTraining] Duration: {self.config.session_duration_hours}h")
        print(f"[HistoricalTraining] Epochs: {self.config.epochs}")
        
    def load_historical_data(self) -> pd.DataFrame:
        """Load historical data from DuckDB or arrow files"""
        print(f"\n[HistoricalTraining] Loading historical data...")
        
        if self.config.data_source == "duckdb":
            try:
                from hrm.duck_store import DuckStore
                duck_store = DuckStore(self.config.data_path)
                
                # Query data for the symbol and timeframe
                symbol_filter = self.config.symbol.replace("-", "_").replace("_", "")
                query = f"""
                    SELECT * FROM market_data 
                    WHERE symbol = '{symbol_filter}' 
                    AND timestamp >= '{self.config.start_date.isoformat()}'
                    AND timestamp <= '{self.config.end_date.isoformat()}'
                    ORDER BY timestamp
                """
                
                df = duck_store.conn.execute(query).fetchdf()
                print(f"[HistoricalTraining] Loaded {len(df)} rows from DuckDB")
                return df
                
            except Exception as e:
                print(f"[HistoricalTraining] Failed to load from DuckDB: {e}")
        
        # Fallback to arrow files
        arrow_file = Path("hrm/data/arrow") / f"{self.config.symbol.replace('-', '_')}.feather"
        if arrow_file.exists():
            df = pd.read_feather(arrow_file)
            print(f"[HistoricalTraining] Loaded {len(df)} rows from arrow file")
            return df
        
        # Generate synthetic data as last resort
        print(f"[HistoricalTraining] No data found, generating synthetic data")
        return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic data for testing"""
        dates = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq=self.config.timeframe
        )
        
        # Generate price series with trends
        n = len(dates)
        base_price = 50000
        trend = np.linspace(1, 1.2, n)  # 20% upward trend
        noise = np.random.normal(0, 0.02, n)  # 2% daily noise
        
        prices = base_price * trend * (1 + noise)
        
        # Create OHLCV
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * (1 + np.random.uniform(0.001, 0.01, n)),
            'low': prices * (1 - np.random.uniform(0.001, 0.01, n)),
            'close': prices * (1 + np.random.uniform(-0.005, 0.005, n)),
            'volume': np.random.uniform(100, 10000, n),
        })
        
        # Add 48-column schema
        df = self._add_48_column_schema(df)
        
        print(f"[HistoricalTraining] Generated {len(df)} synthetic rows")
        return df
    
    def _add_48_column_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 48-column schema to DataFrame"""
        if df.empty:
            return df
        
        # Basic OHLCV is already there
        # Add technical indicators
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_15'] = df['close'].rolling(15).mean()
        df['sma_60'] = df['close'].rolling(60).mean()
        df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema_15'] = df['close'].ewm(span=15, adjust=False).mean()
        df['ema_60'] = df['close'].ewm(span=60, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Add other columns as placeholders
        df['returns_1h'] = df['close'].pct_change(1)
        df['regime_label'] = 1  # Flat
        df['stochastic_compass'] = 0.0
        df['hrm_reward'] = 0.0
        df['veto_flag'] = False
        df['position_size_usd'] = 0.0
        df['equity_curve'] = 1000.0
        
        return df
    
    def train_epoch(self, epoch: int, data: pd.DataFrame) -> Dict[str, Any]:
        """Train one epoch and return metrics"""
        if data.empty:
            return {}
        
        # Simulate training process
        # In real implementation, this would call actual HRM training
        
        # Generate synthetic metrics for demonstration
        base_loss = 0.5
        loss_improvement = max(0.0, base_loss - (epoch * 0.005))
        accuracy = 50 + (epoch * 0.5)  # Gradually improves
        if accuracy > 95: accuracy = 95
        
        # Simulate trading decisions
        if epoch > 10:  # Start trading after some training
            # Generate synthetic trades
            n_trades = np.random.poisson(3)  # Average 3 trades per epoch
            for i in range(n_trades):
                trade = {
                    'timestamp': datetime.now(),
                    'entry_price': np.random.uniform(49000, 51000),
                    'exit_price': np.random.uniform(49000, 51000),
                    'direction': np.random.choice(['long', 'short']),
                    'size': np.random.uniform(100, 1000),
                    'pnl': np.random.normal(50, 100),  # Normal distribution with mean 50
                    'holding_time': np.random.uniform(0.1, 2.0),  # hours
                }
                trade['exit_price'] = trade['entry_price'] * (1 + trade['pnl'] / (trade['size'] * 100))
                self.trades.append(trade)
        
        # Calculate trader metrics from trades
        trader_metrics = self.trader_metrics.calculate_from_trades(self.trades)
        
        # Calculate winning agent specific stats
        winning_agent_stats = self.trader_metrics._calculate_winning_agent_stats(self.trades)
        
        # Calculate HRM-specific metrics
        hrm_reward = np.random.uniform(0.1, 0.9) if epoch > 20 else epoch * 0.02
        veto_rate = max(0, 0.3 - epoch * 0.002)  # Decreases with training
        regime_confidence = min(0.95, 0.5 + epoch * 0.004)  # Increases
        
        # Update prevailing agent metrics with winning agent stats
        metrics_kwargs = {
            "loss": loss_improvement,
            "accuracy": accuracy,
            "profit_factor": trader_metrics.get('profit_factor', 0),
            "sharpe_ratio": trader_metrics.get('sharpe_ratio', 0),
            "win_rate": trader_metrics.get('win_rate', 0),
            "max_drawdown": trader_metrics.get('max_drawdown', 0),
            "equity": trader_metrics.get('final_equity', 1000),
            "position_size": np.random.uniform(0, 500),
            "trade_count": trader_metrics.get('total_trades', 0),
            "total_pnl": trader_metrics.get('total_pnl', 0),
            "hrm_reward": hrm_reward,
            "veto_rate": veto_rate,
            "regime_confidence": regime_confidence,
            # Add winning agent stats
            "winning_trades": winning_agent_stats.get('winning_trades', 0),
            "losing_trades": winning_agent_stats.get('losing_trades', 0),
            "avg_win": winning_agent_stats.get('avg_win', 0),
            "avg_loss": winning_agent_stats.get('avg_loss', 0),
            "largest_win": winning_agent_stats.get('largest_win', 0),
            "largest_loss": winning_agent_stats.get('largest_loss', 0),
            "profit_factor_composite": winning_agent_stats.get('profit_factor_composite', 0),
            "expectancy": winning_agent_stats.get('expectancy', 0),
            "recovery_factor": winning_agent_stats.get('recovery_factor', 0),
            "risk_adjusted_return": winning_agent_stats.get('risk_adjusted_return', 0),
            "drawdown_depth": winning_agent_stats.get('drawdown_depth', 0),
            "drawdown_duration": winning_agent_stats.get('drawdown_duration', 0),
            "consecutive_wins": winning_agent_stats.get('consecutive_wins', 0),
            "consecutive_losses": winning_agent_stats.get('consecutive_losses', 0),
            "win_streak": winning_agent_stats.get('win_streak', 0),
            "loss_streak": winning_agent_stats.get('loss_streak', 0),
        }
        
        self.prevailing_agent.update(epoch, **metrics_kwargs)
        
        return {
            "epoch": epoch,
            "hrm_metrics": metrics_kwargs,
            "trader_metrics": trader_metrics,
            "trade_count": len(self.trades),
        }
    
    def display_realtime_metrics(self, epoch: int, metrics: Dict[str, Any]):
        """Display real-time metrics in terminal"""
        hrm_metrics = metrics.get('hrm_metrics', {})
        trader_metrics = metrics.get('trader_metrics', {})
        
        # Clear screen (Unix/Linux/Mac)
        if os.name == 'posix':
            os.system('clear')
        
        print("="*100)
        print(f"HISTORICAL TRAINING SESSION - HRM LEARNING PROGRESS")
        print("="*100)
        print(f"Epoch: {epoch}/{self.config.epochs} | Time: {time.time() - self.start_time:.1f}s")
        print(f"Symbol: {self.config.symbol} | Timeframe: {self.config.timeframe}")
        print(f"Training Period: {self.config.start_date.date()} to {self.config.end_date.date()}")
        print("-"*100)
        
        # HRM Learning Metrics
        print("HRM LEARNING METRICS:")
        print(f"  Loss:           {hrm_metrics.get('loss', 0):.4f}")
        print(f"  Accuracy:       {hrm_metrics.get('accuracy', 0):.1f}%")
        print(f"  HRM Reward:     {hrm_metrics.get('hrm_reward', 0):.3f}")
        print(f"  Veto Rate:      {hrm_metrics.get('veto_rate', 0):.3f}")
        print(f"  Regime Conf:    {hrm_metrics.get('regime_confidence', 0):.3f}")
        print()
        
        # Winning Agent Statistics
        print("WINNING AGENT STATISTICS:")
        print(f"  Wins/Losses:    {hrm_metrics.get('winning_trades', 0)}/{hrm_metrics.get('losing_trades', 0)}")
        print(f"  Avg Win:        ${hrm_metrics.get('avg_win', 0):.2f}")
        print(f"  Avg Loss:       ${hrm_metrics.get('avg_loss', 0):.2f}")
        print(f"  Largest Win:    ${hrm_metrics.get('largest_win', 0):.2f}")
        print(f"  Largest Loss:   ${hrm_metrics.get('largest_loss', 0):.2f}")
        print(f"  Exp. Value:     ${hrm_metrics.get('expectancy', 0):.2f}")
        print(f"  Recov. Factor:  {hrm_metrics.get('recovery_factor', 0):.2f}")
        print(f"  Risk Adj. Ret:  {hrm_metrics.get('risk_adjusted_return', 0):.2f}")
        print()
        
        # Trader Performance Metrics
        print("TRADER PERFORMANCE METRICS:")
        print(f"  Total Trades:   {trader_metrics.get('total_trades', 0)}")
        print(f"  Win Rate:       {trader_metrics.get('win_rate', 0):.1f}%")
        print(f"  Profit Factor:  {trader_metrics.get('profit_factor', 0):.2f}")
        print(f"  Sharpe Ratio:   {trader_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown:   {trader_metrics.get('max_drawdown', 0):.2%}")
        print(f"  Total P&L:      ${trader_metrics.get('total_pnl', 0):.2f}")
        print(f"  Final Equity:   ${trader_metrics.get('final_equity', 1000):.2f}")
        print()
        
        # Drawdown & Streak Metrics
        print("DRAWDOWN & STREAK METRICS:")
        print(f"  DD Depth:       {hrm_metrics.get('drawdown_depth', 0):.2f}%")
        print(f"  DD Duration:    {hrm_metrics.get('drawdown_duration', 0)} bars")
        print(f"  Curr Wins:      {hrm_metrics.get('consecutive_wins', 0)}")
        print(f"  Curr Losses:    {hrm_metrics.get('consecutive_losses', 0)}")
        print(f"  Win Streak:     {hrm_metrics.get('win_streak', 0)}")
        print(f"  Loss Streak:    {hrm_metrics.get('loss_streak', 0)}")
        print()
        
        # Trade Details
        print(f"TRADES THIS SESSION: {len(self.trades)}")
        if self.trades:
            latest_trades = self.trades[-5:]  # Show last 5 trades
            for i, trade in enumerate(latest_trades, 1):
                pnl_color = "\033[92m" if trade['pnl'] > 0 else "\033[91m"
                reset_color = "\033[0m"
                print(f"  Trade {i}: {trade['direction']:5s} | "
                      f"PNL: {pnl_color}${trade['pnl']:.2f}{reset_color} | "
                      f"Hold: {trade['holding_time']:.2f}h")
        
        # Progress bar
        progress = (epoch / self.config.epochs) * 100
        bar_length = 50
        filled_length = int(bar_length * progress / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"\nPROGRESS: [{bar}] {progress:.1f}%")
        
        # Status messages
        if epoch == 0:
            print("\n[STATUS] Initializing training session...")
        elif epoch < 10:
            print("\n[STATUS] Warming up - collecting initial data...")
        elif epoch < 50:
            print("\n[STATUS] Learning phase - optimizing strategy...")
        elif epoch < 90:
            print("\n[STATUS] Refinement phase - fine-tuning parameters...")
        else:
            print("\n[STATUS] Convergence phase - final adjustments...")
    
    def generate_summary_report(self):
        """Generate final summary report"""
        history_df = self.prevailing_agent.get_history()
        
        if history_df.empty:
            return
        
        # Create output directory
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.output_dir / f"session_{session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics to JSON
        if self.config.save_metrics:
            metrics_file = session_dir / "training_metrics.json"
            metrics_data = {
                "config": self.config.__dict__,
                "final_metrics": self.prevailing_agent.get_latest(),
                "history": history_df.to_dict('records'),
                "trades": self.trades,
                "session_duration": time.time() - self.start_time,
            }
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            print(f"\n[REPORT] Metrics saved to: {metrics_file}")
        
        # Generate plots
        if self.config.save_plots:
            self._generate_plots(history_df, session_dir)
        
        # Print final summary
        print("\n" + "="*100)
        print("FINAL SUMMARY REPORT")
        print("="*100)
        
        latest = self.prevailing_agent.get_latest()
        trader_final = self.trader_metrics.calculate_from_trades(self.trades)
        
        print("\nHRM LEARNING SUMMARY:")
        print(f"  Final Loss:       {latest['loss']:.4f}")
        print(f"  Final Accuracy:   {latest['accuracy']:.1f}%")
        print(f"  Final HRM Reward: {latest['hrm_reward']:.3f}")
        print(f"  Final Veto Rate:  {latest['veto_rate']:.3f}")
        
        print("\nWINNING AGENT FINAL STATISTICS:")
        print(f"  Wins/Losses:      {latest['winning_trades']}/{latest['losing_trades']}")
        print(f"  Avg Win:          ${latest['avg_win']:.2f}")
        print(f"  Avg Loss:         ${latest['avg_loss']:.2f}")
        print(f"  Largest Win:      ${latest['largest_win']:.2f}")
        print(f"  Largest Loss:     ${latest['largest_loss']:.2f}")
        print(f"  Exp. Value:       ${latest['expectancy']:.2f}")
        print(f"  Recov. Factor:    {latest['recovery_factor']:.2f}")
        print(f"  Risk Adj. Ret:    {latest['risk_adjusted_return']:.2f}")
        
        print("\nDRAWDOWN STATISTICS:")
        print(f"  DD Depth:         {latest['drawdown_depth']:.2f}%")
        print(f"  DD Duration:      {latest['drawdown_duration']} bars")
        print(f"  Win Streak:       {latest['win_streak']}")
        print(f"  Loss Streak:      {latest['loss_streak']}")
        
        print("\nTRADER PERFORMANCE SUMMARY:")
        print(f"  Total Trades:     {trader_final['total_trades']}")
        print(f"  Win Rate:         {trader_final['win_rate']:.1f}%")
        print(f"  Profit Factor:    {trader_final['profit_factor']:.2f}")
        print(f"  Sharpe Ratio:     {trader_final['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:     {trader_final['max_drawdown']:.2%}")
        print(f"  Total P&L:        ${trader_final['total_pnl']:.2f}")
        print(f"  Final Equity:     ${trader_final['final_equity']:.2f}")
        
        print("\nSESSION STATISTICS:")
        print(f"  Duration:        {time.time() - self.start_time:.1f} seconds")
        print(f"  Epochs:          {self.config.epochs}")
        print(f"  Trade Count:     {len(self.trades)}")
        print(f"  Avg P&L/Trade:   ${trader_final['avg_trade']:.2f}")
        
        print(f"\n[REPORT] Session ID: {session_id}")
        print(f"[REPORT] Files saved in: {session_dir}")
        
        # Save session ID to file
        with open("latest_training_session.txt", "w") as f:
            f.write(session_id)
        
        return session_dir
    
    def _generate_plots(self, history_df: pd.DataFrame, session_dir: Path):
        """Generate visualization plots"""
        print(f"\n[REPORT] Generating plots...")
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'HRM Training Session - {datetime.now().strftime("%Y-%m-%d %H:%M")}', fontsize=16)
        
        # 1. Loss and Accuracy
        axes[0, 0].plot(history_df['epoch'], history_df['loss'], 'b-', linewidth=2, label='Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss', color='b')
        axes[0, 0].tick_params(axis='y', labelcolor='b')
        axes[0, 0].set_title('HRM Learning Loss')
        
        ax2 = axes[0, 0].twinx()
        ax2.plot(history_df['epoch'], history_df['accuracy'], 'r-', linewidth=2, label='Accuracy')
        ax2.set_ylabel('Accuracy (%)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # 2. Profit Factor and Sharpe Ratio
        axes[0, 1].plot(history_df['epoch'], history_df['profit_factor'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Profit Factor', color='g')
        axes[0, 1].tick_params(axis='y', labelcolor='g')
        axes[0, 1].set_title('Trading Metrics')
        
        ax2 = axes[0, 1].twinx()
        ax2.plot(history_df['epoch'], history_df['sharpe_ratio'], 'm-', linewidth=2)
        ax2.set_ylabel('Sharpe Ratio', color='m')
        ax2.tick_params(axis='y', labelcolor='m')
        
        # 3. Win Rate and Max Drawdown
        axes[1, 0].plot(history_df['epoch'], history_df['win_rate'], 'c-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Win Rate (%)', color='c')
        axes[1, 0].tick_params(axis='y', labelcolor='c')
        axes[1, 0].set_title('Win Rate & Drawdown')
        
        ax2 = axes[1, 0].twinx()
        ax2.plot(history_df['epoch'], history_df['max_drawdown'], 'r-', linewidth=2)
        ax2.set_ylabel('Max Drawdown (%)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # 4. Equity Curve
        axes[1, 1].plot(history_df['epoch'], history_df['equity'], 'b-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Equity ($)', color='b')
        axes[1, 1].tick_params(axis='y', labelcolor='b')
        axes[1, 1].set_title('Equity Curve')
        
        # 5. HRM Reward and Veto Rate
        axes[2, 0].plot(history_df['epoch'], history_df['hrm_reward'], 'g-', linewidth=2)
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('HRM Reward', color='g')
        axes[2, 0].tick_params(axis='y', labelcolor='g')
        axes[2, 0].set_title('HRM Internal Metrics')
        
        ax2 = axes[2, 0].twinx()
        ax2.plot(history_df['epoch'], history_df['veto_rate'], 'r-', linewidth=2)
        ax2.set_ylabel('Veto Rate', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # 6. Position Size and Trade Count
        axes[2, 1].plot(history_df['epoch'], history_df['position_size'], 'm-', linewidth=2)
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Position Size ($)', color='m')
        axes[2, 1].tick_params(axis='y', labelcolor='m')
        axes[2, 1].set_title('Position Management')
        
        ax2 = axes[2, 1].twinx()
        ax2.plot(history_df['epoch'], history_df['trade_count'], 'c-', linewidth=2)
        ax2.set_ylabel('Trade Count', color='c')
        ax2.tick_params(axis='y', labelcolor='c')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = session_dir / "training_progress.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[REPORT] Plot saved to: {plot_file}")
        
        # Create equity curve plot separately
        self._plot_equity_curve(history_df, session_dir)
    
    def _plot_equity_curve(self, history_df: pd.DataFrame, session_dir: Path):
        """Create dedicated equity curve plot"""
        plt.figure(figsize=(12, 6))
        
        # Main equity curve
        plt.plot(history_df['epoch'], history_df['equity'], 'b-', linewidth=2, label='Equity')
        
        # Add drawdown shading
        if 'max_drawdown' in history_df.columns:
            drawdown = history_df['max_drawdown']
            plt.fill_between(history_df['epoch'], 
                           history_df['equity'] * (1 - abs(drawdown)/100),
                           history_df['equity'],
                           alpha=0.2, color='red', label='Drawdown')
        
        plt.xlabel('Epoch')
        plt.ylabel('Equity ($)')
        plt.title('24-Hour HRM Training - Equity Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add annotations for key metrics
        latest = self.prevailing_agent.get_latest()
        textstr = '\n'.join([
            f"Final Equity: ${latest['equity']:.2f}",
            f"Total P&L: ${latest['total_pnl']:.2f}",
            f"Win Rate: {latest['win_rate']:.1f}%",
            f"Profit Factor: {latest['profit_factor']:.2f}",
        ])
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save
        plot_file = session_dir / "equity_curve.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[REPORT] Equity curve saved to: {plot_file}")
    
    def run(self):
        """Run the historical training session"""
        print(f"\n[HistoricalTraining] Starting session...")
        
        # Load data
        data = self.load_historical_data()
        
        if data.empty:
            print("[ERROR] No data available for training")
            return
        
        # Training loop
        print(f"\n[HistoricalTraining] Starting training loop...")
        print(f"[HistoricalTraining] Will display metrics every {self.config.update_frequency_seconds} seconds")
        
        last_update = time.time()
        
        for epoch in range(self.config.epochs + 1):
            self.current_epoch = epoch
            
            # Train one epoch
            metrics = self.train_epoch(epoch, data)
            
            # Display real-time metrics periodically
            current_time = time.time()
            if (current_time - last_update >= self.config.update_frequency_seconds) or epoch == self.config.epochs:
                self.display_realtime_metrics(epoch, metrics)
                last_update = current_time
            
            # Small delay to make visualization observable
            time.sleep(0.1)
        
        # Generate final report
        print(f"\n[HistoricalTraining] Training complete!")
        session_dir = self.generate_summary_report()
        
        return session_dir


async def main():
    """Run historical training session"""
    print("HISTORICAL TRAINING SESSION - HRM LEARNING & METRICS")
    print("="*80)
    
    # Create configuration
    config = HistoricalTrainingConfig(
        symbol="BTC-USD",
        timeframe="1h",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 8),
        epochs=50,  # Reduced for demonstration
        session_duration_hours=24,
        update_frequency_seconds=2,  # Update every 2 seconds
        data_source="synthetic",  # Use synthetic data for demo
    )
    
    # Create and run session
    session = HistoricalTrainingSession(config)
    session_dir = session.run()
    
    if session_dir:
        print(f"\n✅ Training session completed successfully!")
        print(f"📁 Results saved to: {session_dir}")
        print(f"📊 Open the PNG files to view training progress")
    else:
        print(f"\n❌ Training session failed")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())