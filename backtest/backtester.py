"""
Backtester for HRM strategies.
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime


@dataclass
class BacktestConfig:
    """Backtest configuration"""
    initial_capital: float = 10000.0
    commission: float = 0.001  # 0.1% commission
    slippage: float = 0.001    # 0.1% slippage
    lookback_period: int = 20
    risk_per_trade: float = 0.01  # 1% risk per trade


@dataclass
class Trade:
    """Trade record"""
    timestamp: datetime
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    size: float
    price: float
    pnl: float = 0.0
    reason: str = ""


@dataclass
class BacktestResult:
    """Backtest results"""
    trades: List[Trade]
    equity_curve: np.ndarray
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_pnl: float
    trade_count: int


class Backtester:
    """
    Backtester for trading strategies.
    
    Responsibilities:
    - Simulate trading with historical data
    - Track positions and P&L
    - Compute performance metrics
    - Generate reports
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.position: float = 0.0
        self.cash: float = config.initial_capital if config else 10000.0
        self.initial_capital = self.cash
        
    def run_backtest(self,
                    prices: np.ndarray,
                    signals: np.ndarray,
                    symbols: List[str],
                    timestamps: List[datetime]) -> BacktestResult:
        """
        Run backtest with given signals.
        
        Args:
            prices: Price data [time, symbols]
            signals: Signal data [time, symbols]
            symbols: Symbol names
            timestamps: Timestamps
            
        Returns:
            BacktestResult
        """
        self.trades.clear()
        self.equity_curve.clear()
        self.position = 0.0
        self.cash = self.initial_capital
        
        n_time = len(timestamps)
        n_symbols = len(symbols)
        
        # Check data shapes
        if prices.shape != (n_time, n_symbols):
            raise ValueError(f"Prices shape mismatch: expected ({n_time}, {n_symbols}), got {prices.shape}")
        if signals.shape != (n_time, n_symbols):
            raise ValueError(f"Signals shape mismatch: expected ({n_time}, {n_symbols}), got {signals.shape}")
        
        # Run simulation
        for t in range(1, n_time):
            timestamp = timestamps[t]
            current_prices = prices[t]
            previous_prices = prices[t - 1]
            current_signals = signals[t]
            
            # Update equity curve
            equity = self._compute_equity(current_prices)
            self.equity_curve.append(equity)
            
            # Process each symbol
            for i, symbol in enumerate(symbols):
                signal = current_signals[i]
                current_price = current_prices[i]
                previous_price = previous_prices[i]
                
                # Check if we should trade
                action, size, reason = self._determine_trade(
                    symbol, signal, current_price, previous_price, equity
                )
                
                if action != 'hold':
                    # Execute trade
                    trade = Trade(
                        timestamp=timestamp,
                        symbol=symbol,
                        action=action,
                        size=size,
                        price=current_price,
                        reason=reason
                    )
                    self.trades.append(trade)
                    
                    # Update position and cash
                    if action == 'buy':
                        cost = size * current_price * (1 + self.config.commission + self.config.slippage)
                        self.cash -= cost
                        self.position += size
                    elif action == 'sell':
                        revenue = size * current_price * (1 - self.config.commission - self.config.slippage)
                        self.cash += revenue
                        self.position -= size
        
        # Compute final metrics
        return self.compute_metrics()
    
    def _determine_trade(self,
                        symbol: str,
                        signal: float,
                        current_price: float,
                        previous_price: float,
                        equity: float) -> Tuple[str, float, str]:
        """Determine trade action"""
        if abs(signal) < 0.3:  # Minimum threshold
            return 'hold', 0.0, 'signal_too_weak'
        
        # Calculate position size based on signal strength
        position_size = self.config.risk_per_trade * equity / current_price
        position_size *= abs(signal)  # Scale by signal strength
        
        if position_size == 0:
            return 'hold', 0.0, 'zero_position'
        
        # Determine action
        if signal > 0 and self.position <= 0:
            return 'buy', position_size, 'positive_signal'
        elif signal < 0 and self.position >= 0:
            return 'sell', position_size, 'negative_signal'
        elif signal > 0 and self.position > 0:
            # Already in position, maybe add
            if signal > 0.7:  # Strong signal
                return 'buy', position_size * 0.5, 'strong_buy_signal'
            else:
                return 'hold', 0.0, 'already_long'
        elif signal < 0 and self.position < 0:
            # Already in short position
            if signal < -0.7:  # Strong signal
                return 'sell', position_size * 0.5, 'strong_sell_signal'
            else:
                return 'hold', 0.0, 'already_short'
        
        return 'hold', 0.0, 'default'
    
    def _compute_equity(self, current_prices: np.ndarray) -> float:
        """Compute current equity"""
        position_value = self.position * np.mean(current_prices) if len(current_prices) > 0 else 0.0
        return self.cash + position_value
    
    def compute_metrics(self) -> BacktestResult:
        """Compute performance metrics"""
        if len(self.trades) == 0:
            return BacktestResult(
                trades=self.trades,
                equity_curve=np.array(self.equity_curve),
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.5,
                total_pnl=0.0,
                trade_count=0
            )
        
        # Extract P&L from trades
        pnls = []
        for trade in self.trades:
            if trade.action == 'buy':
                # Look for corresponding sell
                for later_trade in self.trades:
                    if (later_trade.timestamp > trade.timestamp and 
                        later_trade.symbol == trade.symbol and 
                        later_trade.action == 'sell'):
                        pnl = (later_trade.price - trade.price) * trade.size
                        pnls.append(pnl)
                        break
        
        if not pnls:
            # Use equity curve for P&L
            equity_array = np.array(self.equity_curve)
            if len(equity_array) > 1:
                returns = np.diff(equity_array) / equity_array[:-1]
                total_pnl = np.sum(returns) * self.initial_capital
            else:
                total_pnl = 0.0
            win_rate = 0.5
        else:
            pnls = np.array(pnls)
            total_pnl = np.sum(pnls)
            win_rate = np.mean(pnls > 0) if len(pnls) > 0 else 0.5
        
        # Total return
        final_equity = self.cash + self.position * np.mean(np.array(self.equity_curve)[-1]) if len(self.equity_curve) > 0 else self.cash
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        # Sharpe ratio
        equity_array = np.array(self.equity_curve)
        if len(equity_array) > 1:
            returns = np.diff(equity_array) / equity_array[:-1]
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        # Max drawdown
        if len(equity_array) > 0:
            running_max = np.maximum.accumulate(equity_array)
            drawdown = (equity_array - running_max) / running_max
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
        else:
            max_drawdown = 0.0
        
        return BacktestResult(
            trades=self.trades,
            equity_curve=equity_array,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_pnl=total_pnl,
            trade_count=len(self.trades)
        )
    
    def generate_report(self, result: BacktestResult) -> Dict[str, Any]:
        """Generate performance report"""
        return {
            'total_return': f"{result.total_return:.2%}",
            'sharpe_ratio': f"{result.sharpe_ratio:.2f}",
            'max_drawdown': f"{result.max_drawdown:.2%}",
            'win_rate': f"{result.win_rate:.2%}",
            'total_pnl': f"${result.total_pnl:,.2f}",
            'trade_count': result.trade_count,
            'initial_capital': f"${self.initial_capital:,.2f}",
            'final_capital': f"${self.cash + self.position * np.mean(np.array(self.equity_curve)[-1]) if len(self.equity_curve) > 0 else self.cash:,.2f}"
        }