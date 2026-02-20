"""
Risk management module - position sizing, risk limits, drawdown control.

Pure logic, no framework dependencies.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime


@dataclass
class RiskConfig:
    """Risk management configuration"""
    initial_capital: float = 10000.0  # Initial portfolio value
    max_position_size: float = 0.1  # Max 10% per position
    max_total_exposure: float = 0.8  # Max 80% total exposure
    max_drawdown_limit: float = -0.2  # -20% max drawdown
    risk_per_trade: float = 0.01  # 1% risk per trade
    stop_loss_multiplier: float = 2.0  # 2x ATR stop loss
    take_profit_multiplier: float = 3.0  # 3x ATR take profit
    volatility_adjustment: bool = True
    correlation_penalty: float = 0.5  # Penalty for correlated positions


@dataclass
class Position:
    """Single position"""
    symbol: str
    size: float  # Number of units
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    timestamp: datetime
    risk_score: float = 0.0


class RiskManager:
    """
    Manages risk across positions and portfolio.
    
    Responsibilities:
    - Position sizing
    - Stop loss/take profit calculation
    - Correlation analysis
    - Drawdown monitoring
    - Risk limits enforcement
    """
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.portfolio_value: float = config.initial_capital
        self.initial_portfolio_value: float = config.initial_capital
        self.history: List[Dict] = []
        
    def calculate_position_size(self, 
                               symbol: str,
                               signal_strength: float,
                               volatility: float,
                               confidence: float = 1.0) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            symbol: Asset symbol
            signal_strength: Signal strength [-1, 1]
            volatility: Current volatility
            confidence: Model confidence [0, 1]
            
        Returns:
            Position size in units
        """
        if abs(signal_strength) < 0.3:  # Too weak
            return 0.0
        
        # Base risk per trade (1% of portfolio)
        risk_amount = self.portfolio_value * self.config.risk_per_trade
        
        # Adjust for volatility
        if self.config.volatility_adjustment:
            if volatility > 0:
                volatility_factor = max(0.1, min(2.0, 0.05 / volatility))
                risk_amount *= volatility_factor
        
        # Adjust for confidence
        risk_amount *= confidence
        
        # Adjust for signal strength
        risk_amount *= abs(signal_strength)
        
        # Calculate position size based on stop distance
        # Assuming stop loss distance is proportional to volatility
        stop_distance = volatility * self.config.stop_loss_multiplier
        
        if stop_distance <= 0:
            return 0.0
        
        position_size = risk_amount / stop_distance
        
        # Apply max position size limit
        current_price = self.get_current_price(symbol)
        if current_price > 0:
            max_size = self.config.max_position_size * self.portfolio_value / current_price
            position_size = min(position_size, max_size)
        
        return position_size
    
    def calculate_stop_loss_take_profit(self,
                                       entry_price: float,
                                       volatility: float,
                                       signal_direction: float) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels.
        
        Args:
            entry_price: Entry price
            volatility: Current volatility
            signal_direction: Signal direction (1 = long, -1 = short)
            
        Returns:
            (stop_loss, take_profit)
        """
        stop_distance = volatility * self.config.stop_loss_multiplier
        profit_distance = volatility * self.config.take_profit_multiplier
        
        if signal_direction > 0:  # Long
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + profit_distance
        else:  # Short
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - profit_distance
            
        return stop_loss, take_profit
    
    def can_add_position(self, symbol: str, size: float, price: float) -> bool:
        """
        Check if we can add a position given current exposure.
        
        Args:
            symbol: Asset symbol
            size: Position size
            price: Current price
            
        Returns:
            True if position can be added
        """
        # Check total exposure
        total_exposure = self.get_total_exposure()
        if total_exposure + (size * price) / self.portfolio_value > self.config.max_total_exposure:
            return False
        
        # Check if we already have a position in this symbol
        if symbol in self.positions:
            # Could implement adding to existing position logic
            return False
        
        return True
    
    def update_position(self, 
                       symbol: str, 
                       current_price: float,
                       timestamp: datetime) -> Dict[str, float]:
        """
        Update position with current price and check stop/take profit.
        
        Args:
            symbol: Asset symbol
            current_price: Current price
            timestamp: Current timestamp
            
        Returns:
            Position update info
        """
        if symbol not in self.positions:
            return {'action': 'none', 'pnl': 0.0}
        
        position = self.positions[symbol]
        old_pnl = position.current_price - position.entry_price
        
        # Update current price
        position.current_price = current_price
        
        # Check stop loss
        if position.size > 0:  # Long position
            if current_price <= position.stop_loss:
                self.close_position(symbol, 'stop_loss', timestamp)
                return {'action': 'close', 'reason': 'stop_loss', 'pnl': position.current_price - position.entry_price}
        
        # Check take profit
            elif current_price >= position.take_profit:
                self.close_position(symbol, 'take_profit', timestamp)
                return {'action': 'close', 'reason': 'take_profit', 'pnl': position.current_price - position.entry_price}
        else:  # Short position
            if current_price >= position.stop_loss:
                self.close_position(symbol, 'stop_loss', timestamp)
                return {'action': 'close', 'reason': 'stop_loss', 'pnl': position.entry_price - position.current_price}
            elif current_price <= position.take_profit:
                self.close_position(symbol, 'take_profit', timestamp)
                return {'action': 'close', 'reason': 'take_profit', 'pnl': position.entry_price - position.current_price}
        
        # No action
        new_pnl = position.current_price - position.entry_price
        return {'action': 'hold', 'pnl': new_pnl - old_pnl}
    
    def open_position(self,
                     symbol: str,
                     size: float,
                     price: float,
                     volatility: float,
                     signal_direction: float,
                     timestamp: datetime) -> bool:
        """
        Open a new position.
        
        Args:
            symbol: Asset symbol
            size: Position size
            price: Entry price
            volatility: Current volatility
            signal_direction: Signal direction (1 = long, -1 = short)
            timestamp: Current timestamp
            
        Returns:
            True if position was opened
        """
        if size <= 0:
            return False
            
        if not self.can_add_position(symbol, size, price):
            return False
        
        # Calculate stop loss and take profit
        stop_loss, take_profit = self.calculate_stop_loss_take_profit(
            price, volatility, signal_direction
        )
        
        # Create position
        position = Position(
            symbol=symbol,
            size=size * signal_direction,  # Positive for long, negative for short
            entry_price=price,
            current_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=timestamp
        )
        
        self.positions[symbol] = position
        
        # Record in history
        self.history.append({
            'timestamp': timestamp,
            'action': 'open',
            'symbol': symbol,
            'size': size,
            'price': price,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        })
        
        return True
    
    def close_position(self, symbol: str, reason: str, timestamp: datetime) -> float:
        """
        Close a position.
        
        Args:
            symbol: Asset symbol
            reason: Reason for closing
            timestamp: Current timestamp
            
        Returns:
            P&L
        """
        if symbol not in self.positions:
            return 0.0
        
        position = self.positions[symbol]
        pnl = position.current_price - position.entry_price
        
        # Record in history
        self.history.append({
            'timestamp': timestamp,
            'action': 'close',
            'symbol': symbol,
            'reason': reason,
            'pnl': pnl,
            'size': position.size,
            'price': position.current_price
        })
        
        # Update portfolio value
        self.portfolio_value += pnl * position.size
        
        # Remove position
        del self.positions[symbol]
        
        return pnl
    
    def get_total_exposure(self) -> float:
        """Get total exposure as fraction of portfolio value"""
        if self.portfolio_value == 0:
            return 0.0
        
        total_exposure = 0.0
        for position in self.positions.values():
            total_exposure += abs(position.size * position.current_price)
        
        return total_exposure / self.portfolio_value
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        if symbol in self.positions:
            return self.positions[symbol].current_price
        return 100.0  # Default price
    
    def get_drawdown(self) -> float:
        """Get current drawdown (negative value for drawdown)"""
        if self.initial_portfolio_value == 0:
            return 0.0
        # Drawdown is how much we've lost from peak
        # For simplicity, we compare current value to initial
        return (self.portfolio_value - self.initial_portfolio_value) / self.initial_portfolio_value
    
    def check_risk_limits(self) -> Tuple[bool, List[str]]:
        """
        Check if risk limits are violated.
        
        Returns:
            (is_safe, violations)
        """
        violations = []
        
        # Check drawdown
        drawdown = self.get_drawdown()
        if drawdown < self.config.max_drawdown_limit:
            violations.append(f"Max drawdown exceeded: {drawdown:.2%}")
        
        # Check total exposure
        exposure = self.get_total_exposure()
        if exposure > self.config.max_total_exposure:
            violations.append(f"Max exposure exceeded: {exposure:.2%}")
        
        # Check position concentration
        if len(self.positions) > 0:
            max_position = max(abs(p.size * p.current_price) for p in self.positions.values())
            concentration = max_position / self.portfolio_value
            if concentration > self.config.max_position_size:
                violations.append(f"Position concentration too high: {concentration:.2%}")
        
        is_safe = len(violations) == 0
        return is_safe, violations
    
    def compute_portfolio_metrics(self) -> Dict[str, float]:
        """Compute portfolio performance metrics"""
        if len(self.history) == 0:
            return {
                'total_pnl': 0.0,
                'win_rate': 0.5,
                'sharpe': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0
            }
        
        # Extract P&L from closed positions
        pnls = [h.get('pnl', 0.0) for h in self.history if h['action'] == 'close']
        
        if len(pnls) == 0:
            return {
                'total_pnl': 0.0,
                'win_rate': 0.5,
                'sharpe': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0
            }
        
        pnls = np.array(pnls)
        
        # Compute metrics
        total_pnl = np.sum(pnls)
        win_rate = np.mean(pnls > 0)
        volatility = np.std(pnls) if len(pnls) > 1 else 0.0
        
        # Sharpe ratio (assuming annualized)
        if volatility > 0:
            sharpe = np.mean(pnls) / volatility * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Max drawdown (simplified)
        cumulative = np.cumsum(pnls)
        if len(cumulative) > 0:
            max_drawdown = np.min(cumulative) / max(1.0, cumulative[0])
        else:
            max_drawdown = 0.0
        
        return {
            'total_pnl': float(total_pnl),
            'win_rate': float(win_rate),
            'sharpe': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'volatility': float(volatility)
        }
    
    def reset(self):
        """Reset risk manager"""
        self.positions.clear()
        self.portfolio_value = self.initial_portfolio_value
        self.history.clear()


# Factory functions
def create_risk_manager(config: RiskConfig = None) -> RiskManager:
    """Factory function to create risk manager"""
    if config is None:
        config = RiskConfig()
    return RiskManager(config)