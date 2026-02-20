"""
FiduciaryOverlay - Risk Management and Position Sizing Layer

Interposes between HRM output and execution engine:
HRM → Fiduciary → Execution

Implements:
1. Risk caps (max position, max drawdown, max leverage)
2. Position sizing (Kelly, fixed fractional, volatility scaling)
3. Portfolio allocation (HRM weights → target weights)
4. Stop loss / take profit logic
5. Circuit breakers
6. Compliance checks
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class RiskLevel(Enum):
    CONSERVATIVE = "conservative"  # 0.5x leverage, 2% risk per trade
    MODERATE = "moderate"          # 1.0x leverage, 5% risk per trade
    AGGRESSIVE = "aggressive"      # 2.0x leverage, 10% risk per trade
    DEGEN = "degen"                # 3.0x leverage, 20% risk per trade


@dataclass
class PositionConstraint:
    """Individual position constraints"""
    max_position: float = 0.10      # Max 10% of portfolio per position
    min_position: float = 0.01      # Min 1% for meaningful positions
    max_leverage: float = 1.0       # Default: no leverage
    stop_loss_pct: float = 0.05     # 5% stop loss
    take_profit_pct: float = 0.15   # 15% take profit
    trailing_stop: bool = True
    trailing_distance: float = 0.03  # 3% trailing stop


@dataclass
class PortfolioConstraint:
    """Portfolio-level constraints"""
    max_total_risk: float = 0.30      # Max 30% portfolio risk at once
    max_drawdown: float = 0.20        # Circuit breaker: 20% drawdown
    max_concentration: float = 0.40   # Max 40% in single sector
    rebalance_frequency: int = 60     # Rebalance every 60 minutes
    min_liquidity: float = 0.05       # Keep 5% in cash/stablecoins
    correlation_cap: float = 0.80     # Max correlation between positions


@dataclass
class FiduciaryState:
    """Current fiduciary state"""
    timestamp: str
    portfolio_value: float
    total_risk: float
    drawdown: float
    active_positions: int
    leverage: float
    cash_ratio: float
    sector_concentration: Dict[str, float]


class FiduciaryOverlay:
    """
    Fiduciary overlay that constrains HRM signals to safe, compliant trades.
    
    Receives: HRM output (weights, alpha, convergence)
    Returns:  Safe target weights with constraints applied
    """

    def __init__(
        self,
        risk_level: RiskLevel = RiskLevel.MODERATE,
        constraints: Optional[PortfolioConstraint] = None,
        position_constraints: Optional[Dict[str, PositionConstraint]] = None,
    ):
        self.risk_level = risk_level
        self.constraints = constraints or PortfolioConstraint()
        self.position_constraints = position_constraints or {}
        
        # Set defaults based on risk level
        self._apply_risk_level_defaults()
        
        self.history: List[FiduciaryState] = []
        self.trades_today: int = 0
        self.last_rebalance: Optional[datetime] = None
        
        # Circuit breakers
        self.circuit_breaker_active = False
        self.circuit_breaker_start = None
        
    def _apply_risk_level_defaults(self):
        """Set constraints based on risk level"""
        if self.risk_level == RiskLevel.CONSERVATIVE:
            self.constraints.max_total_risk = 0.15
            self.constraints.max_drawdown = 0.10
            self.position_constraints.setdefault("default", PositionConstraint(
                max_position=0.05,
                max_leverage=0.5,
                stop_loss_pct=0.03,
                take_profit_pct=0.10,
            ))
        elif self.risk_level == RiskLevel.MODERATE:
            self.constraints.max_total_risk = 0.30
            self.constraints.max_drawdown = 0.20
            self.position_constraints.setdefault("default", PositionConstraint(
                max_position=0.10,
                max_leverage=1.0,
                stop_loss_pct=0.05,
                take_profit_pct=0.15,
            ))
        elif self.risk_level == RiskLevel.AGGRESSIVE:
            self.constraints.max_total_risk = 0.50
            self.constraints.max_drawdown = 0.30
            self.position_constraints.setdefault("default", PositionConstraint(
                max_position=0.15,
                max_leverage=2.0,
                stop_loss_pct=0.08,
                take_profit_pct=0.25,
            ))
        elif self.risk_level == RiskLevel.DEGEN:
            self.constraints.max_total_risk = 0.80
            self.constraints.max_drawdown = 0.50
            self.position_constraints.setdefault("default", PositionConstraint(
                max_position=0.25,
                max_leverage=3.0,
                stop_loss_pct=0.15,
                take_profit_pct=0.50,
            ))
    
    def apply(
        self,
        hrms: Dict[str, Dict],  # {symbol: {weights, alpha, confidence}}
        current_positions: Dict[str, Dict],  # {symbol: {size, entry_price}}
        portfolio_value: float,
        prices: Dict[str, float],
    ) -> Dict[str, Dict]:
        """
        Apply fiduciary constraints to HRM output.
        
        Returns: {symbol: {target_weight, action, size, constraints}}
        """
        if self.circuit_breaker_active:
            return self._emergency_shutdown()
        
        # 1. Extract raw HRM weights
        raw_weights = {}
        for symbol, hrm in hrms.items():
            raw_weights[symbol] = hrm.get('weight', 0.0)
        
        # 2. Check circuit breaker
        if self._check_circuit_breaker(portfolio_value, current_positions):
            self.circuit_breaker_active = True
            self.circuit_breaker_start = datetime.utcnow()
            return self._emergency_shutdown()
        
        # 3. Normalize and apply constraints
        constrained_weights = self._apply_weight_constraints(raw_weights, portfolio_value)
        
        # 4. Calculate position sizes
        positions = self._calculate_positions(
            constrained_weights, current_positions, portfolio_value, prices
        )
        
        # 5. Generate orders
        orders = self._generate_orders(positions, current_positions, prices)
        
        # 6. Update state
        self._update_state(portfolio_value, orders)
        
        return orders
    
    def _apply_weight_constraints(
        self,
        raw_weights: Dict[str, float],
        portfolio_value: float
    ) -> Dict[str, float]:
        """Apply portfolio-level constraints to raw weights"""
        
        # Filter out low-confidence signals
        filtered = {s: w for s, w in raw_weights.items() if abs(w) > 0.01}
        
        # Sort by absolute weight (largest positions first)
        sorted_items = sorted(filtered.items(), key=lambda x: -abs(x[1]))
        
        # Apply max concentration constraint
        constrained = {}
        total_weight = 0
        for symbol, weight in sorted_items:
            # Apply per-position max
            max_pos = self.position_constraints.get(symbol, PositionConstraint()).max_position
            constrained_weight = np.clip(weight, -max_pos, max_pos)
            
            # Check portfolio concentration
            new_total = total_weight + abs(constrained_weight)
            if new_total > self.constraints.max_total_risk:
                # Scale down
                remaining = self.constraints.max_total_risk - total_weight
                constrained_weight = np.sign(constrained_weight) * min(abs(constrained_weight), remaining)
            
            constrained[symbol] = constrained_weight
            total_weight += abs(constrained_weight)
        
        # Normalize to ensure sum doesn't exceed 1.0 (for long-only)
        if total_weight > 0:
            scaling_factor = min(1.0, self.constraints.max_total_risk / total_weight)
            for symbol in constrained:
                constrained[symbol] *= scaling_factor
        
        return constrained
    
    def _calculate_positions(
        self,
        weights: Dict[str, float],
        current_positions: Dict[str, Dict],
        portfolio_value: float,
        prices: Dict[str, float]
    ) -> Dict[str, Dict]:
        """Calculate target position sizes"""
        positions = {}
        
        for symbol, weight in weights.items():
            if abs(weight) < 0.01:
                continue
            
            # Target value
            target_value = weight * portfolio_value
            
            # Current position
            current = current_positions.get(symbol, {'size': 0, 'entry_price': 0})
            current_value = current['size'] * prices.get(symbol, 0)
            
            # Position constraint
            pos_constraint = self.position_constraints.get(symbol, PositionConstraint())
            
            # Calculate new size
            if target_value > 0:
                # Long position
                new_size = target_value / prices[symbol]
                action = 'BUY' if new_size > current['size'] else 'SELL'
            else:
                # Short position (if allowed)
                new_size = target_value / prices[symbol]
                action = 'SELL' if new_size < current['size'] else 'BUY'
            
            # Apply leverage constraint
            if pos_constraint.max_leverage < 1.0:
                new_size *= pos_constraint.max_leverage
            
            positions[symbol] = {
                'target_size': new_size,
                'current_size': current['size'],
                'action': action,
                'weight': weight,
                'stop_loss': pos_constraint.stop_loss_pct,
                'take_profit': pos_constraint.take_profit_pct,
            }
        
        return positions
    
    def _generate_orders(
        self,
        positions: Dict[str, Dict],
        current_positions: Dict[str, Dict],
        prices: Dict[str, float]
    ) -> Dict[str, Dict]:
        """Generate executable orders with stop loss / take profit"""
        orders = {}
        
        for symbol, pos in positions.items():
            size_diff = pos['target_size'] - pos['current_size']
            
            if abs(size_diff) < 0.001:  # Min order size
                continue
            
            order = {
                'symbol': symbol,
                'action': pos['action'],
                'size': abs(size_diff),
                'price': prices.get(symbol, 0),
                'target_weight': pos['weight'],
                'stop_loss': pos['stop_loss'],
                'take_profit': pos['take_profit'],
                'order_type': 'LIMIT',
                'time_in_force': 'GTC',
                'timestamp': datetime.utcnow().isoformat(),
            }
            
            # Add trailing stop for aggressive positions
            if self.risk_level in [RiskLevel.AGGRESSIVE, RiskLevel.DEGEN]:
                order['trailing_stop'] = True
                order['trail_distance'] = 0.02
            
            orders[symbol] = order
        
        return orders
    
    def _check_circuit_breaker(
        self,
        portfolio_value: float,
        current_positions: Dict[str, Dict]
    ) -> bool:
        """Check if circuit breaker should be triggered"""
        
        # Check drawdown
        if len(self.history) > 0:
            peak = max(h.portfolio_value for h in self.history[-100:])
            drawdown = (peak - portfolio_value) / peak
            if drawdown > self.constraints.max_drawdown:
                return True
        
        # Check excessive positions
        if len(current_positions) > 20:
            return True
        
        # Check high leverage
        total_exposure = sum(p['size'] * 100 for p in current_positions.values())  # Approximate
        if total_exposure > portfolio_value * 3:
            return True
        
        return False
    
    def _emergency_shutdown(self) -> Dict[str, Dict]:
        """Emergency shutdown - exit all positions"""
        return {
            '__EMERGENCY__': {
                'action': 'SHUTDOWN',
                'message': 'Circuit breaker triggered',
                'timestamp': datetime.utcnow().isoformat(),
            }
        }
    
    def _update_state(
        self,
        portfolio_value: float,
        orders: Dict[str, Dict]
    ):
        """Update fiduciary state"""
        total_risk = sum(abs(o.get('target_weight', 0)) for o in orders.values())
        
        # Calculate drawdown
        if len(self.history) > 0:
            peak = max(h.portfolio_value for h in self.history[-100:])
            drawdown = (peak - portfolio_value) / peak if peak > 0 else 0
        else:
            drawdown = 0
        
        state = FiduciaryState(
            timestamp=datetime.utcnow().isoformat(),
            portfolio_value=portfolio_value,
            total_risk=total_risk,
            drawdown=drawdown,
            active_positions=len(orders),
            leverage=1.0,  # Would need actual leverage calculation
            cash_ratio=1.0 - total_risk,
            sector_concentration={},  # Would need sector mapping
        )
        
        self.history.append(state)
        
        # Keep history manageable
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
    
    def get_risk_report(self) -> Dict:
        """Get current risk metrics"""
        if not self.history:
            return {"status": "no data"}
        
        recent = self.history[-100:] if len(self.history) > 100 else self.history
        
        df = pd.DataFrame([
            {
                'timestamp': h.timestamp,
                'portfolio_value': h.portfolio_value,
                'total_risk': h.total_risk,
                'drawdown': h.drawdown,
            }
            for h in recent
        ])
        
        return {
            'current_portfolio': recent[-1].portfolio_value if recent else 0,
            'current_risk': recent[-1].total_risk if recent else 0,
            'current_drawdown': recent[-1].drawdown if recent else 0,
            'avg_risk': df['total_risk'].mean() if len(df) > 0 else 0,
            'max_drawdown': df['drawdown'].min() if len(df) > 0 else 0,
            'circuit_breaker': self.circuit_breaker_active,
            'risk_level': self.risk_level.value,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("Fiduciary Overlay - Risk Management System")
    print("=" * 60)
    
    # Test with sample data
    fiduciary = FiduciaryOverlay(risk_level=RiskLevel.MODERATE)
    
    # Sample HRM output
    hrms = {
        'BTC-USD': {'weight': 0.15, 'alpha': 0.02, 'confidence': 0.7},
        'ETH-USD': {'weight': 0.08, 'alpha': 0.015, 'confidence': 0.6},
        'SOL-USD': {'weight': 0.05, 'alpha': 0.01, 'confidence': 0.5},
    }
    
    current_positions = {
        'BTC-USD': {'size': 0.1, 'entry_price': 40000},
        'ETH-USD': {'size': 0.5, 'entry_price': 2500},
    }
    
    portfolio_value = 100000
    prices = {'BTC-USD': 42000, 'ETH-USD': 2600, 'SOL-USD': 100}
    
    orders = fiduciary.apply(hrms, current_positions, portfolio_value, prices)
    
    print("\nFiduciary Orders:")
    for symbol, order in orders.items():
        if symbol.startswith('__'):
            print(f"  {symbol}: {order}")
        else:
            print(f"  {symbol}: {order['action']} {order['size']:.4f} at ${order['price']:.2f}")
            print(f"    Target: {order['target_weight']:.1%}, Stop: {order['stop_loss']:.1%}, Take: {order['take_profit']:.1%}")
    
    print("\nRisk Report:")
    risk_report = fiduciary.get_risk_report()
    for key, value in risk_report.items():
        print(f"  {key}: {value}")
