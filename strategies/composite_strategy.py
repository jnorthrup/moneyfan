"""
Composite strategy combining multiple strategies.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from datetime import datetime


@dataclass
class CompositeStrategyConfig:
    """Configuration for composite strategy"""
    trend_weight: float = 0.3
    mean_reversion_weight: float = 0.3
    volatility_weight: float = 0.4
    signal_threshold: float = 0.3  # Higher threshold


class CompositeStrategy:
    """
    Composite strategy combining trend, mean reversion, and volatility.
    """
    
    def __init__(self, config: Optional[CompositeStrategyConfig] = None):
        self.config = config or CompositeStrategyConfig()
        
    def compute_signal(self, 
                      prices: np.ndarray,
                      timestamp: datetime,
                      metadata: dict = None) -> Tuple[float, float, str]:
        """
        Compute composite signal.
        
        Args:
            prices: Price series
            timestamp: Current timestamp
            metadata: Additional metadata
            
        Returns:
            (signal_strength, confidence, regime)
        """
        if len(prices) < 20:
            return 0.0, 0.0, "composite"
        
        # Compute individual signals
        trend_signal = self._compute_trend_signal(prices)
        mean_rev_signal = self._compute_mean_reversion_signal(prices)
        vol_signal = self._compute_volatility_signal(prices)
        
        # Weighted combination
        signal_strength = (
            self.config.trend_weight * trend_signal +
            self.config.mean_reversion_weight * mean_rev_signal +
            self.config.volatility_weight * vol_signal
        )
        
        # Average confidence
        confidence = (abs(trend_signal) + abs(mean_rev_signal) + abs(vol_signal)) / 3
        
        # Determine dominant regime
        if abs(trend_signal) > abs(mean_rev_signal) and abs(trend_signal) > abs(vol_signal):
            regime = "trend"
        elif abs(mean_rev_signal) > abs(trend_signal) and abs(mean_rev_signal) > abs(vol_signal):
            regime = "mean_reversion"
        elif abs(vol_signal) > abs(trend_signal) and abs(vol_signal) > abs(mean_rev_signal):
            regime = "volatility"
        else:
            regime = "composite"
            
        # Apply threshold
        if abs(signal_strength) < self.config.signal_threshold:
            signal_strength = 0.0
            
        return signal_strength, confidence, regime
    
    def _compute_trend_signal(self, prices: np.ndarray) -> float:
        """Compute trend signal"""
        if len(prices) < 20:
            return 0.0
            
        ma_20 = np.mean(prices[-20:])
        ma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else ma_20
        
        if ma_20 > ma_50:
            return 0.5
        elif ma_20 < ma_50:
            return -0.5
        return 0.0
    
    def _compute_mean_reversion_signal(self, prices: np.ndarray) -> float:
        """Compute mean reversion signal"""
        if len(prices) < 14:
            return 0.0
            
        # Simple RSI-like computation
        delta = np.diff(prices[-15:])
        gain = np.maximum(delta, 0)
        loss = np.maximum(-delta, 0)
        
        avg_gain = np.mean(gain) if len(gain) > 0 else 0.0
        avg_loss = np.mean(loss) if len(loss) > 0 else 0.0
        
        if avg_loss == 0:
            return 0.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi_norm = rsi / 100.0
        
        if rsi_norm < 0.3:
            return 0.5
        elif rsi_norm > 0.7:
            return -0.5
        return 0.0
    
    def _compute_volatility_signal(self, prices: np.ndarray) -> float:
        """Compute volatility signal"""
        if len(prices) < 20:
            return 0.0
            
        returns = np.diff(prices[-20:])
        volatility = np.std(returns)
        
        # Higher volatility might mean breakout opportunities
        if volatility > 0.05:
            current_price = prices[-1]
            mean_price = np.mean(prices[-20:])
            
            if current_price > mean_price:
                return 0.4
            else:
                return -0.4
        return 0.0
    
    def compute_position_size(self,
                            signal_strength: float,
                            current_price: float,
                            portfolio_value: float) -> float:
        """
        Compute position size based on signal strength.
        
        Args:
            signal_strength: Signal strength [-1, 1]
            current_price: Current price
            portfolio_value: Portfolio value
            
        Returns:
            Position size in units
        """
        if abs(signal_strength) < self.config.signal_threshold:
            return 0.0
            
        # Composite strategy uses moderate position sizes
        # Scale by signal strength and ensure reasonable size
        base_size = 0.6 * portfolio_value / current_price
        size = base_size * abs(signal_strength)
        
        # Cap at 10% of portfolio
        max_size = 0.1 * portfolio_value / current_price
        return min(size, max_size)