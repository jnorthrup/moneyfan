"""
Trend-following strategy.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from datetime import datetime


@dataclass
class TrendStrategyConfig:
    """Configuration for trend strategy"""
    lookback_period: int = 20
    ma_slow: int = 50
    ma_fast: int = 20
    signal_threshold: float = 0.1
    position_size: float = 0.8


class TrendStrategy:
    """
    Trend-following strategy using moving averages and momentum.
    """
    
    def __init__(self, config: Optional[TrendStrategyConfig] = None):
        self.config = config or TrendStrategyConfig()
        
    def compute_signal(self, 
                      prices: np.ndarray,
                      timestamp: datetime,
                      metadata: dict = None) -> Tuple[float, float, str]:
        """
        Compute trend signal.
        
        Args:
            prices: Price series
            timestamp: Current timestamp
            metadata: Additional metadata
            
        Returns:
            (signal_strength, confidence, regime)
        """
        if len(prices) < self.config.ma_slow:
            return 0.0, 0.0, "trend"
        
        # Compute moving averages
        ma_fast = np.mean(prices[-self.config.ma_fast:])
        ma_slow = np.mean(prices[-self.config.ma_slow:])
        
        # Compute momentum
        returns = (prices[-1] - prices[-self.config.lookback_period]) / prices[-self.config.lookback_period]
        
        # Compute signal
        if ma_fast > ma_slow and returns > 0:
            signal_strength = 1.0  # Strong buy
        elif ma_fast < ma_slow and returns < 0:
            signal_strength = -1.0  # Strong sell
        elif ma_fast > ma_slow:
            signal_strength = 0.5  # Weak buy
        elif ma_fast < ma_slow:
            signal_strength = -0.5  # Weak sell
        else:
            signal_strength = 0.0
            
        # Compute confidence
        confidence = min(1.0, abs(returns) * 10)
        
        # Apply threshold
        if abs(signal_strength) < self.config.signal_threshold:
            signal_strength = 0.0
            
        return signal_strength, confidence, "trend"
    
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
            
        # Scale position by signal strength
        base_size = self.config.position_size * portfolio_value / current_price
        return base_size * abs(signal_strength)