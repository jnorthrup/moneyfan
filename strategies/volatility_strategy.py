"""
Volatility breakout strategy.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from datetime import datetime


@dataclass
class VolatilityStrategyConfig:
    """Configuration for volatility strategy"""
    lookback_period: int = 20
    volatility_threshold: float = 0.05
    breakout_multiple: float = 2.0
    signal_threshold: float = 0.1
    position_size: float = 0.3


class VolatilityStrategy:
    """
    Volatility breakout strategy.
    """
    
    def __init__(self, config: Optional[VolatilityStrategyConfig] = None):
        self.config = config or VolatilityStrategyConfig()
        
    def compute_signal(self, 
                      prices: np.ndarray,
                      timestamp: datetime,
                      metadata: dict = None) -> Tuple[float, float, str]:
        """
        Compute volatility breakout signal.
        
        Args:
            prices: Price series
            timestamp: Current timestamp
            metadata: Additional metadata
            
        Returns:
            (signal_strength, confidence, regime)
        """
        if len(prices) < self.config.lookback_period:
            return 0.0, 0.0, "volatility"
        
        # Compute volatility
        returns = np.diff(prices)
        volatility = np.std(returns)
        
        # Compute recent volatility
        recent_volatility = np.std(returns[-self.config.lookback_period:])
        
        # Compute Bollinger Band width
        middle = np.mean(prices[-self.config.lookback_period:])
        std = np.std(prices[-self.config.lookback_period:])
        band_width = 2 * std / middle
        
        current_price = prices[-1]
        
        # Determine signal
        signal_strength = 0.0
        
        # Check if volatility is increasing
        if recent_volatility > volatility * 1.5:
            # Volatility is increasing - potential breakout
            if current_price > middle + self.config.breakout_multiple * std:
                signal_strength = 0.8  # Strong buy
            elif current_price < middle - self.config.breakout_multiple * std:
                signal_strength = -0.8  # Strong sell
            elif current_price > middle + std:
                signal_strength = 0.5  # Buy
            elif current_price < middle - std:
                signal_strength = -0.5  # Sell
                
        # Check Bollinger Band width
        if band_width > self.config.volatility_threshold:
            # High volatility regime
            if current_price > middle:
                signal_strength = 0.4
            else:
                signal_strength = -0.4
                
        # Compute confidence
        confidence = min(1.0, recent_volatility / volatility) if volatility > 0 else 0.5
        
        # Apply threshold
        if abs(signal_strength) < self.config.signal_threshold:
            signal_strength = 0.0
            
        return signal_strength, confidence, "volatility"
    
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
            
        # Volatility strategies typically use smaller positions due to higher risk
        base_size = self.config.position_size * portfolio_value / current_price
        return base_size * abs(signal_strength)