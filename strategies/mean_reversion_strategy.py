"""
Mean reversion strategy.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from datetime import datetime


@dataclass
class MeanReversionStrategyConfig:
    """Configuration for mean reversion strategy"""
    lookback_period: int = 14
    rsi_oversold: float = 0.3
    rsi_overbought: float = 0.7
    bollinger_band_period: int = 20
    signal_threshold: float = 0.1
    position_size: float = 0.5


class MeanReversionStrategy:
    """
    Mean reversion strategy using RSI and Bollinger Bands.
    """
    
    def __init__(self, config: Optional[MeanReversionStrategyConfig] = None):
        self.config = config or MeanReversionStrategyConfig()
        
    def _compute_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Compute RSI"""
        if len(prices) < period + 1:
            return 0.5
            
        delta = np.diff(prices)
        gain = np.maximum(delta, 0)
        loss = np.maximum(-delta, 0)
        
        avg_gain = np.mean(gain[-period:])
        avg_loss = np.mean(loss[-period:])
        
        if avg_loss == 0:
            return 1.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi / 100.0  # Normalize to [0, 1]
    
    def _compute_bollinger_bands(self, prices: np.ndarray) -> Tuple[float, float, float]:
        """Compute Bollinger Bands"""
        if len(prices) < self.config.bollinger_band_period:
            return np.mean(prices), 0.0, 0.0
            
        period_prices = prices[-self.config.bollinger_band_period:]
        middle = np.mean(period_prices)
        std = np.std(period_prices)
        
        upper = middle + 2 * std
        lower = middle - 2 * std
        
        return middle, upper, lower
    
    def compute_signal(self, 
                      prices: np.ndarray,
                      timestamp: datetime,
                      metadata: dict = None) -> Tuple[float, float, str]:
        """
        Compute mean reversion signal.
        
        Args:
            prices: Price series
            timestamp: Current timestamp
            metadata: Additional metadata
            
        Returns:
            (signal_strength, confidence, regime)
        """
        if len(prices) < self.config.lookback_period:
            return 0.0, 0.0, "mean_reversion"
        
        current_price = prices[-1]
        
        # Compute RSI
        rsi = self._compute_rsi(prices)
        
        # Compute Bollinger Bands
        middle, upper, lower = self._compute_bollinger_bands(prices)
        
        # Determine signal
        signal_strength = 0.0
        
        if rsi < self.config.rsi_oversold and current_price < lower:
            # Oversold and below lower band - strong buy
            signal_strength = 1.0
        elif rsi > self.config.rsi_overbought and current_price > upper:
            # Overbought and above upper band - strong sell
            signal_strength = -1.0
        elif rsi < self.config.rsi_oversold:
            # Oversold - buy
            signal_strength = 0.5
        elif rsi > self.config.rsi_overbought:
            # Overbought - sell
            signal_strength = -0.5
        elif current_price < lower:
            # Below lower band - weak buy
            signal_strength = 0.3
        elif current_price > upper:
            # Above upper band - weak sell
            signal_strength = -0.3
            
        # Compute confidence
        rsi_deviation = abs(rsi - 0.5)
        confidence = rsi_deviation * 2
        
        # Apply threshold
        if abs(signal_strength) < self.config.signal_threshold:
            signal_strength = 0.0
            
        return signal_strength, confidence, "mean_reversion"
    
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
            
        # Mean reversion typically uses smaller positions
        base_size = self.config.position_size * portfolio_value / current_price
        return base_size * abs(signal_strength)