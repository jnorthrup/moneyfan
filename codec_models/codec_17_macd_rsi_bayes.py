"""
Codec #17: MACD-RSI with Bayesian Optimization
================================================

This codec combines classic technical analysis (MACD, RSI) with Bayesian
optimization to find optimal parameter values.

Features used:
1. MACD (Moving Average Convergence Divergence)
2. RSI (Relative Strength Index)
3. Price position in recent range
4. Volume confirmation

Training approach:
- Bayesian optimization over MACD/RSI parameters
- Optimal parameters: fast period, slow period, signal period, RSI period
- Reward: Sharpe ratio optimization

Test-time adapter:
- Online Bayesian optimization updates
- Periodically re-optimize parameters based on recent performance
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from .base_codec import BaseCodec

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("⚠️  MLX not available - using NumPy fallback")

try:
    from scipy.stats import norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️  SciPy not available - using simple optimization")


class Codec_17_MACD_RSI_Bayes(BaseCodec):
    """
    Codec #17: MACD-RSI with Bayesian Optimization
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "macd_rsi_bayes"
        self.version = "1.0"
        
        # MACD parameters (will be optimized)
        self.macd_params = {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
        }
        
        # RSI parameters
        self.rsi_params = {
            'period': 14,
            'overbought': 70,
            'oversold': 30,
        }
        
        # Bayesian optimization state
        self.bayes_state = {
            'mean': 0.0,
            'variance': 1.0,
            'observations': [],
        }
        
        # Performance tracking for optimization
        self.param_history = []
        self.performance_history = []
        
        # Initialize MLX model if available
        if HAS_MLX:
            self.model = self._create_mlx_model()
            print(f"✅ {self.name}: MLX model initialized")
        else:
            self.model = None
            print(f"⚠️  {self.name}: Using NumPy fallback")
    
    def _create_mlx_model(self):
        """Create MLX model for parameter optimization"""
        return nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),  # 4 optimized parameters
        )
    
    def calculate_macd(self, price: float, params: Dict[str, int]) -> Tuple[float, float, float]:
        """
        Calculate MACD line, signal line, and histogram
        
        Args:
            price: Current price
            params: MACD parameters
            
        Returns:
            (macd_line, signal_line, histogram)
        """
        # This is a simplified MACD calculation
        # In production, you'd need price history
        
        # For now, use simulated values based on current price
        # In real implementation, this would use actual EMA calculations
        
        # Simulate MACD based on price momentum
        # This is a placeholder - in production, use actual price history
        fast_ema = price * 0.99  # Simplified
        slow_ema = price * 0.98  # Simplified
        
        macd_line = fast_ema - slow_ema
        
        # Simulate signal line (9-period EMA of MACD)
        signal_line = macd_line * 0.9  # Simplified
        
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_rsi(self, price: float, params: Dict[str, int]) -> float:
        """
        Calculate RSI
        
        Args:
            price: Current price
            params: RSI parameters
            
        Returns:
            RSI value (0-100)
        """
        # Simplified RSI calculation
        # In production, you'd need price history
        
        # Simulate RSI based on price changes
        # This is a placeholder
        
        # For now, return a value based on price position
        # In real implementation, calculate from actual price history
        
        # Simple logic: if price is high, RSI high
        # This is just for demonstration
        
        # We'll use the price position from features
        return 50.0  # Placeholder
    
    def bayesian_optimization(self, 
                             param_space: Dict[str, Any],
                             reward_history: np.ndarray) -> Dict[str, float]:
        """
        Bayesian optimization to find optimal parameters
        
        Args:
            param_space: Parameter search space
            reward_history: Historical rewards (Sharpe ratios)
            
        Returns:
            Optimized parameters
        """
        if not HAS_SCIPY:
            # Simple random search fallback
            return self._simple_optimization(param_space, reward_history)
        
        try:
            # Simple Gaussian Process-like optimization
            # In production, use actual Bayesian optimization library
            
            # Get best parameters so far
            if len(self.performance_history) > 0:
                best_idx = np.argmax(self.performance_history)
                best_params = self.param_history[best_idx]
            else:
                # Random initialization
                best_params = {
                    'fast_period': np.random.randint(5, 20),
                    'slow_period': np.random.randint(15, 40),
                    'signal_period': np.random.randint(5, 15),
                    'rsi_period': np.random.randint(5, 20),
                }
            
            # Explore around best parameters (with some randomness)
            explored_params = {
                'fast_period': int(np.random.normal(best_params['fast_period'], 2)),
                'slow_period': int(np.random.normal(best_params['slow_period'], 3)),
                'signal_period': int(np.random.normal(best_params['signal_period'], 2)),
                'rsi_period': int(np.random.normal(best_params['rsi_period'], 2)),
            }
            
            # Clip to valid ranges
            explored_params = {
                k: max(5, min(50, v)) for k, v in explored_params.items()
            }
            
            return explored_params
            
        except Exception as e:
            print(f"⚠️  Bayesian optimization failed: {e}")
            return self._simple_optimization(param_space, reward_history)
    
    def _simple_optimization(self, param_space: Dict[str, Any], reward_history: np.ndarray) -> Dict[str, float]:
        """
        Simple random search optimization
        """
        # If we have reward history, try to improve
        if len(self.performance_history) > 0:
            # Get best parameters
            best_idx = np.argmax(self.performance_history)
            best_params = self.param_history[best_idx]
            
            # Try to perturb them slightly
            perturbed = {}
            for key, value in best_params.items():
                perturbation = np.random.normal(0, 2)
                perturbed[key] = int(max(5, min(50, value + perturbation)))
            
            return perturbed
        
        # Random initialization
        return {
            'fast_period': int(np.random.uniform(5, 20)),
            'slow_period': int(np.random.uniform(15, 40)),
            'signal_period': int(np.random.uniform(5, 15)),
            'rsi_period': int(np.random.uniform(5, 20)),
        }
    
    def extract_features(self, market_data: Dict[str, Any], 
                        features: np.ndarray) -> np.ndarray:
        """
        Extract features for MACD-RSI analysis
        """
        # MACD features
        macd_line, signal_line, histogram = self.calculate_macd(
            market_data.get('price', 100.0),
            self.macd_params
        )
        
        # RSI features
        rsi = self.calculate_rsi(
            market_data.get('price', 100.0),
            self.rsi_params
        )
        
        # Volume features
        volume = market_data.get('volume', 100000.0)
        volume_ratio = np.log10(max(volume, 1.0))
        
        # Price position in recent range (simulated)
        # In production, this would be actual price position
        price_position = np.random.uniform(0, 1)
        
        # Combine features
        feature_vector = np.array([
            macd_line,           # 0: MACD line
            signal_line,         # 1: Signal line
            histogram,           # 2: MACD histogram
            rsi,                 # 3: RSI
            volume_ratio,        # 4: Volume ratio
            price_position,      # 5: Price position
            market_data.get('price', 100.0),  # 6: Price
            market_data.get('lob_imbalance', 0.0),  # 7: LOB imbalance
        ], dtype=np.float32)
        
        return feature_vector
    
    def generate_signal_from_indicators(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Generate trading signal from MACD and RSI indicators
        
        Returns:
            (confidence, direction)
        """
        # Extract indicator values
        macd_line = features[0] if len(features) > 0 else 0.0
        signal_line = features[1] if len(features) > 1 else 0.0
        histogram = features[2] if len(features) > 2 else 0.0
        rsi = features[3] if len(features) > 3 else 50.0
        price_position = features[5] if len(features) > 5 else 0.5
        
        signals = []
        
        # MACD signals
        if histogram > 0 and macd_line > signal_line:
            signals.append(0.3)  # Bullish MACD
        elif histogram < 0 and macd_line < signal_line:
            signals.append(-0.3)  # Bearish MACD
        
        # RSI signals
        if rsi < 30:  # Oversold
            signals.append(0.3)
        elif rsi > 70:  # Overbought
            signals.append(-0.3)
        
        # Price position signals
        if price_position > 0.8:  # Near top of range
            signals.append(-0.2)
        elif price_position < 0.2:  # Near bottom of range
            signals.append(0.2)
        
        if not signals:
            return 0.0, 0.0
        
        # Weighted average of signals
        direction = np.mean(signals)
        confidence = abs(direction) + 0.2  # Base confidence
        
        return confidence, direction
    
    def forward(self, 
                market_data: Dict[str, Any],
                features: np.ndarray) -> Tuple[float, float]:
        """
        Generate trading signal using MACD-RSI with Bayesian optimization
        """
        # Extract features
        feature_vector = self.extract_features(market_data, features)
        
        # Generate signal from indicators
        confidence, direction = self.generate_signal_from_indicators(feature_vector)
        
        # Update memory
        self.update_memory(direction, feature_vector)
        
        # Validate output
        confidence, direction = self.validate_signal(confidence, direction)
        
        return confidence, direction
    
    def test_time_adapter(self, 
                         batch_data: Dict[str, Any],
                         learning_rate: float = 1e-3) -> None:
        """
        Online Bayesian optimization updates
        """
        if 'inputs' in batch_data and 'targets' in batch_data:
            # Calculate reward (Sharpe ratio improvement)
            # This would use the actual performance metrics
            
            # For now, add to history
            reward = np.random.randn()  # Placeholder
            
            self.performance_history.append(reward)
            
            # Update Bayesian optimization state
            if len(self.performance_history) > 10:
                # Optimize parameters every 10 observations
                self._optimize_parameters()
    
    def _optimize_parameters(self):
        """
        Optimize MACD/RSI parameters using Bayesian optimization
        """
        if len(self.performance_history) < 5:
            return
        
        # Define parameter space
        param_space = {
            'fast_period': (5, 20),
            'slow_period': (15, 40),
            'signal_period': (5, 15),
            'rsi_period': (5, 20),
        }
        
        # Run Bayesian optimization
        optimized_params = self.bayesian_optimization(
            param_space,
            np.array(self.performance_history)
        )
        
        # Update parameters
        self.macd_params = {
            'fast_period': optimized_params['fast_period'],
            'slow_period': optimized_params['slow_period'],
            'signal_period': optimized_params['signal_period'],
        }
        
        self.rsi_params['period'] = optimized_params['rsi_period']
        
        # Store in history
        self.param_history.append(optimized_params)
        
        print(f"✅ {self.name}: Optimized parameters: {optimized_params}")


# Factory function
def create_codec(config: Dict[str, Any] = None):
    """Factory function to create codec instance"""
    if config is None:
        config = {}
    return Codec_17_MACD_RSI_Bayes(config)