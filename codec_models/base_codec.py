"""
Abstract Base Class for all 24 SOTA Codecs
===========================================

Defines the interface that all codec implementations must follow.
Each codec must support:
1. Forward pass for signal generation
2. Test-time adaptation (online fine-tuning)
3. Fixed memory management (512-timestep context window)
4. MLX compatibility for Apple Silicon
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("⚠️  MLX not available - using NumPy fallback")


class BaseCodec(ABC):
    """
    Abstract base class for all 24 SOTA codec implementations
    
    Each codec must implement:
    - __init__: Initialize model and memory
    - forward: Generate trading signal
    - test_time_adapter: Online fine-tuning via MLX
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize codec with configuration
        
        Args:
            config: Dictionary with codec-specific parameters
        """
        self.config = config
        self.name = config.get('name', 'base_codec')
        self.version = config.get('version', '1.0')
        
        # Fixed memory: 512-timestep context window
        # Each timestep is 64-dimensional feature vector
        self.memory = np.zeros((512, 64), dtype=np.float32)
        self.memory_idx = 0
        
        # Performance tracking
        self.performance = {
            'sharpe': 0.0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'trade_count': 0,
        }
    
    @abstractmethod
    def forward(self, 
                market_data: Dict[str, Any],
                features: np.ndarray) -> Tuple[float, float]:
        """
        Generate trading signal from market data and features
        
        Args:
            market_data: Dictionary with market state:
                - 'price': float (current price)
                - 'volume': float (trading volume)
                - 'lob_imbalance': float (order book imbalance)
                - 'bid_ask_spread': float (bid-ask spread)
                - 'funding_rate': float (funding rate, optional)
                - 'onchain_active_addresses': float (on-chain metric, optional)
                - 'timestamp': datetime (current timestamp)
            
            features: Computed technical indicators/features [n_features]
            
        Returns:
            Tuple[confidence, direction] where:
                - confidence: float in [0, 1] - signal confidence
                - direction: float in [-1, 1] - negative = sell, positive = buy
        """
        pass
    
    @abstractmethod
    def test_time_adapter(self, 
                         batch_data: Dict[str, Any],
                         learning_rate: float = 1e-3) -> None:
        """
        Online fine-tuning of codec using MLX value_and_grad
        
        Args:
            batch_data: Dictionary with training data:
                - 'inputs': np.ndarray [batch, n_features]
                - 'targets': np.ndarray [batch, 2] - [confidence, direction]
                - 'weights': np.ndarray [batch] (optional, sample weights)
            
            learning_rate: Low LR for stable online updates (1e-3 typical)
        """
        pass
    
    def update_memory(self, signal: float, context: np.ndarray) -> None:
        """
        Update fixed 512-timestep memory with new context
        
        Args:
            signal: Trading signal (direction)
            context: Feature vector [64] for current timestep
        """
        # Ensure context is 64-dimensional
        if len(context) != 64:
            # Pad or truncate
            if len(context) < 64:
                padded = np.zeros(64, dtype=np.float32)
                padded[:len(context)] = context
                context = padded
            else:
                context = context[:64]
        
        self.memory[self.memory_idx] = context
        self.memory_idx = (self.memory_idx + 1) % 512
    
    def get_memory_summary(self) -> Dict[str, float]:
        """
        Get summary statistics for long-term state
        
        Returns:
            Dictionary with mean, std, quantiles of memory
        """
        if self.memory_idx == 0:
            return {}
        
        memory_slice = self.memory[:self.memory_idx]
        
        return {
            'mean': float(np.mean(memory_slice)),
            'std': float(np.std(memory_slice)),
            'q25': float(np.quantile(memory_slice, 0.25)),
            'q75': float(np.quantile(memory_slice, 0.75)),
            'count': self.memory_idx,
        }
    
    def update_performance(self, 
                          pnl: float, 
                          direction: float, 
                          actual: float) -> None:
        """
        Update performance tracking metrics
        
        Args:
            pnl: Profit/loss for this trade
            direction: Predicted direction
            actual: Actual outcome
        """
        self.performance['total_pnl'] += pnl
        self.performance['trade_count'] += 1
        
        # Update win rate
        if pnl > 0:
            self.performance['win_rate'] = (
                (self.performance['win_rate'] * (self.performance['trade_count'] - 1) + 1) /
                self.performance['trade_count']
            )
        else:
            self.performance['win_rate'] = (
                (self.performance['win_rate'] * (self.performance['trade_count'] - 1)) /
                self.performance['trade_count']
            )
    
    def get_performance(self) -> Dict[str, Any]:
        """
        Get current performance metrics
        
        Returns:
            Dictionary with performance statistics
        """
        performance = self.performance.copy()
        
        # Calculate Sharpe ratio if we have enough trades
        if performance['trade_count'] >= 10:
            # Simplified Sharpe calculation (would need return history for proper Sharpe)
            performance['sharpe'] = performance['total_pnl'] / max(1.0, performance['trade_count'])
        
        return performance
    
    def reset_performance(self) -> None:
        """Reset performance tracking"""
        self.performance = {
            'sharpe': 0.0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'trade_count': 0,
        }
    
    def __repr__(self) -> str:
        """String representation"""
        return f"{self.name} (v{self.version})"
    
    def validate_signal(self, confidence: float, direction: float) -> Tuple[float, float]:
        """
        Validate and normalize signal output
        
        Args:
            confidence: Raw confidence value
            direction: Raw direction value
            
        Returns:
            Normalized (confidence, direction)
        """
        # Clip to valid ranges
        confidence = max(0.0, min(1.0, float(confidence)))
        direction = max(-1.0, min(1.0, float(direction)))
        
        return confidence, direction


class CodecFactory:
    """
    Factory for creating codec instances
    """
    
    @staticmethod
    def create_codec(codec_id: int, config: Dict[str, Any] = None) -> BaseCodec:
        """
        Create a codec instance by ID
        
        Args:
            codec_id: Codec number (1-24)
            config: Codec configuration
            
        Returns:
            Codec instance
        """
        if config is None:
            config = {}
        
        config['codec_id'] = codec_id
        config['name'] = f"codec_{codec_id:02d}"
        
        # Import codec dynamically
        codec_module = f"codecs.codec_{codec_id:02d}_generic"
        
        try:
            module = __import__(codec_module, fromlist=[f'Codec_{codec_id:02d}'])
            codec_class = getattr(module, f'Codec_{codec_id:02d}')
            return codec_class(config)
        except (ImportError, AttributeError):
            # Fallback to generic codec
            from .codec_generic import GenericCodec
            return GenericCodec(config)


# Factory function for getting all codecs
def get_all_codecs(config: Dict[str, Any] = None) -> list:
    """
    Get all 24 codecs as a list
    
    Args:
        config: Base configuration for all codecs
        
    Returns:
        List of 24 codec instances
    """
    if config is None:
        config = {}
    
    codecs = []
    for i in range(1, 25):
        codec = CodecFactory.create_codec(i, config)
        codecs.append(codec)
    
    return codecs