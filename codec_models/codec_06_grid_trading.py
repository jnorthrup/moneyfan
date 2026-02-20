"""
Codec 06: Grid Trading
Profits from oscillating markets with grid levels.
"""

import numpy as np
from typing import Tuple, Dict, Any
from .base_codec import BaseCodec

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


class Codec06(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'grid_trading'
        super().__init__(config)
        
        self.grid_spacing = config.get('grid_spacing', 0.01)
        
        if HAS_MLX:
            self.model = nn.Sequential(
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 3)
            )
    
    def forward(self, market_data: Dict[str, Any], features: np.ndarray) -> Tuple[float, float]:
        price = market_data.get('price', 0)
        sma = market_data.get('sma_15', price)
        atr = market_data.get('atr_14', price * 0.01)
        
        grid_signal = 0.0
        if sma > 0:
            deviation = (price - sma) / sma
            grid_signal = -np.sign(deviation) * min(abs(deviation) / self.grid_spacing, 1.0)
        
        volatility_factor = 1.0
        if atr > 0 and sma > 0:
            vol_ratio = atr / sma
            volatility_factor = min(vol_ratio / 0.05, 1.5)
        
        direction = grid_signal * volatility_factor
        confidence = min(abs(grid_signal) + 0.2, 1.0)
        
        regime = market_data.get('regime_label', 1)
        if regime == 1:  # Sideways
            confidence *= 1.2
        else:
            confidence *= 0.7
        
        if HAS_MLX and self.model is not None and len(features) >= 64:
            try:
                mx_features = mx.array(features[:64].reshape(1, -1).astype(np.float32))
                output = self.model(mx_features)
                direction = direction * 0.6 + float(np.tanh(output[0, 1])) * 0.4
            except:
                pass
        
        return self.validate_signal(confidence, direction)
    
    def test_time_adapter(self, batch_data: Dict[str, Any], learning_rate: float = 1e-3) -> None:
        pass
