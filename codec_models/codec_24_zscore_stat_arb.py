"""
Codec 24: Z-Score Stat Arb
Multi-asset z-score mean-reversion baskets.
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


class Codec24(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'zscore_stat_arb'
        super().__init__(config)
        
        self.z_threshold = config.get('z_threshold', 2.0)
        self.lookback = config.get('lookback', 20)
        
        if HAS_MLX:
            self.model = nn.Sequential(
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 3)
            )
    
    def forward(self, market_data: Dict[str, Any], features: np.ndarray) -> Tuple[float, float]:
        price = market_data.get('price', 0)
        sma = market_data.get('sma_15', price)
        vol = market_data.get('vol_5m', price * 0.01)
        
        z_score = 0.0
        if vol > 0 and sma > 0:
            z_score = (price - sma) / vol
        
        direction = 0.0
        confidence = 0.2
        
        if abs(z_score) > self.z_threshold:
            direction = -np.sign(z_score) * min(abs(z_score) / 3, 1.0)
            confidence = min(abs(z_score) / 4 + 0.3, 1.0)
        elif abs(z_score) > 1.5:
            direction = -np.sign(z_score) * min(abs(z_score) / 4, 0.6)
            confidence = min(abs(z_score) / 5 + 0.2, 0.8)
        
        correlation = market_data.get('correlation', 0)
        if abs(correlation) > 0.7:
            confidence *= 1.2
        
        regime = market_data.get('regime_label', 1)
        if regime == 1:
            confidence *= 1.3
        
        if HAS_MLX and self.model is not None and len(features) >= 64:
            try:
                mx_features = mx.array(features[:64].reshape(1, -1).astype(np.float32))
                output = self.model(mx_features)
                direction = direction * 0.5 + float(np.tanh(output[0, 1])) * 0.5
            except:
                pass
        
        return self.validate_signal(confidence, direction)
    
    def test_time_adapter(self, batch_data: Dict[str, Any], learning_rate: float = 1e-3) -> None:
        pass
