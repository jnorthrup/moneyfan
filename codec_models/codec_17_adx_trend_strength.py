"""
Codec 17: ADX Trend Strength
Directional movement + ADX power filter.
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


class Codec17(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'adx_trend_strength'
        super().__init__(config)
        
        self.adx_threshold = config.get('adx_threshold', 25)
        
        if HAS_MLX:
            self.model = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 3)
            )
    
    def forward(self, market_data: Dict[str, Any], features: np.ndarray) -> Tuple[float, float]:
        adx = market_data.get('adx_14', 0)
        plus_di = market_data.get('plus_di', 0)
        minus_di = market_data.get('minus_di', 0)
        
        direction = 0.0
        confidence = 0.2
        
        if adx > self.adx_threshold:
            di_diff = plus_di - minus_di
            direction = np.sign(di_diff) * min(abs(di_diff) / 20, 1.0)
            confidence = min(adx / 50 + 0.3, 1.0)
        else:
            confidence = 0.2
            direction = 0.0
        
        momentum = market_data.get('momentum', 0)
        if np.sign(direction) == np.sign(momentum) and adx > 30:
            confidence = min(confidence * 1.3, 1.0)
        
        if HAS_MLX and self.model is not None and len(features) >= 64:
            try:
                mx_features = mx.array(features[:64].reshape(1, -1).astype(np.float32))
                output = self.model(mx_features)
                if adx > self.adx_threshold:
                    direction = direction * 0.6 + float(np.tanh(output[0, 1])) * 0.4
            except:
                pass
        
        return self.validate_signal(confidence, direction)
    
    def test_time_adapter(self, batch_data: Dict[str, Any], learning_rate: float = 1e-3) -> None:
        pass
    
    def online_adapter(self, *args, **kwargs) -> None:
        pass

