"""
Codec 07: Volume Profile
Analyzes volume distribution to find support/resistance.
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


class Codec07(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'volume_profile'
        super().__init__(config)
        
        if HAS_MLX:
            self.model = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 3)
            )
    
    def forward(self, market_data: Dict[str, Any], features: np.ndarray) -> Tuple[float, float]:
        volume = market_data.get('volume', 0)
        avg_volume = market_data.get('avg_volume', volume)
        vwap = market_data.get('vwap', market_data.get('price', 0))
        price = market_data.get('price', 0)
        
        vol_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        vwap_signal = 0.0
        if vwap > 0:
            vwap_deviation = (price - vwap) / vwap
            vwap_signal = -np.sign(vwap_deviation) * min(abs(vwap_deviation) * 10, 1.0)
        
        volume_signal = 0.0
        if vol_ratio > 2.0:
            momentum = market_data.get('momentum', 0)
            volume_signal = np.sign(momentum) * min(vol_ratio / 5, 1.0)
        
        direction = vwap_signal * 0.5 + volume_signal * 0.5
        confidence = min((vol_ratio - 1) * 0.3 + 0.3, 1.0)
        
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
    
    def online_adapter(self, *args, **kwargs) -> None:
        pass

