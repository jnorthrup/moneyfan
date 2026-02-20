"""
Codec 11: Sector Rotation
Rotates between sectors based on relative strength.
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


class Codec11(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'sector_rotation'
        super().__init__(config)
        
        if HAS_MLX:
            self.model = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 3)
            )
    
    def forward(self, market_data: Dict[str, Any], features: np.ndarray) -> Tuple[float, float]:
        sector_momentum = market_data.get('sector_momentum', 0)
        relative_strength = market_data.get('relative_strength', 0)
        market_regime = market_data.get('market_regime', 0)
        
        direction = 0.0
        if relative_strength > 0.1:
            direction = sector_momentum * 0.6 + relative_strength * 0.4
        elif relative_strength < -0.1:
            direction = -sector_momentum * 0.4 + relative_strength * 0.6
        
        if market_regime == -1:
            direction *= 0.5
        
        confidence = min(abs(relative_strength) + abs(sector_momentum) * 0.5 + 0.2, 1.0)
        
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
