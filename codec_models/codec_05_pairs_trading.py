"""
Codec 05: Pairs Trading
Statistical arbitrage between correlated assets.
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


class Codec05(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'pairs_trading'
        super().__init__(config)
        
        self.z_entry = config.get('z_entry', 2.0)
        
        if HAS_MLX:
            self.model = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 3)
            )
    
    def forward(self, market_data: Dict[str, Any], features: np.ndarray) -> Tuple[float, float]:
        spread = market_data.get('spread_pct', 0)
        correlation = market_data.get('correlation', 0)
        
        signal = 0.0
        if abs(correlation) > 0.7:
            z_spread = spread / 0.001 if spread != 0 else 0
            if abs(z_spread) > self.z_entry:
                signal = -np.sign(z_spread) * min(abs(z_spread) / 4, 1.0)
        
        ob_imbalance = market_data.get('ob_imbalance', 0)
        imbalance_boost = ob_imbalance * 0.3 if abs(ob_imbalance) > 0.2 else 0
        
        direction = signal + imbalance_boost
        confidence = abs(correlation) * 0.5 + abs(signal) * 0.5
        
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
