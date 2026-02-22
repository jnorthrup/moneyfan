"""
Codec 10: Liquidity Making
Provides liquidity and profits from spread.
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


class Codec10(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'liquidity_making'
        super().__init__(config)
        
        self.min_spread = config.get('min_spread', 0.001)
        
        if HAS_MLX:
            self.model = nn.Sequential(
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 3)
            )
    
    def forward(self, market_data: Dict[str, Any], features: np.ndarray) -> Tuple[float, float]:
        spread = market_data.get('spread_pct', 0)
        volatility = market_data.get('vol_5m', 0)
        ob_imbalance = market_data.get('ob_imbalance', 0)
        price = market_data.get('price', 0)
        vwap = market_data.get('vwap', price)
        
        if spread < self.min_spread:
            return self.validate_signal(0.1, 0.0)
        
        direction = -ob_imbalance * 0.5
        
        if vwap > 0:
            vwap_dev = (price - vwap) / vwap
            direction -= np.sign(vwap_dev) * min(abs(vwap_dev) * 5, 0.5)
        
        vol_penalty = min(volatility * 10, 0.5) if volatility > 0.02 else 0
        confidence = min(spread * 100 - vol_penalty, 1.0)
        
        regime = market_data.get('regime_label', 1)
        if regime == 1:
            confidence *= 1.3
        else:
            confidence *= 0.6
        
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
    
    def online_adapter(self, *args, **kwargs) -> None:
        pass

