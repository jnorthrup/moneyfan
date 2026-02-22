"""
Codec 08: Order Flow
Analyzes bid/ask dynamics and trade flow.
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


class Codec08(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'order_flow'
        super().__init__(config)
        
        if HAS_MLX:
            self.model = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.Linear(64, 3)
            )
    
    def forward(self, market_data: Dict[str, Any], features: np.ndarray) -> Tuple[float, float]:
        ob_imbalance = market_data.get('ob_imbalance', 0)
        taker_buy = market_data.get('taker_buy_base', 0)
        volume = market_data.get('volume', 1)
        spread = market_data.get('spread_pct', 0)
        
        flow_signal = ob_imbalance
        
        if volume > 0:
            buy_ratio = taker_buy / volume
            flow_signal = flow_signal * 0.5 + (buy_ratio - 0.5) * 0.5
        
        spread_penalty = 0.0
        if spread > 0.002:
            spread_penalty = -0.2
        
        direction = flow_signal + spread_penalty
        confidence = min(abs(ob_imbalance) + 0.3, 1.0)
        
        depth_bid = market_data.get('depth_5_bid', 0)
        depth_ask = market_data.get('depth_5_ask', 0)
        if depth_bid + depth_ask > 0:
            depth_imbalance = (depth_bid - depth_ask) / (depth_bid + depth_ask)
            direction += depth_imbalance * 0.3
        
        if HAS_MLX and self.model is not None and len(features) >= 64:
            try:
                mx_features = mx.array(features[:64].reshape(1, -1).astype(np.float32))
                output = self.model(mx_features)
                direction = direction * 0.4 + float(np.tanh(output[0, 1])) * 0.6
            except:
                pass
        
        return self.validate_signal(confidence, direction)
    
    def test_time_adapter(self, batch_data: Dict[str, Any], learning_rate: float = 1e-3) -> None:
        pass
    
    def online_adapter(self, *args, **kwargs) -> None:
        pass

