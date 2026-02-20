"""
Codec 12: Composite Alpha
Combines multiple alpha sources into unified signal.
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


class Codec12(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'composite_alpha'
        super().__init__(config)
        
        if HAS_MLX:
            self.model = nn.Sequential(
                nn.Linear(64, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 3)
            )
    
    def forward(self, market_data: Dict[str, Any], features: np.ndarray) -> Tuple[float, float]:
        signals = []
        
        momentum = market_data.get('momentum', 0)
        if abs(momentum) > 0.01:
            signals.append(momentum)
        
        rsi = market_data.get('rsi_14', 50)
        rsi_signal = (50 - rsi) / 50
        if abs(rsi_signal) > 0.2:
            signals.append(rsi_signal * 0.5)
        
        ob_imbalance = market_data.get('ob_imbalance', 0)
        if abs(ob_imbalance) > 0.1:
            signals.append(ob_imbalance * 0.3)
        
        macd = market_data.get('macd', 0)
        macd_signal = market_data.get('macd_signal', 0)
        macd_hist = macd - macd_signal
        if abs(macd_hist) > 0:
            signals.append(np.sign(macd_hist) * min(abs(macd_hist) * 10, 0.5))
        
        if not signals:
            return self.validate_signal(0.2, 0.0)
        
        direction = np.mean(signals)
        confidence = min(np.std(signals) * 2 + abs(direction) + 0.3, 1.0)
        
        if HAS_MLX and self.model is not None and len(features) >= 64:
            try:
                mx_features = mx.array(features[:64].reshape(1, -1).astype(np.float32))
                output = self.model(mx_features)
                ml_dir = float(np.tanh(output[0, 1]))
                ml_conf = float(mx.sigmoid(output[0, 0]))
                direction = direction * 0.4 + ml_dir * 0.6
                confidence = confidence * 0.5 + ml_conf * 0.5
            except:
                pass
        
        return self.validate_signal(confidence, direction)
    
    def test_time_adapter(self, batch_data: Dict[str, Any], learning_rate: float = 1e-3) -> None:
        pass
