"""
Codec 23: Transformer Attention
Self-attention on multi-timeframe patches.
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


class Codec23(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'transformer_attention'
        super().__init__(config)
        
        self.n_heads = config.get('n_heads', 4)
        self.d_model = config.get('d_model', 64)
        
        if HAS_MLX:
            self.model = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 3)
            )
    
    def _self_attention(self, x: np.ndarray) -> np.ndarray:
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        d_k = x.shape[-1]
        scores = x @ x.T / np.sqrt(d_k)
        attention = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention = attention / np.sum(attention, axis=-1, keepdims=True)
        return attention @ x
    
    def forward(self, market_data: Dict[str, Any], features: np.ndarray) -> Tuple[float, float]:
        if len(features) >= 64:
            attended = self._self_attention(features[:64])
            pooled = np.mean(attended, axis=0) if len(attended.shape) > 1 else attended
        else:
            pooled = features
        
        direction = np.tanh(np.sum(pooled[:16]) * 0.5)
        confidence = min(np.std(pooled) * 2 + 0.3, 1.0)
        
        trend_features = [
            market_data.get('returns_5m', 0),
            market_data.get('returns_15m', 0),
            market_data.get('returns_1h', 0),
        ]
        trend_signal = np.mean(trend_features)
        direction = direction * 0.7 + np.sign(trend_signal) * min(abs(trend_signal) * 10, 0.3)
        
        if HAS_MLX and self.model is not None and len(features) >= 64:
            try:
                mx_features = mx.array(features[:64].reshape(1, -1).astype(np.float32))
                output = self.model(mx_features)
                ml_dir = float(np.tanh(output[0, 1]))
                ml_conf = float(mx.sigmoid(output[0, 0]))
                direction = direction * 0.3 + ml_dir * 0.7
                confidence = confidence * 0.3 + ml_conf * 0.7
            except:
                pass
        
        return self.validate_signal(confidence, direction)
    
    def test_time_adapter(self, batch_data: Dict[str, Any], learning_rate: float = 1e-3) -> None:
        pass
