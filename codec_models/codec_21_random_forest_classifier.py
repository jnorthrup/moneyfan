"""
Codec 21: Random Forest Classifier
Ensemble of tree-based feature signals.
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


class Codec21(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'random_forest_classifier'
        super().__init__(config)
        
        self.n_trees = config.get('n_trees', 10)
        self.tree_weights = np.random.dirichlet(np.ones(self.n_trees))
        
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
        
        rsi = market_data.get('rsi_14', 50)
        signals.append((rsi - 50) / 50)
        
        macd_hist = market_data.get('macd_hist', 0)
        signals.append(np.sign(macd_hist) * min(abs(macd_hist) * 5, 1.0))
        
        ob_imbalance = market_data.get('ob_imbalance', 0)
        signals.append(ob_imbalance)
        
        momentum = market_data.get('momentum', 0)
        signals.append(np.sign(momentum) * min(abs(momentum) * 10, 1.0))
        
        sma_5 = market_data.get('sma_5', 0)
        sma_15 = market_data.get('sma_15', 0)
        if sma_15 > 0:
            signals.append(np.sign(sma_5 - sma_15))
        
        bb_upper = market_data.get('bb_upper', 0)
        bb_lower = market_data.get('bb_lower', 0)
        price = market_data.get('price', 0)
        if bb_upper > bb_lower:
            bb_pos = (price - bb_lower) / (bb_upper - bb_lower)
            signals.append((bb_pos - 0.5) * 2)
        
        weighted_votes = np.array(signals) * self.tree_weights[:len(signals)]
        direction = np.sum(weighted_votes)
        
        confidence = min(np.std(signals) * 2 + abs(direction) * 0.5 + 0.3, 1.0)
        
        if HAS_MLX and self.model is not None and len(features) >= 64:
            try:
                mx_features = mx.array(features[:64].reshape(1, -1).astype(np.float32))
                output = self.model(mx_features)
                ml_dir = float(np.tanh(output[0, 1]))
                ml_conf = float(mx.sigmoid(output[0, 0]))
                direction = direction * 0.4 + ml_dir * 0.6
                confidence = confidence * 0.4 + ml_conf * 0.6
            except:
                pass
        
        return self.validate_signal(confidence, direction)
    
    def test_time_adapter(self, batch_data: Dict[str, Any], learning_rate: float = 1e-3) -> None:
        pass
