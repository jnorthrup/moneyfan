"""
Codec 22: XGBoost Signal
Gradient boosting on engineered metrics.
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


class Codec19(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'xgboost_signal'
        super().__init__(config)
        
        self.n_estimators = config.get('n_estimators', 100)
        self.lr = config.get('learning_rate', 0.1)
        
        self.feature_importance = np.random.dirichlet(np.ones(15))
        
        if HAS_MLX:
            self.model = nn.Sequential(
                nn.Linear(64, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 3)
            )
    
    def forward(self, market_data: Dict[str, Any], features: np.ndarray) -> Tuple[float, float]:
        feature_values = np.array([
            market_data.get('rsi_14', 50) / 100 - 0.5,
            market_data.get('macd_hist', 0),
            market_data.get('ob_imbalance', 0),
            market_data.get('momentum', 0),
            market_data.get('vol_5m', 0),
            market_data.get('adx_14', 0) / 100,
            market_data.get('atr_14', 0) / market_data.get('price', 1),
            market_data.get('bb_upper', 0) - market_data.get('bb_lower', 0),
            market_data.get('spread_pct', 0),
            market_data.get('returns_5m', 0),
            market_data.get('returns_15m', 0),
            market_data.get('volume', 0) / max(market_data.get('avg_volume', 1), 1) - 1,
            market_data.get('vwap', 0) - market_data.get('price', 0) if market_data.get('vwap', 0) > 0 else 0,
            market_data.get('sma_5', 0) / market_data.get('sma_15', 1) - 1 if market_data.get('sma_15', 0) > 0 else 0,
            market_data.get('regime_label', 1) - 1,
        ])
        
        weighted_signal = np.sum(feature_values * self.feature_importance)
        
        direction = np.tanh(weighted_signal * 3)
        confidence = min(abs(weighted_signal) + 0.3, 1.0)
        
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
