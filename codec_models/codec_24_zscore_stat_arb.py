import numpy as np
from typing import Dict, Any
from .base_codec import BaseExpert

class Codec24ZScoreStatArb(BaseExpert):
    """
    Expert 24: zscore_stat_arb
    Multi-asset z-score mean-reversion baskets.
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
    def forward(self, context) -> Dict[str, float]:
        if "close" not in context or len(context["close"]) < 10:
            return {'signal_conviction': 0.0, 'direction': 0.0, 'regime_fit': 0.0}
            
        recent = np.array(context["close"][-10:])
        mean_val = float(np.mean(recent))
        std_val = float(np.std(recent))
        
        current = float(recent[-1])
        z_score = (current - mean_val) / (std_val + 1e-8)
        
        # Mean reversion: fade the z-score
        direction = -1.0 if z_score > 1.5 else 1.0 if z_score < -1.5 else 0.0
        
        return {
            'signal_conviction': min(1.0, abs(z_score) / 3.0) if direction != 0 else 0.0,
            'direction': direction,
            'regime_fit': 0.75
        }
        
    def online_adapter(self, *args, **kwargs) -> None:
        pass
