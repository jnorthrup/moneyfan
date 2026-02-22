import numpy as np
from typing import Dict, Any
from .base_codec import BaseExpert

class Codec23TransformerAttention(BaseExpert):
    """
    Expert 23: transformer_attention
    Self-attention simulated over multi-timeframe bar patches.
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
    def forward(self, context) -> Dict[str, float]:
        if "close" not in context or len(context["close"]) < 5:
            return {'signal_conviction': 0.0, 'direction': 0.0, 'regime_fit': 0.0}
            
        recent = np.array(context["close"][-5:])
        weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3]) # Simulated attention map
        
        attended_val = float(np.sum(recent * weights))
        last_close = float(recent[-1])
        
        direction = 1.0 if attended_val > last_close else -1.0
        conviction = min(1.0, float(np.abs(attended_val - last_close) / last_close) * 100)
        
        return {
            'signal_conviction': conviction,
            'direction': direction,
            'regime_fit': 0.9
        }
        
    def online_adapter(self, *args, **kwargs) -> None:
        pass
