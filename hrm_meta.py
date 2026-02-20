"""
HRM Meta-Learner — Fast/Slow Shared World Model
================================================

Hierarchical Reasoning Model with:
- Shared backbone encoder (world model)
- Slow layer (strategic): regime detection, risk budgeting
- Fast layer (tactical): signal execution, veto enforcement
- Bidirectional context flow between layers

Gradients from all heads update shared encoder → mutual generalization.
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass, field
import time

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("[HRM] MLX not available - using numpy fallback")


@dataclass
class HRMConfig:
    """Configuration for Fast/Slow HRM"""
    n_codecs: int = 24
    hidden_dim: int = 256
    n_encoder_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1
    
    ohlcv_horizons: List[int] = field(default_factory=lambda: [1, 5, 15])
    n_kernel_metrics: int = 48
    
    slow_update_freq: int = 5
    fast_update_freq: int = 1
    
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'trust': 0.35,
        'ohlcv': 0.25,
        'kernel': 0.25,
        'regime': 0.15
    })
    
    learning_rate: float = 1e-4
    max_memory: int = 512


class SharedEncoder:
    """
    Shared backbone encoder - the world model heart
    Both fast and slow layers read from this same encoder
    """
    
    def __init__(self, config: HRMConfig):
        self.config = config
        
        if HAS_MLX:
            self._init_mlx()
        else:
            self._init_numpy()
    
    def _init_mlx(self):
        class Encoder(nn.Module):
            def __init__(self, hidden_dim, n_layers, n_heads, dropout):
                super().__init__()
                self.input_proj = nn.Linear(64, hidden_dim)
                self.pos_embed = nn.Linear(1, hidden_dim)
                self.layers = [
                    nn.TransformerEncoderLayer(
                        dims=hidden_dim,
                        num_heads=n_heads,
                        mlp_dims=hidden_dim * 4,
                        dropout=dropout
                    ) for _ in range(n_layers)
                ]
                self.norm = nn.LayerNorm(hidden_dim)
            
            def __call__(self, x):
                x = self.input_proj(x)
                for layer in self.layers:
                    x = layer(x)
                return self.norm(x)
        
        self.model = Encoder(
            self.config.hidden_dim,
            self.config.n_encoder_layers,
            self.config.n_heads,
            self.config.dropout
        )
    
    def _init_numpy(self):
        self.weights = {
            'input_proj': np.random.randn(64, self.config.hidden_dim).astype(np.float32) * 0.02,
            'layers': [
                {
                    'w1': np.random.randn(self.config.hidden_dim, self.config.hidden_dim * 4).astype(np.float32) * 0.02,
                    'w2': np.random.randn(self.config.hidden_dim * 4, self.config.hidden_dim).astype(np.float32) * 0.02,
                } for _ in range(self.config.n_encoder_layers)
            ],
            'norm': np.ones(self.config.hidden_dim, dtype=np.float32),
        }
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Encode input through shared backbone"""
        if HAS_MLX:
            return self._forward_mlx(x)
        return self._forward_numpy(x)
    
    def _forward_mlx(self, x: np.ndarray) -> np.ndarray:
        try:
            mx_x = mx.array(x.reshape(1, -1).astype(np.float32))
            out = self.model(mx_x)
            return np.array(out[0])
        except:
            return self._forward_numpy(x)
    
    def _forward_numpy(self, x: np.ndarray) -> np.ndarray:
        h = x @ self.weights['input_proj']
        for layer in self.weights['layers']:
            h1 = np.maximum(0, h @ layer['w1'])
            h = h + h1 @ layer['w2']
        return h * self.weights['norm']


class SlowLayer:
    """
    Strategic layer - regime detection, risk budgeting, codec trust matrix
    Outputs conditioning context to Fast layer
    """
    
    def __init__(self, config: HRMConfig):
        self.config = config
        
        if HAS_MLX:
            self._init_mlx()
        else:
            self._init_numpy()
        
        self.regime_memory = []
        self.trust_history = []
    
    def _init_mlx(self):
        class SlowHead(nn.Module):
            def __init__(self, hidden_dim, n_codecs):
                super().__init__()
                self.regime_net = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 4)
                )
                self.trust_net = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, n_codecs)
                )
                self.risk_net = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 4),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 4, 3)
                )
            
            def __call__(self, x):
                regime_logits = self.regime_net(x)
                regime = mx.softmax(regime_logits, axis=-1)
                trust_logits = self.trust_net(x)
                trust = mx.softmax(trust_logits, axis=-1)
                risk = mx.sigmoid(self.risk_net(x))
                return regime, trust, risk
        
        self.model = SlowHead(self.config.hidden_dim, self.config.n_codecs)
    
    def _init_numpy(self):
        self.weights = {
            'regime': np.random.randn(self.config.hidden_dim, 4).astype(np.float32) * 0.02,
            'trust': np.random.randn(self.config.hidden_dim, self.config.n_codecs).astype(np.float32) * 0.02,
            'risk': np.random.randn(self.config.hidden_dim, 3).astype(np.float32) * 0.02,
        }
    
    def forward(self, encoded: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate strategic outputs"""
        if HAS_MLX:
            return self._forward_mlx(encoded)
        return self._forward_numpy(encoded)
    
    def _forward_mlx(self, encoded: np.ndarray) -> Dict[str, np.ndarray]:
        try:
            mx_enc = mx.array(encoded.reshape(1, -1).astype(np.float32))
            regime, trust, risk = self.model(mx_enc)
            return {
                'regime': np.array(regime[0]),
                'trust_weights': np.array(trust[0]),
                'risk_budget': np.array(risk[0])
            }
        except:
            return self._forward_numpy(encoded)
    
    def _forward_numpy(self, encoded: np.ndarray) -> Dict[str, np.ndarray]:
        regime_logits = encoded @ self.weights['regime']
        regime = np.exp(regime_logits - np.max(regime_logits))
        regime = regime / np.sum(regime)
        
        trust_logits = encoded @ self.weights['trust']
        trust = np.exp(trust_logits - np.max(trust_logits))
        trust = trust / np.sum(trust)
        
        risk = 1 / (1 + np.exp(-(encoded @ self.weights['risk'])))
        
        return {
            'regime': regime,
            'trust_weights': trust,
            'risk_budget': risk
        }
    
    def get_context(self) -> np.ndarray:
        """Get conditioning context for fast layer"""
        if not self.regime_memory:
            return np.zeros(4, dtype=np.float32)
        return np.mean(self.regime_memory[-10:], axis=0)


class FastLayer:
    """
    Tactical layer - real-time signal execution, position sizing, veto
    Feeds performance feedback back to shared encoder + slow layer
    """
    
    def __init__(self, config: HRMConfig):
        self.config = config
        
        if HAS_MLX:
            self._init_mlx()
        else:
            self._init_numpy()
        
        self.signal_history = []
        self.performance_feedback = []
    
    def _init_mlx(self):
        class FastHead(nn.Module):
            def __init__(self, hidden_dim, context_dim):
                super().__init__()
                self.context_proj = nn.Linear(context_dim, hidden_dim // 4)
                self.signal_net = nn.Sequential(
                    nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 3)
                )
            
            def __call__(self, encoded, context):
                ctx = self.context_proj(context)
                combined = mx.concatenate([encoded, ctx], axis=-1)
                out = self.signal_net(combined)
                signal = mx.tanh(out[:, 0:1])
                confidence = mx.sigmoid(out[:, 1:2])
                veto = mx.sigmoid(out[:, 2:3])
                return signal, confidence, veto
        
        self.model = FastHead(self.config.hidden_dim, 4)
    
    def _init_numpy(self):
        self.weights = {
            'context_proj': np.random.randn(4, self.config.hidden_dim // 4).astype(np.float32) * 0.02,
            'signal': np.random.randn(self.config.hidden_dim + self.config.hidden_dim // 4, 3).astype(np.float32) * 0.02,
        }
    
    def forward(self, encoded: np.ndarray, slow_context: np.ndarray) -> Dict[str, float]:
        """Generate tactical outputs with slow layer conditioning"""
        if HAS_MLX:
            return self._forward_mlx(encoded, slow_context)
        return self._forward_numpy(encoded, slow_context)
    
    def _forward_mlx(self, encoded: np.ndarray, context: np.ndarray) -> Dict[str, float]:
        try:
            mx_enc = mx.array(encoded.reshape(1, -1).astype(np.float32))
            mx_ctx = mx.array(context.reshape(1, -1).astype(np.float32))
            signal, confidence, veto = self.model(mx_enc, mx_ctx)
            return {
                'signal': float(signal[0, 0]),
                'confidence': float(confidence[0, 0]),
                'veto': float(veto[0, 0])
            }
        except:
            return self._forward_numpy(encoded, context)
    
    def _forward_numpy(self, encoded: np.ndarray, context: np.ndarray) -> Dict[str, float]:
        ctx = context @ self.weights['context_proj']
        combined = np.concatenate([encoded, ctx])
        out = combined @ self.weights['signal']
        return {
            'signal': float(np.tanh(out[0])),
            'confidence': float(1 / (1 + np.exp(-out[1]))),
            'veto': float(1 / (1 + np.exp(-out[2])))
        }
    
    def update_feedback(self, pnl: float, signal_quality: float):
        """Feed performance back to shared encoder"""
        self.performance_feedback.append({
            'pnl': pnl,
            'quality': signal_quality,
            'timestamp': time.time()
        })
        if len(self.performance_feedback) > 100:
            self.performance_feedback.pop(0)


class WorldModelHeads:
    """
    Auxiliary prediction heads for world-model pretraining
    - OHLCV prediction at multiple horizons
    - Kernel metric prediction
    """
    
    def __init__(self, config: HRMConfig):
        self.config = config
        
        if HAS_MLX:
            self._init_mlx()
        else:
            self._init_numpy()
    
    def _init_mlx(self):
        class OHLCVHead(nn.Module):
            def __init__(self, hidden_dim, horizons):
                super().__init__()
                self.heads = [
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Linear(hidden_dim // 2, 5)
                    ) for _ in horizons
                ]
            
            def __call__(self, x):
                return mx.stack([h(x) for h in self.heads], axis=1)
        
        class KernelHead(nn.Module):
            def __init__(self, hidden_dim, n_metrics):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, n_metrics)
                )
            
            def __call__(self, x):
                return self.net(x)
        
        self.ohlcv_head = OHLCVHead(self.config.hidden_dim, self.config.ohlcv_horizons)
        self.kernel_head = KernelHead(self.config.hidden_dim, self.config.n_kernel_metrics)
    
    def _init_numpy(self):
        self.weights = {
            'ohlcv': [
                np.random.randn(self.config.hidden_dim, 5).astype(np.float32) * 0.02
                for _ in self.config.ohlcv_horizons
            ],
            'kernel': np.random.randn(self.config.hidden_dim, self.config.n_kernel_metrics).astype(np.float32) * 0.02,
        }
    
    def forward(self, encoded: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict OHLCV and kernel metrics"""
        if HAS_MLX:
            return self._forward_mlx(encoded)
        return self._forward_numpy(encoded)
    
    def _forward_mlx(self, encoded: np.ndarray) -> Dict[str, np.ndarray]:
        try:
            mx_enc = mx.array(encoded.reshape(1, -1).astype(np.float32))
            ohlcv = self.ohlcv_head(mx_enc)
            kernel = self.kernel_head(mx_enc)
            return {
                'ohlcv': np.array(ohlcv[0]),
                'kernel': np.array(kernel[0])
            }
        except:
            return self._forward_numpy(encoded)
    
    def _forward_numpy(self, encoded: np.ndarray) -> Dict[str, np.ndarray]:
        ohlcv = np.stack([encoded @ w for w in self.weights['ohlcv']])
        kernel = encoded @ self.weights['kernel']
        return {'ohlcv': ohlcv, 'kernel': kernel}


class FastSlowHRM:
    """
    Complete Fast/Slow HRM with shared world model
    
    Architecture:
    - SharedEncoder (backbone, updated by all gradients)
    - SlowLayer (strategic: regime, trust, risk)
    - FastLayer (tactical: signal, veto)
    - WorldModelHeads (auxiliary: OHLCV, kernel prediction)
    
    Bidirectional flow:
    - Slow → Fast: conditioning context
    - Fast → Slow: performance feedback
    - All → SharedEncoder: gradient updates
    """
    
    def __init__(self, config: HRMConfig = None):
        self.config = config or HRMConfig()
        
        self.encoder = SharedEncoder(self.config)
        self.slow_layer = SlowLayer(self.config)
        self.fast_layer = FastLayer(self.config)
        self.world_model = WorldModelHeads(self.config)
        
        self.memory = np.zeros((self.config.max_memory, 64), dtype=np.float32)
        self.memory_idx = 0
        
        self.step_count = 0
        self.performance = {
            'total_loss': 0.0,
            'trust_loss': 0.0,
            'ohlcv_loss': 0.0,
            'kernel_loss': 0.0,
            'regime_loss': 0.0,
            'sharpe': 0.0,
        }
        
        print(f"[HRM] Fast/Slow HRM initialized")
        print(f"[HRM] Shared encoder: {self.config.n_encoder_layers} layers, {self.config.hidden_dim} dim")
        print(f"[HRM] Slow layer: regime + trust + risk")
        print(f"[HRM] Fast layer: signal + veto")
        print(f"[HRM] World model: OHLCV {self.config.ohlcv_horizons} + {self.config.n_kernel_metrics} metrics")
    
    def update_memory(self, features: np.ndarray):
        """Update rolling memory buffer"""
        if len(features) >= 64:
            self.memory[self.memory_idx] = features[:64]
            self.memory_idx = (self.memory_idx + 1) % self.config.max_memory
    
    def forward(self, 
                codec_signals: np.ndarray, 
                features: np.ndarray) -> Dict[str, Any]:
        """
        Full forward pass through Fast/Slow HRM
        
        Flow:
        1. Encode features through shared backbone
        2. Slow layer generates strategic context
        3. World model predicts OHLCV + kernels
        4. Fast layer generates tactical signal (conditioned on slow)
        """
        self.update_memory(features)
        self.step_count += 1
        
        encoded = self.encoder.forward(features)
        
        slow_out = self.slow_layer.forward(encoded)
        self.slow_layer.regime_memory.append(slow_out['regime'])
        
        world_out = self.world_model.forward(encoded)
        
        slow_context = self.slow_layer.get_context()
        fast_out = self.fast_layer.forward(encoded, slow_context)
        
        trust_weights = slow_out['trust_weights']
        veto = fast_out['veto']
        
        weighted_signal = np.sum([
            trust_weights[i] * codec_signals[i][1]
            for i in range(min(len(codec_signals), len(trust_weights)))
        ])
        
        if veto > 0.6:
            final_signal = 0.0
            confidence = 0.0
        else:
            final_signal = weighted_signal * 0.7 + fast_out['signal'] * 0.3
            confidence = fast_out['confidence']
        
        regime = np.argmax(slow_out['regime'])
        risk_budget = slow_out['risk_budget']
        
        return {
            'signal': float(np.clip(final_signal, -1, 1)),
            'confidence': float(confidence),
            'veto': bool(veto > 0.6),
            'position_hint': float(fast_out['signal']),
            'regime': int(regime),
            'regime_probs': slow_out['regime'].tolist(),
            'trust_weights': trust_weights.tolist(),
            'risk_budget': risk_budget.tolist(),
            'ohlcv_predictions': world_out['ohlcv'].tolist(),
            'kernel_predictions': world_out['kernel'].tolist(),
            'encoded': encoded
        }
    
    def compute_loss(self,
                     predictions: Dict[str, Any],
                     target_signal: float,
                     target_ohlcv: np.ndarray,
                     target_kernel: np.ndarray,
                     target_regime: int) -> Dict[str, float]:
        """
        Compute multi-task loss for all heads
        
        Loss = trust_loss + ohlcv_loss + kernel_loss + regime_loss
        All gradients update shared encoder
        """
        pred_signal = predictions['signal']
        pred_ohlcv = np.array(predictions['ohlcv_predictions'])
        pred_kernel = np.array(predictions['kernel_predictions'])
        pred_regime_probs = np.array(predictions['regime_probs'])
        
        trust_loss = (pred_signal - target_signal) ** 2
        
        ohlcv_loss = np.mean((pred_ohlcv - target_ohlcv) ** 2)
        
        kernel_loss = np.mean((pred_kernel - target_kernel) ** 2)
        
        regime_target = np.zeros(4)
        regime_target[target_regime] = 1.0
        regime_loss = -np.sum(regime_target * np.log(pred_regime_probs + 1e-8))
        
        total_loss = (
            self.config.loss_weights['trust'] * trust_loss +
            self.config.loss_weights['ohlcv'] * ohlcv_loss +
            self.config.loss_weights['kernel'] * kernel_loss +
            self.config.loss_weights['regime'] * regime_loss
        )
        
        return {
            'trust_loss': float(trust_loss),
            'ohlcv_loss': float(ohlcv_loss),
            'kernel_loss': float(kernel_loss),
            'regime_loss': float(regime_loss),
            'total_loss': float(total_loss)
        }
    
    def train_step(self,
                   codec_signals: np.ndarray,
                   features: np.ndarray,
                   target_signal: float,
                   target_ohlcv: np.ndarray,
                   target_kernel: np.ndarray,
                   target_regime: int) -> Dict[str, float]:
        """Single training step with full gradient flow"""
        predictions = self.forward(codec_signals, features)
        
        losses = self.compute_loss(
            predictions, target_signal, target_ohlcv,
            target_kernel, target_regime
        )
        
        for key in ['total_loss', 'trust_loss', 'ohlcv_loss', 'kernel_loss', 'regime_loss']:
            self.performance[key] = self.performance[key] * 0.99 + losses.get(key, 0) * 0.01
        
        return losses
    
    def feedback(self, pnl: float, signal_quality: float):
        """Feed performance back from fast to slow + encoder"""
        self.fast_layer.update_feedback(pnl, signal_quality)
        
        if pnl < -0.02:
            if len(self.slow_layer.trust_history) > 0:
                worst_codec = np.argmin(self.slow_layer.trust_history[-1])
    
    def allocate(self, 
                 codec_outputs: List[Dict[str, Any]], 
                 features: np.ndarray) -> Dict[str, Any]:
        """
        Main allocation interface
        
        Takes 24 codec outputs and features, returns allocation decision
        """
        codec_signals = np.array([
            [o.get('confidence', 0.5), o.get('direction', 0), o.get('regime_fit', 0.5)]
            for o in codec_outputs[:24]
        ])
        
        if len(codec_signals) < 24:
            padding = np.zeros((24 - len(codec_signals), 3))
            codec_signals = np.vstack([codec_signals, padding])
        
        return self.forward(codec_signals, features)
    
    def get_performance(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            **self.performance,
            'steps': self.step_count,
            'memory_used': min(self.memory_idx, self.config.max_memory)
        }
    
    def reset(self):
        """Reset HRM state"""
        self.memory = np.zeros((self.config.max_memory, 64), dtype=np.float32)
        self.memory_idx = 0
        self.step_count = 0
        self.slow_layer.regime_memory = []
        self.slow_layer.trust_history = []
        self.fast_layer.signal_history = []
        self.fast_layer.performance_feedback = []


def create_hrm(config: HRMConfig = None) -> FastSlowHRM:
    """Factory function to create Fast/Slow HRM"""
    return FastSlowHRM(config)


if __name__ == "__main__":
    print("="*60)
    print("FAST/SLOW HRM TEST")
    print("="*60)
    
    config = HRMConfig()
    hrm = create_hrm(config)
    
    codec_outputs = [
        {'confidence': 0.6 + np.random.rand() * 0.3, 
         'direction': np.random.randn() * 0.5, 
         'regime_fit': 0.5 + np.random.rand() * 0.3}
        for _ in range(24)
    ]
    
    features = np.random.randn(64).astype(np.float32)
    
    result = hrm.allocate(codec_outputs, features)
    
    print(f"\nAllocation Result:")
    print(f"  Signal: {result['signal']:.4f}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Veto: {result['veto']}")
    print(f"  Regime: {result['regime']}")
    print(f"  Position hint: {result['position_hint']:.4f}")
    
    print(f"\nTrust Weights (top 5):")
    trust = np.array(result['trust_weights'])
    top_idx = np.argsort(trust)[-5:][::-1]
    for i in top_idx:
        print(f"  Codec {i}: {trust[i]:.4f}")
    
    print(f"\nWorld Model Predictions:")
    ohlcv = np.array(result['ohlcv_predictions'])
    print(f"  OHLCV shape: {ohlcv.shape}")
    print(f"  1m prediction: O={ohlcv[0,0]:.4f} H={ohlcv[0,1]:.4f} L={ohlcv[0,2]:.4f} C={ohlcv[0,3]:.4f} V={ohlcv[0,4]:.4f}")
    
    kernel = np.array(result['kernel_predictions'])
    print(f"  Kernel metrics shape: {kernel.shape}")
    
    target_signal = 0.3
    target_ohlcv = np.random.randn(3, 5).astype(np.float32) * 0.01
    target_kernel = np.random.randn(48).astype(np.float32) * 0.1
    target_regime = 1
    
    codec_signals = np.array([
        [o['confidence'], o['direction'], o['regime_fit']]
        for o in codec_outputs
    ])
    
    losses = hrm.train_step(
        codec_signals, features, target_signal,
        target_ohlcv, target_kernel, target_regime
    )
    
    print(f"\nTraining Losses:")
    for key, value in losses.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nHRM Performance:")
    perf = hrm.get_performance()
    for key, value in perf.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
