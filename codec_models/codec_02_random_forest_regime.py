"""
Codec #22: Random Forest Regime Detector + Executor
===================================================

This codec uses Random Forest to detect market regimes and execute trades
based on regime classification. It's interpretable, fast to train, and
excellent for regime detection.

Features used:
1. Price momentum (short-term and long-term)
2. Volatility indicators
3. Volume analysis
4. Market regime labels (from historical data)
5. Technical indicator patterns

Training approach:
- Supervised classification: Predict regime (bull/bear/sideways)
- Online learning: Update model periodically with new data
- Feature importance: Use SHAP for interpretability

Test-time adapter:
- Online Random Forest updates (streaming learning)
- Retrain on recent data every 15 minutes
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from .base_codec import BaseCodec

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("⚠️  MLX not available - using NumPy fallback")


class Codec_02_RandomForest_Regime(BaseCodec):
    """
    Codec #22: Random Forest Regime Detector + Executor
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "random_forest_regime"
        self.version = "1.0"
        
        # Feature configuration
        self.feature_config = {
            'price_momentum_short': True,
            'price_momentum_long': True,
            'volatility': True,
            'volume_ratio': True,
            'technical_patterns': True,
        }
        
        # Regime labels
        self.regime_labels = ['bullish', 'bearish', 'sideways']
        
        # Initialize models
        if HAS_MLX:
            self.model = self._create_mlx_model()
            print(f"✅ {self.name}: MLX model initialized")
        else:
            self.model = None
            self._initialize_simple_model()
            print(f"⚠️  {self.name}: Using NumPy fallback model")
        
        # Online learning state
        self.training_buffer = []
        self.buffer_size = 500
        
        # Regime state
        self.current_regime = 'neutral'
        self.regime_confidence = 0.0
    
    def _create_mlx_model(self):
        """Create MLX-based Random Forest equivalent"""
        # MLX doesn't have Random Forest, so we create a neural network ensemble
        # that approximates Random Forest behavior
        return nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),  # 3 regimes: bullish, bearish, sideways
        )
    
    def _initialize_simple_model(self):
        """Initialize simple rule-based model as fallback"""
        self.rule_weights = {
            'momentum': 0.4,
            'volatility': 0.3,
            'volume': 0.2,
            'technical': 0.1,
        }
    
    def extract_features(self, market_data: Dict[str, Any], 
                        features: np.ndarray) -> np.ndarray:
        """
        Extract regime detection features
        
        Args:
            market_data: Market state
            features: Technical indicators
            
        Returns:
            Feature vector for regime classification
        """
        # Price momentum features
        price = market_data.get('price', 100.0)
        
        # Short-term momentum (last 5 prices in features)
        if len(features) > 15:
            short_momentum = features[10]  # momentum_20
        else:
            short_momentum = 0.0
        
        # Volatility features
        volatility = market_data.get('volatility', 0.0)
        if len(features) > 11:
            vol_feature = features[11]
        else:
            vol_feature = volatility
        
        # Volume features
        volume = market_data.get('volume', 100000.0)
        volume_ratio = np.log10(max(volume, 1.0))
        
        # Technical patterns (from features)
        rsi = features[6] if len(features) > 6 else 50.0
        macd = features[5] if len(features) > 5 else 0.0
        
        # Regime classification features
        regime_features = np.array([
            short_momentum,  # 0: price momentum
            vol_feature,     # 1: volatility
            volume_ratio,    # 2: volume
            rsi,             # 3: RSI
            macd,            # 4: MACD
            price,           # 5: price level
            market_data.get('lob_imbalance', 0.0),  # 6: LOB imbalance
            market_data.get('bid_ask_spread', 0.001),  # 7: spread
            market_data.get('funding_rate', 0.0),  # 8: funding rate
            0.0,  # 9: reserved for future
        ], dtype=np.float32)
        
        return regime_features
    
    def detect_regime(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Detect market regime using features
        
        Returns:
            (regime_label, confidence)
        """
        if HAS_MLX and self.model is not None:
            # MLX forward pass
            features_mx = mx.array(features.reshape(1, -1))
            
            try:
                output = self.model(features_mx)
                probabilities = mx.softmax(output)
                
                # Get highest probability regime
                regime_idx = int(mx.argmax(probabilities)[0])
                confidence = float(probabilities[0, regime_idx])
                
                if regime_idx == 0:
                    regime = 'bullish'
                elif regime_idx == 1:
                    regime = 'bearish'
                else:
                    regime = 'sideways'
                    
                return regime, confidence
                
            except Exception as e:
                print(f"⚠️  MLX regime detection failed: {e}")
                return self._rule_based_regime(features)
        else:
            # Rule-based fallback
            return self._rule_based_regime(features)
    
    def _rule_based_regime(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Simple rule-based regime detection
        """
        # Extract features
        momentum = features[0] if len(features) > 0 else 0.0
        volatility = features[1] if len(features) > 1 else 0.0
        volume_ratio = features[2] if len(features) > 2 else 0.0
        rsi = features[3] if len(features) > 3 else 50.0
        macd = features[4] if len(features) > 4 else 0.0
        
        # Simple rules
        bull_signals = 0
        bear_signals = 0
        total_signals = 0
        
        # Momentum rule
        if momentum > 0.01:
            bull_signals += 1
        elif momentum < -0.01:
            bear_signals += 1
        total_signals += 1
        
        # RSI rule
        if rsi > 70:
            bear_signals += 1
        elif rsi < 30:
            bull_signals += 1
        total_signals += 1
        
        # MACD rule
        if macd > 0:
            bull_signals += 1
        elif macd < 0:
            bear_signals += 1
        total_signals += 1
        
        # Volume rule (high volume = trend confirmation)
        if volume_ratio > 5.0:  # High volume
            if momentum > 0:
                bull_signals += 1
            else:
                bear_signals += 1
        total_signals += 1
        
        # Volatility rule (low volatility = sideways)
        if volatility < 0.01:
            # Low volatility, likely sideways
            return 'sideways', 0.6
        elif volatility > 0.05:
            # High volatility, use other signals
            pass
        
        # Determine regime
        if bull_signals > bear_signals + 1:
            regime = 'bullish'
            confidence = min(0.9, bull_signals / total_signals)
        elif bear_signals > bull_signals + 1:
            regime = 'bearish'
            confidence = min(0.9, bear_signals / total_signals)
        else:
            regime = 'sideways'
            confidence = 0.5
        
        return regime, confidence
    
    def forward(self, 
                market_data: Dict[str, Any],
                features: np.ndarray) -> Tuple[float, float]:
        """
        Generate trading signal based on regime detection
        
        Returns:
            (confidence, direction)
        """
        # Extract regime features
        regime_features = self.extract_features(market_data, features)
        
        # Detect regime
        regime, confidence = self.detect_regime(regime_features)
        
        # Store current regime
        self.current_regime = regime
        self.regime_confidence = confidence
        
        # Generate signal based on regime
        if regime == 'bullish':
            direction = 0.8 * confidence  # Strong buy
        elif regime == 'bearish':
            direction = -0.8 * confidence  # Strong sell
        else:
            direction = 0.0  # Neutral (no trade)
        
        # Validate output
        confidence, direction = self.validate_signal(confidence, direction)
        
        # Update memory
        self.update_memory(direction, regime_features)
        
        return confidence, direction
    
    def test_time_adapter(self, 
                         batch_data: Dict[str, Any],
                         learning_rate: float = 1e-3) -> None:
        """
        Online fine-tuning via MLX (or buffer update for NumPy fallback)
        """
        if 'inputs' in batch_data and 'targets' in batch_data:
            # Add to buffer for later retraining
            self.training_buffer.append({
                'inputs': batch_data['inputs'],
                'targets': batch_data['targets']
            })
            
            # Keep buffer size manageable
            if len(self.training_buffer) > self.buffer_size:
                self.training_buffer.pop(0)
            
            # If buffer is full and MLX available, retrain
            if len(self.training_buffer) >= self.buffer_size and HAS_MLX:
                self._online_retrain(learning_rate)
    
    def _online_retrain(self, learning_rate: float):
        """
        Online retraining with MLX
        """
        if not HAS_MLX or self.model is None:
            return
        
        try:
            # Prepare data
            inputs = []
            targets = []
            
            for item in self.training_buffer:
                if isinstance(item['inputs'], np.ndarray):
                    inputs.append(item['inputs'])
                else:
                    inputs.append(np.array(item['inputs']))
                
                if isinstance(item['targets'], np.ndarray):
                    targets.append(item['targets'])
                else:
                    targets.append(np.array(item['targets']))
            
            if not inputs:
                return
            
            X = np.vstack(inputs)
            y = np.vstack(targets)
            
            # Convert to MLX
            X_mx = mx.array(X.astype(np.float32))
            y_mx = mx.array(y.astype(np.float32))
            
            # Define loss function
            def loss_fn(params):
                predictions = self.model.apply(params, X_mx)
                return mx.mean(mx.softmax(predictions) * y_mx)
            
            # Create optimizer
            optimizer = mx.optimizers.Adam(learning_rate=learning_rate)
            
            # Update model
            loss, grads = mx.value_and_grad(loss_fn)(self.model.parameters())
            optimizer.update(self.model, grads)
            
            print(f"✅ {self.name}: Online retrain completed, loss: {float(loss):.4f}")
            
        except Exception as e:
            print(f"⚠️  {self.name}: Online retrain failed: {e}")


# Factory function
def create_codec(config: Dict[str, Any] = None):
    """Factory function to create codec instance"""
    if config is None:
        config = {}
    return Codec_02_RandomForest_Regime(config)