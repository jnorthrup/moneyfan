"""
Codec #3: XGBoost on Orderbook Imbalance + TA
==============================================

This codec uses XGBoost for regression on orderbook features combined with technical indicators.
It demonstrates the "copy-paste ready" money-making flow for Coinbase.

Features used:
1. Orderbook imbalance (bid vs ask volume)
2. Bid-ask spread
3. Volume ratios
4. TA indicators: EMA, MACD, RSI, Bollinger Bands
5. Price momentum
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from .base_codec import BaseCodec

try:
    import xgboost as xgb
    HAS_XGBOOST = True
    XGBOOST_INSTALLED = True
except ImportError:
    HAS_XGBOOST = False
    XGBOOST_INSTALLED = False
    print("⚠️  XGBoost not available - using MLX fallback")

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("⚠️  MLX not available - using NumPy fallback")


class Codec_03_XGBoost_Orderbook(BaseCodec):
    """
    Codec #3: XGBoost on orderbook imbalance + TA
    
    Training approach:
    - Offline: Train on historical Coinbase data
    - Online: Test-time adaptation with low LR updates
    
    Money-making flow:
    - Input: LOB imbalance + TA features every 5 seconds
    - Output: [confidence, direction] for next 1-5 minutes
    - Reward: Realized PnL - slippage + direction accuracy bonus
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "xgboost_orderbook"
        self.version = "1.0"
        
        # Feature mapping
        self.feature_map = {
            'lob_imbalance': 0,
            'bid_ask_spread': 1,
            'volume_ratio': 2,
            'ema_12': 3,
            'ema_26': 4,
            'macd': 5,
            'rsi': 6,
            'bb_upper': 7,
            'bb_lower': 8,
            'bb_middle': 9,
            'momentum_20': 10,
            'volatility': 11,
            'volume_momentum': 12,
            'price_position': 13,
            'funding_rate': 14,
        }
        
        # Initialize models
        if HAS_XGBOOST:
            self.xgb_model = self._create_xgboost_model()
            print(f"✅ {self.name}: XGBoost model initialized")
        elif HAS_MLX:
            self.mlx_model = self._create_mlx_model()
            print(f"✅ {self.name}: MLX model initialized (XGBoost fallback)")
        else:
            self.mlx_model = None
            print(f"⚠️  {self.name}: Using simple linear model (fallback)")
        
        # Online learning state
        self.online_buffer = []
        self.online_buffer_size = 100
        
    def _create_xgboost_model(self):
        """Create XGBoost regression model"""
        params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'reg:squarederror',
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42,
            'n_jobs': -1,
        }
        
        model = xgb.XGBRegressor(**params)
        return model
    
    def _create_mlx_model(self):
        """Create MLX neural network as fallback"""
        return nn.Sequential(
            nn.Linear(15, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )
    
    def extract_features(self, market_data: Dict[str, Any], 
                        features: np.ndarray) -> np.ndarray:
        """
        Extract and combine features from market data and TA features
        
        Returns:
            Combined feature vector [15]
        """
        # Get LOB features from market_data
        lob_imbalance = market_data.get('lob_imbalance', 0.0)
        bid_ask_spread = market_data.get('bid_ask_spread', 0.001)
        volume = market_data.get('volume', 100000.0)
        price = market_data.get('price', 70000.0)
        funding_rate = market_data.get('funding_rate', 0.0001)
        
        # Calculate volume ratio (if available)
        volume_ratio = np.log10(max(volume, 1.0))
        
        # Ensure features array has at least 15 elements
        if len(features) < 15:
            padded = np.zeros(15, dtype=np.float32)
            padded[:len(features)] = features
            features = padded
        elif len(features) > 15:
            features = features[:15]
        
        # Combine all features
        combined = np.array([
            lob_imbalance,
            bid_ask_spread,
            volume_ratio,
            features[3],  # ema_12
            features[4],  # ema_26
            features[5],  # macd
            features[6],  # rsi
            features[7],  # bb_upper
            features[8],  # bb_lower
            features[9],  # bb_middle
            features[10], # momentum_20
            features[11], # volatility
            features[12], # volume_momentum
            features[13], # price_position
            funding_rate,
        ], dtype=np.float32)
        
        return combined
    
    def forward(self, 
                market_data: Dict[str, Any],
                features: np.ndarray) -> Tuple[float, float]:
        """
        Generate trading signal using XGBoost/MLX model
        
        Returns:
            (confidence, direction) where:
                - confidence: 0-1 signal strength
                - direction: -1 to 1 (negative = sell, positive = buy)
        """
        # Extract features
        input_features = self.extract_features(market_data, features)
        
        # Add batch dimension for MLX
        if HAS_MLX and self.mlx_model is not None:
            features_mx = mx.array(input_features.reshape(1, -1))
            
            try:
                # Forward pass
                if hasattr(self.mlx_model, 'forward'):
                    output = self.mlx_model.forward(features_mx)
                else:
                    output = self.mlx_model(features_mx)
                
                # Extract predictions
                if output.shape == (1, 2):
                    confidence = float(output[0, 0])
                    direction = float(output[0, 1])
                elif output.shape == (1, 1):
                    confidence = 0.5
                    direction = float(output[0, 0])
                else:
                    # Fallback
                    confidence = 0.5
                    direction = 0.0
            except Exception as e:
                print(f"⚠️  MLX forward pass failed: {e}")
                # Fallback to simple logic
                confidence, direction = self._simple_fallback(input_features)
        elif HAS_XGBOOST and hasattr(self, 'xgb_model'):
            # XGBoost prediction
            try:
                prediction = self.xgb_model.predict(input_features.reshape(1, -1))
                if prediction.ndim == 2:
                    confidence = float(prediction[0, 0])
                    direction = float(prediction[0, 1])
                else:
                    confidence = 0.5
                    direction = float(prediction[0])
            except Exception as e:
                print(f"⚠️  XGBoost prediction failed: {e}")
                confidence, direction = self._simple_fallback(input_features)
        else:
            # Simple fallback logic
            confidence, direction = self._simple_fallback(input_features)
        
        # Validate output
        confidence, direction = self.validate_signal(confidence, direction)
        
        # Update memory
        self.update_memory(direction, input_features)
        
        return confidence, direction
    
    def _simple_fallback(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Simple fallback logic when MLX/XGBoost not available
        
        Uses LOB imbalance and basic TA signals
        """
        # Extract key features
        lob_imbalance = features[0]  # bid-ask imbalance
        rsi = features[6] if len(features) > 6 else 50.0
        macd = features[5] if len(features) > 5 else 0.0
        
        # Simple logic:
        # - Positive LOB imbalance + RSI < 30 => BUY
        # - Negative LOB imbalance + RSI > 70 => SELL
        # - MACD positive => BUY signal
        # - MACD negative => SELL signal
        
        confidence_signals = []
        
        if lob_imbalance > 0.1:
            confidence_signals.append(lob_imbalance)
        elif lob_imbalance < -0.1:
            confidence_signals.append(-lob_imbalance)
        
        if rsi < 30:
            confidence_signals.append(0.3)
        elif rsi > 70:
            confidence_signals.append(-0.3)
        
        if macd > 0:
            confidence_signals.append(0.2)
        elif macd < 0:
            confidence_signals.append(-0.2)
        
        if not confidence_signals:
            return 0.0, 0.0
        
        # Average the signals
        direction = np.mean(confidence_signals)
        confidence = abs(direction) + 0.3  # Base confidence + signal strength
        
        return confidence, direction
    
    def test_time_adapter(self, 
                         batch_data: Dict[str, Any],
                         learning_rate: float = 1e-3) -> None:
        """
        Online fine-tuning via MLX value_and_grad
        
        For XGBoost, we can only add data to buffer and retrain periodically
        For MLX model, we can do online gradient updates
        """
        if not HAS_MLX or self.mlx_model is None:
            # For XGBoost or fallback, add to buffer
            if 'inputs' in batch_data and 'targets' in batch_data:
                self.online_buffer.append({
                    'inputs': batch_data['inputs'],
                    'targets': batch_data['targets']
                })
                
                # Keep buffer size manageable
                if len(self.online_buffer) > self.online_buffer_size:
                    self.online_buffer.pop(0)
                
                # Retrain XGBoost if buffer is full
                if len(self.online_buffer) >= self.online_buffer_size and HAS_XGBOOST:
                    self._retrain_xgboost()
            return
        
        # MLX online training
        if 'inputs' in batch_data and 'targets' in batch_data:
            try:
                inputs_mx = mx.array(batch_data['inputs'].astype(np.float32))
                targets_mx = mx.array(batch_data['targets'].astype(np.float32))
                
                # Define loss function
                def loss_fn(params):
                    predictions = self.mlx_model.apply(params, inputs_mx)
                    return mx.mean((predictions - targets_mx) ** 2)
                
                # Create optimizer
                optimizer = optim.Adam(learning_rate=learning_rate)
                
                # Get current parameters and compute gradients
                loss, grads = mx.value_and_grad(loss_fn)(self.mlx_model.parameters())
                
                # Update model parameters
                optimizer.update(self.mlx_model, grads)
                
                print(f"✅ {self.name}: Online update completed, loss: {float(loss):.4f}")
            except Exception as e:
                print(f"⚠️  {self.name}: Online update failed: {e}")
    
    def _retrain_xgboost(self):
        """Retrain XGBoost with buffer data"""
        if not HAS_XGBOOST or not self.online_buffer:
            return
        
        try:
            # Prepare data
            inputs = []
            targets = []
            
            for item in self.online_buffer:
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
            
            # Retrain model
            self.xgb_model.fit(X, y, verbose=False)
            
            print(f"✅ {self.name}: Retrained XGBoost with {len(inputs)} samples")
        except Exception as e:
            print(f"⚠️  {self.name}: Retrain failed: {e}")


# Factory registration (optional)
def create_codec(config: Dict[str, Any] = None):
    """Factory function to create codec instance"""
    if config is None:
        config = {}
    return Codec_03_XGBoost_Orderbook(config)