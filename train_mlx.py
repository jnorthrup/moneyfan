"""
Real MLX Training - Simplified
"""

import numpy as np
import sys

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
except ImportError:
    print("MLX not available!")
    sys.exit(1)


class FullModel(nn.Module):
    """Combined codec + HRM model"""
    
    def __init__(self, n_codecs=24):
        super().__init__()
        self.n_codecs = n_codecs
        
        # Shared feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        # Trust prediction
        self.trust_layer = nn.Linear(128, n_codecs)
        
        # Signal prediction
        self.signal_layer = nn.Linear(128 + n_codecs, 1)
        
        # World model
        self.ohlcv_layer = nn.Linear(128, 15)
        self.kernel_layer = nn.Linear(128, 48)
    
    def __call__(self, features):
        h = self.encoder(features)
        
        trust = mx.softmax(self.trust_layer(h), axis=-1)
        
        signal_in = mx.concatenate([h, trust], axis=-1)
        signal = mx.tanh(self.signal_layer(signal_in)[:, 0])
        
        ohlcv = self.ohlcv_layer(h)
        kernel = self.kernel_layer(h)
        
        return trust, signal, ohlcv, kernel


def generate_data(n=2000):
    np.random.seed(42)
    
    # Generate price series
    returns = np.random.randn(n) * 0.02
    prices = 50000 * np.cumprod(1 + returns)
    
    features = np.random.randn(n, 64).astype(np.float32)
    features[:, 0] = (prices - 50000) / 50000  # Normalized price
    features[:, 1] = returns  # Current return
    features[:, 2] = np.roll(returns, -1)  # Future return (target info)
    
    # Target: next return direction
    target_signal = np.roll(returns, -1).astype(np.float32) * 10  # Scaled
    target_signal = np.clip(target_signal, -1, 1)
    
    target_ohlcv = np.random.randn(n, 15).astype(np.float32) * 0.01
    target_kernel = features.copy()[:, :48]  # Use first 48 features as kernel target
    
    return features, target_signal, target_ohlcv, target_kernel


def train():
    print("="*60)
    print("MLX TRAINING")
    print("="*60)
    
    model = FullModel()
    optimizer = optim.Adam(learning_rate=1e-3)
    
    features, target_signal, target_ohlcv, target_kernel = generate_data()
    
    features_mx = mx.array(features)
    signal_mx = mx.array(target_signal)
    ohlcv_mx = mx.array(target_ohlcv)
    kernel_mx = mx.array(target_kernel)
    
    def loss_fn(model):
        trust, signal, ohlcv, kernel = model(features_mx)
        
        signal_loss = mx.mean((signal - signal_mx)**2)
        ohlcv_loss = mx.mean((ohlcv - ohlcv_mx)**2)
        kernel_loss = mx.mean((kernel - kernel_mx)**2)
        
        return 0.5 * signal_loss + 0.25 * ohlcv_loss + 0.25 * kernel_loss
    
    print(f"\nTraining on {len(features)} samples")
    
    # Create gradient function for model
    grad_fn = mx.value_and_grad(loss_fn)
    
    for epoch in range(20):
        loss, grads = grad_fn(model)
        optimizer.update(model, grads)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1}: loss = {float(loss):.4f}")
    
    print(f"Final loss = {float(loss):.4f}")
    
    return model


def validate(model, n_ticks=500):
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)
    
    np.random.seed(123)
    
    equity = 1000.0
    position = 0.0
    entry = 50000.0
    trades = []
    
    prices = 50000 * np.cumprod(1 + np.random.randn(n_ticks) * 0.02)
    
    for i in range(n_ticks):
        price = prices[i]
        
        # Features
        f = np.random.randn(64).astype(np.float32)
        f[0] = (price - 50000) / 50000
        
        features = mx.array(f.reshape(1, -1))
        
        trust, signal, _, _ = model(features)
        
        sig = float(signal) * 3  # Scale up signal
        
        if abs(sig) > 0.15:  # Lower threshold
            direction = 1 if sig > 0 else -1
            
            if position == 0:
                position = direction * equity * 0.02
                entry = price
            elif np.sign(position) != direction:
                pnl = (price - entry) / entry * position
                equity += pnl
                trades.append(pnl)
                position = direction * equity * 0.02
                entry = price
        
        if i % 100 == 0:
            print(f"  Tick {i}: Equity=${equity:.2f}, Signal={sig:.3f}")
    
    if position != 0:
        pnl = (prices[-1] - entry) / entry * position
        equity += pnl
        trades.append(pnl)
    
    print(f"\nFinal Equity: ${equity:.2f}")
    print(f"Return: {(equity - 1000) / 1000 * 100:.2f}%")
    print(f"Trades: {len(trades)}")
    
    if trades:
        wins = [t for t in trades if t > 0]
        print(f"Win Rate: {len(wins)/len(trades)*100:.1f}%")


if __name__ == "__main__":
    model = train()
    validate(model)
