"""
HRM Training - No Veto, Pure Backprop
=====================================
Backprop synthesizes best response from all codec signals.
No manual veto - let the model learn when to trade.
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


class HRM(nn.Module):
    """
    HRM Architecture (no veto):
    1. Shared encoder
    2. 24 codec simulators  
    3. Trust allocation (learned via backprop)
    4. Regime detection
    5. Signal synthesis
    """
    
    def __init__(self, n_codecs=24):
        super().__init__()
        self.n_codecs = n_codecs
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # 24 codec simulators
        self.codecs = [
            nn.Sequential(
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 2)  # [confidence, direction]
            ) for _ in range(n_codecs)
        ]
        
        # Trust allocation (who to believe)
        self.trust_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_codecs)
        )
        
        # Regime detection (slow layer)
        self.regime_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # 4 regime classes
        )
        
        # Signal synthesis (combines everything)
        # Input: encoder(128) + trust(24) + regime(4) + weighted_codec_signal(1)
        self.signal_head = nn.Sequential(
            nn.Linear(128 + n_codecs + 4 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def __call__(self, x):
        # Encode
        h = self.encoder(x)
        
        # Get codec outputs
        codec_conf = []
        codec_dir = []
        for codec in self.codecs:
            out = codec(h)
            codec_conf.append(mx.sigmoid(out[:, 0]))
            codec_dir.append(mx.tanh(out[:, 1]))
        
        # Trust weights
        trust = mx.softmax(self.trust_head(h), axis=-1)
        
        # Regime
        regime = mx.softmax(self.regime_head(h), axis=-1)
        
        # Weighted codec signal (trust * direction)
        weighted = sum(
            trust[:, i] * codec_dir[i] * codec_conf[i]
            for i in range(self.n_codecs)
        )
        
        # Signal synthesis (backprop learns optimal combination)
        signal_in = mx.concatenate([h, trust, regime, weighted.reshape(-1, 1)], axis=-1)
        signal = mx.tanh(self.signal_head(signal_in)[:, 0])
        
        return {
            'signal': signal,
            'trust': trust,
            'regime': regime,
            'codec_dir': codec_dir,
            'codec_conf': codec_conf,
            'weighted': weighted
        }


def generate_data(n=20000):
    """Generate training data with regime structure"""
    np.random.seed(42)
    
    # Simulate regimes
    returns = np.zeros(n)
    regimes = np.zeros(n, dtype=int)
    
    current_regime = 1
    for i in range(n):
        # Regime switching
        if np.random.rand() < 0.005:
            current_regime = np.random.randint(4)
        regimes[i] = current_regime
        
        # Regime-dependent returns
        if current_regime == 0:  # Down
            returns[i] = np.random.randn() * 0.015 - 0.003
        elif current_regime == 1:  # Sideways
            returns[i] = np.random.randn() * 0.005
        elif current_regime == 2:  # Up
            returns[i] = np.random.randn() * 0.015 + 0.003
        else:  # Volatile
            returns[i] = np.random.randn() * 0.03
    
    prices = 50000 * np.cumprod(1 + returns)
    
    # Features (64-dim)
    features = np.zeros((n, 64), dtype=np.float32)
    features[:, 0] = (prices - np.mean(prices)) / np.std(prices)
    features[:, 1] = returns
    
    # Lagged returns
    for lag in range(2, 15):
        features[:, lag] = np.roll(returns, lag-1)
    
    # Rolling stats
    for w in [5, 10, 20]:
        features[:, 15 + w] = np.sqrt(np.convolve(returns**2, np.ones(w)/w, mode='same'))
    
    # Random features for remaining
    features[:, 40:] = np.random.randn(n, 24) * 0.1
    
    # Target: next return (scaled)
    target_signal = np.roll(returns, -1).astype(np.float32)
    target_signal = np.clip(target_signal * 25, -1, 1)
    
    # Regime target
    target_regime = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        target_regime[i, regimes[i]] = 1.0
    
    return features, target_signal, target_regime, returns


def train(n_epochs=100):
    print("="*60)
    print("HRM TRAINING - NO VETO, PURE BACKPROP")
    print("="*60)
    
    model = HRM()
    optimizer = optim.Adam(learning_rate=1e-3)
    
    features, target_signal, target_regime, returns = generate_data(20000)
    
    x = mx.array(features)
    y_signal = mx.array(target_signal)
    y_regime = mx.array(target_regime)
    
    def loss_fn(model):
        out = model(x)
        
        # Signal prediction loss
        signal_loss = mx.mean((out['signal'] - y_signal)**2)
        
        # Regime classification loss
        regime_loss = mx.mean(-mx.sum(y_regime * mx.log(out['regime'] + 1e-8), axis=-1))
        
        # Trust should be concentrated on best-performing codecs
        # (encourages specialization)
        trust_entropy = -mx.sum(out['trust'] * mx.log(out['trust'] + 1e-8), axis=-1)
        trust_reg = mx.mean(trust_entropy) * 0.1
        
        return 0.6 * signal_loss + 0.3 * regime_loss + 0.1 * trust_reg
    
    print(f"\nTraining on {len(features)} samples\n")
    
    grad_fn = mx.value_and_grad(loss_fn)
    
    for epoch in range(n_epochs):
        loss, grads = grad_fn(model)
        optimizer.update(model, grads)
        
        if epoch % 10 == 0:
            out = model(x[:1000])
            sig = out['signal']
            corr = float(mx.corrcoef(mx.stack([sig, y_signal[:1000]]))[0, 1])
            trust = np.array(out['trust'][0])
            print(f"Epoch {epoch:3d}: loss={float(loss):.4f}, signal_corr={corr:.3f}, trust_top3={trust.argsort()[-3:]}")
    
    return model


def validate(model, n_ticks=2000):
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)
    
    np.random.seed(123)
    
    # Realistic price series
    returns = np.random.randn(n_ticks) * 0.02
    prices = 50000 * np.cumprod(1 + returns)
    
    equity = 1000.0
    cash = 1000.0
    position = 0.0
    entry_price = 50000.0
    trades = []
    
    for i in range(n_ticks):
        price = prices[i]
        
        # Features
        f = np.zeros(64, dtype=np.float32)
        f[0] = (price - 50000) / 10000
        f[1] = returns[i]
        
        x = mx.array(f.reshape(1, -1))
        out = model(x)
        
        signal = float(out['signal'])
        trust = np.array(out['trust'][0])
        regime = int(np.argmax(out['regime']))
        
        # Trade based on signal (no veto - backprop learned when to trade)
        if abs(signal) > 0.15:
            direction = 1 if signal > 0 else -1
            confidence = abs(signal)
            size = equity * 0.02 * confidence
            
            if position == 0:
                position = direction * size
                cash -= abs(size)
                entry_price = price
            elif np.sign(position) != direction:
                # Close
                pnl = (price - entry_price) / entry_price * position
                cash += abs(position) + pnl
                trades.append(pnl)
                
                # Open new
                position = direction * size
                cash -= abs(size)
                entry_price = price
        
        equity = cash + abs(position)
        
        if i % 400 == 0:
            print(f"  Tick {i}: Equity=${equity:.2f}, Regime={regime}, Signal={signal:.2f}")
    
    # Close final
    if position != 0:
        pnl = (prices[-1] - entry_price) / entry_price * position
        cash += abs(position) + pnl
        trades.append(pnl)
    
    equity = cash
    
    print(f"\n{'='*40}")
    print(f"Final Equity: ${equity:.2f}")
    print(f"Return: {(equity - 1000) / 10:.2f}%")
    print(f"Trades: {len(trades)}")
    
    if trades:
        wins = [t for t in trades if t > 0]
        print(f"Win Rate: {len(wins)/len(trades)*100:.1f}%")
        if len(trades) > 5:
            sharpe = np.mean(trades) / (np.std(trades) + 1e-8) * np.sqrt(252)
            print(f"Sharpe: {sharpe:.2f}")
    
    return equity, trades


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MONEYFAN HRM - NO VETO")
    print("="*60 + "\n")
    
    model = train(n_epochs=100)
    equity, trades = validate(model, n_ticks=2000)
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)
