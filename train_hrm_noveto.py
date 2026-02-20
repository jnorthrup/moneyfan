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
        
        # Shared encoder (smaller for better generalization)
        self.encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # 24 codec simulators (simplified)
        self.codecs = [
            nn.Sequential(
                nn.Linear(64, 16),
                nn.ReLU(),
                nn.Linear(16, 2)  # [confidence, direction]
            ) for _ in range(n_codecs)
        ]
        
        # Trust allocation (who to believe)
        self.trust_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_codecs)
        )
        
        # Regime detection (slow layer)
        self.regime_head = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 4)  # 4 regime classes
        )
        
        # Signal synthesis (combines everything)
        # Input: encoder(64) + trust(24) + regime(4) + weighted_codec_signal(1)
        self.signal_head = nn.Sequential(
            nn.Linear(64 + n_codecs + 4 + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
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
        raw_signal = self.signal_head(signal_in)[:, 0]
        # Scale to [-1, 1] range with learnable amplification
        signal = mx.tanh(raw_signal * 3)  # Amplify before tanh
        
        return {
            'signal': signal,
            'trust': trust,
            'regime': regime,
            'codec_dir': codec_dir,
            'codec_conf': codec_conf,
            'weighted': weighted
        }


def generate_data(n=20000):
    """Load real data from feather files"""
    import pandas as pd
    from pathlib import Path
    
    # Try to load real data
    data_dir = Path("hrm/data/public_binance")
    feather_files = list(data_dir.glob("*.feather"))
    
    if feather_files:
        # Load 5m data (reasonable size)
        f = [f for f in feather_files if "5m" in f.name]
        if f:
            df = pd.read_feather(f[0])
            print(f"Loaded real data from {f[0]}: {len(df)} rows")
            
            prices = df['close'].values.astype(np.float32)
            returns = np.diff(prices, prepend=prices[0]) / prices
            
            n = min(len(df), n)
            
            # Features (64-dim from available columns)
            features = np.zeros((n, 64), dtype=np.float32)
            
            # Use existing computed features
            feature_cols = ['open', 'high', 'low', 'close', 'volume', 'rsi_14', 
                           'macd', 'macd_hist', 'bb_upper', 'bb_lower', 'atr_14', 
                           'adx_14', 'ob_imbalance', 'spread_pct', 'vol_5m',
                           'returns_1m', 'returns_5m', 'returns_15m', 'returns_1h']
            
            for i, col in enumerate(feature_cols[:20]):
                if col in df.columns:
                    val = df[col].values[:n].astype(np.float32)
                    val = np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
                    features[:, i] = val
            
            # Add more features
            if 'sma_5' in df.columns and 'sma_15' in df.columns:
                sma5 = np.nan_to_num(df['sma_5'].values[:n], nan=1.0)
                sma15 = np.nan_to_num(df['sma_15'].values[:n], nan=1.0)
                features[:, 20] = np.where(sma15 != 0, (sma5 / sma15 - 1) * 100, 0)
            
            if 'bid_price' in df.columns and 'ask_price' in df.columns:
                bid = np.nan_to_num(df['bid_price'].values[:n], nan=prices[:n])
                ask = np.nan_to_num(df['ask_price'].values[:n], nan=prices[:n])
                features[:, 21] = np.where(prices[:n] != 0, (ask - bid) / prices[:n] * 10000, 0)
            
            # Normalize
            mean = np.nan_to_num(features.mean(axis=0), nan=0.0)
            std = np.nan_to_num(features.std(axis=0), nan=1.0) + 1e-8
            features = np.nan_to_num((features - mean) / std, nan=0.0)
            
            # Target: use returns_5m if available, else compute
            if 'returns_5m' in df.columns:
                target_signal = np.nan_to_num(df['returns_5m'].values[:n].astype(np.float32), nan=0.0)
            else:
                target_signal = np.roll(returns[:n], -5)
            
            # Scale based on actual distribution
            target_std = np.nan_to_num(np.std(target_signal), nan=1.0) + 1e-8
            target_signal = target_signal / target_std  # Normalize to std=1
            target_signal = np.clip(target_signal, -3, 3) / 3  # Scale to [-1, 1]
            target_signal = np.roll(target_signal, -1)  # Predict next step
            target_signal[-1] = 0
            target_signal = np.nan_to_num(target_signal, nan=0.0)
            
            # Regime (simplified: based on return sign)
            target_regime = np.zeros((n, 4), dtype=np.float32)
            for i in range(n):
                if target_signal[i] > 0.3:
                    target_regime[i, 2] = 1  # Up
                elif target_signal[i] < -0.3:
                    target_regime[i, 0] = 1  # Down
                elif abs(returns[i]) > 0.01:
                    target_regime[i, 3] = 1  # Volatile
                else:
                    target_regime[i, 1] = 1  # Sideways
            
            return features, target_signal, target_regime, returns[:n]
    
    # Fallback to synthetic data
    print("Using synthetic data (no real data found)")
    np.random.seed(42)
    
    returns = np.zeros(n)
    regimes = np.zeros(n, dtype=int)
    
    current_regime = 1
    for i in range(n):
        if np.random.rand() < 0.005:
            current_regime = np.random.randint(4)
        regimes[i] = current_regime
        
        if current_regime == 0:
            returns[i] = np.random.randn() * 0.015 - 0.003
        elif current_regime == 1:
            returns[i] = np.random.randn() * 0.005
        elif current_regime == 2:
            returns[i] = np.random.randn() * 0.015 + 0.003
        else:
            returns[i] = np.random.randn() * 0.03
    
    prices = 50000 * np.cumprod(1 + returns)
    
    features = np.zeros((n, 64), dtype=np.float32)
    features[:, 0] = (prices - np.mean(prices)) / np.std(prices)
    features[:, 1] = returns
    
    for lag in range(2, 15):
        features[:, lag] = np.roll(returns, lag-1)
    
    for w in [5, 10, 20]:
        features[:, 15 + w] = np.sqrt(np.convolve(returns**2, np.ones(w)/w, mode='same'))
    
    features[:, 40:] = np.random.randn(n, 24) * 0.1
    
    target_signal = np.roll(returns, -1).astype(np.float32)
    target_signal = np.clip(target_signal * 50, -1, 1)
    
    target_regime = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        target_regime[i, regimes[i]] = 1.0
    
    return features, target_signal, target_regime, returns


def train(n_epochs=100):
    print("="*60)
    print("HRM TRAINING - NO VETO, PURE BACKPROP")
    print("="*60)
    
    # Fixed seed for reproducibility
    np.random.seed(42)
    
    model = HRM()
    optimizer = optim.Adam(learning_rate=5e-4)  # Lower LR for better generalization
    
    features, target_signal, target_regime, returns = generate_data(20000)
    
    # Add noise for regularization
    noise_scale = 0.1
    features_noisy = features + np.random.randn(*features.shape).astype(np.float32) * noise_scale
    
    x = mx.array(features_noisy)
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
        trust_reg = mx.mean(trust_entropy) * 0.05  # Lower weight
        
        # L2 regularization on signal head weights
        l2_reg = mx.array(0.0)
        
        return 0.7 * signal_loss + 0.25 * regime_loss + 0.05 * trust_reg
    
    print(f"\nTraining on {len(features)} samples\n")
    
    grad_fn = mx.value_and_grad(loss_fn)
    
    for epoch in range(n_epochs):
        loss, grads = grad_fn(model)
        optimizer.update(model, grads)
        
        if epoch % 10 == 0:
            out = model(x[:1000])
            sig = out['signal']
            # Manual correlation
            sig_np = np.array(sig)
            y_np = np.array(y_signal[:1000])
            corr = np.corrcoef(sig_np, y_np)[0, 1]
            trust = np.array(out['trust'][0])
            print(f"Epoch {epoch:3d}: loss={float(loss):.4f}, signal_corr={corr:.3f}, trust_top3={trust.argsort()[-3:]}")
    
    return model


def validate(model, n_ticks=2000):
    print("\n" + "="*60)
    print("VALIDATION (Cross-Asset: ETH)")
    print("="*60)
    
    # Load ETH data for cross-asset validation
    import pandas as pd
    from pathlib import Path
    
    data_dir = Path("hrm/data/public_binance")
    feather_files = list(data_dir.glob("*ETH*5m*.feather"))  # Use ETH 5m for validation
    
    if not feather_files:
        feather_files = list(data_dir.glob("*15m*.feather"))  # Fallback to 15m
    
    if feather_files:
        df = pd.read_feather(feather_files[0])
        print(f"Loaded validation data from {feather_files[0]}: {len(df)} rows")
        
        prices = df['close'].values.astype(np.float32)
        n_ticks = min(len(prices), n_ticks)
        
        # Build features same way as training
        features = np.zeros((n_ticks, 64), dtype=np.float32)
        
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'rsi_14', 
                       'macd', 'macd_hist', 'bb_upper', 'bb_lower', 'atr_14', 
                       'adx_14', 'ob_imbalance', 'spread_pct', 'vol_5m',
                       'returns_1m', 'returns_5m', 'returns_15m', 'returns_1h']
        
        for i, col in enumerate(feature_cols[:20]):
            if col in df.columns:
                val = df[col].values[:n_ticks].astype(np.float32)
                val = np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
                features[:, i] = val
        
        if 'sma_5' in df.columns and 'sma_15' in df.columns:
            sma5 = np.nan_to_num(df['sma_5'].values[:n_ticks], nan=1.0)
            sma15 = np.nan_to_num(df['sma_15'].values[:n_ticks], nan=1.0)
            features[:, 20] = np.where(sma15 != 0, (sma5 / sma15 - 1) * 100, 0)
        
        mean = np.nan_to_num(features.mean(axis=0), nan=0.0)
        std = np.nan_to_num(features.std(axis=0), nan=1.0) + 1e-8
        features = np.nan_to_num((features - mean) / std, nan=0.0)
        
        np.random.seed(123)  # For reproducibility
    else:
        # Fallback to synthetic
        np.random.seed(123)
        returns = np.random.randn(n_ticks) * 0.02
        prices = 50000 * np.cumprod(1 + returns)
        features = np.zeros((n_ticks, 64), dtype=np.float32)
        features[:, 0] = (prices - 50000) / 10000
    
    equity = 1000.0
    cash = 1000.0
    position = 0.0
    entry_price = prices[0] if len(prices) > 0 else 50000.0
    trades = []
    
    for i in range(min(n_ticks, len(features))):
        price = prices[i]
        
        x = mx.array(features[i].reshape(1, -1))
        out = model(x)
        
        signal = float(out['signal'])
        trust = np.array(out['trust'][0])
        regime = int(np.argmax(out['regime']))
        
        # Trade based on signal (no veto - backprop learned when to trade)
        if abs(signal) > 0.08:  # Lower threshold
            direction = 1 if signal > 0 else -1
            confidence = abs(signal) * 2  # Scale confidence
            size = equity * 0.02 * min(confidence, 1.5)
            
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
    
    model = train(n_epochs=150)  # More epochs
    equity, trades = validate(model, n_ticks=2000)
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)
