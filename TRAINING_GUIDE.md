# Training Guide: Making the Hierarchical Codec a Killer System

## Current Problem

The codec is trained with:
1. **Pre-train**: MSE on signal prediction → learns signal structure
2. **Fine-tune**: `-mean(pred_return * confidence * actual_return)` → naive profit maximization

This fails because:
- Predicting returns ≠ making money
- No risk adjustment (Sharpe, Sortino)
- No position sizing (Kelly)
- No sequential compounding
- No regime awareness

---

## Reward Functions for Trading ML

### 1. Sharpe Reward (Risk-Adjusted Returns)
```
reward = mean(returns) / std(returns)
```
- Pro: Risk-adjusted, standard metric
- Con: Assumes normal distribution, penalizes both upside and downside volatility

### 2. Sortino Reward (Downside Risk Only)
```
reward = mean(returns) / downside_std
```
- Pro: Only penalizes losses
- Con: Can encourage tail risk

### 3. Differential Sharpe (Online)
```
dS_t = (r_t - S_{t-1}) / (N * sigma_t)
```
- Pro: Online updating, suitable for streaming
- Con: Sensitive to initialization

### 4. Kelly Criterion (Optimal Position Sizing)
```
f* = (p * b - q) / b  where b = win/loss ratio
reward = log(1 + f * r_t)
```
- Pro: Optimal long-term growth
- Con: Can suggest extreme positions

### 5. Profit Factor Reward
```
reward = sum(positive_returns) / sum(negative_returns)
```
- Pro: Simple, intuitive
- Con: Doesn't account for magnitude

### 6. Utility-Based Reward (Risk Aversion)
```
reward = -exp(-lambda * wealth_change)
```
- Pro: Accounts for risk aversion
- Con: Requires tuning lambda

### 7. Calmar Ratio (Drawdown-Adjusted)
```
reward = annual_return / max_drawdown
```
- Pro: Penalizes catastrophic losses
- Con: Slow to update

---

## How to Train a Killer Codec

### Phase 1: Pre-training (Signal Understanding)
**Goal**: Learn signal structure, correlations, regimes

**Current**: MSE on signal prediction
**Better**: Contrastive learning on regime clusters

```python
# Instead of: loss = MSE(pred, target)
# Use: loss = contrastive_loss(signal_embedding, regime_label)
```

This teaches the model to recognize:
- Trend regime signals cluster together
- Mean reversion signals cluster together
- Volatility signals cluster together

### Phase 2: Policy Training (Reward Maximization)
**Goal**: Maximize risk-adjusted returns

**Current**: `-mean(pred * conf * return)`
**Better**: Policy gradient with Sharpe reward

```python
# Reinforcement learning approach
log_prob = model.get_action_prob(signal_state)
action = model.act(signal_state)
reward = sharpe(returns_from_action)
loss = -log_prob * reward  # Policy gradient
```

### Phase 3: Regime-Conditioned Training
**Goal**: Learn when to trust which signals

**Approach**: Train separate heads for each regime

```python
regime = classify_regime(signals)  # trend/mean_reversion/volatility
alpha = regime_head[regime](signals)
loss = -sharpe(alpha * actual_returns)
```

---

## Training Loop Improvements

### 1. Curriculum Learning
Start easy, increase difficulty:
- Epoch 1-5: Only clear trends (high Sharpe periods)
- Epoch 6-10: Add mean reversion periods
- Epoch 11-15: Add choppy/sideways periods
- Epoch 16+: Full stochastic bag

### 2. Hard Negative Mining
Focus on where the model fails:
```python
# Track losses per regime
regime_losses = {trend: [], mean_reversion: [], ...}
# Sample more from high-loss regimes
sample_weights = 1 / (regime_losses + epsilon)
```

### 3. Ensemble Self-Play
Train multiple codec instances:
```python
# N codecs with different seeds
codecs = [Codec(seed=i) for i in range(N)]
# Each trains on others' weaknesses
for codec in codecs:
    hard_samples = get_hard_samples(other_codecs)
    codec.train(hard_samples)
```

### 4. Reward Shaping
Guide the model toward good behavior:
```python
# Base reward
reward = sharpe(returns)

# Bonus for correct direction
reward += 0.1 * (pred_direction == actual_direction)

# Bonus for high confidence on correct predictions
reward += 0.05 * confidence * correct_direction

# Penalty for overtrading
reward -= 0.01 * abs(position_change)

# Penalty for high drawdown periods
if current_drawdown > 0.1:
    reward -= 0.1
```

---

## Key Insight: Position Sizing > Direction Prediction

The current codec predicts direction. But in trading:

```
Profit = Direction_Accuracy × Position_Size × Return_Magnitude
```

A model with 51% accuracy can be profitable with proper position sizing.
A model with 70% accuracy can lose money with poor position sizing.

**Train the codec to output position size, not just direction:**

```python
# Instead of: output = [return_prediction, confidence]
# Use: output = [position_size]  # ∈ [-1, 1]

# Loss = -sharpe(position_size * actual_returns)
# This directly optimizes for risk-adjusted profit
```

---

## Practical Training Schedule

### Week 1: Foundation
- Pre-train on 30 pairs, 100 epochs
- Curriculum: start with high-volatility periods
- Reward: MSE on signal prediction (understanding)

### Week 2: Direction
- Fine-tune on direction prediction
- Reward: `-cross_entropy(pred_direction, actual_direction)`
- Target: 55%+ direction accuracy

### Week 3: Position Sizing
- Train position sizing head
- Reward: `-sharpe(position * returns)`
- Target: Positive Sharpe on test set

### Week 4: Regime Awareness
- Add regime classification head
- Reward: `-regime_loss + sharpe_reward`
- Target: Beat all baseline signals

### Week 5+: Refinement
- Hard negative mining
- Ensemble training
- Online learning on live data

---

## Metrics to Track

| Metric | Current | Target | Killer Level |
|--------|---------|--------|--------------|
| Direction Acc | 45% | 55% | 60%+ |
| Sharpe | 0.02 | 0.5 | 1.0+ |
| Sortino | - | 0.7 | 1.5+ |
| Max DD | - | 10% | 5% |
| Profit Factor | - | 1.2 | 1.5+ |
| Win Rate | - | 55% | 60%+ |

---

## Summary: How to Make It a Killer

1. **Change the objective** from predicting returns to maximizing Sharpe
2. **Add position sizing** as the primary output (not just direction)
3. **Use curriculum learning** to start with easy regimes
4. **Add regime conditioning** so the model knows when to trust which signals
5. **Train longer** with more data and hard negative mining

The key insight: **A killer trading system isn't about predicting the future perfectly - it's about position sizing and risk management.**

The codec should learn:
- "In high volatility, reduce position size"
- "When signals disagree, lower confidence"
- "In trend regime, trust momentum signals"
- "After a drawdown, reduce exposure"
