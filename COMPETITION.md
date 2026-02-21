# HRM Agent Competition - Watch the Battle!

## 🚀 **Ready to Watch the Action?**

Your HRM system is now set up with a full **24-agent competition framework**! Here's how to watch the agents battle for dominance:

---

## 📋 **Quick Start Commands**

### 1. **Run the Competition**
```bash
python3 agent_competition.py
```

This will simulate 10 rounds of trading with all 24 agents competing. Watch them battle for the top spot!

### 2. **Start the Live Dashboard**
```bash
streamlit run live_dashboard.py
```

This opens an interactive dashboard where you can:
- Watch real-time agent rankings
- See performance charts
- Track individual agent statistics
- Monitor the competition evolution

---

## 🎯 **What You'll See**

### **Agent Lineup (24 Total)**

1. **Momentum Breakout** - Detects acceleration + volume confirmation
2. **Mean Reversion RSI** - Classic overbought/oversold with dynamic thresholds  
3. **Volatility Regime GARCH** - Regime switching via GARCH volatility forecast
4. **Trend Following EMA** - Multi-timeframe EMA stack with slope filter
5. **MACD Crossover** - Signal-line + histogram divergence
6. **Bollinger Bands Squeeze** - Volatility contraction → expansion plays
7. **Stochastic KD** - %K/%D crossover with overbought filter
8. **Ichimoku Cloud** - Full cloud, conversion, base, lagging span signals
9. **ADX Trend Strength** - Directional movement + ADX power filter
10. **CCI Commodity Channel** - Cyclical deviation from statistical mean
11. **Parabolic SAR** - Trailing stop + reversal detection
12. **VWAP Mean Reversion** - Volume-weighted anchor reversion
13. **Order Book Imbalance** - Live depth delta (when exchange feed available)
14. **Kalman Filter Trend** - Adaptive smoothing of price + velocity
15. **ARIMA Predictor** - Statistical time-series forecast (p,d,q tuned per asset)
16. **Hurst Regime** - Long-memory detection for trend vs mean-reversion
17. **Fractal Dimension** - Chaos vs structure classification
18. **Random Forest Classifier** - Ensemble of tree-based feature signals
19. **XGBoost Signal** - Gradient boosting on engineered metrics
20. **LSTM Sequence Predictor** - Recurrent net on normalized candle sequences
21. **Transformer Attention** - Self-attention on multi-timeframe patches
22. **RL DQN Policy** - Deep Q-network tactical execution agent
23. **Pair Correlation Arb** - Statistical arbitrage on coin pairs
24. **Z-Score Stat Arb** - Multi-asset z-score mean-reversion baskets

---

## 📊 **Dashboard Features**

### **Live Rankings**
- **Top 5 Agents**: Watch the leaders battle for dominance
- **Bottom 5 Agents**: See who's struggling and might get eliminated
- **Real-time Updates**: Auto-refresh every 5 seconds

### **Agent Performance**
- **Individual Agent Stats**: Total P&L, Win Rate, Trade Count, Max Drawdown
- **Performance Charts**: Cumulative P&L, Trade Signals, Confidence Levels
- **Specialization Tracking**: See which strategies perform best

### **Competition Metrics**
- **Total P&L**: Overall competition performance
- **Win Rates**: Agent success rates
- **Trade Volume**: Activity levels
- **Market Impact**: How agents affect the market

---

## 🎯 **What to Watch For**

### **Early Rounds**
- **Initial Rankings**: See which agents jump out to early leads
- **Strategy Performance**: Watch different approaches compete
- **Market Adaptation**: How agents adjust to market conditions

### **Mid Competition**
- **Dominance Shifts**: Watch rankings change as agents learn
- **Strategy Evolution**: See which approaches prove most effective
- **Risk Management**: Watch how agents handle drawdowns

### **Final Stages**
- **Final Rankings**: See which agents emerge victorious
- **Performance Analysis**: Understand why winners succeeded
- **Learning Insights**: What the competition reveals about market dynamics

---

## 🔧 **Technical Details**

### **Competition Mechanics**
- **24 Agents**: Each with unique strategy and specialization
- **Real-time Trading**: Simulated market data with realistic conditions
- **Performance Tracking**: P&L, Win Rate, Drawdown, Trade Count
- **Ranking System**: Based on total P&L and risk-adjusted returns

### **HRM Integration**
- **Fast/Slow Layers**: Agents feed into hierarchical reasoning
- **Trust Allocation**: HRM learns which agents to trust
- **Performance Feedback**: Agents get rewarded/punished based on results
- **Dynamic Adaptation**: System evolves based on agent performance

---

## 📈 **Expected Outcomes**

### **Performance Targets**
- **Top Agents**: Should achieve Sharpe >1.8
- **Win Rates**: Target 55%+ for successful agents
- **Risk Management**: Max drawdown <15%
- **Alpha Generation**: Consistent positive returns

### **Hierarchy Validation**
- **Trust Learning**: HRM should learn to trust top performers
- **Strategy Selection**: System should favor effective approaches
- **Risk Adjustment**: Better risk management over time
- **Alpha Amplification**: Hierarchy should boost overall performance

---

## 🎯 **Ready to Watch?**

**Start the Competition:**
```bash
python3 agent_competition.py
```

**Watch Live:**
```bash
streamlit run live_dashboard.py
```

**Enjoy the Battle!** 👹👹👹

Watch as these 24 agents compete for dominance, learn from each other, and evolve into a powerful trading system. The HRM hierarchy will learn to trust the best performers and create a winning combination!

---

**Note**: The competition uses synthetic market data for simulation. For real trading, connect to live market feeds and implement proper risk management.

**Performance Targets**: Sharpe ≥1.8, Max DD ≤15%, Annualized Return >20%