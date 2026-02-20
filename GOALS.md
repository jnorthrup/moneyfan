# moneyfan — Project Goals

## Core Definition

**Stochastic Bag**: 30 pairs + 1 main currency (USD), $100 randomly distributed for backtesting. Identical $100 capital to all agents for fair comparison.

**Stochastic Extent**: Variable time windows within the stochastic bag. Trade-pairs can have up to 75% missing data (empty candles) to allow new coin issues to profit from online recognition.

**Stochastic Length**: Random length for each training sequence (64-256 time steps).

**Codec**: A signal agent that runs on instrument-metric inputs. Each codec is a trained model that outputs [confidence, direction, regime_fit].

**HRM**: The meta-allocator that learns which codecs to trust and when. HRM has 24 codecs trained for the 24 SOTA crypto strategies.

**Instrument-Metric**: Any calculated indicator like EMA, MACD, RSI, etc. These are the inputs to codecs.

**Agent/Tradebot**: A backtrade automated strategy (SOTA precedent). The 24 agents are the 24 codecs.

## Architecture

for the purposes of clarity an Agent is training a codec in a slot at training time while the codec is trained and running filling that slot in the harness for the higher executive, maybe the fast voice informing the slow voice 


```
Data Flow:
Binance Arrow data (296 pairs) 
  → Stochastic Bag (30 random pairs + USD)
  → Stochastic Extent (75% max missing data allowed)
  → Stochastic Length (64-256 time steps)
  → 24 Instrument-Metric Inputs (EMA, MACD, RSI, etc.)
  → 24 Codec Agents (each trained separately)
  → HRM Meta-Allocator (learns codec composition)
  → Output: Trading decisions per pair
```

## Codec Training (24 Separate Models)

Each of the 24 codecs is trained independently:

1. **Input**: All instrument-metrics (EMA, MACD, RSI, etc.)
2. **Codec Code**: 24 different SOTA crypto strategies
3. **Training**: Each codec learns to output [confidence, direction, regime_fit] for its strategy
4. **Same Inputs**: All codecs get the SAME instrument-metric inputs
5. **Different Outputs**: Each codec outputs its own signal

## HRM Training

HRM is trained AFTER all codecs are trained:

1. **Input**: 24 codec outputs (confidence, direction, regime_fit from each)
2. **Goal**: Learn composition - which codecs to trust and when
3. **Convergence**: ≥ 2 codecs agree on direction with confidence
4. **CPR Loop**: Converge → Perturb → Repeat, keep improvements

## Stochastic Bag Implementation

```
- 30 random pairs from 296 available
- 1 main currency: USD
- $100 identical capital for all agents
- Drop out: Some pairs may be empty (up to 75%)
- Random pair selection every training round
```

## Stochastic Extent & Length

```
- Variable extent: 32-256 time steps per instrument
- Random length per bag: 64-256 total steps
- Missing data: Allowed up to 75% (new coins)
- Scale preservation: Use previous candle as reference
```

## Instrument-Metrics

These are the inputs to ALL codecs:
- EMA (multiple periods)
- MACD
- RSI
- Bollinger Bands
- Volume ratios
- Price positions
- Momentum
- Volatility
- Trend indicators

Each codec can use whatever subset it needs.

## 24 Codecs (SOTA Crypto Strategies)

1. momentum_breakout
2. mean_reversion
3. volatility_regime
4. trend_following
5. pairs_trading
6. grid_trading
7. volume_profile
8. order_flow
9. correlation_trading
10. liquidity_making
11. sector_rotation
12. composite_alpha
13. rsi_reversal
14. bollinger_bands
15. macd_cross
16. atr_breakout
17. tick_momentum
18. dca_baseline
19. technical_ml
20. hrm_mean_reversion
21. volatility_x_momentum
22. mean_reversion_v2
23. sector_rotation_v2
24. composite_trend
 
 #25 HRM A/B A CPU model, B MLX port 

 hyperparemters: not static

## Current Architecture

Hierarchical Codec: H/L nested cycles with sparkline memory (206k params), pre-trades with MSE loss, trades with pred_loss - 0.1*pnl (fractional return) to route plays to ceiling.

Stochastic Training: Seed-based replay (SEED+iteration) with 296 Binance files, 200-candle windows, 75% dropout, $100 bag, per-iteration PnL isolation, outlier logging.

MLX Implementation: Native MLX (preserves architecture) - NO tiling, sequential H/L cycles, cascading sparkline updates, automatic optimization via lazy evaluation and Metal acceleration. 2.78ms for B=4, T=32. 


