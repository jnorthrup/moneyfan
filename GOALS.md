# moneyfan — High-Entropy Goals

## High-Entropy Performance Targets
**20%+ net annualized return, Sharpe ≥ 1.8, MaxDD ≤ 15%**  
**100% turnover cap, 2023-2026 out-of-sample testing**  
**30-day live paper trading + 10K Monte-Carlo slippage simulations**

## High-Entropy Architecture Choices
**Vector Store Replacement**: Replace hyperbolic memory with 64-dim numpy vectors  
**3-Predictor MVP**: 5m Transformer + 15m XGBoost + 1h LightGBM  
**4-Stage HRM Rollout**: EarnHFT-inspired, 80% benefit, debuggable in weekend  
**Predictor/Live Split**: Pure numpy+MLX inference vs pandas training (50ms latency)**

## High-Entropy Risk Controls
**Veto Layer**: HRM high-level rejects trades when regime_confidence < 0.75  
**Portfolio Limits**: 20% max per symbol, 3-5 uncorrelated positions  
**Hard Stops**: 2% loss max, 5% daily drawdown freeze, 3 losing trades → 1hr pause  
**Kelly/Fixed-Fraction**: 1-2% risk per trade, ATR-based stops/targets

## High-Entropy Validation Protocol
**Walk-Forward**: 12-month train, 3-month test, 4 cycles (3 years total)  
**Ablation Test**: Merge vs hierarchy must show ≥0.5 Sharpe improvement  
**Statistical Significance**: 10K random shuffles, p < 0.05 alpha validation  
**Live Paper**: 30 days minimum, real market data with 0.1-0.3% slippage

## High-Entropy Trading Recipe
**Regime Detection**: 15-60min updates, veto threshold 0.75 confidence  
**Entry Threshold**: |aggregated_signal| > 0.3, 1-2% risk per trade  
**Take Profit**: 2-3× ATR or trailing stop, stop loss = 1× ATR or 2% hard stop  
