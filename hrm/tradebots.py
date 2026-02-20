"""
TradeBots - Signal-generating bots that wrap QuantModels

PANDAS -> instruments -> {tradebots} -> HRM IO

Each TradeBot:
- Reads from InstrumentRegistry (lazy pandas)
- Computes signals using QuantModel kernels
- Reports state for HRM selection
- Outputs to HRM IO membrane

TradeBot types:
- MomentumBot: trend-following strategies
- ReversionBot: mean-reversion strategies
- VolatilityBot: volatility breakout (proven winner)
- StatArbBot: pairs trading, bent penny
- CompositeBot: combines other bot signals
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime

from hrm.instruments import LazyInstrument, InstrumentRegistry, QuantModel


class BotType(Enum):
    # From DeFlorio Thesis 2022 - Chapter 2.3.2 Trading Algorithms
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY_BREAKOUT = "volatility_breakout"
    STAT_ARB = "stat_arb"
    COMPOSITE = "composite"
    TREND_FOLLOWING = "trend_following"
    PATTERN_MATCHING = "pattern_matching"
    STOP_LOSS = "stop_loss"
    MOVING_AVERAGE_CROSS = "moving_average_cross"
    HOUR_CHOPPY = "hour_choppy"
    DAILY_MOMENTUM = "daily_momentum"
    STATISTICAL_BREAKOUT = "statistical_breakout"
    DCA_WEIGHTED = "dca_weighted"
    SUPPORT_RESISTANCE = "support_resistance"
    TIME_BASED = "time_based"
    CORRELATION = "correlation"


@dataclass
class BotState:
    """State snapshot for HRM to read"""
    bot_id: str
    bot_type: str
    status: str = "idle"
    last_signal: float = 0.0
    last_return: float = 0.0
    confidence: float = 0.0
    energy: float = 0.0
    pnl: float = 0.0
    trades_count: int = 0
    win_rate: float = 0.5
    signal_count: int = 0
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'bot_id': self.bot_id,
            'model_id': self.bot_id,
            'bot_type': self.bot_type,
            'status': self.status,
            'last_signal': self.last_signal,
            'last_return': self.last_return,
            'confidence': self.confidence,
            'energy': self.energy,
            'pnl': self.pnl,
            'trades_count': self.trades_count,
            'win_rate': self.win_rate,
            'signal_count': self.signal_count,
            'timestamp': self.timestamp,
        }


@dataclass
class TradeBot:
    """
    A trading bot that generates signals from instruments.
    
    Wraps a QuantModel and adds:
    - Position tracking
    - P&L calculation
    - State reporting for HRM
    """
    bot_id: str
    bot_type: BotType
    model: QuantModel
    instruments: List[str]
    
    state: BotState = field(default=None)
    position: float = 0.0
    entry_price: float = 0.0
    trade_history: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        if self.state is None:
            self.state = BotState(
                bot_id=self.bot_id,
                bot_type=self.bot_type.value,
            )
    
    def compute(self, registry: InstrumentRegistry) -> pd.DataFrame:
        """Compute signals from instruments"""
        self.state.status = "computing"
        
        signals = self.model.compute_signals(registry)
        
        if not signals.empty and 'signal' in signals.columns:
            self.state.last_signal = signals['signal'].iloc[-1]
            self.state.confidence = abs(self.state.last_signal)
        
        self.state.status = "active"
        self.state.timestamp = datetime.utcnow().isoformat()
        
        return signals
    
    def update_pnl(self, current_price: float):
        """Update P&L based on current position"""
        if self.position != 0 and self.entry_price != 0:
            self.state.pnl = self.position * (current_price - self.entry_price)
    
    def execute_signal(self, signal: float, price: float, size: float = 1.0):
        """Execute a trade based on signal"""
        if signal > 0.5 and self.position <= 0:
            self.position = size
            self.entry_price = price
            self.state.trades_count += 1
            self.trade_history.append({
                'type': 'LONG',
                'price': price,
                'size': size,
                'timestamp': datetime.utcnow().isoformat(),
            })
        elif signal < -0.5 and self.position >= 0:
            self.position = -size
            self.entry_price = price
            self.state.trades_count += 1
            self.trade_history.append({
                'type': 'SHORT',
                'price': price,
                'size': size,
                'timestamp': datetime.utcnow().isoformat(),
            })
    
    def get_state(self) -> Dict[str, Any]:
        """Get state for HRM"""
        return self.state.to_dict()


class TradeBotRegistry:
    """
    Registry of tradebots that HRM can select from.
    """
    
    def __init__(self):
        self._bots: Dict[str, TradeBot] = {}
        self._bot_states: Dict[str, Dict[str, Any]] = {}
    
    def register(self, bot: TradeBot):
        """Register a tradebot"""
        self._bots[bot.bot_id] = bot
    
    def get(self, bot_id: str) -> Optional[TradeBot]:
        """Get bot by ID"""
        return self._bots.get(bot_id)
    
    def list_bots(self) -> List[str]:
        """List all bot IDs"""
        return list(self._bots.keys())
    
    def compute_all(self, registry: InstrumentRegistry) -> Dict[str, pd.DataFrame]:
        """Compute signals from all bots"""
        results = {}
        for bot_id, bot in self._bots.items():
            results[bot_id] = bot.compute(registry)
            self._bot_states[bot_id] = bot.get_state()
        return results
    
    def get_all_states(self) -> pd.DataFrame:
        """Get all bot states as DataFrame for HRM"""
        states = [bot.get_state() for bot in self._bots.values()]
        return pd.DataFrame(states)
    
    def get_bot_by_type(self, bot_type: BotType) -> List[TradeBot]:
        """Get all bots of a type"""
        return [b for b in self._bots.values() if b.bot_type == bot_type]

    def rank_bots(self) -> pd.DataFrame:
        """
        Return all bot states ranked by composite score.

        score = confidence * 0.5 + energy * 0.3 + win_rate * 0.2

        - confidence: |last_signal|, available immediately after compute()
        - energy:     EWMA of realized returns, non-zero after update_performance() calls
        - win_rate:   historical win rate, non-zero after trades settle (defaults to 0.5)
        """
        states = self.get_all_states()
        if states.empty:
            return states

        states['score'] = (
            states['confidence'] * 0.5
            + states['energy'] * 0.3
            + states['win_rate'] * 0.2
        )
        states = states.sort_values('score', ascending=False).reset_index(drop=True)
        states.insert(0, 'rank', range(1, len(states) + 1))
        return states


def create_default_bots(models: Optional[Dict[str, QuantModel]] = None) -> Dict[str, TradeBot]:
    """
    Create default tradebots from quant models.
    
    Implements 24 strategies from DeFlorio Thesis 2022:
    - Chapter 2.3.2: Trading Algorithms (manual methods)
    - Chapter 3.2.2: AI Pattern Matching
    - SMA 50/200 crossover
    - Hourly swing analysis
    - Statistical breakout (current vs highest)
    - Stop loss risk management
    """
    if models is None:
        from instruments import create_default_models
        models = create_default_models()
    
    bots = {}
    
    # 24 Tradebot Strategies from Thesis
    bot_configs = [
        # 1. Trend Following (SMA Crossover) - Thesis p.11-12
        ('sma_50_200_cross', BotType.TREND_FOLLOWING, 'sma_cross'),
        
        # 2. Momentum (Hourly Swing) - Thesis p.10
        ('hourly_swing_momentum', BotType.MOMENTUM, 'hourly_swing'),
        
        # 3. Statistical Breakout - Thesis p.10 (current % vs highest %)
        ('statistical_breakout', BotType.STATISTICAL_BREAKOUT, 'stat_breakout'),
        
        # 4. Stop Loss Risk Management - Thesis p.11
        ('stop_loss_risk', BotType.STOP_LOSS, 'stop_loss'),
        
        # 5. DCA Weighted - Thesis p.11 (multiple small trades)
        ('weighted_dca', BotType.DCA_WEIGHTED, 'dca_weighted'),
        
        # 6. Pattern Matching (AI) - Thesis p.14-15
        ('pattern_matching_ai', BotType.PATTERN_MATCHING, 'pattern_match'),
        
        # 7. Daily Momentum (Percent Change) - Thesis p.10
        ('daily_percent_change', BotType.DAILY_MOMENTUM, 'daily_mom'),
        
        # 8. Support/Resistance - Technical analysis
        ('support_resistance', BotType.SUPPORT_RESISTANCE, 'sup_res'),
        
        # 9. Time-based Trading - Thesis p.14 (time-of-day patterns)
        ('time_of_day', BotType.TIME_BASED, 'time_based'),
        
        # 10. Cross-asset Correlation - Thesis p.15
        ('cross_correlation', BotType.CORRELATION, 'correlation'),
        
        # 11. Choppy Market Detector - Hourly swing analysis
        ('choppy_detector', BotType.HOUR_CHOPPY, 'choppy'),
        
        # 12. Volatility Breakout - Thesis p.15
        ('volatility_breakout', BotType.VOLATILITY_BREAKOUT, 'volatility_breakout'),
        
        # 13. Mean Reversion - Counter-trend
        ('mean_reversion', BotType.MEAN_REVERSION, 'mean_rev'),
        
        # 14. Stat Arb (Pairs) - Thesis p.15
        ('pairs_spread', BotType.STAT_ARB, 'pairs_spread'),
        
        # 15. Composite Ensemble - Thesis p.15
        ('composite_ensemble', BotType.COMPOSITE, 'composite'),
        
        # 16. Weighted SMA - Thesis p.11
        ('weighted_sma', BotType.TREND_FOLLOWING, 'weighted_sma'),
        
        # 17. EMA Crossover - Faster SMA
        ('ema_cross', BotType.TREND_FOLLOWING, 'ema_cross'),
        
        # 18. MACD Signal - Momentum
        ('macd_signal', BotType.MOMENTUM, 'macd'),
        
        # 19. RSI Overbought/Oversold - Thesis p.15
        ('rsi_reversion', BotType.MEAN_REVERSION, 'rsi_reversion'),
        
        # 20. Bollinger Band Breakout - Volatility
        ('bollinger_breakout', BotType.VOLATILITY_BREAKOUT, 'bollinger'),
        
        # 21. ATR Trailing Stop - Risk management
        ('atr_trailing', BotType.STOP_LOSS, 'atr_trailing'),
        
        # 22. VWAP Reversion - Mean reversion
        ('vwap_reversion', BotType.MEAN_REVERSION, 'vwap'),
        
        # 23. Order Flow - Volume analysis
        ('order_flow', BotType.MOMENTUM, 'order_flow'),
        
        # 24. Stochastic Oscillator - Momentum
        ('stochastic_osc', BotType.MOMENTUM, 'stochastic'),
    ]
    
    for bot_id, bot_type, model_id in bot_configs:
        if model_id in models:
            model = models[model_id]
            bots[bot_id] = TradeBot(
                bot_id=bot_id,
                bot_type=bot_type,
                model=model,
                instruments=model.required_instruments,
            )
    
    return bots


def create_bot_registry(bots: Optional[Dict[str, TradeBot]] = None) -> TradeBotRegistry:
    """Create and populate a bot registry"""
    registry = TradeBotRegistry()
    
    if bots is None:
        bots = create_default_bots()
    
    for bot in bots.values():
        registry.register(bot)
    
    return registry


if __name__ == "__main__":
    print("TradeBots: PANDAS -> instruments -> {tradebots} -> HRM IO")
    print("=" * 60)
    
    np.random.seed(42)
    
    market_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='5min'),
        'asset': 'BTC-USD',
        'open': 40000 + np.random.randn(100).cumsum() * 10,
        'high': 40100 + np.random.randn(100).cumsum() * 10,
        'low': 39900 + np.random.randn(100).cumsum() * 10,
        'close': 40050 + np.random.randn(100).cumsum() * 10,
        'volume': np.abs(np.random.randn(100)) * 1000,
    })
    
    from instruments import InstrumentRegistry, LazyInstrument
    
    inst_reg = InstrumentRegistry()
    inst_reg.register('market_data', LazyInstrument('market_data', lambda: market_data))
    inst_reg.register('features', LazyInstrument('features', lambda: pd.DataFrame({
        'momentum': np.random.randn(100) * 0.1,
        'ma_ratio': 1 + np.random.randn(100) * 0.02,
        'rsi': np.random.uniform(0.3, 0.7, 100),
    })))
    inst_reg.register('sectors', LazyInstrument('sectors', lambda: pd.DataFrame([{'majors': ['BTC-USD']}])))
    inst_reg.register('signals', LazyInstrument('signals', lambda: pd.DataFrame({'signal': [0]})))
    
    print("\n1. Instruments registered:", inst_reg.list_instruments())
    
    bots = create_default_bots()
    bot_registry = create_bot_registry(bots)
    
    print("\n2. TradeBots registered:", bot_registry.list_bots())
    
    signals = bot_registry.compute_all(inst_reg)
    
    print("\n3. Signals computed:")
    for bot_id, df in signals.items():
        if not df.empty:
            print(f"   {bot_id}: {len(df)} signals, last={df['signal'].iloc[-1]:.4f}")
    
    states_df = bot_registry.get_all_states()
    print("\n4. Bot states for HRM:")
    print(states_df[['bot_id', 'bot_type', 'last_signal', 'confidence']].to_string())
