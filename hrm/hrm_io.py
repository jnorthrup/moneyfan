"""
HRM IO - Input/Output layer for HRM

PANDAS -> instruments -> {tradebots} -> HRM IO

This module provides the IO boundary between:
1. PANDAS DataFrames (raw data)
2. LazyInstruments (lazy evaluation)
3. TradeBots (signal generation)
4. HRM (decision making)

HRM IO responsibilities:
- Accept pandas data and create instruments
- Route instruments to tradebots
- Collect bot states for HRM input
- Format HRM output as actions

Usage:
    io = HRMIO()
    
    # Input: pandas DataFrame
    io.ingest(market_df, 'market_data')
    
    # Process: instruments -> tradebots
    signals = io.process()
    
    # Output: HRM decision
    decision = io.decide()
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import pandas as pd
import numpy as np
from datetime import datetime
from enum import Enum

from instruments import (
    LazyInstrument, 
    InstrumentRegistry, 
    QuantModel,
    hrm_read_all_states,
    hrm_choose_model,
    HRMDecision,
)
from tradebots import (
    TradeBot,
    TradeBotRegistry,
    BotType,
    BotState,
    create_default_bots,
    create_bot_registry,
)
from membrane import (
    CoinbaseMembrane,
    MembraneConfig,
    MembraneState,
    MarketRegime,
)
from convergence import convergence_from_snapshot


class HRMInputType(Enum):
    PANDAS = "pandas"
    DICT = "dict"
    PARQUET = "parquet"
    SQL = "sql"


@dataclass
class HRMOutput:
    """HRM decision output"""
    decision: HRMDecision
    active_bot: str
    signal: float
    confidence: float
    regime: str
    timestamp: str
    convergence: float = 0.0
    agreement_ratio: float = 0.0
    confidence_support: float = 0.0
    membrane_state: Optional[MembraneState] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'decision': self.decision.model_id,
            'active_bot': self.active_bot,
            'signal': self.signal,
            'confidence': self.confidence,
            'regime': self.regime,
            'convergence': self.convergence,
            'agreement_ratio': self.agreement_ratio,
            'confidence_support': self.confidence_support,
            'timestamp': self.timestamp,
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])


class HRMIO:
    """
    HRM Input/Output layer.
    
    Connects: PANDAS -> instruments -> tradebots -> HRM
    
    Example:
        io = HRMIO()
        
        # Ingest pandas data
        io.ingest(market_df, 'market_data')
        io.ingest(features_df, 'features')
        
        # Process through bots
        signals = io.process()
        
        # Get HRM decision
        output = io.decide()
    """
    
    def __init__(
        self,
        use_membrane: bool = True,
        membrane_config: Optional[MembraneConfig] = None,
        bots: Optional[Dict[str, TradeBot]] = None,
    ):
        self.instrument_registry = InstrumentRegistry()
        self.bot_registry = create_bot_registry(bots)
        
        self.use_membrane = use_membrane
        if use_membrane:
            self.membrane = CoinbaseMembrane(membrane_config or MembraneConfig())
        else:
            self.membrane = None
        
        self._signals_cache: Dict[str, pd.DataFrame] = {}
        self._last_output: Optional[HRMOutput] = None
        self._history: List[HRMOutput] = []
    
    def ingest(
        self,
        data: pd.DataFrame,
        name: str,
        lazy: bool = True,
    ) -> LazyInstrument:
        """
        Ingest pandas DataFrame as an instrument.
        
        Args:
            data: pandas DataFrame to ingest
            name: Instrument name (e.g., 'market_data', 'features')
            lazy: If True, wrap in lazy evaluation
        
        Returns:
            The created LazyInstrument
        """
        if lazy:
            instrument = LazyInstrument(name, lambda: data)
        else:
            instrument = LazyInstrument(name, lambda: data)
            instrument.compute()  # Eager evaluation
        
        self.instrument_registry.register(name, instrument)
        return instrument
    
    def ingest_dict(
        self,
        data: Dict[str, pd.DataFrame],
    ) -> List[LazyInstrument]:
        """
        Ingest multiple DataFrames at once.
        
        Args:
            data: Dict of {name: DataFrame}
        
        Returns:
            List of created LazyInstruments
        """
        instruments = []
        for name, df in data.items():
            instruments.append(self.ingest(df, name))
        return instruments
    
    def process(self) -> Dict[str, pd.DataFrame]:
        """
        Process instruments through tradebots.
        
        Returns:
            Dict of {bot_id: signals_df}
        """
        self._signals_cache = self.bot_registry.compute_all(self.instrument_registry)
        return self._signals_cache
    
    def get_bot_states(self) -> pd.DataFrame:
        """Get all bot states as DataFrame for HRM"""
        return self.bot_registry.get_all_states()

    def rank(self) -> pd.DataFrame:
        """
        Return the ranking of tradebots.

        - If signals have not been computed yet, it triggers `process()`.
        - Delegates to `TradeBotRegistry.rank_bots()` which scores and sorts.
        """
        # Ensure we have latest signals (some scores like confidence rely on them)
        if not self._signals_cache:
            self.process()
        return self.bot_registry.rank_bots()
    
    def decide(
        self,
        market_context: Optional[Dict[str, Any]] = None,
    ) -> HRMOutput:
        """
        Make HRM decision based on bot states.
        
        Returns:
            HRMOutput with decision, signal, and metadata
        """
        states_df = self.get_bot_states()
        
        decision = hrm_choose_model(states_df, market_context)
        
        bot = self.bot_registry.get(decision.model_id)
        
        if bot and bot.bot_id in self._signals_cache:
            signals = self._signals_cache[bot.bot_id]
            if not signals.empty and 'signal' in signals.columns:
                signal = signals['signal'].iloc[-1]
                confidence = abs(signal)
            else:
                signal = 0.0
                confidence = 0.0
        else:
            signal = 0.0
            confidence = 0.0
        
        regime = self._detect_regime(states_df)
        convergence_payload = self._compute_convergence_payload(states_df)
        
        membrane_state = None
        if self.use_membrane and self.membrane:
            membrane_state = self.membrane.state
        
        output = HRMOutput(
            decision=decision,
            active_bot=decision.model_id,
            signal=signal,
            confidence=confidence,
            regime=regime,
            convergence=convergence_payload['convergence'],
            agreement_ratio=convergence_payload['agreement_ratio'],
            confidence_support=convergence_payload['confidence_support'],
            timestamp=datetime.utcnow().isoformat(),
            membrane_state=membrane_state,
        )
        
        self._last_output = output
        self._history.append(output)
        
        return output

    def _compute_convergence_payload(self, states_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute convergence fields for normalized decision payloads.
        """
        if states_df.empty or 'last_signal' not in states_df.columns:
            return {
                'convergence': 0.0,
                'agreement_ratio': 0.0,
                'confidence_support': 0.0,
            }

        signals = states_df['last_signal'].to_numpy(dtype=np.float32)
        if 'confidence' in states_df.columns:
            confidences = states_df['confidence'].to_numpy(dtype=np.float32)
        else:
            confidences = np.abs(signals).astype(np.float32)

        convergence = convergence_from_snapshot(
            signals,
            confidences,
            min_agree=2,
            min_confidence_sum=0.25,
        )

        pos = signals > 0
        neg = signals < 0
        n_pos = int(np.sum(pos))
        n_neg = int(np.sum(neg))
        n_nonzero = max(1, n_pos + n_neg)
        agreement_ratio = float(max(n_pos, n_neg) / n_nonzero)
        confidence_support = float(
            max(
                np.sum(confidences[pos]) if n_pos > 0 else 0.0,
                np.sum(confidences[neg]) if n_neg > 0 else 0.0,
            )
        )

        return {
            'convergence': float(convergence),
            'agreement_ratio': agreement_ratio,
            'confidence_support': confidence_support,
        }
    
    def _detect_regime(self, states_df: pd.DataFrame) -> str:
        """Detect market regime from bot states"""
        if states_df.empty:
            return "transition"
        
        momentum_score = states_df[states_df['bot_type'] == 'momentum']['energy'].sum()
        reversion_score = states_df[states_df['bot_type'] == 'reversion']['energy'].sum()
        volatility_score = states_df[states_df['bot_type'] == 'volatility']['energy'].sum()
        
        max_score = max(momentum_score, reversion_score, volatility_score)
        
        if max_score == momentum_score and momentum_score > 0:
            return "trending"
        elif max_score == reversion_score and reversion_score > 0:
            return "ranging"
        elif max_score == volatility_score and volatility_score > 0:
            return "volatile"
        else:
            return "transition"
    
    def run_pipeline(
        self,
        data: Dict[str, pd.DataFrame],
        market_context: Optional[Dict[str, Any]] = None,
    ) -> HRMOutput:
        """
        Run full pipeline: ingest -> process -> decide
        
        Args:
            data: Dict of {instrument_name: DataFrame}
            market_context: Optional market context for HRM
        
        Returns:
            HRMOutput with decision
        """
        self.ingest_dict(data)
        self.process()
        return self.decide(market_context)
    
    def get_history(self, limit: int = 100) -> pd.DataFrame:
        """Get decision history as DataFrame"""
        if not self._history:
            return pd.DataFrame()
        
        records = [h.to_dict() for h in self._history[-limit:]]
        return pd.DataFrame(records)
    
    def summary(self) -> str:
        """Get summary of HRM IO state"""
        lines = [
            "HRM IO Summary",
            "=" * 50,
            f"Instruments: {len(self.instrument_registry.list_instruments())}",
            f"TradeBots: {len(self.bot_registry.list_bots())}",
            f"Membrane: {'enabled' if self.use_membrane else 'disabled'}",
            f"Decisions made: {len(self._history)}",
        ]
        
        if self._last_output:
            lines.extend([
                "",
                "Last Decision:",
                f"  Bot: {self._last_output.active_bot}",
                f"  Signal: {self._last_output.signal:.4f}",
                f"  Confidence: {self._last_output.confidence:.4f}",
                f"  Regime: {self._last_output.regime}",
            ])
        
        return "\n".join(lines)


def create_hrm_io(
    market_data: Optional[pd.DataFrame] = None,
    features: Optional[pd.DataFrame] = None,
    **kwargs,
) -> HRMIO:
    """
    Factory to create HRMIO with optional initial data.
    
    Args:
        market_data: Optional market data DataFrame
        features: Optional features DataFrame
        **kwargs: Additional arguments for HRMIO
    
    Returns:
        Configured HRMIO instance
    """
    io = HRMIO(**kwargs)
    
    if market_data is not None:
        io.ingest(market_data, 'market_data')
    
    if features is not None:
        io.ingest(features, 'features')
    
    return io


if __name__ == "__main__":
    print("HRM IO: PANDAS -> instruments -> {tradebots} -> HRM IO")
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
    
    features = pd.DataFrame({
        'momentum': np.random.randn(100) * 0.1,
        'ma_ratio': 1 + np.random.randn(100) * 0.02,
        'rsi': np.random.uniform(0.3, 0.7, 100),
    })
    
    sectors = {'majors': ['BTC-USD', 'ETH-USD']}
    
    signals = pd.DataFrame({
        'signal': np.random.randn(10) * 0.5,
        'confidence': np.random.uniform(0.5, 1.0, 10),
    })
    
    sectors_df = pd.DataFrame([{'majors': ['BTC-USD', 'ETH-USD']}])
    
    print("\n1. Creating HRM IO...")
    io = HRMIO(use_membrane=True)
    
    print("\n2. Ingesting PANDAS data...")
    io.ingest(market_data, 'market_data')
    io.ingest(features, 'features')
    io.ingest(sectors_df, 'sectors')
    io.ingest(signals, 'signals')
    print(f"   Instruments: {io.instrument_registry.list_instruments()}")
    
    print("\n3. Processing instruments -> tradebots...")
    signals = io.process()
    for bot_id, df in signals.items():
        if not df.empty:
            print(f"   {bot_id}: signal={df['signal'].iloc[-1]:.4f}")
    
    print("\n4. HRM decision...")
    output = io.decide()
    print(f"   Active bot: {output.active_bot}")
    print(f"   Signal: {output.signal:.4f}")
    print(f"   Confidence: {output.confidence:.4f}")
    print(f"   Regime: {output.regime}")
    
    print("\n" + io.summary())
    
    print("\n5. Full pipeline in one call:")
    io2 = HRMIO()
    output2 = io2.run_pipeline({
        'market_data': market_data,
        'features': features,
        'sectors': sectors_df,
        'signals': signals,
    })
    print(f"   Decision: {output2.to_dict()}")
