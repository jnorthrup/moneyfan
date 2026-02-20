"""
Lazy Pandas Services as Instruments

HRM reads model states lazily, chooses ONE to act.
No eager computation. No swarm complexity.

Pattern:
    instrument = LazyInstrument(lambda: load_data())
    result = instrument.compute()  # Only executes when called
"""

import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from functools import wraps
import inspect


# =============================================================================
# LAZY INSTRUMENT
# =============================================================================

@dataclass
class LazyInstrument:
    """
    Lazy pandas computation.
    
    Wraps a function that produces a DataFrame.
    Only executes when .compute() is called.
    
    Example:
        market = LazyInstrument(lambda: pd.read_parquet('data.parquet'))
        df = market.compute()  # Loads data now
    """
    name: str
    compute_fn: Callable[[], pd.DataFrame]
    _cached: Optional[pd.DataFrame] = field(default=None, repr=False)
    _dependencies: List['LazyInstrument'] = field(default_factory=list, repr=False)
    
    def compute(self, force: bool = False) -> pd.DataFrame:
        """Execute the lazy computation"""
        if self._cached is None or force:
            self._cached = self.compute_fn()
        return self._cached
    
    def clear_cache(self):
        """Clear cached result"""
        self._cached = None
    
    def pipe(self, fn: Callable, *args, **kwargs) -> 'LazyInstrument':
        """Chain lazy operations"""
        def new_compute():
            return fn(self.compute(), *args, **kwargs)
        return LazyInstrument(
            name=f"{self.name}_piped",
            compute_fn=new_compute,
            _dependencies=[self]
        )
    
    def __repr__(self):
        cached_status = "cached" if self._cached is not None else "lazy"
        return f"LazyInstrument({self.name}, {cached_status})"


# =============================================================================
# INSTRUMENT REGISTRY
# =============================================================================

class InstrumentRegistry:
    """
    Registry of lazy instruments.
    
    HRM reads from here to get model states.
    """
    
    def __init__(self):
        self._instruments: Dict[str, LazyInstrument] = {}
        self._model_states: Dict[str, Dict[str, Any]] = {}
    
    def register(self, name: str, instrument: LazyInstrument):
        """Register an instrument"""
        self._instruments[name] = instrument
    
    def get(self, name: str) -> LazyInstrument:
        """Get instrument by name"""
        return self._instruments.get(name)
    
    def compute(self, name: str) -> pd.DataFrame:
        """Compute an instrument"""
        inst = self.get(name)
        if inst:
            return inst.compute()
        raise KeyError(f"Instrument not found: {name}")
    
    def list_instruments(self) -> List[str]:
        """List all registered instruments"""
        return list(self._instruments.keys())
    
    def set_model_state(self, model_id: str, state: Dict[str, Any]):
        """Store model state for HRM to read"""
        self._model_states[model_id] = state
    
    def get_model_states(self) -> Dict[str, Dict[str, Any]]:
        """Get all model states (HRM reads this)"""
        return self._model_states.copy()
    
    def get_model_state(self, model_id: str) -> Dict[str, Any]:
        """Get state for one model"""
        return self._model_states.get(model_id, {})


# Global registry
registry = InstrumentRegistry()


# =============================================================================
# QUANT MODELS AS INSTRUMENT READERS
# =============================================================================

class QuantModel:
    """
    A quant model that reads instruments and produces signals.
    
    HRM reads model.state before deciding to activate.
    Model is LAZY - only computes when activated.
    """
    
    def __init__(self, 
                 model_id: str,
                 description: str,
                 required_instruments: List[str],
                 compute_fn: Callable[[Dict[str, pd.DataFrame]], pd.DataFrame]):
        """
        Args:
            model_id: Unique identifier
            description: What this model does
            required_instruments: List of instrument names to read
            compute_fn: Function that takes {name: df} and returns signals df
        """
        self.model_id = model_id
        self.description = description
        self.required_instruments = required_instruments
        self.compute_fn = compute_fn
        
        # State that HRM reads (updated after each compute)
        self.state = {
            'model_id': model_id,
            'status': 'idle',
            'last_return': 0.0,
            'signal_count': 0,
            'confidence': 0.0,
            'energy': 0.0,
            'timestamp': None,
        }
    
    def read_instruments(self, reg: InstrumentRegistry) -> Dict[str, pd.DataFrame]:
        """Read all required instruments"""
        return {name: reg.compute(name) for name in self.required_instruments}
    
    def compute_signals(self, reg: InstrumentRegistry) -> pd.DataFrame:
        """
        Compute signals. Only called when HRM activates this model.
        """
        self.state['status'] = 'computing'
        
        # Read instruments
        instruments = self.read_instruments(reg)
        
        # Compute signals
        signals = self.compute_fn(instruments)
        
        # Update state
        self.state['status'] = 'active'
        self.state['signal_count'] = len(signals)
        if 'signal' in signals.columns:
            self.state['confidence'] = signals['signal'].abs().mean()
        
        # Store state in registry for HRM
        reg.set_model_state(self.model_id, self.state)
        
        return signals
    
    def update_performance(self, returns: float):
        """Update state with actual returns"""
        self.state['last_return'] = returns
        self.state['energy'] = 0.9 * self.state['energy'] + 0.1 * returns
    
    def __repr__(self):
        return f"QuantModel({self.model_id}, {self.state['status']})"


# =============================================================================
# BUILT-IN MODELS
# =============================================================================

def model_volatility_breakout(instruments: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    PROVEN WINNER: volatility × breakout
    
    Reads: market_data
    """
    df = instruments['market_data'].copy()
    
    # Features
    df['volatility'] = (df['high'] - df['low']) / df['open']
    df['position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    df['breakout'] = 2 * df['position'] - 1
    
    # Signal
    df['signal'] = df['volatility'].clip(0, 1) * df['breakout']
    
    return df[['signal']]


def model_momentum_trend(instruments: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    momentum × trend
    
    Reads: features
    """
    df = instruments['features'].copy()
    
    trend = np.sign(df.get('ma_ratio', 1) - 1)
    strength = df.get('momentum', 0).abs().clip(0, 1)
    
    df['signal'] = strength * trend
    return df[['signal']]


def model_mean_reversion(instruments: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    mean reversion
    
    Reads: features
    """
    df = instruments['features'].copy()
    
    deviation = df.get('ma_ratio', 1) - 1
    rsi_signal = (50 - df.get('rsi', 50) * 100) / 50
    
    df['signal'] = 0.5 * (-np.sign(deviation)) + 0.5 * rsi_signal
    return df[['signal']]


def model_sector_rotation(instruments: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Sector-based rotation.
    
    Reads: market_data, sectors
    """
    df = instruments['market_data'].copy()
    sectors = instruments.get('sectors', {})
    
    # Simple: momentum per sector
    df['returns'] = df.groupby('asset')['close'].pct_change()
    
    signals = []
    for asset in df['asset'].unique():
        asset_df = df[df['asset'] == asset]
        mom = asset_df['returns'].rolling(10).mean().iloc[-1]
        signals.append({'asset': asset, 'signal': mom})
    
    return pd.DataFrame(signals)


def model_composite(instruments: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Composite of other model signals.
    
    Reads: signals (from other models)
    """
    signals_input = instruments.get('signals')
    
    if signals_input is None:
        return pd.DataFrame({'signal': [0]})
    
    if isinstance(signals_input, dict):
        return pd.DataFrame({'signal': [0]})
    
    df = signals_input.copy() if hasattr(signals_input, 'copy') else pd.DataFrame(signals_input)
    
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame({'signal': [0]})
    
    if 'confidence' in df.columns:
        df['signal'] = df['signal'] * df['confidence']
    
    return df[['signal']] if 'signal' in df.columns else pd.DataFrame({'signal': [0]})


# =============================================================================
# HRM INTERFACE
# =============================================================================

@dataclass
class HRMDecision:
    """HRM output - which model to activate"""
    model_id: str
    parameters: Dict[str, Any]
    confidence: float
    reason: str


def hrm_read_all_states(reg: InstrumentRegistry, models: Dict[str, QuantModel]) -> pd.DataFrame:
    """
    HRM reads all model states as a DataFrame.
    
    This is the INPUT to HRM's decision.
    """
    states = []
    for model_id, model in models.items():
        state = model.state.copy()
        state['model_id'] = model_id
        states.append(state)
    
    return pd.DataFrame(states)


def hrm_choose_model(states_df: pd.DataFrame,
                      market_context: Dict[str, Any] = None) -> HRMDecision:
    """
    HRM chooses ONE model to act.

    This is where HRM's intelligence lives.
    For now, simple heuristic - can be replaced with neural HRM.
    """
    if states_df.empty:
        return HRMDecision(
            model_id='volatility_breakout',
            parameters={},
            confidence=1.0,
            reason='default: no states available'
        )

    # Simple heuristic: choose model with best composite score
    states_df = states_df.copy()
    states_df['score'] = (
        states_df['confidence'] * 0.5 +
        states_df['energy'] * 0.3 +
        states_df['win_rate'] * 0.2
    )
    best = states_df.loc[states_df['score'].idxmax()]

    return HRMDecision(
        model_id=best['model_id'],
        parameters={},
        confidence=best['score'],
        reason=f"highest score: {best['score']:.4f}"
    )


# =============================================================================
# BUILT-IN MODELS REGISTRY
# =============================================================================

def create_default_models() -> Dict[str, QuantModel]:
    """Create the default set of quant models"""
    return {
        'volatility_breakout': QuantModel(
            model_id='volatility_breakout',
            description='PROVEN WINNER: volatility × breakout signal',
            required_instruments=['market_data'],
            compute_fn=model_volatility_breakout,
        ),
        'momentum_trend': QuantModel(
            model_id='momentum_trend',
            description='momentum × trend signal',
            required_instruments=['features'],
            compute_fn=model_momentum_trend,
        ),
        'mean_reversion': QuantModel(
            model_id='mean_reversion',
            description='mean reversion signal',
            required_instruments=['features'],
            compute_fn=model_mean_reversion,
        ),
        'sector_rotation': QuantModel(
            model_id='sector_rotation',
            description='sector-based rotation',
            required_instruments=['market_data', 'sectors'],
            compute_fn=model_sector_rotation,
        ),
        'composite': QuantModel(
            model_id='composite',
            description='composite of other signals',
            required_instruments=['signals'],
            compute_fn=model_composite,
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Lazy Pandas Instruments for HRM")
    print("=" * 50)
    
    # Create some fake instruments
    np.random.seed(42)
    
    market_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='5min'),
        'asset': 'BTC-USD',
        'open': 40000 + np.random.randn(100).cumsum() * 10,
        'high': 40100 + np.random.randn(100).cumsum() * 10,
        'low': 39900 + np.random.randn(100).cumsum() * 10,
        'close': 40050 + np.random.randn(100).cumsum() * 10,
        'volume': np.random.randn(100).abs() * 1000,
    })
    
    features = pd.DataFrame({
        'momentum': np.random.randn(100) * 0.1,
        'ma_ratio': 1 + np.random.randn(100) * 0.02,
        'rsi': np.random.uniform(0.3, 0.7, 100),
    })
    
    # Register instruments
    registry.register('market_data', LazyInstrument('market_data', lambda: market_data))
    registry.register('features', LazyInstrument('features', lambda: features))
    registry.register('sectors', LazyInstrument('sectors', lambda: {'majors': [0, 1, 2]}))
    registry.register('signals', LazyInstrument('signals', lambda: pd.DataFrame()))
    
    print("Instruments:", registry.list_instruments())
    
    # Create models
    models = create_default_models()
    
    print("\nModels:", list(models.keys()))
    
    # HRM reads all states
    states = hrm_read_all_states(registry, models)
    print("\nModel states:")
    print(states)
    
    # HRM chooses one model
    decision = hrm_choose_model(states)
    print(f"\nHRM Decision: {decision}")
    
    # Activate chosen model
    chosen_model = models[decision.model_id]
    signals = chosen_model.compute_signals(registry)
    print(f"\nSignals from {decision.model_id}:")
    print(signals.head())
