"""
Concurrent Lazy Signal Orchestrator
Uses pandas event hooks to compose and multiply trading signals

Architecture:
1. Signal Services - Individual strategy implementations
2. Event Hooks - Pandas-based event triggers
3. Lazy Evaluation - Signals computed only when accessed
4. Composition Engine - Combine signals multiplicatively
5. Orchestrator - Manage concurrent execution
"""

import asyncio
import numpy as np
import pandas as pd
from pandas.api.extensions import register_dataframe_accessor
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Any, Awaitable
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor
import time
from collections import defaultdict
import json

# ==============================================================================
# LAZY SIGNAL EVALUATION
# ==============================================================================

class LazySignal:
    """Lazily evaluated signal that only computes when accessed"""
    
    def __init__(self, name: str, func: Callable, deps: List[str] = None):
        self.name = name
        self.func = func
        self.deps = deps or []
        self._value = None
        self._computed = False
        self._timestamp = None
    
    def __call__(self, df: pd.DataFrame, cache: Dict = None) -> pd.Series:
        """Compute signal lazily"""
        if cache is not None and self.name in cache:
            return cache[self.name]
        
        if not self._computed or self._timestamp != df.index[-1]:
            self._value = self.func(df)
            self._computed = True
            self._timestamp = df.index[-1]
            if cache is not None:
                cache[self.name] = self._value
        
        return self._value
    
    def __repr__(self):
        return f"LazySignal({self.name}, deps={self.deps})"


class LazySignalStore:
    """Store and manage lazy signals"""
    
    def __init__(self):
        self.signals: Dict[str, LazySignal] = {}
        self.cache: Dict[str, pd.Series] = {}
        self._graph = defaultdict(list)
    
    def register(self, name: str, func: Callable, deps: List[str] = None):
        """Register a new lazy signal"""
        signal = LazySignal(name, func, deps)
        self.signals[name] = signal
        
        # Build dependency graph
        for dep in (deps or []):
            self._graph[dep].append(name)
        
        return signal
    
    def get(self, name: str, df: pd.DataFrame) -> pd.Series:
        """Get signal value, computing if needed"""
        if name not in self.signals:
            raise KeyError(f"Signal '{name}' not registered")
        
        # Compute dependencies first
        signal = self.signals[name]
        for dep in signal.deps:
            if dep in self.signals:
                self.get(dep, df)  # Ensure deps computed
        
        return signal(df, self.cache)
    
    def clear_cache(self):
        """Clear computed cache"""
        self.cache.clear()
        for signal in self.signals.values():
            signal._computed = False


# ==============================================================================
# PANDAS EVENT HOOKS
# ==============================================================================

@dataclass
class Event:
    """Event fired by pandas operations"""
    name: str
    timestamp: int
    data: Any = None
    source: str = None


class PandasEventHook:
    """Event hooks for pandas DataFrame operations"""
    
    def __init__(self):
        self.listeners: Dict[str, List[Callable]] = defaultdict(list)
        self.event_log: List[Event] = []
    
    def on(self, event_name: str, callback: Callable):
        """Register listener for event"""
        self.listeners[event_name].append(callback)
        return callback
    
    def emit(self, event_name: str, timestamp: int, data: Any = None, source: str = None):
        """Emit event to all listeners"""
        event = Event(event_name, timestamp, data, source)
        self.event_log.append(event)
        
        for callback in self.listeners.get(event_name, []):
            try:
                callback(event)
            except Exception as e:
                print(f"Event handler error: {e}")
    
    def emit_async(self, event_name: str, timestamp: int, data: Any = None, source: str = None):
        """Emit event asynchronously"""
        event = Event(event_name, timestamp, data, source)
        self.event_log.append(event)
        
        async def _emit():
            for callback in self.listeners.get(event_name, []):
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    print(f"Async event handler error: {e}")
        
        return _emit()


@register_dataframe_accessor("events")
class EventsAccessor:
    """Pandas accessor for event hooks"""
    
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self._hooks = {}
    
    def on_new_row(self, callback: Callable):
        """Hook for new row events"""
        self._hooks['new_row'] = callback
        return self
    
    def on_price_change(self, callback: Callable):
        """Hook for price change events"""
        self._hooks['price_change'] = callback
        return self
    
    def on_volatility_spike(self, callback: Callable):
        """Hook for volatility spike events"""
        self._hooks['volatility_spike'] = callback
        return self
    
    def fire(self, event_name: str, row: pd.Series):
        """Fire event for row"""
        if event_name in self._hooks:
            self._hooks[event_name](row)


# ==============================================================================
# SIGNAL SERVICES
# ==============================================================================

class SignalService:
    """Base class for signal services"""
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self._last_value = None
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute signal - override in subclass"""
        raise NotImplementedError
    
    def __call__(self, df: pd.DataFrame) -> pd.Series:
        if not self.enabled:
            return pd.Series(0, index=df.index)
        
        self._last_value = self.compute(df)
        return self._last_value


class GridService(SignalService):
    """Grid Trading signal service"""
    
    def __init__(self, spacing: float = 0.02, levels: int = 5):
        super().__init__("grid")
        self.spacing = spacing
        self.levels = levels
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute grid signal based on price position"""
        if 'close' not in df.columns:
            return pd.Series(0, index=df.index)
        
        close = df['close']
        ma = close.rolling(20).mean()
        std = close.rolling(20).std()
        
        # Signal: distance from mean, normalized
        zscore = (close - ma) / (std + 1e-8)
        
        # Grid signal: buy when oversold, sell when overbought
        signal = -np.tanh(zscore / 2)  # Mean reversion
        
        return signal


class MomentumService(SignalService):
    """Momentum signal service"""
    
    def __init__(self, lookback: int = 20):
        super().__init__("momentum")
        self.lookback = lookback
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute momentum signal"""
        if 'close' not in df.columns:
            return pd.Series(0, index=df.index)
        
        close = df['close']
        mom = close.pct_change(self.lookback)
        
        # Normalize momentum
        signal = np.tanh(mom * 10)
        
        return signal


class RSIService(SignalService):
    """RSI signal service"""
    
    def __init__(self, period: int = 14):
        super().__init__("rsi")
        self.period = period
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute RSI signal"""
        if 'close' not in df.columns:
            return pd.Series(0, index=df.index)
        
        close = df['close']
        delta = close.diff()
        
        gain = delta.where(delta > 0, 0).rolling(self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean()
        
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        # Signal: oversold = buy, overbought = sell
        signal = -(rsi - 50) / 50  # -1 to 1
        
        return signal.fillna(0)


class TrendService(SignalService):
    """Trend following signal service"""
    
    def __init__(self, fast: int = 10, slow: int = 30):
        super().__init__("trend")
        self.fast = fast
        self.slow = slow
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute trend signal"""
        if 'close' not in df.columns:
            return pd.Series(0, index=df.index)
        
        close = df['close']
        fast_ma = close.rolling(self.fast).mean()
        slow_ma = close.rolling(self.slow).mean()
        
        # Trend signal: fast above slow = bullish
        trend = (fast_ma - slow_ma) / slow_ma
        signal = np.tanh(trend * 20)
        
        return signal.fillna(0)


class VolatilityService(SignalService):
    """Volatility-based signal service"""
    
    def __init__(self, lookback: int = 20):
        super().__init__("volatility")
        self.lookback = lookback
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute volatility signal"""
        if 'close' not in df.columns:
            return pd.Series(0, index=df.index)
        
        close = df['close']
        rets = close.pct_change()
        vol = rets.rolling(self.lookback).std()
        
        # Signal: low vol = higher position
        vol_rank = vol.rolling(50).rank(pct=True)
        signal = 1 - vol_rank  # Inverse volatility
        
        return signal.fillna(0)


class VolumeService(SignalService):
    """Volume-based signal service"""
    
    def __init__(self, lookback: int = 20):
        super().__init__("volume")
        self.lookback = lookback
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute volume signal"""
        if 'volume' not in df.columns:
            return pd.Series(0, index=df.index)
        
        vol = df['volume']
        vol_ma = vol.rolling(self.lookback).mean()
        
        # Signal: high volume = strong conviction
        vol_ratio = vol / (vol_ma + 1e-8)
        signal = np.tanh((vol_ratio - 1) * 2)
        
        return signal.fillna(0)


# ==============================================================================
# COMPOSITIONAL MODELS
# ==============================================================================

class CompositionModel:
    """Compose multiple signals into one"""
    
    def __init__(self, name: str):
        self.name = name
        self.weights: Dict[str, float] = {}
        self.operations: Dict[str, str] = {}
    
    def add_signal(self, signal_name: str, weight: float = 1.0, op: str = 'multiply'):
        """Add signal to composition"""
        self.weights[signal_name] = weight
        self.operations[signal_name] = op
        return self
    
    def compose(self, signals: Dict[str, pd.Series]) -> pd.Series:
        """Compose signals"""
        result = None
        
        for name, signal in signals.items():
            if name not in self.weights:
                continue
            
            weight = self.weights[name]
            op = self.operations[name]
            
            weighted = signal * weight
            
            if result is None:
                result = weighted
            elif op == 'multiply':
                result = result * signal
            elif op == 'add':
                result = result + weighted
            elif op == 'takeda':  # Takeda (Japanese multiplication)
                result = result * (1 + signal)
        
        return result if result is not None else pd.Series(0, index=list(signals.values())[0].index)


class SignalMultiplier:
    """Multiply signals to find synergies"""
    
    def __init__(self):
        self.signals: Dict[str, SignalService] = {}
        self.compositions: Dict[str, CompositionModel] = {}
        self._synergy_cache = {}
    
    def register_signal(self, signal: SignalService):
        """Register a signal service"""
        self.signals[signal.name] = signal
        return signal
    
    def create_composition(self, name: str, signals: List[str], weights: List[float] = None) -> CompositionModel:
        """Create a new composition model"""
        model = CompositionModel(name)
        
        for i, sig in enumerate(signals):
            w = weights[i] if weights and i < len(weights) else 1.0
            model.add_signal(sig, w)
        
        self.compositions[name] = model
        return model
    
    def find_synergies(self, df: pd.DataFrame, top_n: int = 10) -> List[Dict]:
        """Find signal combinations with highest synergy"""
        # Compute all individual signals
        computed = {}
        for name, service in self.signals.items():
            computed[name] = service(df)
        
        # Test all pairwise combinations
        synergies = []
        signal_names = list(computed.keys())
        
        for i in range(len(signal_names)):
            for j in range(i + 1, len(signal_names)):
                s1, s2 = signal_names[i], signal_names[j]
                
                # Multiply signals
                combined = computed[s1] * computed[s2]
                
                # Calculate synergy score
                # Synergy = combined performance - sum of individual performances
                if 'close' in df.columns:
                    returns = df['close'].pct_change().shift(-1)
                    
                    ind1_perf = (computed[s1] * returns).mean() / (computed[s1].std() + 1e-8)
                    ind2_perf = (computed[s2] * returns).mean() / (computed[s2].std() + 1e-8)
                    comb_perf = (combined * returns).mean() / (combined.std() + 1e-8)
                    
                    synergy = comb_perf - (ind1_perf + ind2_perf)
                    
                    synergies.append({
                        'signals': (s1, s2),
                        'individual_1': ind1_perf,
                        'individual_2': ind2_perf,
                        'combined': comb_perf,
                        'synergy': synergy,
                        'model': 'multiply'
                    })
                
                # Test Takeda composition
                takeda = computed[s1] * (1 + computed[s2])
                if 'close' in df.columns:
                    takeda_perf = (takeda * returns).mean() / (takeda.std() + 1e-8)
                    takeda_synergy = takeda_perf - (ind1_perf + ind2_perf)
                    
                    synergies.append({
                        'signals': (s1, s2),
                        'individual_1': ind1_perf,
                        'individual_2': ind2_perf,
                        'combined': takeda_perf,
                        'synergy': takeda_synergy,
                        'model': 'takeda'
                    })
        
        # Sort by synergy
        synergies.sort(key=lambda x: x['synergy'], reverse=True)
        
        return synergies[:top_n]


# ==============================================================================
# CONCURRENT ORCHESTRATOR
# ==============================================================================

class Orchestrator:
    """Concurrent lazy service orchestrator"""
    
    def __init__(self, max_workers: int = 4):
        self.services: Dict[str, SignalService] = {}
        self.compositions: Dict[str, CompositionModel] = {}
        self.multiplier = SignalMultiplier()
        self.event_hook = PandasEventHook()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.loop = None
        self._running = False
        self._results: Dict[str, asyncio.Future] = {}
        self._signal_store = LazySignalStore()
        
        # Setup default event listeners
        self._setup_events()
    
    def _setup_events(self):
        """Setup default event listeners"""
        
        async def on_data_update(event: Event):
            """Handle data update events"""
            self._signal_store.clear_cache()
        
        def on_signal_computed(event: Event):
            """Handle signal computed events"""
            pass  # silent — no diagnostic logging
        
        self.event_hook.on('data_update', on_data_update)
        self.event_hook.on('signal_computed', on_signal_computed)
    
    def register_service(self, service: SignalService) -> SignalService:
        """Register a signal service"""
        self.services[service.name] = service
        self.multiplier.register_signal(service)
        
        # Also register as lazy signal
        self._signal_store.register(
            service.name,
            lambda df: service(df),
            deps=[]
        )
        
        return service
    
    def register_composition(self, composition: CompositionModel) -> CompositionModel:
        """Register a composition model"""
        self.compositions[composition.name] = composition
        return composition
    
    async def compute_signal(self, name: str, df: pd.DataFrame) -> pd.Series:
        """Compute a single signal asynchronously"""
        if name in self.services:
            service = self.services[name]
            
            # Run in thread pool for CPU-bound work
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                service,
                df
            )
            
            # Emit event
            await self.event_hook.emit_async(
                'signal_computed',
                df.index[-1] if hasattr(df.index, '__getitem__') else 0,
                {'name': name, 'result': result},
                source='orchestrator'
            )
            
            return result
        
        raise KeyError(f"Service '{name}' not registered")
    
    async def compute_all(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute all signals concurrently"""
        tasks = {
            name: self.compute_signal(name, df)
            for name in self.services
        }
        
        results = {}
        for name, task in tasks.items():
            results[name] = await task
        
        return results
    
    def compute_lazy(self, name: str, df: pd.DataFrame) -> pd.Series:
        """Compute signal lazily (synchronous)"""
        return self._signal_store.get(name, df)
    
    def find_best_compositions(self, df: pd.DataFrame, n: int = 10) -> List[Dict]:
        """Find best signal compositions"""
        return self.multiplier.find_synergies(df, n)
    
    async def run_pipeline(self, df: pd.DataFrame, compositions: List[str] = None) -> Dict[str, Any]:
        """Run full pipeline: compute signals, find synergies, compose"""
        
        # 1. Compute all signals concurrently
        signals = await self.compute_all(df)
        
        # 2. Find synergies
        synergies = self.find_best_compositions(df)
        
        # 3. Apply compositions
        composed = {}
        for name, model in self.compositions.items():
            if compositions and name not in compositions:
                continue
            composed[name] = model.compose(signals)
        
        return {
            'signals': signals,
            'synergies': synergies,
            'compositions': composed,
            'timestamp': df.index[-1] if len(df) > 0 else None
        }
    
    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run pipeline synchronously"""
        return asyncio.run(self.run_pipeline(df))
    
    def start(self):
        """Start orchestrator"""
        self._running = True
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def stop(self):
        """Stop orchestrator"""
        self._running = False
        self.executor.shutdown(wait=True)
        if self.loop:
            self.loop.close()


# ==============================================================================
# DATA LOADER
# ==============================================================================

class DataLoader:
    """Load and prepare data for signal computation using DuckStore"""
    
    def __init__(self, db_path: str = "hrm/data/coinbase.duckdb"):
        self.db_path = db_path
        self._cache = {}
        try:
            from hrm.duck_store import DuckStore
            self.store = DuckStore(db_path)
        except ImportError:
            from duck_store import DuckStore
            self.store = DuckStore(db_path)
    
    @lru_cache(maxsize=32)
    def load_symbol(self, symbol: str, lookback: int = 100) -> pd.DataFrame:
        """Load symbol data with caching"""
        df = self.store.load(symbol)
        
        if len(df) == 0:
            return pd.DataFrame()
        
        df = df.tail(lookback)
        return df
    
    def load_multi(self, symbols: List[str], lookback: int = 100) -> Dict[str, pd.DataFrame]:
        """Load multiple symbols"""
        return {sym: self.load_symbol(sym, lookback) for sym in symbols]


# ==============================================================================
# MAIN DEMO
# ==============================================================================

def main():
    print("="*70)
    print("  CONCURRENT LAZY SIGNAL ORCHESTRATOR")
    print("="*70)
    
    # Initialize
    loader = DataLoader()
    orchestrator = Orchestrator(max_workers=4)
    
    # Register services
    print("\n1. Registering signal services...")
    
    orchestrator.register_service(GridService())
    orchestrator.register_service(MomentumService())
    orchestrator.register_service(RSIService())
    orchestrator.register_service(TrendService())
    orchestrator.register_service(VolatilityService())
    
    print(f"   Registered: {list(orchestrator.services.keys())}")
    
    # Create compositions
    print("\n2. Creating composition models...")
    
    # Grid x Momentum (mean-reversion + trend confirmation)
    orchestrator.register_composition(
        CompositionModel('grid_momentum')
            .add_signal('grid', 0.6)
            .add_signal('momentum', 0.4, op='multiply')
    )
    
    # RSI x Trend (mean-reversion with trend filter)
    orchestrator.register_composition(
        CompositionModel('rsi_trend')
            .add_signal('rsi', 0.7)
            .add_signal('trend', 0.3, op='multiply')
    )
    
    # All signals multiplied
    orchestrator.register_composition(
        CompositionModel('all_multiply')
            .add_signal('grid', 1.0, op='multiply')
            .add_signal('momentum', 1.0, op='multiply')
            .add_signal('rsi', 1.0, op='multiply')
            .add_signal('trend', 1.0, op='multiply')
    )
    
    print(f"   Created: {list(orchestrator.compositions.keys())}")
    
    # Load data
    print("\n3. Loading data...")
    
    symbols = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE']
    data = loader.load_multi(symbols, lookback=200)
    
    print(f"   Loaded {len(data)} symbols")
    for sym, df in data.items():
        print(f"   - {sym}: {len(df)} candles")
    
    # Run pipeline
    print("\n4. Running signal pipeline...")
    
    results = {}
    for sym, df in data.items():
        if len(df) > 30:
            print(f"\n   Processing {sym}...")
            
            pipeline_result = orchestrator.run(df)
            results[sym] = pipeline_result
            
            print(f"   Signals computed: {len(pipeline_result['signals'])}")
            print(f"   Synergies found: {len(pipeline_result['synergies'])}")
    
    # Find best synergies
    print("\n" + "="*70)
    print("  TOP SIGNAL SYNERGIES")
    print("="*70)
    
    all_synergies = []
    for sym, res in results.items():
        for s in res['synergies']:
            s['symbol'] = sym
            all_synergies.append(s)
    
    all_synergies.sort(key=lambda x: x['synergy'], reverse=True)
    
    print(f"\n{'Rank':<5} {'Signals':<25} {'Model':<10} {'Synergy':>10}")
    print("-"*55)
    for i, s in enumerate(all_synergies[:10], 1):
        sigs = f"{s['signals'][0]} x {s['signals'][1]}"
        print(f"{i:<5} {sigs:<25} {s['model']:<10} {s['synergy']:>10.4f}")
    
    # Best composition per symbol
    print("\n" + "="*70)
    print("  BEST COMPOSITION PER SYMBOL")
    print("="*70)
    
    print(f"\n{'Symbol':<10} {'Composition':<20} {'Sharpe':>10} {'Signal Mean':>12}")
    print("-"*55)
    
    for sym, res in results.items():
        best_comp = None
        best_sharpe = -999
        
        for comp_name, comp_signal in res['compositions'].items():
            if comp_signal.std() > 0:
                sharpe = comp_signal.mean() / comp_signal.std()
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_comp = comp_name
        
        if best_comp:
            print(f"{sym:<10} {best_comp:<20} {best_sharpe:>10.4f} {res['compositions'][best_comp].mean():>12.4f}")
    
    # Save results
    print("\n" + "="*70)
    print("  SAVING RESULTS")
    print("="*70)
    
    output = {
        'synergies': [{
            'signals': list(s['signals']),
            'model': s['model'],
            'synergy': s['synergy'],
            'symbol': s['symbol']
        } for s in all_synergies[:20]],
        'best_per_symbol': {}
    }
    
    for sym, res in results.items():
        best = max(res['synergies'], key=lambda x: x['synergy']) if res['synergies'] else None
        if best:
            output['best_per_symbol'][sym] = {
                'signals': list(best['signals']),
                'model': best['model'],
                'synergy': best['synergy']
            }
    
    with open('/Users/jim/work/moneyfan/signal_synergies.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n   Saved to: signal_synergies.json")
    
    print("\n" + "="*70)
    print("  COMPLETE")
    print("="*70)
    
    # Cleanup
    orchestrator.stop()


if __name__ == "__main__":
    main()
