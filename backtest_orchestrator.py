"""
Concurrent Lazy Backtest Service Orchestrator
Uses pandas event hooks to compose signals and identify best compositions

Architecture:
1. Lazy Signal Services - Computed only when needed
2. Pandas Event Hooks - Event-driven backtest triggers
3. Concurrent Backtest Engine - Parallel strategy testing
4. Composition Finder - Identify best signal combinations
5. Performance Analyzer - Calculate Sharpe, DD, returns
"""

import asyncio
import numpy as np
import pandas as pd
from pandas.api.extensions import register_dataframe_accessor
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Any, Tuple
from functools import lru_cache, wraps
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from collections import defaultdict
import json
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# PANDAS EVENT HOOKS
# ==============================================================================

class BacktestEvent:
    """Events fired during backtest"""
    def __init__(self, name: str, timestamp: int, data: Any = None):
        self.name = name
        self.timestamp = timestamp
        self.data = data
        self.source = 'backtest'


class EventHook:
    """Pandas-style event hooks for backtest events"""
    
    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = defaultdict(list)
        self._event_log: List[BacktestEvent] = []
    
    def on(self, event: str, callback: Callable):
        """Register event listener"""
        self._listeners[event].append(callback)
        return self
    
    def emit(self, event: str, timestamp: int, data: Any = None):
        """Emit event to listeners"""
        evt = BacktestEvent(event, timestamp, data)
        self._event_log.append(evt)
        
        for cb in self._listeners.get(event, []):
            try:
                cb(evt)
            except Exception as e:
                print(f"Event error [{event}]: {e}")
    
    def get_events(self, event: str = None) -> List[BacktestEvent]:
        """Get event log"""
        if event:
            return [e for e in self._event_log if e.name == event]
        return self._event_log


@register_dataframe_accessor("backtest")
class BacktestAccessor:
    """Pandas accessor for backtest operations"""
    
    def __init__(self, df):
        self._df = df
        self._hooks = EventHook()
    
    def on_bar(self, callback):
        """Hook for each bar"""
        self._hooks.on('bar', callback)
        return self
    
    def on_signal(self, callback):
        """Hook for signal generation"""
        self._hooks.on('signal', callback)
        return self
    
    def on_trade(self, callback):
        """Hook for trade execution"""
        self._hooks.on('trade', callback)
        return self
    
    def fire(self, event: str, idx: int, data: Any = None):
        """Fire event at index"""
        ts = self._df.index[idx].timestamp() if hasattr(self._df.index[idx], 'timestamp') else idx
        self._hooks.emit(event, ts, data)


# ==============================================================================
# LAZY SIGNALS
# ==============================================================================

class LazySignal:
    """Signal that computes lazily"""
    
    def __init__(self, name: str, func: Callable[[pd.DataFrame], pd.Series]):
        self.name = name
        self.func = func
        self._cache = None
        self._cache_key = None
    
    def __call__(self, df: pd.DataFrame) -> pd.Series:
        # Cache by last timestamp
        key = df.index[-1] if len(df) > 0 else None
        
        if self._cache_key != key:
            self._cache = self.func(df)
            self._cache_key = key
        
        return self._cache
    
    def __repr__(self):
        return f"LazySignal({self.name})"


class SignalRegistry:
    """Registry of lazy signals"""
    
    def __init__(self):
        self._signals: Dict[str, LazySignal] = {}
        self._dependencies: Dict[str, List[str]] = {}
    
    def register(self, name: str, func: Callable, deps: List[str] = None):
        """Register a signal"""
        self._signals[name] = LazySignal(name, func)
        self._dependencies[name] = deps or []
        return self
    
    def get(self, name: str) -> LazySignal:
        """Get signal by name"""
        if name not in self._signals:
            raise KeyError(f"Signal '{name}' not registered")
        return self._signals[name]
    
    def compute(self, name: str, df: pd.DataFrame) -> pd.Series:
        """Compute signal with dependencies"""
        # Compute deps first
        for dep in self._dependencies.get(name, []):
            if dep in self._signals:
                self.compute(dep, df)
        
        return self._signals[name](df)
    
    def compute_all(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute all signals"""
        return {name: self.compute(name, df) for name in self._signals}
    
    def names(self) -> List[str]:
        """Get all signal names"""
        return list(self._signals.keys())


# ==============================================================================
# BUILT-IN SIGNALS
# ==============================================================================

def create_signals() -> SignalRegistry:
    """Create built-in signal registry"""
    
    registry = SignalRegistry()
    
    # Grid signal (mean reversion)
    registry.register('grid', lambda df: 
        -np.tanh((df['close'] - df['close'].rolling(20).mean()) / 
                 (df['close'].rolling(20).std() + 1e-8) / 2)
    )
    
    # Momentum signal
    registry.register('momentum', lambda df:
        np.tanh(df['close'].pct_change(20) * 10).fillna(0)
    )
    
    # RSI signal
    def rsi_signal(df):
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / (loss + 1e-8)))
        return -(rsi - 50) / 50
    registry.register('rsi', rsi_signal)
    
    # Trend signal
    registry.register('trend', lambda df:
        np.tanh((df['close'].rolling(10).mean() - df['close'].rolling(30).mean()) / 
                df['close'].rolling(30).mean() * 20).fillna(0)
    )
    
    # Volatility signal (inverse)
    registry.register('volatility', lambda df:
        (1 - df['close'].pct_change().rolling(20).std().rolling(50).rank(pct=True)).fillna(0.5)
    )
    
    # Mean reversion signal
    registry.register('meanrev', lambda df:
        -np.tanh((df['close'] - df['close'].rolling(50).mean()) / 
                 (df['close'].rolling(50).std() + 1e-8)).fillna(0)
    )
    
    # Breakout signal
    registry.register('breakout', lambda df:
        np.tanh((df['close'] - df['close'].rolling(20).max().shift(1)) / 
                (df['close'].rolling(20).std() + 1e-8) * 5).fillna(0)
    )
    
    # Volume signal
    registry.register('volume', lambda df:
        np.tanh((df['volume'] / df['volume'].rolling(20).mean() - 1) * 2).fillna(0)
        if 'volume' in df.columns else pd.Series(0, index=df.index)
    )
    
    return registry


# ==============================================================================
# COMPOSITION MODELS
# ==============================================================================

class CompositionModel:
    """Model that composes multiple signals"""
    
    def __init__(self, name: str, signals: List[str], weights: List[float] = None, 
                 operation: str = 'multiply'):
        self.name = name
        self.signals = signals
        self.weights = weights or [1.0] * len(signals)
        self.operation = operation
    
    def compose(self, signal_values: Dict[str, pd.Series]) -> pd.Series:
        """Compose signals into final signal"""
        if not self.signals:
            return pd.Series(0, index=list(signal_values.values())[0].index)
        
        result = None
        
        for i, sig_name in enumerate(self.signals):
            if sig_name not in signal_values:
                continue
            
            sig = signal_values[sig_name]
            w = self.weights[i] if i < len(self.weights) else 1.0
            
            weighted = sig * w
            
            if result is None:
                result = weighted.copy()
            elif self.operation == 'multiply':
                result = result * sig
            elif self.operation == 'takeda':  # Japanese multiplication
                result = result * (1 + sig)
            elif self.operation == 'add':
                result = result + weighted
            elif self.operation == 'avg':
                result = result + weighted
        
        if self.operation == 'avg' and result is not None:
            result = result / len(self.signals)
        
        return result if result is not None else pd.Series(0, index=signal_values[self.signals[0]].index)
    
    def __repr__(self):
        return f"Composition({self.name}, {self.signals}, op={self.operation})"


# ==============================================================================
# BACKTEST ENGINE
# ==============================================================================

@dataclass
class BacktestResult:
    """Result of a backtest"""
    name: str
    final_value: float
    pnl: float
    max_drawdown: float
    sharpe: float
    trades: int
    win_rate: float
    signal_values: Dict[str, float] = None


class BacktestEngine:
    """Run backtests on signal compositions"""
    
    def __init__(self, initial_cash: float = 15000, 
                 maker_fee: float = 0.004, taker_fee: float = 0.006):
        self.initial_cash = initial_cash
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.events = EventHook()
    
    def run(self, df: pd.DataFrame, signal: pd.Series, 
            position_size: float = 0.7, name: str = "backtest") -> BacktestResult:
        """Run backtest with given signal"""
        
        cash = self.initial_cash
        position = 0.0
        holdings = 0.0
        trades = 0
        wins = 0
        portfolio_values = []
        entry_price = 0.0
        
        # Ensure signal aligns with df
        signal = signal.reindex(df.index).fillna(0)
        
        for i, (idx, row) in enumerate(df.iterrows()):
            price = row['close']
            sig = signal.iloc[i] if i < len(signal) else 0
            
            # Generate position from signal
            target_position = sig * position_size
            
            # Trade if significant change
            if abs(target_position - position) > 0.05:
                # Close existing position
                if holdings > 0:
                    sell_value = holdings * price
                    fee = sell_value * self.taker_fee
                    cash += sell_value - fee
                    
                    # Track win/loss
                    if price > entry_price:
                        wins += 1
                    
                    trades += 1
                    self.events.emit('trade', i, {
                        'action': 'SELL',
                        'price': price,
                        'quantity': holdings,
                        'signal': sig
                    })
                
                # Open new position
                if target_position != 0:
                    buy_value = min(cash * abs(target_position), cash * 0.95)
                    shares = buy_value / price
                    fee = buy_value * self.maker_fee
                    cash -= buy_value + fee
                    holdings = shares
                    entry_price = price
                    trades += 1
                    
                    self.events.emit('trade', i, {
                        'action': 'BUY',
                        'price': price,
                        'quantity': shares,
                        'signal': sig
                    })
                
                position = target_position
            
            # Update portfolio value
            portfolio_value = cash + holdings * price
            portfolio_values.append(portfolio_value)
            
            self.events.emit('bar', i, {
                'portfolio_value': portfolio_value,
                'cash': cash,
                'holdings': holdings,
                'signal': sig
            })
        
        # Calculate metrics
        final_value = portfolio_values[-1] if portfolio_values else self.initial_cash
        pnl = final_value - self.initial_cash
        
        # Max drawdown
        peak = portfolio_values[0]
        max_dd = 0
        for v in portfolio_values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd
        
        # Sharpe ratio
        returns = pd.Series(portfolio_values).pct_change().dropna()
        sharpe = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252 * 24)  # Hourly
        
        # Win rate
        win_rate = wins / max(trades // 2, 1)
        
        return BacktestResult(
            name=name,
            final_value=final_value,
            pnl=pnl,
            max_drawdown=max_dd,
            sharpe=sharpe,
            trades=trades,
            win_rate=win_rate
        )


# ==============================================================================
# COMPOSITION FINDER
# ==============================================================================

class CompositionFinder:
    """Find best signal compositions"""
    
    def __init__(self, registry: SignalRegistry, engine: BacktestEngine):
        self.registry = registry
        self.engine = engine
        self._results: List[Tuple[CompositionModel, BacktestResult]] = []
    
    def test_single(self, df: pd.DataFrame, signal_name: str) -> BacktestResult:
        """Test a single signal"""
        signal = self.registry.compute(signal_name, df)
        return self.engine.run(df, signal, name=signal_name)
    
    def test_pair(self, df: pd.DataFrame, sig1: str, sig2: str, 
                  operation: str = 'multiply') -> Tuple[CompositionModel, BacktestResult]:
        """Test a signal pair composition"""
        model = CompositionModel(f"{sig1}_{sig2}_{operation}", [sig1, sig2], operation=operation)
        
        signals = self.registry.compute_all(df)
        composed = model.compose(signals)
        
        result = self.engine.run(df, composed, name=model.name)
        return model, result
    
    def find_best(self, df: pd.DataFrame, max_signals: int = 3, 
                  top_n: int = 20) -> List[Tuple[CompositionModel, BacktestResult]]:
        """Find best compositions by testing combinations"""
        
        signal_names = self.registry.names()
        results = []
        
        # Test all pairs
        print("Testing signal pairs...")
        for sig1, sig2 in combinations(signal_names, 2):
            for op in ['multiply', 'takeda', 'add']:
                model, result = self.test_pair(df, sig1, sig2, op)
                results.append((model, result))
        
        # Test triple combinations
        print("Testing triple combinations...")
        for sigs in combinations(signal_names, 3):
            for op in ['multiply', 'takeda']:
                model = CompositionModel("_".join(sigs), list(sigs), operation=op)
                signals = self.registry.compute_all(df)
                composed = model.compose(signals)
                result = self.engine.run(df, composed, name=model.name)
                results.append((model, result))
        
        # Test all signals combined
        print("Testing all signals combined...")
        for op in ['multiply', 'takeda', 'avg']:
            model = CompositionModel(f"all_{op}", signal_names, operation=op)
            signals = self.registry.compute_all(df)
            composed = model.compose(signals)
            result = self.engine.run(df, composed, name=model.name)
            results.append((model, result))
        
        # Sort by final value
        results.sort(key=lambda x: x[1].final_value, reverse=True)
        
        self._results = results
        return results[:top_n]
    
    def get_best_sharpe(self) -> Tuple[CompositionModel, BacktestResult]:
        """Get composition with best Sharpe ratio"""
        if not self._results:
            return None, None
        return max(self._results, key=lambda x: x[1].sharpe)
    
    def get_best_return(self) -> Tuple[CompositionModel, BacktestResult]:
        """Get composition with best return"""
        if not self._results:
            return None, None
        return max(self._results, key=lambda x: x[1].pnl)


# ==============================================================================
# CONCURRENT ORCHESTRATOR
# ==============================================================================

class BacktestOrchestrator:
    """Orchestrate concurrent backtests using DuckStore"""
    
    def __init__(self, db_path: str = "hrm/data/coinbase.duckdb", max_workers: int = 4):
        self.db_path = db_path
        self.max_workers = max_workers
        
        self.registry = create_signals()
        self.engine = BacktestEngine()
        self.finder = CompositionFinder(self.registry, self.engine)
        
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.events = EventHook()
        
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._results: Dict[str, List[Tuple[CompositionModel, BacktestResult]]] = {}
        
        try:
            from hrm.duck_store import DuckStore
            self.store = DuckStore(db_path)
        except ImportError:
            from duck_store import DuckStore
            self.store = DuckStore(db_path)
    
    def load_symbol(self, symbol: str, lookback: int = 500) -> pd.DataFrame:
        """Load symbol data"""
        if symbol in self._data_cache:
            return self._data_cache[symbol]
        
        df = self.store.load(symbol)
        
        if len(df) == 0:
            return pd.DataFrame()
        
        df = df.tail(lookback)
        
        self._data_cache[symbol] = df
        return df
    
    def load_symbols(self, symbols: List[str], lookback: int = 500) -> Dict[str, pd.DataFrame]:
        """Load multiple symbols"""
        return {sym: self.load_symbol(sym, lookback) for sym in symbols}
    
    def run_backtest(self, symbol: str, df: pd.DataFrame = None) -> List[Tuple[CompositionModel, BacktestResult]]:
        """Run full backtest analysis for a symbol"""
        
        if df is None:
            df = self.load_symbol(symbol)
        
        if len(df) < 50:
            print(f"Skipping {symbol}: insufficient data")
            return []
        
        print(f"\nRunning backtest for {symbol}...")
        self.events.emit('backtest_start', 0, {'symbol': symbol})
        
        # Find best compositions
        results = self.finder.find_best(df, top_n=15)
        
        self.events.emit('backtest_end', 0, {
            'symbol': symbol,
            'results': len(results)
        })
        
        self._results[symbol] = results
        return results
    
    def run_concurrent(self, symbols: List[str], lookback: int = 500) -> Dict[str, List]:
        """Run backtests concurrently"""
        
        # Load all data first
        print(f"Loading {len(symbols)} symbols...")
        data = self.load_symbols(symbols, lookback)
        
        # Run backtests concurrently
        print(f"Running {len(symbols)} backtests concurrently...")
        
        futures = {}
        for symbol, df in data.items():
            future = self.executor.submit(self.run_backtest, symbol, df)
            futures[future] = symbol
        
        results = {}
        for future in futures:
            symbol = futures[future]
            try:
                results[symbol] = future.result(timeout=120)
            except Exception as e:
                print(f"Error for {symbol}: {e}")
                results[symbol] = []
        
        return results
    
    def get_aggregate_best(self, metric: str = 'final_value') -> Tuple[str, CompositionModel, BacktestResult]:
        """Get best overall composition across all symbols"""
        
        best = None
        best_symbol = None
        
        for symbol, results in self._results.items():
            for model, result in results:
                val = getattr(result, metric, 0)
                if best is None or val > getattr(best, metric, 0):
                    best = result
                    best_symbol = symbol
                    best_model = model
        
        if best:
            return best_symbol, best_model, best
        return None, None, None
    
    def summary(self) -> pd.DataFrame:
        """Get summary of all results"""
        
        rows = []
        for symbol, results in self._results.items():
            for model, result in results[:5]:  # Top 5 per symbol
                rows.append({
                    'symbol': symbol,
                    'composition': model.name,
                    'signals': ', '.join(model.signals),
                    'operation': model.operation,
                    'final_value': result.final_value,
                    'pnl': result.pnl,
                    'max_dd': result.max_drawdown,
                    'sharpe': result.sharpe,
                    'trades': result.trades,
                    'win_rate': result.win_rate
                })
        
        return pd.DataFrame(rows)
    
    def shutdown(self):
        """Shutdown orchestrator"""
        self.executor.shutdown(wait=True)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("="*80)
    print("  CONCURRENT LAZY BACKTEST SERVICE ORCHESTRATOR")
    print("="*80)
    
    orchestrator = BacktestOrchestrator(max_workers=4)
    
    # Get available symbols from DuckStore
    symbols = orchestrator.store.get_symbols()[:15]
    
    print(f"\nSymbols: {symbols}")
    
    # Register event listeners
    def on_start(evt):
        print(f"  Starting backtest for {evt.data['symbol']}")
    orchestrator.events.on('backtest_start', on_start)
    
    def on_end(evt):
        print(f"  Completed {evt.data['symbol']}: {evt.data['results']} compositions tested")
    orchestrator.events.on('backtest_end', on_end)
    
    # Run concurrent backtests
    print("\n" + "="*80)
    print("  RUNNING CONCURRENT BACKTESTS")
    print("="*80)
    
    start_time = time.time()
    results = orchestrator.run_concurrent(symbols, lookback=500)
    elapsed = time.time() - start_time
    
    print(f"\nCompleted in {elapsed:.1f}s")
    
    # Show results
    print("\n" + "="*80)
    print("  TOP COMPOSITIONS PER SYMBOL")
    print("="*80)
    
    for symbol, res_list in results.items():
        if not res_list:
            continue
        
        print(f"\n{symbol}:")
        print(f"{'Composition':<30} {'Final':>12} {'PnL':>10} {'Sharpe':>8} {'DD':>8}")
        print("-"*70)
        
        for model, result in res_list[:3]:
            print(f"{model.name:<30} ${result.final_value:>11,.0f} ${result.pnl:>9,.0f} {result.sharpe:>8.2f} {result.max_drawdown*100:>7.1f}%")
    
    # Aggregate best
    print("\n" + "="*80)
    print("  BEST OVERALL COMPOSITIONS")
    print("="*80)
    
    summary = orchestrator.summary()
    summary_sorted = summary.sort_values('final_value', ascending=False)
    
    print(f"\n{'Symbol':<10} {'Composition':<25} {'Final':>12} {'PnL':>10} {'Sharpe':>8}")
    print("-"*70)
    
    for _, row in summary_sorted.head(15).iterrows():
        print(f"{row['symbol']:<10} {row['composition']:<25} ${row['final_value']:>11,.0f} ${row['pnl']:>9,.0f} {row['sharpe']:>8.2f}")
    
    # Find best overall
    best_sym, best_model, best_result = orchestrator.get_aggregate_best('final_value')
    
    if best_result:
        print(f"\n🏆 BEST OVERALL:")
        print(f"   Symbol: {best_sym}")
        print(f"   Composition: {best_model.name}")
        print(f"   Signals: {best_model.signals}")
        print(f"   Operation: {best_model.operation}")
        print(f"   Final Value: ${best_result.final_value:,.2f}")
        print(f"   PnL: ${best_result.pnl:,.2f}")
        print(f"   Sharpe: {best_result.sharpe:.2f}")
        print(f"   Max DD: {best_result.max_drawdown*100:.1f}%")
    
    # Save results
    output = {
        'elapsed_seconds': elapsed,
        'symbols_tested': len(results),
        'best_overall': {
            'symbol': best_sym,
            'composition': best_model.name if best_model else None,
            'signals': best_model.signals if best_model else None,
            'operation': best_model.operation if best_model else None,
            'final_value': best_result.final_value if best_result else None,
            'sharpe': best_result.sharpe if best_result else None,
            'max_dd': best_result.max_drawdown if best_result else None
        },
        'per_symbol': {}
    }
    
    for symbol, res_list in results.items():
        if res_list:
            best_model, best_result = res_list[0]
            output['per_symbol'][symbol] = {
                'best_composition': best_model.name,
                'signals': best_model.signals,
                'operation': best_model.operation,
                'final_value': best_result.final_value,
                'pnl': best_result.pnl,
                'sharpe': best_result.sharpe,
                'max_dd': best_result.max_drawdown,
                'trades': best_result.trades
            }
    
    with open('/Users/jim/work/moneyfan/backtest_compositions.json', 'w') as f:
        json.dump(output, f, indent=2, default=float)
    
    print("\n" + "="*80)
    print("  SAVED TO: backtest_compositions.json")
    print("="*80)
    
    # Signal synergy analysis
    print("\n" + "="*80)
    print("  SIGNAL SYNERGY ANALYSIS")
    print("="*80)
    
    # Count which signals appear most in top compositions
    signal_counts = defaultdict(int)
    signal_performance = defaultdict(list)
    
    for _, res_list in results.items():
        for model, result in res_list[:5]:
            for sig in model.signals:
                signal_counts[sig] += 1
                signal_performance[sig].append(result.final_value)
    
    print(f"\n{'Signal':<15} {'Appearances':>12} {'Avg Final':>12}")
    print("-"*40)
    
    for sig, count in sorted(signal_counts.items(), key=lambda x: -x[1]):
        avg_final = np.mean(signal_performance[sig])
        print(f"{sig:<15} {count:>12} ${avg_final:>11,.0f}")
    
    # Cleanup
    orchestrator.shutdown()
    
    print("\n" + "="*80)
    print("  COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
