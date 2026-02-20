"""
Signal Cache
============

Feather-backed, idempotent per-symbol signal cache.
First run computes orchestrator signals and writes feather; re-runs load via mmap.
"""

import os
import hashlib
import pandas as pd
import pyarrow.feather as feather
from typing import Dict, Optional


class SignalCache:
    """Idempotent per-symbol signal cache stored as Feather files.
    
    Cache key: (symbol, version_hash)
    Storage: {cache_dir}/{SYMBOL}_{version}.feather
    """
    
    def __init__(self, cache_dir: str = "hrm/data/signal_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._version: Optional[str] = None
    
    def _get_path(self, symbol: str) -> str:
        slug = symbol.replace("-", "_").replace("/", "_")
        return os.path.join(self.cache_dir, f"{slug}.feather")
    
    def version_hash(self, orchestrator) -> str:
        """Hash of orchestrator service config for cache busting."""
        if self._version:
            return self._version
        # Hash service names + class names as proxy for config
        parts = sorted(f"{k}:{type(v).__name__}" for k, v in orchestrator.services.items())
        parts += sorted(f"c:{k}" for k in orchestrator.compositions.keys())
        self._version = hashlib.md5("|".join(parts).encode()).hexdigest()[:8]
        return self._version
    
    def get_or_compute(self, symbol: str, df: pd.DataFrame, 
                       orchestrator) -> pd.DataFrame:
        """Load cached signals or compute and cache them.
        
        Returns DataFrame with columns: [grid, momentum, rsi, trend, 
        volatility, volume, composite_alpha, ...] indexed by time.
        """
        path = self._get_path(symbol)
        
        # Check cache hit
        if os.path.exists(path):
            try:
                cached = feather.read_table(path, memory_map=True).to_pandas()
                if 'time' in cached.columns:
                    cached['time'] = pd.to_datetime(cached['time'])
                    cached.set_index('time', inplace=True)
                # Verify coverage: cached must cover the input df range
                if len(cached) >= len(df):
                    return cached
            except Exception:
                pass  # Corrupt cache, recompute
        
        # Cache miss: compute
        result = orchestrator.run(df)
        
        # Build signal DataFrame
        signal_df = pd.DataFrame(index=df.index)
        signal_df.index.name = 'time'
        
        # Individual signals
        for name, series in result.get('signals', {}).items():
            if isinstance(series, pd.Series):
                signal_df[name] = series.values[:len(signal_df)]
        
        # Composed signals
        for name, series in result.get('compositions', {}).items():
            if isinstance(series, pd.Series):
                signal_df[name] = series.values[:len(signal_df)]
        
        # Fill NaN from warmup periods
        signal_df = signal_df.fillna(0.0).astype('float32')
        
        # Write cache
        signal_df.reset_index().to_feather(path, compression='uncompressed')
        
        return signal_df
    
    def has_cache(self, symbol: str) -> bool:
        return os.path.exists(self._get_path(symbol))
    
    def invalidate(self, symbol: str = None):
        """Clear cache. None = clear all."""
        if symbol:
            path = self._get_path(symbol)
            if os.path.exists(path):
                os.remove(path)
        else:
            for f in os.listdir(self.cache_dir):
                if f.endswith('.feather'):
                    os.remove(os.path.join(self.cache_dir, f))
    
    def stats(self) -> Dict[str, int]:
        """Return cache stats."""
        files = [f for f in os.listdir(self.cache_dir) if f.endswith('.feather')]
        total_bytes = sum(
            os.path.getsize(os.path.join(self.cache_dir, f)) for f in files
        )
        return {'symbols': len(files), 'bytes': total_bytes}
