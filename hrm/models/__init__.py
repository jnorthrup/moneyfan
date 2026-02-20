"""
Quant Models - Lazy Computation

Each model:
- Reads instruments lazily
- Uses kernels for computation
- Reports state for HRM selection

Models:
- volatility_breakout: PROVEN WINNER ($37K)
- momentum_trend
- mean_reversion
- sector_rotation
- regime_detection
- composite

BACKLOG: sentiment_analysis (not implemented)
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '..')

from ..instruments import QuantModel
from ..kernels import (
    volatility_breakout_kernel,
    momentum_trend_kernel,
    mean_reversion_kernel,
    cross_sectional_rank,
)


@dataclass
class ModelSpec:
    """Model specification"""
    model_id: str
    description: str
    instruments: List[str]
    kernel: str
    params: Dict
    category: str
    backlog: bool = False


# All model specs
SPECS = {
    # QUANT
    'volatility_breakout': ModelSpec(
        model_id='volatility_breakout',
        description='PROVEN: volatility × breakout ($37K)',
        instruments=['market_data'],
        kernel='volatility_breakout',
        params={'window': 20},
        category='quant',
    ),
    'momentum_trend': ModelSpec(
        model_id='momentum_trend',
        description='momentum × trend',
        instruments=['market_data'],
        kernel='momentum_trend',
        params={'window': 20},
        category='quant',
    ),
    'mean_reversion': ModelSpec(
        model_id='mean_reversion',
        description='mean reversion',
        instruments=['market_data'],
        kernel='mean_reversion',
        params={'window': 20},
        category='quant',
    ),
    # MAPREDUCE
    'cross_sectional': ModelSpec(
        model_id='cross_sectional',
        description='cross-sectional ranking',
        instruments=['market_data'],
        kernel='cross_sectional_rank',
        params={},
        category='mapreduce',
    ),
    # AGENTIC
    'composite': ModelSpec(
        model_id='composite',
        description='composite of models',
        instruments=['model_states'],
        kernel='composite',
        params={},
        category='agentic',
    ),
    # BACKLOG
    'sentiment_analysis': ModelSpec(
        model_id='sentiment_analysis',
        description='BACKLOG: sentiment',
        instruments=['sentiment'],
        kernel='sentiment',
        params={},
        category='quant',
        backlog=True,
    ),
}


def create_model(spec: ModelSpec) -> QuantModel:
    """Create QuantModel from spec"""
    
    def compute_fn(instruments: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        market = instruments.get('market_data', pd.DataFrame())
        if market.empty:
            return pd.DataFrame({'signal': []})
        
        if spec.kernel == 'volatility_breakout':
            s = volatility_breakout_kernel(
                market['open'].values,
                market['high'].values,
                market['low'].values,
                market['close'].values,
                spec.params.get('window', 20)
            )
            return pd.DataFrame({'signal': s}, index=market.index)
        
        elif spec.kernel == 'momentum_trend':
            s = momentum_trend_kernel(market['close'].values, spec.params.get('window', 20))
            return pd.DataFrame({'signal': s}, index=market.index)
        
        elif spec.kernel == 'mean_reversion':
            s = mean_reversion_kernel(market['close'].values, spec.params.get('window', 20))
            return pd.DataFrame({'signal': s}, index=market.index)
        
        elif spec.kernel == 'cross_sectional_rank':
            if 'asset' in market.columns:
                last = market.groupby('asset')['close'].last()
                ranks = cross_sectional_rank(last.values)
                return pd.DataFrame({'asset': last.index, 'signal': ranks})
            return pd.DataFrame({'signal': [0]})
        
        elif spec.kernel == 'composite':
            states = instruments.get('model_states', pd.DataFrame())
            if 'signal' in states.columns:
                return pd.DataFrame({'signal': [states['signal'].mean()]})
            return pd.DataFrame({'signal': [0]})
        
        return pd.DataFrame({'signal': [0]})
    
    return QuantModel(
        model_id=spec.model_id,
        description=spec.description,
        required_instruments=spec.instruments,
        compute_fn=compute_fn,
    )


def create_all_models(exclude_backlog: bool = True) -> Dict[str, QuantModel]:
    """Create all models"""
    return {
        mid: create_model(spec)
        for mid, spec in SPECS.items()
        if not (exclude_backlog and spec.backlog)
    }


def read_states(models: Dict[str, QuantModel]) -> pd.DataFrame:
    """Read model states for HRM"""
    return pd.DataFrame([
        {**m.state, 'model_id': mid, 'category': SPECS[mid].category}
        for mid, m in models.items()
    ])
