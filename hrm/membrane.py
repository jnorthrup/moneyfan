"""
Coinbase Prioritization Membrane

The membrane is a semi-permeable boundary that:
- Filters noise (bad pairs)
- Passes signal (good pairs)  
- Adapts to conditions (dynamic prioritization)

OPTIMAL MEMBRANE PRESCRIPTION:

┌─────────────────────────────────────────────────────────────────┐
│                    COINBASE RAW DATA (~500 pairs)               │
│                                                                 │
│  All pairs, all noise, all volume                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 1: HARD FILTER                         │
│                                                                 │
│  Remove:                                                        │
│  - Volume < $1M/24h (insufficient liquidity)                    │
│  - Spread > 1% (too expensive to trade)                         │
│  - Age < 30 days (unstable pairs)                               │
│  - Delisted/suspended pairs                                     │
│                                                                 │
│  Pass: ~200 pairs                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 2: DEPTH SCORING                       │
│                                                                 │
│  Score = f(route_depth, liquidity_depth, correlation_value)     │
│                                                                 │
│  Route Depth: How many paths use this pair (betweenness)        │
│  Liquidity Depth: Order book depth at ±1%, ±2%, ±5%            │
│  Correlation Value: Information content (low corr = high value) │
│                                                                 │
│  Rank and select top 128                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 3: REGIME ALLOCATION                   │
│                                                                 │
│  Allocate 128 slots by regime:                                  │
│                                                                 │
│  TRENDING: momentum pairs (60 slots)                            │
│    - High β to market, volume surges, breakout candidates       │
│                                                                 │
│  RANGING: volatility pairs (40 slots)                           │
│    - Stable spreads, mean-reverting, grid-friendly              │
│                                                                 │
│  TRANSITION: cross-sectional pairs (28 slots)                   │
│    - Low correlation, sector rotation, arbitrage opportunities  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 4: TIMEFRAME FRACTALS                  │
│                                                                 │
│  Split each regime allocation into timeframes:                  │
│                                                                 │
│  M5:  40% - microstructure, execution                           │
│  M15: 25% - short-term signals                                  │
│  H1:  20% - hourly patterns                                     │
│  H4:  10% - swing trades                                        │
│  D1:  5%  - macro regime                                        │
│                                                                 │
│  Total: 128 pairs × 5 timeframes = 640 parallel observations    │
│  But only 128 computed per tick (lazy)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT: HRM INPUT TENSOR                     │
│                                                                 │
│  Shape: [batch=1, currencies, features, timeframes]             │
│  Sparse: Only 128 active pairs per frame                        │
│  Mask: Which pairs are active this frame                        │
└─────────────────────────────────────────────────────────────────┘
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum
import numpy as np


# =============================================================================
# REGIME DEFINITIONS
# =============================================================================

class MarketRegime(Enum):
    TRENDING = "trending"       # Momentum, breakouts, trend-following
    RANGING = "ranging"         # Mean-reversion, grid, volatility-sell
    TRANSITION = "transition"   # Uncertain, need diversification
    VOLATILE = "volatile"       # High uncertainty, reduce size


# =============================================================================
# MEMBRANE LAYERS
# =============================================================================

@dataclass
class Layer1Filter:
    """
    Layer 1: Hard Filter
    
    Remove pairs that don't meet minimum criteria.
    """
    min_volume_24h: float = 1_000_000      # $1M daily volume
    max_spread: float = 0.01               # 1% max spread
    min_age_days: int = 30                 # 30 days minimum
    
    def passes(self, pair_data: Dict) -> bool:
        """Check if pair passes hard filter"""
        if pair_data.get('volume_24h', 0) < self.min_volume_24h:
            return False
        if pair_data.get('spread', 1) > self.max_spread:
            return False
        if pair_data.get('age_days', 0) < self.min_age_days:
            return False
        if pair_data.get('status') in ['delisted', 'suspended']:
            return False
        return True


@dataclass
class Layer2Score:
    """
    Layer 2: Depth Scoring
    
    Score pairs by route depth, liquidity depth, correlation value.
    """
    # Weights
    route_depth_weight: float = 0.4
    liquidity_depth_weight: float = 0.4
    correlation_value_weight: float = 0.2
    
    def score(self, 
              pair_data: Dict,
              route_depth: float,
              liquidity_depth: float,
              correlation: float) -> float:
        """
        Compute composite depth score.
        
        Higher = better to include.
        """
        # Route depth: normalized betweenness (0-1)
        route_score = min(route_depth, 1.0)
        
        # Liquidity depth: log of order book depth
        liq_score = np.log10(liquidity_depth + 1) / 8  # Normalize to ~0-1
        
        # Correlation value: low correlation = high value
        corr_score = 1.0 - min(abs(correlation), 1.0)
        
        return (
            self.route_depth_weight * route_score +
            self.liquidity_depth_weight * liq_score +
            self.correlation_value_weight * corr_score
        )


@dataclass
class Layer3Allocation:
    """
    Layer 3: Regime Allocation
    
    Allocate 128 slots based on detected regime.
    """
    total_slots: int = 128
    
    # Allocation by regime
    allocations = {
        MarketRegime.TRENDING: {
            'momentum': 60,      # High β pairs
            'cross_sectional': 20,
            'volatility': 20,
        },
        MarketRegime.RANGING: {
            'volatility': 40,    # Stable spread pairs
            'mean_reversion': 48,
            'cross_sectional': 40,
        },
        MarketRegime.TRANSITION: {
            'cross_sectional': 64,  # Diversification
            'volatility': 32,
            'momentum': 32,
        },
        MarketRegime.VOLATILE: {
            'volatility': 96,    # Vol sellers
            'cross_sectional': 32,
        },
    }
    
    def allocate(self, regime: MarketRegime) -> Dict[str, int]:
        """Get allocation for regime"""
        return self.allocations.get(regime, self.allocations[MarketRegime.TRANSITION])


@dataclass
class Layer4Fractals:
    """
    Layer 4: Timeframe Fractals
    
    Split each category into timeframes.
    """
    timeframe_weights = {
        'M5': 0.40,   # Microstructure
        'M15': 0.25,  # Short-term
        'H1': 0.20,   # Hourly
        'H4': 0.10,   # Swing
        'D1': 0.05,   # Macro
    }
    
    def split(self, n_pairs: int) -> Dict[str, int]:
        """Split n pairs into timeframe buckets"""
        result = {}
        remaining = n_pairs
        
        timeframes = list(self.timeframe_weights.keys())
        for i, tf in enumerate(timeframes[:-1]):
            n = int(n_pairs * self.timeframe_weights[tf])
            result[tf] = n
            remaining -= n
        
        result[timeframes[-1]] = remaining
        return result


# =============================================================================
# MEMBRANE CONFIGURATION
# =============================================================================

@dataclass
class MembraneConfig:
    """
    Complete membrane configuration.
    
    This is the PRESCRIPTION for optimal Coinbase prioritization.
    """
    # Layer 1: Hard filter
    min_volume_24h: float = 1_000_000
    max_spread: float = 0.01
    min_age_days: int = 30
    
    # Layer 2: Scoring weights
    route_depth_weight: float = 0.4
    liquidity_depth_weight: float = 0.4
    correlation_value_weight: float = 0.2
    
    # Layer 3: Allocation
    total_slots: int = 128
    
    # Layer 4: Timeframes
    timeframes: List[str] = field(default_factory=lambda: ['M5', 'M15', 'H1', 'H4', 'D1'])
    
    # Adaptive parameters
    regime_detection_lookback: int = 100
    correlation_lookback: int = 50
    refresh_rate: float = 0.05  # 5% of pairs refreshed per tick


# =============================================================================
# MEMBRANE STATE
# =============================================================================

@dataclass
class MembraneState:
    """Current state of the membrane"""
    regime: MarketRegime = MarketRegime.TRANSITION
    
    # Active pairs per layer
    layer1_passed: Set[str] = field(default_factory=set)
    layer2_scored: List[Tuple[str, float]] = field(default_factory=list)
    layer3_allocated: Dict[str, List[str]] = field(default_factory=dict)
    layer4_fractal: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    
    # Current frame
    active_pairs: List[str] = field(default_factory=list)
    active_timeframes: Dict[str, str] = field(default_factory=dict)
    
    # Metrics
    n_total_pairs: int = 0
    n_filtered: int = 0
    n_active: int = 0


# =============================================================================
# MEMBRANE
# =============================================================================

class CoinbaseMembrane:
    """
    Optimal Coinbase Prioritization Membrane
    
    Implements the 4-layer prescription for filtering and prioritizing
    Coinbase trading pairs for HRM input.
    """
    
    def __init__(self, config: MembraneConfig = None):
        self.config = config or MembraneConfig()
        
        # Layers
        self.layer1 = Layer1Filter(
            min_volume_24h=config.min_volume_24h if config else 1_000_000,
            max_spread=config.max_spread if config else 0.01,
            min_age_days=config.min_age_days if config else 30,
        )
        self.layer2 = Layer2Score()
        self.layer3 = Layer3Allocation()
        self.layer4 = Layer4Fractals()
        
        # State
        self.state = MembraneState()
        
        # Data (would be populated from Coinbase API)
        self.pair_data: Dict[str, Dict] = {}
        self.route_depths: Dict[str, float] = {}
        self.liquidity_depths: Dict[str, float] = {}
        self.correlations: Dict[str, float] = {}
    
    def process(self, all_pairs: Dict[str, Dict]) -> MembraneState:
        """
        Process all pairs through the membrane.
        
        Args:
            all_pairs: {pair_symbol: {volume_24h, spread, age_days, ...}}
        
        Returns:
            MembraneState with active pairs
        """
        self.state.n_total_pairs = len(all_pairs)
        
        # Layer 1: Hard filter
        self.state.layer1_passed = {
            symbol for symbol, data in all_pairs.items()
            if self.layer1.passes(data)
        }
        self.state.n_filtered = len(self.state.layer1_passed)
        
        # Layer 2: Score remaining pairs
        scored = []
        for symbol in self.state.layer1_passed:
            data = all_pairs.get(symbol, {})
            score = self.layer2.score(
                pair_data=data,
                route_depth=self.route_depths.get(symbol, 0.5),
                liquidity_depth=self.liquidity_depths.get(symbol, 1_000_000),
                correlation=self.correlations.get(symbol, 0.0),
            )
            scored.append((symbol, score))
        
        # Rank and take top N
        scored.sort(key=lambda x: -x[1])
        self.state.layer2_scored = scored[:self.config.total_slots * 2]  # Keep 2x for regime switching
        
        # Layer 3: Allocate by regime
        allocation = self.layer3.allocate(self.state.regime)
        self.state.layer3_allocated = {cat: [] for cat in allocation}
        
        # Categorize pairs
        for symbol, score in self.state.layer2_scored:
            # Simple categorization based on pair properties
            if '-USD' in symbol:
                # Check volatility
                vol = self.pair_data.get(symbol, {}).get('volatility', 0.02)
                corr = self.correlations.get(symbol, 0.5)
                
                if vol > 0.03:
                    category = 'volatility'
                elif abs(corr) > 0.7:
                    category = 'momentum'
                else:
                    category = 'cross_sectional'
            else:
                category = 'cross_sectional'
            
            # Add to category if space available
            if category in allocation:
                if len(self.state.layer3_allocated[category]) < allocation[category]:
                    self.state.layer3_allocated[category].append(symbol)
        
        # Layer 4: Split into timeframes
        self.state.layer4_fractal = {}
        for category, pairs in self.state.layer3_allocated.items():
            timeframe_split = self.layer4.split(len(pairs))
            self.state.layer4_fractal[category] = {}
            
            idx = 0
            for tf, n in timeframe_split.items():
                self.state.layer4_fractal[category][tf] = pairs[idx:idx+n]
                idx += n
        
        # Compile active pairs
        self.state.active_pairs = []
        self.state.active_timeframes = {}
        
        for category, tf_dict in self.state.layer4_fractal.items():
            for tf, pairs in tf_dict.items():
                for pair in pairs:
                    self.state.active_pairs.append(pair)
                    self.state.active_timeframes[pair] = tf
        
        self.state.n_active = len(self.state.active_pairs)
        
        return self.state
    
    def update_regime(self, regime: MarketRegime):
        """Update detected market regime"""
        self.state.regime = regime
    
    def get_frame(self, n: int = 128) -> List[str]:
        """Get current frame of active pairs"""
        return self.state.active_pairs[:n]
    
    def summary(self) -> str:
        """Get membrane summary"""
        return f"""
COINBASE MEMBRANE STATE
=======================
Total pairs:    {self.state.n_total_pairs}
Layer 1 passed: {self.state.n_filtered}
Layer 2 scored: {len(self.state.layer2_scored)}
Layer 3 active: {self.state.n_active}

Regime: {self.state.regime.value}

Allocation:
{self._format_allocation()}
"""
    
    def _format_allocation(self) -> str:
        lines = []
        for cat, pairs in self.state.layer3_allocated.items():
            tf_str = ", ".join(
                f"{tf}:{len(p)}" 
                for tf, p in self.state.layer4_fractal.get(cat, {}).items()
            )
            lines.append(f"  {cat}: {len(pairs)} pairs ({tf_str})")
        return "\n".join(lines)


# =============================================================================
# OPTIMAL PRESCRIPTION
# =============================================================================

OPTIMAL_PRESCRIPTION = """
OPTIMAL COINBASE PRIORITIZATION MEMBRANE
=========================================

LAYER 1: HARD FILTER (removes ~60% of pairs)
─────────────────────────────────────────────
• Volume ≥ $1M/24h
• Spread ≤ 1%
• Age ≥ 30 days
• Status = active

Why: Eliminates illiquid, expensive, and unstable pairs.
Cost: ~0.1% of trades affected (bad trades anyway).

LAYER 2: DEPTH SCORING (ranks remaining pairs)
──────────────────────────────────────────────
Score = 0.4 × route_depth + 0.4 × liquidity_depth + 0.2 × correlation_value

Route depth: Betweenness centrality in currency graph
  → Pairs that enable multi-hop routing
  
Liquidity depth: Order book depth at ±1%, ±2%, ±5%
  → Pairs with deep books (not just high volume)
  
Correlation value: 1 - |correlation|
  → Pairs that provide unique information

Why: Prioritizes pairs that are USEFUL, not just POPULAR.
Example: ETH-USD might rank higher than BTC-USD if ETH has more cross-pairs.

LAYER 3: REGIME ALLOCATION (allocates 128 slots)
─────────────────────────────────────────────────
TRENDING regime (momentum strong):
  • 60 momentum pairs (high β, volume surges)
  • 20 cross-sectional (diversification)
  • 20 volatility (hedge)
  
RANGING regime (mean-reversion):
  • 40 volatility pairs (grid-friendly)
  • 48 mean-reversion (stable spreads)
  • 40 cross-sectional (sector rotation)
  
TRANSITION regime (uncertain):
  • 64 cross-sectional (max diversification)
  • 32 volatility + 32 momentum
  
VOLATILE regime (high uncertainty):
  • 96 volatility (short vol)
  • 32 cross-sectional (arbitrage)

Why: Different strategies work in different regimes.
Don't use momentum pairs in a ranging market.

LAYER 4: TIMEFRAME FRACTALS (splits into 5 timeframes)
───────────────────────────────────────────────────────
M5:  40% → Microstructure, execution optimization
M15: 25% → Short-term signals (volatility breakout)
H1:  20% → Hourly patterns (intraday momentum)
H4:  10% → Swing trades (multi-day)
D1:  5%  → Macro regime (weekly trends)

Why: Signals exist at multiple timescales.
A pair might have M5 mean-reversion but H4 trend.

REFRESH STRATEGY
────────────────
• Each tick: Refresh 5% of pairs (rotating)
• Stale pairs (>1 timeframe age): Force refresh
• New pairs: Add to bag if depth score > min active

Why: Continuous adaptation without full recomputation.

OUTPUT TO HRM
─────────────
Shape: [batch, currencies, features, timeframes]
Sparse mask: Which of 128 slots are active this frame
Metadata: Regime, allocation, staleness

The HRM receives a CURATED view, not raw noise.
"""


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(OPTIMAL_PRESCRIPTION)
    
    print("\n" + "="*60)
    print("MEMBRANE DEMO")
    print("="*60)
    
    # Create membrane
    config = MembraneConfig()
    membrane = CoinbaseMembrane(config)
    
    # Fake pair data
    fake_pairs = {
        'BTC-USD': {'volume_24h': 10_000_000_000, 'spread': 0.0001, 'age_days': 2000, 'volatility': 0.02},
        'ETH-USD': {'volume_24h': 5_000_000_000, 'spread': 0.0001, 'age_days': 2000, 'volatility': 0.025},
        'SOL-USD': {'volume_24h': 1_000_000_000, 'spread': 0.0002, 'age_days': 500, 'volatility': 0.04},
        'DOGE-USD': {'volume_24h': 300_000_000, 'spread': 0.0003, 'age_days': 1000, 'volatility': 0.06},
        'AVAX-USD': {'volume_24h': 100_000_000, 'spread': 0.0003, 'age_days': 365, 'volatility': 0.035},
        'LINK-USD': {'volume_24h': 100_000_000, 'spread': 0.0003, 'age_days': 1000, 'volatility': 0.03},
        'DOT-USD': {'volume_24h': 100_000_000, 'spread': 0.0003, 'age_days': 1000, 'volatility': 0.03},
        'UNI-USD': {'volume_24h': 50_000_000, 'spread': 0.0004, 'age_days': 500, 'volatility': 0.035},
        # Add some pairs that should be filtered
        'TINY-USD': {'volume_24h': 100_000, 'spread': 0.001, 'age_days': 10, 'volatility': 0.1},
        'WIDE-USD': {'volume_24h': 5_000_000, 'spread': 0.05, 'age_days': 100, 'volatility': 0.1},
    }
    
    # Add route depths
    membrane.route_depths = {
        'BTC-USD': 0.9, 'ETH-USD': 0.95, 'SOL-USD': 0.6,
        'DOGE-USD': 0.3, 'AVAX-USD': 0.5, 'LINK-USD': 0.4,
        'DOT-USD': 0.4, 'UNI-USD': 0.35,
    }
    
    # Add liquidity depths
    membrane.liquidity_depths = {
        'BTC-USD': 100_000_000, 'ETH-USD': 50_000_000,
        'SOL-USD': 10_000_000, 'DOGE-USD': 5_000_000,
    }
    
    # Process
    membrane.pair_data = fake_pairs
    state = membrane.process(fake_pairs)
    
    print(membrane.summary())
    
    print("\nActive pairs (first 10):")
    for pair in state.active_pairs[:10]:
        tf = state.active_timeframes.get(pair, 'M5')
        print(f"  {pair} ({tf})")
