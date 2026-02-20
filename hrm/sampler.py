"""
Fractal Sampler - 128 Pair Routing

Strategy:
1. Full graph: ~500 pairs (Coinbase)
2. Cap at 128 pairs by depth ranking
3. Per frame: randomly sample 128 from bag
4. Fractal scales: multiple bags at different timeframes
5. Continuous pull: refresh % of bag each tick

The "bag" is a reservoir of updates to watch.
Random sampling ensures we don't miss opportunities.
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from collections import deque
from enum import Enum
import time

# Import from parent
import sys
sys.path.insert(0, '.')
from currency_graph import Currency, Pair, CurrencyGraph, DepthBasedRouter


# =============================================================================
# TIMEFRAMES (FRACTAL SCALES)
# =============================================================================

class Timeframe(Enum):
    M5 = "5min"
    M15 = "15min"
    H1 = "1hour"
    H4 = "4hour"
    D1 = "daily"
    
    @property
    def seconds(self) -> int:
        return {
            Timeframe.M5: 300,
            Timeframe.M15: 900,
            Timeframe.H1: 3600,
            Timeframe.H4: 14400,
            Timeframe.D1: 86400,
        }[self]
    
    @property
    def priority(self) -> int:
        """Lower = higher priority for sampling"""
        return {
            Timeframe.M5: 0,
            Timeframe.M15: 1,
            Timeframe.H1: 2,
            Timeframe.H4: 3,
            Timeframe.D1: 4,
        }[self]


# =============================================================================
# PAIR BAG
# =============================================================================

@dataclass
class PairUpdate:
    """A pending update for a pair"""
    pair: Pair
    timeframe: Timeframe
    last_update: float = 0.0
    priority: float = 0.0
    staleness: float = 0.0
    
    def compute_staleness(self, now: float) -> float:
        """How stale is this update"""
        self.staleness = (now - self.last_update) / self.timeframe.seconds
        return self.staleness


class PairBag:
    """
    A bag of pair updates to sample from.
    
    Maintains priority based on:
    - Staleness (how long since update)
    - Depth score (importance of pair)
    - Volatility (recent price movement)
    """
    
    def __init__(self, max_size: int = 256):
        self.max_size = max_size
        self.updates: Dict[Pair, PairUpdate] = {}
        self._dirty = True
        self._sorted: List[PairUpdate] = []
    
    def add(self, pair: Pair, timeframe: Timeframe, priority: float = 0.0):
        """Add a pair to the bag"""
        now = time.time()
        update = PairUpdate(
            pair=pair,
            timeframe=timeframe,
            last_update=now,
            priority=priority,
        )
        self.updates[pair] = update
        self._dirty = True
        
        # Trim if over capacity
        if len(self.updates) > self.max_size:
            self._trim()
    
    def update(self, pair: Pair):
        """Mark a pair as updated (reset staleness)"""
        if pair in self.updates:
            self.updates[pair].last_update = time.time()
            self._dirty = True
    
    def compute_priorities(self, depth_scores: Dict[Currency, float]):
        """Compute priority for all pairs"""
        now = time.time()
        
        for update in self.updates.values():
            staleness = update.compute_staleness(now)
            depth = (depth_scores.get(update.pair.base, 0.5) + 
                    depth_scores.get(update.pair.quote, 0.5)) / 2
            
            # Priority = staleness * depth
            # Higher = more urgent to update
            update.priority = staleness * depth
        
        self._dirty = True
    
    def sample(self, n: int = 128, method: str = "priority") -> List[Pair]:
        """
        Sample n pairs from the bag.
        
        Methods:
          - "random": uniform random
          - "priority": weighted by priority (staleness * depth)
          - "mixed": 50% priority, 50% random
        """
        if not self.updates:
            return []
        
        n = min(n, len(self.updates))
        
        if method == "random":
            return random.sample(list(self.updates.keys()), n)
        
        elif method == "priority":
            if self._dirty:
                self._sorted = sorted(
                    self.updates.values(),
                    key=lambda u: -u.priority
                )
                self._dirty = False
            return [u.pair for u in self._sorted[:n]]
        
        elif method == "mixed":
            n_priority = n // 2
            n_random = n - n_priority
            
            if self._dirty:
                self._sorted = sorted(
                    self.updates.values(),
                    key=lambda u: -u.priority
                )
                self._dirty = False
            
            priority_pairs = [u.pair for u in self._sorted[:n_priority]]
            
            remaining = [p for p in self.updates.keys() if p not in priority_pairs]
            random_pairs = random.sample(remaining, min(n_random, len(remaining)))
            
            return priority_pairs + random_pairs
        
        return []
    
    def _trim(self):
        """Remove lowest priority pairs"""
        if len(self.updates) <= self.max_size:
            return
        
        sorted_updates = sorted(
            self.updates.values(),
            key=lambda u: u.priority
        )
        
        # Keep top max_size
        self.updates = {u.pair: u for u in sorted_updates[-self.max_size:]}
        self._dirty = True
    
    def __len__(self):
        return len(self.updates)


# =============================================================================
# FRACTAL SAMPLER
# =============================================================================

@dataclass
class SampledFrame:
    """A frame of sampled pairs across timeframes"""
    timestamp: float
    pairs: List[Pair]
    timeframes: Dict[Pair, Timeframe]
    depths: Dict[Pair, float]
    
    def get_by_timeframe(self, tf: Timeframe) -> List[Pair]:
        """Get pairs for a specific timeframe"""
        return [p for p in self.pairs if self.timeframes.get(p) == tf]


class FractalSampler:
    """
    Samples 128 pairs per frame from multi-timescale bags.
    
    Architecture:
      Full Graph (~500 pairs)
           ↓
      Rank by Depth → Cap to top 256
           ↓
      Split into Timeframe Bags (M5, M15, H1, H4, D1)
           ↓
      Per Frame: Sample 128 total
        - M5: 64 pairs (most urgent)
        - M15: 32 pairs
        - H1: 16 pairs
        - H4: 8 pairs
        - D1: 8 pairs
           ↓
      Continuous Pull: Each tick refreshes stale pairs
    """
    
    def __init__(self, 
                 graph: CurrencyGraph,
                 max_pairs: int = 256,
                 frame_size: int = 128):
        self.graph = graph
        self.max_pairs = max_pairs
        self.frame_size = frame_size
        
        # Depth metrics
        self.router = DepthBasedRouter(graph)
        self.depth_scores = graph.get_depth_metrics()
        
        # Per-timeframe bags
        self.bags: Dict[Timeframe, PairBag] = {
            tf: PairBag(max_size=max_pairs // 4)
            for tf in Timeframe
        }
        
        # Initialize bags with top pairs by depth
        self._initialize_bags()
        
        # Sampling allocation per timeframe
        self.allocation = {
            Timeframe.M5: 64,
            Timeframe.M15: 32,
            Timeframe.H1: 16,
            Timeframe.H4: 8,
            Timeframe.D1: 8,
        }
        
        # History
        self.frames: deque = deque(maxlen=100)
    
    def _initialize_bags(self):
        """Initialize bags with top-depth pairs"""
        # Get top pairs by combined depth score
        pair_scores = []
        for pair in self.graph.pairs:
            base_score = self.depth_scores.get(pair.base, type('obj', (), {'depth_score': 0.5})()).depth_score
            quote_score = self.depth_scores.get(pair.quote, type('obj', (), {'depth_score': 0.5})()).depth_score
            combined = (base_score + quote_score) / 2
            pair_scores.append((pair, combined))
        
        # Sort and cap
        pair_scores.sort(key=lambda x: -x[1])
        top_pairs = [p for p, _ in pair_scores[:self.max_pairs]]
        
        # Distribute across timeframes
        # M5 gets most volatile, D1 gets most stable
        for i, pair in enumerate(top_pairs):
            # Assign to appropriate timeframe
            if i < len(top_pairs) * 0.5:
                tf = Timeframe.M5
            elif i < len(top_pairs) * 0.7:
                tf = Timeframe.M15
            elif i < len(top_pairs) * 0.85:
                tf = Timeframe.H1
            elif i < len(top_pairs) * 0.95:
                tf = Timeframe.H4
            else:
                tf = Timeframe.D1
            
            self.bags[tf].add(pair, tf)
    
    def sample_frame(self, method: str = "mixed") -> SampledFrame:
        """
        Sample a frame of 128 pairs.
        
        Returns pairs from all timeframes, weighted by allocation.
        """
        now = time.time()
        
        # Compute priorities for each bag
        depth_dict = {cur: m.depth_score for cur, m in self.depth_scores.items()}
        for bag in self.bags.values():
            bag.compute_priorities(depth_dict)
        
        # Sample from each timeframe
        all_pairs = []
        timeframes = {}
        depths = {}
        
        for tf, n in self.allocation.items():
            pairs = self.bags[tf].sample(n, method=method)
            
            for pair in pairs:
                all_pairs.append(pair)
                timeframes[pair] = tf
                depths[pair] = depth_dict.get(pair.base, 0.5) + depth_dict.get(pair.quote, 0.5)
        
        frame = SampledFrame(
            timestamp=now,
            pairs=all_pairs,
            timeframes=timeframes,
            depths=depths,
        )
        
        self.frames.append(frame)
        return frame
    
    def refresh_bag(self, refresh_pct: float = 0.1):
        """
        Refresh a percentage of the bag.
        
        Pulls in new pairs from the graph, replaces lowest priority.
        """
        n_refresh = max(1, int(self.max_pairs * refresh_pct))
        
        # Get pairs not in any bag
        all_in_bag = set()
        for bag in self.bags.values():
            all_in_bag.update(bag.updates.keys())
        
        not_in_bag = [p for p in self.graph.pairs if p not in all_in_bag]
        
        if not not_in_bag:
            return
        
        # Add some new pairs
        new_pairs = random.sample(not_in_bag, min(n_refresh, len(not_in_bag)))
        
        for pair in new_pairs:
            # Assign to timeframe
            tf = random.choice(list(Timeframe))
            self.bags[tf].add(pair, tf)
    
    def mark_updated(self, pairs: List[Pair]):
        """Mark pairs as updated (reset staleness)"""
        for pair in pairs:
            for bag in self.bags.values():
                if pair in bag.updates:
                    bag.update(pair)
                    break
    
    def get_stale_pairs(self, threshold: float = 1.0) -> List[Tuple[Pair, Timeframe, float]]:
        """
        Get pairs that are stale (need update).
        
        Returns: [(pair, timeframe, staleness)]
        """
        now = time.time()
        stale = []
        
        for tf, bag in self.bags.items():
            for update in bag.updates.values():
                staleness = update.compute_staleness(now)
                if staleness > threshold:
                    stale.append((update.pair, tf, staleness))
        
        return sorted(stale, key=lambda x: -x[2])
    
    def stats(self) -> Dict:
        """Get sampler statistics"""
        return {
            "total_pairs": len(self.graph.pairs),
            "max_pairs": self.max_pairs,
            "frame_size": self.frame_size,
            "bag_sizes": {tf.name: len(bag) for tf, bag in self.bags.items()},
            "frames_sampled": len(self.frames),
        }


# =============================================================================
# TICK PROCESSOR
# =============================================================================

class TickProcessor:
    """
    Processes ticks from sampled pairs.
    
    Each tick:
    1. Sample 128 pairs from fractal bags
    2. Route through depth-based graph
    3. Update HRM with new data
    4. Refresh stale pairs
    """
    
    def __init__(self, sampler: FractalSampler):
        self.sampler = sampler
        self.tick_count = 0
    
    def tick(self) -> Dict:
        """Process one tick"""
        self.tick_count += 1
        
        # Sample frame
        frame = self.sampler.sample_frame(method="mixed")
        
        # Get stale pairs for next refresh
        stale = self.sampler.get_stale_pairs(threshold=0.5)
        
        # Refresh bag periodically
        if self.tick_count % 10 == 0:
            self.sampler.refresh_bag(refresh_pct=0.05)
        
        return {
            "tick": self.tick_count,
            "n_pairs": len(frame.pairs),
            "by_timeframe": {
                tf.name: len(frame.get_by_timeframe(tf))
                for tf in Timeframe
            },
            "n_stale": len(stale),
            "top_depth_pairs": [
                (p.symbol, d)
                for p, d in sorted(frame.depths.items(), key=lambda x: -x[1])[:5]
            ],
        }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Fractal Sampler - 128 Pair Routing")
    print("=" * 50)
    
    # Build graph
    from currency_graph import build_coinbase_graph_depth
    graph = build_coinbase_graph_depth()
    
    print(f"\nGraph: {len(graph.currencies)} currencies, {len(graph.pairs)} pairs")
    
    # Create sampler
    sampler = FractalSampler(graph, max_pairs=256, frame_size=128)
    
    print(f"\nSampler stats:")
    for k, v in sampler.stats().items():
        print(f"  {k}: {v}")
    
    # Sample a few frames
    print("\n--- Sampling Frames ---")
    processor = TickProcessor(sampler)
    
    for i in range(5):
        result = processor.tick()
        print(f"\nTick {result['tick']}:")
        print(f"  Pairs sampled: {result['n_pairs']}")
        print(f"  By timeframe: {result['by_timeframe']}")
        print(f"  Stale pairs: {result['n_stale']}")
        print(f"  Top depth: {result['top_depth_pairs'][:3]}")
    
    print("\n--- Stale Pairs ---")
    stale = sampler.get_stale_pairs(threshold=0.0)
    for pair, tf, staleness in stale[:5]:
        print(f"  {pair.symbol} ({tf.name}): staleness={staleness:.2f}")
    
    print("\n✓ 128 pairs per frame from multi-timescale bags")
    print("✓ Random sampling ensures coverage across all pairs")
    print("✓ Depth-based routing prioritizes well-connected currencies")
