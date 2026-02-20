"""
Currency Graph - Depth-Based Routing

Route through currencies with DEPTH (many connections),
not just WIDTH (high volume on favorite pairs).

Depth = number of alternative paths
Width = volume on a single pair

Coinbase selection based on route depth, not BTC favoritism.
"""

from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict, deque
import heapq


# =============================================================================
# CURRENCY NODE
# =============================================================================

@dataclass(frozen=True)
class Currency:
    """A currency node"""
    symbol: str
    
    def __repr__(self):
        return self.symbol
    
    def __hash__(self):
        return hash(self.symbol)

    def __lt__(self, other):
        return self.symbol < other.symbol


# =============================================================================
# PAIR EDGE
# =============================================================================

@dataclass
class Pair:
    """An edge between two currencies"""
    base: Currency
    quote: Currency
    exchange: str = "coinbase"
    
    volume_24h: float = 0.0
    spread: float = 0.0
    
    @property
    def symbol(self) -> str:
        return f"{self.base.symbol}-{self.quote.symbol}"
    
    def inverse(self) -> 'Pair':
        return Pair(base=self.quote, quote=self.base, exchange=self.exchange,
                   volume_24h=self.volume_24h, spread=self.spread)
    
    def __hash__(self):
        return hash((self.base, self.quote))


# =============================================================================
# DEPTH METRICS
# =============================================================================

@dataclass
class DepthMetrics:
    """Depth metrics for a currency"""
    currency: Currency
    
    # Direct connections
    out_degree: int = 0       # Pairs where this is base
    in_degree: int = 0        # Pairs where this is quote
    total_degree: int = 0     # Total direct connections
    
    # Path depth (how many routes pass through)
    routes_through: int = 0   # Number of routes using this as intermediate
    betweenness: float = 0.0  # Betweenness centrality
    
    # Reachability
    reachable_in_1_hop: int = 0
    reachable_in_2_hops: int = 0
    reachable_in_3_hops: int = 0
    
    # Combined depth score
    depth_score: float = 0.0
    
    def __repr__(self):
        return f"DepthMetrics({self.currency.symbol}, degree={self.total_degree}, depth={self.depth_score:.2f})"


def compute_depth_metrics(graph: 'CurrencyGraph') -> Dict[Currency, DepthMetrics]:
    """
    Compute depth metrics for all currencies.
    
    Depth prioritizes currencies with:
    - Many direct connections (high degree)
    - Many routes passing through (high betweenness)
    - Good reachability (can reach many others)
    """
    metrics = {}
    
    for cur in graph.currencies:
        m = DepthMetrics(currency=cur)
        
        # Direct connections
        m.out_degree = len(graph.get_pairs_from(cur))
        m.in_degree = len([p for p in graph.pairs if p.quote == cur])
        m.total_degree = m.out_degree + m.in_degree
        
        # Reachability
        m.reachable_in_1_hop = m.total_degree
        m.reachable_in_2_hops = len(graph.reachable_in(cur, max_hops=2))
        m.reachable_in_3_hops = len(graph.reachable_in(cur, max_hops=3))
        
        metrics[cur] = m
    
    # Compute betweenness centrality (routes through)
    for source in graph.currencies:
        for target in graph.currencies:
            if source == target:
                continue
            
            # Find shortest path
            path = find_shortest_path(graph, source, target, max_hops=3)
            if path and len(path) > 1:
                # Intermediate currencies on this route
                for cur in path[1:-1]:
                    metrics[cur].routes_through += 1
    
    # Normalize betweenness
    max_routes = max(m.routes_through for m in metrics.values()) if metrics else 1
    for m in metrics.values():
        m.betweenness = m.routes_through / max_routes if max_routes > 0 else 0
    
    # Compute combined depth score
    for m in metrics.values():
        m.depth_score = (
            0.3 * (m.total_degree / max(1, max(m2.total_degree for m2 in metrics.values()))) +
            0.3 * m.betweenness +
            0.2 * (m.reachable_in_2_hops / max(1, max(m2.reachable_in_2_hops for m2 in metrics.values()))) +
            0.2 * (m.reachable_in_3_hops / max(1, max(m2.reachable_in_3_hops for m2 in metrics.values())))
        )
    
    return metrics


# =============================================================================
# CURRENCY GRAPH
# =============================================================================

class CurrencyGraph:
    """
    Graph of currencies and trading pairs.
    
    Routes by DEPTH, not WIDTH.
    """
    
    def __init__(self):
        self.currencies: Set[Currency] = set()
        self.pairs: Set[Pair] = set()
        
        self._outgoing: Dict[Currency, Set[Pair]] = defaultdict(set)
        self._incoming: Dict[Currency, Set[Pair]] = defaultdict(set)
        self._pair_map: Dict[Tuple[Currency, Currency], Pair] = {}
        
        self._depth_metrics: Dict[Currency, DepthMetrics] = {}
        self._metrics_valid = False
    
    def add_currency(self, symbol: str) -> Currency:
        c = Currency(symbol)
        self.currencies.add(c)
        self._metrics_valid = False
        return c
    
    def add_pair(self, base: str, quote: str, volume_24h: float = 0.0, spread: float = 0.0) -> Pair:
        base_c = Currency(base)
        quote_c = Currency(quote)
        
        self.currencies.add(base_c)
        self.currencies.add(quote_c)
        
        pair = Pair(base=base_c, quote=quote_c, volume_24h=volume_24h, spread=spread)
        
        self.pairs.add(pair)
        self._outgoing[base_c].add(pair)
        self._incoming[quote_c].add(pair)
        self._pair_map[(base_c, quote_c)] = pair
        
        self._metrics_valid = False
        return pair
    
    def get_pair(self, base: Currency, quote: Currency) -> Optional[Pair]:
        return self._pair_map.get((base, quote))
    
    def get_pairs_from(self, currency: Currency) -> Set[Pair]:
        return self._outgoing.get(currency, set())
    
    def get_pairs_to(self, currency: Currency) -> Set[Pair]:
        return self._incoming.get(currency, set())
    
    def neighbors(self, currency: Currency) -> Set[Currency]:
        outgoing = {p.quote for p in self._outgoing.get(currency, set())}
        incoming = {p.base for p in self._incoming.get(currency, set())}
        return outgoing | incoming
    
    def reachable_in(self, start: Currency, max_hops: int = 3) -> Set[Currency]:
        """Find all currencies reachable within N hops"""
        visited = {start}
        frontier = {start}
        
        for _ in range(max_hops):
            new_frontier = set()
            for cur in frontier:
                new_frontier.update(self.neighbors(cur))
            new_frontier -= visited
            visited.update(new_frontier)
            frontier = new_frontier
            if not frontier:
                break
        
        visited.discard(start)
        return visited
    
    def get_depth_metrics(self) -> Dict[Currency, DepthMetrics]:
        """Get depth metrics (computed lazily)"""
        if not self._metrics_valid:
            self._depth_metrics = compute_depth_metrics(self)
            self._metrics_valid = True
        return self._depth_metrics
    
    def get_depth_score(self, currency: Currency) -> float:
        """Get depth score for a currency"""
        metrics = self.get_depth_metrics()
        return metrics.get(currency, DepthMetrics(currency)).depth_score
    
    def rank_by_depth(self) -> List[Tuple[Currency, float]]:
        """Rank currencies by depth score"""
        metrics = self.get_depth_metrics()
        return sorted(
            [(cur, m.depth_score) for cur, m in metrics.items()],
            key=lambda x: -x[1]
        )
    
    def get_top_30_pairs_by_depth(self) -> List[str]:
        """
        Get top 30 currency symbols by depth score (avoiding BTC favoritism)
        This selects for structural liquidity and arbitrage surface
        as per GOALS.md: "30 pairs chosen by currency_graph.py route depth"
        """
        rankings = self.rank_by_depth()
        
        # Get top 30 currencies, but deprioritize BTC
        top_30 = []
        for cur, depth_score in rankings:
            if cur.symbol != "BTC":  # Avoid BTC favoritism
                top_30.append(cur.symbol)
            if len(top_30) >= 30:
                break
        
        # If we don't have 30 after avoiding BTC, add BTC back
        if len(top_30) < 30:
            for cur, depth_score in rankings:
                if cur.symbol == "BTC" and cur.symbol not in top_30:
                    top_30.append(cur.symbol)
                if len(top_30) >= 30:
                    break
        
        return top_30[:30]  # Ensure exactly 30


# =============================================================================
# DEPTH-BASED ROUTER
# =============================================================================

def find_shortest_path(graph: CurrencyGraph, 
                        source: Currency, 
                        target: Currency,
                        max_hops: int = 4) -> Optional[List[Currency]]:
    """BFS shortest path"""
    if source == target:
        return [source]
    
    queue = deque([(source, [source])])
    visited = {source}
    
    while queue:
        current, path = queue.popleft()
        
        if len(path) > max_hops:
            continue
        
        for neighbor in graph.neighbors(current):
            if neighbor == target:
                return path + [neighbor]
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None


@dataclass
class Route:
    """A route with depth-based scoring"""
    path: List[Pair]
    total_spread: float
    min_depth_score: float  # Minimum depth score along route
    avg_depth_score: float
    depth_weighted_cost: float
    
    @property
    def currencies(self) -> List[Currency]:
        if not self.path:
            return []
        curs = [self.path[0].base]
        for p in self.path:
            curs.append(p.quote)
        return curs
    
    def __repr__(self):
        path_str = " → ".join(c.symbol for c in self.currencies)
        return f"Route({path_str}, depth={self.avg_depth_score:.2f})"


class DepthBasedRouter:
    """
    Routes based on DEPTH, not WIDTH.
    
    Prioritizes:
    - Currencies with many connections (high degree)
    - Currencies on many routes (high betweenness)
    - Paths with good depth at each hop
    
    NOT:
    - Just routing through BTC because it has high volume
    """
    
    def __init__(self, graph: CurrencyGraph):
        self.graph = graph
        self.metrics = graph.get_depth_metrics()
    
    def find_route(self, 
                   source: Currency, 
                   target: Currency,
                   max_hops: int = 3,
                   prefer_depth: bool = True) -> Optional[Route]:
        """
        Find route prioritizing depth.
        
        When prefer_depth=True:
          Routes through well-connected currencies
          Even if spread is slightly higher
        
        When prefer_depth=False:
          Routes by spread (width-based, traditional)
        """
        if source == target:
            return Route(path=[], total_spread=0.0, min_depth_score=1.0, 
                        avg_depth_score=1.0, depth_weighted_cost=0.0)
        
        # Dijkstra with depth-weighted cost
        # Priority: (cost, currency, path)
        pq = [(0.0, source, [])]
        visited = {}
        
        while pq:
            cost, current, path = heapq.heappop(pq)
            
            if current in visited:
                continue
            visited[current] = cost
            
            if current == target and path:
                # Build route
                route_pairs = []
                # path contains nodes visited before current.
                # Use full path including target for iteration
                full_path_nodes = path + [target]
                
                for i in range(len(full_path_nodes) - 1):
                    u, v = full_path_nodes[i], full_path_nodes[i+1]
                    p1 = self.graph.get_pair(u, v)
                    p2 = self.graph.get_pair(v, u)
                    pair = p1 or p2
                    
                    if pair:
                        route_pairs.append(pair)
                
                if len(route_pairs) == len(full_path_nodes) - 1:
                    total_spread = sum(p.spread for p in route_pairs)
                    depth_scores = [self.metrics.get(c, DepthMetrics(c)).depth_score 
                                   for c in path + [target]]
                    
                    return Route(
                        path=route_pairs,
                        total_spread=total_spread,
                        min_depth_score=min(depth_scores),
                        avg_depth_score=sum(depth_scores) / len(depth_scores),
                        depth_weighted_cost=total_spread / (sum(depth_scores) / len(depth_scores) + 0.1),
                    )
            
            if len(path) >= max_hops:
                continue
            
            # Explore neighbors (OUTGOING: current -> next)
            for pair in self.graph.get_pairs_from(current):
                next_cur = pair.quote
                self._process_neighbor(current, next_cur, pair, cost, path, visited, pq, prefer_depth)

            # Explore neighbors (INCOMING: next -> current, i.e. traversing pair in reverse)
            for pair in self.graph.get_pairs_to(current):
                next_cur = pair.base
                self._process_neighbor(current, next_cur, pair, cost, path, visited, pq, prefer_depth)
        
        return None

    def _process_neighbor(self, current, next_cur, pair, cost, path, visited, pq, prefer_depth):
        if next_cur in visited:
            return
        
        # Cost: spread penalized by LOW depth (want high depth)
        next_depth = self.metrics.get(next_cur, DepthMetrics(next_cur)).depth_score
        
        if prefer_depth:
            # Lower cost for higher depth
            edge_cost = pair.spread / (next_depth + 0.1)
        else:
            edge_cost = pair.spread
        
        new_cost = cost + edge_cost
        heapq.heappush(pq, (new_cost, next_cur, path + [current]))
    
    def find_all_routes(self,
                        source: Currency,
                        target: Currency,
                        max_hops: int = 3,
                        limit: int = 5) -> List[Route]:
        """Find top routes by depth-weighted cost"""
        routes = []
        queue = deque([(source, [], 0.0)])
        visited_paths = set()
        
        while queue and len(routes) < limit * 2:
            current, path_currencies, total_spread = queue.popleft()
            
            path_key = tuple(c.symbol for c in path_currencies)
            if path_key in visited_paths:
                continue
            visited_paths.add(path_key)
            
            if current == target and path_currencies:
                # Build route
                route_pairs = []
                for i in range(len(path_currencies) - 1):
                    # Try both directions
                    p1 = self.graph.get_pair(path_currencies[i], path_currencies[i+1])
                    p2 = self.graph.get_pair(path_currencies[i+1], path_currencies[i])
                    pair = p1 or p2
                    
                    if pair:
                        route_pairs.append(pair)
                
                if route_pairs:
                    depth_scores = [self.metrics.get(c, DepthMetrics(c)).depth_score 
                                   for c in path_currencies]
                    routes.append(Route(
                        path=route_pairs,
                        total_spread=total_spread,
                        min_depth_score=min(depth_scores),
                        avg_depth_score=sum(depth_scores) / len(depth_scores),
                        depth_weighted_cost=total_spread / (sum(depth_scores) / len(depth_scores) + 0.1),
                    ))
                continue
            
            if len(path_currencies) >= max_hops:
                continue
            
            # Outgoing
            for pair in self.graph.get_pairs_from(current):
                next_cur = pair.quote
                if next_cur not in path_currencies:
                    queue.append((next_cur, path_currencies + [current, next_cur], total_spread + pair.spread))

            # Incoming (Reverse)
            for pair in self.graph.get_pairs_to(current):
                next_cur = pair.base
                if next_cur not in path_currencies:
                    queue.append((next_cur, path_currencies + [current, next_cur], total_spread + pair.spread))
        
        # Sort by depth-weighted cost
        return sorted(routes, key=lambda r: r.depth_weighted_cost)[:limit]
    
    def suggest_intermediate(self, source: Currency, target: Currency) -> Optional[Currency]:
        """
        Suggest best intermediate currency based on depth.
        
        Returns currency with highest depth score that connects both.
        """
        source_neighbors = self.graph.neighbors(source)
        target_neighbors = self.graph.neighbors(target)
        
        common = source_neighbors & target_neighbors
        
        if not common:
            return None
        
        # Return one with highest depth score
        return max(common, key=lambda c: self.metrics.get(c, DepthMetrics(c)).depth_score)


# =============================================================================
# BUILD COINBASE GRAPH (DEPTH-FOCUSED)
# =============================================================================

def build_coinbase_graph_depth() -> CurrencyGraph:
    """
    Build Coinbase graph with realistic pair structure.
    
    Note: Depth comes from cross-pairs, not just USD pairs.
    This graph is expanded to match the 30 pairs from Binance.
    """
    graph = CurrencyGraph()
    
    # Base currency
    graph.add_currency("USD")
    
    # All pairs (base, quote, volume, spread)
    # Prioritize cross-pairs for depth
    # Match the 30 pairs from Binance: BTC, ETH, BNB, XRP, ADA, DOGE, SOL, DOT, MATIC, LTC, 
    # AVAX, LINK, ATOM, UNI, ETC, XLM, ALGO, VET, FIL, ICP, THETA, AAVE, NEAR, AXS, FTM, SAND, MANA, GALA, ENJ, COMP
    pairs = [
        # Major USD pairs (10 pairs)
        ("BTC", "USD", 1e10, 0.0001),
        ("ETH", "USD", 5e9, 0.0001),
        ("BNB", "USD", 2e9, 0.0002),
        ("SOL", "USD", 1e9, 0.0002),
        ("XRP", "USD", 5e8, 0.0002),
        ("ADA", "USD", 2e8, 0.0003),
        ("DOGE", "USD", 3e8, 0.0003),
        ("DOT", "USD", 1e8, 0.0003),
        ("MATIC", "USD", 8e7, 0.0004),
        ("LTC", "USD", 5e7, 0.0004),
        
        # Cross pairs (CREATE DEPTH!) - 20 pairs
        ("ETH", "BTC", 5e8, 0.0002),   # ETH-BTC creates depth
        ("BNB", "ETH", 2e8, 0.0003),   # BNB-ETH
        ("SOL", "ETH", 2e8, 0.0003),   # SOL-ETH creates depth
        ("SOL", "BTC", 1e8, 0.0003),
        ("XRP", "BTC", 5e7, 0.0004),
        ("XRP", "ETH", 3e7, 0.0005),
        ("ADA", "BTC", 5e7, 0.0004),
        ("ADA", "ETH", 3e7, 0.0005),
        ("DOGE", "BTC", 1e8, 0.0004),
        ("DOGE", "ETH", 5e7, 0.0005),
        ("DOT", "ETH", 2e8, 0.0003),
        ("DOT", "BTC", 3e7, 0.0005),
        ("MATIC", "ETH", 1e8, 0.0004),
        ("LTC", "ETH", 3e7, 0.0005),
        ("LTC", "BTC", 2e7, 0.0006),
        
        # Additional pairs to reach 30 (6 pairs)
        ("AVAX", "USD", 1e8, 0.0003),
        ("AVAX", "ETH", 3e8, 0.0002),
        ("LINK", "USD", 1e8, 0.0003),
        ("LINK", "ETH", 2e8, 0.0003),
        ("ATOM", "USD", 5e7, 0.0004),
        ("ATOM", "ETH", 8e7, 0.0004),
        
        # Remaining pairs from Binance list (14 pairs)
        ("UNI", "USD", 5e7, 0.0004),
        ("UNI", "ETH", 1e8, 0.0003),
        ("ETC", "USD", 3e7, 0.0005),
        ("XLM", "USD", 3e7, 0.0005),
        ("ALGO", "USD", 2e7, 0.0006),
        ("VET", "USD", 2e7, 0.0006),
        ("FIL", "USD", 3e7, 0.0005),
        ("ICP", "USD", 3e7, 0.0005),
        ("THETA", "USD", 2e7, 0.0006),
        ("AAVE", "USD", 2e7, 0.0006),
        ("NEAR", "USD", 2e7, 0.0006),
        ("AXS", "USD", 2e7, 0.0006),
        ("FTM", "USD", 2e7, 0.0006),
        ("SAND", "USD", 2e7, 0.0006),
        # Add remaining 3 pairs from Binance list: MANA, GALA, ENJ, COMP
        ("MANA", "USD", 2e7, 0.0006),
        ("GALA", "USD", 2e7, 0.0006),
        ("ENJ", "USD", 2e7, 0.0006),
        ("COMP", "USD", 2e7, 0.0006),
    ]
    
    for base, quote, vol, spread in pairs:
        graph.add_pair(base, quote, vol, spread)
    
    return graph


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Currency Graph - Depth-Based Routing")
    print("=" * 50)
    
    # Build graph
    graph = build_coinbase_graph_depth()
    print(f"\n{graph}")
    
    # Compute depth metrics
    metrics = graph.get_depth_metrics()
    
    print("\n--- Depth Rankings ---")
    rankings = graph.rank_by_depth()
    for cur, score in rankings[:10]:
        m = metrics[cur]
        print(f"  {cur.symbol}: depth={score:.3f}, degree={m.total_degree}, betweenness={m.betweenness:.3f}")
    
    print("\n--- Depth vs Width Comparison ---")
    
    # BTC has highest volume (width) but may not have highest depth
    btc = Currency("BTC")
    eth = Currency("ETH")
    usd = Currency("USD")
    sol = Currency("SOL")
    
    print(f"BTC depth score: {graph.get_depth_score(btc):.3f}")
    print(f"ETH depth score: {graph.get_depth_score(eth):.3f}")
    print(f"USD depth score: {graph.get_depth_score(usd):.3f}")
    
    print("\n--- Routing Examples ---")
    
    router = DepthBasedRouter(graph)
    
    # Route SOL → USD
    print("\nSOL → USD:")
    print("  Depth-based:")
    route = router.find_route(sol, usd, prefer_depth=True)
    if route:
        print(f"    {route}")
        print(f"    Currencies: {[c.symbol for c in route.currencies]}")
    
    print("  Width-based:")
    route = router.find_route(sol, usd, prefer_depth=False)
    if route:
        print(f"    {route}")
    
    # All routes
    print("\n  All routes SOL → USD:")
    for r in router.find_all_routes(sol, usd):
        print(f"    {r}")
    
    # Suggest intermediate
    print("\n--- Intermediate Suggestions ---")
    for source, target in [(sol, usd), (Currency("DOGE"), Currency("AVAX"))]:
        intermediate = router.suggest_intermediate(source, target)
        if intermediate:
            print(f"  {source} → {target}: via {intermediate} (depth={graph.get_depth_score(intermediate):.3f})")
    
    print("\n✓ Routes favor DEPTH (many connections), not WIDTH (high volume)")
    print("✓ ETH may be preferred over BTC if ETH has more cross-pairs")
