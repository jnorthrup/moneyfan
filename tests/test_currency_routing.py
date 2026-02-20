import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hrm.currency_graph import CurrencyGraph, DepthBasedRouter, Currency

class TestCurrencyRouting(unittest.TestCase):
    def setUp(self):
        # Build a small test graph
        self.graph = CurrencyGraph()
        
        # Add basic currencies
        self.usd = self.graph.add_currency("USD")
        self.btc = self.graph.add_currency("BTC")
        self.eth = self.graph.add_currency("ETH")
        self.sol = self.graph.add_currency("SOL")
        self.deep = self.graph.add_currency("DEEP") # A highly connected currency
        self.shallow = self.graph.add_currency("SHALLOW") # A poorly connected currency
        
        # Add Pairs
        # USD <-> BTC (High Volume)
        self.graph.add_pair("BTC", "USD", volume_24h=1e9, spread=0.0001)
        
        # USD <-> ETH (High Volume)
        self.graph.add_pair("ETH", "USD", volume_24h=5e8, spread=0.0001)
        
        # DEEP is connected to everyone (simulating depth)
        self.graph.add_pair("DEEP", "USD", volume_24h=1e7, spread=0.0005)
        self.graph.add_pair("DEEP", "BTC", volume_24h=1e7, spread=0.0005)
        self.graph.add_pair("DEEP", "ETH", volume_24h=1e7, spread=0.0005)
        self.graph.add_pair("DEEP", "SOL", volume_24h=1e7, spread=0.0005)
        
        # SHALLOW is only connected to SOL
        self.graph.add_pair("SHALLOW", "SOL", volume_24h=1e5, spread=0.0005)
        
        # Build Metrics (must call this to populate depth)
        self.graph.get_depth_metrics()
        
        self.router = DepthBasedRouter(self.graph)

    def test_depth_metrics_calculation(self):
        """Test that depth metrics are calculated and DEEP > SHALLOW"""
        metrics = self.graph.get_depth_metrics()
        
        score_deep = metrics[self.deep].depth_score
        score_shallow = metrics[self.shallow].depth_score
        
        print(f"\nDepth Scores: DEEP={score_deep:.3f}, SHALLOW={score_shallow:.3f}")
        self.assertGreater(score_deep, score_shallow, "DEEP currency should have higher depth score")
        
        # BTC should also be reasonably deep due to connections
        score_btc = metrics[self.btc].depth_score
        self.assertGreater(score_btc, 0.0)

    def test_direct_route(self):
        """Test direct routing functionality"""
        route = self.router.find_route(self.usd, self.btc, prefer_depth=True)
        self.assertIsNotNone(route)
        self.assertEqual(len(route.path), 1)
        self.assertEqual(route.path[0].base.symbol, "BTC")
        self.assertEqual(route.path[0].quote.symbol, "USD")

    def test_multi_hop_depth_preference(self):
        """
        Test that router prefers a route through a DEEP node.
        
        Scenario: Route SOL -> USD
        Path A: SOL -> SHALLOW -> ... (Dead end or long)
        Path B: SOL -> DEEP -> USD
        """
        route = self.router.find_route(self.sol, self.usd, prefer_depth=True)
        
        self.assertIsNotNone(route)
        print(f"\nRoute SOL->USD: {route}")
        
        # Should go through DEEP because it connects both and has high depth
        # Path: SOL -> DEEP -> USD
        # Depending on implementation details, it might find this.
        
        currencies_in_path = [c.symbol for c in route.currencies]
        self.assertIn("DEEP", currencies_in_path, "Should route through DEEP currency")
        
    def test_suggest_intermediate(self):
        """Test intermediate currency suggestion"""
        # SOL and BTC are connected via DEEP
        inter = self.router.suggest_intermediate(self.sol, self.btc)
        self.assertEqual(inter, self.deep, "Should suggest DEEP as intermediate")

if __name__ == "__main__":
    unittest.main()
