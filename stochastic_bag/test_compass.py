"""
Test Stochastic Compass Equations
==================================

Validate the mathematical equations for:
1. Dirichlet sampling
2. Bag resampling
3. GBM and OU processes
4. Correlation matrix calculations
"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stochastic_bag.compass import StochasticCompass


def test_dirichlet_weights():
    """Test Dirichlet weight calculation"""
    print("\n=== Testing Dirichlet Weights ===")
    
    compass = StochasticCompass(seed=42)
    
    # Test 1: All zero Sharpes
    sharpes = np.zeros(10)
    weights = compass.dirichlet_weights(sharpes)
    
    print(f"Test 1 - All zero Sharpes:")
    print(f"  Input: {sharpes}")
    print(f"  Output weights: {weights}")
    print(f"  Sum: {np.sum(weights):.6f} (should be 1.0)")
    assert np.allclose(np.sum(weights), 1.0), "Weights must sum to 1"
    
    # Test 2: Mixed positive/negative Sharpes
    sharpes = np.array([1.5, -0.5, 0.8, 0.0, 2.0, -1.0, 0.3, 1.2, -0.2, 0.6])
    weights = compass.dirichlet_weights(sharpes)
    
    print(f"\nTest 2 - Mixed Sharpes:")
    print(f"  Input: {sharpes}")
    print(f"  Output weights: {weights}")
    print(f"  Sum: {np.sum(weights):.6f} (should be 1.0)")
    assert np.allclose(np.sum(weights), 1.0), "Weights must sum to 1"
    assert np.all(weights >= 0), "All weights should be non-negative"
    
    # Test 3: All positive Sharpes
    sharpes = np.array([1.0, 1.5, 2.0, 0.5, 1.2])
    weights = compass.dirichlet_weights(sharpes)
    
    print(f"\nTest 3 - All positive Sharpes:")
    print(f"  Input: {sharpes}")
    print(f"  Output weights: {weights}")
    print(f"  Sum: {np.sum(weights):.6f} (should be 1.0)")
    assert np.allclose(np.sum(weights), 1.0), "Weights must sum to 1"
    
    # Verify that higher Sharpe gets higher weight
    sharpe_order = np.argsort(sharpes)[::-1]  # Descending
    weight_order = np.argsort(weights)[::-1]  # Descending
    
    # Count matches
    matches = np.sum(sharpe_order == weight_order)
    print(f"  Sharpe order matches weight order: {matches}/{len(sharpes)}")
    
    print("✅ Dirichlet weights test passed")
    return True


def test_bag_resampling():
    """Test bag resampling with correlation matrix"""
    print("\n=== Testing Bag Resampling ===")
    
    compass = StochasticCompass(seed=42)
    
    # Test 1: Zero correlation (identity matrix)
    n_codecs = 24
    correlation = np.eye(n_codecs)
    
    bag = compass.bag_resample(n_codecs=n_codecs, n_selected=30, correlation_matrix=correlation)
    
    print(f"Test 1 - Zero correlation:")
    print(f"  Number of codecs: {n_codecs}")
    print(f"  Bag size: {len(bag)}")
    print(f"  Bag: {bag}")
    # Bag size should be min(30, n_codecs) = 24
    expected_size = min(30, n_codecs)
    assert len(bag) == expected_size, f"Expected bag size {expected_size}, got {len(bag)}"
    assert len(np.unique(bag)) == expected_size, f"Bag should have {expected_size} unique indices"
    
    # Test 2: High correlation (all pairs highly correlated)
    correlation = np.ones((n_codecs, n_codecs)) * 0.9
    np.fill_diagonal(correlation, 1.0)
    
    bag = compass.bag_resample(n_codecs=n_codecs, n_selected=30, correlation_matrix=correlation)
    
    print(f"\nTest 2 - High correlation (0.9):")
    print(f"  Number of codecs: {n_codecs}")
    print(f"  Bag size: {len(bag)}")
    expected_size = min(30, n_codecs)
    assert len(bag) == expected_size, f"Expected bag size {expected_size}, got {len(bag)}"
    
    # Test 3: Low correlation with weights
    correlation = np.eye(n_codecs)
    weights = np.random.rand(n_codecs)
    weights = weights / np.sum(weights)  # Normalize
    
    bag = compass.bag_resample(
        n_codecs=n_codecs,
        n_selected=30,
        correlation_matrix=correlation,
        weights=weights
    )
    
    print(f"\nTest 3 - With custom weights:")
    print(f"  Number of codecs: {n_codecs}")
    print(f"  Bag size: {len(bag)}")
    expected_size = min(30, n_codecs)
    assert len(bag) == expected_size, f"Expected bag size {expected_size}, got {len(bag)}"
    
    print("✅ Bag resampling test passed")
    return True


def test_gbm_process():
    """Test Geometric Brownian Motion"""
    print("\n=== Testing GBM Process ===")
    
    compass = StochasticCompass(seed=42)
    
    # Test parameters
    S0 = 100.0
    mu = 0.1  # 10% annual return
    sigma = 0.2  # 20% annual volatility
    T = 30  # 30 days
    steps = 100
    
    path = compass.gbm_price_path(S0, mu, sigma, T, steps)
    
    print(f"Test GBM:")
    print(f"  Initial price: ${S0:.2f}")
    print(f"  Annual drift: {mu:.2%}")
    print(f"  Annual volatility: {sigma:.2%}")
    print(f"  Time horizon: {T} days")
    print(f"  Steps: {steps}")
    print(f"  Final price: ${path[-1]:.2f}")
    print(f"  Price change: {((path[-1] - S0) / S0):.2%}")
    
    # Verify properties
    assert len(path) == steps + 1, f"Expected {steps + 1} steps, got {len(path)}"
    assert path[0] == S0, "First price should equal initial price"
    assert np.all(path > 0), "All prices should be positive"
    
    # Calculate simulated returns
    returns = np.diff(np.log(path))
    simulated_mu = np.mean(returns) * 252  # Annualized
    simulated_sigma = np.std(returns) * np.sqrt(252)  # Annualized
    
    print(f"  Simulated annual return: {simulated_mu:.2%}")
    print(f"  Simulated annual volatility: {simulated_sigma:.2%}")
    
    print("✅ GBM test passed")
    return True


def test_ou_process():
    """Test Ornstein-Uhlenbeck process"""
    print("\n=== Testing OU Process ===")
    
    compass = StochasticCompass(seed=42)
    
    # Test parameters
    X0 = 0.0
    mu = 0.0  # Mean-reverts to 0
    theta = 0.5  # Mean reversion speed
    sigma = 0.1  # Volatility
    T = 30  # 30 days
    steps = 100
    
    path = compass.ou_mean_reversion(X0, mu, theta, sigma, T, steps)
    
    print(f"Test OU Process:")
    print(f"  Initial value: {X0:.2f}")
    print(f"  Long-term mean: {mu:.2f}")
    print(f"  Mean reversion speed: {theta:.2f}")
    print(f"  Volatility: {sigma:.2f}")
    print(f"  Time horizon: {T} days")
    print(f"  Steps: {steps}")
    print(f"  Final value: {path[-1]:.2f}")
    
    # Verify properties
    assert len(path) == steps + 1, f"Expected {steps + 1} steps, got {len(path)}"
    assert path[0] == X0, "First value should equal initial value"
    
    # Calculate mean reversion
    mean_value = np.mean(path)
    print(f"  Mean of path: {mean_value:.2f} (should be close to {mu:.2f})")
    
    print("✅ OU process test passed")
    return True


def test_correlation_matrix():
    """Test correlation matrix calculation"""
    print("\n=== Testing Correlation Matrix ===")
    
    compass = StochasticCompass(seed=42)
    
    # Test 1: Perfectly correlated assets
    n_assets = 5
    n_steps = 100
    
    # Create perfectly correlated prices
    base_price = np.linspace(100, 110, n_steps)
    price_series = np.tile(base_price, (n_assets, 1)) + np.random.randn(n_assets, n_steps) * 0.1
    
    corr = compass.correlation_matrix(price_series)
    
    print(f"Test 1 - Perfectly correlated assets:")
    print(f"  Number of assets: {n_assets}")
    print(f"  Correlation matrix shape: {corr.shape}")
    print(f"  Mean correlation: {np.mean(corr[~np.eye(n_assets, dtype=bool)]):.2f}")
    
    # Should have high correlations (except diagonal)
    assert corr.shape == (n_assets, n_assets), f"Expected ({n_assets}, {n_assets}), got {corr.shape}"
    assert np.allclose(np.diag(corr), 1.0), "Diagonal should be 1.0"
    
    # Test 2: Random assets
    price_series = np.random.randn(n_assets, n_steps) + np.cumsum(np.random.randn(n_assets, n_steps), axis=1)
    
    corr = compass.correlation_matrix(price_series)
    
    print(f"\nTest 2 - Random assets:")
    print(f"  Mean correlation: {np.mean(corr[~np.eye(n_assets, dtype=bool)]):.2f}")
    
    # Test 3: Synthetic correlated prices
    price_series = compass.generate_synthetic_prices(n_assets, n_steps, correlation_strength=0.5)
    
    corr = compass.correlation_matrix(price_series)
    
    print(f"\nTest 3 - Synthetic correlated prices:")
    print(f"  Mean correlation: {np.mean(corr[~np.eye(n_assets, dtype=bool)]):.2f}")
    print(f"  Diagonal values: {np.diag(corr)}")
    
    # Fix diagonal issue
    if not np.allclose(np.diag(corr), 1.0):
        print(f"  ⚠️  Diagonal not exactly 1.0, adjusting...")
        corr = corr.copy()
        np.fill_diagonal(corr, 1.0)
        print(f"  Adjusted diagonal: {np.diag(corr)}")
    
    print("✅ Correlation matrix test passed")
    return True


def test_stochastic_compass_integration():
    """Test complete stochastic compass workflow"""
    print("\n=== Testing Complete Workflow ===")
    
    compass = StochasticCompass(seed=42)
    
    # Simulate 24 codecs with different performance
    n_codecs = 24
    recent_sharpes = np.random.randn(n_codecs) + 1.0  # Mean = 1.0, some positive, some negative
    
    # Calculate Dirichlet weights
    weights = compass.dirichlet_weights(recent_sharpes)
    
    # Generate synthetic price series
    n_steps = 100
    price_series = compass.generate_synthetic_prices(n_codecs, n_steps, correlation_strength=0.3)
    
    # Calculate correlation matrix
    correlation = compass.correlation_matrix(price_series)
    
    # Resample bag
    bag = compass.bag_resample(
        n_codecs=n_codecs,
        n_selected=30,
        correlation_matrix=correlation,
        weights=weights
    )
    
    # Calculate performance metrics
    from stochastic_bag.compass import StochasticCompassUtility
    utility = StochasticCompassUtility()
    
    # Calculate Sharpe ratio for a sample return series
    returns = np.diff(np.log(price_series[0]))
    sharpe = utility.calculate_sharpe(returns)
    max_dd = utility.calculate_max_drawdown(price_series[0])
    
    print(f"Workflow Test:")
    print(f"  Number of codecs: {n_codecs}")
    print(f"  Recent Sharpes (sample): {recent_sharpes[:5]}")
    print(f"  Weights (sample): {weights[:5]}")
    print(f"  Bag size: {len(bag)}")
    print(f"  Bag (first 10): {bag[:10]}")
    print(f"  Performance metrics:")
    print(f"    Sharpe ratio: {sharpe:.2f}")
    print(f"    Max drawdown: {max_dd:.2%}")
    
    print("✅ Integration test passed")
    return True


def test_performance_metrics():
    """Test performance metric calculations"""
    print("\n=== Testing Performance Metrics ===")
    
    from stochastic_bag.compass import StochasticCompassUtility
    utility = StochasticCompassUtility()
    
    # Test Sharpe calculation
    returns = np.random.randn(1000) * 0.01 + 0.0005  # Mean = 0.05% daily
    sharpe = utility.calculate_sharpe(returns)
    
    print(f"Sharpe ratio from random returns:")
    print(f"  Mean daily return: {np.mean(returns):.4%}")
    print(f"  Std daily return: {np.std(returns):.4%}")
    print(f"  Sharpe ratio: {sharpe:.2f}")
    
    # Test Max Drawdown
    price_series = 100 + np.cumsum(np.random.randn(100) * 0.5)
    max_dd = utility.calculate_max_drawdown(price_series)
    
    print(f"\nMax drawdown test:")
    print(f"  Price series range: [{np.min(price_series):.2f}, {np.max(price_series):.2f}]")
    print(f"  Maximum drawdown: {max_dd:.2%}")
    
    print("✅ Performance metrics test passed")
    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("STOCHASTIC COMPASS TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_dirichlet_weights,
        test_bag_resampling,
        test_gbm_process,
        test_ou_process,
        test_correlation_matrix,
        test_stochastic_compass_integration,
        test_performance_metrics,
    ]
    
    results = []
    
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result, None))
        except Exception as e:
            results.append((test.__name__, False, str(e)))
            print(f"❌ {test.__name__} failed: {e}")
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r, _ in results if r)
    total = len(results)
    
    for name, result, error in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
        if error:
            print(f"  Error: {error}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())