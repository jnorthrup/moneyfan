"""
Stochastic Compass Module
=========================

Implements the exact mathematical compass equations for steering the stochastic bag:

1. Dirichlet Sampling:
   w ~ Dirichlet(α) where α_i ∝ (recent_Sharpe_i + ε)^k

2. Bag Resampling:
   bag_resample = multinomial(N=30, p = softmax(-β * C @ w))

3. GBM (Geometric Brownian Motion):
   dS = μ S dt + σ S dW

4. OU (Ornstein-Uhlenbeck):
   dX = θ(μ - X)dt + σ dW

These equations prevent over-concentration and auto-rotate to uncorrelated pairs.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class StochasticCompass:
    """
    Stochastic compass with Dirichlet, GBM, and OU equations
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        
    def dirichlet_weights(self, 
                         recent_sharpes: np.ndarray, 
                         epsilon: float = 0.1,
                         k: float = 2.0) -> np.ndarray:
        """
        Dirichlet sampling: w ~ Dirichlet(α) where α_i ∝ (recent_Sharpe_i + ε)^k
        
        Args:
            recent_sharpes: Array of Sharpe ratios for each codec [n_codecs]
            epsilon: Small constant to avoid zero (default: 0.1)
            k: Exponent for power transformation (default: 2.0)
            
        Returns:
            Dirichlet weights [n_codecs] that sum to 1
        """
        # Ensure positive values
        sharpes = recent_sharpes.copy().astype(float)
        sharpes = np.maximum(sharpes, 0.0)
        
        # Calculate alpha values: α_i ∝ (Sharpe_i + ε)^k
        alpha = (sharpes + epsilon) ** k
        
        # Normalize alpha (ensure it's not all zeros)
        alpha_sum = np.sum(alpha)
        if alpha_sum < 1e-10:
            # All sharpes are zero, use uniform weights
            alpha = np.ones_like(sharpes)
        else:
            # Normalize so that sum(alpha) = n_codecs (typical for Dirichlet)
            alpha = alpha / alpha_sum * len(sharpes)
        
        # Sample from Dirichlet distribution
        weights = self.rng.dirichlet(alpha)
        
        return weights
    
    def bag_resample(self, 
                    n_codecs: int,
                    n_selected: int = 30,
                    correlation_matrix: Optional[np.ndarray] = None,
                    beta: float = 1.5,
                    weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Bag resampling: bag_resample = multinomial(N=30, p = softmax(-β * C @ w))
        
        Args:
            n_codecs: Total number of codecs
            n_selected: Number to select (default: 30 for 30-pair bag)
            correlation_matrix: Correlation matrix [n_codecs, n_codecs] (optional)
            beta: Scaling factor for correlation (default: 1.5)
            weights: Dirichlet weights [n_codecs] (optional)
            
        Returns:
            Selected codec indices [n_selected]
        """
        if correlation_matrix is None:
            # Default: zero correlation (identity matrix)
            correlation_matrix = np.eye(n_codecs)
        
        if weights is None:
            # Uniform weights
            weights = np.ones(n_codecs) / n_codecs
        
        # Calculate scores: scores = -β * C @ w
        scores = -beta * correlation_matrix @ weights
        
        # Softmax: p = softmax(scores)
        p = self._softmax(scores)
        
        # Ensure probabilities are valid (sum to 1, no zeros)
        p = np.maximum(p, 1e-10)
        p = p / np.sum(p)
        
        # Multinomial selection without replacement
        selected = self.rng.choice(
            n_codecs, 
            size=min(n_selected, n_codecs),
            replace=False,
            p=p
        )
        
        return selected
    
    def gbm_price_path(self, 
                      S0: float, 
                      mu: float, 
                      sigma: float, 
                      T: float, 
                      steps: int) -> np.ndarray:
        """
        Geometric Brownian Motion: dS = μ S dt + σ S dW
        
        Args:
            S0: Initial price (e.g., 70000.0 for BTC)
            mu: Drift (expected return, e.g., 0.1 for 10% annual)
            sigma: Volatility (e.g., 0.5 for 50% annual)
            T: Time horizon in days (e.g., 30 for 30 days)
            steps: Number of time steps (e.g., 100)
            
        Returns:
            Price path [steps+1]
        """
        dt = T / steps
        price_path = np.zeros(steps + 1)
        price_path[0] = S0
        
        for i in range(1, steps + 1):
            # Wiener process increment
            dW = self.rng.normal(0, np.sqrt(dt))
            
            # GBM update
            price_path[i] = price_path[i-1] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * dW
            )
        
        return price_path
    
    def ou_mean_reversion(self, 
                         X0: float, 
                         mu: float, 
                         theta: float, 
                         sigma: float, 
                         T: float, 
                         steps: int) -> np.ndarray:
        """
        Ornstein-Uhlenbeck: dX = θ(μ - X)dt + σ dW
        
        Args:
            X0: Initial value (e.g., 0.0 for mean reversion)
            mu: Long-term mean (e.g., 0.0 for centered around zero)
            theta: Mean reversion speed (e.g., 0.5)
            sigma: Volatility (e.g., 0.1)
            T: Time horizon in days
            steps: Number of time steps
            
        Returns:
            OU process path [steps+1]
        """
        dt = T / steps
        path = np.zeros(steps + 1)
        path[0] = X0
        
        for i in range(1, steps + 1):
            # Wiener process increment
            dW = self.rng.normal(0, np.sqrt(dt))
            
            # OU update
            path[i] = path[i-1] + theta * (mu - path[i-1]) * dt + sigma * dW
        
        return path
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax function for probability calculation"""
        x = x - np.max(x)  # Numerical stability
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)
    
    def correlation_matrix(self, 
                          price_series: np.ndarray) -> np.ndarray:
        """
        Calculate correlation matrix from price series
        
        Args:
            price_series: Array of price series [n_series, n_timesteps]
            
        Returns:
            Correlation matrix [n_series, n_series]
        """
        # Handle empty input
        if price_series.shape[0] < 2 or price_series.shape[1] < 2:
            return np.eye(price_series.shape[0])
        
        # Convert prices to returns
        try:
            returns = np.diff(np.log(price_series), axis=1)
            
            # Handle cases where log fails (zero or negative prices)
            if np.any(np.isnan(returns)):
                # Fallback to simple returns
                returns = np.diff(price_series, axis=1) / price_series[:, :-1]
        except Exception:
            # Fallback to simple differences
            returns = np.diff(price_series, axis=1)
        
        # Calculate correlation matrix
        try:
            corr = np.corrcoef(returns)
            
            # Handle NaN values (zero variance)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Ensure symmetry
            corr = (corr + corr.T) / 2
            
            # Ensure diagonal is 1
            np.fill_diagonal(corr, 1.0)
            
            # Clip to valid range
            corr = np.clip(corr, -0.99, 0.99)
            
        except Exception as e:
            print(f"⚠️  Error calculating correlation matrix: {e}")
            # Return identity matrix as fallback
            corr = np.eye(price_series.shape[0])
            
        # Final validation: ensure diagonal is exactly 1.0
        np.fill_diagonal(corr, 1.0)
        
        return corr
    
    def generate_synthetic_prices(self, 
                                 n_assets: int, 
                                 n_steps: int,
                                 mu: float = 0.1,
                                 sigma: float = 0.2,
                                 correlation_strength: float = 0.3) -> np.ndarray:
        """
        Generate synthetic correlated price series for testing
        
        Args:
            n_assets: Number of assets
            n_steps: Number of time steps
            mu: Expected return
            sigma: Volatility
            correlation_strength: Base correlation between assets
            
        Returns:
            Price series [n_assets, n_steps]
        """
        # Create correlation matrix with base correlation
        base_corr = np.ones((n_assets, n_assets)) * correlation_strength
        np.fill_diagonal(base_corr, 1.0)
        
        # Generate correlated returns
        try:
            from scipy.stats import multivariate_normal
            mean = np.zeros(n_assets)
            cov = base_corr * sigma**2
            mvn = multivariate_normal(mean=mean, cov=cov)
            
            returns = mvn.rvs(n_steps)
            prices = np.zeros((n_assets, n_steps + 1))
            prices[:, 0] = 100.0  # Start price
            
            for i in range(1, n_steps + 1):
                prices[:, i] = prices[:, i-1] * (1 + returns[:, i-1] + mu/n_steps)
            
        except ImportError:
            # Fallback without scipy
            prices = np.zeros((n_assets, n_steps + 1))
            prices[:, 0] = 100.0
            
            for i in range(1, n_steps + 1):
                # Simple correlated returns
                base_return = self.rng.normal(mu/n_steps, sigma*np.sqrt(1/n_steps))
                for a in range(n_assets):
                    # Add correlation
                    correlated_return = base_return + self.rng.normal(0, sigma*np.sqrt(1-correlation_strength))
                    prices[a, i] = prices[a, i-1] * (1 + correlated_return)
        
        return prices
    
    def estimate_portfolio_weights(self, 
                                  price_series: np.ndarray,
                                  target_volatility: float = 0.15) -> np.ndarray:
        """
        Estimate optimal portfolio weights using risk parity approach
        
        Args:
            price_series: Price series [n_assets, n_timesteps]
            target_volatility: Target portfolio volatility
            
        Returns:
            Portfolio weights [n_assets]
        """
        # Calculate returns
        returns = np.diff(np.log(price_series), axis=1)
        
        # Calculate covariance matrix
        cov = np.cov(returns)
        
        # Risk parity weights (inverse volatility)
        volatilities = np.sqrt(np.diag(cov))
        inv_vol = 1.0 / (volatilities + 1e-10)
        
        # Normalize
        weights = inv_vol / np.sum(inv_vol)
        
        # Scale to target volatility
        portfolio_vol = np.sqrt(weights @ cov @ weights)
        if portfolio_vol > 0:
            scaling = target_volatility / portfolio_vol
            weights = weights * min(scaling, 1.0)  # Don't leverage up
        
        return weights


class StochasticCompassUtility:
    """
    Utility functions for the stochastic compass
    """
    
    @staticmethod
    def calculate_sharpe(returns: np.ndarray, 
                        annualization_factor: float = 252) -> float:
        """
        Calculate Sharpe ratio from returns
        
        Args:
            returns: Array of returns
            annualization_factor: Days per year (default: 252 for trading)
            
        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return < 1e-10:
            return 0.0
        
        sharpe = (mean_return / std_return) * np.sqrt(annualization_factor)
        return sharpe
    
    @staticmethod
    def calculate_max_drawdown(price_series: np.ndarray) -> float:
        """
        Calculate maximum drawdown from price series
        
        Args:
            price_series: Array of prices
            
        Returns:
            Maximum drawdown (negative value, e.g., -0.15 for 15% drawdown)
        """
        if len(price_series) < 2:
            return 0.0
        
        peak = price_series[0]
        max_dd = 0.0
        
        for price in price_series:
            if price > peak:
                peak = price
            
            dd = (price - peak) / peak
            if dd < max_dd:
                max_dd = dd
        
        return max_dd
    
    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate KL divergence between two distributions
        
        Args:
            p: First distribution
            q: Second distribution
            
        Returns:
            KL divergence
        """
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)
        
        return np.sum(p * np.log(p / q))