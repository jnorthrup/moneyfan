"""
Correlation Matrix Module
=========================

Handles correlation matrix calculations and portfolio optimization
for the stochastic bag system.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class CorrelationMatrix:
    """
    Correlation matrix calculator and analyzer
    """
    
    def __init__(self, min_periods: int = 10):
        """
        Initialize correlation matrix calculator
        
        Args:
            min_periods: Minimum number of periods for correlation calculation
        """
        self.min_periods = min_periods
        
    def calculate(self, 
                  returns: np.ndarray, 
                  method: str = 'pearson') -> np.ndarray:
        """
        Calculate correlation matrix from returns
        
        Args:
            returns: Array of returns [n_assets, n_timesteps]
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Correlation matrix [n_assets, n_assets]
        """
        # Validate input
        if returns.shape[0] < 2 or returns.shape[1] < self.min_periods:
            n_assets = returns.shape[0]
            return np.eye(n_assets)
        
        try:
            if method == 'pearson':
                corr = self._pearson_correlation(returns)
            elif method == 'spearman':
                corr = self._spearman_correlation(returns)
            elif method == 'kendall':
                corr = self._kendall_correlation(returns)
            else:
                corr = self._pearson_correlation(returns)
            
            # Post-process correlation matrix
            corr = self._post_process(corr)
            
            return corr
            
        except Exception as e:
            print(f"⚠️  Error calculating correlation matrix: {e}")
            n_assets = returns.shape[0]
            return np.eye(n_assets)
    
    def _pearson_correlation(self, returns: np.ndarray) -> np.ndarray:
        """Pearson correlation (linear correlation)"""
        n_assets = returns.shape[0]
        
        # Handle edge cases
        if n_assets == 0:
            return np.array([])
        
        if n_assets == 1:
            return np.array([[1.0]])
        
        # Calculate Pearson correlation
        corr = np.corrcoef(returns)
        
        return corr
    
    def _spearman_correlation(self, returns: np.ndarray) -> np.ndarray:
        """Spearman correlation (rank correlation)"""
        try:
            from scipy.stats import spearmanr
            n_assets = returns.shape[0]
            
            if n_assets == 0:
                return np.array([])
            
            if n_assets == 1:
                return np.array([[1.0]])
            
            # Calculate Spearman correlation for each pair
            corr = np.eye(n_assets)
            
            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    try:
                        rho, _ = spearmanr(returns[i], returns[j])
                        corr[i, j] = rho
                        corr[j, i] = rho
                    except Exception:
                        # Fallback to Pearson
                        rho = np.corrcoef(returns[i], returns[j])[0, 1]
                        corr[i, j] = rho
                        corr[j, i] = rho
            
            return corr
            
        except ImportError:
            # Fallback to Pearson if scipy not available
            return self._pearson_correlation(returns)
    
    def _kendall_correlation(self, returns: np.ndarray) -> np.ndarray:
        """Kendall tau correlation"""
        try:
            from scipy.stats import kendalltau
            n_assets = returns.shape[0]
            
            if n_assets == 0:
                return np.array([])
            
            if n_assets == 1:
                return np.array([[1.0]])
            
            # Calculate Kendall tau for each pair
            corr = np.eye(n_assets)
            
            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    try:
                        tau, _ = kendalltau(returns[i], returns[j])
                        corr[i, j] = tau
                        corr[j, i] = tau
                    except Exception:
                        # Fallback to Pearson
                        rho = np.corrcoef(returns[i], returns[j])[0, 1]
                        corr[i, j] = rho
                        corr[j, i] = rho
            
            return corr
            
        except ImportError:
            # Fallback to Pearson if scipy not available
            return self._pearson_correlation(returns)
    
    def _post_process(self, corr: np.ndarray) -> np.ndarray:
        """Post-process correlation matrix"""
        # Handle NaN values
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure symmetry
        corr = (corr + corr.T) / 2
        
        # Ensure diagonal is 1
        np.fill_diagonal(corr, 1.0)
        
        # Clip to valid range
        corr = np.clip(corr, -0.99, 0.99)
        
        return corr
    
    def calculate_rolling_correlation(self, 
                                     returns: np.ndarray, 
                                     window: int = 20,
                                     min_periods: int = 10) -> np.ndarray:
        """
        Calculate rolling correlation matrix
        
        Args:
            returns: Array of returns [n_assets, n_timesteps]
            window: Rolling window size
            min_periods: Minimum periods required
            
        Returns:
            Rolling correlation [n_assets, n_assets, n_windows]
        """
        n_assets, n_timesteps = returns.shape
        
        if n_timesteps < min_periods:
            return np.eye(n_assets)[:, :, np.newaxis]
        
        n_windows = max(0, n_timesteps - min_periods + 1)
        rolling_corr = np.zeros((n_assets, n_assets, n_windows))
        
        for w in range(n_windows):
            start = w
            end = w + min_periods + window
            if end > n_timesteps:
                break
            
            window_returns = returns[:, start:end]
            rolling_corr[:, :, w] = self.calculate(window_returns)
        
        return rolling_corr
    
    def estimate_covariance(self, 
                           returns: np.ndarray, 
                           method: str = 'ledoit_wolf') -> np.ndarray:
        """
        Estimate covariance matrix with regularization
        
        Args:
            returns: Array of returns [n_assets, n_timesteps]
            method: Estimation method ('ledoit_wolf', 'shrinkage', 'sample')
            
        Returns:
            Covariance matrix [n_assets, n_assets]
        """
        n_assets, n_timesteps = returns.shape
        
        if n_assets == 0:
            return np.array([])
        
        if n_assets == 1:
            return np.array([[np.var(returns[0])]])
        
        if method == 'sample':
            # Sample covariance
            cov = np.cov(returns)
            
        elif method == 'ledoit_wolf':
            # Ledoit-Wolf shrinkage
            try:
                from sklearn.covariance import LedoitWolf
                lw = LedoitWolf()
                cov = lw.covariance_
            except ImportError:
                # Fallback to sample with diagonal regularization
                cov = np.cov(returns)
                cov = self._regularize_covariance(cov)
        
        elif method == 'shrinkage':
            # Simple shrinkage to identity
            sample_cov = np.cov(returns)
            n = n_timesteps
            p = n_assets
            
            # Shrinkage factor
            delta = (p - 2) / (n * p) if n > 0 else 1.0
            
            # Shrink towards identity
            cov = (1 - delta) * sample_cov + delta * np.eye(n_assets)
        
        else:
            cov = np.cov(returns)
        
        # Post-process
        cov = self._regularize_covariance(cov)
        
        return cov
    
    def _regularize_covariance(self, cov: np.ndarray) -> np.ndarray:
        """Regularize covariance matrix"""
        # Ensure symmetry
        cov = (cov + cov.T) / 2
        
        # Ensure positive definite
        try:
            # Try to make it positive definite
            eigvals, eigvecs = np.linalg.eigh(cov)
            
            # Set negative eigenvalues to small positive
            eigvals = np.maximum(eigvals, 1e-8)
            
            # Reconstruct covariance
            cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
        except Exception:
            # Fallback: add small positive constant to diagonal
            cov = cov + np.eye(cov.shape[0]) * 1e-6
        
        return cov
    
    def calculate_optimal_weights(self, 
                                 cov_matrix: np.ndarray,
                                 method: str = 'risk_parity',
                                 target_volatility: float = 0.15) -> np.ndarray:
        """
        Calculate optimal portfolio weights
        
        Args:
            cov_matrix: Covariance matrix [n_assets, n_assets]
            method: Weighting method ('risk_parity', 'minimum_variance', 'equal')
            target_volatility: Target portfolio volatility
            
        Returns:
            Portfolio weights [n_assets]
        """
        n_assets = cov_matrix.shape[0]
        
        if n_assets == 0:
            return np.array([])
        
        if method == 'equal':
            weights = np.ones(n_assets) / n_assets
            
        elif method == 'risk_parity':
            # Risk parity: weights inversely proportional to volatility
            volatilities = np.sqrt(np.diag(cov_matrix))
            inv_vol = 1.0 / (volatilities + 1e-10)
            
            weights = inv_vol / np.sum(inv_vol)
            
            # Scale to target volatility
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            if portfolio_vol > 0 and target_volatility > 0:
                scaling = target_volatility / portfolio_vol
                weights = weights * min(scaling, 1.0)  # Don't leverage
                
        elif method == 'minimum_variance':
            # Minimum variance portfolio (Markowitz)
            try:
                # Solve: min w'Σw s.t. w'1 = 1
                ones = np.ones(n_assets)
                weights = np.linalg.inv(cov_matrix) @ ones
                weights = weights / np.sum(weights)
            except Exception:
                # Fallback to risk parity
                weights = self.calculate_optimal_weights(cov_matrix, 'risk_parity', target_volatility)
        
        else:
            weights = np.ones(n_assets) / n_assets
        
        # Ensure no negative weights
        weights = np.maximum(weights, 0.0)
        
        # Renormalize
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(n_assets) / n_assets
        
        return weights
    
    def calculate_portfolio_risk(self, 
                                weights: np.ndarray,
                                cov_matrix: np.ndarray) -> Dict[str, float]:
        """
        Calculate portfolio risk metrics
        
        Args:
            weights: Portfolio weights [n_assets]
            cov_matrix: Covariance matrix [n_assets, n_assets]
            
        Returns:
            Dictionary with risk metrics
        """
        if len(weights) == 0 or len(cov_matrix) == 0:
            return {}
        
        # Portfolio variance
        portfolio_variance = weights @ cov_matrix @ weights
        
        # Portfolio volatility
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Component contributions
        marginal_risk = cov_matrix @ weights
        component_risk = weights * marginal_risk
        component_contrib = component_risk / portfolio_variance if portfolio_variance > 0 else np.zeros_like(weights)
        
        # Diversification ratio
        avg_volatility = np.sqrt(np.mean(np.diag(cov_matrix)))
        if avg_volatility > 0 and portfolio_volatility > 0:
            diversification_ratio = avg_volatility / portfolio_volatility
        else:
            diversification_ratio = 1.0
        
        # Effective number of assets (concentration)
        effective_assets = 1.0 / np.sum(weights ** 2) if np.sum(weights ** 2) > 0 else 0.0
        
        return {
            'portfolio_variance': float(portfolio_variance),
            'portfolio_volatility': float(portfolio_volatility),
            'diversification_ratio': float(diversification_ratio),
            'effective_assets': float(effective_assets),
            'concentration': float(1.0 / effective_assets) if effective_assets > 0 else 0.0,
            'max_component_contrib': float(np.max(component_contrib)) if len(component_contrib) > 0 else 0.0,
            'min_component_contrib': float(np.min(component_contrib)) if len(component_contrib) > 0 else 0.0,
        }


class PortfolioOptimizer:
    """
    Portfolio optimization utilities
    """
    
    @staticmethod
    def calculate_expected_returns(price_series: np.ndarray, 
                                  method: str = 'historical') -> np.ndarray:
        """
        Calculate expected returns for each asset
        
        Args:
            price_series: Price series [n_assets, n_timesteps]
            method: Method for estimation
            
        Returns:
            Expected returns [n_assets]
        """
        n_assets, n_timesteps = price_series.shape
        
        if n_assets == 0:
            return np.array([])
        
        if method == 'historical':
            # Historical mean returns
            returns = np.diff(np.log(price_series), axis=1)
            expected_returns = np.mean(returns, axis=1) * 252  # Annualized
            
        elif method == 'constant':
            # Constant expected return
            expected_returns = np.ones(n_assets) * 0.1  # 10% annual
            
        else:
            # Default to historical
            returns = np.diff(np.log(price_series), axis=1)
            expected_returns = np.mean(returns, axis=1) * 252
        
        return expected_returns
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, 
                              risk_free_rate: float = 0.02,
                              annualization_factor: float = 252) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (annual)
            annualization_factor: Annualization factor
            
        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return < 1e-10:
            return 0.0
        
        excess_return = mean_return - (risk_free_rate / annualization_factor)
        sharpe = (excess_return / std_return) * np.sqrt(annualization_factor)
        
        return sharpe
    
    @staticmethod
    def calculate_max_drawdown(price_series: np.ndarray) -> float:
        """
        Calculate maximum drawdown
        
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


# Factory functions
def create_correlation_matrix(min_periods: int = 10) -> CorrelationMatrix:
    """Create correlation matrix calculator"""
    return CorrelationMatrix(min_periods)


def create_portfolio_optimizer() -> PortfolioOptimizer:
    """Create portfolio optimizer"""
    return PortfolioOptimizer()