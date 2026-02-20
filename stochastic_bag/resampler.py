"""
Stochastic Bag Resampler
=========================

Handles the daily resampling of the 30-pair stochastic bag using:
- Dirichlet weights based on recent Sharpe ratios
- Correlation matrix to avoid over-concentration
- Multinomial selection for bag selection
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .compass import StochasticCompass


class StochasticBagResampler:
    """
    Handles daily resampling of stochastic bag
    """
    
    def __init__(self, 
                 n_codecs: int = 24,
                 bag_size: int = 30,
                 seed: int = 42):
        """
        Initialize stochastic bag resampler
        
        Args:
            n_codecs: Total number of codecs (24)
            bag_size: Size of stochastic bag (30)
            seed: Random seed
        """
        self.n_codecs = n_codecs
        self.bag_size = bag_size
        self.seed = seed
        
        # Initialize compass
        self.compass = StochasticCompass(seed=seed)
        
        # Performance tracking
        self.recent_sharpes = np.zeros(n_codecs)
        self.performance_history = []
        
        # Current bag
        self.current_bag = None
        self.current_weights = None
        
    def resample_daily(self, 
                      performance_data: Dict[int, Dict[str, float]] = None,
                      price_series: Optional[np.ndarray] = None) -> Dict[str, any]:
        """
        Resample the stochastic bag for the day
        
        Args:
            performance_data: Dictionary of codec_id -> performance metrics
            price_series: Optional price series for correlation calculation
            
        Returns:
            Dictionary with:
                - 'bag': list of codec indices
                - 'weights': Dirichlet weights
                - 'correlation_matrix': correlation matrix used
                - 'selected_count': number selected
        """
        # Update recent sharpes from performance data
        if performance_data is not None:
            for codec_id, metrics in performance_data.items():
                if 0 <= codec_id < self.n_codecs:
                    self.recent_sharpes[codec_id] = metrics.get('sharpe', 0.0)
        
        # Calculate Dirichlet weights
        weights = self.compass.dirichlet_weights(self.recent_sharpes)
        
        # Calculate correlation matrix if price series provided
        correlation_matrix = None
        if price_series is not None and price_series.shape[0] >= self.n_codecs:
            correlation_matrix = self.compass.correlation_matrix(price_series)
        else:
            # Use identity matrix (no correlation)
            correlation_matrix = np.eye(self.n_codecs)
        
        # Resample bag
        bag_indices = self.compass.bag_resample(
            n_codecs=self.n_codecs,
            n_selected=self.bag_size,
            correlation_matrix=correlation_matrix,
            weights=weights
        )
        
        # Store current state
        self.current_bag = bag_indices
        self.current_weights = weights
        
        # Record performance
        self.performance_history.append({
            'timestamp': np.datetime64('now'),
            'bag_size': len(bag_indices),
            'avg_weight': float(np.mean(weights)),
            'max_weight': float(np.max(weights)),
            'min_weight': float(np.min(weights)),
        })
        
        return {
            'bag': bag_indices,
            'weights': weights,
            'correlation_matrix': correlation_matrix,
            'selected_count': len(bag_indices),
            'avg_sharpe': float(np.mean(self.recent_sharpes)),
        }
    
    def update_performance(self, 
                          codec_id: int, 
                          sharpe: float,
                          win_rate: float = None,
                          pnl: float = None) -> None:
        """
        Update performance metrics for a codec
        
        Args:
            codec_id: Codec index (0-23)
            sharpe: Sharpe ratio
            win_rate: Optional win rate
            pnl: Optional PnL
        """
        if 0 <= codec_id < self.n_codecs:
            # Update Sharpe (use exponential moving average)
            alpha = 0.1  # Weight for new observation
            self.recent_sharpes[codec_id] = (
                (1 - alpha) * self.recent_sharpes[codec_id] + alpha * sharpe
            )
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get summary of current performance"""
        if len(self.recent_sharpes) == 0:
            return {}
        
        return {
            'mean_sharpe': float(np.mean(self.recent_sharpes)),
            'std_sharpe': float(np.std(self.recent_sharpes)),
            'max_sharpe': float(np.max(self.recent_sharpes)),
            'min_sharpe': float(np.min(self.recent_sharpes)),
            'positive_count': int(np.sum(self.recent_sharpes > 0)),
            'negative_count': int(np.sum(self.recent_sharpes < 0)),
        }
    
    def get_current_bag(self) -> Optional[List[int]]:
        """Get current bag indices"""
        return self.current_bag.tolist() if self.current_bag is not None else None
    
    def get_current_weights(self) -> Optional[np.ndarray]:
        """Get current Dirichlet weights"""
        return self.current_weights
    
    def estimate_portfolio_risk(self, 
                               cov_matrix: np.ndarray = None) -> Dict[str, float]:
        """
        Estimate portfolio risk metrics
        
        Args:
            cov_matrix: Covariance matrix of returns
            
        Returns:
            Dictionary with risk metrics
        """
        if self.current_weights is None:
            return {}
        
        if cov_matrix is None:
            # Use identity matrix as fallback
            cov_matrix = np.eye(self.n_codecs)
        
        weights = self.current_weights
        
        # Portfolio variance
        portfolio_variance = weights @ cov_matrix @ weights
        
        # Portfolio volatility
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Diversification ratio (1 if perfectly correlated, >1 if diversified)
        avg_volatility = np.sqrt(np.mean(np.diag(cov_matrix)))
        if avg_volatility > 0:
            diversification_ratio = avg_volatility / portfolio_volatility
        else:
            diversification_ratio = 1.0
        
        # Effective number of assets (concentration measure)
        effective_assets = 1.0 / np.sum(weights ** 2)
        
        return {
            'portfolio_variance': float(portfolio_variance),
            'portfolio_volatility': float(portfolio_volatility),
            'diversification_ratio': float(diversification_ratio),
            'effective_assets': float(effective_assets),
            'concentration': float(1.0 / effective_assets) if effective_assets > 0 else 0.0,
        }


class BagStatistics:
    """
    Statistics for stochastic bag operations
    """
    
    @staticmethod
    def calculate_entropy(weights: np.ndarray) -> float:
        """
        Calculate entropy of weight distribution (measure of diversity)
        
        Args:
            weights: Weight distribution
            
        Returns:
            Entropy value (higher = more diverse)
        """
        weights = np.clip(weights, 1e-10, 1.0)
        return -np.sum(weights * np.log(weights))
    
    @staticmethod
    def calculate_gini(weights: np.ndarray) -> float:
        """
        Calculate Gini coefficient (measure of inequality)
        
        Args:
            weights: Weight distribution
            
        Returns:
            Gini coefficient (0 = perfect equality, 1 = perfect inequality)
        """
        weights = np.sort(weights)
        n = len(weights)
        index = np.arange(1, n + 1)
        
        return (np.sum((2 * index - n - 1) * weights)) / (n * np.sum(weights))
    
    @staticmethod
    def calculate_portfolio_turnover(old_weights: np.ndarray, 
                                    new_weights: np.ndarray) -> float:
        """
        Calculate portfolio turnover (0-1)
        
        Args:
            old_weights: Previous weights
            new_weights: New weights
            
        Returns:
            Turnover (fraction of portfolio that needs to be traded)
        """
        if len(old_weights) != len(new_weights):
            return 1.0  # Complete rebalance
        
        turnover = np.sum(np.abs(old_weights - new_weights)) / 2.0
        return float(turnover)
    
    @staticmethod
    def analyze_correlation_structure(corr_matrix: np.ndarray) -> Dict[str, float]:
        """
        Analyze correlation matrix structure
        
        Args:
            corr_matrix: Correlation matrix
            
        Returns:
            Dictionary with correlation statistics
        """
        n = corr_matrix.shape[0]
        
        # Remove diagonal
        off_diagonal = corr_matrix[~np.eye(n, dtype=bool)]
        
        return {
            'mean_correlation': float(np.mean(off_diagonal)),
            'std_correlation': float(np.std(off_diagonal)),
            'max_correlation': float(np.max(off_diagonal)),
            'min_correlation': float(np.min(off_diagonal)),
            'positive_correlation': float(np.sum(off_diagonal > 0) / len(off_diagonal)),
        }


# Factory function for creating resampler
def create_resampler(n_codecs: int = 24, 
                    bag_size: int = 30,
                    seed: int = 42) -> StochasticBagResampler:
    """
    Create a stochastic bag resampler
    
    Args:
        n_codecs: Number of codecs (24)
        bag_size: Size of bag (30)
        seed: Random seed
        
    Returns:
        StochasticBagResampler instance
    """
    return StochasticBagResampler(n_codecs, bag_size, seed)