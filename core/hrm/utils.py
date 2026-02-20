"""
Utility functions for HRM modules.

Pure logic, no framework dependencies.
"""
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class HRMConstants:
    """Constants for HRM system"""
    N_REGIMES = 6  # TREND, MEAN_REVERSION, VOLATILITY, STAT_ARB, SYSTEMATIC, ML
    REGIME_NAMES = [
        "trend",
        "mean_reversion", 
        "volatility",
        "stat_arb",
        "systematic",
        "ml"
    ]


def safe_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax.
    
    Args:
        logits: Input logits
        axis: Axis to compute softmax over
        
    Returns:
        Softmax probabilities
    """
    # Subtract max for numerical stability
    max_logits = np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    
    # Normalize
    sum_exp = np.sum(exp_logits, axis=axis, keepdims=True)
    
    # Avoid division by zero
    sum_exp = np.maximum(sum_exp, 1e-10)
    
    return exp_logits / sum_exp


def clip_and_normalize(values: np.ndarray, 
                      min_val: float = -1.0, 
                      max_val: float = 1.0) -> np.ndarray:
    """
    Clip values to range and normalize.
    
    Args:
        values: Input values
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clipped and normalized values
    """
    clipped = np.clip(values, min_val, max_val)
    # Normalize to [-1, 1] if needed
    if np.max(np.abs(clipped)) > 0:
        clipped = clipped / np.max(np.abs(clipped))
    return clipped


def compute_entropy(probabilities: np.ndarray) -> float:
    """
    Compute entropy of probability distribution.
    
    Args:
        probabilities: Probability distribution
        
    Returns:
        Entropy value
    """
    probabilities = np.clip(probabilities, 1e-10, 1.0)
    return -np.sum(probabilities * np.log(probabilities))


def compute_regime_scores(signals: np.ndarray, 
                         regime_weights: np.ndarray) -> np.ndarray:
    """
    Compute regime-weighted scores.
    
    Args:
        signals: Signal matrix [n_models, n_regimes]
        regime_weights: Regime weights [n_regimes]
        
    Returns:
        Weighted scores [n_models]
    """
    if len(signals.shape) == 1:
        signals = signals.reshape(1, -1)
    
    # Weight each model's signal by regime weights
    weighted = signals * regime_weights
    
    # Sum over regimes for each model
    scores = np.sum(weighted, axis=-1)
    
    return scores


def create_regime_weights(n_regimes: int = 6, 
                         strategy: str = "balanced") -> np.ndarray:
    """
    Create regime weights for different strategies.
    
    Args:
        n_regimes: Number of regimes
        strategy: Weighting strategy
        
    Returns:
        Regime weights
    """
    if strategy == "balanced":
        weights = np.ones(n_regimes) / n_regimes
    elif strategy == "trend_focused":
        weights = np.array([0.4, 0.1, 0.2, 0.1, 0.1, 0.1])
    elif strategy == "mean_reversion_focused":
        weights = np.array([0.1, 0.4, 0.1, 0.1, 0.2, 0.1])
    elif strategy == "volatility_focused":
        weights = np.array([0.1, 0.1, 0.4, 0.1, 0.2, 0.1])
    elif strategy == "conservative":
        weights = np.array([0.1, 0.2, 0.1, 0.2, 0.3, 0.1])
    elif strategy == "aggressive":
        weights = np.array([0.3, 0.1, 0.3, 0.1, 0.1, 0.1])
    else:
        weights = np.ones(n_regimes) / n_regimes
    
    return weights / np.sum(weights)


def validate_input_dimensions(input_array: np.ndarray, 
                            expected_shape: Tuple[int, ...],
                            name: str = "input") -> bool:
    """
    Validate input dimensions.
    
    Args:
        input_array: Input array to validate
        expected_shape: Expected shape
        name: Name for error messages
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    if input_array.shape != expected_shape:
        raise ValueError(
            f"{name} shape mismatch: expected {expected_shape}, "
            f"got {input_array.shape}"
        )
    return True


def batch_process(func: callable, 
                 inputs: List[np.ndarray], 
                 batch_size: int = 32) -> List[np.ndarray]:
    """
    Process inputs in batches.
    
    Args:
        func: Function to apply
        inputs: List of input arrays
        batch_size: Batch size
        
    Returns:
        List of processed outputs
    """
    results = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]
        batch_results = func(batch)
        results.extend(batch_results)
    return results


def moving_average(data: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Compute moving average.
    
    Args:
        data: Input data
        window: Window size
        
    Returns:
        Moving average
    """
    if len(data) < window:
        return data
    
    weights = np.ones(window) / window
    return np.convolve(data, weights, mode='valid')


def compute_returns(prices: np.ndarray) -> np.ndarray:
    """
    Compute returns from prices.
    
    Args:
        prices: Price series
        
    Returns:
        Returns series
    """
    if len(prices) < 2:
        return np.zeros_like(prices)
    
    returns = (prices[1:] - prices[:-1]) / prices[:-1]
    # Pad with zero at beginning
    padded_returns = np.zeros_like(prices)
    padded_returns[1:] = returns
    return padded_returns


def compute_sharpe_ratio(returns: np.ndarray, 
                        risk_free_rate: float = 0.0) -> float:
    """
    Compute Sharpe ratio.
    
    Args:
        returns: Returns series
        risk_free_rate: Risk-free rate
        
    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns)
    
    if std_return == 0:
        return 0.0
    
    return mean_return / std_return * np.sqrt(252)  # Annualized


def compute_max_drawdown(prices: np.ndarray) -> float:
    """
    Compute maximum drawdown.
    
    Args:
        prices: Price series
        
    Returns:
        Maximum drawdown
    """
    if len(prices) < 2:
        return 0.0
    
    cumulative = np.maximum.accumulate(prices)
    drawdown = (prices - cumulative) / cumulative
    return np.min(drawdown)


def compute_var(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """
    Compute Value at Risk.
    
    Args:
        returns: Returns series
        confidence_level: Confidence level
        
    Returns:
        VaR value
    """
    if len(returns) == 0:
        return 0.0
    
    # Sort returns
    sorted_returns = np.sort(returns)
    
    # Compute VaR index
    var_index = int((1 - confidence_level) * len(sorted_returns))
    
    return sorted_returns[var_index]


def compute_cvar(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """
    Compute Conditional Value at Risk.
    
    Args:
        returns: Returns series
        confidence_level: Confidence level
        
    Returns:
        CVaR value
    """
    if len(returns) == 0:
        return 0.0
    
    # Sort returns
    sorted_returns = np.sort(returns)
    
    # Compute VaR index
    var_index = int((1 - confidence_level) * len(sorted_returns))
    
    # Compute average of returns worse than VaR
    if var_index == 0:
        return np.mean(sorted_returns[:1])
    
    return np.mean(sorted_returns[:var_index])


class HRMStateManager:
    """Manages HRM module states"""
    
    def __init__(self):
        self.states = {}
        self.history = []
        
    def store_state(self, name: str, state: Dict[str, Any]):
        """Store module state"""
        self.states[name] = state
        self.history.append({
            'timestamp': len(self.history),
            'name': name,
            'state': state.copy()
        })
        
        # Keep history manageable
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
    
    def get_state(self, name: str) -> Dict[str, Any]:
        """Get module state"""
        return self.states.get(name, {})
    
    def clear(self):
        """Clear all states"""
        self.states.clear()
        self.history.clear()


# Factory functions
def create_hrm_constants() -> HRMConstants:
    """Create HRM constants"""
    return HRMConstants()


def create_hrm_state_manager() -> HRMStateManager:
    """Create HRM state manager"""
    return HRMStateManager()