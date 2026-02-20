"""
High-level HRM module - high-level reasoning and decision making.

Pure logic, no framework dependencies.
"""
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np


@dataclass
class HighLevelConfig:
    """Configuration for high-level module"""
    n_regimes: int = 6
    n_models: int = 5
    n_assets: int = 128
    hidden_dim: int = 128
    cycles: int = 2
    layers: int = 2


class HighLevelModule:
    """
    High-level reasoning module.
    
    Responsibilities:
    - Long-term pattern recognition
    - Regime classification
    - Portfolio allocation across regimes
    - Strategic decision making
    """
    
    def __init__(self, config: HighLevelConfig):
        self.config = config
        self.state = np.zeros(config.hidden_dim)
        self.memory = []
        
    def forward(self, 
                low_level_output: np.ndarray,
                context: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Forward pass of high-level module.
        
        Args:
            low_level_output: Output from low-level module [hidden_dim]
            context: Additional context information [context_dim]
            
        Returns:
            regime_weights: Probability distribution over regimes [n_regimes]
            metadata: Additional information
        """
        # Combine low-level output with context
        combined = np.concatenate([low_level_output, context])
        
        # Update internal state
        self.state = self._update_state(combined)
        
        # Store in memory
        self.memory.append(self.state.copy())
        if len(self.memory) > 100:
            self.memory.pop(0)
            
        # Compute regime weights
        regime_weights = self._compute_regime_weights(self.state)
        
        metadata = {
            'state_norm': np.linalg.norm(self.state),
            'memory_size': len(self.memory),
            'regime_distribution': regime_weights.tolist()
        }
        
        return regime_weights, metadata
    
    def _update_state(self, combined: np.ndarray) -> np.ndarray:
        """Update internal state"""
        # Simple update - in production this would be more sophisticated
        if len(combined) >= self.config.hidden_dim:
            return 0.9 * self.state + 0.1 * combined[:self.config.hidden_dim]
        else:
            # Pad with zeros if combined is too short
            padded = np.zeros(self.config.hidden_dim)
            padded[:len(combined)] = combined
            return 0.9 * self.state + 0.1 * padded
    
    def _compute_regime_weights(self, state: np.ndarray) -> np.ndarray:
        """Compute regime weights from state"""
        # Simple softmax-like computation
        if len(state.shape) == 0:
            scores = np.random.randn(self.config.n_regimes)
        else:
            # Use first few dimensions for scoring
            n_scores = min(self.config.n_regimes, len(state))
            scores = state[:n_scores] + np.random.randn(n_scores) * 0.1
        
        # Ensure we have enough scores
        if len(scores) < self.config.n_regimes:
            # Pad with random scores
            padding = np.random.randn(self.config.n_regimes - len(scores))
            scores = np.concatenate([scores, padding])
        
        # Add regularization to prevent extreme weights
        scores = scores + 0.1
        weights = np.exp(scores - np.max(scores))
        return weights / np.sum(weights)
    
    def reset(self):
        """Reset state"""
        self.state = np.zeros(self.config.hidden_dim)
        self.memory.clear()


class HighLevelDecision:
    """High-level decision output"""
    
    def __init__(self, 
                 regime_weights: np.ndarray,
                 confidence: float,
                 metadata: Dict[str, Any]):
        self.regime_weights = regime_weights
        self.confidence = confidence
        self.metadata = metadata
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'regime_weights': self.regime_weights.tolist(),
            'confidence': float(self.confidence),
            'metadata': self.metadata
        }


class HighLevelController:
    """
    Controller for high-level reasoning.
    
    Orchestrates multiple high-level modules and manages state transitions.
    """
    
    def __init__(self, config: HighLevelConfig):
        self.config = config
        self.modules = {
            'primary': HighLevelModule(config),
            'backup': HighLevelModule(config)
        }
        self.active_module = 'primary'
        
    def decide(self, 
               signals: np.ndarray,
               market_state: np.ndarray) -> HighLevelDecision:
        """
        Make high-level decision.
        
        Args:
            signals: Input signals [n_models, signal_dim]
            market_state: Market state [state_dim]
            
        Returns:
            HighLevelDecision
        """
        # Aggregate signals
        aggregated_signal = np.mean(signals, axis=0) if len(signals.shape) > 1 else signals
        
        # Get active module
        module = self.modules[self.active_module]
        
        # Forward pass
        regime_weights, metadata = module.forward(aggregated_signal, market_state)
        
        # Compute confidence
        confidence = self._compute_confidence(regime_weights)
        
        return HighLevelDecision(regime_weights, confidence, metadata)
    
    def _compute_confidence(self, regime_weights: np.ndarray) -> float:
        """Compute confidence score from regime weights"""
        # Confidence based on weight distribution entropy
        entropy = -np.sum(regime_weights * np.log(regime_weights + 1e-10))
        max_entropy = np.log(len(regime_weights))
        return 1.0 - (entropy / max_entropy)
    
    def switch_module(self):
        """Switch to backup module"""
        self.active_module = 'backup' if self.active_module == 'primary' else 'primary'
    
    def reset(self):
        """Reset all modules"""
        for module in self.modules.values():
            module.reset()


# Factory functions for creating high-level modules
def create_high_level_module(config: HighLevelConfig = None) -> HighLevelModule:
    """Factory function to create high-level module"""
    if config is None:
        config = HighLevelConfig()
    return HighLevelModule(config)


def create_high_level_controller(config: HighLevelConfig = None) -> HighLevelController:
    """Factory function to create high-level controller"""
    if config is None:
        config = HighLevelConfig()
    return HighLevelController(config)