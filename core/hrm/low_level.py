"""
Low-level HRM module - fast pattern recognition and feature extraction.

Pure logic, no framework dependencies.
"""
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np


@dataclass
class LowLevelConfig:
    """Configuration for low-level module"""
    n_features: int = 15
    n_assets: int = 128
    hidden_dim: int = 64
    cycles: int = 2
    layers: int = 2
    lookback: int = 16


class LowLevelModule:
    """
    Low-level reasoning module.
    
    Responsibilities:
    - Fast pattern recognition
    - Feature extraction
    - Local signal processing
    - Time-series analysis
    """
    
    def __init__(self, config: LowLevelConfig):
        self.config = config
        self.buffer = np.zeros((config.lookback, config.hidden_dim))
        self.buffer_idx = 0
        
    def forward(self, 
                input_features: np.ndarray,
                context: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Forward pass of low-level module.
        
        Args:
            input_features: Raw input features [n_assets, n_features]
            context: Additional context information [context_dim]
            
        Returns:
            processed_features: Processed features [hidden_dim]
            metadata: Additional information
        """
        # Extract features from input
        extracted = self._extract_features(input_features)
        
        # Combine with context
        combined = np.concatenate([extracted, context])
        
        # Update buffer
        self._update_buffer(combined)
        
        # Process buffer
        processed = self._process_buffer()
        
        metadata = {
            'buffer_size': len(self.buffer),
            'extracted_norm': np.linalg.norm(extracted),
            'processed_norm': np.linalg.norm(processed)
        }
        
        return processed, metadata
    
    def _extract_features(self, input_features: np.ndarray) -> np.ndarray:
        """Extract features from input"""
        # Basic feature extraction - in production this would be more sophisticated
        if len(input_features.shape) == 2:
            # Multiple assets: aggregate
            features = np.mean(input_features, axis=0)
        else:
            features = input_features
            
        # Ensure output size matches hidden_dim
        if features.shape[0] != self.config.hidden_dim:
            # Simple padding/truncation
            if features.shape[0] < self.config.hidden_dim:
                padded = np.zeros(self.config.hidden_dim)
                padded[:len(features)] = features
                return padded
            else:
                return features[:self.config.hidden_dim]
        return features
    
    def _update_buffer(self, features: np.ndarray):
        """Update circular buffer"""
        # Ensure features match buffer size
        if len(features) != self.config.hidden_dim:
            if len(features) < self.config.hidden_dim:
                # Pad with zeros
                padded = np.zeros(self.config.hidden_dim)
                padded[:len(features)] = features
                features = padded
            else:
                # Truncate
                features = features[:self.config.hidden_dim]
        
        self.buffer[self.buffer_idx] = features
        self.buffer_idx = (self.buffer_idx + 1) % len(self.buffer)
    
    def _process_buffer(self) -> np.ndarray:
        """Process buffer to extract temporal patterns"""
        if np.all(self.buffer == 0):
            return np.zeros(self.config.hidden_dim)
        
        # Simple temporal processing - average of recent states
        recent = self.buffer[:self.buffer_idx]
        if len(recent) == 0:
            recent = self.buffer
            
        # Compute temporal features
        mean = np.mean(recent, axis=0)
        std = np.std(recent, axis=0)
        trend = recent[-1] - recent[0] if len(recent) > 1 else np.zeros_like(mean)
        
        # Combine features
        processed = np.concatenate([mean, std, trend])[:self.config.hidden_dim]
        
        return processed
    
    def reset(self):
        """Reset buffer"""
        self.buffer = np.zeros((self.config.lookback, self.config.hidden_dim))
        self.buffer_idx = 0


class LowLevelFeature:
    """Low-level feature output"""
    
    def __init__(self, 
                 features: np.ndarray,
                 confidence: float,
                 metadata: Dict[str, Any]):
        self.features = features
        self.confidence = confidence
        self.metadata = metadata
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'features': self.features.tolist(),
            'confidence': float(self.confidence),
            'metadata': self.metadata
        }


class LowLevelProcessor:
    """
    Processor for low-level features.
    
    Handles feature extraction and preprocessing.
    """
    
    def __init__(self, config: LowLevelConfig):
        self.config = config
        self.modules = {
            'primary': LowLevelModule(config),
            'secondary': LowLevelModule(config)
        }
        self.active_module = 'primary'
        
    def process(self, 
                input_data: np.ndarray,
                context: np.ndarray = None) -> LowLevelFeature:
        """
        Process input data.
        
        Args:
            input_data: Input features [n_assets, n_features]
            context: Additional context [context_dim]
            
        Returns:
            LowLevelFeature
        """
        if context is None:
            context = np.zeros(self.config.hidden_dim)
            
        # Get active module
        module = self.modules[self.active_module]
        
        # Forward pass
        processed, metadata = module.forward(input_data, context)
        
        # Compute confidence
        confidence = self._compute_confidence(processed)
        
        return LowLevelFeature(processed, confidence, metadata)
    
    def _compute_confidence(self, features: np.ndarray) -> float:
        """Compute confidence score from features"""
        # Confidence based on feature magnitude
        magnitude = np.linalg.norm(features)
        return min(1.0, magnitude / 10.0)
    
    def switch_module(self):
        """Switch to backup module"""
        self.active_module = 'secondary' if self.active_module == 'primary' else 'primary'
    
    def reset(self):
        """Reset all modules"""
        for module in self.modules.values():
            module.reset()


# Factory functions for creating low-level modules
def create_low_level_module(config: LowLevelConfig = None) -> LowLevelModule:
    """Factory function to create low-level module"""
    if config is None:
        config = LowLevelConfig()
    return LowLevelModule(config)


def create_low_level_processor(config: LowLevelConfig = None) -> LowLevelProcessor:
    """Factory function to create low-level processor"""
    if config is None:
        config = LowLevelConfig()
    return LowLevelProcessor(config)