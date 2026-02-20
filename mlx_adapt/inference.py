"""
MLX inference module for HRM.
"""
import mlx.core as mx
import numpy as np
from typing import Optional, Dict, Any
from .hrm_model import HRMModel, HRMConfig


class HRMInference:
    """
    Inference wrapper for HRM model.
    """
    
    def __init__(self, model: HRMModel, config: HRMConfig):
        self.model = model
        self.config = config
        
    def predict(self, 
                features: np.ndarray,
                batch_size: int = 1) -> np.ndarray:
        """
        Run inference on features.
        
        Args:
            features: Input features [B, T, input_dim]
            batch_size: Batch size
            
        Returns:
            Predictions [B, n_models]
        """
        # Convert to MLX array
        if isinstance(features, np.ndarray):
            features_mx = mx.array(features, dtype=mx.float32)
        else:
            features_mx = features
            
        # Ensure correct shape
        if len(features_mx.shape) == 2:
            # Add sequence dimension
            features_mx = features_mx[None, None, :]
        elif len(features_mx.shape) == 1:
            # Add batch and sequence dimensions
            features_mx = features_mx[None, None, :]
            
        # Run inference
        with mx.no_grad():
            predictions = self.model(features_mx)
            
        # Convert back to numpy
        return np.array(predictions)
    
    def predict_batch(self, 
                     features_list: list,
                     batch_size: int = 32) -> np.ndarray:
        """
        Predict on batch of features.
        
        Args:
            features_list: List of feature arrays
            batch_size: Batch size
            
        Returns:
            Combined predictions
        """
        all_predictions = []
        
        for i in range(0, len(features_list), batch_size):
            batch = features_list[i:i + batch_size]
            
            # Stack batch
            batch_array = np.stack(batch, axis=0)
            
            # Predict
            batch_predictions = self.predict(batch_array)
            all_predictions.append(batch_predictions)
            
        if not all_predictions:
            return np.array([])
            
        return np.concatenate(all_predictions, axis=0)
    
    def get_model_weights(self) -> Dict[str, mx.array]:
        """Get model weights"""
        return self.model.parameters()
    
    def load_weights(self, weights_path: str):
        """Load model weights from file"""
        self.model.load_weights(weights_path)
    
    def save_weights(self, weights_path: str):
        """Save model weights to file"""
        self.model.save_weights(weights_path)


def enable_ane_optimization():
    """Enable ANE optimizations for MLX"""
    # Set default device to GPU (which includes ANE)
    mx.set_default_device(mx.gpu)
    
    # Ensure float32 for ANE compatibility
    mx.set_default_dtype(mx.float32)


def setup_mlx_device():
    """Setup MLX device for optimal performance"""
    enable_ane_optimization()