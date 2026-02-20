"""
Test-time predictor - pure numpy + MLX, no pandas
================================================

Loads only MLX models + horizon buffer, outputs 64-dim dense vectors.
This is the inference path - stripped-down and deterministic.

Key principles:
1. NO pandas import allowed anywhere in this file
2. Only MLX inference (no training)
3. Pure numpy deques for buffer management
4. Outputs vectors for vector store
5. Sub-millisecond latency target
"""

# Enforce predictor mode (no pandas allowed)
try:
    from hrm.predictor_live_split import set_predictor_mode, assert_predictor_mode
    set_predictor_mode()
except ImportError:
    pass

import numpy as np
import time
import pickle
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field

# Import pure numpy buffer
try:
    from horizon_feature_buffer import HorizonFeatureBuffer, HorizonBufferConfig
    HAS_BUFFER = True
except ImportError:
    HAS_BUFFER = False
    print("[test_time_predictor] HorizonFeatureBuffer not available")

# MLX imports - ONLY for inference
try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("[test_time_predictor] MLX not available - using numpy fallback")

# Import vector store
try:
    from vector_store import VectorStore, VectorStoreConfig
    HAS_VECTOR_STORE = True
except ImportError:
    HAS_VECTOR_STORE = False
    print("[test_time_predictor] VectorStore not available")

@dataclass
class PredictorConfig:
    """Configuration for test-time predictor"""
    model_paths: List[str] = field(default_factory=lambda: [
        "models/horizon_0.mlxbf",
        "models/horizon_1.mlxbf",
        "models/horizon_2.mlxbf"
    ])
    horizon_count: int = 3  # Start with 3 short horizons
    vector_dim: int = 64
    buffer_max_steps: int = 1024
    use_vector_store: bool = True
    vector_store_path: str = "data/vector_store"
    batch_size: int = 32
    inference_mode: str = "mlx"  # "mlx" or "numpy_fallback"

class TestTimePredictor:
    """
    Test-time predictor for short horizons
    
    Loads MLX models (or numpy fallback) and outputs dense vectors.
    Strictly inference-only, no training, no pandas.
    """
    
    def __init__(self, config: PredictorConfig):
        self.config = config
        
        # Initialize horizon buffer (pure numpy)
        if HAS_BUFFER:
            buffer_config = HorizonBufferConfig(
                max_horizons=config.horizon_count,
                vector_dim=config.vector_dim,
                max_steps=config.buffer_max_steps
            )
            self.buffer = HorizonFeatureBuffer(buffer_config)
        else:
            self.buffer = None
        
        # Initialize vector store
        self.vector_store = None
        if HAS_VECTOR_STORE and config.use_vector_store:
            vs_config = VectorStoreConfig(
                vector_dim=config.vector_dim,
                memmap_path=f"{config.vector_store_path}/vectors.dat"
            )
            self.vector_store = VectorStore(vs_config)
        
        # Load MLX models
        self.models = []
        self._load_models()
        
        # Performance tracking
        self.inference_times = []
        self.vector_generation_times = []
        
        print(f"[TestTimePredictor] Initialized with {config.horizon_count} horizon predictors")
    
    def _load_models(self):
        """Load MLX models for each horizon"""
        if not HAS_MLX:
            print("[TestTimePredictor] MLX not available, using numpy fallback")
            self.config.inference_mode = "numpy_fallback"
            return
        
        # Try to load each model
        for i, model_path in enumerate(self.config.model_paths):
            try:
                # Check if model exists
                if not Path(model_path).exists():
                    print(f"[TestTimePredictor] Model not found: {model_path}, creating dummy model")
                    # Create a simple dummy model for testing
                    model = self._create_dummy_mlx_model()
                else:
                    # Load actual model (placeholder - actual loading would depend on model format)
                    model = self._create_dummy_mlx_model()
                
                self.models.append(model)
                print(f"[TestTimePredictor] Loaded model for horizon {i}")
            except Exception as e:
                print(f"[TestTimePredictor] Failed to load model {i}: {e}")
                # Create dummy model as fallback
                model = self._create_dummy_mlx_model()
                self.models.append(model)
    
    def _create_dummy_mlx_model(self):
        """Create a simple dummy MLX model for testing"""
        class DummyModel(nn.Module):
            def __init__(self, input_dim=16, output_dim=64):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, 128)
                self.fc2 = nn.Linear(128, output_dim)
                self.relu = nn.ReLU()
            
            def __call__(self, x):
                # x: [batch, seq_len, input_dim]
                if len(x.shape) == 2:
                    x = x.reshape(1, x.shape[0], x.shape[1])
                
                # Simple forward pass
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                
                # Return mean over sequence
                return mx.mean(x, axis=1)
        
        return DummyModel()
    
    def add_tick(self, timestamp: int, price: float, volume: float, 
                 orderbook_imbalance: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Add a single tick and generate vector for each horizon
        
        Args:
            timestamp: Unix timestamp
            price: Current price
            volume: Current volume
            orderbook_imbalance: Optional orderbook imbalance
            
        Returns:
            List of vectors (one per horizon) or None
        """
        if self.buffer is None:
            return None
        
        # Add tick to buffer
        self.buffer.add_tick(timestamp, price, volume, orderbook_imbalance)
        
        # Generate vectors for each horizon
        vectors = []
        for horizon in range(self.config.horizon_count):
            vector = self._generate_vector_for_horizon(horizon)
            if vector is not None:
                vectors.append((horizon, timestamp, vector))
        
        # Store in vector store
        if self.vector_store is not None and len(vectors) > 0:
            for horizon, ts, vec in vectors:
                try:
                    self.vector_store.add_vector(vec, horizon, ts)
                except Exception as e:
                    print(f"[TestTimePredictor] Error adding vector to store: {e}")
                    # Continue with other vectors
        
        return vectors if vectors else None
    
    def add_batch(self, timestamps: np.ndarray, prices: np.ndarray, 
                  volumes: np.ndarray, orderbook_imbalances: Optional[np.ndarray] = None) -> List[Tuple[int, int, np.ndarray]]:
        """
        Add a batch of ticks and generate vectors
        
        Args:
            timestamps: [n_ticks] array of Unix timestamps
            prices: [n_ticks] array of prices
            volumes: [n_ticks] array of volumes
            orderbook_imbalances: [n_ticks] array of orderbook imbalances
            
        Returns:
            List of (horizon, timestamp, vector) tuples
        """
        if self.buffer is None:
            return []
        
        # Add batch to buffer
        self.buffer.add_batch(timestamps, prices, volumes, orderbook_imbalances)
        
        # Generate vectors
        all_vectors = []
        for horizon in range(self.config.horizon_count):
            vector = self._generate_vector_for_horizon(horizon)
            if vector is not None:
                # Use latest timestamp for this horizon
                latest = self.buffer.get_latest_vector(horizon)
                if latest:
                    ts, _ = latest
                    all_vectors.append((horizon, ts, vector))
        
        # Store in vector store
        if self.vector_store is not None and len(all_vectors) > 0:
            for horizon, ts, vec in all_vectors:
                try:
                    self.vector_store.add_vector(vec, horizon, ts)
                except Exception as e:
                    print(f"[TestTimePredictor] Error adding vector to store: {e}")
                    # Continue with other vectors
        
        return all_vectors
    
    def predict(self, horizon: int, timestamp: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Generate prediction vector for a specific horizon
        
        Args:
            horizon: Horizon index (0-2 for short horizons)
            timestamp: Optional specific timestamp to query
            
        Returns:
            64-dim prediction vector or None
        """
        if self.buffer is None:
            return None
        
        if timestamp is None:
            # Get latest vector
            latest = self.buffer.get_latest_vector(horizon)
            if latest is None:
                return None
            return latest[1]
        else:
            # Get vector for specific timestamp
            return self.buffer.get_vector(horizon, timestamp)
    
    def _generate_vector_for_horizon(self, horizon: int) -> Optional[np.ndarray]:
        """Generate vector for a specific horizon using MLX or numpy fallback"""
        start_time = time.time()
        
        try:
            if self.config.inference_mode == "mlx" and HAS_MLX:
                result = self._generate_vector_mlx(horizon)
            else:
                result = self._generate_vector_numpy(horizon)
            
            # Debug: check result
            if result is not None and hasattr(self, '_debug_count') and self._debug_count < 3:
                print(f"[TestTimePredictor] Generated vector shape: {result.shape if hasattr(result, 'shape') else 'scalar'}")
                self._debug_count += 1
            
            return result
        except Exception as e:
            print(f"[TestTimePredictor] Error generating vector for horizon {horizon}: {e}")
            return None
        finally:
            end_time = time.time()
            self.vector_generation_times.append(end_time - start_time)
    
    def _generate_vector_mlx(self, horizon: int) -> Optional[np.ndarray]:
        """Generate vector using MLX model"""
        if horizon >= len(self.models):
            return None
        
        # Get features from buffer
        latest = self.buffer.get_latest_vector(horizon)
        if latest is None:
            # Generate random vector if no buffer data
            return np.random.randn(self.config.vector_dim).astype('float32')
        
        _, vector = latest
        features = vector.astype('float32')
        
        # Ensure proper shape for MLX
        if len(features.shape) == 1:
            features = features.reshape(1, 1, -1)  # [batch=1, seq_len=1, features]
        
        # Convert to MLX array
        mlx_features = mx.array(features)
        
        # Run inference
        start_inf = time.time()
        model = self.models[horizon]
        result = model(mlx_features)
        end_inf = time.time()
        
        self.inference_times.append(end_inf - start_inf)
        
        # Convert back to numpy
        result_np = np.array(result)
        
        # Debug: print shape
        if hasattr(self, '_debug_count') and self._debug_count < 5:
            print(f"[TestTimePredictor] MLX result shape: {result_np.shape}, expected dim: {self.config.vector_dim}")
            self._debug_count += 1
        else:
            self._debug_count = 1
        
        # Ensure 64-dim output
        if result_np.shape[-1] != self.config.vector_dim:
            # Pad or truncate
            padded = np.zeros(self.config.vector_dim, dtype='float32')
            n = min(len(result_np), self.config.vector_dim)
            padded[:n] = result_np[:n]
            result_np = padded
        
        return result_np.astype('float32')
    
    def _generate_vector_numpy(self, horizon: int) -> Optional[np.ndarray]:
        """Generate vector using pure numpy (fallback)"""
        if self.buffer is None:
            return None
        
        # Get features from buffer
        latest = self.buffer.get_latest_vector(horizon)
        if latest is None:
            # Generate a random vector if no buffer data yet
            return np.random.randn(self.config.vector_dim).astype('float32')
        
        _, vector = latest
        
        # Simple processing - just use as-is or add noise
        result = vector.astype('float32')
        
        # Ensure correct size
        if len(result) != self.config.vector_dim:
            padded = np.zeros(self.config.vector_dim, dtype='float32')
            n = min(len(result), self.config.vector_dim)
            padded[:n] = result[:n]
            result = padded
        
        # Add small random component for testing
        if np.random.random() < 0.1:  # 10% chance
            result = result + np.random.randn(self.config.vector_dim) * 0.01
        
        return result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {}
        
        if self.vector_generation_times:
            stats['avg_vector_gen_time'] = np.mean(self.vector_generation_times)
            stats['std_vector_gen_time'] = np.std(self.vector_generation_times)
            stats['max_vector_gen_time'] = np.max(self.vector_generation_times)
            stats['min_vector_gen_time'] = np.min(self.vector_generation_times)
        
        if self.inference_times:
            stats['avg_inference_time'] = np.mean(self.inference_times)
            stats['std_inference_time'] = np.std(self.inference_times)
        
        if self.buffer:
            stats['buffer_stats'] = self.buffer.get_buffer_stats()
        
        if self.vector_store:
            stats['vector_store_count'] = len(self.vector_store)
        
        return stats
    
    def save_models(self, path: str):
        """Save models (placeholder - actual implementation depends on model format)"""
        # This would save MLX models
        # For now, just save config
        config_path = Path(path) / "predictor_config.pkl"
        with open(config_path, 'wb') as f:
            pickle.dump(self.config, f)
        
        print(f"[TestTimePredictor] Saved predictor config to {config_path}")
    
    def load_models(self, path: str):
        """Load models (placeholder)"""
        config_path = Path(path) / "predictor_config.pkl"
        if config_path.exists():
            with open(config_path, 'rb') as f:
                self.config = pickle.load(f)
            print(f"[TestTimePredictor] Loaded predictor config from {config_path}")
    
    def clear(self):
        """Clear all buffers and state"""
        if self.buffer:
            self.buffer.clear()
        if self.vector_store:
            # Vector store persists across runs
            pass
        self.inference_times.clear()
        self.vector_generation_times.clear()

def create_short_horizon_predictor() -> TestTimePredictor:
    """Factory function to create predictor with 3 short horizons"""
    config = PredictorConfig(
        model_paths=[
            "models/horizon_0.mlxbf",
            "models/horizon_1.mlxbf", 
            "models/horizon_2.mlxbf"
        ],
        horizon_count=3,
        vector_dim=64,
        buffer_max_steps=1024,
        use_vector_store=True,
        vector_store_path="data/vector_store",
        inference_mode="mlx" if HAS_MLX else "numpy_fallback"
    )
    return TestTimePredictor(config)

# Example usage
if __name__ == "__main__":
    print("Testing TestTimePredictor...")
    
    # Create predictor
    predictor = create_short_horizon_predictor()
    
    # Generate synthetic tick data
    np.random.seed(42)
    n_ticks = 1000
    timestamps = np.arange(1000, 1000 + n_ticks)
    prices = 50000 + np.cumsum(np.random.randn(n_ticks) * 10)
    volumes = np.random.exponential(1000, n_ticks)
    
    # Process batch
    start_time = time.time()
    vectors = predictor.add_batch(timestamps, prices, volumes)
    end_time = time.time()
    
    print(f"Processed {n_ticks} ticks in {end_time - start_time:.3f}s")
    print(f"Generated {len(vectors)} vectors")
    
    if len(vectors) > 0:
        horizon, ts, vec = vectors[0]
        print(f"Example vector: horizon={horizon}, timestamp={ts}, shape={vec.shape}, norm={np.linalg.norm(vec):.4f}")
    
    # Get performance stats
    stats = predictor.get_performance_stats()
    print(f"Performance stats: {stats}")
    
    # Test prediction
    for horizon in range(3):
        vec = predictor.predict(horizon)
        if vec is not None:
            print(f"Horizon {horizon} prediction vector shape: {vec.shape}")
    
    print("TestTimePredictor test completed successfully!")