"""
Pure numpy horizon feature buffer - no pandas allowed
====================================================

Rolling windows for 24 horizons, pure numpy operations only.
This is the inference path buffer - lightweight, deterministic, fast.

Predictor codecs stay pure numpy deques + MLX inference only (no pandas import even allowed in this file).
"""

import numpy as np
from typing import Deque, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field
import time

@dataclass
class HorizonBufferConfig:
    """Configuration for horizon feature buffer"""
    max_horizons: int = 24  # 24 horizons (0-23)
    vector_dim: int = 64  # 64-dim dense vectors
    max_steps: int = 1024  # Maximum steps per horizon
    min_steps: int = 64  # Minimum steps to generate valid vectors
    # Horizon durations in minutes (geometric progression)
    horizon_durations: Optional[List[int]] = None  # Will be set dynamically
    
    def __post_init__(self):
        if self.horizon_durations is None:
            # Create geometric progression for specified number of horizons
            self.horizon_durations = [
                int(2 ** i) for i in range(self.max_horizons)
            ]

class HorizonFeatureBuffer:
    """
    Pure numpy rolling windows for 24 horizons
    
    Each horizon maintains its own rolling window of features.
    No pandas DataFrame used anywhere.
    
    Input: Raw tick data (price, volume, orderbook) as numpy arrays
    Output: 64-dim dense vectors per horizon per step
    """
    
    def __init__(self, config: HorizonBufferConfig):
        self.config = config
        
        # Validate config
        assert len(config.horizon_durations) == config.max_horizons
        
        # Create deque buffers for each horizon
        # Each horizon stores: (timestamp, vector, raw_features)
        self.buffers: List[Deque[Tuple[int, np.ndarray, np.ndarray]]] = []
        for _ in range(config.max_horizons):
            self.buffers.append(deque(maxlen=config.max_steps))
        
        # Cache for computed vectors (horizon, timestamp) -> vector
        self.vector_cache: Dict[Tuple[int, int], np.ndarray] = {}
        
        # Last processed timestamp per horizon
        self.last_timestamps: List[int] = [-1] * config.max_horizons
        
        # Feature extraction state
        self.feature_extractor = FeatureExtractor()
        
        print(f"[HorizonBuffer] Initialized with {config.max_horizons} horizons")
    
    def add_tick(self, timestamp: int, price: float, volume: float, 
                 orderbook_imbalance: Optional[float] = None) -> None:
        """
        Add a single tick to all horizons
        
        Args:
            timestamp: Unix timestamp (seconds)
            price: Current price
            volume: Current volume
            orderbook_imbalance: Optional orderbook imbalance [0,1]
        """
        # Raw features for this tick (6-dim)
        raw_features = self._extract_raw_features(price, volume, orderbook_imbalance)
        
        # Update each horizon
        for horizon in range(self.config.max_horizons):
            # Check if this horizon should process this tick
            # (based on horizon duration and last processed time)
            if self._should_process_tick(horizon, timestamp):
                # Get features for this horizon (16-dim)
                horizon_features = self._get_horizon_features(horizon, timestamp, price, volume)
                
                # Store horizon features in buffer (16-dim)
                self.buffers[horizon].append((timestamp, horizon_features, raw_features))
                self.last_timestamps[horizon] = timestamp
                
                # Generate vector for this horizon (64-dim)
                vector = self._generate_vector(horizon)
                
                # Store in vector cache
                self.vector_cache[(horizon, timestamp)] = vector
    
    def add_batch(self, timestamps: np.ndarray, prices: np.ndarray, 
                  volumes: np.ndarray, orderbook_imbalances: Optional[np.ndarray] = None) -> None:
        """
        Add a batch of ticks
        
        Args:
            timestamps: [n_ticks] array of Unix timestamps
            prices: [n_ticks] array of prices
            volumes: [n_ticks] array of volumes
            orderbook_imbalances: [n_ticks] array of orderbook imbalances (optional)
        """
        n_ticks = len(timestamps)
        
        for i in range(n_ticks):
            obs_imb = orderbook_imbalances[i] if orderbook_imbalances is not None else None
            self.add_tick(timestamps[i], prices[i], volumes[i], obs_imb)
    
    def get_vector(self, horizon: int, timestamp: int) -> Optional[np.ndarray]:
        """Get vector for specific horizon and timestamp"""
        key = (horizon, timestamp)
        return self.vector_cache.get(key)
    
    def get_all_vectors_for_horizon(self, horizon: int) -> List[Tuple[int, np.ndarray]]:
        """Get all vectors for a specific horizon"""
        if horizon < 0 or horizon >= self.config.max_horizons:
            return []
        
        results = []
        for ts, vec, _ in self.buffers[horizon]:
            if vec is not None:
                results.append((ts, vec))
        return results
    
    def get_latest_vector(self, horizon: int) -> Optional[Tuple[int, np.ndarray]]:
        """Get the latest vector for a horizon"""
        if horizon < 0 or horizon >= self.config.max_horizons:
            return None
        
        buffer = self.buffers[horizon]
        if len(buffer) == 0:
            return None
        
        timestamp, _, _ = buffer[-1]
        
        # Get vector from cache
        vector = self.vector_cache.get((horizon, timestamp))
        if vector is not None:
            return (timestamp, vector)
        
        return None
    
    def get_all_vectors(self) -> List[Tuple[int, int, np.ndarray]]:
        """Get all vectors across all horizons"""
        results = []
        for horizon in range(self.config.max_horizons):
            for timestamp, vector, _ in self.buffers[horizon]:
                if vector is not None:
                    results.append((horizon, timestamp, vector))
        return results
    
    def get_buffer_stats(self) -> dict:
        """Get statistics about buffer state"""
        stats = {
            "horizons": [],
            "total_vectors": 0,
            "total_steps": 0
        }
        
        for horizon in range(self.config.max_horizons):
            buffer = self.buffers[horizon]
            n_steps = len(buffer)
            stats["horizons"].append({
                "horizon": horizon,
                "duration_min": self.config.horizon_durations[horizon],
                "steps": n_steps,
                "last_timestamp": self.last_timestamps[horizon] if n_steps > 0 else None
            })
            stats["total_steps"] += n_steps
        
        stats["total_vectors"] = len(self.vector_cache)
        return stats
    
    def clear(self) -> None:
        """Clear all buffers"""
        for buffer in self.buffers:
            buffer.clear()
        self.vector_cache.clear()
        self.last_timestamps = [-1] * self.config.max_horizons
    
    # Private methods
    
    def _extract_raw_features(self, price: float, volume: float, 
                             orderbook_imbalance: Optional[float]) -> np.ndarray:
        """Extract raw features from a single tick"""
        features = []
        
        # Price-based features
        features.append(price)
        features.append(np.log(price) if price > 0 else 0.0)
        
        # Volume features
        features.append(volume)
        features.append(np.log(volume + 1e-10))
        
        # Orderbook imbalance (if available)
        if orderbook_imbalance is not None:
            features.append(orderbook_imbalance)
            features.append(1.0 - orderbook_imbalance)  # Ask side
        else:
            features.extend([0.0, 0.0])
        
        # Convert to numpy array (6 dimensions)
        return np.array(features, dtype='float32')
    
    def _should_process_tick(self, horizon: int, timestamp: int) -> bool:
        """Check if this horizon should process this tick"""
        if len(self.buffers[horizon]) == 0:
            return True  # First tick for this horizon
        
        last_ts = self.last_timestamps[horizon]
        horizon_duration = self.config.horizon_durations[horizon]
        
        # Process if enough time has passed
        return (timestamp - last_ts) >= horizon_duration
    
    def _get_horizon_features(self, horizon: int, timestamp: int, 
                             price: float, volume: float) -> np.ndarray:
        """Get features for a specific horizon"""
        # Get recent data for this horizon
        buffer = self.buffers[horizon]
        
        if len(buffer) == 0:
            # First tick - use current values
            features = np.zeros(16, dtype='float32')
            features[0] = price  # current price
            features[1] = volume  # current volume
            return features
        
        # Extract features from recent data
        recent_data = list(buffer)[-10:]  # Last 10 steps
        
        prices = np.array([d[2][0] for d in recent_data])  # price is first raw feature
        volumes = np.array([d[2][2] for d in recent_data])  # volume is third raw feature
        
        # Calculate statistics
        features = np.zeros(16, dtype='float32')
        
        # Price features
        features[0] = price  # current price
        features[1] = np.mean(prices) if len(prices) > 0 else price
        features[2] = np.std(prices) if len(prices) > 1 else 0.0
        features[3] = np.min(prices) if len(prices) > 0 else price
        features[4] = np.max(prices) if len(prices) > 0 else price
        features[5] = np.median(prices) if len(prices) > 0 else price
        
        # Volume features
        features[6] = volume  # current volume
        features[7] = np.mean(volumes) if len(volumes) > 0 else volume
        features[8] = np.std(volumes) if len(volumes) > 1 else 0.0
        features[9] = np.sum(volumes) if len(volumes) > 0 else volume
        
        # Trend features
        if len(prices) >= 3:
            features[10] = (prices[-1] - prices[0]) / prices[0]  # Return
            features[11] = np.sign(features[10])  # Direction
        else:
            features[10] = 0.0
            features[11] = 0.0
        
        # Momentum features
        if len(prices) >= 2:
            features[12] = prices[-1] - prices[-2]  # Price change
        else:
            features[12] = 0.0
        
        # Volatility features
        if len(prices) >= 2:
            returns = np.diff(prices) / prices[:-1]
            features[13] = np.std(returns) if len(returns) > 1 else 0.0
        else:
            features[13] = 0.0
        
        # Time features
        features[14] = horizon  # Horizon index
        features[15] = timestamp  # Timestamp
        
        return features
    
    def _generate_vector(self, horizon: int) -> np.ndarray:
        """Generate 64-dim dense vector from horizon features"""
        buffer = self.buffers[horizon]
        
        if len(buffer) == 0:
            return np.random.randn(self.config.vector_dim).astype('float32')
        
        # Get all horizon features from buffer (16-dim features)
        features_list = []
        for _, horizon_features, _ in buffer:
            features_list.append(horizon_features)
        
        if len(features_list) == 0:
            return np.random.randn(self.config.vector_dim).astype('float32')
        
        # Convert to numpy array
        all_features = np.stack(features_list)  # [n_steps, 16]
        n_steps, n_features = all_features.shape
        
        # Generate vector via multiple aggregations
        vector = np.zeros(self.config.vector_dim, dtype='float32')
        
        # Block 1: First 16 dimensions - mean of horizon features
        mean_features = np.mean(all_features, axis=0)
        vector[:16] = mean_features
        
        # Block 2: Next 16 dimensions - std of horizon features
        if n_steps > 1:
            std_features = np.std(all_features, axis=0)
            vector[16:32] = std_features
        else:
            vector[16:32] = 0.0
        
        # Block 3: Next 16 dimensions - quantiles
        if n_steps >= 5:
            for i, q in enumerate([10, 50, 90]):
                q_features = np.percentile(all_features, q, axis=0)
                offset = 32 + i * 5  # Spread across 15 dims
                vector[offset:offset+5] = q_features[:5]
        
        # Block 4: Last 16 dimensions - additional statistics
        if n_steps >= 2:
            # Min/max (16 dims total)
            min_features = np.min(all_features, axis=0)
            max_features = np.max(all_features, axis=0)
            vector[48:56] = min_features[:8]  # 8 dims
            vector[56:64] = max_features[:8]  # 8 dims
        
        # Normalize vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector

class FeatureExtractor:
    """Pure numpy feature extractor - no pandas"""
    
    @staticmethod
    def compute_moving_average(data: np.ndarray, window: int) -> np.ndarray:
        """Compute moving average using numpy"""
        if len(data) < window:
            # Pad with zeros
            result = np.zeros_like(data)
            result[:len(data)] = data
            return result
        
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    @staticmethod
    def compute_ema(data: np.ndarray, span: int) -> np.ndarray:
        """Compute exponential moving average"""
        if len(data) == 0:
            return np.array([])
        
        alpha = 2 / (span + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    @staticmethod
    def compute_rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
        """Compute RSI using pure numpy"""
        if len(data) < period + 1:
            return np.full(len(data), 50.0)
        
        deltas = np.diff(data)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.zeros_like(data)
        avg_loss = np.zeros_like(data)
        
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])
        
        for i in range(period + 1, len(data)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period
        
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

# Example usage
if __name__ == "__main__":
    print("Testing HorizonFeatureBuffer...")
    
    # Create buffer
    config = HorizonBufferConfig()
    buffer = HorizonFeatureBuffer(config)
    
    # Generate synthetic tick data
    np.random.seed(42)
    n_ticks = 1000
    timestamps = np.arange(1000, 1000 + n_ticks)
    prices = 50000 + np.cumsum(np.random.randn(n_ticks) * 10)  # Random walk
    volumes = np.random.exponential(1000, n_ticks)
    
    # Add ticks
    start_time = time.time()
    buffer.add_batch(timestamps, prices, volumes)
    end_time = time.time()
    
    print(f"Processed {n_ticks} ticks in {end_time - start_time:.3f}s")
    
    # Get stats
    stats = buffer.get_buffer_stats()
    print(f"Buffer stats: {stats}")
    
    # Get vectors for horizon 0
    horizon_0_vectors = buffer.get_all_vectors_for_horizon(0)
    print(f"Horizon 0 has {len(horizon_0_vectors)} vectors")
    
    if len(horizon_0_vectors) > 0:
        ts, vec = horizon_0_vectors[-1]
        print(f"Latest vector for horizon 0: shape={vec.shape}, norm={np.linalg.norm(vec):.4f}")
    
    print("HorizonFeatureBuffer test completed successfully!")