"""
HRM Agent with Vector Cache Support
===================================

This agent pulls from the vector cache for "skewed" features.
Instead of hyperbolic memory, it uses simple vector lookup from the store.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field

try:
    from vector_store import VectorStore, VectorStoreConfig
    HAS_VECTOR_STORE = True
except ImportError:
    HAS_VECTOR_STORE = False
    print("[VectorCacheAgent] VectorStore not available")

@dataclass
class VectorCacheAgentConfig:
    """Configuration for vector cache agent"""
    vector_dim: int = 64
    n_horizons: int = 24
    vector_store_path: str = "data/vector_store"
    use_cosine_similarity: bool = True
    n_neighbors: int = 3  # Number of neighbors to look up
    skew_factor: float = 0.3  # How much to skew features using vectors

class VectorCacheAgent:
    """
    HRM agent that uses vector cache for skewed features
    
    Replaces hyperbolic memory with simple vector lookup.
    """
    
    def __init__(self, config: VectorCacheAgentConfig):
        self.config = config
        
        # Initialize vector store
        if HAS_VECTOR_STORE:
            vs_config = VectorStoreConfig(
                vector_dim=config.vector_dim,
                memmap_path=f"{config.vector_store_path}/vectors.dat"
            )
            self.vector_store = VectorStore(vs_config)
            # Try to load existing data
            self.vector_store.load()
        else:
            self.vector_store = None
        
        print(f"[VectorCacheAgent] Initialized with {config.n_horizons} horizons")
    
    def get_skewed_features(self, base_features: np.ndarray, horizon: int, 
                           timestamp: int) -> np.ndarray:
        """
        Get skewed features by looking up similar vectors in cache
        
        Args:
            base_features: Original features (16-dim or 64-dim)
            horizon: Horizon index (0-23)
            timestamp: Current timestamp
            
        Returns:
            Skewed features (same shape as base_features)
        """
        if self.vector_store is None or not HAS_VECTOR_STORE:
            # Fallback: return base features unchanged
            return base_features
        
        # Create query vector from base features
        if len(base_features.shape) == 1:
            query_vector = base_features
        else:
            query_vector = base_features.mean(axis=0)
        
        # Ensure query vector is 64-dim
        if len(query_vector) != self.config.vector_dim:
            padded = np.zeros(self.config.vector_dim, dtype='float32')
            n = min(len(query_vector), self.config.vector_dim)
            padded[:n] = query_vector[:n]
            query_vector = padded
        
        # Look up similar vectors
        if self.config.use_cosine_similarity:
            similar_vectors = self.vector_store.cosine_similarity(
                query_vector, k=self.config.n_neighbors
            )
        else:
            similar_vectors = self.vector_store.nearest_neighbor(
                query_vector, k=self.config.n_neighbors
            )
        
        if not similar_vectors:
            # No similar vectors found, return base features
            return base_features
        
        # Aggregate similar vectors
        similar_vecs_list = []
        for distance_or_sim, (vec_horizon, vec_timestamp) in similar_vectors:
            vec = self.vector_store.get_vector(vec_horizon, vec_timestamp)
            if vec is not None:
                similar_vecs_list.append(vec)
        
        if not similar_vecs_list:
            return base_features
        
        # Compute skew from similar vectors
        similar_mean = np.mean(similar_vecs_list, axis=0)
        
        # Blend base features with skewed features
        if self.config.use_cosine_similarity:
            # For cosine similarity, higher is better
            skew = self.config.skew_factor
        else:
            # For distance, closer is better - invert
            skew = self.config.skew_factor * 0.5
        
        # Apply skew
        skewed = base_features + skew * (similar_mean - base_features)
        
        # Ensure output shape matches input
        if len(base_features.shape) == 1:
            return skewed[:len(base_features)]
        else:
            # For 2D arrays, broadcast appropriately
            if len(skewed.shape) == 1 and len(base_features.shape) == 2:
                skewed = np.tile(skewed, (base_features.shape[0], 1))
            return skewed[:base_features.shape[0], :base_features.shape[1]]
    
    def update_cache(self, vector: np.ndarray, horizon: int, timestamp: int):
        """Update the vector cache with a new vector"""
        if self.vector_store is None or not HAS_VECTOR_STORE:
            return
        
        try:
            self.vector_store.add_vector(vector, horizon, timestamp)
        except Exception as e:
            print(f"[VectorCacheAgent] Error updating cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get vector cache statistics"""
        if self.vector_store is None or not HAS_VECTOR_STORE:
            return {"count": 0, "error": "VectorStore not available"}
        
        return self.vector_store.get_stats()
    
    def get_similar_vectors(self, query_vector: np.ndarray, k: int = 3) -> List[Tuple[float, Tuple[int, int]]]:
        """Get k similar vectors from cache"""
        if self.vector_store is None or not HAS_VECTOR_STORE:
            return []
        
        if self.config.use_cosine_similarity:
            return self.vector_store.cosine_similarity(query_vector, k=k)
        else:
            return self.vector_store.nearest_neighbor(query_vector, k=k)


# Example usage
if __name__ == "__main__":
    print("Testing VectorCacheAgent...")
    
    # Create agent
    config = VectorCacheAgentConfig(
        vector_dim=64,
        n_horizons=24,
        vector_store_path="data/vector_store",
        use_cosine_similarity=True,
        n_neighbors=3,
        skew_factor=0.3
    )
    
    agent = VectorCacheAgent(config)
    
    # Add some vectors to cache
    print("\nAdding vectors to cache...")
    for i in range(10):
        vector = np.random.randn(64).astype('float32')
        horizon = i % 24  # 24 horizons
        timestamp = 1000 + i
        agent.update_cache(vector, horizon, timestamp)
    
    print(f"Cache stats: {agent.get_cache_stats()}")
    
    # Test skewed features
    print("\nTesting skewed features...")
    base_features = np.random.randn(64).astype('float32')
    skewed = agent.get_skewed_features(base_features, horizon=0, timestamp=1100)
    
    print(f"Base features shape: {base_features.shape}")
    print(f"Skewed features shape: {skewed.shape}")
    print(f"Features changed: {np.linalg.norm(base_features - skewed):.4f}")
    
    # Test similar vectors lookup
    print("\nTesting similar vectors lookup...")
    query = np.random.randn(64).astype('float32')
    similar = agent.get_similar_vectors(query, k=5)
    print(f"Found {len(similar)} similar vectors")
    for dist, (horizon, ts) in similar[:3]:
        print(f"  - Horizon {horizon}, timestamp {ts}, distance/similarity: {dist:.4f}")
    
    print("\nVectorCacheAgent test completed!")