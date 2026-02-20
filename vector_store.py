"""
Simple vector store for horizon features - replaces hyperbolic memory
===============================================================

Simple numpy memmap + optional FAISS index for nearest-neighbor lookup.
Keys: (horizon + timestamp) → 64-dim dense vector.

Training: instrument/agent codecs → 64-dim dense vector per step → store
Test-time: predictor outputs tiny vector → nearest-neighbor lookup

10x simpler than hyperbolic ops, sub-millisecond lookup.
"""

import numpy as np
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("[vector_store] FAISS not available, using numpy-only mode")

@dataclass
class VectorStoreConfig:
    """Configuration for vector store"""
    vector_dim: int = 64
    memmap_path: str = "data/vectors.dat"
    index_path: str = "data/vectors.index"
    use_faiss: bool = False
    faiss_metric: int = faiss.METRIC_L2 if HAS_FAISS else None
    max_vectors: int = 1_000_000  # Maximum vectors to store
    chunk_size: int = 10000  # Chunk size for memmap operations

class VectorStore:
    """
    Simple vector store with numpy memmap + optional FAISS index
    
    Key design: (horizon + timestamp) → 64-dim dense vector
    - horizon: 24 integer values (0-23)
    - timestamp: integer timestamp (e.g., Unix seconds)
    - Combined key ensures uniqueness per horizon per step
    """
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.data_dir = Path(config.memmap_path).parent
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize memmap for vectors
        self.vectors_memmap = None
        self.timestamps_memmap = None
        self.horizons_memmap = None
        
        # Index for fast lookup
        self.index = None
        self.key_to_idx: Dict[Tuple[int, int], int] = {}  # (horizon, timestamp) → idx
        
        # Current count of vectors
        self.count = 0
        
        self._init_memmap()
        self._init_index()
    
    def _init_memmap(self):
        """Initialize memory-mapped arrays"""
        # Vectors: max_vectors × vector_dim
        vectors_path = self.config.memmap_path
        self.vectors_memmap = np.memmap(
            vectors_path,
            dtype='float32',
            mode='w+',
            shape=(self.config.max_vectors, self.config.vector_dim)
        )
        
        # Timestamps: max_vectors × 1
        timestamps_path = str(Path(self.config.memmap_path).parent / "timestamps.dat")
        self.timestamps_memmap = np.memmap(
            timestamps_path,
            dtype='int64',
            mode='w+',
            shape=(self.config.max_vectors,)
        )
        
        # Horizons: max_vectors × 1
        horizons_path = str(Path(self.config.memmap_path).parent / "horizons.dat")
        self.horizons_memmap = np.memmap(
            horizons_path,
            dtype='int16',
            mode='w+',
            shape=(self.config.max_vectors,)
        )
    
    def _init_index(self):
        """Initialize FAISS index if available"""
        if self.config.use_faiss and HAS_FAISS:
            try:
                self.index = faiss.IndexFlatL2(self.config.vector_dim)
                print(f"[vector_store] FAISS index initialized with metric L2")
            except Exception as e:
                print(f"[vector_store] Failed to init FAISS index: {e}")
                self.index = None
                self.config.use_faiss = False
        else:
            self.index = None
    
    def add_vector(self, vector: np.ndarray, horizon: int, timestamp: int) -> int:
        """
        Add a vector to the store
        
        Args:
            vector: 64-dim dense vector (can be 1D or 2D)
            horizon: horizon index (0-23)
            timestamp: integer timestamp
            
        Returns:
            Index where vector was stored
        """
        # Handle both 1D and 2D arrays
        if isinstance(vector, np.ndarray):
            if len(vector.shape) == 1:
                vector_dim = len(vector)
            elif len(vector.shape) == 2:
                vector_dim = vector.shape[-1]
                # Squeeze if it's (1, 64) or (64, 1)
                if vector.shape[0] == 1:
                    vector = vector[0]
                elif vector.shape[1] == 1:
                    vector = vector[:, 0]
            else:
                raise ValueError(f"Vector must be 1D or 2D, got shape {vector.shape}")
        else:
            vector_dim = len(vector)
        
        if vector_dim != self.config.vector_dim:
            raise ValueError(f"Vector dim {vector_dim} != config dim {self.config.vector_dim}")
        
        if self.count >= self.config.max_vectors:
            # Cycle around - overwrite oldest
            idx = self.count % self.config.max_vectors
        else:
            idx = self.count
            self.count += 1
        
        # Store in memmap
        self.vectors_memmap[idx] = vector.astype('float32')
        self.timestamps_memmap[idx] = timestamp
        self.horizons_memmap[idx] = horizon
        
        # Update key mapping
        key = (horizon, timestamp)
        self.key_to_idx[key] = idx
        
        # Update FAISS index if available
        if self.index is not None:
            # Add vector to FAISS index
            try:
                self.index.add(vector.astype('float32').reshape(1, -1))
            except Exception as e:
                print(f"[vector_store] FAISS add failed: {e}")
        
        return idx
    
    def add_batch(self, vectors: np.ndarray, horizons: List[int], timestamps: List[int]) -> List[int]:
        """
        Add multiple vectors at once
        
        Args:
            vectors: [n_vectors, vector_dim] array
            horizons: List of horizon indices
            timestamps: List of timestamps
            
        Returns:
            List of indices where vectors were stored
        """
        n_vectors = len(vectors)
        indices = []
        
        for i in range(n_vectors):
            idx = self.add_vector(vectors[i], horizons[i], timestamps[i])
            indices.append(idx)
        
        return indices
    
    def get_vector(self, horizon: int, timestamp: int) -> Optional[np.ndarray]:
        """Get vector by key"""
        key = (horizon, timestamp)
        if key not in self.key_to_idx:
            return None
        
        idx = self.key_to_idx[key]
        return self.vectors_memmap[idx].copy()
    
    def nearest_neighbor(self, query_vector: np.ndarray, k: int = 1) -> List[Tuple[float, Tuple[int, int]]]:
        """
        Find k nearest neighbors to query vector
        
        Args:
            query_vector: 64-dim query vector
            k: number of neighbors to return
            
        Returns:
            List of (distance, (horizon, timestamp)) tuples, sorted by distance
        """
        if self.count == 0:
            return []
        
        if self.index is not None and self.config.use_faiss:
            # Use FAISS for fast search
            try:
                distances, indices = self.index.search(
                    query_vector.astype('float32').reshape(1, -1), k
                )
                results = []
                for dist, idx in zip(distances[0], indices[0]):
                    if idx >= 0 and idx < self.count:
                        horizon = self.horizons_memmap[idx]
                        timestamp = self.timestamps_memmap[idx]
                        results.append((float(dist), (horizon, timestamp)))
                return results
            except Exception as e:
                print(f"[vector_store] FAISS search failed: {e}, falling back to numpy")
        
        # Fallback: numpy brute-force search
        if self.count == 0:
            return []
        
        # Get all stored vectors
        all_vectors = self.vectors_memmap[:self.count]
        
        # Compute distances
        distances = np.linalg.norm(all_vectors - query_vector, axis=1)
        
        # Get top-k indices
        top_k_indices = np.argsort(distances)[:k]
        
        results = []
        for idx in top_k_indices:
            horizon = self.horizons_memmap[idx]
            timestamp = self.timestamps_memmap[idx]
            results.append((float(distances[idx]), (horizon, timestamp)))
        
        return results
    
    def cosine_similarity(self, query_vector: np.ndarray, k: int = 1) -> List[Tuple[float, Tuple[int, int]]]:
        """
        Find k most similar vectors using cosine similarity
        
        Args:
            query_vector: 64-dim query vector
            k: number of neighbors to return
            
        Returns:
            List of (similarity, (horizon, timestamp)) tuples, sorted by similarity (descending)
        """
        if self.count == 0:
            return []
        
        # Get all stored vectors
        all_vectors = self.vectors_memmap[:self.count]
        
        # Compute cosine similarities
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return []
        
        similarities = []
        for i in range(self.count):
            vec_norm = np.linalg.norm(all_vectors[i])
            if vec_norm == 0:
                similarities.append((0.0, i))
            else:
                dot_product = np.dot(query_vector, all_vectors[i])
                cos_sim = dot_product / (query_norm * vec_norm)
                similarities.append((float(cos_sim), i))
        
        # Sort by similarity (descending)
        similarities.sort(reverse=True)
        
        # Get top-k
        results = []
        for sim, idx in similarities[:k]:
            horizon = self.horizons_memmap[idx]
            timestamp = self.timestamps_memmap[idx]
            results.append((sim, (horizon, timestamp)))
        
        return results
    
    def get_vectors_by_horizon(self, horizon: int) -> List[Tuple[np.ndarray, int]]:
        """Get all vectors for a specific horizon"""
        if self.count == 0:
            return []
        
        results = []
        for idx in range(self.count):
            if self.horizons_memmap[idx] == horizon:
                vector = self.vectors_memmap[idx].copy()
                timestamp = self.timestamps_memmap[idx]
                results.append((vector, timestamp))
        
        return results
    
    def save(self, path: Optional[str] = None):
        """Save the key mapping and metadata"""
        if path is None:
            path = str(Path(self.config.memmap_path).parent / "vector_store.pkl")
        
        metadata = {
            'count': self.count,
            'key_to_idx': self.key_to_idx,
            'config': self.config,
            'timestamp': time.time()
        }
        
        with open(path, 'wb') as f:
            pickle.dump(metadata, f)
        
        # Flush memmaps
        self.vectors_memmap.flush()
        self.timestamps_memmap.flush()
        self.horizons_memmap.flush()
        
        print(f"[vector_store] Saved to {path}")
    
    def load(self, path: Optional[str] = None):
        """Load the key mapping and metadata"""
        if path is None:
            path = str(Path(self.config.memmap_path).parent / "vector_store.pkl")
        
        if not os.path.exists(path):
            print(f"[vector_store] No saved data found at {path}")
            return
        
        with open(path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.count = metadata['count']
        self.key_to_idx = metadata['key_to_idx']
        
        print(f"[vector_store] Loaded {self.count} vectors from {path}")
    
    def __len__(self):
        return self.count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        if self.count == 0:
            return {"count": 0, "unique_horizons": 0, "unique_timestamps": 0}
        
        horizons = set(self.horizons_memmap[:self.count])
        timestamps = set(self.timestamps_memmap[:self.count])
        
        # Get vector statistics
        all_vectors = self.vectors_memmap[:self.count]
        norms = np.linalg.norm(all_vectors, axis=1)
        
        return {
            "count": self.count,
            "unique_horizons": len(horizons),
            "unique_timestamps": len(timestamps),
            "avg_norm": float(np.mean(norms)),
            "std_norm": float(np.std(norms)),
            "max_norm": float(np.max(norms)),
            "min_norm": float(np.min(norms)),
            "config": self.config
        }

def create_simple_vector_store(vector_dim: int = 64, use_faiss: bool = False) -> VectorStore:
    """Factory function to create a simple vector store"""
    config = VectorStoreConfig(
        vector_dim=vector_dim,
        use_faiss=use_faiss,
        memmap_path="data/vector_store/vectors.dat",
        index_path="data/vector_store/vectors.index"
    )
    return VectorStore(config)

# Example usage
if __name__ == "__main__":
    # Test the vector store
    print("Testing Vector Store...")
    
    # Create store
    store = create_simple_vector_store(vector_dim=64, use_faiss=False)
    
    # Add some random vectors
    np.random.seed(42)
    for i in range(100):
        vector = np.random.randn(64).astype('float32')
        horizon = i % 24  # 24 horizons
        timestamp = 1000 + i
        store.add_vector(vector, horizon, timestamp)
    
    print(f"Added {len(store)} vectors")
    
    # Query with a random vector
    query = np.random.randn(64).astype('float32')
    
    # Nearest neighbor search
    nn_results = store.nearest_neighbor(query, k=3)
    print(f"Nearest neighbors: {nn_results}")
    
    # Cosine similarity search
    cos_results = store.cosine_similarity(query, k=3)
    print(f"Cosine similarity: {cos_results}")
    
    # Get stats
    stats = store.get_stats()
    print(f"Store stats: {stats}")
    
    # Save
    store.save()
    print("Vector store saved successfully!")