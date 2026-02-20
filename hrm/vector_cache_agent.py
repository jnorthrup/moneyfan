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
    from hrm.duck_store import DuckStore
    HAS_DUCK_STORE = True
except ImportError:
    HAS_DUCK_STORE = False
    print("[VectorCacheAgent] DuckStore not available")

@dataclass
class VectorCacheAgentConfig:
    """Configuration for vector cache agent"""
    vector_dim: int = 64
    n_horizons: int = 24
    duck_db_path: str = "hrm/data/market.duckdb"
    use_cosine_similarity: bool = True
    n_neighbors: int = 3  # Number of neighbors to look up
    skew_factor: float = 0.3  # How much to skew features using vectors

class VectorCacheAgent:
    """
    HRM agent that uses DuckDB vector cache for skewed features
    
    Replaces hyperbolic memory with SQL-based vector lookup.
    """
    
    def __init__(self, config: VectorCacheAgentConfig):
        self.config = config
        
        # Initialize DuckDB store
        if HAS_DUCK_STORE:
            self.duck_store = DuckStore(config.duck_db_path)
            
            # Ensure vectors table exists
            self.duck_store.conn.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    id INTEGER PRIMARY KEY,
                    horizon INTEGER,
                    timestamp INTEGER,
                    vector DOUBLE[],
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        else:
            self.duck_store = None
        
        print(f"[VectorCacheAgent] Initialized with {config.n_horizons} horizons")
    
    def get_skewed_features(self, base_features: np.ndarray, horizon: int, 
                           timestamp: int) -> np.ndarray:
        """
        Get skewed features by looking up similar vectors in DuckDB
        
        Args:
            base_features: Original features (16-dim or 64-dim)
            horizon: Horizon index (0-23)
            timestamp: Current timestamp
            
        Returns:
            Skewed features (same shape as base_features)
        """
        if self.duck_store is None or not HAS_DUCK_STORE:
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
        
        # Look up similar vectors from DuckDB
        # Convert vector to SQL array format
        query_vector_str = "[" + ", ".join([str(x) for x in query_vector]) + "]"
        
        try:
            if self.config.use_cosine_similarity:
                # DuckDB doesn't have built-in cosine similarity
                # We'll use simple Euclidean distance
                query = f"""
                    SELECT horizon, timestamp, vector, 
                           array_distance(vector, {query_vector_str}) as distance
                    FROM vectors
                    WHERE horizon = ?
                    ORDER BY distance
                    LIMIT ?
                """
                results = self.duck_store.conn.execute(
                    query, (horizon, self.config.n_neighbors)
                ).fetchall()
            else:
                query = f"""
                    SELECT horizon, timestamp, vector,
                           array_distance(vector, {query_vector_str}) as distance
                    FROM vectors
                    WHERE horizon = ?
                    ORDER BY distance
                    LIMIT ?
                """
                results = self.duck_store.conn.execute(
                    query, (horizon, self.config.n_neighbors)
                ).fetchall()
            
            if not results:
                # No similar vectors found, return base features
                return base_features
            
            # Aggregate similar vectors
            similar_vecs_list = []
            for row in results:
                horizon_val, ts_val, vec_str, distance = row
                # Parse vector string back to array
                try:
                    vec = np.array(json.loads(vec_str), dtype='float32')
                    similar_vecs_list.append(vec)
                except:
                    continue
            
            if not similar_vecs_list:
                return base_features
        
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
        """Update the vector cache with a new vector in DuckDB"""
        if self.duck_store is None or not HAS_DUCK_STORE:
            return
        
        try:
            # Convert vector to SQL array format
            vector_str = "[" + ", ".join([str(x) for x in vector]) + "]"
            
            # Insert into DuckDB
            self.duck_store.conn.execute(
                "INSERT INTO vectors (horizon, timestamp, vector) VALUES (?, ?, ?)",
                (int(horizon), int(timestamp), vector_str)
            )
        except Exception as e:
            print(f"[VectorCacheAgent] Error updating cache: {e}")
        
        # Vector already inserted in DuckDB above
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get vector cache statistics from DuckDB"""
        if self.duck_store is None or not HAS_DUCK_STORE:
            return {"count": 0, "error": "DuckStore not available"}
        
        try:
            result = self.duck_store.conn.execute("SELECT COUNT(*) as count FROM vectors").fetchone()
            count = result[0] if result else 0
            
            return {
                "count": count,
                "table": "vectors",
                "database": self.duck_store.db_path
            }
        except Exception as e:
            return {"count": 0, "error": str(e)}
    
    def get_similar_vectors(self, query_vector: np.ndarray, k: int = 3) -> List[Tuple[float, Tuple[int, int]]]:
        """Get k similar vectors from DuckDB cache"""
        if self.duck_store is None or not HAS_DUCK_STORE:
            return []
        
        try:
            # Convert vector to SQL array format
            query_vector_str = "[" + ", ".join([str(x) for x in query_vector]) + "]"
            
            # Query DuckDB for similar vectors
            query = f"""
                SELECT horizon, timestamp, array_distance(vector, {query_vector_str}) as distance
                FROM vectors
                ORDER BY distance
                LIMIT ?
            """
            results = self.duck_store.conn.execute(query, (k,)).fetchall()
            
            return [(float(distance), (int(horizon), int(timestamp))) for horizon, timestamp, distance in results]
        except Exception as e:
            print(f"[VectorCacheAgent] Error querying similar vectors: {e}")
            return []


# Example usage
if __name__ == "__main__":
    print("Testing VectorCacheAgent...")
    
    # Create agent
    config = VectorCacheAgentConfig(
        vector_dim=64,
        n_horizons=24,
        duck_db_path="hrm/data/market.duckdb",
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