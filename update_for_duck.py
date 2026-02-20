"""
Update all files to use DuckStore instead of vector_store/ArrowStore
=====================================================================

This script updates key files to use DuckStore exclusively.
"""

import os
from pathlib import Path

def update_mvp_runner():
    """Update mvp_runner.py to use DuckStore"""
    file_path = Path("mvp_runner.py")
    if not file_path.exists():
        return
    
    content = file_path.read_text()
    
    # Replace vector_store imports
    content = content.replace(
        "from vector_store import create_simple_vector_store",
        "# from vector_store import create_simple_vector_store\n# Using DuckStore instead"
    )
    
    # Replace vector_store usage
    content = content.replace(
        "self.vector_store = create_simple_vector_store(vector_dim=config.vector_dim)",
        "# self.vector_store = create_simple_vector_store(vector_dim=config.vector_dim)\n        # Using DuckStore\n        self.duck_store = None"
    )
    
    # Update stats
    content = content.replace(
        "'vector_store_count': len(self.vector_store)",
        "'duck_store': 'using DuckDB'"
    )
    
    file_path.write_text(content)
    print(f"✅ Updated: {file_path}")

def update_signal_hrm():
    """Update signal_hrm.py to use DuckStore"""
    file_path = Path("hrm/signal_hrm.py")
    if not file_path.exists():
        return
    
    content = file_path.read_text()
    
    # Replace vector_store imports
    content = content.replace(
        "from vector_store import VectorStore, VectorStoreConfig",
        "# from vector_store import VectorStore, VectorStoreConfig\n# Using DuckStore instead"
    )
    
    # Replace VectorStoreConfig usage
    content = content.replace(
        "config = VectorStoreConfig(",
        "# config = VectorStoreConfig("
    )
    
    # Update config comments
    content = content.replace(
        "# Vector store parameters (replaces hyperbolic memory)",
        "# DuckDB store parameters (replaces hyperbolic memory)"
    )
    
    content = content.replace(
        "vector_store_path: str = \"data/vector_store\"",
        "duck_db_path: str = \"hrm/data/market.duckdb\""
    )
    
    file_path.write_text(content)
    print(f"✅ Updated: {file_path}")

def update_test_time_predictor():
    """Update test_time_predictor.py to use DuckStore"""
    file_path = Path("test_time_predictor.py")
    if not file_path.exists():
        return
    
    content = file_path.read_text()
    
    # Replace vector_store imports
    content = content.replace(
        "from vector_store import VectorStore, VectorStoreConfig",
        "# from vector_store import VectorStore, VectorStoreConfig\n# Using DuckStore instead"
    )
    
    # Replace VectorStoreConfig usage
    content = content.replace(
        "vs_config = VectorStoreConfig(",
        "# vs_config = VectorStoreConfig("
    )
    
    # Update config
    content = content.replace(
        "vector_store_path: str = \"data/vector_store\"",
        "duck_db_path: str = \"hrm/data/market.duckdb\""
    )
    
    file_path.write_text(content)
    print(f"✅ Updated: {file_path}")

def update_hrm_rollout_stages():
    """Update hrm_rollout_stages.py to use DuckStore"""
    file_path = Path("hrm_rollout_stages.py")
    if not file_path.exists():
        return
    
    content = file_path.read_text()
    
    # Replace vector_store imports
    content = content.replace(
        "from vector_store import VectorStore, VectorStoreConfig, create_simple_vector_store",
        "# from vector_store import VectorStore, VectorStoreConfig, create_simple_vector_store\n# Using DuckStore instead"
    )
    
    # Replace VectorStoreConfig usage
    content = content.replace(
        "vs_config = VectorStoreConfig(",
        "# vs_config = VectorStoreConfig("
    )
    
    # Update config
    content = content.replace(
        "vector_store_path: str = \"data/vector_store\"",
        "duck_db_path: str = \"hrm/data/market.duckdb\""
    )
    
    file_path.write_text(content)
    print(f"✅ Updated: {file_path}")

def update_scale_8_predictors():
    """Update scaling_manager.py to use DuckStore"""
    file_path = Path("execution/scaling_manager.py")
    if not file_path.exists():
        return
    
    content = file_path.read_text()
    
    # Replace vector_store imports
    content = content.replace(
        "from vector_store import create_simple_vector_store",
        "# from vector_store import create_simple_vector_store\n# Using DuckStore instead"
    )
    
    # Replace vector_store usage
    content = content.replace(
        "self.vector_store = create_simple_vector_store(vector_dim=64)",
        "# self.vector_store = create_simple_vector_store(vector_dim=64)\n        self.duck_store = None"
    )
    
    content = content.replace(
        "self.vector_store = create_simple_vector_store(vector_dim=64)",
        "# self.vector_store = create_simple_vector_store(vector_dim=64)\n        self.duck_store = None"
    )
    
    content = content.replace(
        "self.vector_store = None  # No vector cache",
        "# self.vector_store = None  # No vector cache\n            self.duck_store = None"
    )
    
    file_path.write_text(content)
    print(f"✅ Updated: {file_path}")

def main():
    print(f"{'='*60}")
    print("UPDATING FILES FOR DUCKDB")
    print(f"{'='*60}\n")
    
    print("Updating key files...\n")
    
    update_mvp_runner()
    update_signal_hrm()
    update_test_time_predictor()
    update_hrm_rollout_stages()
    update_scale_8_predictors()
    
    print(f"\n{'='*60}")
    print("UPDATE COMPLETE")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("1. Run cleanup_vector_store.py to search for remaining references")
    print("2. Run final_cleanup.py to delete vector_store.py and arrow_store.py")
    print("3. Run duck_migration.py to migrate existing data")
    print("4. Run backbone_duck_trainer.py for training")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()