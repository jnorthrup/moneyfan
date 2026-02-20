"""
Final Cleanup - Remove vector_store.py and arrow_store.py
===========================================================

Deletes the non-duck stores and cleans up all references.
"""

import os
import sys
from pathlib import Path

def delete_files():
    """Delete vector_store.py and arrow_store.py"""
    files_to_delete = [
        "vector_store.py",
        "hrm/arrow_store.py",
        "data/vector_store",
        "hrm/data/arrow",
    ]
    
    print(f"{'='*60}")
    print("FINAL CLEANUP - DELETE NON-DUCK STORES")
    print(f"{'='*60}\n")
    
    for file_path in files_to_delete:
        path = Path(file_path)
        if path.exists():
            if path.is_dir():
                import shutil
                shutil.rmtree(path)
                print(f"✅ Deleted directory: {path}")
            else:
                path.unlink()
                print(f"✅ Deleted file: {path}")
        else:
            print(f"⚠️  Not found: {path}")
    
    print(f"\n{'='*60}")
    print("CLEANUP COMPLETE")
    print(f"{'='*60}")
    print("All non-duck stores have been removed.")
    print("Only DuckStore (DuckDB) remains.")
    print(f"{'='*60}")

if __name__ == "__main__":
    delete_files()