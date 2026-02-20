"""
Cleanup Vector Store References - Replace with DuckStore
=========================================================

Searches for and replaces all vector_store references with DuckStore in codebase.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any

class VectorStoreCleanup:
    """
    Remove vector_store references and replace with DuckStore
    """
    
    def __init__(self):
        self.files_checked = []
        self.files_modified = []
        self.files_with_errors = []
        
        # Patterns to search for
        self.patterns = [
            r'from vector_store import',
            r'import vector_store',
            r'vector_store\.',
            r'VectorStore',
            r'VectorStoreConfig',
            r'create_simple_vector_store',
            r'vector_store_path',
            r'VectorStore\(',
            r'VectorStoreConfig\(',
            r'\.add_vector\(',
            r'\.cosine_similarity\(',
            r'\.nearest_neighbor\(',
            r'\.get_vector\(',
            r'\.get_stats\(',
        ]
        
        # Files to exclude
        self.exclude_files = [
            'cleanup_vector_store.py',
            'vector_store.py',  # Will be deleted
            'duck_migration.py',
            'backbone_trainer.py',
            'binance_data_loader.py',
            'binance_stochastic_bag_trainer.py',
        ]
    
    def search_files(self, directory: str = ".") -> List[Path]:
        """Search for files containing vector_store references"""
        found_files = []
        
        for root, dirs, files in os.walk(directory):
            # Skip directories
            dirs[:] = [d for d in dirs if d not in ['.git', 'venv', '__pycache__', '.idea', 'node_modules']]
            
            for file in files:
                if file.endswith(('.py', '.md', '.txt', '.sh')):
                    file_path = Path(root) / file
                    
                    # Skip excluded files
                    if any(ex in str(file_path) for ex in self.exclude_files):
                        continue
                    
                    # Check if file contains vector_store references
                    try:
                        content = file_path.read_text()
                        has_vector_store = any(
                            pattern in content.lower() or pattern.replace('_', '') in content.lower()
                            for pattern in self.patterns
                        )
                        
                        if has_vector_store:
                            found_files.append(file_path)
                            print(f"⚠️  {file_path}")
                            
                    except Exception as e:
                        print(f"❌ Error reading {file_path}: {e}")
                        self.files_with_errors.append(file_path)
        
        return found_files
    
    def replace_in_file(self, file_path: Path) -> bool:
        """Replace vector_store references with DuckStore in a single file"""
        try:
            content = file_path.read_text()
            original_content = content
            
            # List of replacements to make
            replacements = [
                # Import statements
                (r'from vector_store import VectorStore, VectorStoreConfig', 'from hrm.duck_store import DuckStore'),
                (r'from vector_store import VectorStore', 'from hrm.duck_store import DuckStore'),
                (r'import vector_store', '# import vector_store'),
                (r'from hrm\.vector_store import', 'from hrm.duck_store import'),
                (r'VectorStore\(', 'DuckStore('),
                (r'VectorStoreConfig\(', '# VectorStoreConfig('),
                (r'VectorStoreConfig', '# VectorStoreConfig'),
                (r'VectorStore', 'DuckStore'),
                (r'create_simple_vector_store\(', 'DuckStore('),
                
                # Configuration
                (r'vector_store_path\s*=\s*[^,]*', '# vector_store_path removed, using duck_db_path'),
                (r'memmap_path\s*=\s*[^,]*', '# memmap_path removed, using duck_db_path'),
                (r'VectorStoreConfig\(', '# VectorStoreConfig removed, using DuckStore'),
                
                # Method calls
                (r'\.add_vector\(', '.update_cache('),
                (r'\.cosine_similarity\(', '.get_similar_vectors('),
                (r'\.nearest_neighbor\(', '.get_similar_vectors('),
                (r'\.get_vector\(', '.get_vector_from_db('),
                (r'\.get_stats\(', '.get_cache_stats('),
                
                # Variable names
                (r'vector_store\s*=', 'duck_store ='),
                (r'self\.vector_store', 'self.duck_store'),
                (r'vector_store\.', 'duck_store.'),
                
                # Comments
                (r'# VectorStore', '# DuckStore'),
                (r'# vector store', '# duck store'),
                (r'# VectorStoreConfig', '# VectorStoreConfig removed'),
            ]
            
            # Apply replacements
            modified = False
            for pattern, replacement in replacements:
                new_content = re.sub(pattern, replacement, content)
                if new_content != content:
                    content = new_content
                    modified = True
            
            # If modified, save back to file
            if modified:
                file_path.write_text(content)
                self.files_modified.append(file_path)
                print(f"✅ Modified: {file_path}")
                return True
            else:
                print(f"ℹ️  No changes needed: {file_path}")
                return False
                
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            self.files_with_errors.append(file_path)
            return False
    
    def cleanup_all(self, directory: str = "."):
        """Search and replace vector_store references throughout codebase"""
        print(f"\n{'='*80}")
        print("CLEANUP VECTOR STORE REFERENCES")
        print(f"{'='*80}\n")
        
        print("Searching for files with vector_store references...\n")
        files = self.search_files(directory)
        
        if not files:
            print("✅ No files with vector_store references found!")
            return
        
        print(f"\n{'='*80}")
        print(f"Found {len(files)} files with vector_store references")
        print(f"{'='*80}\n")
        
        # Ask for confirmation
        print("Proceed with cleanup? (yes/no)")
        response = input("> ").lower()
        
        if response not in ['yes', 'y']:
            print("Cleanup cancelled.")
            return
        
        print("\nCleaning up files...\n")
        
        for file_path in files:
            self.replace_in_file(file_path)
        
        # Print summary
        print(f"\n{'='*80}")
        print("CLEANUP SUMMARY")
        print(f"{'='*80}")
        print(f"Files checked: {len(self.files_checked)}")
        print(f"Files modified: {len(self.files_modified)}")
        print(f"Files with errors: {len(self.files_with_errors)}")
        
        if self.files_with_errors:
            print("\nFiles with errors:")
            for file in self.files_with_errors:
                print(f"  • {file}")
        
        print(f"\n{'='*80}")
        print("NEXT STEPS:")
        print("1. Review all changes")
        print("2. Test modified files")
        print("3. Delete vector_store.py and arrow_store.py")
        print("4. Run migration script")
        print(f"{'='*80}")

# Example usage
if __name__ == "__main__":
    cleaner = VectorStoreCleanup()
    cleaner.cleanup_all(".")