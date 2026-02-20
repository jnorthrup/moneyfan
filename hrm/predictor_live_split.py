"""
Enforce strict predictor vs live-agents split
==============================================

This module enforces the architectural split that separates:
1. PREDICTOR PATH: Pure numpy + MLX inference only (no pandas)
2. LIVE AGENTS PATH: Full pandas DataFrames + depth snapshots

The predictor path is the inference engine that must be:
- Deterministic
- Sub-millisecond latency
- No pandas imports allowed
- Pure MLX + numpy operations
- No training, only inference

The live agents path can have:
- Lazy-loaded pandas DataFrames
- Full depth snapshots
- Training capabilities
- Complex data structures

This enforces production HFT practices where:
- Inference path is stripped-down and deterministic
- Training/analysis path is rich
- Saves RAM and keeps latency <50 ms
"""

import sys
import importlib
from typing import Dict, Any
import inspect

# List of pandas-dependent modules that should NOT be in predictor path
PANDAS_DEPENDENT_MODULES = [
    'pandas',
    'pd',
    'pandas.core',
    'pandas.io',
    'pandas.plotting',
    'pandas.util',
    'pandas._libs',
    'pyarrow',  # Often used with pandas
    'polars',   # Alternative to pandas
    'dask',     # Often used with pandas
    'xarray',   # Often used with pandas
]

# List of allowed modules for predictor path
ALLOWED_MODULES = [
    'numpy',
    'np',
    'mlx',
    'mx',
    'mlx.core',
    'mlx.nn',
    'mlx.optimizers',
    'array',
    'collections',
    'deque',
    'pathlib',
    'typing',
    'dataclasses',
    'time',
    'math',
    'random',
    'json',
    'pickle',
    'os',
    'sys',
]

class PredictorLiveSplit:
    """
    Enforces the predictor vs live-agents architectural split
    
    Usage:
        # In predictor modules:
        predictor_split = PredictorLiveSplit()
        predictor_split.assert_predictor_mode()
        
        # In live-agents modules:
        predictor_split.assert_live_agent_mode()
    """
    
    def __init__(self, mode: str = "predictor"):
        """
        Initialize with mode: "predictor" or "live_agent"
        """
        self.mode = mode
        self.checkpoint_modules = set()
    
    def assert_predictor_mode(self, check_pandas: bool = True):
        """
        Assert that we are in predictor mode (no pandas allowed)
        
        Args:
            check_pandas: Whether to check for pandas imports
        """
        if self.mode != "predictor":
            raise RuntimeError(
                f"Predictor mode required, but module is in {self.mode} mode. "
                f"Predictor modules should only use pure numpy + MLX inference."
            )
        
        if check_pandas:
            self._check_for_pandas()
    
    def assert_live_agent_mode(self):
        """Assert that we are in live-agent mode (pandas allowed)"""
        if self.mode != "live_agent":
            raise RuntimeError(
                f"Live agent mode required, but module is in {self.mode} mode. "
                f"Live agent modules can use pandas and full data structures."
            )
    
    def _check_for_pandas(self):
        """Check if any pandas-dependent modules are imported"""
        imported_modules = set(sys.modules.keys())
        
        for module in imported_modules:
            for pandas_module in PANDAS_DEPENDENT_MODULES:
                if module.startswith(pandas_module):
                    raise ImportError(
                        f"Pandas dependency detected in predictor mode: {module}\n"
                        f"Predictor modules must be pure numpy + MLX only.\n"
                        f"Move pandas operations to live-agent path."
                    )
    
    def wrap_module(self, module_name: str, mode: str = "predictor"):
        """
        Wrap a module to enforce mode restrictions
        
        Args:
            module_name: Name of the module to wrap
            mode: "predictor" or "live_agent"
        """
        self.mode = mode
        
        # Get the module
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            print(f"[PredictorLiveSplit] Could not import module: {module_name}")
            return
        
        # Store original __import__ and __builtins__ to restore later
        original_import = __builtins__.__import__
        original_getattr = getattr
        
        def restricted_import(name, *args, **kwargs):
            """Custom import function that restricts pandas in predictor mode"""
            if self.mode == "predictor":
                for pandas_module in PANDAS_DEPENDENT_MODULES:
                    if name.startswith(pandas_module):
                        raise ImportError(
                            f"Pandas dependency '{name}' not allowed in predictor mode.\n"
                            f"Use pure numpy + MLX instead."
                        )
            return original_import(name, *args, **kwargs)
        
        def restricted_getattr(obj, name, *args, **kwargs):
            """Custom getattr that checks for pandas-like objects"""
            if self.mode == "predictor":
                # Check if accessing pandas-like attributes
                obj_name = str(type(obj).__module__) + "." + str(type(obj).__name__)
                for pandas_module in PANDAS_DEPENDENT_MODULES:
                    if pandas_module in obj_name:
                        raise AttributeError(
                            f"Pandas dependency '{name}' not allowed in predictor mode.\n"
                            f"Object type: {obj_name}"
                        )
            return original_getattr(obj, name, *args, **kwargs)
        
        # Replace import and getattr in the module's namespace
        module.__builtins__.__import__ = restricted_import
        module.__builtins__.getattr = restricted_getattr
        
        # Store checkpoint
        self.checkpoint_modules.add(module_name)
        
        return module
    
    def create_predictor_module(self, name: str):
        """
        Create a new predictor module with enforced restrictions
        
        Returns:
            A module-like object that enforces predictor restrictions
        """
        import types
        
        # Create a new module
        module = types.ModuleType(name)
        module.__file__ = f"<predictor_module_{name}>"
        
        # Add restriction methods
        module.assert_predictor_mode = self.assert_predictor_mode
        module.assert_live_agent_mode = self.assert_live_agent_mode
        
        # Replace import in module's globals
        original_import = __builtins__.__import__
        
        def predictor_import(name, *args, **kwargs):
            if name.startswith(tuple(PANDAS_DEPENDENT_MODULES)):
                raise ImportError(
                    f"Pandas dependency '{name}' not allowed in predictor module '{name}'.\n"
                    f"Predictor modules must use pure numpy + MLX only."
                )
            return original_import(name, *args, **kwargs)
        
        module.__builtins__ = {'__import__': predictor_import}
        
        return module
    
    def create_live_agent_module(self, name: str):
        """
        Create a new live-agent module (pandas allowed)
        
        Returns:
            A module-like object that allows pandas
        """
        import types
        
        # Create a new module
        module = types.ModuleType(name)
        module.__file__ = f"<live_agent_module_{name}>"
        
        # Add restriction methods
        module.assert_predictor_mode = self.assert_predictor_mode
        module.assert_live_agent_mode = self.assert_live_agent_mode
        
        return module


# Global instance for the current session
_predictor_split = PredictorLiveSplit()

def set_predictor_mode():
    """Set current session to predictor mode (no pandas)"""
    global _predictor_split
    _predictor_split.mode = "predictor"

def set_live_agent_mode():
    """Set current session to live-agent mode (pandas allowed)"""
    global _predictor_split
    _predictor_split.mode = "live_agent"

def assert_predictor_mode():
    """Assert current session is in predictor mode"""
    global _predictor_split
    _predictor_split.assert_predictor_mode()

def assert_live_agent_mode():
    """Assert current session is in live-agent mode"""
    global _predictor_split
    _predictor_split.assert_live_agent_mode()


# Decorators for function-level enforcement
def predictor_only(func):
    """
    Decorator: Function only allowed in predictor mode
    """
    def wrapper(*args, **kwargs):
        assert_predictor_mode()
        return func(*args, **kwargs)
    return wrapper


def live_agent_only(func):
    """
    Decorator: Function only allowed in live-agent mode
    """
    def wrapper(*args, **kwargs):
        assert_live_agent_mode()
        return func(*args, **kwargs)
    return wrapper


# Test utilities
def test_predictor_restrictions():
    """Test that predictor restrictions work"""
    print("Testing predictor restrictions...")
    
    # Should work
    try:
        import numpy as np
        print("✓ numpy import allowed")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
    
    # Should fail if pandas is installed
    try:
        import pandas as pd
        print("✗ pandas import should have been blocked")
    except ImportError:
        print("✓ pandas import correctly blocked")
    
    print("Predictor restriction test completed")


def test_live_agent_restrictions():
    """Test that live-agent mode allows pandas"""
    set_live_agent_mode()
    
    # Should work
    try:
        import numpy as np
        print("✓ numpy import allowed in live-agent mode")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
    
    # Should work in live-agent mode
    try:
        import pandas as pd
        print("✓ pandas import allowed in live-agent mode")
    except ImportError as e:
        print(f"pandas not available: {e}")
    
    set_predictor_mode()
    print("Live-agent restriction test completed")


# Example usage
if __name__ == "__main__":
    print("Testing Predictor vs Live-Agent Split...")
    
    # Test predictor mode
    print("\n1. Testing predictor mode:")
    set_predictor_mode()
    test_predictor_restrictions()
    
    # Test live-agent mode
    print("\n2. Testing live-agent mode:")
    test_live_agent_restrictions()
    
    # Test decorator usage
    print("\n3. Testing decorators:")
    
    @predictor_only
    def predictor_function():
        return "This is a predictor function"
    
    @live_agent_only
    def live_agent_function():
        return "This is a live-agent function"
    
    try:
        result = predictor_function()
        print(f"✓ predictor_function() works: {result}")
    except RuntimeError as e:
        print(f"✗ predictor_function() failed: {e}")
    
    try:
        result = live_agent_function()
        print(f"✗ live_agent_function() should have failed in predictor mode")
    except RuntimeError as e:
        print(f"✓ live_agent_function() correctly blocked: {e}")
    
    print("\nPredictor vs Live-Agent Split test completed!")