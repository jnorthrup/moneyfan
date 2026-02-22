"""
Codec implementations for the 24 SOTA models
"""

import os
import glob
import importlib
from typing import List, Type
from .base_codec import BaseExpert

def load_all_codecs() -> List[Type[BaseExpert]]:
    """
    Dynamically loads all 24 codec experts from the codec_models directory.
    Guarantees canonical ordering according to GOALS.md.
    """
    codec_dir = os.path.dirname(__file__)
    codec_files = sorted(glob.glob(os.path.join(codec_dir, "codec_[0-2][0-9]_*.py")))
    
    codecs = []
    
    for file_path in codec_files:
        module_name = os.path.basename(file_path)[:-3]
        module = importlib.import_module(f".{module_name}", package="codec_models")
        
        # Find the class that inherits from BaseExpert
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, BaseExpert) and attr is not BaseExpert:
                codecs.append(attr)
                break
                
    return codecs

__all__ = ['BaseExpert', 'load_all_codecs']