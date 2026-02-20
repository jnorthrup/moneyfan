"""
MLX adaptation layer for HRM.

MLX-specific wrappers and inference for Apple Silicon.
"""
from .hrm_model import HRMModel, HRMConfig
from .trainer import HRMTrainer
from .inference import HRMInference
from .utils import enable_ane_optimization, setup_mlx_device

__all__ = [
    'HRMModel',
    'HRMConfig',
    'HRMTrainer',
    'HRMInference',
    'enable_ane_optimization',
    'setup_mlx_device'
]