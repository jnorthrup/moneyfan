"""
Compatibility shim.

This module is kept only to avoid breaking older notebooks/scripts.
Active development lives in `fiduciary_controller.py`.
"""

from fiduciary_controller import (
    AllocationControllerConfig as JoystickConfig,
    FiduciaryControllerHRM as JoystickHRM,
    MarketFrame as Frame,
    MarketFrameBag as FrameBag,
    StatelessTrainer,
    CONTESTANT_MODELS,
    compute_raw_features,
    compute_model_signals,
    compute_holdings_gravity,
    create_fiduciary_hrm,
)


def create_joystick_hrm(n_instruments: int = 64, n_models: int = 12):
    return create_fiduciary_hrm(n_instruments=n_instruments, n_models=n_models)
