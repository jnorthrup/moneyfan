"""Configuration for HRM trading system"""
from .assets import (
    TRADE_PAIRS,
    ASSET_NAMES, 
    COINBASE_PAIRS,
    SECTORS,
    SECTOR_RISK_LIMITS,
    get_asset_index,
    get_sector_for_index,
    get_risk_limit,
)

__all__ = [
    "TRADE_PAIRS",
    "ASSET_NAMES",
    "COINBASE_PAIRS", 
    "SECTORS",
    "SECTOR_RISK_LIMITS",
    "get_asset_index",
    "get_sector_for_index",
    "get_risk_limit",
]
