from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RiskTier(str, Enum):
    NORMAL = "normal"
    CAUTION = "caution"
    PROTECTIVE = "protective"


@dataclass(frozen=True)
class VetoDecision:
    vetoed: bool
    reason: Optional[str] = None
    risk_tier: RiskTier = RiskTier.NORMAL


@dataclass(frozen=True)
class NormalizedTradeIntent:
    """
    Broker-agnostic trade output shape produced from HRM trade heads.

    This is intentionally independent of Coinbase/other broker payload fields.
    Lane D will map this schema into broker-specific order requests.
    """
    symbol: str
    direction: float
    pred_fwd_return: float
    confidence: float
    position_fraction: float
    stop_loss_pct: float
    take_profit_pct: float
    vetoed: bool = False
    veto_reason: Optional[str] = None
    risk_tier: RiskTier = RiskTier.NORMAL

