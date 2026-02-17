from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class ScorecardEntry:
    timestamp: datetime
    symbol: str
    alpha: float = 0.0
    pnl: float = 0.0
    win_rate: float = 0.5
    sharpe: float = 0.0
    convergence: float = 0.0
    regime: str = "transition"
    signal: float = 0.0
    confidence: float = 0.0
    position_size: float = 0.0
    window: str = "decision"


@dataclass
class DecisionPayload:
    signal: float
    confidence: float
    regime: str
    metrics: Dict[str, float]
    timestamp: datetime


class AggregationWindow(Enum):
    DECISION = "decision"
    EPISODE = "episode"
    RUN = "run"


class Scorecard:
    def __init__(self):
        self.entries: List[ScorecardEntry] = []

    def add(self, entry: ScorecardEntry) -> None:
        self.entries.append(entry)

    def get_entries(self) -> List[ScorecardEntry]:
        return self.entries.copy()

    def filter(self, symbol: Optional[str] = None, regime: Optional[str] = None) -> "Scorecard":
        filtered = Scorecard()
        for entry in self.entries:
            if symbol is not None and entry.symbol != symbol:
                continue
            if regime is not None and entry.regime != regime:
                continue
            filtered.add(entry)
        return filtered

    def filter_by_symbol(self, symbol: str) -> "Scorecard":
        return self.filter(symbol=symbol)

    def filter_by_regime(self, regime: str) -> "Scorecard":
        return self.filter(regime=regime)

    def aggregate(self, window: AggregationWindow) -> Dict:
        if not self.entries:
            return {
                "mean_alpha": 0.0,
                "total_pnl": 0.0,
                "mean_sharpe": 0.0,
                "entry_count": 0,
                "regime_distribution": {},
            }

        alphas = [e.alpha for e in self.entries]
        pnls = [e.pnl for e in self.entries]
        sharpes = [e.sharpe for e in self.entries]

        regime_counts: Dict[str, int] = {}
        for entry in self.entries:
            regime_counts[entry.regime] = regime_counts.get(entry.regime, 0) + 1

        return {
            "mean_alpha": float(np.mean(alphas)),
            "total_pnl": float(sum(pnls)),
            "mean_sharpe": float(np.mean(sharpes)),
            "entry_count": len(self.entries),
            "regime_distribution": regime_counts,
        }

    def summary(self) -> Dict:
        return self.aggregate(AggregationWindow.DECISION)


def compute_alpha(returns: np.ndarray, benchmark: np.ndarray) -> float:
    if len(returns) != len(benchmark):
        raise ValueError("returns and benchmark must have the same length")
    if len(returns) == 0:
        return 0.0
    excess_returns = returns - benchmark
    return float(np.mean(excess_returns))


def compute_pnl(positions: np.ndarray, prices: np.ndarray) -> float:
    if len(positions) == 0 or len(prices) == 0:
        return 0.0
    pnl = 0.0
    for i in range(1, len(prices)):
        price_change = prices[i] - prices[i - 1]
        pnl += positions[i - 1] * price_change
    return float(pnl)


def compute_win_rate(trade_returns: np.ndarray) -> float:
    if len(trade_returns) == 0:
        return 0.0
    winners = np.sum(trade_returns > 0)
    return float(winners / len(trade_returns))


def compute_sharpe(returns: np.ndarray, risk_free: float = 0.0) -> float:
    if len(returns) == 0:
        return 0.0
    excess = returns - risk_free
    std = np.std(excess, ddof=0)
    if std == 0:
        return float("inf")
    return float(np.mean(excess) / std)


def compute_sortino(returns: np.ndarray, risk_free: float = 0.0) -> float:
    if len(returns) == 0:
        return 0.0
    excess = returns - risk_free
    downside_returns = excess[excess < 0]
    if len(downside_returns) == 0:
        return float("inf")
    downside_std = np.sqrt(np.mean(downside_returns ** 2))
    if downside_std == 0:
        return float("inf")
    return float(np.mean(excess) / downside_std)


def compute_max_drawdown(cumulative_returns: np.ndarray) -> float:
    if len(cumulative_returns) == 0:
        return 0.0
    first = cumulative_returns[0]
    if first <= 0:
        return 0.0
    min_val = np.min(cumulative_returns)
    return float((first - min_val) / first)


def compute_convergence(signals: np.ndarray, confidences: np.ndarray) -> float:
    if len(signals) == 0 or len(confidences) == 0:
        return 0.0
    weighted_signals = signals * confidences
    weighted_sum = np.sum(weighted_signals)
    total_weight = np.sum(np.abs(weighted_signals))
    if total_weight == 0:
        return 0.0
    return float(abs(weighted_sum) / total_weight)


def score_decision(payload: DecisionPayload, outcomes: Dict) -> float:
    if not outcomes:
        return 0.0

    alpha_weight = 0.4
    pnl_weight = 0.3
    win_rate_weight = 0.3

    alpha = outcomes.get("realized_alpha", payload.metrics.get("alpha", 0.0))
    pnl = outcomes.get("realized_pnl", payload.metrics.get("pnl", 0.0))
    win_rate = payload.metrics.get("win_rate", 0.5)

    max_pnl = max(abs(outcomes.get("realized_pnl", 1.0)), 1000.0)
    normalized_pnl = np.tanh(pnl / max_pnl)

    alpha_clamped = max(-1.0, min(1.0, alpha * 10))
    win_rate_normalized = win_rate

    base_score = (
        alpha_weight * (alpha_clamped + 1) / 2
        + pnl_weight * (normalized_pnl + 1) / 2
        + win_rate_weight * win_rate_normalized
    )

    confidence_multiplier = payload.confidence

    market_direction = outcomes.get("market_direction", 0)
    signal_aligned = (payload.signal * market_direction > 0) if market_direction != 0 else True
    regime_bonus = 0.1 if signal_aligned else 0.0

    score = base_score * confidence_multiplier + regime_bonus
    score = max(0.0, min(1.0, score))

    return score
