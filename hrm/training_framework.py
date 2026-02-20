"""
Pragmatic HRM training framework.

Canonical pipeline under evaluation:
    bots + candles -> HRM -> trade control

This module stays orthogonal to runtime execution code. It provides:
1) Stochastic expanse sampling with normalized "screws"
2) Weighted objective scoring for candidate-vs-baseline
3) Failfast gate evaluation
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from .currency_graph import CurrencyGraph
except ImportError:  # pragma: no cover - direct module import path
    from currency_graph import CurrencyGraph


CANONICAL_PIPELINE = "bots+candles -> HRM -> trade control"

# 24 baseline competitors (pipeline catalog minus hrm_mean_reversion).
DEFAULT_COMPETITOR_MODELS_24: List[str] = [
    "macd_crossover",
    "sota_momentum",
    "momentum_trend",
    "sector_rotation",
    "rsi_mean_reversion",
    "bollinger_reversion",
    "grid_reversion",
    "harvest_rebalance",
    "kilo_rebalance",
    "volatility_breakout",
    "bollinger_vol_regime",
    "vol_inverse_sizing",
    "bent_penny",
    "pairs_spread",
    "dca_baseline",
    "weekly_cadence",
    "technical_ml",
    "grid_x_trend",
    "rsi_x_trend",
    "momentum_x_vol",
    "vol_x_breakout_proven",
    "mom_trend_additive",
    "rsi_trend_additive",
    "macd_momentum_dual",
]


def _validate_portion(name: str, value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return float(value)


def _validate_norm_metric(name: str, value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be normalized to [0, 1], got {value}")
    return float(value)


@dataclass(frozen=True)
class TrainingScrews:
    """High-leverage knobs for small stochastic benchmark runs."""

    history_depth_portion: float = 1.0
    coinbase_slot_portion: float = 1.0
    random_seed: int = 42

    def validated(self) -> "TrainingScrews":
        return TrainingScrews(
            history_depth_portion=_validate_portion(
                "history_depth_portion", self.history_depth_portion
            ),
            coinbase_slot_portion=_validate_portion(
                "coinbase_slot_portion", self.coinbase_slot_portion
            ),
            random_seed=int(self.random_seed),
        )


@dataclass(frozen=True)
class ObjectiveWeights:
    """Weighted objective with flow capture support."""

    pnl: float = 0.5
    flow_capture: float = 0.0
    convergence: float = 0.3
    stability: float = 0.2

    @classmethod
    def pnl_first(cls) -> "ObjectiveWeights":
        """
        Practical default for training:
        - maximize PnL first
        - require meaningful liquidity-flow capture
        - keep smaller pressure for convergence/stability
        """
        return cls(pnl=0.70, flow_capture=0.20, convergence=0.07, stability=0.03)

    def normalized(self) -> Tuple[float, float, float, float]:
        values = np.array(
            [self.pnl, self.flow_capture, self.convergence, self.stability],
            dtype=np.float64,
        )
        if np.any(values < 0):
            raise ValueError(f"Objective weights must be non-negative, got {values.tolist()}")
        total = float(values.sum())
        if total <= 0.0:
            raise ValueError("Objective weights sum must be > 0")
        values = values / total
        return float(values[0]), float(values[1]), float(values[2]), float(values[3])


@dataclass(frozen=True)
class FailFastThresholds:
    """Guardrails that invalidate a run regardless of objective score."""

    min_rows_per_symbol: int = 200
    max_nan_ratio: float = 0.0
    require_monotonic_timestamps: bool = True
    max_gradient_norm: float = 5.0
    max_loss: float = 10.0
    max_epoch_seconds: float = 300.0
    max_exposure: float = 1.0
    max_turnover: float = 0.35
    max_concentration: float = 0.30
    min_confidence: float = 0.25
    min_convergence: float = 0.20
    min_flow_capture: float = 0.10


@dataclass(frozen=True)
class CandidateMetrics:
    pnl_norm: float
    convergence: float
    stability: float
    flow_capture: float = 0.0

    def validated(self) -> "CandidateMetrics":
        return CandidateMetrics(
            pnl_norm=_validate_norm_metric("pnl_norm", self.pnl_norm),
            convergence=_validate_norm_metric("convergence", self.convergence),
            stability=_validate_norm_metric("stability", self.stability),
            flow_capture=_validate_norm_metric("flow_capture", self.flow_capture),
        )


@dataclass(frozen=True)
class ScoredCandidate:
    pnl_norm: float
    flow_capture: float
    convergence: float
    stability: float
    objective: float


@dataclass(frozen=True)
class FailFastResult:
    ok: bool
    reasons: Tuple[str, ...] = ()


@dataclass(frozen=True)
class BenchmarkVerdict:
    accepted: bool
    objective_delta: float
    baseline: ScoredCandidate
    candidate: ScoredCandidate
    failfast: FailFastResult
    screws: TrainingScrews
    objective_weights: ObjectiveWeights
    competitors: Tuple[str, ...]
    pipeline: str = CANONICAL_PIPELINE

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["failfast"]["reasons"] = list(self.failfast.reasons)
        d["competitors"] = list(self.competitors)
        return d


@dataclass(frozen=True)
class FractalVerdict:
    """
    Aggregate result across many expanse/scenario evaluations.
    """

    scenario_count: int
    wins: int
    win_rate: float
    failfast_pass_rate: float
    worst_delta: float
    median_delta: float
    fractal_score: float
    is_fractal_winner: bool

    def to_dict(self) -> Dict:
        return asdict(self)


def compute_weighted_objective(
    metrics: CandidateMetrics, weights: ObjectiveWeights
) -> ScoredCandidate:
    m = metrics.validated()
    w_pnl, w_flow, w_conv, w_stability = weights.normalized()
    objective = (
        m.pnl_norm * w_pnl
        + m.flow_capture * w_flow
        + m.convergence * w_conv
        + m.stability * w_stability
    )
    return ScoredCandidate(
        pnl_norm=m.pnl_norm,
        flow_capture=m.flow_capture,
        convergence=m.convergence,
        stability=m.stability,
        objective=float(objective),
    )


def _split_pair_symbol(symbol: str) -> Optional[Tuple[str, str]]:
    if not symbol or "-" not in symbol:
        return None
    parts = symbol.split("-", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return parts[0], parts[1]


def build_coinpair_graph(
    pair_stats_df: pd.DataFrame,
    *,
    base_col: str = "base",
    quote_col: str = "quote",
    symbol_col: str = "symbol",
    volume_col: str = "volume_24h",
    spread_col: str = "spread",
) -> CurrencyGraph:
    """
    Build a currency graph from pair stats.

    Accepts either:
    - explicit base/quote columns, or
    - symbol column formatted like BASE-QUOTE (e.g. BTC-USD).
    """
    graph = CurrencyGraph()
    if pair_stats_df.empty:
        return graph

    for _, row in pair_stats_df.iterrows():
        base = row.get(base_col)
        quote = row.get(quote_col)

        if (not isinstance(base, str) or not base) or (not isinstance(quote, str) or not quote):
            symbol = row.get(symbol_col)
            parsed = _split_pair_symbol(str(symbol)) if symbol is not None else None
            if parsed is None:
                continue
            base, quote = parsed

        volume = float(row.get(volume_col, 0.0) or 0.0)
        spread = float(row.get(spread_col, 0.0) or 0.0)
        graph.add_pair(base=str(base), quote=str(quote), volume_24h=volume, spread=spread)

    return graph


def compute_liquidity_flow_capture(
    *,
    pair_stats_df: pd.DataFrame,
    trade_control: Mapping[str, float],
) -> float:
    """
    Score how well trade control routes across high-liquidity, high-depth pair flows.

    Returns a normalized [0,1] score where 1 is best.
    """
    graph = build_coinpair_graph(pair_stats_df)
    if not graph.pairs:
        return 0.0

    depth = graph.get_depth_metrics()
    max_volume = max((p.volume_24h for p in graph.pairs), default=0.0)
    max_spread = max((p.spread for p in graph.pairs), default=0.0)

    total_weight = 0.0
    weighted_score = 0.0

    for pair in graph.pairs:
        symbol = pair.symbol
        reverse_symbol = f"{pair.quote.symbol}-{pair.base.symbol}"
        intensity = abs(float(trade_control.get(symbol, 0.0))) + abs(
            float(trade_control.get(reverse_symbol, 0.0))
        )

        depth_score = (
            depth[pair.base].depth_score + depth[pair.quote].depth_score
        ) / 2.0

        if max_volume > 0.0:
            volume_norm = min(1.0, pair.volume_24h / max_volume)
        else:
            volume_norm = 0.0

        if max_spread > 0.0:
            spread_penalty = min(1.0, pair.spread / max_spread)
        else:
            spread_penalty = 0.0

        liquidity_quality = (0.6 * volume_norm + 0.4 * depth_score) * (1.0 - spread_penalty)
        liquidity_quality = float(np.clip(liquidity_quality, 0.0, 1.0))

        total_weight += intensity
        weighted_score += intensity * liquidity_quality

    # No routing signal supplied: fall back to graph-wide average quality.
    if total_weight <= 0.0:
        quality_scores = []
        for pair in graph.pairs:
            depth_score = (depth[pair.base].depth_score + depth[pair.quote].depth_score) / 2.0
            if max_volume > 0.0:
                volume_norm = min(1.0, pair.volume_24h / max_volume)
            else:
                volume_norm = 0.0
            if max_spread > 0.0:
                spread_penalty = min(1.0, pair.spread / max_spread)
            else:
                spread_penalty = 0.0
            quality_scores.append((0.6 * volume_norm + 0.4 * depth_score) * (1.0 - spread_penalty))
        if not quality_scores:
            return 0.0
        return float(np.clip(np.mean(quality_scores), 0.0, 1.0))

    return float(np.clip(weighted_score / total_weight, 0.0, 1.0))


def build_fractal_screws_grid(
    *,
    history_depths: Sequence[float] = (0.20, 0.40, 0.60, 0.80, 1.00),
    slot_portions: Sequence[float] = (0.25, 0.50, 0.75, 1.00),
    seeds: Sequence[int] = (11, 23, 37),
) -> Tuple[TrainingScrews, ...]:
    """
    Produce a deterministic expanse/scenario grid for fractal evaluation.
    """
    screws: List[TrainingScrews] = []
    for d in history_depths:
        _validate_portion("history_depth_portion", float(d))
        for s in slot_portions:
            _validate_portion("coinbase_slot_portion", float(s))
            for seed in seeds:
                screws.append(
                    TrainingScrews(
                        history_depth_portion=float(d),
                        coinbase_slot_portion=float(s),
                        random_seed=int(seed),
                    )
                )
    return tuple(screws)


def judge_fractal_winner(verdicts: Sequence[BenchmarkVerdict]) -> FractalVerdict:
    """
    Determine whether candidate is a fractal winner across all scenarios.

    Strict winner condition:
    - every scenario passes failfast
    - objective delta > 0 in every scenario
    """
    if not verdicts:
        return FractalVerdict(
            scenario_count=0,
            wins=0,
            win_rate=0.0,
            failfast_pass_rate=0.0,
            worst_delta=0.0,
            median_delta=0.0,
            fractal_score=0.0,
            is_fractal_winner=False,
        )

    deltas = np.array([v.objective_delta for v in verdicts], dtype=np.float64)
    failfast_flags = np.array([1.0 if v.failfast.ok else 0.0 for v in verdicts], dtype=np.float64)
    wins = np.array([1.0 if (v.failfast.ok and v.objective_delta > 0.0) else 0.0 for v in verdicts], dtype=np.float64)

    scenario_count = int(len(verdicts))
    wins_count = int(wins.sum())
    win_rate = float(wins.mean())
    failfast_pass_rate = float(failfast_flags.mean())
    worst_delta = float(deltas.min())
    median_delta = float(np.median(deltas))

    # Fractal score favors broad wins and robust worst-case margin.
    worst_margin_norm = float(np.clip((worst_delta + 1.0) / 2.0, 0.0, 1.0))
    fractal_score = float(np.clip(win_rate * failfast_pass_rate * worst_margin_norm, 0.0, 1.0))
    is_fractal_winner = bool(wins_count == scenario_count and failfast_pass_rate == 1.0)

    return FractalVerdict(
        scenario_count=scenario_count,
        wins=wins_count,
        win_rate=win_rate,
        failfast_pass_rate=failfast_pass_rate,
        worst_delta=worst_delta,
        median_delta=median_delta,
        fractal_score=fractal_score,
        is_fractal_winner=is_fractal_winner,
    )


def build_data_failfast_metrics(
    candles_df: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    timestamp_col: str = "timestamp",
) -> Dict[str, float]:
    """
    Build common failfast inputs from a candles expanse.
    """
    if candles_df.empty:
        return {
            "rows_per_symbol_min": 0.0,
            "nan_ratio": 1.0,
            "timestamp_monotonic": 0.0,
        }

    metrics: Dict[str, float] = {}

    if symbol_col in candles_df.columns:
        counts = candles_df.groupby(symbol_col).size()
        metrics["rows_per_symbol_min"] = float(counts.min())
    else:
        metrics["rows_per_symbol_min"] = float(len(candles_df))

    total_cells = int(candles_df.shape[0] * candles_df.shape[1])
    nan_cells = int(candles_df.isna().sum().sum())
    metrics["nan_ratio"] = float(nan_cells / total_cells) if total_cells > 0 else 1.0

    if timestamp_col in candles_df.columns and symbol_col in candles_df.columns:
        monotonic = True
        for _, grp in candles_df.groupby(symbol_col, sort=False):
            ts = pd.to_datetime(grp[timestamp_col], utc=True, errors="coerce")
            if not ts.is_monotonic_increasing:
                monotonic = False
                break
        metrics["timestamp_monotonic"] = 1.0 if monotonic else 0.0
    else:
        metrics["timestamp_monotonic"] = 1.0

    return metrics


def evaluate_failfast(
    metrics: Mapping[str, float], thresholds: FailFastThresholds
) -> FailFastResult:
    reasons: List[str] = []

    rows_per_symbol_min = float(metrics.get("rows_per_symbol_min", thresholds.min_rows_per_symbol))
    if rows_per_symbol_min < thresholds.min_rows_per_symbol:
        reasons.append(
            f"rows_per_symbol_min={rows_per_symbol_min:.0f} < {thresholds.min_rows_per_symbol}"
        )

    nan_ratio = float(metrics.get("nan_ratio", 0.0))
    if nan_ratio > thresholds.max_nan_ratio:
        reasons.append(f"nan_ratio={nan_ratio:.6f} > {thresholds.max_nan_ratio:.6f}")

    monotonic_flag = bool(metrics.get("timestamp_monotonic", 1.0))
    if thresholds.require_monotonic_timestamps and not monotonic_flag:
        reasons.append("timestamps are not monotonic within symbol")

    gradient_norm = float(metrics.get("gradient_norm", 0.0))
    if gradient_norm > thresholds.max_gradient_norm:
        reasons.append(
            f"gradient_norm={gradient_norm:.4f} > {thresholds.max_gradient_norm:.4f}"
        )

    loss = float(metrics.get("loss", 0.0))
    if loss > thresholds.max_loss:
        reasons.append(f"loss={loss:.4f} > {thresholds.max_loss:.4f}")

    epoch_seconds = float(metrics.get("epoch_seconds", 0.0))
    if epoch_seconds > thresholds.max_epoch_seconds:
        reasons.append(
            f"epoch_seconds={epoch_seconds:.2f} > {thresholds.max_epoch_seconds:.2f}"
        )

    exposure = float(metrics.get("max_exposure", 0.0))
    if exposure > thresholds.max_exposure:
        reasons.append(f"max_exposure={exposure:.4f} > {thresholds.max_exposure:.4f}")

    turnover = float(metrics.get("turnover", 0.0))
    if turnover > thresholds.max_turnover:
        reasons.append(f"turnover={turnover:.4f} > {thresholds.max_turnover:.4f}")

    concentration = float(metrics.get("concentration", 0.0))
    if concentration > thresholds.max_concentration:
        reasons.append(
            f"concentration={concentration:.4f} > {thresholds.max_concentration:.4f}"
        )

    confidence = float(metrics.get("confidence", 1.0))
    if confidence < thresholds.min_confidence:
        reasons.append(f"confidence={confidence:.4f} < {thresholds.min_confidence:.4f}")

    convergence = float(metrics.get("convergence", 1.0))
    if convergence < thresholds.min_convergence:
        reasons.append(
            f"convergence={convergence:.4f} < {thresholds.min_convergence:.4f}"
        )

    flow_capture = float(metrics.get("flow_capture", 1.0))
    if flow_capture < thresholds.min_flow_capture:
        reasons.append(
            f"flow_capture={flow_capture:.4f} < {thresholds.min_flow_capture:.4f}"
        )

    return FailFastResult(ok=len(reasons) == 0, reasons=tuple(reasons))


def sample_stochastic_expanse(
    candles_df: pd.DataFrame,
    screws: TrainingScrews,
    *,
    symbol_col: str = "symbol",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Slice a small stochastic dataset from full history using normalized screws.

    - history_depth_portion: per-symbol contiguous depth window in [0,1]
    - coinbase_slot_portion: symbol universe fraction in [0,1]
    """
    cfg = screws.validated()
    if candles_df.empty:
        return candles_df.copy()
    if symbol_col not in candles_df.columns:
        raise ValueError(f"Missing symbol column: {symbol_col}")

    rng = np.random.default_rng(cfg.random_seed)

    symbols = list(dict.fromkeys(candles_df[symbol_col].astype(str).tolist()))
    if cfg.coinbase_slot_portion <= 0.0 or not symbols:
        return candles_df.iloc[0:0].copy()
    n_symbols = max(1, int(round(len(symbols) * cfg.coinbase_slot_portion)))
    chosen_symbols = set(rng.choice(symbols, size=n_symbols, replace=False).tolist())

    scoped = candles_df[candles_df[symbol_col].astype(str).isin(chosen_symbols)].copy()
    if scoped.empty:
        return scoped

    if timestamp_col in scoped.columns:
        scoped = scoped.sort_values([symbol_col, timestamp_col])
    else:
        scoped = scoped.sort_values([symbol_col])

    if cfg.history_depth_portion <= 0.0:
        return scoped.iloc[0:0].copy()

    sampled_chunks: List[pd.DataFrame] = []
    for _, grp in scoped.groupby(symbol_col, sort=False):
        n_rows = len(grp)
        keep = max(1, int(round(n_rows * cfg.history_depth_portion)))
        if keep >= n_rows:
            sampled_chunks.append(grp)
            continue
        start_max = n_rows - keep
        start = int(rng.integers(0, start_max + 1))
        sampled_chunks.append(grp.iloc[start : start + keep])

    sampled = pd.concat(sampled_chunks, axis=0)
    if timestamp_col in sampled.columns:
        sampled = sampled.sort_values([timestamp_col, symbol_col])
    return sampled.reset_index(drop=True)


class HRMTrainingFramework:
    """
    Orthogonal training evaluation framework for HRM experiments.
    """

    def __init__(
        self,
        *,
        screws: Optional[TrainingScrews] = None,
        objective_weights: Optional[ObjectiveWeights] = None,
        failfast_thresholds: Optional[FailFastThresholds] = None,
        competitors: Optional[Sequence[str]] = None,
    ):
        self.screws = (screws or TrainingScrews()).validated()
        self.objective_weights = objective_weights or ObjectiveWeights.pnl_first()
        self.failfast_thresholds = failfast_thresholds or FailFastThresholds()
        chosen_competitors = list(competitors) if competitors is not None else DEFAULT_COMPETITOR_MODELS_24
        if len(chosen_competitors) != 24:
            raise ValueError(f"Competitor roster must have exactly 24 models, got {len(chosen_competitors)}")
        if len(set(chosen_competitors)) != len(chosen_competitors):
            raise ValueError("Competitor roster contains duplicate model names")
        self.competitors = tuple(chosen_competitors)

    def prepare_expanse(
        self,
        candles_df: pd.DataFrame,
        *,
        symbol_col: str = "symbol",
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        return sample_stochastic_expanse(
            candles_df,
            self.screws,
            symbol_col=symbol_col,
            timestamp_col=timestamp_col,
        )

    def judge_candidate(
        self,
        *,
        baseline: CandidateMetrics,
        candidate: CandidateMetrics,
        failfast_metrics: Optional[Mapping[str, float]] = None,
    ) -> BenchmarkVerdict:
        baseline_scored = compute_weighted_objective(baseline, self.objective_weights)
        candidate_scored = compute_weighted_objective(candidate, self.objective_weights)

        failfast = evaluate_failfast(
            failfast_metrics or {}, self.failfast_thresholds
        )

        delta = float(candidate_scored.objective - baseline_scored.objective)
        accepted = bool(failfast.ok and delta > 0.0)

        return BenchmarkVerdict(
            accepted=accepted,
            objective_delta=delta,
            baseline=baseline_scored,
            candidate=candidate_scored,
            failfast=failfast,
            screws=self.screws,
            objective_weights=self.objective_weights,
            competitors=self.competitors,
        )
