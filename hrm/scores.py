"""
HRM Algorithm Score Display
============================

Shows live algorithm scores per symbol — NOT prices.

    python hrm/scores.py                        # all symbols, latest bar
    python hrm/scores.py --symbols BTC-USD ETH-USD
    python hrm/scores.py --watch 60             # refresh every 60s
    python hrm/scores.py --regime volatility    # filter by regime
    python hrm/scores.py --sort bull            # sort by bull score
    python hrm/scores.py --metrics              # show per-model backtest metrics

Display columns:
    MODEL         — algorithm name
    REGIME        — signal family (trend/mean_rev/vol/stat_arb/sys/ml)
    SIGNAL ▐▌     — bar chart [-1 → +1]  red=short  green=long
    CONF  ░░░░    — confidence bar [0 → 1]
    SCORE         — signal × confidence (the actual alpha contribution)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.rule import Rule

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
# Import directly from pipeline module — avoids hrm/__init__.py numba chain
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("hrm_pipeline",
    os.path.join(os.path.dirname(__file__), "pipeline.py"))
_mod = _ilu.module_from_spec(_spec)
sys.modules["hrm_pipeline"] = _mod   # register before exec so @dataclass resolves
_spec.loader.exec_module(_mod)

load_candles          = _mod.load_candles
compute_features      = _mod.compute_features
compute_all_signals   = _mod.compute_all_signals
compute_regime_alphas = _mod.compute_regime_alphas
compute_final_alpha   = _mod.compute_final_alpha
hrm_regime_weights    = _mod.hrm_regime_weights
DB_PATH               = _mod.DB_PATH
Regime                = _mod.Regime

console = Console()

# ── Colour mapping ─────────────────────────────────────────────────────────
REGIME_COLOUR = {
    Regime.TREND:          "cyan",
    Regime.MEAN_REVERSION: "yellow",
    Regime.VOLATILITY:     "magenta",
    Regime.STAT_ARB:       "blue",
    Regime.SYSTEMATIC:     "white",
    Regime.ML:             "green",
}

REGIME_ICON = {
    Regime.TREND:          "↑",
    Regime.MEAN_REVERSION: "↔",
    Regime.VOLATILITY:     "~",
    Regime.STAT_ARB:       "⊕",
    Regime.SYSTEMATIC:     "⏱",
    Regime.ML:             "🤖",
}

# Abbreviated regime labels for compact display
REGIME_SHORT = {
    Regime.TREND:          "trend",
    Regime.MEAN_REVERSION: "mean_rev",
    Regime.VOLATILITY:     "vol",
    Regime.STAT_ARB:       "stat_arb",
    Regime.SYSTEMATIC:     "sys",
    Regime.ML:             "ml",
}

# ── Bar helpers ─────────────────────────────────────────────────────────────

def _signal_bar(signal: float, width: int = 20) -> Text:
    """
    Render signal in [-1, 1] as a bidirectional bar.
    Centre = 0.  Left = short (red).  Right = long (green).
    """
    half = width // 2
    pos = int(round(signal * half))
    t = Text()
    if pos >= 0:
        t.append(" " * half, style="on red")
        t.append("▐" + "█" * pos + " " * (half - pos), style="bold green")
    else:
        filled = -pos
        t.append(" " * (half - filled) + "█" * filled + "▌", style="bold red")
        t.append(" " * half, style="on green")
    return t


def _conf_bar(conf: float, width: int = 8) -> Text:
    n = int(round(conf * width))
    t = Text()
    colour = "green" if conf > 0.6 else ("yellow" if conf > 0.3 else "red")
    t.append("█" * n + "░" * (width - n), style=colour)
    return t


def _score_colour(score: float) -> str:
    if score > 0.3:  return "bold green"
    if score > 0.0:  return "green"
    if score > -0.3: return "red"
    return "bold red"


# ── Per-model backtest metrics table ─────────────────────────────────────────

def build_model_metrics_table(signals_df: pd.DataFrame,
                               candles_df: pd.DataFrame) -> Table:
    """
    Compute per-model backtested performance metrics over the full history
    and return a Rich Table with one stable row per model, sorted by Sharpe.

    Metrics:
        PnL%       — cumulative return of (signal * next_bar_return) * 100
        Max DD%    — worst peak-to-trough of cumulative PnL
        Sharpe     — annualized Sharpe (hourly: sqrt(8760)) = mean/std of hourly PnL
        Win Rate%  — fraction of bars where signal*return > 0
        Signal     — latest bar signal averaged across symbols
        Confidence — latest bar confidence averaged across symbols

    All computations are vectorized with pandas — no Python row loops.
    """
    # ── Step 1: compute per-bar returns from close prices ──────────────────
    # next_bar_return = (close[t+1] - close[t]) / close[t]
    candles_sorted = candles_df.sort_values(["symbol", "timestamp"]).copy()
    candles_sorted["next_return"] = (
        candles_sorted.groupby("symbol")["close"]
        .transform(lambda s: s.shift(-1) / s - 1)
    )
    # Drop last bar per symbol (no next return available)
    returns_df = candles_sorted[["symbol", "timestamp", "next_return"]].dropna(
        subset=["next_return"]
    )

    # ── Step 2: merge signals with returns ────────────────────────────────
    merged = signals_df.merge(
        returns_df[["symbol", "timestamp", "next_return"]],
        on=["symbol", "timestamp"],
        how="inner",
    )
    # bar_pnl = signal * next_return  (vectorized)
    merged["bar_pnl"] = merged["signal"] * merged["next_return"]

    # ── Step 3: aggregate across symbols per (model, timestamp) ───────────
    # Average the bar_pnl across symbols for each model-timestamp pair so that
    # all symbols contribute equally regardless of price magnitude.
    model_ts_pnl = (
        merged.groupby(["model", "timestamp"])["bar_pnl"]
        .mean()
        .reset_index()
        .sort_values(["model", "timestamp"])
    )

    # ── Step 4: per-model vectorized metrics ──────────────────────────────
    def _model_metrics(grp: pd.DataFrame) -> pd.Series:
        pnl = grp["bar_pnl"].values.astype(np.float64)
        cum = np.cumsum(pnl)

        # PnL%
        total_pnl_pct = float(cum[-1]) * 100.0 if len(cum) > 0 else 0.0

        # Max drawdown (peak-to-trough of cumulative PnL)
        peak = np.maximum.accumulate(cum)
        drawdown = cum - peak  # always <= 0
        max_dd_pct = float(drawdown.min()) * 100.0 if len(drawdown) > 0 else 0.0

        # Sharpe — annualised, hourly bars → annualisation factor = sqrt(8760)
        mean_pnl = np.mean(pnl)
        std_pnl  = np.std(pnl, ddof=1)
        sharpe = (mean_pnl / std_pnl * np.sqrt(8760)) if std_pnl > 1e-12 else 0.0

        # Win rate
        wins = float(np.sum(pnl > 0))
        total_bars = float(len(pnl))
        win_rate_pct = (wins / total_bars * 100.0) if total_bars > 0 else 0.0

        return pd.Series({
            "pnl_pct":     total_pnl_pct,
            "max_dd_pct":  max_dd_pct,
            "sharpe":      float(sharpe),
            "win_rate_pct": win_rate_pct,
        })

    perf = model_ts_pnl.groupby("model").apply(_model_metrics).reset_index()

    # ── Step 5: latest-bar signal & confidence (averaged across symbols) ───
    latest_ts = signals_df["timestamp"].max()
    latest = (
        signals_df[signals_df["timestamp"] == latest_ts]
        .groupby("model")[["signal", "confidence"]]
        .mean()
        .reset_index()
    )

    # ── Step 6: regime lookup (from first occurrence) ─────────────────────
    regime_map = (
        signals_df.groupby("model")["regime"].first().reset_index()
    )

    # ── Step 7: merge everything ───────────────────────────────────────────
    metrics = (
        perf
        .merge(latest, on="model", how="left")
        .merge(regime_map, on="model", how="left")
        .fillna({"signal": 0.0, "confidence": 0.0})
        .sort_values("sharpe", ascending=False)
    )

    # ── Step 8: build Rich Table ───────────────────────────────────────────
    tbl = Table(
        title="[bold]Per-Model Backtest Metrics[/bold]  (signal × next-bar return, full history)",
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold dim",
        expand=False,
        min_width=100,
    )
    tbl.add_column("MODEL",   width=24, no_wrap=True)
    tbl.add_column("REGIME",  width=10, no_wrap=True)
    tbl.add_column("PNL%",    width=8,  justify="right", no_wrap=True)
    tbl.add_column("DD%",     width=7,  justify="right", no_wrap=True)
    tbl.add_column("SHARPE",  width=7,  justify="right", no_wrap=True)
    tbl.add_column("WIN%",    width=5,  justify="right", no_wrap=True)
    tbl.add_column("SIG",     width=6,  no_wrap=True)
    tbl.add_column("CONF" + "█" * 4, width=10, no_wrap=True)

    for _, row in metrics.iterrows():
        pnl_pct    = float(row["pnl_pct"])
        dd_pct     = float(row["max_dd_pct"])
        sharpe     = float(row["sharpe"])
        win_rate   = float(row["win_rate_pct"])
        signal     = float(row["signal"])
        conf       = float(row["confidence"])
        regime     = str(row["regime"]) if pd.notna(row["regime"]) else ""
        regime_short = REGIME_SHORT.get(regime, regime[:8])
        regime_colour = REGIME_COLOUR.get(regime, "white")

        # PnL colour
        pnl_style = "bold green" if pnl_pct > 0 else "bold red"
        pnl_str   = f"{pnl_pct:+.1f}%"

        # Drawdown (always negative, show in red)
        dd_str   = f"{dd_pct:.1f}%"
        dd_style = "red" if dd_pct < -5 else "yellow"

        # Sharpe colour
        if sharpe > 1.0:
            sharpe_style = "bold bright_green"
        elif sharpe > 0.0:
            sharpe_style = "green"
        else:
            sharpe_style = "red"
        sharpe_str = f"{sharpe:.2f}"

        win_str = f"{win_rate:.0f}%"

        # Compact signal indicator: 3-char wide
        sig_n = int(round(signal * 2)) + 2   # map [-1,1] -> [0,4] out of 4 blocks
        sig_n = max(0, min(4, sig_n))
        sig_text = Text()
        sig_style = "bold green" if signal > 0.05 else ("bold red" if signal < -0.05 else "dim")
        sig_text.append(f"{'▐' if signal >= 0 else ''}{'█' * sig_n:4s}", style=sig_style)

        tbl.add_row(
            Text(str(row["model"])[:23], style=regime_colour),
            Text(regime_short, style=regime_colour),
            Text(pnl_str, style=pnl_style),
            Text(dd_str,  style=dd_style),
            Text(sharpe_str, style=sharpe_style),
            Text(win_str),
            sig_text,
            _conf_bar(conf, width=8),
        )

    return tbl


# ── Per-symbol signal table ─────────────────────────────────────────────────

def build_signal_table(signals_snapshot: pd.DataFrame, symbol: str,
                        regime_filter: Optional[str] = None) -> Table:
    """
    Render all model scores for one symbol at the latest bar.
    """
    sym_df = signals_snapshot[signals_snapshot["symbol"] == symbol].copy()
    if regime_filter:
        sym_df = sym_df[sym_df["regime"] == regime_filter]

    sym_df = sym_df.sort_values("regime")
    sym_df["score"] = sym_df["signal"] * sym_df["confidence"]

    tbl = Table(
        title=f"[bold]{symbol}[/bold]",
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold dim",
        expand=False,
        min_width=72,
    )
    tbl.add_column("MODEL",  style="dim",  width=24, no_wrap=True)
    tbl.add_column("REGIME", width=12)
    tbl.add_column(f"SIGNAL{'':>5}[-1 ← 0 → +1]", width=22, no_wrap=True)
    tbl.add_column("CONF",   width=10, no_wrap=True)
    tbl.add_column("SCORE",  width=7, justify="right")

    for _, row in sym_df.iterrows():
        regime   = row["regime"]
        colour   = REGIME_COLOUR.get(regime, "white")
        icon     = REGIME_ICON.get(regime, "")
        score    = row["score"]

        tbl.add_row(
            Text(row["model"][:23], style=colour),
            Text(f"{icon} {regime[:10]}", style=colour),
            _signal_bar(row["signal"]),
            _conf_bar(row["confidence"]),
            Text(f"{score:+.3f}", style=_score_colour(score)),
        )

    return tbl


# ── Regime summary panel ─────────────────────────────────────────────────────

def build_regime_summary(decisions: pd.DataFrame, symbol: str) -> Panel:
    """Bull/bear/regime breakdown panel for one symbol."""
    row = decisions[decisions["symbol"] == symbol]
    if row.empty:
        return Panel("No data", title=symbol)

    row = row.iloc[0]
    alpha = row["alpha"]
    bull  = row["bull_score"]
    bear  = row["bear_score"]
    dom   = row["dominant_regime"]
    model = row["dominant_model"]

    alpha_bar = _signal_bar(alpha, width=30)
    bull_bar  = _conf_bar(max(bull, 0), width=10)
    bear_bar  = _conf_bar(max(-bear, 0), width=10)

    t = Text()
    t.append(f"ALPHA  ", style="bold")
    t.append(alpha_bar)
    t.append(f"  {alpha:+.3f}\n", style=_score_colour(alpha))
    t.append(f"BULL ↑ ", style="bold green")
    t.append(bull_bar)
    t.append(f"  {bull:+.3f}\n", style="green")
    t.append(f"BEAR ↓ ", style="bold red")
    t.append(bear_bar)
    t.append(f"  {bear:+.3f}\n", style="red")
    t.append(f"REGIME ", style="bold")
    colour = REGIME_COLOUR.get(dom, "white")
    t.append(f"{REGIME_ICON.get(dom,'')} {dom}", style=colour)
    t.append(f"\nTOP    ", style="bold")
    t.append(model, style=colour)

    return Panel(t, title=f"[bold]{symbol}[/bold] scores", expand=False)


# ── Universe leaderboard ─────────────────────────────────────────────────────

def build_leaderboard(decisions: pd.DataFrame,
                       sort_by: str = "alpha") -> Table:
    """
    Ranked table across all symbols — algo scores only.
    """
    df = decisions.copy()
    if sort_by == "bull":
        df = df.sort_values("bull_score", ascending=False)
    elif sort_by == "bear":
        df = df.sort_values("bear_score", ascending=True)
    else:
        df = df.sort_values("alpha", ascending=False)

    tbl = Table(
        title="[bold]Universe Algo Scores[/bold]  (no prices)",
        box=box.MARKDOWN,
        show_header=True,
        header_style="bold",
        expand=False,
    )
    tbl.add_column("SYMBOL",  width=12, no_wrap=True)
    tbl.add_column("ALPHA  [-1 ← 0 → +1]", width=24)
    tbl.add_column("BULL ↑", width=10, no_wrap=True)
    tbl.add_column("BEAR ↓", width=10, no_wrap=True)
    tbl.add_column("SCORE",  width=7,  justify="right")
    tbl.add_column("REGIME", width=14)
    tbl.add_column("TOP MODEL", width=24, no_wrap=True)

    for _, row in df.iterrows():
        alpha = row["alpha"]
        bull  = row["bull_score"]
        bear  = row["bear_score"]
        dom   = row["dominant_regime"]
        colour = REGIME_COLOUR.get(dom, "white")
        icon   = REGIME_ICON.get(dom, "")
        tbl.add_row(
            Text(str(row["symbol"]), style="bold"),
            _signal_bar(alpha, width=20),
            _conf_bar(max(float(bull), 0), width=8),
            _conf_bar(max(float(-bear), 0), width=8),
            Text(f"{alpha:+.3f}", style=_score_colour(alpha)),
            Text(f"{icon} {dom[:12]}", style=colour),
            Text(str(row["dominant_model"])[:23], style=colour),
        )

    return tbl


# ── Main render ──────────────────────────────────────────────────────────────

def render_scores(symbols: Optional[List[str]] = None,
                  regime_filter: Optional[str] = None,
                  sort_by: str = "alpha",
                  checkpoint: Optional[Path] = None,
                  limit_hours: int = 500,
                  show_metrics: bool = True) -> None:
    """Load data, run pipeline, render algo scores."""

    with console.status("[bold green]Loading candles...", spinner="dots"):
        candles = load_candles(symbols=symbols, limit_hours=limit_hours)

    with console.status("[bold green]Computing features...", spinner="dots"):
        features = compute_features(candles)

    with console.status("[bold green]Running 25 signal models...", spinner="dots"):
        signals = compute_all_signals(candles, features)

    with console.status("[bold green]HRM regime inference...", spinner="dots"):
        regime_weights = hrm_regime_weights(features, checkpoint_path=checkpoint,
                                             symbols=symbols)

    with console.status("[bold green]Aggregating alphas...", spinner="dots"):
        regime_alphas = compute_regime_alphas(signals, regime_weights)
        decisions_all = compute_final_alpha(regime_alphas)
        # One row per symbol — latest bar only (for display)
        decisions = (decisions_all.sort_values("timestamp")
                                   .groupby("symbol", as_index=False)
                                   .last())

    # Latest bar only for signal table
    latest_ts = signals["timestamp"].max()
    snapshot = signals[signals["timestamp"] == latest_ts]

    console.print()
    console.print(Rule(f"[bold]HRM Algorithm Scores[/bold]  —  {latest_ts}  —  "
                       f"device: {'MPS' if 'mps' in str(regime_weights) else 'CPU'}"))

    # ── Per-model metrics table (default first output) ──────────────────────
    if show_metrics:
        with console.status("[bold green]Computing per-model backtest metrics...",
                            spinner="dots"):
            metrics_tbl = build_model_metrics_table(signals, candles)
        console.print()
        console.print(metrics_tbl)
        console.print()

    # Regime weight strip
    regime_row = []
    for reg in Regime.ALL:
        w = regime_weights.get(reg, 0)
        colour = REGIME_COLOUR.get(reg, "white")
        icon   = REGIME_ICON.get(reg, "")
        regime_row.append(
            Panel(Text(f"{icon} {reg}\n{_conf_bar(w, 6)}\n{w:.2%}", justify="center"),
                  style=colour, expand=False, padding=(0, 1))
        )
    console.print(Columns(regime_row))
    console.print()

    # Universe leaderboard
    console.print(build_leaderboard(decisions, sort_by=sort_by))
    console.print()

    # Per-symbol detail (top 5 by |alpha| or filtered symbols)
    if symbols:
        detail_syms = [s for s in symbols if s in decisions["symbol"].values]
    else:
        detail_syms = decisions.nlargest(5, "alpha")["symbol"].tolist() + \
                      decisions.nsmallest(2, "alpha")["symbol"].tolist()
        detail_syms = list(dict.fromkeys(detail_syms))  # dedup, preserve order

    for sym in detail_syms:
        console.print(build_regime_summary(decisions, sym))
        console.print(build_signal_table(snapshot, sym, regime_filter=regime_filter))
        console.print()


def watch_loop(interval: int, **kwargs) -> None:
    """Refresh score display every `interval` seconds."""
    console.print(f"[dim]Watching every {interval}s — Ctrl+C to stop[/dim]")
    while True:
        console.clear()
        try:
            render_scores(**kwargs)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
        console.print(f"[dim]Next refresh in {interval}s...[/dim]")
        time.sleep(interval)


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="HRM Algorithm Score Display")
    ap.add_argument("--symbols",    nargs="*",            help="Symbol filter")
    ap.add_argument("--regime",     default=None,
                    choices=Regime.ALL,                   help="Filter signal table by regime")
    ap.add_argument("--sort",       default="alpha",
                    choices=["alpha", "bull", "bear"],    help="Leaderboard sort key")
    ap.add_argument("--watch",      type=int, default=0,  help="Auto-refresh interval (seconds, 0=off)")
    ap.add_argument("--checkpoint", default=None,         help="HRM checkpoint .pt path")
    ap.add_argument("--limit",      type=int, default=500, help="Recent hours of candle data")
    ap.add_argument("--metrics",    action="store_true",  default=True,
                    help="Show per-model backtest metrics table (default: on)")
    ap.add_argument("--no-metrics", action="store_false", dest="metrics",
                    help="Suppress per-model backtest metrics table")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint) if args.checkpoint else None

    kwargs = dict(symbols=args.symbols, regime_filter=args.regime,
                  sort_by=args.sort, checkpoint=ckpt, limit_hours=args.limit,
                  show_metrics=args.metrics)

    if args.watch:
        watch_loop(args.watch, **kwargs)
    else:
        render_scores(**kwargs)
