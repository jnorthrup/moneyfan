#!/usr/bin/env python3
"""
Isolated HRM paper-tuning runner.

Runs short `train.py` episode simulations under multiple config variants and writes
all outputs to a timestamped experiment directory so the main training artifacts
(`training_results.json`, `training_checkpoint.json`) remain untouched.
"""

import argparse
import json
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from train import EpisodeTrainingConfig, EpochEpisodeTrainer


VARIANT_PRESETS: Dict[str, Dict[str, Any]] = {
    "baseline": {},
    "veto": {"use_mechanical_veto": True},
    "veto_replay": {"use_mechanical_veto": True, "replay_coalescing": True},
    "small_pair": {"pair_width": 12},
    "hyperbolic": {"ob_decay_mode": "hyperbolic"},
}


class ExperimentTrainer(EpochEpisodeTrainer):
    def __init__(self, config: EpisodeTrainingConfig, run_dir: Path):
        self._run_dir = run_dir
        self._run_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(config)

    def _save_checkpoint(self, completed_episodes: int):
        checkpoint = {
            "completed_episodes": completed_episodes,
            "total_episodes": self.config.n_epoch_episodes,
            "session_start_time": self.session_start_time,
            "checkpoint_time": datetime.now().isoformat(),
            "results": self.results,
        }
        with open(self._run_dir / "training_checkpoint.json", "w") as f:
            json.dump(checkpoint, f, indent=2)

    def _save_final_results(self):
        realized = [r["realized_pnl"] for r in self.results if "realized_pnl" in r]
        hit_rates = [r["hit_rate"] for r in self.results if "hit_rate" in r]
        finals = [r["final_capital"] for r in self.results if "final_capital" in r]
        summary = {
            "total_episodes": len(self.results),
            "session_start_time": self.session_start_time,
            "session_end_time": datetime.now().isoformat(),
            "avg_realized_pnl": float(np.mean(realized)) if realized else 0.0,
            "avg_hit_rate": float(np.mean(hit_rates)) if hit_rates else 0.0,
            "avg_final_capital": float(np.mean(finals)) if finals else 0.0,
            "total_notional": float(sum(finals)) if finals else 0.0,
            "results": self.results,
        }

        with open(self._run_dir / "training_results.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(
            f"[Experiment] {self._run_dir.name}: episodes={summary['total_episodes']} "
            f"avg_pnl={summary['avg_realized_pnl']:.4f} avg_hit={summary['avg_hit_rate']:.2%}"
        )


def _coerce_overrides(base: Dict[str, Any], preset: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    merged.update(preset)
    return merged


def _summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    pnl = np.array([r.get("realized_pnl", 0.0) for r in results], dtype=float)
    hit = np.array([r.get("hit_rate", 0.0) for r in results], dtype=float)
    trades = np.array([r.get("total_trades", 0) for r in results], dtype=float)
    vetoes = np.array([r.get("veto_count", 0) for r in results], dtype=float)
    replays = np.array([r.get("optimizer_replays", 0) for r in results], dtype=float)
    return {
        "episodes": int(len(results)),
        "avg_realized_pnl": float(np.mean(pnl)) if len(pnl) else 0.0,
        "median_realized_pnl": float(np.median(pnl)) if len(pnl) else 0.0,
        "std_realized_pnl": float(np.std(pnl)) if len(pnl) else 0.0,
        "avg_hit_rate": float(np.mean(hit)) if len(hit) else 0.0,
        "avg_total_trades": float(np.mean(trades)) if len(trades) else 0.0,
        "avg_veto_count": float(np.mean(vetoes)) if len(vetoes) else 0.0,
        "avg_optimizer_replays": float(np.mean(replays)) if len(replays) else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Run isolated HRM paper-tuning experiments")
    parser.add_argument("--episodes", type=int, default=8, help="Episodes per variant")
    parser.add_argument("--notional", type=float, default=100.0, help="Starting notional")
    parser.add_argument("--pair-width", type=int, default=30, help="Pairs per episode")
    parser.add_argument("--bar-seqs", type=int, default=40, help="Bar windows per episode")
    parser.add_argument("--min-window", type=int, default=64, help="Minimum bar window")
    parser.add_argument("--max-window", type=int, default=192, help="Maximum bar window")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "lion", "muon"])
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument(
        "--variants",
        type=str,
        default="baseline,veto",
        help=f"Comma-separated variants from: {', '.join(sorted(VARIANT_PRESETS))}",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Repeat count per variant")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument("--out-dir", type=str, default="", help="Explicit output directory")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = Path(args.out_dir) if args.out_dir else Path("experiments") / f"paper_tune_{ts}"
    root_dir.mkdir(parents=True, exist_ok=True)

    requested_variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in requested_variants if v not in VARIANT_PRESETS]
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}. Valid: {sorted(VARIANT_PRESETS)}")

    base_kwargs: Dict[str, Any] = {
        "n_epoch_episodes": args.episodes,
        "notional": args.notional,
        "pair_width": args.pair_width,
        "bar_sequences_per_episode": args.bar_seqs,
        "min_bar_window": args.min_window,
        "max_bar_window": args.max_window,
        "optimizer_name": args.optimizer,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    }

    experiment_index: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "root_dir": str(root_dir.resolve()),
        "base_config": base_kwargs,
        "variants": {},
    }

    for variant_name in requested_variants:
        preset = VARIANT_PRESETS[variant_name]
        variant_runs: List[Dict[str, Any]] = []
        for rep in range(args.repeat):
            run_seed = args.seed + (1000 * rep) + (10000 * requested_variants.index(variant_name))
            np.random.seed(run_seed)
            random.seed(run_seed)

            cfg_kwargs = _coerce_overrides(base_kwargs, preset)
            config = EpisodeTrainingConfig(**cfg_kwargs)
            run_slug = f"{variant_name}_rep{rep+1:02d}"
            run_dir = root_dir / run_slug
            print(f"[Experiment] Starting {run_slug} (seed={run_seed})")
            trainer = ExperimentTrainer(config, run_dir)
            trainer.run_episode_training()

            run_summary = {
                "run_id": run_slug,
                "seed": run_seed,
                "config_overrides": preset,
                "summary": _summarize_results(trainer.results),
                "run_dir": str(run_dir.resolve()),
            }
            variant_runs.append(run_summary)
            with open(run_dir / "run_summary.json", "w") as f:
                json.dump(run_summary, f, indent=2)

        # Aggregate repeats
        agg_pnl = [r["summary"]["avg_realized_pnl"] for r in variant_runs]
        agg_hit = [r["summary"]["avg_hit_rate"] for r in variant_runs]
        agg_trades = [r["summary"]["avg_total_trades"] for r in variant_runs]
        experiment_index["variants"][variant_name] = {
            "preset": preset,
            "runs": variant_runs,
            "aggregate": {
                "repeat_count": len(variant_runs),
                "mean_avg_realized_pnl": float(np.mean(agg_pnl)) if agg_pnl else 0.0,
                "mean_avg_hit_rate": float(np.mean(agg_hit)) if agg_hit else 0.0,
                "mean_avg_total_trades": float(np.mean(agg_trades)) if agg_trades else 0.0,
            },
        }

    with open(root_dir / "experiment_index.json", "w") as f:
        json.dump(experiment_index, f, indent=2)

    print("\n[Experiment] Complete")
    print(json.dumps(experiment_index["variants"], indent=2))


if __name__ == "__main__":
    main()
