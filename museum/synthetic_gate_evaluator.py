#!/usr/bin/env python3
"""
Synthetic gate evaluator for HRM readiness milestones.

This suite does not certify HRM by itself. It defines the cheap tasks and
baseline references that HRM must eventually beat or match before promotion.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class SyntheticTask:
    name: str
    inputs: np.ndarray
    targets: np.ndarray
    primary_signal: np.ndarray
    horizons: Tuple[int, ...]
    expected_outcome: str


def _ema(values: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    if values.size == 0:
        return out
    out[0] = float(values[0])
    for idx in range(1, values.size):
        out[idx] = (alpha * float(values[idx])) + ((1.0 - alpha) * float(out[idx - 1]))
    return out


def _rolling_std(values: np.ndarray, window: int = 8) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    for idx in range(values.size):
        start = max(0, idx - window + 1)
        chunk = values[start : idx + 1]
        out[idx] = float(np.std(chunk)) if chunk.size else 0.0
    return out


def build_feature_matrix(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    prev = np.concatenate(([signal[0]], signal[:-1]))
    momentum = signal - prev
    ema = _ema(signal)
    vol = _rolling_std(signal)
    return np.column_stack([signal, prev, momentum, ema, vol]).astype(np.float32)


def generate_identity_task(length: int = 256, seed: int = 7) -> SyntheticTask:
    rng = np.random.default_rng(seed)
    signal = rng.normal(loc=0.0, scale=1.0, size=length).astype(np.float32)
    inputs = signal.reshape(-1, 1)
    targets = signal.reshape(-1, 1)
    return SyntheticTask(
        name="identity",
        inputs=inputs,
        targets=targets,
        primary_signal=signal,
        horizons=(0,),
        expected_outcome="near_zero",
    )


def generate_sine_task(length: int = 256, seed: int = 7) -> SyntheticTask:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 8.0 * np.pi, length + 1, dtype=np.float64)
    signal = (
        np.sin(t)
        + 0.35 * np.sin((2.3 * t) + 0.4)
        + 0.1 * np.sin((5.1 * t) - 0.3)
        + rng.normal(loc=0.0, scale=0.01, size=length + 1)
    ).astype(np.float32)
    inputs = build_feature_matrix(signal[:-1])
    targets = signal[1:].reshape(-1, 1)
    return SyntheticTask(
        name="sine",
        inputs=inputs,
        targets=targets,
        primary_signal=signal[:-1],
        horizons=(1,),
        expected_outcome="near_zero_or_better_than_baseline",
    )


def generate_feature_plus_one_task(length: int = 256, seed: int = 7) -> SyntheticTask:
    signal = generate_sine_task(length=length, seed=seed).primary_signal
    full_signal = np.concatenate([signal, signal[-1:]])
    features = build_feature_matrix(full_signal)
    inputs = features[:-1]
    targets = features[1:, :1]
    return SyntheticTask(
        name="feature_plus_1",
        inputs=inputs,
        targets=targets,
        primary_signal=features[:-1, 0],
        horizons=(1,),
        expected_outcome="beats_persistence_and_ema",
    )


def generate_multi_horizon_task(
    length: int = 256,
    horizons: Sequence[int] = (1, 2, 4, 8),
    seed: int = 7,
) -> SyntheticTask:
    horizon_tuple = tuple(int(h) for h in horizons)
    max_h = max(horizon_tuple)
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 10.0 * np.pi, length + max_h, dtype=np.float64)
    regime = np.where(np.arange(length + max_h) < (length + max_h) // 2, 1.0, 0.55)
    signal = (
        regime * (np.sin(t) + 0.25 * np.sin((3.0 * t) + 0.7))
        + 0.02 * np.arange(length + max_h)
        + rng.normal(loc=0.0, scale=0.015, size=length + max_h)
    ).astype(np.float32)
    features = build_feature_matrix(signal)
    inputs = features[:-max_h]
    targets = np.column_stack(
        [signal[h : h + inputs.shape[0]] for h in horizon_tuple]
    ).astype(np.float32)
    return SyntheticTask(
        name="feature_plus_n",
        inputs=inputs,
        targets=targets,
        primary_signal=signal[: inputs.shape[0]],
        horizons=horizon_tuple,
        expected_outcome="graceful_multi_horizon_degradation",
    )


def evaluate_predictions(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    tgt = np.asarray(targets, dtype=np.float64)
    pred = np.asarray(predictions, dtype=np.float64)
    err = pred - tgt
    return {
        "mse": float(np.mean(np.square(err))),
        "mae": float(np.mean(np.abs(err))),
    }


def persistence_predictions(task: SyntheticTask) -> np.ndarray:
    current = np.asarray(task.primary_signal, dtype=np.float64)
    if task.horizons == (0,):
        return current.reshape(-1, 1)
    return np.column_stack([current for _ in task.horizons])


def ema_predictions(task: SyntheticTask, alpha: float = 0.25) -> np.ndarray:
    ema = _ema(np.asarray(task.primary_signal, dtype=np.float64), alpha=alpha)
    return np.column_stack([ema for _ in task.horizons])


def linear_predictions(task: SyntheticTask) -> np.ndarray:
    current = np.asarray(task.primary_signal, dtype=np.float64)
    diff = np.concatenate(([0.0], np.diff(current)))
    return np.column_stack([current + (float(h) * diff) for h in task.horizons])


def evaluate_task_against_baselines(task: SyntheticTask) -> Dict[str, object]:
    baselines = {
        "persistence": persistence_predictions(task),
        "ema": ema_predictions(task),
        "linear": linear_predictions(task),
    }
    baseline_metrics = {
        name: evaluate_predictions(task.targets, pred) for name, pred in baselines.items()
    }
    best_name = min(baseline_metrics, key=lambda name: baseline_metrics[name]["mse"])
    return {
        "name": task.name,
        "expected_outcome": task.expected_outcome,
        "horizons": list(task.horizons),
        "n_samples": int(task.targets.shape[0]),
        "input_width": int(task.inputs.shape[1]),
        "target_width": int(task.targets.shape[1]),
        "baselines": baseline_metrics,
        "best_baseline": best_name,
    }


def run_reference_suite(length: int = 256, seed: int = 7) -> Dict[str, object]:
    tasks = [
        generate_identity_task(length=length, seed=seed),
        generate_sine_task(length=length, seed=seed),
        generate_feature_plus_one_task(length=length, seed=seed),
        generate_multi_horizon_task(length=length, seed=seed),
    ]
    evaluations = [evaluate_task_against_baselines(task) for task in tasks]
    return {
        "suite": "hrm_synthetic_reference",
        "seed": int(seed),
        "length": int(length),
        "tasks": evaluations,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the HRM synthetic gate reference suite.")
    parser.add_argument("--length", type=int, default=256, help="Base sequence length per synthetic task.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for deterministic task generation.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable text.")
    args = parser.parse_args(argv)

    suite = run_reference_suite(length=int(args.length), seed=int(args.seed))
    if args.json:
        print(json.dumps(suite, indent=2))
        return 0

    print("HRM Synthetic Gate Reference Suite")
    print(f"seed={suite['seed']} length={suite['length']}")
    for task in suite["tasks"]:
        print(
            f"- {task['name']}: expected={task['expected_outcome']} "
            f"best_baseline={task['best_baseline']} "
            f"mse={task['baselines'][task['best_baseline']]['mse']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
