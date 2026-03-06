import numpy as np

from synthetic_gate_evaluator import (
    evaluate_predictions,
    generate_identity_task,
    generate_multi_horizon_task,
    persistence_predictions,
    run_reference_suite,
)


def test_identity_task_has_matching_input_and_target_shapes():
    task = generate_identity_task(length=32, seed=3)

    assert task.inputs.shape == (32, 1)
    assert task.targets.shape == (32, 1)
    assert task.horizons == (0,)


def test_perfect_predictions_produce_zero_error():
    task = generate_identity_task(length=16, seed=5)
    metrics = evaluate_predictions(task.targets, task.targets)

    assert metrics == {"mse": 0.0, "mae": 0.0}


def test_persistence_is_perfect_for_identity_task():
    task = generate_identity_task(length=24, seed=11)
    pred = persistence_predictions(task)
    metrics = evaluate_predictions(task.targets, pred)

    assert metrics["mse"] == 0.0
    assert metrics["mae"] == 0.0


def test_multi_horizon_task_target_width_matches_horizons():
    task = generate_multi_horizon_task(length=40, horizons=(1, 2, 4), seed=2)

    assert task.targets.shape[1] == 3
    assert task.horizons == (1, 2, 4)


def test_reference_suite_contains_expected_tasks():
    suite = run_reference_suite(length=48, seed=7)
    task_names = [task["name"] for task in suite["tasks"]]

    assert task_names == ["identity", "sine", "feature_plus_1", "feature_plus_n"]
    assert all(task["best_baseline"] in {"persistence", "ema", "linear"} for task in suite["tasks"])
