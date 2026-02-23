from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import (
    EpisodeTrainingConfig,
    EpochEpisodeTrainer,
    objective_weight_config_from_config,
)


def _trainer_stub(tmp_path: Path) -> EpochEpisodeTrainer:
    trainer = EpochEpisodeTrainer.__new__(EpochEpisodeTrainer)
    trainer.config = EpisodeTrainingConfig(n_epoch_episodes=2, bar_sequences_per_episode=100)
    trainer.session_start_time = "2026-02-23T00:00:00"
    trainer.results = [
        {
            "episode_id": 0,
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "final_capital": 1001.5,
            "realized_pnl": 1.5,
            "hit_rate": 0.6,
            "total_trades": 12,
            "predictor_loss": 0.18,
            "trade_train_loss_mean": -0.04,
            "trade_train_eval_count": 8,
            "energy_train_loss_mean": 0.12,
            "energy_train_eval_count": 5,
            "outlier_extents": 2,
            "optimizer_replays": 1,
        },
        {
            "episode_id": 1,
            "symbols": ["BTCUSDT", "SOLUSDT"],
            "final_capital": 998.8,
            "realized_pnl": -1.2,
            "hit_rate": 0.4,
            "total_trades": 6,
            "predictor_loss": 0.12,
            "trade_train_loss_mean": -0.02,
            "trade_train_eval_count": 4,
            "energy_train_loss_mean": 0.08,
            "energy_train_eval_count": 3,
            "outlier_extents": 0,
            "optimizer_replays": 0,
        },
    ]
    trainer._save_hrm_artifacts = lambda out_dir, stem: {"saved": False, "reason": "test"}
    return trainer


def test_save_checkpoint_includes_objective_decomposition(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trainer = _trainer_stub(tmp_path)

    trainer._save_checkpoint(completed_episodes=2)

    checkpoint = __import__("json").loads((tmp_path / "training_checkpoint.json").read_text())
    telemetry = checkpoint["objective_telemetry"]
    assert telemetry["version"] == 1
    assert "components" in telemetry
    assert set(telemetry["components"]) >= {
        "world_model_term",
        "trade_head_term",
        "energy_routing_term",
        "cost_turnover_term",
        "regime_weighting_term",
    }


def test_save_final_results_backfills_episode_and_summary_objective_telemetry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trainer = _trainer_stub(tmp_path)

    trainer._save_final_results()

    payload = __import__("json").loads((tmp_path / "training_results.json").read_text())
    assert "objective_telemetry" in payload
    assert payload["objective_telemetry"]["episode_count"] == 2
    assert payload["results"][0]["objective_telemetry"]["version"] == 1
    episode_components = payload["results"][0]["objective_telemetry"]["components"]
    assert set(episode_components) >= {
        "world_model_term",
        "trade_head_term",
        "energy_routing_term",
        "cost_turnover_term",
        "regime_weighting_term",
    }


def test_objective_weight_config_defaults_and_overrides():
    defaults = objective_weight_config_from_config(EpisodeTrainingConfig())
    assert defaults["world_model_weight"] == 1.0
    assert defaults["trade_head_weight"] == 1.0
    assert defaults["energy_routing_weight"] == 0.0
    assert defaults["cost_turnover_weight"] == 0.0
    assert defaults["regime_weight_scale"] == 1.0

    cfg = EpisodeTrainingConfig(
        objective_world_model_weight=0.75,
        objective_trade_head_weight=1.25,
        objective_energy_routing_weight=0.4,
        objective_cost_turnover_weight=0.05,
        objective_regime_weight_scale=1.5,
    )
    custom = objective_weight_config_from_config(cfg)
    assert custom["world_model_weight"] == 0.75
    assert custom["trade_head_weight"] == 1.25
    assert custom["energy_routing_weight"] == 0.4
    assert custom["cost_turnover_weight"] == 0.05
    assert custom["regime_weight_scale"] == 1.5


def test_save_outputs_persist_objective_weight_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trainer = _trainer_stub(tmp_path)
    trainer.config = EpisodeTrainingConfig(
        n_epoch_episodes=2,
        objective_world_model_weight=0.6,
        objective_trade_head_weight=1.4,
        objective_energy_routing_weight=0.2,
        objective_cost_turnover_weight=0.03,
        objective_regime_weight_scale=1.2,
    )

    trainer._save_checkpoint(completed_episodes=2)
    trainer._save_final_results()

    checkpoint = __import__("json").loads((tmp_path / "training_checkpoint.json").read_text())
    results = __import__("json").loads((tmp_path / "training_results.json").read_text())
    for payload in (checkpoint, results):
        cfg = payload["objective_weight_config"]
        assert cfg["world_model_weight"] == 0.6
        assert cfg["trade_head_weight"] == 1.4
        assert cfg["energy_routing_weight"] == 0.2
        assert cfg["cost_turnover_weight"] == 0.03
        assert cfg["regime_weight_scale"] == 1.2
