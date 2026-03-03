"""
Training smoke profile tests - validates bounded training profiles for fast iteration.

These profiles provide operator-facing configurations for quick training loops
that can be evaluated by the Freqtrade ring agent.
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from train import (
    EpisodeTrainingConfig,
    objective_weight_config_from_config,
)


class TestTrainingSmokeProfiles:
    """Tests for bounded MLX training smoke profiles."""

    def test_smoke_profile_minimal_config(self):
        """Smoke profile should have minimal viable configuration."""
        config = EpisodeTrainingConfig(
            n_epoch_episodes=1,
            bar_sequences_per_episode=10,
            pair_width=3,
            min_bar_window=32,
            max_bar_window=64,
            epochs=1,
            learning_rate=1e-3,
        )

        # Verify minimal config is valid
        assert config.n_epoch_episodes >= 1
        assert config.bar_sequences_per_episode >= 10
        assert config.pair_width >= 3

    def test_smoke_profile_rapid_iteration_config(self):
        """Rapid iteration profile should use small episode count."""
        config = EpisodeTrainingConfig(
            n_epoch_episodes=5,
            bar_sequences_per_episode=20,
            pair_width=5,
            epochs=1,
            learning_rate=1e-3,
            max_training_seconds=300,  # 5 min max
        )

        # Should complete quickly
        assert config.n_epoch_episodes <= 10
        assert config.max_training_seconds <= 600

    def test_smoke_profile_objective_weights_default(self):
        """Smoke profile should use default objective weights."""
        config = EpisodeTrainingConfig()
        weights = objective_weight_config_from_config(config)

        # Default weights should be defined
        assert "world_model_weight" in weights
        assert "trade_head_weight" in weights
        assert weights["world_model_weight"] == 1.0
        assert weights["trade_head_weight"] == 1.0

    def test_smoke_profile_profit_oriented_weights(self):
        """Smoke profile should support profit-oriented objective weights."""
        config = EpisodeTrainingConfig(
            objective_world_model_weight=1.0,
            objective_trade_head_weight=2.0,
            objective_cost_turnover_weight=0.5,
            objective_regime_weight_scale=1.5,
        )
        weights = objective_weight_config_from_config(config)

        assert weights["trade_head_weight"] == 2.0
        assert weights["cost_turnover_weight"] == 0.5
        assert weights["regime_weight_scale"] == 1.5

    def test_smoke_profile_trade_step_scheduling(self):
        """Smoke profile should support trade-step scheduling controls."""
        config = EpisodeTrainingConfig(
            trade_step_schedule_mode="density_gated",
            trade_step_min_density=0.3,
            trade_step_schedule_interval=0,
        )
        weights = objective_weight_config_from_config(config)

        assert weights["trade_step_schedule_mode"] == "density_gated"
        assert weights["trade_step_min_density"] == 0.3

    def test_smoke_profile_train_val_test_splits(self):
        """Smoke profile should use standard ML data splits."""
        config = EpisodeTrainingConfig(
            train_split=0.7,
            val_split=0.15,
            test_split=0.15,
            split_mode="symbols",
        )

        # Verify splits sum to 1.0
        assert abs(config.train_split + config.val_split + config.test_split - 1.0) < 0.001

    def test_smoke_profile_reproducibility_seed(self):
        """Smoke profile should support reproducibility seed."""
        config = EpisodeTrainingConfig(
            random_seed=42,
            use_true_randomness=False,
        )

        assert config.random_seed == 42
        assert config.use_true_randomness is False


class TestTrainingArtifactCompatibility:
    """Tests for Freqtrade evaluation pipeline compatibility."""

    def test_checkpoint_contains_objective_telemetry(self, tmp_path, monkeypatch):
        """Training checkpoint should include objective telemetry for Freqtrade."""
        monkeypatch.chdir(tmp_path)

        # Import here to avoid MLX dependency in test
        from train import (
            build_episode_objective_telemetry,
            EpochEpisodeTrainer,
        )

        # Create minimal trainer stub
        trainer = EpochEpisodeTrainer.__new__(EpochEpisodeTrainer)
        trainer.config = EpisodeTrainingConfig(n_epoch_episodes=1)
        trainer.session_start_time = "2026-01-01T00:00:00"
        trainer.model_config = None
        trainer._train_symbols = []
        trainer._val_symbols = []
        trainer._test_symbols = []
        trainer.results = []
        trainer._save_hrm_artifacts = lambda out_dir, stem: {"saved": False}

        # Simulate episode with metrics
        episode_metrics = {
            "episode_id": 0,
            "symbols": ["BTCUSDT"],
            "final_capital": 100.0,
            "realized_pnl": 0.0,
            "total_trades": 10,
            "predictor_loss": 0.15,
            "trade_train_loss_mean": -0.05,
            "trade_train_eval_count": 5,
            "energy_train_loss_mean": 0.1,
            "energy_train_eval_count": 3,
            "outlier_extents": 0,
            "optimizer_replays": 0,
        }

        config = EpisodeTrainingConfig()
        telemetry = build_episode_objective_telemetry(episode_metrics, config)

        # Verify telemetry structure for Freqtrade
        assert "version" in telemetry
        assert "components" in telemetry
        assert "world_model_term" in telemetry["components"]
        assert "trade_head_term" in telemetry["components"]

    def test_training_results_preserve_objective_config(self, tmp_path, monkeypatch):
        """Training results should preserve objective configuration metadata."""
        monkeypatch.chdir(tmp_path)

        from train import EpochEpisodeTrainer

        trainer = EpochEpisodeTrainer.__new__(EpochEpisodeTrainer)
        config = EpisodeTrainingConfig(
            n_epoch_episodes=2,
            objective_trade_head_weight=2.0,
            objective_cost_turnover_weight=0.5,
            trade_step_schedule_mode="density_gated",
            trade_step_min_density=0.3,
        )
        trainer.config = config
        trainer.session_start_time = "2026-01-01T00:00:00"
        trainer.model_config = None
        trainer._train_symbols = []
        trainer._val_symbols = []
        trainer._test_symbols = []
        trainer.results = []
        trainer._save_hrm_artifacts = lambda out_dir, stem: {"saved": False}

        # Save final results
        trainer._save_final_results()

        # Verify results contain objective config
        import json
        results = json.loads((tmp_path / "training_results.json").read_text())

        assert "objective_weights" in results
        assert results["objective_weights"]["trade_head_weight"] == 2.0
        assert results["objective_weights"]["cost_turnover_weight"] == 0.5


class TestTrainingBaselineEvidence:
    """Tests for training baseline evidence capture."""

    def test_objective_weight_config_captures_all_terms(self):
        """Objective weight config should capture all profit-oriented terms."""
        config = EpisodeTrainingConfig(
            objective_world_model_weight=1.0,
            objective_trade_head_weight=1.5,
            objective_energy_routing_weight=0.5,
            objective_cost_turnover_weight=0.25,
            objective_regime_weight_scale=1.2,
        )

        weights = objective_weight_config_from_config(config)

        required_keys = [
            "world_model_weight",
            "trade_head_weight",
            "energy_routing_weight",
            "cost_turnover_weight",
            "regime_weight_scale",
            "trade_step_schedule_mode",
            "trade_step_min_density",
            "trade_step_schedule_interval",
        ]

        for key in required_keys:
            assert key in weights, f"Missing required key: {key}"

    def test_episode_telemetry_includes_regime_info(self):
        """Episode telemetry should include regime-related information."""
        from train import build_episode_objective_telemetry, EpisodeTrainingConfig

        episode_metrics = {
            "episode_id": 0,
            "outlier_extents": 5,
            "optimizer_replays": 2,
            "predictor_loss": 0.1,
            "trade_train_loss_mean": -0.02,
            "trade_train_eval_count": 10,
            "total_trades": 15,
        }

        config = EpisodeTrainingConfig(objective_regime_weight_scale=1.5)
        telemetry = build_episode_objective_telemetry(episode_metrics, config)

        # Should have regime weighting term
        assert "regime_weighting_term" in telemetry["components"]
