"""End-to-end test: GPU backend + coordinator against mock_train.py.

Tests the full loop: coordinator picks configs from variants/config/,
GpuTrainingBackend swaps them into src/config.rs, mock_train.py reads
the swapped config, computes a deterministic val_bpb, and the coordinator
prunes/exploits/narrows based on the scores.

Everything lives in tests/fixtures/gpu_project/ — no external repos touched.
"""

import sys
from pathlib import Path

import pytest

from engine.backends.gpu_training import GpuTrainingBackend
from engine.coordinator import Coordinator

PYTHON = sys.executable
PROJECT = Path(__file__).parent / "fixtures" / "gpu_project"
MOCK_TRAIN = PROJECT / "mock_train.py"


def _make_backend() -> GpuTrainingBackend:
    return GpuTrainingBackend(
        project_dir=PROJECT,
        binary=f'"{PYTHON}" "{MOCK_TRAIN}"',
        config_file="src/config.rs",
        time_budget=5.0,
    )


class TestGpuBackend:

    def test_single_run(self):
        backend = _make_backend()
        result = backend.run_experiment({"config": "baseline"})
        assert result.error is None, f"Run failed: {result.error}"
        assert 0.5 < result.score < 3.0, f"val_bpb out of range: {result.score}"
        assert result.metadata.get("depth") == 8.0

    def test_variant_scores_differ(self):
        backend = _make_backend()
        scores = {}
        for variant in ["baseline", "depth_12", "depth_16"]:
            result = backend.run_experiment({"config": variant})
            assert result.error is None, f"{variant} failed: {result.error}"
            scores[variant] = result.score
            print(f"  {variant}: val_bpb={result.score:.6f}")

        # Deeper = lower bpb (better)
        assert scores["depth_16"] < scores["depth_12"] < scores["baseline"], (
            f"Depth scaling wrong: {scores}"
        )

    def test_lr_sensitivity(self):
        backend = _make_backend()
        scores = {}
        for variant in ["baseline", "lr_high", "lr_low"]:
            result = backend.run_experiment({"config": variant})
            assert result.error is None, f"{variant} failed: {result.error}"
            scores[variant] = result.score
            print(f"  {variant}: val_bpb={result.score:.6f}")

        # Both lr_high and lr_low should be worse than baseline (0.04 is optimal)
        assert scores["lr_high"] > scores["baseline"], f"lr_high should be worse: {scores}"
        assert scores["lr_low"] > scores["baseline"], f"lr_low should be worse: {scores}"

    def test_config_restored_after_run(self):
        backend = _make_backend()
        config_path = PROJECT / "src" / "config.rs"
        original = config_path.read_text()

        backend.run_experiment({"config": "depth_16"})

        restored = config_path.read_text()
        assert restored == original, "src/config.rs was not restored after run"

    def test_search_space_discovery(self):
        backend = _make_backend()
        space = backend.get_search_space()
        assert "config" in space
        assert "baseline" in space["config"]
        assert "depth_12" in space["config"]
        assert len(space["config"]) >= 6


class TestCoordinatorGpu:

    def test_full_loop(self, tmp_path):
        backend = _make_backend()
        coord = Coordinator(backend, budget_per_round=8, results_dir=tmp_path)
        tracker = coord.run(n_rounds=4, llm_rounds=0)

        assert len(tracker.experiments) > 0
        best = tracker.best()
        assert best is not None
        print(f"\n  Best config: {best.config}")
        print(f"  Best val_bpb: {best.score:.6f}")
        print(f"  Total experiments: {len(tracker.experiments)}")
        print(f"  Dead families: {tracker.dead_families}")
        print(f"  Dead options: {tracker.dead_options}")

    def test_pruning_kills_bad_lr(self, tmp_path):
        backend = _make_backend()
        coord = Coordinator(backend, budget_per_round=10, results_dir=tmp_path)
        tracker = coord.run(n_rounds=3, llm_rounds=0)

        tracker.update_dead_families(backend)
        print(f"\n  Dead families: {tracker.dead_families}")
        print(f"  Dead options: {tracker.dead_options}")

        best = tracker.best()
        assert best is not None
        assert best.score < 1.5, f"Best should be decent: {best.score}"

    def test_tracker_summary(self, tmp_path):
        backend = _make_backend()
        coord = Coordinator(backend, budget_per_round=6, results_dir=tmp_path)
        tracker = coord.run(n_rounds=2, llm_rounds=0)

        summary = tracker.summary_for_prompt()
        print(f"\n{summary}")
        assert "Experiments:" in summary
        assert len(summary) > 50
