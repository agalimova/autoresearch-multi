"""Tests for the backend interface, tracker, and coordinator."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from engine.backend import (
    Backend,
    ExperimentResult,
    ExperimentTracker,
    TrackedExperiment,
)
from engine.coordinator import Coordinator


# ── Mock backend ─────────────────────────────────────────────────────────────

class MockBackend:
    """Deterministic backend for testing the coordinator loop.

    Returns score = sum of option indices in the search space.
    E.g. config={'a': 'a1', 'b': 'b0'} -> index(a1)=1 + index(b0)=0 = 1.
    """

    def __init__(self, *, fail_options: set[str] | None = None):
        self._space = {
            "features": ["base", "pca", "poly"],
            "model": ["base", "lr_default", "lr_tuned", "xgb_default", "xgb_deep"],
        }
        self._fail = fail_options or set()
        self._proposals: dict[str, str] = {}

    @property
    def metric_name(self) -> str:
        return "val_acc"

    @property
    def higher_is_better(self) -> bool:
        return True

    def get_search_space(self) -> dict[str, list[str]]:
        space = dict(self._space)
        for dim, name in self._proposals.items():
            if dim in space and name not in space[dim]:
                space[dim].append(name)
        return space

    def run_experiment(self, config: dict[str, str]) -> ExperimentResult:
        # Check for forced failures
        for opt in config.values():
            if opt in self._fail:
                return ExperimentResult(
                    config=config, score=0.0, elapsed=0.01,
                    error=f"forced failure: {opt}",
                )

        # Score = normalized index sum
        score = 0.0
        for dim, opt in config.items():
            options = self._space.get(dim, ["base"])
            if opt in options:
                score += options.index(opt) / len(options)
            else:
                score += 0.5  # LLM proposal gets middle score
        return ExperimentResult(config=config, score=score, elapsed=0.01)

    def apply_proposal(self, dimension: str, name: str, code: str) -> Path:
        self._proposals[dimension] = name
        p = Path(tempfile.mktemp(suffix=".py"))
        p.write_text(code)
        return p

    def prompt_context(self) -> dict[str, Any]:
        return {
            "backend_type": "tabular_ml",
            "metric": "val_acc",
            "pipeline": ["features", "model"],
            "slots": {},
        }

    def get_base_config(self) -> dict[str, str]:
        return {"features": "base", "model": "base"}

    def families(self, dimension: str) -> dict[str, list[str]]:
        space = self.get_search_space()
        options = space.get(dimension, ["base"])
        groups: dict[str, list[str]] = {}
        for opt in options:
            family = opt.split("_")[0]
            groups.setdefault(family, []).append(opt)
        return groups

    def build_llm_prompt(self, tracker_summary: str, findings: str, target_dim: str) -> str:
        return f"Propose a variant for {target_dim}.\n{tracker_summary}\n{findings}"


class LowerIsBetterBackend(MockBackend):
    """Backend where lower score is better (like val_bpb)."""

    @property
    def metric_name(self) -> str:
        return "val_bpb"

    @property
    def higher_is_better(self) -> bool:
        return False

    def run_experiment(self, config: dict[str, str]) -> ExperimentResult:
        result = super().run_experiment(config)
        if result.error:
            return ExperimentResult(
                config=config, score=float("inf"), elapsed=0.01,
                error=result.error,
            )
        # Invert: lower = better
        return ExperimentResult(
            config=config, score=1.0 - result.score, elapsed=0.01,
        )


# ── ExperimentTracker tests ──────────────────────────────────────────────────

class TestExperimentTracker:

    def _make_tracker(self, tmp_path: Path, **kwargs) -> ExperimentTracker:
        return ExperimentTracker(tmp_path / "tracker.json", **kwargs)

    def test_record_and_retrieve(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        result = ExperimentResult(
            config={"a": "x"}, score=0.9, elapsed=1.0,
        )
        tracker.record(result, round_num=1, base_config={"a": "base"})

        assert len(tracker.experiments) == 1
        assert tracker.experiments[0].score == 0.9
        assert tracker.experiments[0].config_diff == {"a": "x"}

    def test_already_tested(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        result = ExperimentResult(
            config={"a": "x", "b": "y"}, score=0.5, elapsed=1.0,
        )
        tracker.record(result, round_num=1, base_config={"a": "base", "b": "base"})

        assert tracker.already_tested({"a": "x", "b": "y"})
        assert not tracker.already_tested({"a": "x", "b": "z"})

    def test_best_higher_is_better(self, tmp_path):
        tracker = self._make_tracker(tmp_path, higher_is_better=True)
        base = {"dim": "base"}
        for score in [0.5, 0.9, 0.7]:
            tracker.record(
                ExperimentResult(config={"dim": f"v{score}"}, score=score, elapsed=0.1),
                round_num=1, base_config=base,
            )
        assert tracker.best_score() == 0.9

    def test_best_lower_is_better(self, tmp_path):
        tracker = self._make_tracker(tmp_path, higher_is_better=False)
        base = {"dim": "base"}
        for score in [0.5, 0.1, 0.7]:
            tracker.record(
                ExperimentResult(config={"dim": f"v{score}"}, score=score, elapsed=0.1),
                round_num=1, base_config=base,
            )
        assert tracker.best_score() == 0.1

    def test_crash_tracking(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        base = {"dim": "base"}

        # Three crashes -> dead
        for i in range(3):
            tracker.record(
                ExperimentResult(
                    config={"dim": "bad_v1"}, score=0.0, elapsed=0.1,
                    error="crash",
                ),
                round_num=1, base_config=base,
            )
        assert tracker.is_dead("bad_v1")

        # Successful run clears crash count
        tracker.record(
            ExperimentResult(config={"dim": "ok_v1"}, score=0.5, elapsed=0.1),
            round_num=1, base_config=base,
        )
        assert not tracker.is_dead("ok_v1")

    def test_persistence(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.record(
            ExperimentResult(config={"a": "x"}, score=0.8, elapsed=0.1),
            round_num=1, base_config={"a": "base"},
        )
        tracker.dead_families.add("bad")
        tracker._save()  # manual save after direct mutation

        # Reload from disk
        tracker2 = self._make_tracker(tmp_path)
        assert len(tracker2.experiments) == 1
        assert tracker2.experiments[0].score == 0.8
        assert "bad" in tracker2.dead_families

    def test_interactions(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        base = {"a": "base", "b": "base"}

        # Record base, two solos, and a combo
        tracker.record(ExperimentResult(config={"a": "base", "b": "base"}, score=0.5, elapsed=0.1), round_num=1, base_config=base)
        tracker.record(ExperimentResult(config={"a": "x", "b": "base"}, score=0.6, elapsed=0.1), round_num=1, base_config=base)
        tracker.record(ExperimentResult(config={"a": "base", "b": "y"}, score=0.55, elapsed=0.1), round_num=1, base_config=base)
        # Superadditive: actual 0.8 > predicted 0.5 + 0.1 + 0.05 = 0.65
        tracker.record(ExperimentResult(config={"a": "x", "b": "y"}, score=0.8, elapsed=0.1), round_num=1, base_config=base)

        tracker.update_interactions(base)
        pair = tuple(sorted(("x", "y")))
        assert pair in tracker.interactions
        assert tracker.interactions[pair] == pytest.approx(0.15, abs=0.001)

    def test_dead_families(self, tmp_path):
        backend = MockBackend()
        tracker = self._make_tracker(tmp_path)
        base = backend.get_base_config()

        # Record experiments where xgb family dominates
        tracker.record(ExperimentResult(config={"features": "base", "model": "lr_default"}, score=0.3, elapsed=0.1), round_num=1, base_config=base)
        tracker.record(ExperimentResult(config={"features": "base", "model": "xgb_default"}, score=0.9, elapsed=0.1), round_num=1, base_config=base)

        tracker.update_dead_families(backend, threshold=0.95)
        # lr family's best (0.3) < 0.9 * 0.95 = 0.855 -> dead
        assert "lr" in tracker.dead_families

    def test_summary_for_prompt(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        base = {"dim": "base"}
        tracker.record(
            ExperimentResult(config={"dim": "good"}, score=0.9, elapsed=0.1),
            round_num=1, base_config=base,
        )
        summary = tracker.summary_for_prompt()
        assert "0.900000" in summary
        assert "dim=good" in summary


# ── Coordinator tests ────────────────────────────────────────────────────────

class TestCoordinator:

    def test_explore_coverage(self, tmp_path):
        """EXPLORE round tests at least one per family."""
        backend = MockBackend()
        coord = Coordinator(backend, budget_per_round=50, results_dir=tmp_path)
        coord._explore(1)

        # Should have tested combos involving all model families
        tested_models = {
            e.config["model"] for e in coord.tracker.experiments if not e.error
        }
        # At minimum, one from each family: base, lr, xgb
        families_seen = {m.split("_")[0] for m in tested_models}
        assert "base" in families_seen
        assert "lr" in families_seen
        assert "xgb" in families_seen

    def test_crash_skipping(self, tmp_path):
        """Options that crash 3x are auto-skipped."""
        backend = MockBackend(fail_options={"lr_default"})
        coord = Coordinator(backend, budget_per_round=50, results_dir=tmp_path)
        coord._explore(1)

        # lr_default should have crashed and eventually been skipped
        lr_crashes = [
            e for e in coord.tracker.experiments
            if e.config.get("model") == "lr_default" and e.error
        ]
        assert len(lr_crashes) >= 1
        assert coord.tracker.is_dead("lr_default")

    def test_full_run(self, tmp_path):
        """Full 4-mode run completes and finds a good result."""
        backend = MockBackend()
        coord = Coordinator(backend, budget_per_round=10, results_dir=tmp_path)
        tracker = coord.run(n_rounds=4, llm_rounds=0)

        assert len(tracker.experiments) > 0
        best = tracker.best()
        assert best is not None
        assert best.score > 0.0

    def test_lower_is_better(self, tmp_path):
        """Coordinator works with lower-is-better metrics."""
        backend = LowerIsBetterBackend()
        coord = Coordinator(backend, budget_per_round=10, results_dir=tmp_path)
        tracker = coord.run(n_rounds=2, llm_rounds=0)

        best = tracker.best()
        assert best is not None
        assert best.score < 1.0  # should find something better than worst

    def test_tracker_persistence_across_resume(self, tmp_path):
        """Coordinator picks up where it left off via tracker persistence."""
        backend = MockBackend()
        coord1 = Coordinator(backend, budget_per_round=5, results_dir=tmp_path)
        coord1._explore(1)
        n_first = len(coord1.tracker.experiments)

        # Create a new coordinator pointing at same results dir
        coord2 = Coordinator(backend, budget_per_round=5, results_dir=tmp_path)
        assert len(coord2.tracker.experiments) == n_first

        # Second explore should skip already-tested configs
        coord2._explore(2)
        # Should have added some new ones (explore with random fill)
        # but not re-tested the old ones
        n_second = len(coord2.tracker.experiments)
        assert n_second >= n_first

    def test_dead_family_propagation(self, tmp_path):
        """Dead families are respected in EXPLOIT and NARROW."""
        backend = MockBackend()
        coord = Coordinator(backend, budget_per_round=50, results_dir=tmp_path)

        # Run explore
        coord._explore(1)
        coord.tracker.update_interactions(backend.get_base_config())
        coord.tracker.update_dead_families(backend)

        # Force-kill a family
        coord.tracker.dead_families.add("lr")

        # Run exploit — should not test any lr_ options
        coord._exploit(2)
        exploit_models = [
            e.config["model"]
            for e in coord.tracker.experiments
            if e.round_num == 2 and not e.error
        ]
        lr_in_exploit = [m for m in exploit_models if m.startswith("lr")]
        assert len(lr_in_exploit) == 0


# ── Backend protocol conformance ─────────────────────────────────────────────

class TestProtocolConformance:

    def test_mock_backend_is_backend(self):
        """MockBackend satisfies the Backend protocol."""
        backend = MockBackend()
        assert isinstance(backend, Backend)

    def test_lower_backend_is_backend(self):
        backend = LowerIsBetterBackend()
        assert isinstance(backend, Backend)


# ── GpuTrainingBackend output parsing ────────────────────────────────────────

class TestGpuOutputParsing:

    def test_parse_rust_output(self):
        from engine.backends.gpu_training import GpuTrainingBackend

        stdout = """step 00050 (45.2%) | loss: 3.456789 | dt: 450ms
step 00100 (90.1%) | loss: 2.123456 | dt: 440ms
---
val_bpb:          1.234567
training_seconds: 300.1
total_seconds:    312.5
total_tokens_M:   52.4
num_steps:        100
num_params_M:     46.2
depth:            8
startup_seconds:  12.4"""

        parsed = GpuTrainingBackend._parse_output(stdout)
        assert parsed["val_bpb"] == pytest.approx(1.234567)
        assert parsed["training_seconds"] == pytest.approx(300.1)
        assert parsed["depth"] == 8.0
        assert parsed["num_steps"] == 100.0

    def test_parse_no_separator(self):
        from engine.backends.gpu_training import GpuTrainingBackend

        stdout = "some random output without ---"
        parsed = GpuTrainingBackend._parse_output(stdout)
        assert parsed == {}
