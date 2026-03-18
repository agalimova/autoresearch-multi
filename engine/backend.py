"""
Backend interface for autoresearch experiment engines.

A Backend wraps a specific experiment type (tabular ML, GPU training, etc.)
behind a common interface. The adaptive search, pruning, interaction detection,
and LLM proposer all operate on this interface without knowing what runs
underneath.

Usage:
    class MyBackend(Backend):
        def run_experiment(self, config): ...
        def get_search_space(self): ...
        ...

    coordinator = Coordinator(backend=MyBackend(...))
    coordinator.run()
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable


# ── Result ───────────────────────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    """Outcome of a single experiment run.

    Every backend returns this. The coordinator, adaptive search, pruning,
    and interaction detection all consume it.
    """
    config: dict[str, Any]          # backend-specific config that was tested
    score: float                    # primary metric (direction depends on backend)
    elapsed: float                  # wall-clock seconds
    error: Optional[str] = None     # non-None means the run crashed
    metadata: dict[str, Any] = field(default_factory=dict)  # extra info (tokens/s, loss curve, etc.)


# ── Shared helpers ───────────────────────────────────────────────────────────

def group_by_family(options: list[str]) -> dict[str, list[str]]:
    """Group option names by prefix before the first '_'."""
    groups: dict[str, list[str]] = {}
    for opt in options:
        groups.setdefault(opt.split("_")[0], []).append(opt)
    return groups


# ── Backend protocol ─────────────────────────────────────────────────────────

@runtime_checkable
class Backend(Protocol):
    """Interface that every experiment backend must implement.

    The coordinator calls these methods without knowing whether it's running
    sklearn grid search or a 5-minute GPU training loop.
    """

    @property
    def metric_name(self) -> str:
        """Name of the primary metric (e.g. 'val_acc', 'val_bpb')."""
        ...

    @property
    def higher_is_better(self) -> bool:
        """True if higher metric = better (accuracy). False if lower = better (bpb)."""
        ...

    def get_search_space(self) -> dict[str, list[str]]:
        """Return the search space as {dimension_name: [option_names]}.

        For tabular: {'engineer_features': ['base', 'pca', ...], 'build_model': ['lr', 'xgb', ...]}
        For GPU:     {'config': ['baseline', 'depth_12', 'lr_0.01', ...]}
        """
        ...

    def run_experiment(self, config: dict[str, str]) -> ExperimentResult:
        """Run one experiment with the given config and return the result.

        config maps dimension names to chosen options, matching get_search_space() keys.
        """
        ...

    def apply_proposal(self, dimension: str, name: str, code: str) -> Path:
        """Write an LLM-proposed variant into the search space.

        Returns the path of the written file. The next call to get_search_space()
        must include this new option.
        """
        ...

    def prompt_context(self) -> dict[str, Any]:
        """Return backend-specific context for the LLM proposer."""
        ...

    def get_base_config(self) -> dict[str, str]:
        """Return the baseline config (all defaults / 'base' values)."""
        ...

    def families(self, dimension: str) -> dict[str, list[str]]:
        """Group options in a dimension by family prefix."""
        ...

    def build_llm_prompt(
        self,
        tracker_summary: str,
        findings: str,
        target_dim: str,
    ) -> str:
        """Build a prompt for the LLM to propose a new variant.

        Default implementation works for most backends. Override for
        custom prompt formatting.
        """
        ...


# ── Experiment tracker ───────────────────────────────────────────────────────

@dataclass
class TrackedExperiment:
    """One row in the experiment history."""
    config: dict[str, str]
    score: float
    error: Optional[str]
    round_num: int
    timestamp: float = field(default_factory=time.time)
    config_diff: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ExperimentTracker:
    """Backend-agnostic experiment history with config diffs and pruning state.

    This is the 'results.tsv equivalent with richer metadata' that lets the
    agent see history, not just the last run. Persists to JSON after every
    experiment.
    """

    def __init__(self, path: Path, *, higher_is_better: bool = True):
        self.path = path
        self.higher_is_better = higher_is_better
        self.experiments: list[TrackedExperiment] = []
        self._tested: set[tuple] = set()

        # Pruning state
        self.dead_families: set[str] = set()
        self.dead_options: set[str] = set()
        self.crash_counts: dict[str, int] = {}

        # Interaction tracking: (option_a, option_b) -> delta
        self.interactions: dict[tuple[str, ...], float] = {}

        if path.exists():
            self._load()

    def is_better(self, a: float, b: float) -> bool:
        """True if score `a` is strictly better than score `b`."""
        return a > b if self.higher_is_better else a < b

    def best_of(self, scores: list[float]) -> float:
        return max(scores) if self.higher_is_better else min(scores)

    def worst_score(self) -> float:
        return 0.0 if self.higher_is_better else float("inf")

    def _config_key(self, config: dict[str, str]) -> tuple:
        return tuple(sorted(config.items()))

    def already_tested(self, config: dict[str, str]) -> bool:
        return self._config_key(config) in self._tested

    def record(
        self,
        result: ExperimentResult,
        *,
        round_num: int,
        base_config: dict[str, str],
    ) -> TrackedExperiment:
        """Record an experiment result. Saves immediately."""
        config_diff = {
            k: v for k, v in result.config.items()
            if base_config.get(k) != v
        }

        entry = TrackedExperiment(
            config=result.config,
            score=result.score,
            error=result.error,
            round_num=round_num,
            config_diff=config_diff,
            metadata=result.metadata,
        )
        self.experiments.append(entry)
        self._tested.add(self._config_key(result.config))

        # Track crashes
        if result.error:
            for option in result.config.values():
                if option != "base":
                    self.crash_counts[option] = self.crash_counts.get(option, 0) + 1
                    if self.crash_counts[option] >= 3:
                        self.dead_options.add(option)
        else:
            for option in result.config.values():
                if option != "base":
                    self.crash_counts[option] = 0

        self._save()
        return entry

    def is_dead(self, option: str) -> bool:
        """Check if an option is dead (by family, dominance, or crash)."""
        if option in self.dead_options:
            return True
        if self.crash_counts.get(option, 0) >= 3:
            return True
        family = option.split("_")[0]
        return family in self.dead_families

    def best(self) -> Optional[TrackedExperiment]:
        """Return the best experiment so far."""
        valid = [e for e in self.experiments if not e.error]
        if not valid:
            return None
        return max(valid, key=lambda e: e.score if self.higher_is_better else -e.score)

    def best_score(self) -> float:
        b = self.best()
        return b.score if b else self.worst_score()

    def valid_sorted(self) -> list[TrackedExperiment]:
        """All successful experiments, sorted best-first."""
        valid = [e for e in self.experiments if not e.error]
        valid.sort(key=lambda e: e.score, reverse=self.higher_is_better)
        return valid

    def _cutoff(self, reference: float, threshold: float) -> float:
        """Score that an option must beat to survive pruning."""
        return reference * threshold if self.higher_is_better else reference / threshold

    def _is_worse_than(self, score: float, cutoff: float) -> bool:
        """True if score falls below the pruning cutoff."""
        return score < cutoff if self.higher_is_better else score > cutoff

    def update_dead_families(
        self,
        backend: Backend,
        threshold: float = 0.95,
    ):
        """Mark dead families and dominated options.

        Family pruning: best-in-family doesn't meet threshold vs global best.
        Dominance pruning: option doesn't meet threshold vs best in its dimension.
        """
        best_score = self.best_score()
        if best_score == self.worst_score():
            return

        global_cutoff = self._cutoff(best_score, threshold)

        for dim, options in backend.get_search_space().items():
            # Family-level pruning
            for family, members in backend.families(dim).items():
                if family == "base":
                    continue
                best_in_family = self._best_score_for_options(members)
                if best_in_family is not None and self._is_worse_than(best_in_family, global_cutoff):
                    self.dead_families.add(family)

            # Dominance pruning within dimension
            option_bests = {
                opt: score
                for opt in options
                if (score := self._best_score_for_options([opt])) is not None
            }
            if len(option_bests) < 2:
                continue
            dim_best = self.best_of(list(option_bests.values()))
            dim_cutoff = self._cutoff(dim_best, threshold)

            for opt, score in option_bests.items():
                if opt != "base" and self._is_worse_than(score, dim_cutoff):
                    self.dead_options.add(opt)

        self._save()

    def update_interactions(self, base_config: dict[str, str]):
        """Compute interaction deltas for all multi-dimension configs tested.

        delta = actual - predicted, where predicted = base + sum(solo improvements).
        Positive delta = superadditive.
        """
        valid = {
            self._config_key(e.config): e.score
            for e in self.experiments if not e.error
        }
        if not valid:
            return

        base_key = self._config_key(base_config)
        if base_key not in valid:
            return
        base_score = valid[base_key]

        for config_key, actual in valid.items():
            config = dict(config_key)
            non_base = {d: v for d, v in config.items() if base_config.get(d) != v}
            if len(non_base) < 2:
                continue

            predicted = base_score
            for dim, opt in non_base.items():
                solo = dict(base_config)
                solo[dim] = opt
                solo_score = valid.get(self._config_key(solo), base_score)
                predicted += (solo_score - base_score)

            delta = actual - predicted
            pair = tuple(sorted(non_base.values()))
            self.interactions[pair] = delta

    def _best_score_for_options(self, options: list[str]) -> Optional[float]:
        """Best score across all experiments that used any of the given options."""
        option_set = set(options)
        scores = [
            e.score for e in self.experiments
            if not e.error and any(v in option_set for v in e.config.values())
        ]
        return self.best_of(scores) if scores else None

    def scores_by_option(self, dim: str) -> dict[str, float]:
        """Best score per option in a dimension."""
        result: dict[str, float] = {}
        for e in self.experiments:
            if e.error:
                continue
            opt = e.config.get(dim, "base")
            prev = result.get(opt)
            if prev is None or self.is_better(e.score, prev):
                result[opt] = e.score
        return result

    def best_in_dim(self, dim: str) -> str:
        """Best-performing option in a dimension."""
        scores = self.scores_by_option(dim)
        if not scores:
            return "base"
        return max(scores, key=scores.get) if self.higher_is_better else min(scores, key=scores.get)  # type: ignore[arg-type]

    def near_misses(self, var_dims: list[str]) -> dict[str, list[str]]:
        """Options within 1% of best per dimension."""
        near: dict[str, list[str]] = {}
        for dim in var_dims:
            scores = self.scores_by_option(dim)
            if not scores:
                continue
            best_val = self.best_of(list(scores.values()))
            threshold = best_val * (0.99 if self.higher_is_better else 1.01)
            misses = [o for o, v in scores.items()
                      if v != best_val and not self._is_worse_than(v, threshold)]
            if misses:
                near[dim] = misses
        return near

    def summary_for_prompt(self, *, max_entries: int = 15) -> str:
        """Compact text summary for LLM context injection."""
        lines = []
        valid = self.valid_sorted()
        lines.append(f"Experiments: {len(valid)} successful, {len(self.experiments) - len(valid)} failed")

        if valid:
            lines.append(f"\nTop results:")
            for e in valid[:max_entries]:
                diff_str = ", ".join(f"{k}={v}" for k, v in e.config_diff.items()) or "(baseline)"
                lines.append(f"  {e.score:.6f} | {diff_str}")

        if self.dead_families:
            lines.append(f"\nDead families: {', '.join(sorted(self.dead_families))}")
        if self.dead_options - self.dead_families:
            lines.append(f"Dead options: {', '.join(sorted(self.dead_options - self.dead_families))}")

        if self.interactions:
            top_ix = sorted(self.interactions.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            lines.append(f"\nInteraction effects:")
            for pair, delta in top_ix:
                tag = "superadditive" if delta > 0 else "subadditive"
                lines.append(f"  {'+'.join(pair)}: {delta:+.4f} ({tag})")

        return "\n".join(lines)

    # ── Persistence ──────────────────────────────────────────────────────

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "experiments": [asdict(e) for e in self.experiments],
            "dead_families": sorted(self.dead_families),
            "dead_options": sorted(self.dead_options),
            "crash_counts": self.crash_counts,
            "interactions": {
                "|".join(k): v for k, v in self.interactions.items()
            },
        }
        self.path.write_text(json.dumps(state, indent=2))

    def _load(self):
        raw = json.loads(self.path.read_text())
        self.experiments = [
            TrackedExperiment(**e) for e in raw.get("experiments", [])
        ]
        self._tested = {
            self._config_key(e.config) for e in self.experiments
        }
        self.dead_families = set(raw.get("dead_families", []))
        self.dead_options = set(raw.get("dead_options", []))
        self.crash_counts = raw.get("crash_counts", {})
        self.interactions = {
            tuple(k.split("|")): v
            for k, v in raw.get("interactions", {}).items()
        }
