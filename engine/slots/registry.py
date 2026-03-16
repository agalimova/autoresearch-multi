"""
Entity registry: tracks per-slot version history with pruning.

Pruning rules:
  1. Dominance:  if A's best combo < B's solo, A is archived
  2. Staleness:  submissions not in winning combo for N cycles are archived
  3. Compounds:  multi-slot changes only stay if interaction_delta > threshold

Features integrated from other autoresearch extensions:
  - Near-miss tracking (n-autoresearch, Apache-2.0)
  - Darwinian weighting (atlas-gic, MIT)
  - Crash tracking (n-autoresearch, Apache-2.0)

The summary view returns only active + top N archived for prompt injection.
Full history is always available for claudemem search.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Submission:
    """One submission to a slot."""
    impl_name: str
    agent: str
    metric: float                              # solo metric (paired with base for all other slots)
    timestamp: float = field(default_factory=time.time)
    active: bool = True
    best_combo_metric: Optional[float] = None  # best metric in any combination
    cycles_since_win: int = 0                  # how many combo rounds since this was in the best
    # Near-miss tracking (from n-autoresearch)
    near_miss: bool = False                    # within threshold of best
    # Darwinian weighting (from atlas-gic)
    weight: float = 1.0                        # 0.3 to 2.5, updated after each sweep
    # Crash tracking (from n-autoresearch)
    consecutive_crashes: int = 0               # abort after 3


@dataclass
class SlotState:
    """All submissions for one slot."""
    slot_name: str
    submissions: list[Submission] = field(default_factory=list)

    @property
    def active(self) -> list[Submission]:
        return [s for s in self.submissions if s.active]

    @property
    def archived(self) -> list[Submission]:
        return [s for s in self.submissions if not s.active]

    @property
    def current_best(self) -> Optional[Submission]:
        active = self.active
        return max(active, key=lambda s: s.metric) if active else None


@dataclass
class Compound:
    """A multi-slot change that must be used as a unit."""
    slots: dict[str, str]           # slot_name -> impl_name
    metric: float                   # combo metric
    interaction_delta: float        # actual - predicted (positive = superadditive)
    agent: str
    timestamp: float = field(default_factory=time.time)
    active: bool = True


class Registry:
    """Persistent per-slot version tracker with pruning."""

    def __init__(self, path: Path):
        self.path = path
        self._slots: dict[str, SlotState] = {}
        self._compounds: list[Compound] = []
        if path.exists():
            self._load()

    def _load(self):
        raw = json.loads(self.path.read_text())
        for name, data in raw.get("slots", raw).items():
            if name == "_compounds":
                continue
            subs = [Submission(**s) for s in data.get("submissions", [])]
            self._slots[name] = SlotState(slot_name=name, submissions=subs)
        for c in raw.get("_compounds", []):
            self._compounds.append(Compound(**c))

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw = {}
        for name, state in self._slots.items():
            raw[name] = {
                "slot_name": name,
                "submissions": [asdict(s) for s in state.submissions],
            }
        raw["_compounds"] = [asdict(c) for c in self._compounds]
        self.path.write_text(json.dumps(raw, indent=2))

    # ── Recording ────────────────────────────────────────────────────────

    def record(
        self,
        slot_name: str,
        impl_name: str,
        *,
        agent: str,
        metric: float,
        best_combo_metric: Optional[float] = None,
    ):
        """Record a submission. Saves immediately."""
        if slot_name not in self._slots:
            self._slots[slot_name] = SlotState(slot_name=slot_name)

        state = self._slots[slot_name]
        for s in state.submissions:
            if s.impl_name == impl_name:
                s.metric = metric
                if best_combo_metric is not None:
                    s.best_combo_metric = best_combo_metric
                self._save()
                return

        state.submissions.append(Submission(
            impl_name=impl_name, agent=agent, metric=metric,
            best_combo_metric=best_combo_metric,
        ))
        self._save()

    # ── Near-miss tracking (from n-autoresearch) ────────────────────────

    def update_near_misses(self, threshold: float = 0.01):
        """Mark submissions within threshold of the best as near-misses."""
        for state in self._slots.values():
            best = state.current_best
            if not best:
                continue
            for s in state.active:
                gap = abs(best.metric - s.metric)
                s.near_miss = 0 < gap <= threshold
        self._save()

    def get_near_misses(self, slot_name: str) -> list[Submission]:
        """Get near-miss submissions for a slot (candidates for combination)."""
        state = self._slots.get(slot_name)
        if not state:
            return []
        return [s for s in state.active if s.near_miss]

    # ── Darwinian weighting (from atlas-gic) ──────────────────────────

    def update_weights(self, combo_results: list, *, metric_name: str = "val_acc"):
        """
        Adjust submission weights based on combo performance.
        Top quartile: weight * 1.05. Bottom quartile: weight * 0.95.
        Clamped to [0.3, 2.5].
        """
        valid = [r for r in combo_results if not r.error]
        if len(valid) < 4:
            return

        # Score each submission by its average combo performance
        sub_scores: dict[str, dict[str, list[float]]] = {}
        for r in valid:
            m = r.metrics.get(metric_name, 0)
            for slot, impl in r.combo.items():
                sub_scores.setdefault(slot, {}).setdefault(impl, []).append(m)

        for slot_name, impl_scores in sub_scores.items():
            state = self._slots.get(slot_name)
            if not state:
                continue

            # Rank by mean score
            ranked = sorted(impl_scores.items(), key=lambda x: -sum(x[1]) / len(x[1]))
            n = len(ranked)
            top_quartile = {name for name, _ in ranked[: max(1, n // 4)]}
            bot_quartile = {name for name, _ in ranked[max(1, n - n // 4) :]}

            for s in state.submissions:
                if s.impl_name in top_quartile:
                    s.weight = min(2.5, s.weight * 1.05)
                elif s.impl_name in bot_quartile:
                    s.weight = max(0.3, s.weight * 0.95)

        self._save()

    # ── Crash tracking (from n-autoresearch) ──────────────────────────

    def record_crash(self, slot_name: str, impl_name: str) -> int:
        """Record a crash for a submission. Returns consecutive crash count."""
        state = self._slots.get(slot_name)
        if not state:
            return 0
        for s in state.submissions:
            if s.impl_name == impl_name:
                s.consecutive_crashes += 1
                if s.consecutive_crashes >= 3:
                    s.active = False  # auto-archive after 3 crashes
                self._save()
                return s.consecutive_crashes
        return 0

    def clear_crashes(self, slot_name: str, impl_name: str):
        """Clear crash count on successful run."""
        state = self._slots.get(slot_name)
        if not state:
            return
        for s in state.submissions:
            if s.impl_name == impl_name:
                s.consecutive_crashes = 0
                self._save()
                return

    # ── Pruning ──────────────────────────────────────────────────────────

    def prune(self, *, stale_cycles: int = 5):
        """Run all pruning rules. Call after each combo round."""
        for name in list(self._slots):
            self._prune_dominated(name)
            self._prune_stale(name, stale_cycles)
        self._prune_compounds()
        self._save()

    def _prune_dominated(self, slot_name: str):
        """Archive submissions whose best combo < any other's solo metric."""
        state = self._slots.get(slot_name)
        if not state or len(state.active) <= 1:
            return

        # Best solo metric among all active submissions
        best_solo = max(s.metric for s in state.active)

        for s in state.active:
            if s.impl_name == "base":
                continue  # never prune base
            if s.best_combo_metric is not None and s.best_combo_metric < best_solo:
                s.active = False

    def _prune_stale(self, slot_name: str, max_cycles: int):
        """Archive submissions that haven't been in a winning combo for N cycles."""
        state = self._slots.get(slot_name)
        if not state:
            return

        for s in state.active:
            if s.impl_name == "base":
                continue
            if s.cycles_since_win >= max_cycles:
                s.active = False

    def _prune_compounds(self, threshold: float = 0.005):
        """Decompose compounds with interaction_delta below threshold."""
        self._compounds = [
            c for c in self._compounds
            if c.active and c.interaction_delta > threshold
        ]

    # ── Combo results ────────────────────────────────────────────────────

    def update_from_combo_results(
        self,
        combo_results: list,
        *,
        metric_name: str = "val_acc",
    ):
        """
        After running all combos, update each submission's best_combo_metric
        and cycles_since_win. Also detect compounds with positive interaction.
        """
        # Find the overall best combo
        valid = [r for r in combo_results if not r.error]
        if not valid:
            return

        best_result = max(valid, key=lambda r: r.metrics.get(metric_name, 0.0))
        best_metric = best_result.metrics.get(metric_name, 0.0)
        best_combo = best_result.combo

        # Update per-submission best combo metric
        best_combos: dict[str, dict[str, float]] = {}
        for r in valid:
            m = r.metrics.get(metric_name, 0.0)
            for slot_name, impl_name in r.combo.items():
                best_combos.setdefault(slot_name, {})
                current = best_combos[slot_name].get(impl_name, 0.0)
                if m > current:
                    best_combos[slot_name][impl_name] = m

        for slot_name, impl_metrics in best_combos.items():
            state = self._slots.get(slot_name)
            if not state:
                continue
            for s in state.submissions:
                if s.impl_name in impl_metrics:
                    s.best_combo_metric = impl_metrics[s.impl_name]

        # Update cycles_since_win
        for state in self._slots.values():
            for s in state.submissions:
                if best_combo.get(state.slot_name) == s.impl_name:
                    s.cycles_since_win = 0
                else:
                    s.cycles_since_win += 1

        # Detect interaction effects in the best combo
        self._check_interactions(combo_results, best_combo, metric_name)

        self._save()

    def _check_interactions(
        self,
        combo_results: list,
        best_combo: dict[str, str],
        metric_name: str,
    ):
        """
        Check if the best combo has superadditive interactions.

        Predicted = base + sum of individual improvements.
        Actual = measured combo metric.
        interaction_delta = actual - predicted.
        """
        valid = {
            tuple(sorted(r.combo.items())): r.metrics.get(metric_name, 0.0)
            for r in combo_results if not r.error
        }

        # Find base metric (all slots = "base")
        base_key = tuple(sorted((s, "base") for s in best_combo))
        base_metric = valid.get(base_key, 0.0)

        if base_metric == 0.0:
            return

        # Sum of individual improvements
        non_base_slots = {s: v for s, v in best_combo.items() if v != "base"}
        if len(non_base_slots) < 2:
            return  # need at least 2 non-base slots for interaction

        predicted = base_metric
        for slot_name, impl_name in non_base_slots.items():
            # Solo key: this slot changed, all others base
            solo_combo = {s: "base" for s in best_combo}
            solo_combo[slot_name] = impl_name
            solo_key = tuple(sorted(solo_combo.items()))
            solo_metric = valid.get(solo_key, base_metric)
            predicted += (solo_metric - base_metric)

        actual_key = tuple(sorted(best_combo.items()))
        actual = valid.get(actual_key, 0.0)

        interaction_delta = actual - predicted

        if abs(interaction_delta) > 0.001:
            print(f"  Interaction detected in best combo:")
            print(f"    Base:        {base_metric:.4f}")
            print(f"    Predicted:   {predicted:.4f} (additive)")
            print(f"    Actual:      {actual:.4f}")
            print(f"    Delta:       {interaction_delta:+.4f} "
                  f"({'superadditive' if interaction_delta > 0 else 'subadditive'})")

    # ── Views ────────────────────────────────────────────────────────────

    def summary(self, *, top_archived: int = 2) -> str:
        """Compact view for agent prompts."""
        lines = []
        for name, state in sorted(self._slots.items()):
            best = state.current_best
            best_str = f" (best: {best.impl_name} {best.metric:.4f})" if best else ""
            lines.append(f"[{name}]{best_str}")

            for s in state.active:
                marker = " *" if best and s.impl_name == best.impl_name else ""
                combo = f" combo={s.best_combo_metric:.4f}" if s.best_combo_metric else ""
                stale = f" stale={s.cycles_since_win}" if s.cycles_since_win > 0 else ""
                lines.append(
                    f"  {s.impl_name} by {s.agent}: "
                    f"{s.metric:.4f}{combo}{stale}{marker}"
                )

            top = sorted(state.archived, key=lambda s: s.metric, reverse=True)[:top_archived]
            for s in top:
                lines.append(f"  {s.impl_name} by {s.agent}: {s.metric:.4f} (archived)")

        if self._compounds:
            lines.append("\n[compounds]")
            for c in self._compounds:
                slots = ", ".join(f"{s}={v}" for s, v in c.slots.items())
                lines.append(
                    f"  {slots} by {c.agent}: {c.metric:.4f} "
                    f"delta={c.interaction_delta:+.4f}"
                )

        return "\n".join(lines)

    def get_state(self, slot_name: str) -> Optional[SlotState]:
        return self._slots.get(slot_name)

    def all_slots(self) -> dict[str, SlotState]:
        return dict(self._slots)

    def active_count(self) -> dict[str, int]:
        """Number of active submissions per slot."""
        return {name: len(state.active) for name, state in self._slots.items()}

    def total_combos(self) -> int:
        """How many combinations the exhaustive engine needs to test."""
        counts = self.active_count()
        if not counts:
            return 0
        result = 1
        for c in counts.values():
            result *= max(c, 1)
        return result
