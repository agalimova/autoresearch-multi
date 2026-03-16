"""
Adaptive grid search with coverage guarantee.

Round 1 (coverage): one representative per family x every feature variant.
         Guarantees no family-level interaction is missed.
Round 1 (explore):  fill remaining budget with random combos.
Round 2 (exploit):  drill into the winning family+feature region.
Round 3 (narrow):   fine-tune within the winning region.

Usage:
    from engine.adaptive import AdaptiveSearch
    search = AdaptiveSearch(runner, budget_per_round=20)
    results = search.run(n_rounds=3)
"""

from __future__ import annotations

import itertools
import json
import random
import time
from pathlib import Path
from typing import Optional

from engine.slots.runner import SlotRunner, SlotResult


def _has_optuna() -> bool:
    """Check if Optuna is installed."""
    try:
        import optuna  # noqa: F401
        return True
    except ImportError:
        return False


class AdaptiveSearch:
    """Coverage-first adaptive grid search across slots."""

    def __init__(
        self,
        runner: SlotRunner,
        *,
        budget_per_round: int = 20,
        metric_name: str = "val_acc",
        results_path: Optional[Path] = None,
        seed: int = 42,
    ):
        self.runner = runner
        self.budget = budget_per_round
        self.metric = metric_name
        self.rng = random.Random(seed)
        self.results_path = results_path
        self.all_results: list[dict] = []
        self._tested: set[tuple] = set()

        if results_path and results_path.exists():
            self.all_results = json.loads(results_path.read_text())
            self._tested = {
                tuple(sorted(r["combo"].items())) for r in self.all_results
            }

    def _save(self):
        if self.results_path:
            self.results_path.parent.mkdir(parents=True, exist_ok=True)
            self.results_path.write_text(json.dumps(self.all_results, indent=2))

    def save_state(self, path: Optional[Path] = None):
        """Save search state for resume. Includes all results + tested set."""
        p = path or (self.results_path.with_suffix(".state.json") if self.results_path else None)
        if not p:
            return
        state = {
            "results": self.all_results,
            "tested": [list(t) for t in self._tested],
            "budget": self.budget,
            "metric": self.metric,
        }
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(state, indent=2))

    def load_state(self, path: Optional[Path] = None) -> bool:
        """Load saved state to resume. Returns True if state was loaded."""
        p = path or (self.results_path.with_suffix(".state.json") if self.results_path else None)
        if not p or not p.exists():
            return False
        state = json.loads(p.read_text())
        self.all_results = state.get("results", [])
        self._tested = {tuple(tuple(x) if isinstance(x, list) else x for x in t) for t in state.get("tested", [])}
        print(f"Resumed from {len(self.all_results)} prior results", flush=True)
        return True

    def _test(self, combo: dict, round_num: int) -> Optional[dict]:
        """Test one combo. Skip if already tested. Save incrementally."""
        key = tuple(sorted(combo.items()))
        if key in self._tested:
            return None

        r = self.runner.run(combo, metric_name=self.metric)
        entry = {
            "combo": combo,
            "val_acc": r.metrics.get(self.metric, 0),
            "error": r.error,
            "round": round_num,
        }
        self.all_results.append(entry)
        self._tested.add(key)
        self._save()
        return entry

    def _slot_names(self) -> list[str]:
        return [name for name, _ in self.runner.pipeline]

    def _variable_slots(self) -> list[str]:
        """Slots with more than 1 implementation (the ones worth varying)."""
        avail = self.runner.discover()
        return [name for name in self._slot_names() if len(avail.get(name, [])) > 1]

    def _families(self, slot: str) -> dict[str, list[str]]:
        """Group implementations by family prefix."""
        avail = self.runner.discover()
        impls = avail.get(slot, ["base"])
        groups: dict[str, list[str]] = {}
        for impl in impls:
            family = impl.split("_")[0]
            groups.setdefault(family, []).append(impl)
        return groups

    def run(self, n_rounds: int = 4) -> list[dict]:
        """
        Run the full adaptive search with named strategy modes.

        Modes (from n-autoresearch + atlas-gic):
          explore:  coverage guarantee + random sampling (round 1)
          exploit:  drill into winning model family (round 2)
          combine:  test near-miss pairs (round 3, from n-autoresearch)
          narrow:   fine-tune within winning region (round 4)
        """
        avail = self.runner.discover()
        slot_names = self._slot_names()
        var_slots = self._variable_slots()

        total_possible = 1
        for impls in avail.values():
            total_possible *= len(impls)

        # Resume from saved state if available
        self.load_state()

        optuna_status = "Optuna (if expensive)" if _has_optuna() else "grid"
        print(f"Adaptive search: {total_possible} possible combos, "
              f"budget={self.budget}/round, {n_rounds} rounds, exploit={optuna_status}", flush=True)

        # Mode 1: EXPLORE — coverage + random
        print(f"\n[EXPLORE]", flush=True)
        self._round_coverage(1)
        self.save_state()

        if n_rounds < 2:
            return self._valid_sorted()

        best_combo, best_acc = self._current_best()
        print(f"\nExplore best: {best_acc:.4f}", flush=True)
        for s in var_slots:
            print(f"  {s} = {best_combo[s]}", flush=True)

        # Mode 2: EXPLOIT — Optuna Bayesian or grid fallback
        print(f"\n[EXPLOIT]", flush=True)
        self._round_exploit(2, best_combo, var_slots)
        self.save_state()

        if n_rounds < 3:
            return self._valid_sorted()

        # Mode 3: COMBINE — test near-miss pairs (from n-autoresearch)
        print(f"\n[COMBINE]", flush=True)
        best_combo, best_acc = self._current_best()
        self._round_combine(3, var_slots)

        if n_rounds < 4:
            return self._valid_sorted()

        # Mode 4: NARROW — fine-tune within winning region
        print(f"\n[NARROW]", flush=True)
        best_combo, best_acc = self._current_best()
        print(f"Combine best: {best_acc:.4f}", flush=True)
        self._round_narrow(4, best_combo, var_slots)

        final_best, final_acc = self._current_best()
        print(f"\nFinal best: {final_acc:.4f}", flush=True)
        for s in var_slots:
            print(f"  {s} = {final_best[s]}", flush=True)

        return self._valid_sorted()

    def _round_coverage(self, round_num: int) -> list[dict]:
        """Round 1: one per family × each feature, then random fill."""
        avail = self.runner.discover()
        slot_names = self._slot_names()
        var_slots = self._variable_slots()

        # Mandatory coverage: one representative per family per variable slot,
        # crossed with all options of other variable slots
        mandatory: list[dict] = []

        if len(var_slots) == 2:
            # Common case: features × models
            s1, s2 = var_slots
            families_1 = self._families(s1)
            families_2 = self._families(s2)

            # Pick one representative per family
            reps_1 = [impls[0] for impls in families_1.values()]
            reps_2 = [impls[0] for impls in families_2.values()]

            for r1 in reps_1:
                for r2 in reps_2:
                    combo = {s: "base" for s in slot_names}
                    combo[s1] = r1
                    combo[s2] = r2
                    mandatory.append(combo)
        else:
            # General case: one per family per slot, base for others
            for vs in var_slots:
                families = self._families(vs)
                for fam, impls in families.items():
                    combo = {s: "base" for s in slot_names}
                    combo[vs] = impls[0]
                    mandatory.append(combo)

        # Deduplicate and remove already-tested
        mandatory_unique = []
        seen = set()
        for combo in mandatory:
            key = tuple(sorted(combo.items()))
            if key not in self._tested and key not in seen:
                mandatory_unique.append(combo)
                seen.add(key)

        coverage_count = len(mandatory_unique)
        explore_budget = max(0, self.budget - coverage_count)

        print(f"\nRound {round_num}: {coverage_count} coverage + {explore_budget} explore "
              f"= {coverage_count + explore_budget} total", flush=True)

        # Test coverage combos
        tested = 0
        for combo in mandatory_unique:
            entry = self._test(combo, round_num)
            if entry:
                tested += 1
                acc = entry["val_acc"]
                label = "+".join(v for v in combo.values() if v != "base") or "base"
                print(f"  [cov {tested}] {acc:.4f} | {label[:60]}", flush=True)

        # Fill remaining budget with random untested combos
        if explore_budget > 0:
            all_options = [avail.get(s, ["base"]) for s in slot_names]
            all_combos = [
                dict(zip(slot_names, c)) for c in itertools.product(*all_options)
            ]
            untested = [
                c for c in all_combos
                if tuple(sorted(c.items())) not in self._tested
            ]
            self.rng.shuffle(untested)

            for combo in untested[:explore_budget]:
                entry = self._test(combo, round_num)
                if entry:
                    tested += 1
                    acc = entry["val_acc"]
                    label = "+".join(v for v in combo.values() if v != "base") or "base"
                    print(f"  [exp {tested}] {acc:.4f} | {label[:60]}", flush=True)

        return [r for r in self.all_results if r["round"] == round_num]

    def _round_exploit(self, round_num: int, best_combo: dict, var_slots: list[str]):
        """Round 2: exploit the winning family. Optuna for expensive evals, grid for cheap."""
        use_optuna = _has_optuna() and self._is_expensive()
        if use_optuna:
            self._exploit_optuna(round_num, best_combo, var_slots)
        else:
            self._exploit_grid(round_num, best_combo, var_slots)

    def _is_expensive(self) -> bool:
        """Check if search space is large enough for Optuna to beat grid."""
        avail = self.runner.discover()
        total_impls = sum(len(v) for v in avail.values())
        # Optuna overhead only worth it with 20+ variants
        return total_impls > 20

    def _exploit_optuna(self, round_num: int, best_combo: dict, var_slots: list[str]):
        """Bayesian exploitation via Optuna TPE sampler with MedianPruner."""
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        avail = self.runner.discover()
        target_slot = max(var_slots, key=lambda s: len(avail.get(s, [])))
        best_impl = best_combo[target_slot]
        best_family = best_impl.split("_")[0]
        family_impls = [i for i in avail.get(target_slot, []) if i.startswith(best_family)]

        if not family_impls:
            return

        print(f"\n[EXPLOIT via Optuna] {len(family_impls)} {best_family} variants, "
              f"{self.budget} trials", flush=True)

        def objective(trial):
            impl = trial.suggest_categorical("impl", family_impls)
            combo = dict(best_combo)
            combo[target_slot] = impl
            entry = self._test(combo, round_num)
            return entry["val_acc"] if entry else 0.0

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=min(self.budget, len(family_impls)))

        best = study.best_trial
        print(f"  Optuna best: {best.value:.4f} | {best.params.get('impl', '?')}", flush=True)

        # Also cross with feature variants if multiple slots
        if len(var_slots) > 1:
            feat_slot = [s for s in var_slots if s != target_slot][0]
            feat_impls = avail.get(feat_slot, ["base"])
            best_model = best.params.get("impl", best_combo[target_slot])
            remaining = max(0, self.budget - len(study.trials))

            for feat in feat_impls[:remaining]:
                combo = dict(best_combo)
                combo[target_slot] = best_model
                combo[feat_slot] = feat
                entry = self._test(combo, round_num)
                if entry:
                    print(f"  [cross] {entry['val_acc']:.4f} | {best_model}+{feat}", flush=True)

    def _exploit_grid(self, round_num: int, best_combo: dict, var_slots: list[str]):
        """Fallback: grid-based exploitation when Optuna is not installed."""
        avail = self.runner.discover()
        slot_names = self._slot_names()

        target_slot = max(var_slots, key=lambda s: len(avail.get(s, [])))
        best_impl = best_combo[target_slot]
        best_family = best_impl.split("_")[0]
        family_impls = [
            i for i in avail.get(target_slot, []) if i.startswith(best_family)
        ]

        print(f"\nRound {round_num}: {len(family_impls)} {best_family} variants "
              f"in {target_slot}", flush=True)

        tested = 0
        for impl in family_impls:
            if tested >= self.budget:
                break
            combo = dict(best_combo)
            combo[target_slot] = impl
            entry = self._test(combo, round_num)
            if entry:
                tested += 1
                acc = entry["val_acc"]
                print(f"  [{tested}] {acc:.4f} | {impl}", flush=True)

        if len(var_slots) > 1:
            feat_slot = [s for s in var_slots if s != target_slot][0]
            best_model = self._current_best()[0][target_slot]

            remaining = self.budget - tested
            for feat in avail.get(feat_slot, ["base"]):
                if remaining <= 0:
                    break
                combo = dict(best_combo)
                combo[target_slot] = best_model
                combo[feat_slot] = feat
                entry = self._test(combo, round_num)
                if entry:
                    tested += 1
                    remaining -= 1
                    acc = entry["val_acc"]
                    print(f"  [{tested}] {acc:.4f} | {best_model}+{feat}", flush=True)

    def _round_combine(self, round_num: int, var_slots: list[str]):
        """
        COMBINE mode (from n-autoresearch): test near-miss pairs.

        Find submissions that individually scored within 1% of the best.
        Test them in combination — a near-miss feature + near-miss model
        might be superadditive.
        """
        avail = self.runner.discover()
        slot_names = self._slot_names()

        # Find near-misses per slot: within 1% of the best in that slot
        near_misses: dict[str, list[str]] = {}
        for vs in var_slots:
            slot_results: dict[str, float] = {}
            for r in self.all_results:
                if not r.get("error"):
                    impl = r["combo"].get(vs, "base")
                    acc = r["val_acc"]
                    slot_results[impl] = max(slot_results.get(impl, 0), acc)

            if not slot_results:
                continue
            best_val = max(slot_results.values())
            threshold = best_val * 0.99
            misses = [impl for impl, val in slot_results.items()
                      if val >= threshold and val < best_val]
            if misses:
                near_misses[vs] = misses

        if not near_misses:
            print(f"  No near-misses found (nothing within 1% of best)", flush=True)
            return

        total_nm = sum(len(v) for v in near_misses.values())
        print(f"  {total_nm} near-misses across {len(near_misses)} slots", flush=True)

        # Test combinations of near-misses with the current best
        best_combo, _ = self._current_best()
        tested = 0
        for vs, misses in near_misses.items():
            for miss_impl in misses:
                if tested >= self.budget:
                    break
                combo = dict(best_combo)
                combo[vs] = miss_impl
                entry = self._test(combo, round_num)
                if entry:
                    tested += 1
                    print(f"  [combine {tested}] {entry['val_acc']:.4f} | "
                          f"{vs}={miss_impl}", flush=True)

        # Also test near-miss × near-miss across slots
        if len(near_misses) >= 2:
            slots_with_nm = list(near_misses.keys())
            for i in range(len(slots_with_nm)):
                for j in range(i + 1, len(slots_with_nm)):
                    if tested >= self.budget:
                        break
                    s1, s2 = slots_with_nm[i], slots_with_nm[j]
                    for m1 in near_misses[s1][:2]:
                        for m2 in near_misses[s2][:2]:
                            if tested >= self.budget:
                                break
                            combo = dict(best_combo)
                            combo[s1] = m1
                            combo[s2] = m2
                            entry = self._test(combo, round_num)
                            if entry:
                                tested += 1
                                print(f"  [combine {tested}] {entry['val_acc']:.4f} | "
                                      f"{s1}={m1} + {s2}={m2}", flush=True)

    def _round_narrow(self, round_num: int, best_combo: dict, var_slots: list[str]):
        """Round 3: test neighbors of the current best in each variable slot."""
        avail = self.runner.discover()
        slot_names = self._slot_names()

        print(f"\nRound {round_num}: narrowing around best", flush=True)

        tested = 0
        for vs in var_slots:
            if tested >= self.budget:
                break
            impls = avail.get(vs, ["base"])
            best_impl = best_combo[vs]
            best_family = best_impl.split("_")[0]

            # Test other impls from the same family
            neighbors = [i for i in impls if i.startswith(best_family) and i != best_impl]
            for impl in neighbors:
                if tested >= self.budget:
                    break
                combo = dict(best_combo)
                combo[vs] = impl
                entry = self._test(combo, round_num)
                if entry:
                    tested += 1
                    acc = entry["val_acc"]
                    print(f"  [{tested}] {acc:.4f} | {vs}={impl}", flush=True)

    def _current_best(self) -> tuple[dict, float]:
        valid = self._valid_sorted()
        if not valid:
            slot_names = self._slot_names()
            return {s: "base" for s in slot_names}, 0.0
        return valid[0]["combo"], valid[0]["val_acc"]

    def _valid_sorted(self) -> list[dict]:
        valid = [r for r in self.all_results if not r.get("error")]
        valid.sort(key=lambda x: -x["val_acc"])
        return valid

    def summary(self) -> str:
        valid = self._valid_sorted()
        lines = [
            f"Tested: {len(valid)} combos",
            f"Best: {valid[0]['val_acc']:.4f}" if valid else "No results",
        ]
        for r in valid[:5]:
            combo = " + ".join(f"{v}" for s, v in r["combo"].items() if v != "base") or "base"
            lines.append(f"  {r['val_acc']:.4f} | {combo}")
        return "\n".join(lines)
