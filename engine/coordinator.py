"""
Unified coordinator — runs the adaptive search loop with any backend.

Takes the 4-mode search (EXPLORE, EXPLOIT, COMBINE, NARROW), pruning,
interaction detection, and LLM proposer from autoresearch-multi and drives
them through the Backend interface. The coordinator doesn't know whether
it's running tabular ML or GPU training — it just calls the backend.

Usage:
    from engine.coordinator import Coordinator
    from engine.backends.tabular import TabularBackend

    backend = TabularBackend(slot_dir=Path("workspace/slots_adult"))
    coord = Coordinator(backend, budget_per_round=20)
    coord.run(n_rounds=4, llm_rounds=3)
"""

from __future__ import annotations

import itertools
import random
from pathlib import Path
from typing import Optional

from engine.backend import Backend, ExperimentResult, ExperimentTracker
from engine.llm_proposer import _call_llm, _clean_code, has_llm


class Coordinator:
    """Drives adaptive search through any Backend implementation."""

    def __init__(
        self,
        backend: Backend,
        *,
        budget_per_round: int = 20,
        results_dir: Path = Path("results"),
        seed: int = 42,
    ):
        self.backend = backend
        self.budget = budget_per_round
        self.rng = random.Random(seed)

        results_dir.mkdir(parents=True, exist_ok=True)
        tracker_path = results_dir / "tracker.json"
        self.tracker = ExperimentTracker(
            tracker_path, higher_is_better=backend.higher_is_better,
        )

    # ── Public API ───────────────────────────────────────────────────────

    _MODES = [
        ("EXPLORE",  "_round_explore"),
        ("EXPLOIT",  "_round_exploit"),
        ("COMBINE",  "_round_combine"),
        ("NARROW",   "_round_narrow"),
    ]

    def run(self, *, n_rounds: int = 4, llm_rounds: int = 3) -> ExperimentTracker:
        """Run the full adaptive search, then optionally LLM rounds."""
        space = self.backend.get_search_space()
        total = 1
        for opts in space.values():
            total *= len(opts)

        print(f"Coordinator: {total} possible configs, "
              f"budget={self.budget}/round, {n_rounds} search rounds", flush=True)
        print(f"  Backend: {self.backend.metric_name} "
              f"({'higher' if self.backend.higher_is_better else 'lower'} is better)", flush=True)

        for i, (label, method_name) in enumerate(self._MODES[:n_rounds]):
            print(f"\n[{label}]", flush=True)
            getattr(self, method_name)(round_num=i + 1)
            self.tracker.update_interactions(self.backend.get_base_config())
            self.tracker.update_dead_families(self.backend)
            if i < n_rounds - 1:
                self._print_best(label.capitalize())

        self._print_best("Final")

        if llm_rounds > 0 and has_llm():
            print(f"\n[LLM ROUNDS]", flush=True)
            self._llm_rounds(n_rounds=llm_rounds, start_round=n_rounds + 1)

        return self.tracker

    # ── Modes ────────────────────────────────────────────────────────────

    def _round_explore(self, round_num: int):
        """Coverage guarantee + random fill."""
        space = self.backend.get_search_space()
        dims = list(space.keys())
        var_dims = [d for d in dims if len(space[d]) > 1]

        # Coverage: one per family per dimension, crossed with others
        mandatory = self._coverage_combos(var_dims)
        coverage_count = len(mandatory)
        explore_budget = max(0, self.budget - coverage_count)

        print(f"  {coverage_count} coverage + {explore_budget} explore", flush=True)

        tested = 0
        for config in mandatory:
            tested += self._test_and_print(config, round_num, "cov", tested + 1)

        # Random fill
        if explore_budget > 0:
            all_configs = self._all_configs()
            untested = [c for c in all_configs if not self.tracker.already_tested(c)]
            self.rng.shuffle(untested)
            for config in untested[:explore_budget]:
                tested += self._test_and_print(config, round_num, "exp", tested + 1)

    def _round_exploit(self, round_num: int):
        """Drill into the winning family."""
        best = self.tracker.best()
        if not best:
            return

        space = self.backend.get_search_space()
        var_dims = [d for d in space if len(space[d]) > 1]
        if not var_dims:
            return

        # Find dimension with most options — that's where we exploit
        target_dim = max(var_dims, key=lambda d: len(space[d]))
        best_opt = best.config.get(target_dim, "baseline")
        best_family = best_opt.split("_")[0]

        # All options in the winning family
        family_opts = [
            o for o in space[target_dim]
            if o.split("_")[0] == best_family and not self.tracker.is_dead(o)
        ]

        print(f"  Exploiting {len(family_opts)} {best_family} variants in {target_dim}", flush=True)

        tested = 0
        for opt in family_opts:
            if tested >= self.budget:
                break
            config = dict(best.config)
            config[target_dim] = opt
            tested += self._test_and_print(config, round_num, "exploit", tested + 1)

        # Cross with other dimensions
        if len(var_dims) > 1:
            other_dims = [d for d in var_dims if d != target_dim]
            best_in_target = self._best_in_dim(target_dim)
            remaining = self.budget - tested
            for od in other_dims:
                for opt in space[od]:
                    if remaining <= 0:
                        break
                    if self.tracker.is_dead(opt):
                        continue
                    config = dict(best.config)
                    config[target_dim] = best_in_target
                    config[od] = opt
                    r = self._test_and_print(config, round_num, "cross", tested + 1)
                    tested += r
                    remaining -= r

    def _round_combine(self, round_num: int):
        """Test near-miss pairs for superadditive interactions."""
        space = self.backend.get_search_space()
        var_dims = [d for d in space if len(space[d]) > 1]
        best = self.tracker.best()
        if not best:
            return

        # Find near-misses: within 1% of best per dimension
        near_misses = self._find_near_misses(var_dims)
        if not near_misses:
            print(f"  No near-misses found", flush=True)
            return

        total_nm = sum(len(v) for v in near_misses.values())
        print(f"  {total_nm} near-misses across {len(near_misses)} dims", flush=True)

        tested = 0

        # Single substitution: near-miss in one dim, best everywhere else
        for dim, misses in near_misses.items():
            for miss in misses:
                if tested >= self.budget:
                    break
                config = dict(best.config)
                config[dim] = miss
                tested += self._test_and_print(config, round_num, "combine", tested + 1)

        # Cross-pair: near-miss x near-miss across dims
        if len(near_misses) >= 2:
            cross = self._cross_near_misses(near_misses)
            for _delta, d1, m1, d2, m2 in cross:
                if tested >= self.budget:
                    break
                if self.tracker.is_dead(m1) or self.tracker.is_dead(m2):
                    continue
                config = dict(best.config)
                config[d1] = m1
                config[d2] = m2
                tested += self._test_and_print(config, round_num, "combine", tested + 1)

    def _round_narrow(self, round_num: int):
        """Fine-tune within the winning region."""
        best = self.tracker.best()
        if not best:
            return

        space = self.backend.get_search_space()
        var_dims = [d for d in space if len(space[d]) > 1]

        tested = 0
        for dim in var_dims:
            if tested >= self.budget:
                break
            best_opt = best.config.get(dim, "baseline")
            best_family = best_opt.split("_")[0]

            neighbors = [
                o for o in space[dim]
                if o.split("_")[0] == best_family and o != best_opt
                and not self.tracker.is_dead(o)
            ]
            for opt in neighbors:
                if tested >= self.budget:
                    break
                config = dict(best.config)
                config[dim] = opt
                tested += self._test_and_print(config, round_num, "narrow", tested + 1)

    def _llm_rounds(self, *, n_rounds: int, start_round: int):
        """Iterative LLM proposal rounds."""
        findings: list[dict] = []
        best_so_far = self.tracker.best_score()

        for rnd in range(n_rounds):
            round_num = start_round + rnd
            print(f"\n  LLM round {rnd + 1}/{n_rounds}", flush=True)

            # Build backend-aware prompt
            prompt = self._build_llm_prompt(findings)
            response = _call_llm(prompt)
            if not response:
                print("    No LLM response", flush=True)
                continue

            code = _clean_code(response)
            if not code:
                print("    LLM returned no valid code", flush=True)
                continue

            # Determine target dimension and name
            target_dim = self._llm_target_dimension()
            name = f"llm_v{round_num}"

            # Apply and test
            path = self.backend.apply_proposal(target_dim, name, code)
            print(f"    Proposed: {target_dim}/{name}", flush=True)

            best = self.tracker.best()
            config = dict(best.config) if best else self.backend.get_base_config()
            config[target_dim] = name
            self._test_and_print(config, round_num, "llm", rnd + 1)

            self.tracker.update_interactions(self.backend.get_base_config())

            new_best = self.tracker.best_score()
            improved = self.tracker.is_better(new_best, best_so_far)
            findings.append({
                "round": rnd + 1,
                "proposed": name,
                "score": new_best,
                "prev_best": best_so_far,
                "improved": improved,
            })
            if improved:
                print(f"    Improved: {best_so_far:.6f} -> {new_best:.6f}", flush=True)
                best_so_far = new_best

    # ── Helpers ──────────────────────────────────────────────────────────

    def _test_and_print(
        self, config: dict[str, str], round_num: int, tag: str, idx: int,
    ) -> int:
        """Test one config, print result, return 1 if tested (0 if skipped)."""
        if self.tracker.already_tested(config):
            return 0

        # Skip configs with dead options
        for opt in config.values():
            if self.tracker.is_dead(opt):
                return 0

        result = self.backend.run_experiment(config)
        self.tracker.record(
            result, round_num=round_num,
            base_config=self.backend.get_base_config(),
        )

        label = self._config_label(config)
        if result.error:
            print(f"  [{tag} {idx}] FAIL | {label[:50]} | {result.error[:40]}", flush=True)
        else:
            print(f"  [{tag} {idx}] {result.score:.6f} | {label[:60]}", flush=True)
        return 1

    def _config_label(self, config: dict[str, str]) -> str:
        """Compact label for a config (only non-base values)."""
        base = self.backend.get_base_config()
        diffs = {k: v for k, v in config.items() if base.get(k) != v}
        if not diffs:
            return "(baseline)"
        return "+".join(f"{v}" for v in diffs.values())

    def _print_best(self, phase: str):
        """Print current best after a search phase."""
        best = self.tracker.best()
        if best:
            print(f"\n{phase} best: {best.score:.6f}", flush=True)
            base = self.backend.get_base_config()
            for k, v in best.config.items():
                if base.get(k) != v:
                    print(f"  {k} = {v}", flush=True)

    def _coverage_combos(self, var_dims: list[str]) -> list[dict[str, str]]:
        """One representative per family per dimension, crossed with others.

        For <= 3 dimensions: full cross product of family representatives.
        For > 3 dimensions: one per family per dim with base elsewhere
        (cross product would be too large).
        """
        base = self.backend.get_base_config()

        # Get one representative per family per dimension
        dim_reps = {
            dim: [members[0] for members in self.backend.families(dim).values()]
            for dim in var_dims
        }

        if len(var_dims) <= 3:
            combos = []
            for combo in itertools.product(*[dim_reps[d] for d in var_dims]):
                config = dict(base)
                for dim, opt in zip(var_dims, combo):
                    config[dim] = opt
                combos.append(config)
            return self._dedup(combos)

        # High-dimensional: one per family per dim, base elsewhere
        combos = []
        for dim in var_dims:
            for opt in dim_reps[dim]:
                config = dict(base)
                config[dim] = opt
                combos.append(config)
        return self._dedup(combos)

    def _dedup(self, configs: list[dict[str, str]]) -> list[dict[str, str]]:
        """Remove duplicates and already-tested configs."""
        seen: set[tuple] = set()
        result = []
        for config in configs:
            key = tuple(sorted(config.items()))
            if key not in seen and not self.tracker.already_tested(config):
                seen.add(key)
                result.append(config)
        return result

    def _all_configs(self) -> list[dict[str, str]]:
        """All possible configs (Cartesian product)."""
        space = self.backend.get_search_space()
        dims = list(space.keys())
        options = [space[d] for d in dims]
        return [dict(zip(dims, combo)) for combo in itertools.product(*options)]

    def _scores_by_option(self, dim: str) -> dict[str, float]:
        """Best score per option in a dimension."""
        t = self.tracker
        result: dict[str, float] = {}
        for e in t.experiments:
            if e.error:
                continue
            opt = e.config.get(dim, "baseline")
            prev = result.get(opt)
            if prev is None or t.is_better(e.score, prev):
                result[opt] = e.score
        return result

    def _best_in_dim(self, dim: str) -> str:
        """Best option for a dimension across all experiments."""
        scores = self._scores_by_option(dim)
        if not scores:
            return "baseline"
        ranked = sorted(scores, key=lambda o: scores[o], reverse=self.backend.higher_is_better)
        return ranked[0]

    def _find_near_misses(self, var_dims: list[str]) -> dict[str, list[str]]:
        """Find options within 1% of the best per dimension."""
        near: dict[str, list[str]] = {}
        for dim in var_dims:
            scores = self._scores_by_option(dim)
            if not scores:
                continue
            best_val = self.tracker.best_of(list(scores.values()))
            threshold = best_val * (0.99 if self.backend.higher_is_better else 1.01)

            misses = [
                o for o, v in scores.items()
                if v != best_val and not self.tracker._is_worse_than(v, threshold)
            ]
            if misses:
                near[dim] = misses
        return near

    def _cross_near_misses(
        self, near_misses: dict[str, list[str]],
    ) -> list[tuple[float, str, str, str, str]]:
        """Build cross-pairs of near-misses, prioritized by interaction signal."""
        dims = list(near_misses.keys())
        pairs: list[tuple[float, str, str, str, str]] = []
        for i in range(len(dims)):
            for j in range(i + 1, len(dims)):
                d1, d2 = dims[i], dims[j]
                for m1 in near_misses[d1]:
                    for m2 in near_misses[d2]:
                        pair_key = tuple(sorted((m1, m2)))
                        delta = self.tracker.interactions.get(pair_key, 0.0)
                        pairs.append((delta, d1, m1, d2, m2))
        pairs.sort(key=lambda x: -x[0])
        return pairs

    def _llm_target_dimension(self) -> str:
        """Which dimension should the LLM propose variants for?"""
        space = self.backend.get_search_space()
        # Pick the dimension with the most options (most room for improvement)
        return max(space, key=lambda d: len(space[d]))

    def _build_llm_prompt(self, prior_findings: list[dict]) -> str:
        """Build a backend-aware prompt for the LLM."""
        tracker_summary = self.tracker.summary_for_prompt()

        findings_str = ""
        if prior_findings:
            for f in prior_findings[-3:]:
                findings_str += (
                    f"\n  Round {f['round']}: {'Improved' if f['improved'] else 'No improvement'}"
                    f" ({f['prev_best']:.6f} -> {f['score']:.6f})"
                    f"\n  Proposed: {f['proposed']}\n"
                )

        target_dim = self._llm_target_dimension()
        return self.backend.build_llm_prompt(tracker_summary, findings_str, target_dim)
