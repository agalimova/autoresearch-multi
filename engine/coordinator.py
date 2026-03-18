"""
Coordinator — adaptive search loop with any backend.

    coord = Coordinator(backend, budget_per_round=20)
    coord.run(n_rounds=4, llm_rounds=3)
"""

from __future__ import annotations

import itertools
import random
from pathlib import Path

from engine.backend import Backend, ExperimentTracker
from engine.llm_proposer import _call_llm, _clean_code, has_llm


class Coordinator:

    _MODES = ["EXPLORE", "EXPLOIT", "COMBINE", "NARROW"]

    def __init__(self, backend: Backend, *, budget_per_round: int = 20,
                 results_dir: Path = Path("results"), tracker_name: str = "tracker",
                 seed: int = 42):
        self.backend = backend
        self.budget = budget_per_round
        self.rng = random.Random(seed)
        results_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = ExperimentTracker(
            results_dir / f".tracker_{tracker_name}.json",
            higher_is_better=backend.higher_is_better,
        )

    # ── Public ───────────────────────────────────────────────────────────

    def run(self, *, n_rounds: int = 4, llm_rounds: int = 3) -> ExperimentTracker:
        space = self.backend.get_search_space()
        total = 1
        for opts in space.values():
            total *= len(opts)
        hib = "higher" if self.backend.higher_is_better else "lower"
        print(f"Coordinator: {total} configs, budget={self.budget}/round, "
              f"{n_rounds} rounds ({self.backend.metric_name}, {hib} is better)", flush=True)

        dispatch = [self._explore, self._exploit, self._combine, self._narrow]
        for i in range(min(n_rounds, 4)):
            print(f"\n[{self._MODES[i]}]", flush=True)
            dispatch[i](i + 1)
            self.tracker.update_interactions(self.backend.get_base_config())
            self.tracker.update_dead_families(self.backend)
            if i < n_rounds - 1:
                self._print_best(self._MODES[i])
        self._print_best("Final")

        if llm_rounds > 0 and has_llm():
            print(f"\n[LLM ROUNDS]", flush=True)
            self._llm(llm_rounds, n_rounds + 1)
        return self.tracker

    def results_as_dicts(self) -> list[dict]:
        return [{"combo": e.config, "val_acc": e.score, "error": e.error, "round": e.round_num}
                for e in self.tracker.experiments]

    def search_state(self) -> dict:
        best = self.tracker.best()
        space = self.backend.get_search_space()
        return {
            "best_combo": best.config if best else self.backend.get_base_config(),
            "best_acc": best.score if best else self.tracker.worst_score(),
            "slot_rankings": {
                dim: [{"impl": o, "best_acc": s, "dead": self.tracker.is_dead(o)}
                      for o, s in sorted(self.tracker.scores_by_option(dim).items(),
                                         key=lambda x: x[1], reverse=self.backend.higher_is_better)]
                for dim in space if len(space[dim]) > 1
            },
            "interactions": [{"pair": list(p), "delta": d}
                             for p, d in sorted(self.tracker.interactions.items(),
                                                key=lambda x: abs(x[1]), reverse=True)[:10]],
            "dead_families": sorted(self.tracker.dead_families),
            "tested_count": len(self.tracker.experiments),
        }

    # ── Modes ────────────────────────────────────────────────────────────

    def _vdims(self) -> list[str]:
        return [d for d, opts in self.backend.get_search_space().items() if len(opts) > 1]

    def _explore(self, rnd: int):
        """Coverage guarantee + random fill."""
        vdims = self._vdims()
        base = self.backend.get_base_config()
        reps = {d: [m[0] for m in self.backend.families(d).values()] for d in vdims}

        if len(vdims) <= 3:
            mandatory = [{**base, **dict(zip(vdims, c))}
                         for c in itertools.product(*(reps[d] for d in vdims))]
        else:
            mandatory = [{**base, d: o} for d in vdims for o in reps[d]]
        mandatory = self._dedup(mandatory)

        budget_left = max(0, self.budget - len(mandatory))
        print(f"  {len(mandatory)} coverage + {budget_left} explore", flush=True)
        tested = sum(self._test(c, rnd, "cov") for c in mandatory)

        if budget_left > 0:
            untested = [c for c in self._all_configs() if not self.tracker.already_tested(c)]
            self.rng.shuffle(untested)
            for c in untested[:budget_left]:
                tested += self._test(c, rnd, "exp")

    def _exploit(self, rnd: int):
        """Drill into the winning family, cross with other dims."""
        best = self.tracker.best()
        if not best:
            return
        space = self.backend.get_search_space()
        vdims = self._vdims()
        if not vdims:
            return

        target = max(vdims, key=lambda d: len(space[d]))
        family = best.config.get(target, "base").split("_")[0]
        opts = [o for o in space[target] if o.split("_")[0] == family and not self.tracker.is_dead(o)]
        print(f"  Exploiting {len(opts)} {family} variants in {target}", flush=True)

        tested = self._test_batch([{**best.config, target: o} for o in opts], rnd, "exploit")

        if len(vdims) > 1:
            best_t = self.tracker.best_in_dim(target)
            cross = [{**best.config, target: best_t, d: o}
                     for d in vdims if d != target
                     for o in space[d] if not self.tracker.is_dead(o)]
            self._test_batch(cross, rnd, "cross", remaining=self.budget - tested)

    def _combine(self, rnd: int):
        """Near-miss pairs, prioritized by interaction signal."""
        best = self.tracker.best()
        if not best:
            return
        near = self.tracker.near_misses(self._vdims())
        if not near:
            print(f"  No near-misses found", flush=True)
            return

        total_nm = sum(len(v) for v in near.values())
        print(f"  {total_nm} near-misses across {len(near)} dims", flush=True)

        # Single subs
        singles = [{**best.config, d: m} for d, ms in near.items() for m in ms]
        tested = self._test_batch(singles, rnd, "combine")

        # Cross-pairs sorted by interaction delta
        if len(near) >= 2:
            dims = list(near.keys())
            pairs = [(self.tracker.interactions.get(tuple(sorted((m1, m2))), 0.0), d1, m1, d2, m2)
                     for i, d1 in enumerate(dims) for d2 in dims[i+1:]
                     for m1 in near[d1] for m2 in near[d2]
                     if not self.tracker.is_dead(m1) and not self.tracker.is_dead(m2)]
            pairs.sort(key=lambda x: -x[0])
            cross = [{**best.config, d1: m1, d2: m2} for _, d1, m1, d2, m2 in pairs]
            self._test_batch(cross, rnd, "combine", remaining=self.budget - tested)

    def _narrow(self, rnd: int):
        """Neighbors of the winner."""
        best = self.tracker.best()
        if not best:
            return
        space = self.backend.get_search_space()
        neighbors = []
        for dim in self._vdims():
            opt = best.config.get(dim, "base")
            fam = opt.split("_")[0]
            neighbors += [{**best.config, dim: o} for o in space[dim]
                          if o.split("_")[0] == fam and o != opt and not self.tracker.is_dead(o)]
        self._test_batch(neighbors, rnd, "narrow")

    def _llm(self, n_rounds: int, start: int):
        findings: list[dict] = []
        best_so_far = self.tracker.best_score()

        for rnd in range(n_rounds):
            num = start + rnd
            print(f"\n  LLM round {rnd+1}/{n_rounds}", flush=True)

            prompt = self._build_llm_prompt(findings)
            code = _clean_code(_call_llm(prompt) or "")
            if not code:
                print("    No valid LLM response", flush=True)
                continue

            space = self.backend.get_search_space()
            dim = max(space, key=lambda d: len(space[d]))
            name = f"llm_v{num}"
            self.backend.apply_proposal(dim, name, code)
            print(f"    Proposed: {dim}/{name}", flush=True)

            best = self.tracker.best()
            config = dict(best.config) if best else self.backend.get_base_config()
            config[dim] = name
            self._test(config, num, "llm")
            self.tracker.update_interactions(self.backend.get_base_config())

            new_best = self.tracker.best_score()
            improved = self.tracker.is_better(new_best, best_so_far)
            findings.append({"round": rnd+1, "proposed": name, "score": new_best,
                             "prev_best": best_so_far, "improved": improved})
            if improved:
                print(f"    Improved: {best_so_far:.6f} -> {new_best:.6f}", flush=True)
                best_so_far = new_best

    # ── Helpers ──────────────────────────────────────────────────────────

    def _test(self, config: dict[str, str], rnd: int, tag: str) -> int:
        if self.tracker.already_tested(config):
            return 0
        if any(self.tracker.is_dead(o) for o in config.values()):
            return 0
        result = self.backend.run_experiment(config)
        self.tracker.record(result, round_num=rnd, base_config=self.backend.get_base_config())
        label = self._label(config)
        if result.error:
            print(f"  [{tag}] FAIL | {label[:50]} | {result.error[:40]}", flush=True)
        else:
            print(f"  [{tag}] {result.score:.6f} | {label[:60]}", flush=True)
        return 1

    def _test_batch(self, configs: list[dict], rnd: int, tag: str, remaining: int | None = None) -> int:
        limit = remaining if remaining is not None else self.budget
        tested = 0
        for c in configs:
            if tested >= limit:
                break
            tested += self._test(c, rnd, tag)
        return tested

    def _label(self, config: dict[str, str]) -> str:
        base = self.backend.get_base_config()
        return "+".join(v for k, v in config.items() if base.get(k) != v) or "(baseline)"

    def _print_best(self, phase: str):
        best = self.tracker.best()
        if not best:
            return
        base = self.backend.get_base_config()
        diffs = [f"  {k} = {v}" for k, v in best.config.items() if base.get(k) != v]
        print(f"\n{phase} best: {best.score:.6f}", flush=True)
        for d in diffs:
            print(d, flush=True)

    def _dedup(self, configs: list[dict[str, str]]) -> list[dict[str, str]]:
        seen: set[tuple] = set()
        return [c for c in configs if (k := tuple(sorted(c.items()))) not in seen
                and not self.tracker.already_tested(c) and not seen.add(k)]  # type: ignore[func-returns-value]

    def _all_configs(self) -> list[dict[str, str]]:
        space = self.backend.get_search_space()
        dims = list(space.keys())
        return [dict(zip(dims, c)) for c in itertools.product(*(space[d] for d in dims))]

    def _build_llm_prompt(self, findings: list[dict]) -> str:
        summary = self.tracker.summary_for_prompt()
        ftext = ""
        for f in (findings or [])[-3:]:
            s = "Improved" if f["improved"] else "No improvement"
            ftext += f"\n  Round {f['round']}: {s} ({f['prev_best']:.6f} -> {f['score']:.6f})\n"
        space = self.backend.get_search_space()
        dim = max(space, key=lambda d: len(space[d]))
        return self.backend.build_llm_prompt(summary, ftext, dim)
