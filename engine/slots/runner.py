"""
Slot pipeline runner + exhaustive combo engine.

Imports one implementation per slot, runs the pipeline, returns metrics.
The combo engine tests all combinations of active submissions across slots.

Usage:
    runner = SlotRunner(Path("workspace/slots"))
    
    # Run one specific combination
    result = runner.run({"engineer_features": "feynman_001", "build_model": "turing_001"})
    
    # Run all combinations exhaustively
    results = runner.run_all_combos()
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


def _normalize_Xy(X, y):
    """Normalize X and y to pandas types with clean integer index.

    LLM-proposed slots may use .iloc, .loc, .unique(), or other
    pandas-specific APIs. Converting numpy arrays to DataFrame/Series
    and resetting the index prevents crashes from type or index mismatches.
    """
    try:
        import pandas as pd
        import numpy as np
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        if hasattr(X, "reset_index"):
            X = X.reset_index(drop=True)
        if hasattr(y, "reset_index"):
            y = y.reset_index(drop=True)
    except Exception:
        pass
    return X, y


@dataclass
class SlotResult:
    """Result of running one pipeline combination."""
    combo: dict[str, str]        # slot_name -> impl_filename (without .py)
    metrics: dict[str, float]
    elapsed: float
    error: Optional[str] = None


class SlotRunner:
    """
    Discovers slot directories, loads implementations, runs the pipeline.
    
    For PyTorch/Keras pipelines, warm-starts models from the best checkpoint
    found so far (if available). Sklearn models are stateless and skip this.
    
    Expected directory structure:
        slots_dir/
            load_data/
                base.py          (must export load_data())
                agent_001.py
            engineer_features/
                base.py          (must export engineer_features())
            build_model/
                base.py          (must export build_model())
            evaluate/
                base.py          (must export evaluate())
    """

    # Known pipeline orderings. If a slots_dir has one of these sets of slots,
    # use the corresponding order. Otherwise, alphabetical.
    KNOWN_PIPELINES: dict[frozenset[str], list[str]] = {
        frozenset(["load_data", "engineer_features", "build_model", "evaluate"]): [
            "load_data", "engineer_features", "build_model", "evaluate",
        ],
        frozenset(["load_data", "vectorize", "build_model", "evaluate"]): [
            "load_data", "vectorize", "build_model", "evaluate",
        ],
        frozenset(["load_data", "build_model", "evaluate"]): [
            "load_data", "build_model", "evaluate",
        ],
        frozenset(["get_transforms", "build_model", "build_optimizer", "evaluate"]): [
            "get_transforms", "build_model", "build_optimizer", "evaluate",
        ],
    }

    def __init__(self, slots_dir: Path):
        self.slots_dir = slots_dir
        self._cache: dict[str, Any] = {}
        self._pipeline: Optional[list[tuple[str, str]]] = None
        self._ckpt = None  # lazy-initialized CheckpointManager

    @property
    def pipeline(self) -> list[tuple[str, str]]:
        """Auto-discover pipeline from directory structure."""
        if self._pipeline is not None:
            return self._pipeline

        slot_names = sorted([
            d.name for d in self.slots_dir.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        ])
        slot_set = frozenset(slot_names)

        # Check known orderings
        if slot_set in self.KNOWN_PIPELINES:
            ordered = self.KNOWN_PIPELINES[slot_set]
        else:
            ordered = slot_names

        # Each slot exports a function with the same name as the slot
        self._pipeline = [(name, name) for name in ordered]
        return self._pipeline

    def discover(self) -> dict[str, list[str]]:
        """Return {slot_name: [impl_names]} for all slots."""
        result = {}
        for slot_name, _ in self.pipeline:
            slot_dir = self.slots_dir / slot_name
            if not slot_dir.is_dir():
                continue
            impls = [
                f.stem for f in sorted(slot_dir.glob("*.py"))
                if f.stem != "__init__"
            ]
            result[slot_name] = impls
        return result

    def _checkpoint(self):
        """Lazy-init checkpoint manager. Returns manager or None."""
        if self._ckpt is None:
            try:
                from engine.checkpoint import CheckpointManager
                ckpt_dir = self.slots_dir.parent / "checkpoints"
                self._ckpt = CheckpointManager(ckpt_dir)
            except Exception:
                pass
        return self._ckpt

    def _warm_start(self, model):
        """Warm-start model from best checkpoint if available."""
        ckpt = self._checkpoint()
        if ckpt and ckpt.has_checkpoint:
            if ckpt.warm_start(model):
                return True
        return False

    def _save_checkpoint(self, model, metric_value: float):
        """Save model if it beats the current best."""
        ckpt = self._checkpoint()
        if ckpt:
            ckpt.save(model, metric_value=metric_value)

    def load_fn(self, slot_name: str, impl_name: str) -> Callable:
        """Load a function from a slot implementation file."""
        cache_key = f"{slot_name}/{impl_name}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        filepath = self.slots_dir / slot_name / f"{impl_name}.py"
        if not filepath.exists():
            raise FileNotFoundError(f"No impl: {filepath}")

        _, fn_name = next(
            (s, f) for s, f in self.pipeline if s == slot_name
        )

        spec = importlib.util.spec_from_file_location(cache_key, filepath)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {filepath}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        fn = getattr(mod, fn_name, None)
        if fn is None:
            raise AttributeError(
                f"{filepath} must export {fn_name}(). "
                f"Found: {[a for a in dir(mod) if not a.startswith('_')]}"
            )

        self._cache[cache_key] = fn
        return fn

    def run(
        self,
        combo: dict[str, str],
        *,
        metric_name: str = "val_acc",
    ) -> SlotResult:
        """
        Run the pipeline with a specific combination of implementations.
        
        combo: {slot_name: impl_name} e.g. {"engineer_features": "feynman_001"}
              Missing slots default to "base".
        """
        t0 = time.monotonic()

        # Fill defaults
        full_combo = {name: "base" for name, _ in self.pipeline}
        full_combo.update(combo)

        try:
            # Load all functions
            fns = {name: self.load_fn(name, full_combo[name]) for name, _ in self.pipeline}

            # Run pipeline: chain slot outputs as inputs to the next slot.
            slot_names = [name for name, _ in self.pipeline]
            model = None
            state = fns[slot_names[0]]()
            for sn in slot_names[1:]:
                args = state if isinstance(state, tuple) else (state,)
                if sn in ("engineer_features", "vectorize"):
                    state = _normalize_Xy(*fns[sn](*args))
                elif sn == "build_model":
                    model = fns[sn]()
                    self._warm_start(model)
                    state = (*args, model) if args else (model,)
                elif sn == "build_optimizer":
                    state = (*args, fns[sn](model))
                else:
                    state = fns[sn](*args)
            metrics = state if isinstance(state, dict) else {"error": "pipeline returned non-dict"}
            if model is not None:
                self._save_checkpoint(model, metrics.get(metric_name, 0))

            return SlotResult(
                combo=full_combo,
                metrics=metrics,
                elapsed=time.monotonic() - t0,
            )
        except Exception as e:
            return SlotResult(
                combo=full_combo,
                metrics={metric_name: 0.0},
                elapsed=time.monotonic() - t0,
                error=str(e),
            )

    def run_all_combos(
        self,
        *,
        metric_name: str = "val_acc",
        higher_is_better: bool = True,
    ) -> list[SlotResult]:
        """
        Exhaustively test all combinations of implementations across slots.
        
        Returns results sorted by metric (best first).
        """
        available = self.discover()
        slot_names = [name for name, _ in self.pipeline]

        # Build list of options per slot
        options_per_slot = [available.get(name, ["base"]) for name in slot_names]

        # All combinations
        all_combos = list(itertools.product(*options_per_slot))
        total = len(all_combos)

        results = []
        for i, combo_tuple in enumerate(all_combos):
            combo = dict(zip(slot_names, combo_tuple))
            result = self.run(combo, metric_name=metric_name)

            status = f"{result.metrics.get(metric_name, 0):.4f}"
            if result.error:
                status = f"FAIL: {result.error[:40]}"

            label = " + ".join(
                f"{v}" if v != "base" else "base"
                for v in combo_tuple
            )
            print(f"  [{i+1}/{total}] {label}: {status}")

            results.append(result)

        # Sort by metric
        rev = higher_is_better
        results.sort(
            key=lambda r: r.metrics.get(metric_name, 0.0),
            reverse=rev,
        )
        return results
