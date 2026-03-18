"""
Tabular ML backend — wraps the existing SlotRunner.

This adapts the slot-based pipeline (load_data -> features -> model -> evaluate)
to the Backend interface. All existing autoresearch-multi functionality
(sklearn, XGBoost, Keras, PyTorch tabular) works through this backend.

Usage:
    from engine.backends.tabular import TabularBackend
    backend = TabularBackend(slot_dir=Path("workspace/slots_adult"))
    result = backend.run_experiment({"engineer_features": "pca", "build_model": "xgb_default"})
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from engine.backend import ExperimentResult, group_by_family
from engine.slots.runner import SlotRunner


class TabularBackend:
    """Backend wrapping the existing slot-based tabular ML pipeline."""

    def __init__(
        self,
        slot_dir: Path,
        *,
        metric: str = "val_acc",
    ):
        self._runner = SlotRunner(slot_dir)
        self._slot_dir = slot_dir
        self._metric = metric

    @property
    def metric_name(self) -> str:
        return self._metric

    @property
    def higher_is_better(self) -> bool:
        return True

    def get_search_space(self) -> dict[str, list[str]]:
        return self._runner.discover()

    def run_experiment(self, config: dict[str, str]) -> ExperimentResult:
        result = self._runner.run(config, metric_name=self._metric)
        return ExperimentResult(
            config=result.combo,
            score=result.metrics.get(self._metric, 0.0),
            elapsed=result.elapsed,
            error=result.error,
            metadata=result.metrics,
        )

    def apply_proposal(self, dimension: str, name: str, code: str) -> Path:
        out_dir = self._slot_dir / dimension
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{name}.py"
        path.write_text(code)
        # SlotRunner doesn't expose a public invalidation method, so we
        # reach into its internals to force re-discovery of the new file.
        if hasattr(self._runner, "_cache"):
            self._runner._cache.clear()
        if hasattr(self._runner, "_pipeline"):
            self._runner._pipeline = None
        return path

    def prompt_context(self) -> dict[str, Any]:
        """Tabular-specific context: slot structure, existing variants, base code."""
        context: dict[str, Any] = {
            "backend_type": "tabular_ml",
            "metric": self._metric,
            "pipeline": [name for name, _ in self._runner.pipeline],
            "slots": {},
        }
        for slot_name, _ in self._runner.pipeline:
            slot_dir = self._slot_dir / slot_name
            if not slot_dir.is_dir():
                continue
            variants: dict[str, str] = {}
            for f in sorted(slot_dir.glob("*.py"))[:6]:
                content = f.read_text()
                if f.name == "base.py":
                    variants[f.stem] = content
                else:
                    variants[f.stem] = content[:300] + ("..." if len(content) > 300 else "")
            context["slots"][slot_name] = variants
        return context

    def get_base_config(self) -> dict[str, str]:
        return {name: "base" for name, _ in self._runner.pipeline}

    def families(self, dimension: str) -> dict[str, list[str]]:
        return group_by_family(self.get_search_space().get(dimension, ["base"]))

    def build_llm_prompt(self, tracker_summary: str, findings: str, target_dim: str) -> str:
        slots = self.prompt_context().get("slots", {})
        slot_code = slots.get(target_dim, {})

        existing = ""
        for name, code in slot_code.items():
            if name == "base":
                existing += f"\n--- {name}.py (REFERENCE: match this signature) ---\n{code}\n"
            else:
                existing += f"\n--- {name}.py ---\n{code}\n"

        return f"""You are an ML researcher. Write a Python function for the "{target_dim}" slot.
Metric: {self._metric} (higher is better).
Pipeline: {' -> '.join(name for name, _ in self._runner.pipeline)}

EXISTING VARIANTS:
{existing}

EXPERIMENT HISTORY:
{tracker_summary}

PRIOR ROUNDS:
{findings or "  (first round)"}

Propose a NEW variant. Match the base.py signature exactly.
Return ONLY the Python code. No explanation, no markdown fences."""
