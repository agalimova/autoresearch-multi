"""
GPU training backend — wraps a training script (Python or Rust).

Mutates config/model files, runs the training command, and reads back
val_bpb. Each experiment gets a fixed wall-clock budget (default 300s).

The search space is one-dimensional: {'config': [variant_names]}.
Each variant is a config file that replaces the current one. The LLM
proposes new variants as code; the backend writes them and runs the binary.

Usage:
    # Python (Karpathy-style)
    backend = GpuTrainingBackend(
        project_dir=Path("path/to/project"),
        binary="python train.py",
        config_file="train.py",
    )

    # Rust
    backend = GpuTrainingBackend(
        project_dir=Path("path/to/autoresearch-rs"),
        binary="cargo run --release --bin train",
        config_file="src/config.rs",
    )
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from engine.backend import ExperimentResult, group_by_family


def _ext(path: str) -> str:
    """Extract file extension including dot, or empty string."""
    return Path(path).suffix or ""


class GpuTrainingBackend:
    """Backend wrapping a GPU training command.

    Variants are stored in project_dir/variants/config/.
    Each run swaps a variant into the config file, runs the command,
    parses val_bpb from stdout, and restores the original.
    """

    def __init__(
        self,
        project_dir: Path,
        *,
        binary: str = "python train.py",
        time_budget: float = 300.0,
        config_file: str = "train.py",
        model_file: str = "",
    ):
        self._project_dir = Path(project_dir)
        self._binary = binary
        self._time_budget = time_budget
        self._config_file = config_file
        self._model_file = model_file
        self._ext = _ext(config_file)

        # Variants directory
        self._variants_dir = self._project_dir / "variants"
        self._config_variants = self._variants_dir / "config"
        self._model_variants = self._variants_dir / "model"

        self._config_variants.mkdir(parents=True, exist_ok=True)
        if model_file:
            self._model_variants.mkdir(parents=True, exist_ok=True)
        self._ensure_baseline()

    def _variant_path(self, name: str) -> Path:
        return self._config_variants / f"{name}{self._ext}"

    def _ensure_baseline(self):
        """Save the current config/model as 'baseline' if not already saved."""
        baseline = self._variant_path("baseline")
        if not baseline.exists():
            src = self._project_dir / self._config_file
            if src.exists():
                shutil.copy2(src, baseline)

        if self._model_file:
            model_ext = _ext(self._model_file)
            baseline_model = self._model_variants / f"baseline{model_ext}"
            if not baseline_model.exists():
                src = self._project_dir / self._model_file
                if src.exists():
                    shutil.copy2(src, baseline_model)

    @property
    def metric_name(self) -> str:
        return "val_bpb"

    @property
    def higher_is_better(self) -> bool:
        return False

    def get_search_space(self) -> dict[str, list[str]]:
        pattern = f"*{self._ext}" if self._ext else "*"
        variants = sorted(f.stem for f in self._config_variants.glob(pattern))
        return {"config": variants or ["baseline"]}

    def run_experiment(self, config: dict[str, str]) -> ExperimentResult:
        variant_name = config.get("config", "baseline")
        variant_file = self._variant_path(variant_name)
        target = self._project_dir / self._config_file
        original_content = target.read_text() if target.exists() else None

        t0 = time.monotonic()
        try:
            if variant_file.exists():
                shutil.copy2(variant_file, target)

            proc = subprocess.run(
                self._binary,
                shell=True,
                capture_output=True,
                text=True,
                timeout=int(self._time_budget * 2),
                cwd=str(self._project_dir),
            )

            elapsed = time.monotonic() - t0

            if proc.returncode != 0:
                return ExperimentResult(
                    config=config, score=float("inf"), elapsed=elapsed,
                    error=f"exit code {proc.returncode}: {proc.stderr[-500:]}",
                )

            parsed = self._parse_output(proc.stdout)
            val_bpb = parsed.get("val_bpb")
            if val_bpb is None:
                return ExperimentResult(
                    config=config, score=float("inf"), elapsed=elapsed,
                    error=f"no val_bpb in output: {proc.stdout[-200:]}",
                )

            return ExperimentResult(
                config=config, score=val_bpb, elapsed=elapsed, metadata=parsed,
            )

        except subprocess.TimeoutExpired:
            return ExperimentResult(
                config=config, score=float("inf"),
                elapsed=time.monotonic() - t0, error="timeout",
            )
        except Exception as e:
            return ExperimentResult(
                config=config, score=float("inf"),
                elapsed=time.monotonic() - t0, error=str(e),
            )
        finally:
            if original_content is not None:
                target.write_text(original_content)

    def apply_proposal(self, dimension: str, name: str, code: str) -> Path:
        if dimension == "model" and self._model_file:
            ext = _ext(self._model_file)
            path = self._model_variants / f"{name}{ext}"
        else:
            path = self._variant_path(name)
        path.write_text(code)
        return path

    def prompt_context(self) -> dict[str, Any]:
        context: dict[str, Any] = {
            "backend_type": "gpu_training",
            "metric": "val_bpb",
            "metric_direction": "lower is better",
            "time_budget_seconds": self._time_budget,
            "binary": self._binary,
            "config_file_path": self._config_file,
            "language": "rust" if self._ext == ".rs" else "python",
        }

        config_path = self._project_dir / self._config_file
        if config_path.exists():
            context["config_file"] = config_path.read_text()

        if self._model_file:
            model_path = self._project_dir / self._model_file
            if model_path.exists():
                content = model_path.read_text()
                context["model_file"] = content[:3000] + ("..." if len(content) > 3000 else "")
                context["model_file_path"] = self._model_file

        context["existing_variants"] = {}
        pattern = f"*{self._ext}" if self._ext else "*"
        for f in sorted(self._config_variants.glob(pattern)):
            context["existing_variants"][f.stem] = f.read_text()[:500]

        return context

    def get_base_config(self) -> dict[str, str]:
        return {"config": "baseline"}

    def families(self, dimension: str) -> dict[str, list[str]]:
        return group_by_family(self.get_search_space().get(dimension, ["baseline"]))

    def build_llm_prompt(self, tracker_summary: str, findings: str, target_dim: str) -> str:
        ctx = self.prompt_context()
        config_path = ctx.get("config_file_path", "train.py")
        lang = ctx.get("language", "python")

        sections = [
            "You are an ML researcher optimizing a language model's val_bpb (bits per byte, lower is better).",
            f"\nTIME BUDGET: {ctx.get('time_budget_seconds', 300)}s per experiment.",
            f"\nCURRENT {config_path}:\n```{lang}\n{ctx.get('config_file', '(not available)')}\n```",
        ]

        if ctx.get("model_file"):
            sections.append(f"\nMODEL ({ctx.get('model_file_path', '')}):\n```{lang}\n{ctx['model_file']}\n```")

        existing = ctx.get("existing_variants", {})
        if existing:
            parts = [f"\n--- {name} ---\n{preview}" for name, preview in existing.items()]
            sections.append(f"\nEXISTING VARIANTS:{''.join(parts)}")

        sections.extend([
            f"\nEXPERIMENT HISTORY:\n{tracker_summary}",
            f"\nPRIOR LLM ROUNDS:\n{findings or '  (first round)'}",
            f"\nPropose a NEW {config_path} variant that might lower val_bpb."
            f"\nConsider: depth, learning rates, batch size, attention patterns, embedding dimensions."
            f"\nReturn ONLY the complete {lang} code for {config_path}."
            f"\nNo explanation, no markdown fences.",
        ])

        return "\n".join(sections)

    @staticmethod
    def _parse_output(stdout: str) -> dict[str, Any]:
        """Parse structured output after '---' separator."""
        result: dict[str, Any] = {}
        in_summary = False
        for line in stdout.splitlines():
            if line.strip() == "---":
                in_summary = True
                continue
            if not in_summary:
                continue
            match = re.match(r"(\w+):\s+(.+)", line.strip())
            if match:
                key, value = match.group(1), match.group(2).strip()
                try:
                    result[key] = float(value)
                except ValueError:
                    result[key] = value
        return result
