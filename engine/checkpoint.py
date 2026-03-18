"""
Warm-start checkpoint manager for neural net pipelines.

Saves the best model state after each combo. Next combo can warm-start
from the best weights instead of random initialization.

Only active for PyTorch and Keras models. Sklearn models are stateless.

Usage:
    ckpt = CheckpointManager("results/checkpoints")
    
    # After training:
    ckpt.save(model, metric_value=0.93)
    
    # Before next training:
    ckpt.warm_start(model)  # loads best weights if available
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class CheckpointManager:
    """Manages best checkpoint for warm-starting across combos."""

    def __init__(self, checkpoint_dir: str | Path = "results/checkpoints"):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.dir / "meta.json"
        self._meta = self._load_meta()

    def _load_meta(self) -> dict:
        if self.meta_path.exists():
            return json.loads(self.meta_path.read_text())
        return {"best_metric": 0.0, "best_file": "", "framework": ""}

    def _save_meta(self) -> None:
        self.meta_path.write_text(json.dumps(self._meta, indent=2))

    def save(self, model, *, metric_value: float) -> bool:
        """Save model if it beats the current best. Returns True if saved."""
        if metric_value <= self._meta.get("best_metric", 0):
            return False

        # Detect framework and save accordingly
        framework = _detect_model_framework(model)
        if not framework:
            return False

        path = self.dir / "best.pt"
        old_metric = self._meta.get("best_metric", 0)

        if framework == "pytorch":
            import torch
            torch.save(model.state_dict(), path)
        elif framework == "keras":
            model.save_weights(str(self.dir / "best.weights.h5"))
            path = self.dir / "best.weights.h5"

        self._meta["best_metric"] = metric_value
        self._meta["best_file"] = path.name
        self._meta["framework"] = framework
        self._save_meta()
        return True

    def warm_start(self, model) -> bool:
        """Load best weights into model. Returns True if loaded."""
        framework = _detect_model_framework(model)
        if not framework or framework != self._meta.get("framework", ""):
            return False

        best_file = self._meta.get("best_file", "")
        if not best_file:
            return False

        path = self.dir / best_file
        if not path.exists():
            return False

        try:
            if framework == "pytorch":
                import torch
                state = torch.load(path, weights_only=True)
                try:
                    model.load_state_dict(state, strict=True)
                except RuntimeError:
                    # Architecture mismatch (different combo) — skip warm-start
                    return False
            elif framework == "keras":
                try:
                    model.load_weights(str(path))
                except ValueError:
                    # Shape mismatch — skip warm-start
                    return False
            return True
        except Exception:
            return False

    @property
    def best_metric(self) -> float:
        return self._meta.get("best_metric", 0.0)

    @property
    def has_checkpoint(self) -> bool:
        best_file = self._meta.get("best_file", "")
        return bool(best_file) and (self.dir / best_file).exists()

    def clear(self) -> None:
        """Delete all checkpoints."""
        for f in self.dir.glob("best.*"):
            f.unlink()
        self._meta = {"best_metric": 0.0, "best_file": "", "framework": ""}
        self._save_meta()


def _detect_model_framework(model) -> str:
    """Detect if model is PyTorch or Keras."""
    cls_name = type(model).__module__
    if "torch" in cls_name:
        return "pytorch"
    if "keras" in cls_name or "tensorflow" in cls_name:
        return "keras"
    return ""
