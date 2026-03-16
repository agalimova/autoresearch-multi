"""
Warm-start checkpoint manager.

Saves the best model state after each run. Next run can load it
instead of training from scratch. Keeps only the best + latest.

Inspired by soveshmohapatra/autoresearch-2.0 and thenamangoyal/autoresearch.

Usage:
    ckpt = CheckpointManager("results/checkpoints")
    
    # After training:
    ckpt.save(model, optimizer, metric_value=0.93, config=config)
    
    # Before next training:
    state = ckpt.load_best()
    if state:
        model.load_state_dict(state["model"])
        print(f"Warm-starting from {state['metric']:.4f}")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


class CheckpointManager:
    """Manages best + latest checkpoints for warm-starting."""

    def __init__(self, checkpoint_dir: str | Path = "results/checkpoints"):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.dir / "meta.json"
        self._meta = self._load_meta()

    def _load_meta(self) -> dict:
        if self.meta_path.exists():
            return json.loads(self.meta_path.read_text())
        return {"best_metric": 0.0, "best_file": "", "latest_file": "", "count": 0}

    def _save_meta(self) -> None:
        self.meta_path.write_text(json.dumps(self._meta, indent=2))

    def save(self, model, optimizer, *, metric_value: float, config: dict) -> Path:
        """Save checkpoint. Keeps best + latest only."""
        try:
            import torch
        except ImportError:
            return Path()

        self._meta["count"] += 1
        count = self._meta["count"]
        filename = f"ckpt_{count}_acc{metric_value:.4f}.pt"
        path = self.dir / filename

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metric": metric_value,
            "config": config,
            "step": count,
        }, path)

        # Update latest
        old_latest = self._meta.get("latest_file", "")
        self._meta["latest_file"] = filename

        # Update best
        if metric_value > self._meta.get("best_metric", 0):
            old_best = self._meta.get("best_file", "")
            self._meta["best_metric"] = metric_value
            self._meta["best_file"] = filename
            # Delete old best (if it's not the new latest)
            if old_best and old_best != filename:
                (self.dir / old_best).unlink(missing_ok=True)
        
        # Delete old latest (if not best)
        if old_latest and old_latest != filename and old_latest != self._meta.get("best_file"):
            (self.dir / old_latest).unlink(missing_ok=True)

        self._save_meta()
        return path

    def load_best(self) -> Optional[dict]:
        """Load the best checkpoint. Returns dict with model, optimizer, metric, config."""
        try:
            import torch
        except ImportError:
            return None

        best_file = self._meta.get("best_file", "")
        if not best_file:
            return None
        path = self.dir / best_file
        if not path.exists():
            return None
        return torch.load(path, weights_only=False)

    def load_latest(self) -> Optional[dict]:
        """Load the most recent checkpoint."""
        try:
            import torch
        except ImportError:
            return None

        latest_file = self._meta.get("latest_file", "")
        if not latest_file:
            return None
        path = self.dir / latest_file
        if not path.exists():
            return None
        return torch.load(path, weights_only=False)

    @property
    def best_metric(self) -> float:
        return self._meta.get("best_metric", 0.0)

    @property
    def count(self) -> int:
        return self._meta.get("count", 0)

    def clear(self) -> None:
        """Delete all checkpoints."""
        for f in self.dir.glob("ckpt_*.pt"):
            f.unlink()
        self._meta = {"best_metric": 0.0, "best_file": "", "latest_file": "", "count": 0}
        self._save_meta()
