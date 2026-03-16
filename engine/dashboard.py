"""
Terminal dashboard for watching experiments in real-time.

Shows: live loss/accuracy, experiment history, best results, sparkline.
Works for both sklearn slot pipeline and PyTorch training demos.

Usage:
    from engine.dashboard import Dashboard
    dash = Dashboard()
    dash.update(metric_name="val_acc", metric_value=0.93, step=5, combo="xgb+onehot")
    dash.finish()

Or standalone:
    python -m engine.dashboard --watch results/  # watch a results directory
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def _term_width() -> int:
    try:
        return os.get_terminal_size().columns
    except Exception:
        return 80


def _sparkline(values: list[float], width: int = 30) -> str:
    """Render a sparkline from a list of values."""
    if not values:
        return ""
    try:
        "▁▂▃".encode(sys.stdout.encoding or "utf-8")
        blocks = " ▁▂▃▄▅▆▇█"
    except (UnicodeEncodeError, LookupError):
        blocks = " .-:=+*#@"
    lo, hi = min(values), max(values)
    rng = hi - lo if hi > lo else 1
    # Downsample to width
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values
    def _char(v):
        try:
            return blocks[min(8, int((v - lo) / rng * 8))]
        except (ValueError, OverflowError):
            return blocks[0]
    return "".join(_char(v) for v in sampled)


def _bar(value: float, max_val: float, width: int = 20) -> str:
    """Render a progress bar."""
    filled = int(value / max(max_val, 1e-9) * width)
    try:
        "█░".encode(sys.stdout.encoding or "utf-8")
        return "█" * filled + "░" * (width - filled)
    except (UnicodeEncodeError, LookupError):
        return "#" * filled + "-" * (width - filled)


class Dashboard:
    """Live terminal dashboard for experiment monitoring."""

    def __init__(self, title: str = "autoresearch"):
        self.title = title
        self.history: list[dict] = []
        self.current: dict = {}
        self.start_time = time.time()
        self._last_draw = 0

    def update(
        self,
        *,
        metric_name: str = "val_acc",
        metric_value: float = 0.0,
        step: int = 0,
        total_steps: int = 0,
        combo: str = "",
        status: str = "running",
        extra: dict | None = None,
    ) -> None:
        """Update the dashboard with new data."""
        self.current = {
            "metric_name": metric_name,
            "metric_value": metric_value,
            "step": step,
            "total_steps": total_steps,
            "combo": combo,
            "status": status,
            "elapsed": time.time() - self.start_time,
            **(extra or {}),
        }

        # Throttle redraws to 4/sec
        now = time.time()
        if now - self._last_draw < 0.25:
            return
        self._last_draw = now
        self._draw()

    def record(self, result: dict) -> None:
        """Record a completed experiment."""
        self.history.append(result)

    def finish(self) -> None:
        """Final draw with summary."""
        self.current["status"] = "done"
        self._draw()
        print()

    def _draw(self) -> None:
        """Render the dashboard to terminal."""
        w = min(_term_width(), 72)
        c = self.current

        # Clear and draw
        lines = []
        lines.append("-" * w)
        lines.append(f"  {self.title}  |  {c.get('status', '?')}  |  {c.get('elapsed', 0):.0f}s")
        lines.append("-" * w)

        # Current experiment
        metric = c.get("metric_name", "?")
        value = c.get("metric_value", 0)
        combo = c.get("combo", "")
        step = c.get("step", 0)
        total = c.get("total_steps", 0)

        if combo:
            lines.append(f"  Current: {combo}")
        lines.append(f"  {metric}: {value:.4f}")
        if total > 0:
            pct = step / total
            lines.append(f"  Progress: {_bar(pct, 1.0)} {step}/{total} ({pct:.0%})")

        # History sparkline
        if self.history:
            accs = [h.get("val_acc", h.get("metric_value", 0)) for h in self.history]
            best = max(accs)
            worst = min(accs)
            lines.append("-" * w)
            lines.append(f"  History ({len(self.history)} experiments)")
            lines.append(f"  {_sparkline(accs, width=w - 4)}")
            lines.append(f"  best: {best:.4f}  worst: {worst:.4f}  latest: {accs[-1]:.4f}")

        # Top 5
        if len(self.history) >= 2:
            sorted_h = sorted(self.history, key=lambda h: h.get("val_acc", 0), reverse=True)
            lines.append("-" * w)
            lines.append("  Top 5:")
            for i, h in enumerate(sorted_h[:5], 1):
                acc = h.get("val_acc", 0)
                label = h.get("combo", h.get("label", "?"))
                if isinstance(label, dict):
                    label = "+".join(v for v in label.values() if v != "base")
                lines.append(f"    {i}. {acc:.4f}  {str(label)[:w-16]}")

        lines.append("-" * w)

        # Move cursor up and overwrite
        n_lines = len(lines)
        if hasattr(self, "_prev_lines") and self._prev_lines > 0:
            sys.stdout.write(f"\033[{self._prev_lines}A")
        self._prev_lines = n_lines

        for line in lines:
            sys.stdout.write(f"\033[2K{line}\n")
        sys.stdout.flush()


def watch_results(results_dir: Path, interval: float = 1.0) -> None:
    """Watch a results directory and display dashboard."""
    dash = Dashboard(title=f"watching {results_dir}")
    seen = set()

    print(f"Watching {results_dir} for results (Ctrl+C to stop)...\n")

    try:
        while True:
            for f in sorted(results_dir.glob("*.json")):
                if f.name in seen:
                    continue
                seen.add(f.name)
                try:
                    data = json.loads(f.read_text())
                    if isinstance(data, list):
                        for entry in data:
                            dash.record(entry)
                            dash.update(
                                metric_value=entry.get("val_acc", 0),
                                combo=str(entry.get("combo", "")),
                                status="loaded",
                            )
                    elif isinstance(data, dict):
                        dash.record(data)
                        dash.update(
                            metric_value=data.get("val_acc", 0),
                            combo=str(data.get("combo", "")),
                            status="loaded",
                        )
                except Exception:
                    pass
            time.sleep(interval)
    except KeyboardInterrupt:
        dash.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", type=Path, default=Path("results"))
    args = parser.parse_args()
    watch_results(args.watch)
