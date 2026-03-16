"""
Semantic diff integration via the `sem` CLI.

Wraps `sem diff file_a file_b --format json` to produce structured,
entity-level diffs that agents can read.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EntityChange:
    """One entity-level change detected by sem."""
    entity_name: str
    entity_type: str  # function, class, etc.
    change_type: str  # added, modified, deleted, renamed
    file_path: str
    before_content: str | None = None
    after_content: str | None = None


@dataclass
class SemanticDiff:
    """The full semantic diff between two versions of a file."""
    changes: list[EntityChange]
    summary: str  # human-readable summary

    def to_dict(self) -> dict:
        return {
            "changes": [
                {
                    "entity_name": c.entity_name,
                    "entity_type": c.entity_type,
                    "change_type": c.change_type,
                    "file_path": c.file_path,
                }
                for c in self.changes
            ],
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SemanticDiff:
        changes = [EntityChange(**c) for c in data.get("changes", [])]
        return cls(changes=changes, summary=data.get("summary", ""))

    def for_prompt(self) -> str:
        """Format for inclusion in an agent prompt."""
        if not self.changes:
            return "No code changes."
        lines = ["Code changes:"]
        for c in self.changes:
            symbol = {"added": "+", "modified": "~", "deleted": "-", "renamed": ">"}.get(
                c.change_type, "?"
            )
            lines.append(f"  {symbol} {c.entity_type} {c.entity_name} [{c.change_type}]")
        lines.append(f"Summary: {self.summary}")
        return "\n".join(lines)


def run_sem_diff(base_path: Path, modified_path: Path) -> SemanticDiff:
    """
    Run `sem diff` between two files and parse the JSON output.

    Falls back to a basic text comparison if sem is not installed.
    """
    try:
        result = subprocess.run(
            ["sem", "diff", str(base_path), str(modified_path), "--format", "json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return _parse_sem_output(result.stdout, str(modified_path))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: basic text diff
    return _fallback_diff(base_path, modified_path)


def _parse_sem_output(raw_json: str, file_path: str) -> SemanticDiff:
    """Parse sem's JSON output into our SemanticDiff."""
    data = json.loads(raw_json)

    changes = []
    for entry in data.get("changes", []):
        changes.append(EntityChange(
            entity_name=entry.get("entityName", "unknown"),
            entity_type=entry.get("entityType", "unknown"),
            change_type=entry.get("changeType", "unknown"),
            file_path=file_path,
            before_content=entry.get("beforeContent"),
            after_content=entry.get("afterContent"),
        ))

    summary_data = data.get("summary", {})
    if isinstance(summary_data, dict):
        added = summary_data.get("added", 0)
        modified = summary_data.get("modified", 0)
        deleted = summary_data.get("deleted", 0)
        summary = f"{added} added, {modified} modified, {deleted} deleted"
    else:
        counts = {"added": 0, "modified": 0, "deleted": 0}
        for c in changes:
            counts[c.change_type] = counts.get(c.change_type, 0) + 1
        summary = ", ".join(f"{v} {k}" for k, v in counts.items() if v > 0)

    return SemanticDiff(changes=changes, summary=summary or "no changes")


def _fallback_diff(base_path: Path, modified_path: Path) -> SemanticDiff:
    """Simple line-based fallback when sem is not available."""
    base_lines = base_path.read_text().splitlines() if base_path.exists() else []
    mod_lines = modified_path.read_text().splitlines() if modified_path.exists() else []

    if base_lines == mod_lines:
        return SemanticDiff(changes=[], summary="no changes")

    return SemanticDiff(
        changes=[EntityChange(
            entity_name=modified_path.name,
            entity_type="file",
            change_type="modified",
            file_path=str(modified_path),
        )],
        summary="file modified (sem not available, line-level diff only)",
    )
