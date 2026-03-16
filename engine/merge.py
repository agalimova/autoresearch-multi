"""
File-level entity merge with A/B testing of all combinations.

When two agents modify the same file:
  - Non-overlapping entities: clean merge (splice both)
  - Overlapping entities: A/B test all combos, best wins

The merge itself becomes a finding with metrics for each combination tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.semantic_diff import run_sem_diff, SemanticDiff


@dataclass
class ComboResult:
    """One combination tested during merge."""
    name: str         # "base", "a_only", "b_only", "a_then_b", "b_then_a"
    metric: float
    content: str      # the file content that produced this metric


@dataclass
class MergeResult:
    """Outcome of merging two agents' changes."""
    clean: bool
    merged_content: Optional[str]
    applied_from_a: list[str]
    applied_from_b: list[str]
    conflicts: list[str]
    # A/B test results (populated only for overlapping entities)
    combos_tested: list[ComboResult] = field(default_factory=list)
    best_combo: Optional[str] = None


def merge_files(
    base_path: Path,
    modified_a: Path,
    modified_b: Path,
    *,
    evaluate_fn: Optional[Callable[[Path], dict]] = None,
    metric_name: str = "val_acc",
    higher_is_better: bool = True,
    work_dir: Optional[Path] = None,
) -> MergeResult:
    """
    Merge two modified versions of a file.

    Non-overlapping entities: splice directly.
    Overlapping entities: A/B test all combinations if evaluate_fn is provided.
    """
    diff_a = run_sem_diff(base_path, modified_a)
    diff_b = run_sem_diff(base_path, modified_b)

    entities_a = {c.entity_name for c in diff_a.changes}
    entities_b = {c.entity_name for c in diff_b.changes}

    only_a = entities_a - entities_b
    only_b = entities_b - entities_a
    overlapping = entities_a & entities_b

    # No conflicts: clean merge
    if not overlapping:
        merged = _splice_entities(base_path, modified_a, modified_b, diff_a, diff_b)
        return MergeResult(
            clean=True,
            merged_content=merged,
            applied_from_a=sorted(entities_a),
            applied_from_b=sorted(only_b),
            conflicts=[],
        )

    # Overlapping entities — A/B test if we have an evaluate function
    if evaluate_fn is None:
        return MergeResult(
            clean=False,
            merged_content=None,
            applied_from_a=sorted(only_a),
            applied_from_b=sorted(only_b),
            conflicts=sorted(overlapping),
        )

    # Run A/B test on all combinations
    return _ab_test_merge(
        base_path, modified_a, modified_b,
        only_a=only_a, only_b=only_b, overlapping=overlapping,
        evaluate_fn=evaluate_fn,
        metric_name=metric_name,
        higher_is_better=higher_is_better,
        work_dir=work_dir or base_path.parent / "merge_test",
    )


def _ab_test_merge(
    base_path: Path,
    modified_a: Path,
    modified_b: Path,
    *,
    only_a: set[str],
    only_b: set[str],
    overlapping: set[str],
    evaluate_fn: Callable[[Path], dict],
    metric_name: str,
    higher_is_better: bool,
    work_dir: Path,
) -> MergeResult:
    """
    A/B test all combinations of overlapping changes.

    Tests:
      1. base        — neither agent's changes
      2. a_only      — agent A's version
      3. b_only      — agent B's version
      4. a_then_b    — start with A, layer B's non-overlapping on top
      5. b_then_a    — start with B, layer A's non-overlapping on top

    The best-performing combination wins.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    base_content = base_path.read_text()
    a_content = modified_a.read_text()
    b_content = modified_b.read_text()

    candidates = {
        "base": base_content,
        "a_only": a_content,
        "b_only": b_content,
    }

    # For "a_then_b": take A's file, splice in B's non-overlapping entities
    if only_b:
        diff_a = run_sem_diff(base_path, modified_a)
        diff_b = run_sem_diff(base_path, modified_b)
        a_then_b = _splice_specific(a_content, b_content, only_b)
        candidates["a_then_b"] = a_then_b

    # For "b_then_a": take B's file, splice in A's non-overlapping entities
    if only_a:
        b_then_a = _splice_specific(b_content, a_content, only_a)
        candidates["b_then_a"] = b_then_a

    # Evaluate each combination
    combos: list[ComboResult] = []
    for name, content in candidates.items():
        test_file = work_dir / f"{name}_{base_path.name}"
        test_file.write_text(content)
        try:
            results = evaluate_fn(test_file)
            metric = results.get(metric_name, 0.0)
        except Exception:
            metric = 0.0
        combos.append(ComboResult(name=name, metric=metric, content=content))

    # Find the best
    if higher_is_better:
        best = max(combos, key=lambda c: c.metric)
    else:
        best = min(combos, key=lambda c: c.metric)

    # Report
    return MergeResult(
        clean=False,  # had conflicts, but resolved via A/B test
        merged_content=best.content,
        applied_from_a=sorted(only_a) if "a" in best.name else [],
        applied_from_b=sorted(only_b) if "b" in best.name else [],
        conflicts=sorted(overlapping),
        combos_tested=combos,
        best_combo=best.name,
    )


def _splice_entities(
    base_path: Path,
    modified_a: Path,
    modified_b: Path,
    diff_a: SemanticDiff,
    diff_b: SemanticDiff,
) -> str:
    """Splice non-overlapping entity changes from both files."""
    a_content = modified_a.read_text()
    b_content = modified_b.read_text()

    entities_a = {c.entity_name for c in diff_a.changes}
    entities_b = {c.entity_name for c in diff_b.changes}
    only_b = entities_b - entities_a

    if not only_b:
        return a_content

    return _splice_specific(a_content, b_content, only_b)


def _splice_specific(
    base_content: str,
    donor_content: str,
    entities_to_splice: set[str],
) -> str:
    """
    Take entities from donor and splice them into base.

    For added functions: append to end.
    For modified functions: replace in-place using definition boundaries.
    """
    base_defs = _extract_definitions(base_content)
    donor_defs = _extract_definitions(donor_content)

    result = base_content

    for name in entities_to_splice:
        if name not in donor_defs:
            continue

        if name in base_defs:
            # Replace existing definition
            result = result.replace(base_defs[name], donor_defs[name])
        else:
            # Append new definition
            result = result.rstrip() + "\n\n" + donor_defs[name] + "\n"

    return result


def _extract_definitions(content: str) -> dict[str, str]:
    """Extract top-level function and class definitions."""
    lines = content.splitlines()
    defs: dict[str, str] = {}
    current_name: Optional[str] = None
    current_lines: list[str] = []

    for line in lines:
        stripped = line.lstrip()
        is_def = stripped.startswith("def ") or stripped.startswith("class ")
        is_top_level = not line.startswith(" ") and not line.startswith("\t")

        if is_def and is_top_level:
            if current_name:
                defs[current_name] = "\n".join(current_lines)
            if stripped.startswith("def "):
                current_name = stripped.split("(")[0].replace("def ", "").strip()
            else:
                current_name = stripped.split("(")[0].split(":")[0].replace("class ", "").strip()
            current_lines = [line]
        elif current_name is not None:
            if line.strip() == "" or line.startswith(" ") or line.startswith("\t"):
                current_lines.append(line)
            else:
                defs[current_name] = "\n".join(current_lines)
                current_name = None
                current_lines = []

    if current_name:
        defs[current_name] = "\n".join(current_lines)

    return defs
