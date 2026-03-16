"""
Auto-decomposer: takes a project file (or directory) and creates a slot directory.

Uses the Rust analyzer (agent-analysis) for slot detection, then extracts
the relevant source code into individual slot files.

Usage:
    # Single file
    python -m engine.decompose workspace/titanic_model.py --output workspace/slots_auto

    # Multi-file project
    python -m engine.decompose workspace/cifar10/ --output workspace/slots_cifar_auto

    # Dry run (show what would be created)
    python -m engine.decompose workspace/titanic_model.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional


def run_analyzer(filepath: Path, command: str = "slots") -> dict:
    """Run the Rust analyzer on a Python file."""
    import shutil
    # Try PATH first, then known locations
    analyzer_name = shutil.which("analyze")
    if analyzer_name:
        analyzer = Path(analyzer_name)
    else:
        repo_root = Path(__file__).resolve().parent.parent
        for suffix in ["", ".exe"]:
            for build in ["release", "debug"]:
                for base in ["tools/agent-analysis", "agent"]:
                    candidate = repo_root / base / "target" / build / f"analyze{suffix}"
                    if candidate.exists():
                        analyzer = candidate
                        break
                else:
                    continue
                break
            else:
                continue
            break
        else:
            raise FileNotFoundError(
                "Rust analyzer not found. Run: cd tools/agent-analysis && cargo build --release"
            )

    result = subprocess.run(
        [str(analyzer), command, str(filepath)],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Analyzer failed on {filepath}: {result.stderr}")
    return json.loads(result.stdout)


def extract_lines(source: str, start: int, end: int) -> str:
    """Extract lines start..end (1-indexed) from source."""
    lines = source.splitlines(keepends=True)
    return "".join(lines[start - 1 : end])


KNOWN_SLOTS = {
    "load_data": "load_data",
    "engineer_features": "engineer_features",
    "build_model": "build_model",
    "evaluate": "evaluate",
    "build_optimizer": "build_optimizer",
    "get_transforms": "get_transforms",
    "vectorize": "vectorize",
    "preprocess": "engineer_features",
    "create_model": "build_model",
    "make_model": "build_model",
    "train": "evaluate",
    "train_and_evaluate": "evaluate",
}


def _decompose_with_ast(filepath: Path, source: str) -> dict[str, str]:
    """Fallback decomposer using Python AST. No Rust required."""
    import ast

    tree = ast.parse(source)
    lines = source.splitlines()

    # Collect imports
    imports = []
    for line in lines:
        s = line.strip()
        if s.startswith("import ") or s.startswith("from "):
            imports.append(line)
        elif s and not s.startswith("#") and not s.startswith('"""'):
            break
    imports_block = "\n".join(imports)

    # Find functions that match known slot names
    slots: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        name = node.name
        slot_name = KNOWN_SLOTS.get(name)
        if not slot_name:
            continue
        start = node.lineno - 1
        end = node.end_lineno or (start + 1)
        func_source = "\n".join(lines[start:end])
        slot_source = imports_block + "\n\n\n" + func_source + "\n"
        slots[slot_name] = slot_source

    return slots


def decompose_file(filepath: Path) -> dict[str, str]:
    """
    Decompose a single Python file into slot files.

    Returns {slot_name: source_code} for each detected slot.
    Uses Rust analyzer if available, falls back to Python AST.
    """
    source = filepath.read_text()
    try:
        slots_info = run_analyzer(filepath, "slots")
        entities_info = run_analyzer(filepath, "entities")
    except FileNotFoundError:
        return _decompose_with_ast(filepath, source)

    # Build entity line ranges lookup
    entity_ranges: dict[str, tuple[int, int]] = {}
    for e in entities_info:
        entity_ranges[e["name"]] = (e["start_line"], e["end_line"])

    # Collect imports from the original file
    imports = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(line)
        elif stripped and not stripped.startswith("#") and not stripped.startswith('"""'):
            # Stop at first non-import, non-comment line
            if not stripped.startswith("import") and not stripped.startswith("from"):
                break

    imports_block = "\n".join(imports)

    # Extract each slot's functions
    slot_files: dict[str, str] = {}
    for slot in slots_info.get("slots", []):
        slot_name = slot["slot_name"]
        functions = slot["functions"]

        # Extract source for each function in this slot
        func_sources = []
        for func_name in functions:
            if func_name in entity_ranges:
                start, end = entity_ranges[func_name]
                func_source = extract_lines(source, start, end)
                func_sources.append(func_source)

        if not func_sources:
            continue

        # Build the slot file: imports + extracted functions
        entry = slot["entry_function"]
        slot_source = f'"""{slot_name} slot. Entry: {entry}."""\n\n'
        slot_source += imports_block + "\n\n\n"
        slot_source += "\n\n".join(func_sources)
        slot_source += "\n"

        slot_files[slot_name] = slot_source

    # Shared: module-level code not in any slot
    shared_entities = slots_info.get("shared", [])
    if shared_entities:
        shared_sources = []
        for name in shared_entities:
            if name in entity_ranges:
                start, end = entity_ranges[name]
                shared_sources.append(extract_lines(source, start, end))
        if shared_sources:
            shared_source = f'"""Shared code from {filepath.name}."""\n\n'
            shared_source += imports_block + "\n\n\n"
            shared_source += "\n\n".join(shared_sources)
            slot_files["_shared"] = shared_source

    return slot_files


def decompose_project(project_dir: Path) -> dict[str, str]:
    """
    Decompose a multi-file project into slots.

    Runs the analyzer on each .py file, merges slot assignments.
    If two files contribute to the same slot, their code is combined.
    """
    all_slots: dict[str, list[str]] = {}

    for pyfile in sorted(project_dir.glob("*.py")):
        if pyfile.name.startswith("_"):
            continue
        try:
            file_slots = decompose_file(pyfile)
            for slot_name, source in file_slots.items():
                if slot_name not in all_slots:
                    all_slots[slot_name] = []
                all_slots[slot_name].append(f"# From {pyfile.name}\n{source}")
        except Exception as e:
            print(f"  Warning: {pyfile.name}: {e}", flush=True)

    # Merge sources for each slot
    merged: dict[str, str] = {}
    for slot_name, sources in all_slots.items():
        merged[slot_name] = "\n\n".join(sources)

    return merged


def write_slots(slot_files: dict[str, str], output_dir: Path):
    """Write slot files to the output directory."""
    for slot_name, source in slot_files.items():
        if slot_name == "_shared":
            slot_dir = output_dir
            filename = "_shared.py"
        else:
            slot_dir = output_dir / slot_name
            filename = "base.py"

        slot_dir.mkdir(parents=True, exist_ok=True)
        (slot_dir / filename).write_text(source)
        print(f"  Created {slot_dir / filename}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Auto-decompose Python code into slots")
    parser.add_argument("path", type=Path, help="Python file or project directory")
    parser.add_argument("--output", type=Path, default=None, help="Output slot directory")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without writing")
    args = parser.parse_args()

    path = args.path.resolve()
    output = args.output or Path(f"workspace/slots_{path.stem}")

    if path.is_file():
        print(f"Decomposing {path.name}...", flush=True)
        slot_files = decompose_file(path)
    elif path.is_dir():
        print(f"Decomposing {path.name}/ ({len(list(path.glob('*.py')))} files)...", flush=True)
        slot_files = decompose_project(path)
    else:
        print(f"Error: {path} not found")
        sys.exit(1)

    print(f"\nDetected {len(slot_files)} slots:")
    for name, source in slot_files.items():
        n_lines = len(source.splitlines())
        print(f"  {name}: {n_lines} lines")

    if args.dry_run:
        print("\nDry run — no files written.")
        for name, source in slot_files.items():
            print(f"\n--- {name} ---")
            print(source[:300])
            if len(source) > 300:
                print("...")
    else:
        print(f"\nWriting to {output}/")
        write_slots(slot_files, output)
        print(f"\nDone. Run the slot pipeline:")
        print(f"  PYTHONPATH=. python -c \"from engine.slots.runner import SlotRunner; "
              f"from pathlib import Path; "
              f"runner = SlotRunner(Path('{output}')); "
              f"print(runner.run_all_combos())\"")



if __name__ == "__main__":
    main()
