"""
autoresearch-multi: point at your data, get results.

Usage:
    python run.py my_data.csv                   # CSV: auto-detect target, run search
    python run.py my_data.csv --target income   # CSV: specify target column
    python run.py my_pipeline.py                # Python: decompose into slots, search
    python run.py --dataset adult               # OpenML: fetch and search

Without an LLM API key: template-based search (fast, offline).
With ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY: LLM proposes novel code.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from engine.adaptive import AdaptiveSearch
from engine.slots.runner import SlotRunner
from engine.hardware import detect
from engine.dashboard import Dashboard


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_csv(path: str, target: str = ""):
    import pandas as pd
    df = pd.read_csv(path)
    if not target:
        target = df.columns[-1]
    print(f"Loaded {path}: {df.shape[0]} rows, {df.shape[1]} cols, target='{target}'")
    return df, target


def _load_openml(name: str):
    from sklearn.datasets import fetch_openml
    data = fetch_openml(name, version=1, as_frame=True, parser="auto")
    df = data.frame
    target = data.target.name if hasattr(data.target, "name") else df.columns[-1]
    print(f"Loaded OpenML/{name}: {df.shape[0]} rows, {df.shape[1]} cols, target='{target}'")
    return df, target


def _fingerprint(df, target: str) -> dict:
    import numpy as np
    y = df[target]
    X = df.drop(columns=[target])
    n_num = X.select_dtypes(include=[np.number]).shape[1]
    n_cat = X.select_dtypes(exclude=[np.number]).shape[1]
    n_cls = y.nunique()
    task = "regression" if n_cls > 20 else ("binary" if n_cls == 2 else "multiclass")
    print(f"  {task} | {n_num} numeric, {n_cat} categorical | {n_cls} classes | {df.isnull().mean().mean():.1%} missing")
    return {"rows": len(df), "cols": X.shape[1], "task": task}


# ── Slot setup ───────────────────────────────────────────────────────────────

def _setup_csv_slots(slot_dir: Path, csv_path: str, openml_name: str, target: str):
    """Generate sklearn slots for CSV/OpenML data."""
    from engine.variants import generate
    for d in ["load_data", "engineer_features", "build_model", "evaluate"]:
        (slot_dir / d).mkdir(parents=True, exist_ok=True)
    _write_loader(slot_dir / "load_data" / "base.py", csv_path, openml_name, target)
    _write_features(slot_dir / "engineer_features")
    generate("sklearn", slot_dir / "build_model")
    _write_evaluator(slot_dir / "evaluate" / "base.py")


def _setup_from_py(py_path: Path, slot_dir: Path):
    """Decompose a Python file into slots, auto-detect framework."""
    source = py_path.read_text()
    framework = "pytorch" if ("import torch" in source or "from torch" in source) else "sklearn"
    print(f"  Detected framework: {framework}")

    from engine.decompose import decompose_file
    from engine.variants import generate

    slots = decompose_file(py_path)
    for name, code in slots.items():
        d = slot_dir / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "base.py").write_text(code)
        print(f"  {name}/base.py ({len(code)} chars)")

    if not slots:
        print("  No slots detected.")
        return

    if (slot_dir / "build_model").exists():
        n = generate(framework, slot_dir / "build_model")
        print(f"  + {n} {framework} model variants")
    if (slot_dir / "engineer_features").exists() and framework == "sklearn":
        _write_features(slot_dir / "engineer_features")
        n = len(list((slot_dir / "engineer_features").glob("*.py"))) - 1
        print(f"  + {n} feature variants")


def _write_loader(path: Path, csv_path: str, openml_name: str, target: str):
    if csv_path:
        path.write_text(
            f'import pandas as pd\n\ndef load_data():\n    return pd.read_csv("{csv_path}"), "{target}"\n'
        )
    else:
        path.write_text(
            'from sklearn.datasets import fetch_openml\n\n'
            'def load_data():\n'
            f'    data = fetch_openml("{openml_name}", version=1, as_frame=True, parser="auto")\n'
            '    df = data.frame\n'
            '    target = data.target.name if hasattr(data.target, "name") else df.columns[-1]\n'
            '    return df, target\n'
        )


def _write_features(feat_dir: Path):
    """4 feature engineering strategies."""
    h = (
        'import pandas as pd\nimport numpy as np\n'
        'from sklearn.preprocessing import LabelEncoder\n\n'
        'def _prep(df, target):\n'
        '    y = df[target].copy()\n'
        '    X = df.drop(columns=[target]).copy()\n'
        '    if y.dtype == object or str(y.dtype) == "category":\n'
        '        y = LabelEncoder().fit_transform(y.astype(str).fillna("missing"))\n'
        '    num = X.select_dtypes(include=[np.number]).columns.tolist()\n'
        '    cat = X.select_dtypes(exclude=[np.number]).columns.tolist()\n'
        '    X[num] = X[num].fillna(0)\n'
        '    for c in cat:\n'
        '        X[c] = X[c].astype(str).fillna("missing")\n'
        '    return X, y, num, cat\n\n'
    )
    (feat_dir / "base.py").write_text(h +
        'def engineer_features(df, target):\n'
        '    X, y, num, cat = _prep(df, target)\n'
        '    for c in cat: X[c] = LabelEncoder().fit_transform(X[c])\n'
        '    return X, y\n')
    (feat_dir / "ordinal.py").write_text(
        'from sklearn.preprocessing import OrdinalEncoder\n' + h +
        'def engineer_features(df, target):\n'
        '    X, y, num, cat = _prep(df, target)\n'
        '    if cat: X[cat] = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(X[cat])\n'
        '    return X, y\n')
    (feat_dir / "onehot.py").write_text(h +
        'def engineer_features(df, target):\n'
        '    X, y, num, cat = _prep(df, target)\n'
        '    low = [c for c in cat if X[c].nunique() <= 20]\n'
        '    high = [c for c in cat if X[c].nunique() > 20]\n'
        '    if low: X = pd.get_dummies(X, columns=low, drop_first=True)\n'
        '    for c in high: X[c] = LabelEncoder().fit_transform(X[c])\n'
        '    return X, y\n')
    (feat_dir / "frequency.py").write_text(h +
        'def engineer_features(df, target):\n'
        '    X, y, num, cat = _prep(df, target)\n'
        '    for c in cat:\n'
        '        freq = X[c].value_counts(normalize=True)\n'
        '        X[c] = X[c].map(freq).fillna(0)\n'
        '    return X, y\n')


def _write_evaluator(path: Path):
    path.write_text(
        'from sklearn.model_selection import cross_val_score\n\n'
        'def evaluate(X, y, model):\n'
        '    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy", n_jobs=-1)\n'
        '    return {"val_acc": float(scores.mean())}\n')


# ── LLM round ───────────────────────────────────────────────────────────────

def _llm_rounds(runner, search, results, slot_dir, budget, n_rounds=3):
    """Iterative LLM: propose → test → read findings → propose better."""
    from engine.llm_proposer import SlotProposer
    from engine.semantic_diff import run_sem_diff

    all_results = list(results)
    best_so_far = max((r.get("val_acc", 0) for r in results), default=0)
    findings = []

    for rnd in range(1, n_rounds + 1):
        print(f"\n{'='*60}")
        print(f"LLM ROUND {rnd}/{n_rounds}")
        print(f"{'='*60}")

        proposer = SlotProposer(slot_dir, all_results)
        proposer.prior_findings = findings
        new_paths = proposer.propose_round(n=2)

        if not new_paths:
            print("  No new variants proposed")
            break

        # Sem diff proposals against current best
        best_r = max(all_results, key=lambda r: r.get("val_acc", 0))
        for path in new_paths:
            slot = path.parent.name
            best_impl = best_r.get("combo", {}).get(slot, "base")
            best_path = slot_dir / slot / f"{best_impl}.py"
            if best_path.exists():
                diff = run_sem_diff(best_path, path)
                print(f"  Diff vs {best_impl}: {diff.for_prompt()}")

        runner._cache.clear()
        runner._pipeline = None
        print(f"\n  Testing LLM proposals...")
        new_results = search.run(n_rounds=1)
        all_results.extend(new_results)

        round_best = max((r.get("val_acc", 0) for r in new_results), default=0)
        improved = round_best > best_so_far

        findings.append({
            "round": rnd,
            "proposed": [p.name for p in new_paths],
            "best_acc": round_best,
            "prev_best": best_so_far,
            "improved": improved,
            "insight": f"{'Improved' if improved else 'No improvement'}: {best_so_far:.2%} -> {round_best:.2%}",
        })

        if improved:
            print(f"  Round {rnd}: {best_so_far:.2%} -> {round_best:.2%} (+{round_best - best_so_far:.2%})")
            best_so_far = round_best
        else:
            print(f"  Round {rnd}: no improvement ({best_so_far:.2%})")

    template_best = max((r.get("val_acc", 0) for r in results), default=0)
    final_best = max((r.get("val_acc", 0) for r in all_results), default=0)
    status = "improved" if final_best > template_best else "no improvement"
    print(f"\n  LLM summary: {len(findings)} rounds, {status} ({template_best:.2%} -> {final_best:.2%})")
    return all_results


# ── Results ──────────────────────────────────────────────────────────────────

def _val(r, name="val_acc"):
    m = r.get("metrics") if isinstance(r, dict) else None
    if isinstance(m, dict) and name in m:
        return m[name]
    return r.get(name, 0) if isinstance(r, dict) else 0


def _print_results(results, name, elapsed, total, hw):
    results.sort(key=_val, reverse=True)
    print(f"\n{'='*60}")
    print(f"RESULTS: {name} ({len(results)} combos, {elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"\n{'Rank':<5}{'Features':<18}{'Model':<22}{'Acc':>8}")
    print("-" * 53)
    for i, r in enumerate(results[:10], 1):
        c = r.get("combo", {})
        print(f"  {i:<3}{c.get('engineer_features', '?'):<18}{c.get('build_model', '?'):<22}{_val(r):>7.2%}")

    accs = [_val(r) for r in results if not r.get("error")]
    if not accs:
        print("\nNo successful results.")
        return
    baseline = next((_val(r) for r in results if r.get("combo", {}).get("build_model") == "lr_default" and not r.get("error")), min(accs))
    print(f"\n{'Best':<25}{max(accs):>7.2%}")
    print(f"{'LogisticRegression base':<25}{baseline:>7.2%}")
    print(f"{'Gain':<25}{max(accs) - baseline:>+6.2%}")
    print(f"{'Combos tested':<25}{len(results):>7}")
    print(f"{'Total combos':<25}{total:>7}")
    print(f"{'Time':<25}{elapsed:>6.1f}s")
    print(f"{'Hardware':<25}{hw.device_name[:30]}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="autoresearch-multi: point at your data, get results.")
    parser.add_argument("csv", nargs="?", help="CSV or Python file")
    parser.add_argument("--dataset", help="OpenML dataset name")
    parser.add_argument("--target", default="", help="Target column")
    parser.add_argument("--budget", type=int, default=15, help="Combos per round")
    parser.add_argument("--rounds", type=int, default=3, help="Search rounds")
    args = parser.parse_args()

    if not args.csv and not args.dataset:
        parser.print_help()
        print("\nExamples:")
        print("  python run.py my_data.csv")
        print("  python run.py my_pipeline.py")
        print("  python run.py --dataset adult")
        sys.exit(1)

    # Hardware + LLM detection
    hw = detect()
    from engine.llm_proposer import has_llm, llm_name
    llm = has_llm()
    print(f"\nHardware: {hw}")
    print(f"Mode: {'llm (' + llm_name() + ')' if llm else 'template'}"
          + ("" if llm else " (set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY for LLM)"))

    # Setup slots based on input type
    is_py = args.csv and args.csv.endswith(".py")
    if is_py:
        name = Path(args.csv).stem
        slot_dir = Path(f"workspace/slots_{name}")
        if not slot_dir.exists():
            print(f"\nDecomposing {args.csv} into slots...")
            _setup_from_py(Path(args.csv), slot_dir)
    elif args.csv:
        df, target = _load_csv(args.csv, args.target)
        name = Path(args.csv).stem
        _fingerprint(df, target)
        slot_dir = Path(f"workspace/slots_{name}")
        if not slot_dir.exists():
            print(f"\nGenerating slots...")
            _setup_csv_slots(slot_dir, args.csv, "", target)
    else:
        df, target = _load_openml(args.dataset)
        name = args.dataset
        _fingerprint(df, target)
        slot_dir = Path(f"workspace/slots_{name}")
        if not slot_dir.exists():
            print(f"\nGenerating slots...")
            _setup_csv_slots(slot_dir, "", args.dataset, target)

    # Run search
    runner = SlotRunner(slot_dir)
    avail = runner.discover()
    n_feat = len(avail.get("engineer_features", []))
    n_model = len(avail.get("build_model", []))
    total = n_feat * n_model
    print(f"{n_feat} feature strategies x {n_model} models = {total} combinations\n")

    Path("results").mkdir(exist_ok=True)
    results_path = Path(f"results/{name}.json")
    dash = Dashboard(f"autoresearch-multi | {name}")
    t0 = time.time()

    search = AdaptiveSearch(runner, budget_per_round=args.budget, results_path=results_path)
    results = search.run(n_rounds=args.rounds)

    # LLM round if available
    if llm:
        results = _llm_rounds(runner, search, results, slot_dir, args.budget)

    elapsed = time.time() - t0

    for r in results:
        dash.record(r)
    if results:
        dash.update(metric_value=max(_val(r) for r in results), status="done")
    dash.finish()

    _print_results(results, name, elapsed, total, hw)

    results_path.write_text(json.dumps([{
        "combo": r.get("combo", {}), "val_acc": _val(r), "error": r.get("error"),
    } for r in results], indent=2))
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
