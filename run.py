"""
autoresearch-multi: point at your data, get results.

    autoresearch my_data.csv                        # sklearn (default)
    autoresearch my_data.csv --framework pytorch    # PyTorch neural nets
    autoresearch my_data.csv --framework keras       # Keras
    autoresearch my_data.csv --framework catboost    # CatBoost
    autoresearch my_pipeline.py                      # decompose existing code
    autoresearch --dataset adult                     # OpenML dataset
"""

import argparse
import json
import sys
import time
from pathlib import Path

from engine.coordinator import Coordinator
from engine.backends.tabular import TabularBackend
from engine.hardware import detect

FRAMEWORKS = ["sklearn", "pytorch", "keras", "tensorflow", "statsmodels", "catboost"]


def main():
    p = argparse.ArgumentParser(description="autoresearch-multi: point at your data, get results.")
    p.add_argument("input", nargs="?", help="CSV file or Python file")
    p.add_argument("--dataset", help="OpenML dataset name")
    p.add_argument("--target", default="", help="Target column")
    p.add_argument("--framework", default="sklearn", choices=FRAMEWORKS)
    p.add_argument("--budget", type=int, default=15, help="Combos per round")
    p.add_argument("--rounds", type=int, default=4, help="Search rounds")
    p.add_argument("--llm-rounds", type=int, default=3, help="LLM proposal rounds")
    args = p.parse_args()

    if not args.input and not args.dataset:
        p.print_help()
        print("\nExamples:\n  autoresearch data.csv\n  autoresearch data.csv --framework pytorch\n  autoresearch --dataset adult")
        sys.exit(1)

    # Environment
    hw = detect()
    from engine.llm_proposer import has_llm, llm_name
    llm = has_llm()
    print(f"\nHardware: {hw}")
    print(f"Mode: {'llm (' + llm_name() + ')' if llm else 'template (set ANTHROPIC_API_KEY for LLM)'}")

    # Resolve input -> slot_dir
    from engine import setup
    is_py = args.input and args.input.endswith(".py")

    if is_py:
        name = Path(args.input).stem
        slot_dir = Path(f"workspace/slots_{name}")
        if not slot_dir.exists():
            print(f"\nDecomposing {args.input}...")
            setup.from_py(Path(args.input), slot_dir)
    else:
        from engine.setup import load
        _, target, name = load(csv_path=args.input, dataset=args.dataset, target=args.target)
        slot_dir = Path(f"workspace/slots_{name}_{args.framework}")
        if not slot_dir.exists():
            print(f"\nGenerating {args.framework} slots...")
            setup.from_data(slot_dir, csv_path=args.input or "", dataset=args.dataset or "",
                            target=target, framework=args.framework)

    # Search
    backend = TabularBackend(slot_dir=slot_dir)
    space = backend.get_search_space()
    total = 1
    for opts in space.values():
        total *= len(opts)
    dims = {k: len(v) for k, v in space.items() if len(v) > 1}
    print(" x ".join(f"{n} {k}" for k, n in dims.items()) + f" = {total} combinations\n")

    t0 = time.time()
    tracker_id = f"{name}_{args.framework}" if not is_py else name
    coord = Coordinator(backend, budget_per_round=args.budget,
                        results_dir=Path("results"), tracker_name=tracker_id)
    coord.run(n_rounds=args.rounds, llm_rounds=args.llm_rounds if llm else 0)
    elapsed = time.time() - t0

    # Results
    results = coord.results_as_dicts()
    results.sort(key=lambda r: r.get("val_acc", 0), reverse=True)

    from engine.dashboard import Dashboard
    dash = Dashboard(f"autoresearch | {name} | {args.framework}")
    for r in results:
        dash.record(r)
    if results:
        dash.update(metric_value=results[0].get("val_acc", 0), status="done")
    dash.finish()

    accs = [r["val_acc"] for r in results if not r.get("error")]
    print(f"\n{'='*60}")
    print(f"RESULTS: {name} ({len(results)} combos, {elapsed:.1f}s, {hw.device_name[:30]})")
    print(f"{'='*60}")
    for i, r in enumerate(results[:10], 1):
        c = r.get("combo", {})
        feats = c.get("engineer_features", "")
        model = c.get("build_model", "?")
        label = f"{feats}+{model}" if feats and feats != "base" else model
        print(f"  {i:>2}. {r['val_acc']:7.2%}  {label}")
    if accs:
        print(f"\n  Best: {max(accs):.2%} | Tested: {len(results)}/{total} | Time: {elapsed:.1f}s")

    out = Path(f"results/{tracker_id}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        [{"combo": r["combo"], "val_acc": r["val_acc"], "error": r.get("error")} for r in results],
        indent=2,
    ))
    print(f"  Saved: {out}\n")


if __name__ == "__main__":
    main()
