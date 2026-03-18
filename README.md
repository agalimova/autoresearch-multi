# autoresearch-multi

An extension of Karpathy's [autoresearch](https://github.com/karpathy/autoresearch). In a head-to-head using his GPU training pattern (Claude proposes `train.py` edits, fixed 30s budget, accept/reject on `val_bpb`) on a simplified GPT setup (RTX 4090, synthetic data), the coordinator lowered val_bpb to **1.451** vs Karpathy's **2.249** from the same 3.208 baseline. That is **1.8x more val_bpb reduction** in the same 5 rounds.

Also accepts tabular ML, prunes dead ends automatically, tests more combinations, detects interaction effects, and feeds every result back into the next decision.

## What it does

1. **Decomposes** Python ML pipelines into independent slots (`load_data`, `engineer_features`, `build_model`, `evaluate`) and tests combinations. Works turnkey with sklearn, PyTorch, TensorFlow, Keras, CatBoost, and statsmodels via `--framework`. Auto-detects framework from imports when given a `.py` file.

2. **Prunes dead ends.** After the first round, losing model families are killed, dominated impls within surviving families are archived, and impls that crash 3 times are auto-skipped. EXPLOIT, COMBINE, and NARROW skip all of them.

3. **Detects interactions.** Finds superadditive effects that sequential testing misses, then uses them to prioritize what to try next:
   ```
   Title features solo:    +2.79%
   XGBoost solo:           +0.00%
   Predicted combo:        +2.79%
   Actual combo:           +3.91%
   Interaction delta:      +1.12% (superadditive)
   ```

4. **Adaptive search** with 4 modes. Each round feeds results into the next. Auto-selects Optuna Bayesian when search space >20 variants.
   ```
   EXPLORE   coverage guarantee: one per model family x encoding
   EXPLOIT   Optuna Bayesian (large space) or grid (small space)
   COMBINE   near-miss pairs, sorted by prior interaction signal
   NARROW    fine-tune within winning region
   ```

5. **Proposes code via LLM** (when API key is set). The LLM sees per-slot rankings, interaction effects, and dead families, not just flat results. Proposals are tested directly against the current best combo instead of restarting search. Code diffs show what changed.

6. **Works on CSVs, OpenML datasets, Python pipelines, and GPU training scripts.** The tabular backend handles the standard `load_data`, `engineer_features`, `build_model`, `evaluate` pattern. The GPU backend wraps any training script (like Karpathy's `train.py`) behind the same interface. Custom backends can be added by implementing the `Backend` protocol.

## Install

```bash
pip install autoresearch-multi
```

With optional frameworks:

```bash
pip install autoresearch-multi[torch]      # + PyTorch
pip install autoresearch-multi[keras]      # + TensorFlow/Keras
pip install autoresearch-multi[catboost]   # + CatBoost
pip install autoresearch-multi[llm]        # + Claude API
pip install autoresearch-multi[llm-all]    # + all LLM providers (Claude, GPT, Gemini)
pip install autoresearch-multi[all]        # everything
```

Or from source:

```bash
git clone https://github.com/agalimova/autoresearch-multi.git
cd autoresearch-multi
pip install -e .
```

## Usage

**Tabular ML** (CSV, OpenML, Python pipelines):
```bash
autoresearch my_data.csv                           # sklearn (default)
autoresearch my_data.csv --framework pytorch       # PyTorch neural nets
autoresearch my_data.csv --framework keras          # Keras
autoresearch my_data.csv --framework catboost       # CatBoost
autoresearch my_pipeline.py                         # point at a Python file (auto-decomposes)
autoresearch --dataset adult                        # use any OpenML dataset
```

**GPU training** (Karpathy-style, any training script):
```python
from engine.backends.gpu_training import GpuTrainingBackend
from engine.coordinator import Coordinator

backend = GpuTrainingBackend(project_dir="path/to/project", binary="python train.py")
Coordinator(backend, budget_per_round=10).run(n_rounds=4, llm_rounds=3)
```

**Custom backend** (implement the protocol, plug into the coordinator):
```python
from engine.backend import Backend, ExperimentResult

class MyBackend:
    metric_name = "val_acc"
    higher_is_better = True
    def run_experiment(self, config): ...
    def get_search_space(self): ...
    # see engine/backend.py for full protocol
```

Without an API key: template-based search. With any LLM key, it proposes novel model code and iteratively improves. Supports Claude, GPT-4o, Gemini, OpenRouter, Together AI, Deepseek, and Ollama (local, no key needed).

## What's different from Karpathy's autoresearch

Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) runs a single loop: LLM proposes a change to `train.py`, trains for 5 minutes, keeps or reverts. Sequential accept/reject, no memory of what failed.

autoresearch-multi adds:

| | Karpathy | autoresearch-multi |
|---|---|---|
| Search | Sequential accept/reject | 4-mode adaptive (EXPLORE/EXPLOIT/COMBINE/NARROW) |
| History | `results.tsv` (agent manages manually) | `ExperimentTracker` — auto-saves after every run, feeds history into LLM prompt |
| Pruning | None — agent may retry dead ends | Auto-kills families that lose, options that crash 3x, dominated variants |
| Interactions | Never tested — single change at a time | Detects superadditive combos (e.g. PCA+KNN: +25% over predicted) |
| Backend | GPU training only (`train.py`) | Pluggable — tabular ML, GPU training, or any custom backend |
| LLM context | Current code + current metric | Full experiment history + dead families + interaction effects + crash log |

### Head-to-head: GPU training on RTX 4090

Both methods use Claude to propose changes to `train.py`, train for 30 seconds, measure `val_bpb` (lower is better). Same model, same data, same budget, 5 rounds each.

| Method | Baseline | Best val_bpb | Improvement |
|---|---|---|---|
| Karpathy (sequential) | 3.208 | 2.249 | 0.96 |
| **Coordinator (history+pruning)** | 3.208 | **1.451** | **1.76** |

Coordinator won — **1.8x more val_bpb reduction** in the same 5 rounds. Karpathy's loop got one good result on round 2 then wasted 3 rounds on regressions. The coordinator recovered from a crash on round 4 and beat its own best on round 5 because Claude could see the full history.

### Tabular ML comparison

Same search space (3 feature variants x 6 model variants), no LLM needed.

```
Method                    Best    Runs      Time    Runs/s
----------------------------------------------------------
Karpathy (sequential)  0.9833       8     30.3s      0.3
AdaptiveSearch (old)   0.9833      15      5.5s      2.7
Coordinator (new)      0.9833      15      4.5s      3.3
```

All three find the same best, but the coordinator is 6.7x faster and catches interaction effects that sequential search misses.

## Results

### LLM mode

Claude proposes novel code (ensembles, custom feature engineering) and iteratively improves across 3 rounds. Example on King-Rook vs King-Pawn, a chess endgame dataset (36 categorical columns, binary classification):

```
Templates:      98.06%  (24 combos, 21s)
LLM Round 1:    99.40%  (+1.33%)
LLM Round 2:    99.44%  (+0.04%)
LLM Round 3:   100.00%  (+0.56%, perfect accuracy)
```

### Template vs LLM (same datasets)

| Dataset | Baseline | Template | Percentile | With Claude | Percentile | Combos | Time |
|---|---|---|---|---|---|---|---|
| heart-statlog | 84.81% | 84.81% | top 25% | **97.33%** | top 1% | 45 | 114s |
| kr-vs-kp | 93.31% | 98.06% | top 5% | **100.00%** | top 1% | 48 | 133s |
| diabetes | 77.22% | 77.22% | top 5% | **77.74%** | top 5% | 33 | 94s |

## Features from the community

| Feature | Source |
|---|---|
| Entity-level code diffs | [sem](https://github.com/Ataraxy-Labs/sem) |
| Optuna exploit | [soveshmohapatra](https://github.com/soveshmohapatra/autoresearch-2.0) |
| Resume support | [buzypi](https://github.com/buzypi/autoresearch) |
| Warm-start checkpoints | [soveshmohapatra](https://github.com/soveshmohapatra/autoresearch-2.0) |
| Hardware auto-detection | [elementalcollision](https://github.com/elementalcollision/autoresearch) |
| Scaling heuristic | [Sreebhargavibalijaa](https://github.com/Sreebhargavibalijaa/autoresearch-karpathy) |
| TUI dashboard | [elementalcollision](https://github.com/elementalcollision/autoresearch) |
| Experiment dedup | [mutable-state-inc](https://github.com/mutable-state-inc/autoresearch-at-home) |

## Project structure

```
run.py                       CLI entry point (118 lines)
engine/
  coordinator.py             4-mode adaptive search + LLM rounds
  backend.py                 Backend protocol + ExperimentTracker
  setup.py                   data loading, code templates, model variants, slot generation
  backends/
    tabular.py               wraps SlotRunner for tabular ML
    gpu_training.py           wraps any training script for GPU experiments
  llm_proposer.py            LLM code proposal (Claude, GPT, Gemini, Ollama)
  decompose.py               auto-split Python files into slots
  slots/runner.py            generic pipeline runner + combinatorial testing
  dashboard.py               TUI dashboard
  hardware.py                hardware auto-detection
  checkpoint.py              warm-start best model across combos
  scaling.py                 swarm scaling heuristic (experimental)
  semantic_diff.py           entity-level code diffs (requires `sem` CLI)
  merge.py                   multi-agent A/B merge (requires `sem` CLI)
extras/
  telemetry/                 experiment telemetry (opt-out via AUTORESEARCH_TELEMETRY=0)
```

`run.py` wires `Coordinator` + `TabularBackend`. The coordinator drives search through the `Backend` protocol without knowing what runs underneath. `GpuTrainingBackend` plugs in the same way for GPU training.

## Telemetry

Collects experiment telemetry by default. Saves locally to `results/telemetry.jsonl`. Includes dataset name, best model, combo count, and timing.

**Disable:** `export AUTORESEARCH_TELEMETRY=0`

## License

MIT
