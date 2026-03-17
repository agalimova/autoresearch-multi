# autoresearch-multi

An extension of Karpathy's [autoresearch](https://github.com/karpathy/autoresearch). Accepts additional ML frameworks, tests more combinations, wastes fewer runs, and feeds every result back into the next decision.

## What it does

1. **Decomposes** Python ML pipelines into independent slots (`load_data`, `engineer_features`, `build_model`, `evaluate`) and tests combinations. Detects the framework from imports and generates appropriate model variants. Supports sklearn, PyTorch, TensorFlow, Keras, HuggingFace, and statsmodels.

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

6. **Works on CSVs, OpenML datasets, and Python pipelines** that follow the standard `load_data`, `engineer_features`, `build_model`, `evaluate` pattern.

## Install

```bash
pip install autoresearch-multi
```

With optional frameworks:

```bash
pip install autoresearch-multi[torch]    # + PyTorch
pip install autoresearch-multi[keras]    # + TensorFlow/Keras
pip install autoresearch-multi[llm]      # + Claude API
pip install autoresearch-multi[all]      # everything
```

Or from source:

```bash
git clone https://github.com/agalimova/autoresearch-multi.git
cd autoresearch-multi
pip install -e .
```

## Usage

```bash
autoresearch my_data.csv                   # point at a CSV
autoresearch my_pipeline.py                # point at a Python file (auto-decomposes)
autoresearch --dataset adult               # use any OpenML dataset
```

Without an API key: template-based search. With any LLM key, it proposes novel model code and iteratively improves. Supports Claude, GPT-4o, Gemini, OpenRouter, Together AI, Deepseek, and Ollama (local, no key needed).

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
| Scaling law | [Sreebhargavibalijaa](https://github.com/Sreebhargavibalijaa/autoresearch-karpathy) |
| TUI dashboard | [elementalcollision](https://github.com/elementalcollision/autoresearch) |
| Experiment dedup | [mutable-state-inc](https://github.com/mutable-state-inc/autoresearch-at-home) |

## Project structure

```
run.py                 one command, any data
engine/
  adaptive.py          4-mode search + Optuna
  variants.py          model variant generation (sklearn, pytorch, tf, keras, hf, statsmodels)
  llm_proposer.py      LLM code proposal (Claude, GPT, Gemini)
  semantic_diff.py     entity-level code diffs (for LLM mode)
  decompose.py         auto-split Python files into slots
  slots/runner.py      combinatorial slot testing
  slots/registry.py    version tracking + interaction detection
  hardware.py          hardware auto-detection
  dashboard.py         TUI dashboard
  checkpoint.py        warm-start best model across combos (PyTorch, Keras)
  scaling.py           swarm scaling law
  merge.py             multi-agent A/B merge
extras/
  telemetry/           PII-stripped experiment telemetry (opt-out)
```

## Telemetry

Collects anonymous experiment telemetry by default. PII auto-stripped from all text (names, emails, paths, IPs, credit cards).

**Disable:** `export AUTORESEARCH_TELEMETRY=0`

## License

MIT
