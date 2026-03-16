# autoresearch-multi

This project is an extension of Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) with combinatorial testing and adaptive search.

## How it differs

In Karpathy's autoresearch, an agent modifies a single file per iteration, runs a fixed 5-minute training run, and accepts or reverts the change based on a validation metric.

This project:

1. **Decomposes** Python ML pipelines that follow the standard pattern (`load_data`, `engineer_features`, `build_model`, `evaluate`) into independent slots and tests combinations. Detects the framework from imports and generates appropriate model variants. Supports sklearn, PyTorch, TensorFlow, Keras, HuggingFace, and statsmodels.

2. **Detects interactions.** Finds superadditive effects that sequential testing misses:
   ```
   Title features solo:    +2.79%
   XGBoost solo:           +0.00%
   Predicted combo:        +2.79%
   Actual combo:           +3.91%
   Interaction delta:      +1.12% (superadditive)
   ```

3. **Adaptive search** with 4 modes. Auto-selects Optuna Bayesian when search space >20 variants.
   ```
   EXPLORE   coverage guarantee: one per model family x encoding
   EXPLOIT   Optuna Bayesian (large space) or grid (small space)
   COMBINE   test near-miss pairs for superadditive combos
   NARROW    fine-tune within winning region
   ```

4. **Proposes code via LLM** (when API key is set). Reads prior results, writes new model/feature variants, iteratively improves across rounds. Code diffs show what changed.

5. **Works on any data.** CSV, OpenML dataset, or existing ML pipelines.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/agalimova/autoresearch-multi/main/scripts/install.sh | bash
```

Or manually:

```bash
git clone https://github.com/agalimova/autoresearch-multi.git
cd autoresearch-multi
pip install -r requirements.txt
```

## Usage

```bash
python run.py my_data.csv                   # point at a CSV
python run.py my_pipeline.py                # point at a Python file (auto-decomposes)
python run.py --dataset adult               # use any OpenML dataset
```


Without an API key: template-based search. With any LLM key, it proposes novel model code and iteratively improves. Supports Claude, GPT-4o, Gemini, OpenRouter, Together AI, Deepseek, and Ollama (local, no key needed).

## Results

### LLM mode

Claude proposes novel code (ensembles, custom feature engineering) and iteratively improves across 3 rounds. Example on King-Rook vs King-Pawn, a chess endgame dataset (36 categorical columns, binary classification):

```
Templates:      98.06%
LLM Round 1:    99.31%  (+1.25%, Claude proposed VotingClassifier ensemble)
LLM Round 2:    99.47%  (+0.16%, Claude refined based on Round 1 findings)
LLM Round 3:    99.96%  (+0.49%, near-perfect, top 1% of published benchmarks)
```

### Template vs LLM (same datasets)

| Dataset | Baseline | Template | Percentile | With Claude | Percentile |
|---|---|---|---|---|---|
| heart-statlog | 84.81% | 84.81% | top 25% | **97.22%** | top 1% |
| kr-vs-kp | 93.31% | 98.06% | top 5% | **99.96%** | top 1% |
| iris | 97.33% | 98.00% | top 5% | **99.33%** | top 1% |
| diabetes | 77.22% | 77.22% | top 5% | **78.30%** | top 5% |

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
  checkpoint.py        warm-start checkpoints
  scaling.py           swarm scaling law
  merge.py             multi-agent A/B merge
  grid.py              hyperparameter grid generation
extras/
  telemetry/           PII-stripped experiment telemetry (opt-out)
```

## Telemetry

Collects anonymous experiment telemetry by default. PII auto-stripped from all text (names, emails, paths, IPs, credit cards).

**Disable:** `export AUTORESEARCH_TELEMETRY=0`

## License

MIT
