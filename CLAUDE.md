# Autoresearch-Multi

Combinatorial ML pipeline search with pruning, interaction detection, and LLM proposals.

Built on: karpathy/autoresearch (engine) + ataraxy-labs/sem (diffs).

## Architecture

```
run.py                      CLI entry point: `autoresearch my_data.csv`
                            Loads data (CSV/OpenML/Python), sets up slots,
                            runs Coordinator + TabularBackend

engine/
  coordinator.py            4-mode adaptive search + LLM rounds
  backend.py                Backend protocol + ExperimentTracker
  setup.py                  Data loading, code templates, model variants, slot generation
  llm_proposer.py           LLM code proposal (Claude, GPT, Gemini, Ollama)
  decompose.py              Auto-split Python files into slots
  hardware.py               Hardware auto-detection (CPU/GPU/RAM)
  dashboard.py              TUI dashboard
  checkpoint.py             Warm-start best model across combos
  scaling.py                Swarm scaling heuristic (experimental)
  merge.py                  Multi-agent A/B merge (requires `sem` CLI)
  semantic_diff.py          Entity-level diffs (requires `sem` CLI)
  adaptive.py               Original standalone 4-mode search (used by demos)

  backends/
    tabular.py              Wraps SlotRunner for tabular ML
    gpu_training.py         Wraps any train script for GPU experiments

  slots/
    runner.py               Generic pipeline runner + combinatorial testing

extras/
  telemetry/
    collector.py            Experiment telemetry (opt-out via AUTORESEARCH_TELEMETRY=0)

tools/
  agent-analysis/           Rust crate: tree-sitter entity extraction + slot detection

tests/
  test_backend.py           Backend protocol + tracker + coordinator tests
  test_gpu_e2e.py           GPU backend end-to-end tests
  fixtures/gpu_project/     Mock GPU project for testing
```

## How the CLI works

```
autoresearch my_data.csv
  1. Loads CSV, auto-detects target column
  2. Generates slots: load_data/, engineer_features/, build_model/, evaluate/
  3. Builds TabularBackend wrapping SlotRunner
  4. Runs Coordinator with 4 search modes:
     - EXPLORE: coverage guarantee (one per family per dimension) + random fill
     - EXPLOIT: drill into winning family (Optuna TPE or grid)
     - COMBINE: near-miss pairs prioritized by interaction signal
     - NARROW: fine-tune within winning region
  5. If LLM API key set: LLM proposes novel variants, tested against best
  6. Prints ranked results, saves to results/<name>.json
```

## Key design decisions

- **Coordinator + Backend protocol**: run.py uses Coordinator with TabularBackend.
  The coordinator drives search without knowing what's underneath. GPU training
  uses GpuTrainingBackend with the same coordinator.

- **ExperimentTracker**: persists experiment history, config diffs, dead families,
  crash counts, and interaction deltas to JSON after every experiment. Provides
  summary_for_prompt() for LLM context injection.

- **Pruning**: family pruning (best-in-family < 95% of global best), dominance
  pruning (per-option within dimension), crash counting (3 crashes = auto-skip).

- **Interaction detection**: for multi-dimension configs, delta = actual - predicted
  where predicted = base + sum(solo improvements). Positive = superadditive.
  COMBINE mode uses interaction history to prioritize cross-pairs.

- **LLM waterfall**: Anthropic -> OpenAI -> Gemini -> OpenRouter -> Together ->
  DeepSeek -> Ollama. First available provider wins. No API key = template-only mode.

## Running

```bash
# Install
pip install -e .                     # core (sklearn, xgboost, lightgbm)
pip install -e ".[llm]"              # + Anthropic Claude
pip install -e ".[llm-all]"          # + all LLM providers
pip install -e ".[all]"              # + torch, tensorflow, optuna, all LLM providers

# CLI
autoresearch my_data.csv                       # sklearn (default)
autoresearch my_data.csv --framework pytorch   # PyTorch
autoresearch my_data.csv --framework catboost  # CatBoost
autoresearch --dataset adult                   # OpenML dataset
autoresearch my_pipeline.py                    # Python file decomposition

# Demos
python demo.py --runs 8              # Hyperparameter comparison
python demo_slots.py                 # Slot-based search
python demo_gpu_karpathy.py          # GPU training (RTX 4090)

# Tests
python -m pytest tests/ -v
```

## Dependencies

- Core: scikit-learn, pandas, numpy, xgboost, lightgbm
- Optional: torch, torchvision, tensorflow, anthropic, openai, google-generativeai, httpx, optuna
- Rust (optional): sem CLI for entity-level diffs, agent-analysis for slot detection
