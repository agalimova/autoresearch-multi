"""
Hyperparameter grid → slot files.

Generates one .py file per grid point. Each file exports a single function
matching the slot name. The combo engine tests compositions across slots.

Usage:
    from engine.grid import generate_model_grid
    generate_model_grid(Path("workspace/slots/build_model"), budget=30)
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any


def generate_model_grid(
    slot_dir: Path,
    *,
    budget: int | None = None,
    seed: int = 42,
) -> int:
    """
    Generate build_model slot files from a hyperparameter grid.

    If budget is set, sample that many points from the grid.
    Returns number of files written.
    """
    slot_dir.mkdir(parents=True, exist_ok=True)

    grids = _model_grids()
    all_configs = []
    for model_name, template, params in grids:
        keys = list(params.keys())
        for values in product(*params.values()):
            config = dict(zip(keys, values))
            all_configs.append((model_name, template, config))

    # Budget sampling
    if budget and len(all_configs) > budget:
        import random
        rng = random.Random(seed)
        all_configs = rng.sample(all_configs, budget)

    count = 0
    for model_name, template, config in all_configs:
        name = _config_to_name(model_name, config)
        filepath = slot_dir / f"{name}.py"
        if filepath.exists():
            continue
        code = template.format(**config)
        filepath.write_text(code)
        count += 1

    return count


def generate_feature_grid(
    slot_dir: Path,
    *,
    dataset_type: str = "tabular",
) -> int:
    """Generate engineer_features slot files for common feature strategies."""
    slot_dir.mkdir(parents=True, exist_ok=True)

    strategies = _feature_strategies(dataset_type)
    count = 0
    for name, code in strategies:
        filepath = slot_dir / f"{name}.py"
        if filepath.exists():
            continue
        filepath.write_text(code)
        count += 1

    return count


def _config_to_name(model_name: str, config: dict) -> str:
    """Convert model name + config to a filename."""
    parts = [model_name]
    for k, v in config.items():
        if v is None:
            parts.append(f"{k}_none")
        elif isinstance(v, float):
            parts.append(f"{k}_{v:.0e}".replace("+", "").replace(".", ""))
        else:
            parts.append(f"{k}_{v}")
    return "_".join(parts)


def _model_grids() -> list[tuple[str, str, dict[str, list]]]:
    """Return (model_name, code_template, param_grid) for each model family."""
    return [
        (
            "xgb",
            '"""XGBoost: depth={max_depth}, lr={learning_rate}, n={n_estimators}."""\n'
            "from xgboost import XGBClassifier\n"
            "from sklearn.base import BaseEstimator\n\n"
            "def build_model() -> BaseEstimator:\n"
            "    return XGBClassifier(\n"
            "        max_depth={max_depth},\n"
            "        learning_rate={learning_rate},\n"
            "        n_estimators={n_estimators},\n"
            '        eval_metric="logloss", verbosity=0, random_state=42,\n'
            "    )\n",
            {
                "max_depth": [3, 4, 5, 6],
                "learning_rate": [0.01, 0.05, 0.1],
                "n_estimators": [100, 200],
            },
        ),
        (
            "lgbm",
            '"""LightGBM: depth={max_depth}, lr={learning_rate}, n={n_estimators}."""\n'
            "from lightgbm import LGBMClassifier\n"
            "from sklearn.base import BaseEstimator\n\n"
            "def build_model() -> BaseEstimator:\n"
            "    return LGBMClassifier(\n"
            "        max_depth={max_depth},\n"
            "        learning_rate={learning_rate},\n"
            "        n_estimators={n_estimators},\n"
            "        verbose=-1, random_state=42,\n"
            "    )\n",
            {
                "max_depth": [3, 4, 5, 6],
                "learning_rate": [0.01, 0.05, 0.1],
                "n_estimators": [100, 200],
            },
        ),
        (
            "rf",
            '"""RandomForest: depth={max_depth}, n={n_estimators}, leaf={min_samples_leaf}."""\n'
            "from sklearn.ensemble import RandomForestClassifier\n"
            "from sklearn.base import BaseEstimator\n\n"
            "def build_model() -> BaseEstimator:\n"
            "    return RandomForestClassifier(\n"
            "        max_depth={max_depth},\n"
            "        n_estimators={n_estimators},\n"
            "        min_samples_leaf={min_samples_leaf},\n"
            "        random_state=42,\n"
            "    )\n",
            {
                "max_depth": [4, 6, 8, 10],
                "n_estimators": [100, 200],
                "min_samples_leaf": [1, 2, 5],
            },
        ),
    ]


def _feature_strategies(dataset_type: str) -> list[tuple[str, str]]:
    """Common feature engineering strategies."""
    if dataset_type != "tabular":
        return []

    # Universal target encoding: try numeric first, then LabelEncoder
    target_block = (
        "    y = df[target]\n"
        "    if y.dtype == 'object' or y.dtype.name == 'category':\n"
        "        try:\n"
        "            y = y.astype(float).values\n"
        "        except (ValueError, TypeError):\n"
        "            from sklearn.preprocessing import LabelEncoder\n"
        "            y = LabelEncoder().fit_transform(y)\n"
        "    else:\n"
        "        y = y.values\n"
        "    df = df.drop(columns=[target])\n"
    )

    return [
        (
            "onehot",
            '"""One-hot encode categoricals."""\n'
            "import numpy as np, pandas as pd\n\n"
            "def engineer_features(df: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray]:\n"
            "    df = df.copy()\n"
            + target_block +
            "    cat = df.select_dtypes(include=['object','category']).columns.tolist()\n"
            "    num = df.select_dtypes(include='number').columns.tolist()\n"
            "    df[num] = df[num].fillna(0)\n"
            "    df = pd.get_dummies(df, columns=cat, drop_first=True).fillna(0)\n"
            "    return df.values.astype(float), y\n",
        ),
        (
            "ordinal",
            '"""Ordinal encode categoricals."""\n'
            "import numpy as np, pandas as pd\n\n"
            "def engineer_features(df: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray]:\n"
            "    df = df.copy()\n"
            + target_block +
            "    for c in df.select_dtypes(include=['object','category']).columns:\n"
            "        df[c] = df[c].astype('category').cat.codes\n"
            "    df = df.fillna(0)\n"
            "    return df.values.astype(float), y\n",
        ),
    ]
