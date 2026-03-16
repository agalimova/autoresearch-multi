"""
Model variant generator — data-driven, one module for all frameworks.

Given a framework name, generates slot files with model alternatives.
Each variant is defined as a spec dict, rendered into a .py file.
"""

from __future__ import annotations

import itertools
import random
from pathlib import Path
from typing import Any


# ── Spec format ──────────────────────────────────────────────────────────────
#
# Each spec is a dict with:
#   name: str           — filename (without .py)
#   code: str           — the Python source code
#
# Specs are generated from _build_* functions below.


def generate(framework: str, model_dir: Path, **kwargs) -> int:
    """Generate model variants for a framework. Returns count."""
    builders = {
        "sklearn": _build_sklearn,
        "pytorch": _build_pytorch,
        "tensorflow": _build_tensorflow,
        "keras": _build_keras,
        "huggingface": _build_huggingface,
        "statsmodels": _build_statsmodels,
    }
    builder = builders.get(framework, _build_sklearn)
    specs = builder(**kwargs)

    for spec in specs:
        (model_dir / f"{spec['name']}.py").write_text(spec["code"])

    return len(specs)


# ── Sklearn ──────────────────────────────────────────────────────────────────

_SKLEARN_MODELS = [
    ("lr_default", "sklearn.linear_model", "LogisticRegression", {"max_iter": 1000, "random_state": 42}),
    ("lr_l1", "sklearn.linear_model", "LogisticRegression", {"max_iter": 1000, "penalty": "'l1'", "solver": "'saga'", "random_state": 42}),
    ("rf_100", "sklearn.ensemble", "RandomForestClassifier", {"n_estimators": 100, "random_state": 42, "n_jobs": -1}),
    ("rf_200_d8", "sklearn.ensemble", "RandomForestClassifier", {"n_estimators": 200, "max_depth": 8, "random_state": 42, "n_jobs": -1}),
    ("rf_200_d12", "sklearn.ensemble", "RandomForestClassifier", {"n_estimators": 200, "max_depth": 12, "random_state": 42, "n_jobs": -1}),
    ("xgb_default", "xgboost", "XGBClassifier", {"n_estimators": 100, "random_state": 42, "verbosity": 0}),
    ("xgb_deep", "xgboost", "XGBClassifier", {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05, "random_state": 42, "verbosity": 0}),
    ("xgb_shallow", "xgboost", "XGBClassifier", {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1, "random_state": 42, "verbosity": 0}),
    ("lgbm_default", "lightgbm", "LGBMClassifier", {"n_estimators": 100, "random_state": 42, "verbosity": -1}),
    ("lgbm_tuned", "lightgbm", "LGBMClassifier", {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05, "random_state": 42, "verbosity": -1}),
    ("knn_5", "sklearn.neighbors", "KNeighborsClassifier", {"n_neighbors": 5}),
    ("knn_11", "sklearn.neighbors", "KNeighborsClassifier", {"n_neighbors": 11}),
]


def _build_sklearn(**_) -> list[dict]:
    specs = []
    for name, module, cls, params in _SKLEARN_MODELS:
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        specs.append({
            "name": name,
            "code": f"from {module} import {cls}\n\ndef build_model():\n    return {cls}({param_str})\n",
        })
    return specs


# ── PyTorch ──────────────────────────────────────────────────────────────────

_TORCH_ACTS = {"relu": "nn.ReLU()", "gelu": "nn.GELU()", "tanh": "nn.Tanh()", "silu": "nn.SiLU()"}


def _build_pytorch(n_variants: int = 60, input_dim: int = 784, output_dim: int = 10, **_) -> list[dict]:
    combos = list(itertools.product(
        [32, 64, 96, 128, 192, 256, 384, 512],
        [1, 2, 3, 4],
        ["relu", "gelu", "tanh", "silu"],
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        [True, False],
    ))
    rng = random.Random(42)
    rng.shuffle(combos)

    specs = []
    for hidden, depth, act, dropout, bn in combos[:n_variants]:
        name = f"h{hidden}_d{depth}_{act}_dr{str(dropout).replace('.','')}_{'bn' if bn else 'nobn'}"
        layers = []
        in_d = input_dim
        for _ in range(depth):
            layers.append(f"        nn.Linear({in_d}, {hidden}),")
            if bn:
                layers.append(f"        nn.BatchNorm1d({hidden}),")
            layers.append(f"        {_TORCH_ACTS[act]},")
            if dropout > 0:
                layers.append(f"        nn.Dropout({dropout}),")
            in_d = hidden
        layers.append(f"        nn.Linear({in_d}, {output_dim}),")
        body = "\n".join(layers)
        specs.append({
            "name": name,
            "code": f"import torch.nn as nn\n\ndef build_model():\n    return nn.Sequential(\n{body}\n    )\n",
        })
    return specs


# ── TensorFlow ───────────────────────────────────────────────────────────────

def _build_tensorflow(**_) -> list[dict]:
    configs = [
        (64, 2, "relu", 0.0, False),
        (64, 2, "gelu", 0.0, False),
        (128, 2, "relu", 0.2, False),
        (128, 3, "gelu", 0.2, False),
        (256, 2, "relu", 0.3, False),
        (256, 3, "gelu", 0.3, False),
        (512, 2, "relu", 0.2, False),
        (128, 4, "relu", 0.1, False),
        (128, 4, "gelu", 0.2, False),
        (128, 2, "relu", 0.0, True),
    ]
    specs = []
    for hidden, depth, act, drop, bn in configs:
        suffix = f"_bn" if bn else ""
        name = f"tf_h{hidden}_d{depth}_{act}{suffix}"
        layers = ["        tf.keras.layers.InputLayer(input_shape=(784,)),"]
        for _ in range(depth):
            layers.append(f"        tf.keras.layers.Dense({hidden}, activation='{act}'),")
            if bn:
                layers.append("        tf.keras.layers.BatchNormalization(),")
            if drop > 0:
                layers.append(f"        tf.keras.layers.Dropout({drop}),")
        layers.append("        tf.keras.layers.Dense(10, activation='softmax'),")
        body = "\n".join(layers)
        specs.append({
            "name": name,
            "code": f"import tensorflow as tf\n\ndef build_model():\n    return tf.keras.Sequential([\n{body}\n    ])\n",
        })
    return specs


# ── Keras (standalone) ───────────────────────────────────────────────────────

def _build_keras(**_) -> list[dict]:
    configs = [
        (64, 2, "relu", 0.0),
        (128, 2, "relu", 0.2),
        (256, 3, "gelu", 0.3),
        (128, 4, "relu", 0.1),
        (512, 2, "relu", 0.2),
        (128, 2, "relu", 0.0),
    ]
    specs = []
    for hidden, depth, act, drop in configs:
        name = f"keras_h{hidden}_d{depth}_{act}"
        layers = ["        keras.layers.Input(shape=(784,)),"]
        for _ in range(depth):
            layers.append(f"        keras.layers.Dense({hidden}, activation='{act}'),")
            if drop > 0:
                layers.append(f"        keras.layers.Dropout({drop}),")
        layers.append("        keras.layers.Dense(10, activation='softmax'),")
        body = "\n".join(layers)
        specs.append({
            "name": name,
            "code": f"import keras\n\ndef build_model():\n    return keras.Sequential([\n{body}\n    ])\n",
        })
    return specs


# ── HuggingFace ──────────────────────────────────────────────────────────────

_HF_MODELS = [
    ("hf_bert_base", "bert-base-uncased"),
    ("hf_bert_small", "prajjwal1/bert-small"),
    ("hf_distilbert", "distilbert-base-uncased"),
    ("hf_roberta", "roberta-base"),
    ("hf_albert", "albert-base-v2"),
]


def _build_huggingface(num_labels: int = 2, **_) -> list[dict]:
    return [{
        "name": name,
        "code": (
            "from transformers import AutoModelForSequenceClassification\n\n"
            "def build_model():\n"
            f"    return AutoModelForSequenceClassification.from_pretrained('{model}', num_labels={num_labels})\n"
        ),
    } for name, model in _HF_MODELS]


# ── Statsmodels ──────────────────────────────────────────────────────────────

_SM_VARIANTS = [
    ("sm_ols", "sm.OLS(y, sm.add_constant(X)).fit()"),
    ("sm_wls", "sm.WLS(y, sm.add_constant(X)).fit()"),
    ("sm_gls", "sm.GLS(y, sm.add_constant(X)).fit()"),
    ("sm_rlm", "sm.RLM(y, sm.add_constant(X)).fit()"),
    ("sm_quantile", "sm.QuantReg(y, sm.add_constant(X)).fit(q=0.5)"),
    ("sm_logit", "sm.Logit(y, sm.add_constant(X)).fit(disp=0)"),
    ("sm_probit", "sm.Probit(y, sm.add_constant(X)).fit(disp=0)"),
]


def _build_statsmodels(**_) -> list[dict]:
    return [{
        "name": name,
        "code": f"import statsmodels.api as sm\n\ndef build_model():\n    return lambda X, y: {constructor}\n",
    } for name, constructor in _SM_VARIANTS]
