"""
Slot directory setup + model variant generation.

One module for everything needed to go from `autoresearch data.csv` to a
runnable slot directory: data loading, code templates, model variants, .py decomposition.
"""

from __future__ import annotations

import itertools
import random
import re
from pathlib import Path


# ── Data loading ─────────────────────────────────────────────────────────────

def load(csv_path: str | None = None, dataset: str | None = None,
         target: str = "") -> tuple:
    """Load CSV or OpenML. Returns (df, target_col, display_name)."""
    import pandas as pd, numpy as np
    if csv_path:
        df = pd.read_csv(csv_path)
        target = target or str(df.columns[-1])
        name = Path(csv_path).stem
    elif dataset:
        from sklearn.datasets import fetch_openml
        data = fetch_openml(dataset, version=1, as_frame=True, parser="auto")
        df = data.frame
        target = target or str(data.target.name if hasattr(data.target, "name") else df.columns[-1])
        name = dataset
    else:
        raise ValueError("Provide csv_path or dataset")
    y, X = df[target], df.drop(columns=[target])
    n_cls = y.nunique()
    task = "regression" if n_cls > 20 else ("binary" if n_cls == 2 else "multiclass")
    print(f"Loaded {name}: {len(df)} rows, {df.shape[1]} cols, target='{target}'")
    print(f"  {task} | {X.select_dtypes(include=[np.number]).shape[1]} numeric, "
          f"{X.select_dtypes(exclude=[np.number]).shape[1]} categorical | "
          f"{n_cls} classes | {df.isnull().mean().mean():.1%} missing")
    return df, target, name


# ── Code templates ───────────────────────────────────────────────────────────

_NN = ("pytorch", "keras", "tensorflow")

_PREP = '''\
import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder
def _prep(df, target):
    y = df[target].copy()
    X = df.drop(columns=[target]).copy()
    if y.dtype == object or str(y.dtype) == "category":
        y = LabelEncoder().fit_transform(y.astype(str).fillna("missing"))
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = X.select_dtypes(exclude=[np.number]).columns.tolist()
    X[num] = X[num].fillna(0)
    for c in cat: X[c] = X[c].astype(str).fillna("missing")
    return X, y, num, cat
'''

_FEATURES = {
    "base": 'def engineer_features(df, t):\n    X,y,_,cat = _prep(df,t)\n    for c in cat: X[c] = LabelEncoder().fit_transform(X[c])\n    return X,y\n',
    "ordinal": 'from sklearn.preprocessing import OrdinalEncoder\ndef engineer_features(df, t):\n    X,y,_,cat = _prep(df,t)\n    if cat: X[cat] = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(X[cat])\n    return X,y\n',
    "onehot": 'def engineer_features(df, t):\n    X,y,_,cat = _prep(df,t)\n    lo = [c for c in cat if X[c].nunique()<=20]\n    hi = [c for c in cat if X[c].nunique()>20]\n    if lo: X = pd.get_dummies(X, columns=lo, drop_first=True)\n    for c in hi: X[c] = LabelEncoder().fit_transform(X[c])\n    return X,y\n',
    "frequency": 'def engineer_features(df, t):\n    X,y,_,cat = _prep(df,t)\n    for c in cat: X[c] = X[c].map(X[c].value_counts(normalize=True)).fillna(0)\n    return X,y\n',
}

EVALUATOR = '''\
import copy, numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
def _to_np(X, y):
    return np.array(X.values if hasattr(X,"values") else X, dtype=np.float32), np.array(y, dtype=np.int64)
def _cv_torch(X, y, model):
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    accs = []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        m = copy.deepcopy(model)
        opt, loss = torch.optim.Adam(m.parameters(), lr=1e-3), nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(torch.tensor(X[tr]), torch.tensor(y[tr])), batch_size=32, shuffle=True)
        m.train()
        for _ in range(20):
            for xb, yb in loader:
                opt.zero_grad(); loss(m(xb), yb).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            accs.append((m(torch.tensor(X[va])).argmax(1) == torch.tensor(y[va])).float().mean().item())
    return float(np.mean(accs))
def _cv_keras(X, y, model):
    accs = []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        m = model.__class__.from_config(model.get_config())
        m.compile(optimizer="adam", loss=model.loss, metrics=["accuracy"])
        m.fit(X[tr], y[tr], epochs=20, batch_size=32, verbose=0)
        accs.append(m.evaluate(X[va], y[va], verbose=0)[1])
    return float(np.mean(accs))
def evaluate(X, y, model):
    mod = type(model).__module__
    if mod.startswith("torch"):
        return {"val_acc": _cv_torch(*_to_np(X, y), model)}
    if "keras" in mod or "tensorflow" in mod:
        return {"val_acc": _cv_keras(*_to_np(X, y), model)}
    return {"val_acc": float(cross_val_score(model, X, y, cv=5, scoring="accuracy", n_jobs=-1).mean())}
'''


# ── Slot setup ───────────────────────────────────────────────────────────────

def from_data(slot_dir: Path, *, csv_path: str = "", dataset: str = "",
              target: str, framework: str = "sklearn"):
    """Generate a complete slot directory for tabular data."""
    for d in ("load_data", "engineer_features", "build_model", "evaluate"):
        (slot_dir / d).mkdir(parents=True, exist_ok=True)
    # Loader
    if csv_path:
        src = f'import pandas as pd\ndef load_data(): return pd.read_csv("{csv_path}"), "{target}"\n'
    else:
        src = (f'from sklearn.datasets import fetch_openml\ndef load_data():\n'
               f'    d = fetch_openml("{dataset}", version=1, as_frame=True, parser="auto")\n'
               f'    return d.frame, (d.target.name if hasattr(d.target,"name") else d.frame.columns[-1])\n')
    (slot_dir / "load_data" / "base.py").write_text(src)
    # Features
    feats = {"base": _FEATURES["base"]} if framework in _NN else _FEATURES
    for name, body in feats.items():
        (slot_dir / "engineer_features" / f"{name}.py").write_text(_PREP + "\n" + body)
    # Models
    kwargs = _infer_shape(csv_path, dataset, target) if framework in _NN else {}
    n = generate(framework, slot_dir / "build_model", **kwargs)
    print(f"  {n} {framework} model variants")
    # Evaluator
    (slot_dir / "evaluate" / "base.py").write_text(EVALUATOR)


def from_py(py_path: Path, slot_dir: Path):
    """Decompose a Python file into slots, auto-detect framework."""
    from engine.decompose import decompose_file
    source = py_path.read_text()
    markers = [("keras", "keras"), ("tensorflow", "tensorflow"),
               ("huggingface", "transformers"), ("pytorch", "torch"), ("statsmodels", "statsmodels")]
    framework = next((n for n, m in markers if f"import {m}" in source or f"from {m}" in source), "sklearn")
    print(f"  framework: {framework}")
    slots = decompose_file(py_path)
    if not slots:
        print("  No slots detected."); return
    for name, code in slots.items():
        d = slot_dir / name; d.mkdir(parents=True, exist_ok=True); (d / "base.py").write_text(code)
    if (slot_dir / "build_model").exists():
        kwargs = _parse_nn_shape(slot_dir / "build_model" / "base.py") if framework in _NN else {}
        generate(framework, slot_dir / "build_model", **kwargs)
    if (slot_dir / "engineer_features").exists() and framework not in _NN:
        for name, body in _FEATURES.items():
            (slot_dir / "engineer_features" / f"{name}.py").write_text(_PREP + "\n" + body)
    eval_dir = slot_dir / "evaluate"; eval_dir.mkdir(parents=True, exist_ok=True)
    if not (eval_dir / "base.py").exists():
        (eval_dir / "base.py").write_text(EVALUATOR)


# ── Model variant generation ────────────────────────────────────────────────

def generate(framework: str, model_dir: Path, **kw) -> int:
    """Generate model variant .py files. Returns count written."""
    builder = _BUILDERS.get(framework, _sklearn)
    specs = builder(**kw)
    for s in specs:
        (model_dir / f"{s['name']}.py").write_text(s["code"])
    return len(specs)


def _simple_model(name: str, module: str, cls: str, params: dict) -> dict:
    """Build spec for a sklearn-API model (one import, one constructor)."""
    ps = ", ".join(f"{k}={v}" for k, v in params.items())
    return {"name": name, "code": f"from {module} import {cls}\n\ndef build_model():\n    return {cls}({ps})\n"}


def _sklearn(**_):
    return [_simple_model(n, m, c, p) for n, m, c, p in [
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
    ]]


def _catboost(**_):
    return [_simple_model(n, "catboost", "CatBoostClassifier", p) for n, p in [
        ("cb_default", {"iterations": 100, "verbose": 0, "random_seed": 42}),
        ("cb_deep", {"iterations": 300, "depth": 8, "learning_rate": 0.03, "verbose": 0, "random_seed": 42}),
        ("cb_shallow", {"iterations": 200, "depth": 4, "learning_rate": 0.1, "verbose": 0, "random_seed": 42}),
        ("cb_fast", {"iterations": 50, "depth": 6, "learning_rate": 0.2, "verbose": 0, "random_seed": 42}),
        ("cb_l2reg", {"iterations": 200, "depth": 6, "l2_leaf_reg": 5, "verbose": 0, "random_seed": 42}),
        ("cb_bagging", {"iterations": 200, "depth": 6, "subsample": 0.8, "verbose": 0, "random_seed": 42, "bootstrap_type": "'Bernoulli'"}),
    ]]


_TORCH_ACTS = {"relu": "nn.ReLU()", "gelu": "nn.GELU()", "tanh": "nn.Tanh()", "silu": "nn.SiLU()"}

def _pytorch(n_variants: int = 60, input_dim: int = 784, output_dim: int = 10, **_):
    combos = list(itertools.product([32, 64, 96, 128, 192, 256, 384, 512], [1, 2, 3, 4],
                                     ["relu", "gelu", "tanh", "silu"], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], [True, False]))
    random.Random(42).shuffle(combos)
    specs = []
    for h, d, act, dr, bn in combos[:n_variants]:
        name = f"h{h}_d{d}_{act}_dr{str(dr).replace('.','')}_{'bn' if bn else 'nobn'}"
        layers, dim = [], input_dim
        for _ in range(d):
            layers.append(f"        nn.Linear({dim}, {h}),")
            if bn: layers.append(f"        nn.BatchNorm1d({h}),")
            layers.append(f"        {_TORCH_ACTS[act]},")
            if dr > 0: layers.append(f"        nn.Dropout({dr}),")
            dim = h
        layers.append(f"        nn.Linear({dim}, {output_dim}),")
        specs.append({"name": name, "code": f"import torch.nn as nn\n\ndef build_model():\n    return nn.Sequential(\n{chr(10).join(layers)}\n    )\n"})
    return specs


def _dense_specs(prefix, layer_mod, configs, *, input_shape=(784,), output_units=10,
                 output_activation="softmax", loss="sparse_categorical_crossentropy", **_):
    specs = []
    for h, d, act, dr in configs:
        name = f"{prefix}_h{h}_d{d}_{act}"
        layers = [f"        {layer_mod}.Input(shape={input_shape}),"]
        for _ in range(d):
            layers.append(f"        {layer_mod}.Dense({h}, activation='{act}'),")
            if dr > 0: layers.append(f"        {layer_mod}.Dropout({dr}),")
        layers.append(f"        {layer_mod}.Dense({output_units}, activation='{output_activation}'),")
        imp = "import keras" if "keras" in layer_mod else "import tensorflow as tf"
        seq = "keras.Sequential" if "keras" in layer_mod else "tf.keras.Sequential"
        specs.append({"name": name, "code": (
            f"{imp}\n\ndef build_model():\n    model = {seq}([\n{chr(10).join(layers)}\n    ])\n"
            f"    model.compile(optimizer='adam', loss='{loss}', metrics=['accuracy'])\n    return model\n")})
    return specs

def _keras(**kw):
    return _dense_specs("keras", "keras.layers", [(64,2,"relu",0),(128,2,"relu",.2),(256,3,"gelu",.3),(128,4,"relu",.1),(512,2,"relu",.2)], **kw)

def _tensorflow(**kw):
    return _dense_specs("tf", "tf.keras.layers", [(64,2,"relu",0),(64,2,"gelu",0),(128,2,"relu",.2),(128,3,"gelu",.2),(256,2,"relu",.3),
                                                   (256,3,"gelu",.3),(512,2,"relu",.2),(128,4,"relu",.1),(128,4,"gelu",.2),(128,2,"relu",0)], **kw)


def _huggingface(num_labels: int = 2, **_):
    return [{"name": n, "code": f"from transformers import AutoModelForSequenceClassification\n\ndef build_model():\n    return AutoModelForSequenceClassification.from_pretrained('{m}', num_labels={num_labels})\n"}
            for n, m in [("hf_bert_base","bert-base-uncased"),("hf_bert_small","prajjwal1/bert-small"),("hf_distilbert","distilbert-base-uncased"),("hf_roberta","roberta-base"),("hf_albert","albert-base-v2")]]


def _statsmodels(**_):
    def _wrap(cls, extra=""):
        e = f", {extra}" if extra else ""
        return f'import numpy as np\nimport statsmodels.api as sm\nfrom sklearn.base import BaseEstimator, ClassifierMixin\n\nclass SMWrapper(BaseEstimator, ClassifierMixin):\n    def __init__(self): self.model_ = None\n    def fit(self, X, y):\n        self.model_ = sm.{cls}(y, sm.add_constant(X, has_constant="add")).fit(disp=0{e})\n        self.classes_ = np.unique(y); return self\n    def predict(self, X):\n        p = self.model_.predict(sm.add_constant(X, has_constant="add"))\n        return (p > 0.5).astype(int) if len(self.classes_) == 2 else np.argmax(p, axis=1)\n\ndef build_model(): return SMWrapper()\n'
    return [{"name": n, "code": _wrap(c, e)} for n, c, e in
            [("sm_logit","Logit",""),("sm_probit","Probit",""),("sm_mnlogit","MNLogit",""),
             ("sm_ols","OLS",""),("sm_wls","WLS",""),("sm_gls","GLS",""),("sm_rlm","RLM",""),("sm_quantile","QuantReg","q=0.5")]]


_BUILDERS = {"sklearn": _sklearn, "pytorch": _pytorch, "tensorflow": _tensorflow,
             "keras": _keras, "huggingface": _huggingface, "statsmodels": _statsmodels, "catboost": _catboost}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _infer_shape(csv_path: str, dataset: str, target: str) -> dict:
    import pandas as pd, numpy as np
    df, tgt, _ = load(csv_path=csv_path or None, dataset=dataset or None, target=target)
    X = df.drop(columns=[tgt])
    dim = X.select_dtypes(include=[np.number]).shape[1] + X.select_dtypes(exclude=[np.number]).shape[1]
    return {"input_dim": dim, "output_dim": int(df[tgt].nunique()), "input_shape": (dim,)}


def _parse_nn_shape(path: Path) -> dict:
    if not path.exists(): return {}
    code = path.read_text(); r = {}
    m = re.search(r'(?:input_shape|shape)\s*=\s*\(([^)]+)\)', code)
    if m:
        try: dims = tuple(int(x.strip()) for x in m.group(1).split(",") if x.strip()); r.update(input_shape=dims, input_dim=dims[0])
        except ValueError: pass
    for m in re.finditer(r'Dense\((\d+)\s*,\s*activation\s*=\s*[\'"](\w+)[\'"]', code):
        r.update(output_units=int(m.group(1)), output_dim=int(m.group(1)), output_activation=m.group(2))
    m = re.search(r'loss\s*=\s*[\'"]([^\'"]+)[\'"]', code)
    if m: r["loss"] = m.group(1)
    for m in re.finditer(r'nn\.Linear\((\d+)\s*,\s*(\d+)\)', code):
        r.setdefault("input_dim", int(m.group(1))); r["output_dim"] = int(m.group(2))
    return r
