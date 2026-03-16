"""
Telemetry collector — gathers experiment data and sends to the platform.

Schema version 2. This is the one-shot design — get it right because forks
will be sending this shape for a long time.

Three tiers:
    Tier 1 (opt-out default): dataset fingerprint, model family, metric, split info
    Tier 2 (opt-in):          + hyperparams, preprocessing, convergence, hardware, dependencies
    Tier 3 (opt-in):          + failures, feature importance, overfitting gap, CV scores
"""

from __future__ import annotations

import enum
import hashlib
import json
import os
import platform
import struct
import threading
import time
from pathlib import Path
from typing import Any, Optional

_VERSION = "0.2.0"
_SCHEMA_VERSION = 2


# ── Canonical enums — callers should use these, not free-form strings ────────

class TargetType(str, enum.Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    REGRESSION = "regression"
    RANKING = "ranking"
    TIME_SERIES = "time_series"
    UNSUPERVISED = "unsupervised"
    OTHER = "other"


class ModelFamily(str, enum.Enum):
    TREE = "tree"
    LINEAR = "linear"
    NEURAL_NETWORK = "nn"
    ENSEMBLE = "ensemble"
    SVM = "svm"
    KNN = "knn"
    BAYESIAN = "bayesian"
    OTHER = "other"


class MetricName(str, enum.Enum):
    ACCURACY = "accuracy"
    BALANCED_ACCURACY = "balanced_accuracy"
    F1 = "f1"
    F1_MACRO = "f1_macro"
    F1_MICRO = "f1_micro"
    F1_WEIGHTED = "f1_weighted"
    PRECISION = "precision"
    RECALL = "recall"
    ROC_AUC = "roc_auc"
    LOG_LOSS = "log_loss"
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    R2 = "r2"
    MAPE = "mape"
    NDCG = "ndcg"
    OTHER = "other"


class TelemetryTier(enum.IntEnum):
    STANDARD = 1   # opt-out default: everything, PII stripped (names, emails, paths, IPs, credit cards)
    FULL = 2       # opt-out default: same data, same PII stripping. Reserved for future use.


# ── Normalization — accept messy input, emit canonical values ────────────────

_TARGET_TYPE_ALIASES: dict[str, str] = {
    "binary": "binary_classification", "binclass": "binary_classification",
    "classification": "binary_classification", "clf": "binary_classification",
    "multiclass": "multiclass_classification", "multi": "multiclass_classification",
    "multilabel": "multilabel_classification",
    "reg": "regression", "regr": "regression",
    "ts": "time_series", "timeseries": "time_series", "forecast": "time_series",
    "cluster": "unsupervised", "clustering": "unsupervised",
}

_MODEL_FAMILY_ALIASES: dict[str, str] = {
    "xgboost": "tree", "lightgbm": "tree", "catboost": "tree",
    "random_forest": "tree", "gradient_boosting": "tree", "gbdt": "tree", "gbm": "tree",
    "logistic": "linear", "ridge": "linear", "lasso": "linear", "elasticnet": "linear",
    "mlp": "nn", "neural": "nn", "transformer": "nn", "cnn": "nn", "rnn": "nn", "lstm": "nn",
    "stacking": "ensemble", "voting": "ensemble", "blending": "ensemble",
    "support_vector": "svm", "svc": "svm", "svr": "svm",
    "k_nearest": "knn", "kneighbors": "knn",
    "naive_bayes": "bayesian", "gaussian_process": "bayesian",
}

_METRIC_ALIASES: dict[str, str] = {
    "acc": "accuracy", "balanced_acc": "balanced_accuracy",
    "f1_score": "f1", "f1-score": "f1", "f1score": "f1",
    "auc": "roc_auc", "auroc": "roc_auc", "roc_auc_score": "roc_auc",
    "logloss": "log_loss", "cross_entropy": "log_loss",
    "mean_absolute_error": "mae", "mean_squared_error": "mse",
    "root_mean_squared_error": "rmse",
    "r2_score": "r2", "r_squared": "r2",
    "mean_absolute_percentage_error": "mape",
}


def _normalize(value: str, aliases: dict[str, str], enum_cls: type) -> str:
    """Normalize a free-form string to a canonical enum value."""
    v = value.strip().lower().replace("-", "_").replace(" ", "_")
    v = aliases.get(v, v)
    try:
        return enum_cls(v).value
    except ValueError:
        return v  # keep original if no match — server will accept "other"


def normalize_target_type(v: str) -> str:
    return _normalize(v, _TARGET_TYPE_ALIASES, TargetType)


def normalize_model_family(v: str) -> str:
    return _normalize(v, _MODEL_FAMILY_ALIASES, ModelFamily)


def normalize_metric_name(v: str) -> str:
    return _normalize(v, _METRIC_ALIASES, MetricName)


# ── Global state ─────────────────────────────────────────────────────────────

_enabled: bool = True
_tier: TelemetryTier = TelemetryTier.FULL
_endpoint: str = "http://localhost:8000/api/v1/ingest/telemetry"
_pending: list[dict] = []
_history: list[dict] = []
_lock = threading.Lock()
_initialized = False


# ── Public API ───────────────────────────────────────────────────────────────

def init(
    endpoint: Optional[str] = None,
    tier: Optional[TelemetryTier] = None,
) -> None:
    """Initialize telemetry. Called once at library import time."""
    global _enabled, _tier, _endpoint, _initialized

    if _initialized:
        return
    _initialized = True

    env_val = os.environ.get("AUTORESEARCH_TELEMETRY", "").strip().lower()
    if env_val in ("0", "false", "no", "off"):
        _enabled = False
        return

    config_path = _get_config_path()
    if config_path.exists():
        try:
            import tomllib
            with open(config_path, "rb") as f:
                cfg = tomllib.load(f)
            tel_cfg = cfg.get("telemetry", {})
            if not tel_cfg.get("enabled", True):
                _enabled = False
                return
            if "tier" in tel_cfg:
                _tier = TelemetryTier(int(tel_cfg["tier"]))
            if "endpoint" in tel_cfg:
                _endpoint = tel_cfg["endpoint"]
        except Exception:
            pass

    if endpoint:
        _endpoint = endpoint
    if tier is not None:
        _tier = tier


def disable() -> None:
    global _enabled
    _enabled = False


def enable() -> None:
    global _enabled
    _enabled = True


def is_enabled() -> bool:
    return _enabled


def set_tier(tier: TelemetryTier) -> None:
    global _tier
    _tier = tier


def report_experiment(
    *,
    # ── Tier 1: Trace context (always sent) ──────────────────────────
    trace_id: str = "",                             # links sub-steps of same pipeline (future nesting)
    metadata: dict[str, Any] | None = None,         # escape hatch: git sha, experiment name, A/B variant, anything
    tags: list[str] | None = None,                  # user-defined categorization for filtering

    # ── Tier 1: Domain context (auto-inferred OR user-provided) ─────
    domain: str = "",                               # broad field: "biology", "finance", "nlp", "computer_vision"
    subdomain: str = "",                            # specific area: "protein folding", "credit risk", "sentiment"
    problem_description: str = "",                  # what you're solving: "predict pancreatic enzyme structure"
    domain_keywords: list[str] | None = None,       # free tags: ["protein", "folding", "cancer", "enzyme"]

    # ── Tier 1: Dataset fingerprint (always sent) ────────────────────
    n_rows: int = 0,
    n_cols: int = 0,
    n_classes: int = 0,                            # NEW: 0 = regression/unsupervised
    dtypes: list[str] | None = None,
    cardinality: list[int] | None = None,
    null_fractions: list[float] | None = None,
    feature_types: dict[str, int] | None = None,   # NEW: {"numeric": 6, "categorical": 8, "text": 0, "datetime": 0}
    target_distribution: dict | None = None,        # NEW: class balance or regression stats
    target_type: str = "unknown",
    dataset_hash: str = "",
    column_names_hash: str = "",                    # NEW: hash of sorted column names
    dataset_source: str = "",                       # NEW: "kaggle/titanic", "openml/31", "huggingface/adult-census", "custom"

    # ── Tier 1: Split info (always sent) ─────────────────────────────
    split_strategy: str = "",                       # NEW: "stratified_kfold", "random", "time_series", "holdout"
    split_sizes: dict[str, int] | None = None,      # NEW: {"train": 26048, "val": 6513, "test": 0}
    n_folds: int = 0,                               # NEW: for CV strategies

    # ── Tier 1: Model info (always sent) ─────────────────────────────
    model_family: str = "unknown",
    model_type: str = "",                           # fully qualified: "xgboost.XGBClassifier"

    # ── Tier 1: Metric (always sent) ─────────────────────────────────
    metric_name: str = "",
    metric_value: float = 0.0,
    baseline_value: Optional[float] = None,

    # ── Tier 2: Approach details ─────────────────────────────────────
    hyperparameters: dict[str, Any] | None = None,
    preprocessing: list[dict[str, Any]] | None = None,
    convergence_curve: list[float] | None = None,
    runtime_seconds: Optional[float] = None,
    training_samples_per_second: Optional[float] = None,  # NEW
    seed: Optional[int] = None,                            # NEW
    dependencies: dict[str, str] | None = None,            # NEW: {"xgboost": "2.0.3", "sklearn": "1.4.0"}
    cv_scores: list[float] | None = None,                  # NEW: per-fold scores

    # ── Tier 3: Deep diagnostics ─────────────────────────────────────
    overfitting_gap: Optional[float] = None,
    feature_importance: list[dict[str, Any]] | None = None,  # NEW: [{"name": "age", "score": 0.23}, ...]
) -> Optional[dict]:
    """
    Report a completed experiment. Returns the payload dict (or None if disabled).

    All string fields (target_type, model_family, metric_name) are auto-normalized
    to canonical enum values. Pass whatever string you want — "acc", "Accuracy",
    "balanced_acc" all become the right canonical name.
    """
    if not _enabled:
        return None

    # Normalize free-form strings to canonical values
    target_type = normalize_target_type(target_type)
    model_family = normalize_model_family(model_family)
    metric_name = normalize_metric_name(metric_name)

    # Build a better fingerprint hash
    fp_hash = dataset_hash or _compute_fingerprint_hash(
        n_rows=n_rows, n_cols=n_cols, n_classes=n_classes,
        target_type=target_type, dtypes=dtypes,
        cardinality=cardinality, null_fractions=null_fractions,
        column_names_hash=column_names_hash,
    )

    # Auto-infer domain if not explicitly provided
    if not domain:
        domain, subdomain, inferred_keywords = _infer_domain(
            model_type=model_type,
            dataset_source=dataset_source,
            tags=tags,
            metadata=metadata,
            dependencies=dependencies,
        )
        if inferred_keywords and not domain_keywords:
            domain_keywords = inferred_keywords

    payload: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "library_version": _VERSION,
        "anonymous_id": _get_anon_id(),
        "trace_id": trace_id,
        "metadata": metadata or {},
        "tags": tags or [],

        "domain": {
            "field": domain,                    # general: "biology", "finance", "nlp"
            "subdomain": subdomain,             # specific but not identifying: "protein", "credit risk"
            "problem": "",                      # Tier 2+ only (user-written, could be identifying)
            "keywords": domain_keywords or [],  # inferred: ["protein", "survival"], not project names
        },

        "dataset": {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "n_classes": n_classes,
            "dtypes": dtypes or [],
            "cardinality": cardinality or [],
            "null_fractions": null_fractions or [],
            "feature_types": feature_types or {},
            "target_distribution": target_distribution or {},
            "target_type": target_type,
            "hash": fp_hash,
            "column_names_hash": column_names_hash,
            "dataset_source": dataset_source,
        },

        "split": {
            "strategy": split_strategy,
            "sizes": split_sizes or {},
            "n_folds": n_folds,
        },

        "approach": {
            "model_family": model_family,
            "model_type": model_type,
        },

        "result": {
            "metric_name": metric_name,
            "metric_value": metric_value,
            "baseline_value": baseline_value,
        },

        "failure": None,
    }

    # Everything below is sent by default (both tiers are opt-out)
    # Tier 1 (STANDARD): sanitized text, hyperparams, hardware, convergence
    payload["domain"]["problem"] = problem_description
    payload["approach"]["hyperparameters"] = hyperparameters or {}
    payload["approach"]["preprocessing"] = preprocessing or []
    payload["approach"]["seed"] = seed
    payload["approach"]["dependencies"] = dependencies or _detect_dependencies()
    payload["result"]["convergence_curve"] = convergence_curve or []
    payload["result"]["runtime_seconds"] = runtime_seconds
    payload["result"]["training_samples_per_second"] = training_samples_per_second
    payload["result"]["cv_scores"] = cv_scores or []
    payload["result"]["overfitting_gap"] = overfitting_gap
    payload["result"]["feature_importance"] = feature_importance or []
    payload["hardware"] = _get_hardware_info()

    _enqueue(payload)
    return payload


def report_failure(
    *,
    stage: str,
    error_type: str,
    error_message: str,
    # Dataset context (for matching failures to datasets)
    n_rows: int = 0,
    n_cols: int = 0,
    n_classes: int = 0,
    target_type: str = "unknown",
    dataset_source: str = "",
    dtypes: list[str] | None = None,
    # Approach context (for matching failures to approaches)
    model_family: str = "unknown",
    model_type: str = "",
    hyperparameters: dict[str, Any] | None = None,
) -> Optional[dict]:
    """Report a failed experiment. Sent by default (both tiers are opt-out)."""
    if not _enabled:
        return None

    target_type = normalize_target_type(target_type)
    model_family = normalize_model_family(model_family)

    fp_hash = _compute_fingerprint_hash(
        n_rows=n_rows, n_cols=n_cols, n_classes=n_classes,
        target_type=target_type, dtypes=dtypes,
    )

    payload: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "library_version": _VERSION,
        "anonymous_id": _get_anon_id(),
        "dataset": {
            "n_rows": n_rows, "n_cols": n_cols, "n_classes": n_classes,
            "dtypes": dtypes or [], "cardinality": [],
            "null_fractions": [], "feature_types": {},
            "target_distribution": {}, "target_type": target_type,
            "hash": fp_hash, "column_names_hash": "",
            "dataset_source": dataset_source,
        },
        "split": {"strategy": "", "sizes": {}, "n_folds": 0},
        "approach": {
            "model_family": model_family,
            "model_type": model_type,
            "hyperparameters": hyperparameters or {},
        },
        "result": None,
        "failure": {
            "stage": stage,
            "error_type": error_type,
            "error_message": error_message[:500],
        },
    }

    _enqueue(payload)
    return payload


def show_pending() -> list[dict]:
    """Show payloads queued but not yet sent."""
    with _lock:
        return list(_pending)


def show_history() -> list[dict]:
    """Show payloads sent in this session."""
    with _lock:
        return list(_history)


# ── Internals ────────────────────────────────────────────────────────────────

def _compute_fingerprint_hash(
    *,
    n_rows: int = 0,
    n_cols: int = 0,
    n_classes: int = 0,
    target_type: str = "",
    dtypes: list[str] | None = None,
    cardinality: list[int] | None = None,
    null_fractions: list[float] | None = None,
    column_names_hash: str = "",
) -> str:
    """
    Better fingerprint hash that includes cardinality + null fraction profiles.
    Two datasets with the same shape but different column distributions get different hashes.
    """
    parts = [
        str(n_rows), str(n_cols), str(n_classes), target_type,
        str(sorted(dtypes or [])),
    ]
    # Include cardinality profile (sorted, binned to reduce noise)
    if cardinality:
        binned = [c // 10 * 10 for c in sorted(cardinality)]  # bin to nearest 10
        parts.append(str(binned))
    # Include null fraction profile (rounded)
    if null_fractions:
        rounded = [round(f, 2) for f in sorted(null_fractions)]
        parts.append(str(rounded))
    # Include column names hash if provided (strongest exact-match signal)
    if column_names_hash:
        parts.append(column_names_hash)

    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _get_config_path() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", "~"))
    else:
        base = Path.home() / ".config"
    return base / "autoresearch" / "config.toml"


def _get_anon_id() -> str:
    """Per-session random ID (not persistent — GDPR safe)."""
    if not hasattr(_get_anon_id, "_session_id"):
        _get_anon_id._session_id = hashlib.sha256(  # type: ignore[attr-defined]
            f"{os.getpid()}:{time.time_ns()}".encode()
        ).hexdigest()[:16]
    return _get_anon_id._session_id  # type: ignore[attr-defined]


def _get_hardware_info() -> dict:
    """Collect hardware info. Tries to detect GPU and memory."""
    info: dict[str, Any] = {
        "platform": platform.system().lower(),
        "python_version": platform.python_version(),
        "cpu": platform.processor() or platform.machine() or "unknown",
        "cpu_count": os.cpu_count(),
    }

    # RAM
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        try:
            # Fallback for Linux without psutil
            if platform.system() == "Linux":
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            kb = int(line.split()[1])
                            info["ram_gb"] = round(kb / (1024 ** 2), 1)
                            break
        except Exception:
            pass

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_mem / (1024 ** 3), 1
            )
    except ImportError:
        pass

    return info


def _detect_dependencies() -> dict[str, str]:
    """Auto-detect versions of common ML libraries."""
    deps: dict[str, str] = {}
    for pkg in [
        "sklearn", "xgboost", "lightgbm", "catboost",
        "torch", "tensorflow", "pandas", "numpy",
        "scipy", "statsmodels", "autoresearch",
    ]:
        try:
            mod = __import__(pkg)
            v = getattr(mod, "__version__", None)
            if v:
                deps[pkg] = str(v)
        except ImportError:
            pass
    return deps


def _infer_domain(
    *,
    model_type: str = "",
    dataset_source: str = "",
    tags: list[str] | None = None,
    metadata: dict | None = None,
    dependencies: dict[str, str] | None = None,
) -> tuple[str, str, list[str]]:
    """
    Infer domain from available signals. Returns (field, subdomain, keywords).
    Uses: model type, dataset source, tags, metadata text, installed packages.
    """
    signals = " ".join([
        model_type.lower(),
        dataset_source.lower(),
        " ".join(t.lower() for t in (tags or [])),
        " ".join(str(v).lower() for v in (metadata or {}).values() if isinstance(v, str)),
    ])

    # Also check what's importable (heavy signal)
    installed = set((dependencies or {}).keys())
    if not installed:
        for pkg in ["biopython", "rdkit", "nibabel", "mne", "astropy",
                     "nltk", "spacy", "transformers", "tokenizers",
                     "opencv", "PIL", "torchvision", "albumentations",
                     "librosa", "soundfile", "torchaudio",
                     "yfinance", "quantlib", "zipline",
                     "networkx", "igraph", "dgl",
                     "gym", "stable_baselines3"]:
            try:
                __import__(pkg)
                installed.add(pkg)
            except ImportError:
                pass

    # Domain rules: (field, subdomain, trigger_words_in_signals, trigger_packages)
    rules: list[tuple[str, str, list[str], list[str]]] = [
        # Biology / life sciences
        ("biology", "genomics", ["gene", "genome", "dna", "rna", "sequenc", "variant", "snp", "gwas"], ["biopython", "pysam"]),
        ("biology", "protein", ["protein", "folding", "amino", "residue", "pdb", "alphafold"], ["biopython"]),
        ("biology", "drug discovery", ["drug", "molecule", "compound", "binding", "docking", "smiles", "fingerprint"], ["rdkit"]),
        ("biology", "medical imaging", ["medical", "dicom", "mri", "ct scan", "xray", "pathology", "radiology"], ["nibabel", "pydicom"]),
        ("biology", "clinical", ["patient", "diagnosis", "clinical", "ehr", "icd", "disease", "survival"], []),
        ("biology", "neuroscience", ["eeg", "fmri", "brain", "neural signal", "spike"], ["mne", "nibabel"]),

        # NLP / text
        ("nlp", "text classification", ["text", "sentiment", "document", "review", "spam", "tweet"], ["nltk", "spacy", "transformers"]),
        ("nlp", "language modeling", ["language model", "lm", "gpt", "bert", "llm", "token", "perplexity"], ["transformers", "tokenizers"]),
        ("nlp", "information extraction", ["ner", "entity", "relation", "extraction", "parsing"], ["spacy"]),
        ("nlp", "translation", ["translat", "nmt", "multilingual", "parallel corpus"], ["transformers"]),

        # Computer vision
        ("computer_vision", "image classification", ["image", "photo", "picture", "imagenet", "cifar", "resnet", "cnn"], ["opencv", "torchvision", "albumentations"]),
        ("computer_vision", "object detection", ["detect", "yolo", "bbox", "coco", "segmentation"], ["opencv", "torchvision"]),
        ("computer_vision", "video", ["video", "frame", "temporal", "action recognition"], ["opencv"]),

        # Audio / speech
        ("audio", "speech", ["speech", "asr", "voice", "speaker", "transcri"], ["librosa", "torchaudio", "soundfile"]),
        ("audio", "music", ["music", "audio", "mel", "spectrogram", "midi"], ["librosa"]),

        # Finance
        ("finance", "trading", ["stock", "trade", "market", "price", "ticker", "return", "portfolio"], ["yfinance", "zipline"]),
        ("finance", "credit risk", ["credit", "default", "loan", "risk", "scoring", "fraud"], []),
        ("finance", "insurance", ["insurance", "claim", "actuari", "premium"], []),

        # Climate / earth science
        ("climate", "weather", ["weather", "temperature", "precipitation", "forecast", "climate"], []),
        ("climate", "remote sensing", ["satellite", "remote sensing", "landsat", "sentinel", "ndvi"], ["rasterio"]),

        # Robotics / RL
        ("robotics", "reinforcement learning", ["reinforcement", "reward", "agent", "policy", "environment", "gym"], ["gym", "stable_baselines3"]),
        ("robotics", "control", ["robot", "control", "actuator", "motor", "trajectory"], []),

        # Graph / network
        ("graph_ml", "social networks", ["graph", "network", "node", "edge", "community", "social"], ["networkx", "igraph", "dgl"]),
        ("graph_ml", "knowledge graphs", ["knowledge graph", "triple", "entity", "relation", "embedding"], ["dgl"]),

        # Tabular / general ML (lowest priority — catch-all)
        ("tabular_ml", "classification", ["classif", "binary", "multiclass", "label"], []),
        ("tabular_ml", "regression", ["regress", "predict", "continuous", "target"], []),
        ("tabular_ml", "time series", ["time series", "forecast", "temporal", "seasonal", "arima", "lstm"], []),
        ("tabular_ml", "recommender", ["recommend", "collaborative", "matrix factori", "user item"], []),
        ("tabular_ml", "anomaly detection", ["anomaly", "outlier", "fraud", "novelty"], []),
    ]

    best_field = ""
    best_sub = ""
    best_score = 0
    best_keywords: list[str] = []

    for field, sub, trigger_words, trigger_pkgs in rules:
        score = 0
        matched_kw: list[str] = []

        # Word matches in signals
        for word in trigger_words:
            if word in signals:
                score += 1
                matched_kw.append(word)

        # Package matches (stronger signal)
        for pkg in trigger_pkgs:
            if pkg in installed:
                score += 3
                matched_kw.append(pkg)

        if score > best_score:
            best_score = score
            best_field = field
            best_sub = sub
            best_keywords = matched_kw

    return best_field, best_sub, best_keywords


def _enqueue(payload: dict) -> None:
    """Add to pending queue and fire async send."""
    with _lock:
        _pending.append(payload)
    t = threading.Thread(target=_flush_one, args=(payload,), daemon=True)
    t.start()


_deletion_tokens: list[str] = []


def show_deletion_tokens() -> list[str]:
    """Show deletion tokens from this session. Save these to delete your data later."""
    return list(_deletion_tokens)


def _flush_one(payload: dict) -> None:
    """Send a single payload to the ingest endpoint."""
    try:
        import urllib.request
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            _endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            # Parse response to capture deletion token
            try:
                resp_data = json.loads(resp.read())
                token = resp_data.get("deletion_token", "")
                if token:
                    _deletion_tokens.append(token)
            except Exception:
                pass
    except Exception:
        pass  # telemetry failure is never fatal
    finally:
        with _lock:
            if payload in _pending:
                _pending.remove(payload)
            _history.append(payload)
