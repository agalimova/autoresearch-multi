"""
Autoresearch telemetry — opt-out, PII-stripped experiment reporting.

Sends everything by default: domain, hypothesis, insights, dead ends,
hyperparams, convergence, hardware, feature names, failures.

PII auto-stripped from all text: names, emails, paths, IPs, credit cards,
SSNs, phone numbers, API keys, org names. Domain terms, model names,
scientific content all preserved.

Disable entirely:  export AUTORESEARCH_TELEMETRY=0
"""

from .collector import (
    init,
    disable,
    enable,
    is_enabled,
    set_tier,
    report_experiment,
    report_failure,
    show_pending,
    show_history,
    show_deletion_tokens,
    TelemetryTier,
    TargetType,
    ModelFamily,
    MetricName,
    normalize_target_type,
    normalize_model_family,
    normalize_metric_name,
)

__all__ = [
    "init", "disable", "enable", "is_enabled", "set_tier",
    "report_experiment", "report_failure",
    "show_pending", "show_history", "show_deletion_tokens",
    "TelemetryTier",
    "TargetType", "ModelFamily", "MetricName",
    "normalize_target_type", "normalize_model_family", "normalize_metric_name",
]
