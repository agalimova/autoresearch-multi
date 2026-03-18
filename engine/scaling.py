"""
Swarm scaling heuristic for multi-agent experiments (EXPERIMENTAL).

This module provides a rough heuristic for estimating how many agents
to run in parallel. The formula is NOT an empirically validated scaling
law — it is a simple parametric model for planning purposes.

    I(N, C, alpha) = log(N) * sqrt(C) * exp(-alpha / N)

Where:
    I     = estimated collective output quality
    N     = number of agents
    C     = compute capacity (normalized 0-1)
    alpha = communication penalty (higher = more coordination overhead)

Usage:
    from engine.scaling import predict_optimal_agents, scaling_curve

    n_opt = predict_optimal_agents(compute=0.5, comm_penalty=0.15)
    print(f"Optimal: {n_opt} agents")

    curve = scaling_curve(compute=0.5, comm_penalty=0.15, max_agents=20)
    for n, score in curve:
        print(f"  {n} agents: {score:.3f}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ScalingPrediction:
    optimal_agents: int
    threshold_agents: int     # beyond this, adding agents hurts
    predicted_scores: list[tuple[int, float]]
    compute: float
    comm_penalty: float


def intelligence(n_agents: int, compute: float, comm_penalty: float) -> float:
    """
    Neural Swarm Scaling Law.
    
    I(N, C, alpha) = log(N) * sqrt(C) * exp(-alpha / N)
    """
    if n_agents <= 0:
        return 0.0
    return math.log(n_agents + 1) * math.sqrt(compute) * math.exp(-comm_penalty / n_agents)


def predict_optimal_agents(
    *,
    compute: float = 0.5,
    comm_penalty: float = 0.15,
    max_agents: int = 50,
) -> int:
    """Find the agent count that maximizes collective intelligence."""
    best_n = 1
    best_score = 0.0
    for n in range(1, max_agents + 1):
        score = intelligence(n, compute, comm_penalty)
        if score > best_score:
            best_score = score
            best_n = n
    return best_n


def find_threshold(
    *,
    compute: float = 0.5,
    comm_penalty: float = 0.15,
    max_agents: int = 50,
) -> int:
    """Find the agent count where adding more starts to hurt (dI/dN < 0)."""
    prev = intelligence(1, compute, comm_penalty)
    for n in range(2, max_agents + 1):
        current = intelligence(n, compute, comm_penalty)
        if current < prev:
            return n - 1
        prev = current
    return max_agents


def scaling_curve(
    *,
    compute: float = 0.5,
    comm_penalty: float = 0.15,
    max_agents: int = 20,
) -> list[tuple[int, float]]:
    """Return (n_agents, intelligence_score) for plotting."""
    return [(n, intelligence(n, compute, comm_penalty)) for n in range(1, max_agents + 1)]


def predict(
    *,
    compute: float = 0.5,
    comm_penalty: float = 0.15,
    max_agents: int = 50,
) -> ScalingPrediction:
    """Full scaling prediction: optimal count, threshold, and curve."""
    curve = scaling_curve(compute=compute, comm_penalty=comm_penalty, max_agents=max_agents)
    optimal = predict_optimal_agents(compute=compute, comm_penalty=comm_penalty, max_agents=max_agents)
    threshold = find_threshold(compute=compute, comm_penalty=comm_penalty, max_agents=max_agents)

    return ScalingPrediction(
        optimal_agents=optimal,
        threshold_agents=threshold,
        predicted_scores=curve,
        compute=compute,
        comm_penalty=comm_penalty,
    )


def estimate_compute_from_hardware() -> float:
    """Estimate normalized compute capacity (0-1) from hardware."""
    try:
        from engine.hardware import detect
        hw = detect()
    except ImportError:
        return 0.3  # conservative default

    if hw.device == "cuda":
        mem = hw.gpu_memory_gb
        if mem >= 40:
            return 1.0    # A100/H100
        if mem >= 16:
            return 0.5    # RTX 4090, V100
        if mem >= 8:
            return 0.3    # RTX 3070, T4
        return 0.15       # smaller
    if hw.device == "mps":
        ram = hw.ram_gb
        if ram >= 64:
            return 0.4    # M Max/Ultra
        if ram >= 16:
            return 0.2    # M Pro
        return 0.1
    return 0.1  # CPU


def auto_predict() -> ScalingPrediction:
    """Auto-detect hardware and predict optimal agent count."""
    compute = estimate_compute_from_hardware()
    return predict(compute=compute)


if __name__ == "__main__":
    from engine.hardware import detect

    hw = detect()
    print(f"Hardware: {hw}")

    compute = estimate_compute_from_hardware()
    result = predict(compute=compute)

    print(f"\nCompute: {compute:.2f} (normalized)")
    print(f"Optimal agents: {result.optimal_agents}")
    print(f"Threshold (adding more hurts): {result.threshold_agents}")
    print(f"\nScaling curve:")
    for n, score in result.predicted_scores[:12]:
        bar = "#" * int(score * 40)
        marker = " <-- optimal" if n == result.optimal_agents else ""
        marker = " <-- THRESHOLD" if n == result.threshold_agents and not marker else marker
        print(f"  {n:3d} agents: {score:.3f} {bar}{marker}")
