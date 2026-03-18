"""Experiment backends for autoresearch.

Each backend wraps a specific experiment type behind the common Backend interface.
"""

from engine.backends.tabular import TabularBackend
from engine.backends.gpu_training import GpuTrainingBackend

__all__ = ["TabularBackend", "GpuTrainingBackend"]
