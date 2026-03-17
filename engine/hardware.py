"""
Hardware auto-detection and configuration suggestions.

Detects GPU/CPU, memory, and suggests batch size and model dimensions.
Inspired by elementalcollision/autoresearch (Apple Silicon detection).

Usage:
    from engine.hardware import detect, suggest_config
    hw = detect()
    print(hw)
    config = suggest_config(hw)
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass


@dataclass
class HardwareInfo:
    device: str          # "cuda", "mps", "cpu"
    device_name: str     # "NVIDIA RTX 3090", "Apple M2 Max", "Intel i7"
    gpu_memory_gb: float # 0 for CPU
    ram_gb: float
    cpu_count: int
    platform: str        # "linux", "darwin", "windows"

    def __str__(self) -> str:
        if self.device == "cpu":
            return f"{self.device_name} | {self.ram_gb:.0f}GB RAM | {self.cpu_count} cores"
        return f"{self.device_name} | {self.gpu_memory_gb:.0f}GB VRAM | {self.ram_gb:.0f}GB RAM"


def detect() -> HardwareInfo:
    """Auto-detect hardware. Returns HardwareInfo."""
    plat = platform.system().lower()
    cpu_count = os.cpu_count() or 1
    ram_gb = _get_ram_gb()

    # Try CUDA
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            mem = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / (1024**3)
            return HardwareInfo("cuda", name, mem, ram_gb, cpu_count, plat)
    except ImportError:
        pass

    # Try MPS (Apple Silicon)
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            chip = _detect_apple_chip()
            return HardwareInfo("mps", chip, 0, ram_gb, cpu_count, plat)
    except ImportError:
        pass

    # CPU fallback
    cpu_name = platform.processor() or platform.machine() or "unknown"
    return HardwareInfo("cpu", cpu_name, 0, ram_gb, cpu_count, plat)


def suggest_config(hw: HardwareInfo) -> dict:
    """Suggest training config based on detected hardware."""
    if hw.device == "cuda":
        return _suggest_cuda(hw)
    if hw.device == "mps":
        return _suggest_mps(hw)
    return _suggest_cpu(hw)


def _suggest_cuda(hw: HardwareInfo) -> dict:
    mem = hw.gpu_memory_gb
    if mem >= 40:      # A100, H100
        return {"batch_size": 256, "hidden_size": 512, "depth": 8, "max_epochs": 20}
    if mem >= 16:      # RTX 4090, A30, V100
        return {"batch_size": 128, "hidden_size": 256, "depth": 6, "max_epochs": 15}
    if mem >= 8:       # RTX 3070, T4
        return {"batch_size": 64, "hidden_size": 256, "depth": 4, "max_epochs": 10}
    return {"batch_size": 32, "hidden_size": 128, "depth": 4, "max_epochs": 10}


def _suggest_mps(hw: HardwareInfo) -> dict:
    ram = hw.ram_gb
    if ram >= 64:      # M Max/Ultra
        return {"batch_size": 64, "hidden_size": 256, "depth": 6, "max_epochs": 10}
    if ram >= 16:      # M Pro
        return {"batch_size": 32, "hidden_size": 128, "depth": 4, "max_epochs": 10}
    return {"batch_size": 16, "hidden_size": 64, "depth": 2, "max_epochs": 10}


def _suggest_cpu(hw: HardwareInfo) -> dict:
    return {"batch_size": 32, "hidden_size": 128, "depth": 2, "max_epochs": 10}


def _get_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / (1024**2)
    except Exception:
        pass
    return 0


def _detect_apple_chip() -> str:
    """Detect Apple Silicon chip model."""
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return f"Apple Silicon ({platform.machine()})"
